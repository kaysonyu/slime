"""Real-checkpoint likelihood/backward probe using slime's Megatron schedule.

This diagnostic uses an explicit synthetic positive advantage to verify the
optimizer path. Actual WER training is exercised by the normal train.py loop.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def add_arguments(parser):
    parser.add_argument("--probe-rollout-file", required=True)
    parser.add_argument("--probe-output-dir", required=True)
    parser.add_argument("--probe-forward-only", action="store_true")
    parser.add_argument("--probe-local-diagnostics", action="store_true")
    parser.add_argument("--probe-hf-reference", action="store_true")
    return parser


def main():
    from slime.utils.arguments import parse_args
    from slime.utils.distributed_utils import init_gloo_group
    from slime.utils.types import Sample

    args = parse_args(add_arguments)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    init_gloo_group()
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import forward_only, initialize_model_and_optimizer, train

    init(args)
    model, optimizer, scheduler, _ = initialize_model_and_optimizer(args)
    if Path(args.probe_rollout_file).suffix == ".pt":
        saved = torch.load(args.probe_rollout_file, weights_only=False)["samples"][0]
        trace = Sample.from_dict(saved).trajectory
    else:
        from slime_plugins.models.moss_tts_local.data import MossLocalTrajectory

        response = json.loads(Path(args.probe_rollout_file).read_text())
        trace = MossLocalTrajectory.from_omni(
            response["meta_info"]["omni_rollout"], response["meta_info"], args.policy_config
        )
    sample = Sample(index=0, group_index=0, trajectory=trace, reward=1.0, advantage=1.0)
    data = dict(samples=[sample], micro_batch_indices=[[0]], num_microbatches=[1], global_batch_sizes=[1])
    iterator = get_data_iterator(data)
    unwrapped = model[0]
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    captured = []
    handle = unwrapped.decoder.register_forward_hook(lambda module, inputs, result: captured.append(result.detach()))
    results = forward_only(args, model, iterator, [1])
    handle.remove()
    scores = results["log_probs"][0]
    _, _, _, mask, old = trace.training_tensors(args.policy_config)
    difference = (scores - old).abs()
    report = dict(
        cp=args.context_parallel_size,
        tp=args.tensor_model_parallel_size,
        max_abs=float(difference[mask].max()),
        mean_abs=float(difference[mask].mean()),
        head_max=[float(difference[:, i][mask[:, i]].max()) for i in range(difference.shape[1])],
    )
    if args.probe_local_diagnostics:
        from sglang_omni.models.moss_tts_local.local_transformer import MossTTSLocalTransformer

        from slime.backends.megatron_utils.data import get_batch

        iterator[0].reset()
        batch = get_batch(iterator[0], ["samples"], args.data_pad_size_multiplier)
        local_config = args.policy_config.local
        local = MossTTSLocalTransformer(
            hidden_size=args.hidden_size,
            num_heads=local_config["num_attention_heads"],
            inner_size=local_config["intermediate_size"],
            num_layers=local_config["num_hidden_layers"],
            max_positions=args.policy_config.n_vq + 1,
            rope_base=args.policy_config.local_rope_base,
            layer_norm_eps=local_config["layer_norm_eps"],
        ).to(device="cuda", dtype=torch.bfloat16)
        local.load_state_dict(unwrapped.local_transformer.state_dict(), strict=True)
        hidden = captured[0][:, 0][batch.prediction_positions]
        with torch.no_grad():
            for mode, chunk in [("batched_step", len(hidden)), ("single_step", 1)]:
                pieces = []
                for start in range(0, len(hidden), chunk):
                    targets = batch.targets[start : start + chunk]
                    h = local.step(hidden[start : start + chunk], 0)
                    logits = unwrapped.local_text_lm_head(h).float()
                    columns = [logits.log_softmax(-1).gather(-1, targets[:, :1]).squeeze(-1)]
                    for depth, head in enumerate(unwrapped.audio_lm_heads):
                        logits = head(h).float()
                        columns.append(logits.log_softmax(-1).gather(-1, targets[:, depth + 1, None]).squeeze(-1))
                        if depth + 1 < args.policy_config.n_vq:
                            h = local.step(unwrapped.audio_embeddings[depth](targets[:, depth + 1]), depth + 1)
                    pieces.append(torch.stack(columns, dim=-1))
                step_scores = torch.cat(pieces).cpu()
                report[f"local_{mode}_max_abs"] = float((step_scores - scores)[mask].abs().max())
                report[f"local_{mode}_mean_abs"] = float((step_scores - scores)[mask].abs().mean())
                report[f"rollout_{mode}_max_abs"] = float((step_scores - old)[mask].abs().max())
    if args.probe_hf_reference and args.model_family == "higgs_tts":
        from transformers import Qwen3Config, Qwen3Model
        from slime.backends.megatron_utils.data import get_batch
        from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader

        iterator[0].reset()
        batch = get_batch(iterator[0], ["samples"], args.data_pad_size_multiplier)
        reader = SafetensorReader(args.hf_checkpoint)
        reference = (
            Qwen3Model(Qwen3Config(**args.policy_config["text_config"]))
            .to(device="cuda:1", dtype=torch.bfloat16)
            .eval()
        )
        state = {
            name.removeprefix("body."): reader.get_tensor(name)
            for name in reader.weight_map
            if name.startswith("body.")
        }
        state["embed_tokens.weight"] = reader.get_tensor("tied.embedding.text_embedding.weight")
        reference.load_state_dict(state, strict=True)
        audio_embedding = reader.get_tensor("tied.embedding.modality_embeddings.0.embedding.weight").to("cuda:1")
        audio_head = (
            reader.get_tensor("tied.head.modality_heads.0.weight").to("cuda:1")
            if "tied.head.modality_heads.0.weight" in reader.weight_map
            else audio_embedding
        )
        rows = batch.input_rows.to("cuda:1")
        n = args.policy_config["audio_encoder_config"]["num_codebooks"]
        vocab = args.policy_config["audio_encoder_config"]["vocab_size"]
        with torch.no_grad():
            audio = rows[:, 0].eq(-100)
            text = reference.embed_tokens(rows[:, 0].masked_fill(audio, 0))
            indices = rows[:, 1:].clamp_min(0) + torch.arange(n, device="cuda:1") * vocab
            embedded_audio = torch.nn.functional.embedding(indices, audio_embedding).sum(-2)
            embedded = torch.where(audio[:, None], embedded_audio, text)
            hidden = reference(inputs_embeds=embedded[None], use_cache=False).last_hidden_state[0]
            report["global_hidden_hf_max_abs"] = float((hidden.cpu() - captured[0][:, 0].cpu()).abs().max())
            logits = (
                torch.nn.functional.linear(hidden[batch.prediction_positions.to("cuda:1")], audio_head)
                .reshape(-1, n, vocab)
                .float()
            )
            reference_scores = (
                (logits / batch.temperatures.to("cuda:1")[..., None])
                .log_softmax(-1)
                .gather(-1, batch.targets.to("cuda:1")[..., None])
                .squeeze(-1)
                .cpu()
            )
            report.update(
                hf_vs_megatron_max_abs=float((reference_scores - scores)[mask].abs().max()),
                hf_vs_megatron_mean_abs=float((reference_scores - scores)[mask].abs().mean()),
                hf_vs_rollout_max_abs=float((reference_scores - old)[mask].abs().max()),
                hf_vs_rollout_mean_abs=float((reference_scores - old)[mask].abs().mean()),
            )
            Path(args.probe_output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(reference_scores, Path(args.probe_output_dir) / "hf_scores.pt")
    elif args.probe_hf_reference:
        raise ValueError(
            "MOSS uses canonical Local weights; verify the original NeoX conversion with the offline conversion tests"
        )
    output = Path(args.probe_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        torch.save(scores, output / "scores.pt")
        (output / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PARITY", json.dumps(report), flush=True)
    if report["max_abs"] > args.logprob_parity_tolerance or report["mean_abs"] > args.logprob_parity_mean_tolerance:
        raise RuntimeError(f"Likelihood parity failed: {report['max_abs']}")
    if not args.probe_forward_only:
        head = unwrapped.local_text_lm_head if args.model_family == "moss_tts_local" else unwrapped.audio_head
        before = head.weight.detach().clone()
        train(0, model, optimizer, scheduler, iterator, [1], [1])
        report["decision_head_change"] = float((head.weight - before).abs().max())
        if report["decision_head_change"] == 0:
            raise RuntimeError("The positive-advantage optimizer probe did not change the decision head")
        if args.rank == 0:
            (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
