"""Export a native Megatron speech checkpoint as a complete Omni/HF artifact.

Launch with torchrun at the desired TP/CP size. No optimizer state is loaded.
The original HF artifact supplies frozen codec tensors and tokenizer/model code.
"""

import json
import os
import shutil
from importlib import import_module
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from safetensors.torch import save_file


def add_arguments(parser):
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--export-shard-bytes", type=int, default=1024**3)
    return parser


def main():
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    from slime.backends.megatron_utils.checkpoint import load_checkpoint
    from slime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
    from slime.utils.arguments import parse_args
    from slime.utils.distributed_utils import init_gloo_group

    args = parse_args(add_arguments)
    destination = Path(args.export_dir).resolve()
    output = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or output.exists() or args.export_shard_bytes <= 0:
        raise ValueError("Export requires a new destination and a positive shard size")
    args.no_load_optim = args.no_load_rng = True
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    init_gloo_group()
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    try:
        init(args)
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        iteration, _ = load_checkpoint(model, None, None, checkpointing_context={})
        reader = SafetensorReader(args.hf_checkpoint)
        adapter = import_module(f"slime_plugins.models.{args.model_family}.weights")
        aliases = getattr(adapter, "checkpoint_aliases", {})
        buffer, weight_map = {}, {}
        size = total_size = shard = 0
        if args.rank == 0:
            output.mkdir(parents=True)
            for source in Path(args.hf_checkpoint).iterdir():
                if source.is_file() and source.suffix in {
                    ".json",
                    ".py",
                    ".txt",
                    ".model",
                    ".tiktoken",
                    ".yaml",
                    ".md",
                    ".jinja",
                }:
                    if source.name != "model.safetensors.index.json":
                        if source.name == "conversion_manifest.json":
                            shutil.copy2(source, output / "base_conversion_manifest.json")
                        elif source.name in {"sglang_omni.yaml", "sglang_omni_teacher.yaml"}:
                            deployment = yaml.safe_load(source.read_text())
                            deployment["model_path"] = str(destination)
                            (output / source.name).write_text(yaml.safe_dump(deployment, sort_keys=False))
                        else:
                            shutil.copy2(source, output / source.name)

        def flush():
            nonlocal size, shard
            if not buffer:
                return
            shard += 1
            name = f"model-{shard:05d}.safetensors"
            save_file(buffer, output / name, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(buffer, name))
            buffer.clear()
            size = 0

        def emit(name, tensor):
            nonlocal size, total_size
            if name in buffer or name in weight_map:
                raise ValueError(f"Duplicate exported tensor {name}")
            value = tensor.detach().cpu().contiguous().clone()
            count = value.numel() * value.element_size()
            if buffer and size + count > args.export_shard_bytes:
                flush()
            buffer[name] = value
            size += count
            total_size += count

        for name, parameter in named_params_and_buffers(args, model):
            tensor = all_gather_param(name, parameter)
            if args.rank == 0:
                for key, value in adapter.export_parameter(args, name, tensor):
                    emit(key, value)
                    for alias in aliases.get(key, []):
                        if alias in reader.weight_map:
                            emit(alias, value)
        if args.rank == 0:
            emitted = weight_map.keys() | buffer.keys()
            missing = adapter.expected_hf_names(args.hf_checkpoint) - emitted
            if missing:
                raise ValueError(f"Native policy export omitted trained tensors: {sorted(missing)}")
            for name in reader.weight_map.keys() - (weight_map.keys() | buffer.keys()):
                emit(name, reader.get_tensor(name))
            flush()
            if set(weight_map) != set(reader.weight_map):
                raise ValueError("The exported HF artifact changed its tensor manifest")
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
            report = dict(
                model_family=args.model_family,
                checkpoint=args.load,
                hf_base=args.hf_checkpoint,
                iteration=iteration,
                tensors=len(weight_map),
                shards=shard,
                total_size=total_size,
                tp=args.tensor_model_parallel_size,
                cp=args.context_parallel_size,
            )
            (output / "slime_export.json").write_text(json.dumps(report, indent=2) + "\n")
            if destination.exists():
                raise FileExistsError(destination)
            output.rename(destination)
            print(json.dumps(report), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
