"""Initialize a model-only Megatron torch_dist directory from canonical Local HF weights."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def conversion_arguments(parser):
    parser.set_defaults(convert_hf_to_torch_dist=True, debug_train_only=True, num_rollout=0, save_interval=1)
    return parser


def main():
    from megatron.core.enums import ModelType
    from megatron.training.checkpointing import save_checkpoint
    from megatron.training.training import get_model

    from slime.backends.megatron_utils.hf_to_megatron import load_hf_weights
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model_provider import get_model_provider_func
    from slime.utils.arguments import parse_args
    from slime.utils.distributed_utils import init_gloo_group

    args = parse_args(conversion_arguments)
    if not args.save:
        raise ValueError("Provide --save for the new torch_dist directory")
    destination = Path(args.save).resolve()
    staging = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or staging.exists():
        raise FileExistsError(f"Choose a new destination: {destination}")
    args.save = str(staging)
    args.no_save_optim = args.no_save_rng = True
    args.async_save = False
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")
    init_gloo_group()
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size()
    try:
        init(args)
        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        load_hf_weights(args, model, args.hf_checkpoint)
        save_checkpoint(
            iteration=0,
            model=model,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )
        dist.barrier()
        if args.rank == 0:
            record = dict(
                kind="initialization",
                format="torch_dist",
                iteration=0,
                model_family=args.model_family,
                hf_checkpoint=args.hf_checkpoint,
                tp=args.tensor_model_parallel_size,
                cp=args.context_parallel_size,
                optimizer_state=False,
                rng_state=False,
            )
            (staging / "slime_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
            if destination.exists():
                raise FileExistsError(destination)
            staging.rename(destination)
            print(json.dumps(record, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
