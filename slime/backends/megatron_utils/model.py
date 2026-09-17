import dataclasses
import gc
import logging
import math
from argparse import Namespace
from collections.abc import Sequence
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

from slime.observability import logging_utils, train_metric_utils
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .data import DataIterator, get_batch
from .loss import structured_policy_loss
from .model_provider import get_model_provider_func

logger = logging.getLogger(__name__)


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = max(
        1, args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Policy role; this branch uses "actor".
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    if args.num_rollout == 0:
        args.no_load_optim = True
        return model, None, None

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    optimizer = get_megatron_optimizer(
        config=config,
        model_chunks=model,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(args, model, data_iterator, num_microbatches, store_prefix=""):
    """Replay supplied actions with the same Megatron schedule used by training."""
    from .loss import collect_policy_scores

    for iterator in data_iterator:
        iterator.reset()
    for module in model:
        module.eval()
    get_model_config(model[0]).timers = None

    def forward_step(iterator, module, return_schedule_plan=False):
        if return_schedule_plan:
            raise ValueError("Speech replay does not use combined 1f1b plans")
        batch = get_batch(iterator, pad_multiplier=args.data_pad_size_multiplier)
        return module(batch), partial(collect_policy_scores, batch=batch)

    outputs = []
    try:
        schedule = get_forward_backward_func()
        for count in num_microbatches:
            outputs.extend(
                schedule(
                    forward_step_func=forward_step,
                    data_iterator=data_iterator,
                    model=model,
                    num_microbatches=count,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    forward_only=True,
                )
            )
    finally:
        for module in model:
            module.train()
    if not mpu.is_pipeline_last_stage():
        return {}
    values = [score for item in outputs for score in item["log_probs"]]
    indices = [index for batch in data_iterator[0].micro_batch_indices for index in batch]
    if len(values) != len(indices):
        raise ValueError("Forward replay returned a different number of samples")
    ordered = [None] * len(indices)
    for index, value in zip(indices, values, strict=True):
        ordered[index] = value
    return {f"{store_prefix}log_probs": ordered}


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(iterator, module, return_schedule_plan=False):
        if return_schedule_plan:
            raise ValueError("Speech policies do not use combined 1f1b schedule plans")
        batch = get_batch(iterator, pad_multiplier=args.data_pad_size_multiplier)
        return module(batch), partial(structured_policy_loss, args, batch, num_microbatches, step_global_batch_size)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = train_metric_utils.reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {
                f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)
    save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far=0,
        checkpointing_context=None,
        train_data_iterator=None,
        preprocess_common_state_dict_fn=None,
    )
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Policy role; this branch uses "actor".

    Returns:
        tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
    )
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
