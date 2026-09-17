"""TTS experiment arguments alongside the pinned Megatron parser."""

import json
import logging

from slime.observability.logging_utils import configure_logger

logger = logging.getLogger(__name__)

_BACKEND_DEFAULTS = {
    "actor_num_nodes": 1,
    "actor_num_gpus_per_node": 8,
    "rollout_num_gpus_per_engine": 1,
    "num_gpus_per_node": 8,
    "distributed_backend": "nccl",
    "distributed_timeout_minutes": 10,
    "train_env_vars": {},
    "hf_checkpoint": None,
    "rollout_function_path": "slime.rollout.sglang_omni_rollout.generate_rollout",
    "rollout_temperature": 1.0,
    "rollout_max_response_len": None,
    "rollout_shuffle": False,
    "rollout_seed": 42,
    "update_weight_buffer_size": 536870912,
    "num_rollout": None,
    "num_epoch": None,
    "rollout_global_dataset": True,
    "prompt_data": None,
    "start_rollout_id": None,
    "rollout_batch_size": 1,
    "n_samples_per_prompt": 1,
    "global_batch_size": None,
    "micro_batch_size": 1,
    "use_dynamic_batch_size": False,
    "eval_function_path": None,
    "load": None,
    "save": None,
    "async_save": False,
    "seed": 1234,
    "calculate_per_token_loss": False,
    "lr": 1e-06,
    "megatron_config_path": None,
    "eps_clip": 0.2,
    "eps_clip_high": None,
    "grpo_std_normalization": True,
    "use_wandb": False,
    "wandb_mode": None,
    "wandb_dir": None,
    "wandb_key": None,
    "wandb_host": None,
    "wandb_team": None,
    "wandb_group": None,
    "wandb_project": None,
    "wandb_random_suffix": True,
    "wandb_always_use_train_step": False,
    "wandb_run_id": None,
    "use_tensorboard": False,
    "tb_project_name": None,
    "tb_experiment_name": None,
    "save_debug_rollout_data": None,
    "save_debug_train_data": None,
    "memory_snapshot_dir": ".",
    "memory_snapshot_num_steps": None,
    "memory_recorder": "torch",
    "record_memory_history": False,
    "data_pad_size_multiplier": 128,
    "ci_save_grad_norm": None,
    "ci_load_grad_norm": None,
    "custom_megatron_init_path": None,
    "custom_megatron_before_train_step_hook_path": None,
    "padded_vocab_size": None,
}


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_arguments(parser):
        existing = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in _BACKEND_DEFAULTS.items() if key not in existing})
        parser.set_defaults(
            balance_data=False,
            balance_by_flops=False,
            reward_key=None,
            eval_reward_key=None,
            use_fault_tolerance=False,
        )
        group = parser.add_argument_group("slime TTS experiment")
        group.add_argument("--train-backend", choices=["megatron"], default="megatron")
        group.add_argument("--debug-rollout-only", action="store_true")
        group.add_argument("--debug-train-only", action="store_true")
        group.add_argument("--load-debug-rollout-data", default=None)
        group.add_argument("--hf-checkpoint", required=True)
        group.add_argument("--actor-num-nodes", type=int, default=1)
        group.add_argument("--actor-num-gpus-per-node", type=int, default=1)
        group.add_argument("--ray-address", default=None)
        group.add_argument("--train-env-vars", type=json.loads, default={})
        group.add_argument("--model-name", default=None)
        group.add_argument("--num-rollout", type=int, default=None)
        group.add_argument("--num-epoch", type=int, default=None)
        group.add_argument("--start-rollout-id", type=int, default=None)
        group.add_argument("--rollout-batch-size", type=int, default=1)
        group.add_argument("--n-samples-per-prompt", type=int, default=4)
        group.add_argument("--prompt-data", default=None)
        group.add_argument("--custom-rm-path", default=None)
        group.add_argument("--group-rm", action="store_true")
        group.add_argument("--reward-config", default=None)
        group.add_argument("--reward-timeout", type=float, default=120)
        group.add_argument("--reward-concurrency", type=int, default=8)
        group.add_argument("--reward-max-retries", type=int, default=2)
        group.add_argument("--rollout-group-max-retries", type=int, default=2)
        group.add_argument("--max-recoverable-rollout-failures", type=int, default=32)
        group.add_argument("--reward-key", default=None)
        group.add_argument("--eval-data", default=None)
        group.add_argument("--eval-config", default=None)
        group.add_argument("--eval-prompt-data", nargs="+", default=None)
        group.add_argument("--n-samples-per-eval-prompt", type=int, default=1)
        group.add_argument("--eval-temperature", type=float, default=None)
        group.add_argument("--eval-max-response-len", type=int, default=None)
        group.add_argument("--rollout-temperature", type=float, default=1.0)
        group.add_argument("--rollout-top-p", type=float, default=1.0)
        group.add_argument("--rollout-top-k", type=int, default=-1)
        group.add_argument("--rollout-max-response-len", type=int, default=512)
        group.add_argument("--rollout-seed", type=int, default=42)
        group.add_argument("--rollout-shuffle", action="store_true")
        group.add_argument("--use-dynamic-batch-size", action="store_true")
        group.add_argument("--max-tokens-per-gpu", type=int, default=2048)
        group.add_argument("--data-pad-size-multiplier", type=int, default=128)
        group.add_argument("--eps-clip", type=float, default=0.2)
        group.add_argument("--eps-clip-high", type=float, default=0.2)
        group.add_argument(
            "--disable-grpo-std-normalization", dest="grpo_std_normalization", action="store_false", default=True
        )
        group.add_argument("--update-weight-buffer-size", type=int, default=256 * 1024 * 1024)
        group.add_argument("--save-debug-rollout-data", default=None)
        group.add_argument("--save-debug-train-data", default=None)
        group.add_argument("--use-tensorboard", action="store_true")
        group.add_argument("--tb-log-dir", default=None)
        group.add_argument("--tb-project-name", default="tts-rl")
        group.add_argument("--tb-experiment-name", default="run")
        group.add_argument("--use-wandb", action="store_true")
        reset_arg(parser, "--wandb-project", default="tts-rl")
        reset_arg(parser, "--wandb-group", default="run")
        group.add_argument("--wandb-mode", choices=["offline", "online", "disabled"], default="offline")
        reset_arg(parser, "--wandb-dir", default=None)
        group.add_argument("--ci-save-grad-norm", default=None)
        group.add_argument("--ci-load-grad-norm", default=None)
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)
        reset_arg(parser, "--eval-interval", type=int, default=None)
        reset_arg(parser, "--micro-batch-size", type=int, default=1)
        reset_arg(parser, "--lr", type=float, default=1e-6)
        reset_arg(parser, "--weight-decay", type=float, default=0.0)
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)
        return parser

    return add_arguments


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()

    def add_tts_arguments(parser):
        group = parser.add_argument_group("Omni TTS RL")
        group.add_argument("--model-family", choices=["moss_tts_local", "higgs_tts"], default="moss_tts_local")
        group.add_argument("--omni-endpoints", nargs="+", default=[])
        group.add_argument("--omni-stage", default="tts_engine")
        group.add_argument("--omni-concurrency", type=int, default=8)
        group.add_argument("--omni-timeout", type=float, default=600)
        group.add_argument("--objective", choices=["grpo", "mopd"], default="grpo")
        group.add_argument("--mopd-teachers", nargs="*", default=[])
        group.add_argument("--mopd-advantage-clip", type=float, default=5)
        group.add_argument("--asr-endpoint", default=None)
        group.add_argument("--asr-model", default="qwen3-asr")
        group.add_argument("--wer-language", default="en", help="Default language for rows without language; en/zh or a canonical language name")
        group.add_argument("--asr-protocol", choices=["openai_audio_transcriptions", "qwen3_asr_chat_path"], default="openai_audio_transcriptions")
        group.add_argument("--asr-auth-token-env", default=None)
        group.add_argument("--train-scope", choices=["full", "audio"], default="full")
        group.add_argument("--local-chunk-size", type=int, default=256)
        group.add_argument("--metrics-jsonl", default=None)
        group.add_argument("--audio-output-dir", default=None)
        group.add_argument(
            "--logprob-parity-tolerance",
            type=float,
            default=1.0,
            help="Maximum BF16 selected-score difference; inspect together with its mean and ratio metrics",
        )
        group.add_argument("--logprob-parity-mean-tolerance", type=float, default=0.1)
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)
        return parser

    add_slime_arguments = get_slime_extra_args_provider(add_tts_arguments)

    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
    )

    configure_tts_args(args)
    if not args.debug_rollout_only:
        megatron_validate_args(args)
    return args


def configure_tts_args(args):
    """Resolve the checkpoint once, before model construction and distributed startup."""
    import sys
    import uuid
    from pathlib import Path

    from slime.backends.megatron_utils.arguments import set_default_megatron_args
    from slime_plugins.models.moss_tts_local.config import MossLocalConfig

    if not args.hf_checkpoint:
        raise ValueError("--hf-checkpoint must identify a complete TTS model")
    if args.model_family == "moss_tts_local":
        args.policy_config = MossLocalConfig.from_pretrained(args.hf_checkpoint)
        language = args.policy_config.language
    else:
        config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
        args.policy_config = config
        language = config.get("text_config", config)
    for source, target in {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "intermediate_size": "ffn_hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
    }.items():
        actual, expected = getattr(args, target, None), language[source]
        explicit = f"--{target.replace('_', '-')}" in {arg.split("=", 1)[0] for arg in sys.argv[1:]}
        if explicit and actual is not None and actual != expected:
            raise ValueError(f"--{target.replace('_', '-')}={actual} disagrees with checkpoint {expected}")
        setattr(args, target, expected)
    args.kv_channels = language.get("head_dim", language["hidden_size"] // language["num_attention_heads"])
    args.vocab_size = language["vocab_size"]
    args.padded_vocab_size = None
    args.max_position_embeddings = language["max_position_embeddings"]
    args.rotary_base = language.get("rope_parameters", {}).get("rope_theta", language.get("rope_theta", 1000000))
    args.norm_epsilon = args.layernorm_epsilon = language.get("rms_norm_eps", 1e-6)
    args.normalization = "RMSNorm"
    args.swiglu = True
    args.qk_layernorm = True
    args.group_query_attention = True
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    args.attention_dropout = args.hidden_dropout = 0.0
    args.accumulate_allreduce_grads_in_fp32 = True
    args.position_embedding_type = "rope"
    args.transformer_impl = "transformer_engine"
    if args.num_experts or args.mtp_num_layers or args.fp8 or args.fp16:
        raise ValueError("The current speech policies use dense BF16 training without MTP")
    if args.pipeline_model_parallel_size != 1 or args.sequence_parallel:
        raise ValueError("This TTS implementation supports DP/TP/CP; PP and SP are not enabled")
    if args.rollout_top_p != 1.0 or args.rollout_top_k != -1 or args.rollout_temperature <= 0:
        raise ValueError("TTS RL requires positive temperature, top_p=1 and top_k=-1")
    if args.local_chunk_size < 1 or args.omni_concurrency < 1:
        raise ValueError("Local chunk size and Omni concurrency must be positive")
    for name in ("actor_num_nodes", "actor_num_gpus_per_node", "rollout_batch_size", "n_samples_per_prompt"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_rollout is not None and args.num_rollout < 0:
        raise ValueError("num_rollout cannot be negative")
    if not preloaded_rollout(args) and args.num_rollout != 0 and not args.prompt_data:
        raise ValueError("Training requires --prompt-data")
    if args.calculate_per_token_loss:
        raise ValueError("TTS objectives use complete-sample normalization, not Megatron token-count normalization")
    if args.global_batch_size is None:
        args.global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt
    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip
    if args.num_rollout is None and args.num_epoch is None:
        raise ValueError("Specify --num-rollout or --num-epoch")
    from slime.utils.eval_config import resolve_eval_datasets
    from slime.rollout.rm_hub.config import get_reward_config
    from slime.rollout.rm_hub.language import resolve_language

    resolve_language(args.wer_language)
    args.eval_datasets = resolve_eval_datasets(args)
    if args.eval_interval is not None and not args.eval_datasets:
        raise ValueError("Evaluation requires --eval-data, --eval-config, or --eval-prompt-data")
    for name in ("rollout_group_max_retries", "max_recoverable_rollout_failures"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.custom_rm_path not in (None, "slime.rollout.rm_hub.wer.reward_func") and args.reward_config:
        raise ValueError("Choose --custom-rm-path or --reward-config")
    if args.group_rm and not args.custom_rm_path:
        raise ValueError("--group-rm requires --custom-rm-path")
    builtin_reward = args.custom_rm_path in (None, "slime.rollout.rm_hub.wer.reward_func")
    if builtin_reward and not preloaded_rollout(args) and (
        args.objective == "grpo" or any(not dataset.reward_config for dataset in args.eval_datasets)
    ):
        get_reward_config(args)
    if args.objective == "grpo" and args.n_samples_per_prompt < 2:
        raise ValueError("GRPO requires at least two samples per prompt")
    if args.objective == "mopd" and not args.mopd_teachers and not preloaded_rollout(args):
        raise ValueError("MOPD requires --mopd-teachers DOMAIN=URL entries")
    if not args.omni_endpoints and not preloaded_rollout(args):
        raise ValueError("Provide --omni-endpoints or --load-debug-rollout-data")
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args.weight_sync_session = uuid.uuid4().hex[:12]
    if args.actor_num_nodes > 1 and not args.ray_address:
        raise ValueError("Multi-node training requires an existing --ray-address")
    args.num_gpus_per_node = args.actor_num_gpus_per_node
    args.initialization_checkpoint = False
    if not args.debug_rollout_only and not getattr(args, "convert_hf_to_torch_dist", False):
        from slime.backends.megatron_utils.checkpoint import _is_megatron_checkpoint

        if not args.load or not _is_megatron_checkpoint(args.load):
            raise ValueError("--load must name a torch_dist directory; use tools/convert_hf_to_torch_dist.py first")
        directory = Path(args.load)
        if directory.name.startswith("iter_"):
            directory = directory.parent
        manifest = directory / "slime_checkpoint.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            args.initialization_checkpoint = metadata.get("kind") == "initialization"
        if args.initialization_checkpoint:
            if args.save and Path(args.save).resolve() == directory.resolve():
                raise ValueError("Save training checkpoints separately from the model-only initialization directory")
            args.finetune = True
            args.no_load_optim = args.no_load_rng = True
            args.use_checkpoint_opt_param_scheduler = False
        elif not args.override_opt_param_scheduler:
            args.use_checkpoint_opt_param_scheduler = True
    args.pretrained_checkpoint = args.load
    args.rollout_seed = args.seed if args.rollout_seed is None else args.rollout_seed
    args.rollout_external = True
    args.rollout_num_gpus = len(args.omni_endpoints)
    args.rollout_num_gpus_per_engine = 1
    args.rollout_global_dataset = True
    args.rollout_data_transport = "object-store"
    args.rollout_function_path = "slime.rollout.sglang_omni_rollout.generate_rollout"
    args.eval_function_path = args.rollout_function_path
    args.data_source_path = "slime.rollout.data_source.RolloutDataSource"
    args.custom_rm_path = args.custom_rm_path or "slime.rollout.rm_hub.wer.reward_func"
    args.input_key = "text"
    args.label_key = "text"
    args.metadata_key = "metadata"
    args.train_env_vars = args.train_env_vars or {}
    args.audio_output_dir = args.audio_output_dir or str(Path(args.save or "outputs") / "audio")
    args.metrics_jsonl = args.metrics_jsonl or str(Path(args.save or "outputs") / "metrics.jsonl")
    args.debug_train_only = bool(args.debug_train_only or preloaded_rollout(args))
    set_default_megatron_args(args)


def preloaded_rollout(args):
    return bool(args.debug_train_only or args.load_debug_rollout_data)
