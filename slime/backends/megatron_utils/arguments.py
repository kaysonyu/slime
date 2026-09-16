import logging

from megatron.training.arguments import parse_args as _megatron_parse_args
from megatron.training.arguments import validate_args as _megatron_validate_args

try:
    from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding as _vocab_size_with_padding
except ImportError:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding

__all__ = ["validate_args", "megatron_parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


def validate_args(args):
    """Run megatron's own validate_args plus slime-specific megatron validations."""

    _megatron_validate_args(args)

    # always use varlen
    args.variable_seq_lengths = True
    if args.pipeline_model_parallel_size == 1:
        assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, (
            "decoder_first_pipeline_num_layers and decoder_last_pipeline_num_layers should be None when "
            "pipeline_model_parallel_size is 1."
        )


def _set_default_megatron_args(args):
    # always use zero optimizer
    args.use_distributed_optimizer = True
    if not hasattr(args, "enable_gloo_process_groups"):
        args.enable_gloo_process_groups = True
    # TODO: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # Checkpoint I/O defaults: these keep checkpoint contents unchanged while
    # reducing repeated validation/planning work and parallelizing load.
    args.use_persistent_ckpt_worker = True
    args.ckpt_assume_constant_structure = True
    args.ckpt_fully_parallel_load = True
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    # Megatron also uses this value as YaRN's original context length. Preserve
    # the checkpoint/model value when the launcher supplied one explicitly.
    if args.max_position_embeddings is None:
        args.max_position_embeddings = args.seq_length
    # TODO: revisit this when megatron(dev) have solved the optimizer-cpu-offload ckpt saving bug
    args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"
    elif not args.tokenizer_model:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
    return args


# Public alias for external tools (e.g. convert_hf_to_torch_dist.py)
set_default_megatron_args = _set_default_megatron_args


def megatron_parse_args(extra_args_provider):
    """Use Megatron's parser; resolve speech geometry in configure_tts_args."""
    args = _megatron_parse_args(extra_args_provider=extra_args_provider, ignore_unknown_args=False)
    args.rank = 0
    args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    return _set_default_megatron_args(args)
