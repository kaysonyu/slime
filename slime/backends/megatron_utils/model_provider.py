"""Construct speech policies on the shared Megatron execution boundary."""

import torch
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.training.arguments import core_transformer_config_from_args


class SpeechPolicyModel(LanguageModule):
    """Shared Megatron global-time execution boundary for discrete speech policies.

    Subclasses provide row embeddings and action heads. Packing, RoPE and CP use
    the same Megatron implementation for every model family.
    """

    def _initialize_backbone(self, config, spec, args, *, pre_process, post_process, pg_collection=None):
        from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
        from megatron.core.transformer.transformer_block import TransformerBlock

        if not pre_process or not post_process:
            raise ValueError("Speech pipeline stages are not enabled; use PP=1")
        if config.sequence_parallel:
            raise ValueError("Speech sequence-parallel head boundary is not enabled")
        self.pre_process, self.post_process = pre_process, post_process
        # No full-vocabulary text output head is part of the speech policy.
        # Audio tying (Higgs) is owned by its model, not LanguageModule's text-head sharing.
        self.share_embeddings_and_output_weights = False
        self.text_embedding = LanguageModelEmbedding(
            config,
            args.padded_vocab_size,
            args.max_position_embeddings,
            position_embedding_type="rope",
            tp_group=pg_collection.tp if pg_collection is not None else None,
        )
        self.decoder = TransformerBlock(
            config=config, spec=spec, pre_process=True, post_process=True, pg_collection=pg_collection
        )
        self.rotary_pos_emb = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=False,
            rotary_base=args.rotary_base,
            use_cpu_initialization=config.use_cpu_initialization,
        )

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        self.decoder.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        return None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.transformer.module import MegatronModule

        # LanguageModule's implementation assumes a single text output_layer.
        # Speech heads have separate vocabularies and their own tying rules.
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def forward(self, batch):
        from megatron.core.packed_seq_params import PackedSeqParams

        embedded = self._embed_rows(batch)
        lengths = batch.cu_seqlens[1:] - batch.cu_seqlens[:-1]
        packed = PackedSeqParams(
            cu_seqlens_q=batch.cu_seqlens,
            cu_seqlens_kv=batch.cu_seqlens,
            max_seqlen_q=int(lengths.max()),
            max_seqlen_kv=int(lengths.max()),
            qkv_format="thd",
        )
        rotary_length = self.rotary_pos_emb.get_rotary_seq_len(None, self.decoder, embedded, self.config, packed)
        hidden = self.decoder(
            hidden_states=embedded,
            attention_mask=None,
            rotary_pos_emb=self.rotary_pos_emb(rotary_length, packed_seq=True),
            packed_seq_params=packed,
        )
        selected = hidden[:, 0].index_select(0, batch.prediction_positions)
        anchor = hidden.sum() * 0
        if not len(selected):
            # CP can assign only prompt/padding rows to a rank. Keep the
            # replicated action parameters in the backward graph on that rank
            # so DDP receives the same gradient-ready events everywhere.
            for module in self.action_modules:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        anchor = anchor + parameter.reshape(-1)[0] * 0
            return hidden.new_empty(batch.targets.shape, dtype=torch.float32) + anchor
        return self._score_actions(selected, batch) + anchor


def get_model_provider_func(args, role="actor"):
    if role != "actor":
        raise ValueError("TTS training has one policy; teachers are external frozen scorers")

    def provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        config = config if config is not None else core_transformer_config_from_args(args)
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        if args.model_family == "moss_tts_local":
            from slime_plugins.models.moss_tts_local.model import MossLocalModel

            model_type = MossLocalModel
        elif args.model_family == "higgs_tts":
            from slime_plugins.models.higgs_tts.model import HiggsModel

            model_type = HiggsModel
        else:
            raise ValueError(f"Unsupported speech model {args.model_family}")
        return model_type(
            config,
            spec,
            args.policy_config,
            args,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
        )

    return provider
