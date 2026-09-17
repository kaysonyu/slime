"""Higgs Qwen3 policy using the same Megatron global-time execution as MOSS."""

import torch
from torch import nn

from slime.backends.megatron_utils.model_provider import SpeechPolicyModel


class HiggsModel(SpeechPolicyModel):
    def __init__(self, config, spec, model_config, args, *, pre_process=True, post_process=True, pg_collection=None):
        super().__init__(config=config, pg_collection=pg_collection)
        self._initialize_backbone(
            config, spec, args, pre_process=pre_process, post_process=post_process, pg_collection=pg_collection
        )
        audio = model_config["audio_encoder_config"]
        if audio["encoder_type"] != "discrete" or not audio["use_delay_pattern"]:
            raise ValueError("Higgs adapter requires discrete delayed codebooks")
        self.codebooks, self.audio_vocab = audio["num_codebooks"], audio["vocab_size"]
        self.audio_embedding = nn.Embedding(self.codebooks * self.audio_vocab, config.hidden_size)
        self.audio_head = nn.Linear(config.hidden_size, self.codebooks * self.audio_vocab, bias=False)
        if audio.get("tie_word_embeddings", True):
            self.audio_head.weight = self.audio_embedding.weight
        self.action_modules = (self.audio_head,)
        for module in (self.audio_embedding, self.audio_head):
            module.to(dtype=config.params_dtype)
            for parameter in module.parameters():
                parameter.tensor_model_parallel = False
                parameter.partition_dim, parameter.partition_stride = -1, 1
                parameter.average_gradients_across_tp_domain = True
        if args.train_scope == "audio":
            self.text_embedding.requires_grad_(False)
            self.decoder.requires_grad_(False)

    def _embed_rows(self, batch):
        rows = batch.input_rows
        audio = rows[:, 0].eq(-100)
        text = self.text_embedding(rows[:, 0].masked_fill(audio, 0)[None], batch.position_ids[None])
        offsets = torch.arange(self.codebooks, device=rows.device) * self.audio_vocab
        codes = rows[:, 1:].clamp_min(0) + offsets
        encoded = self.audio_embedding(codes).sum(-2)[:, None]
        return torch.where(audio[:, None, None], encoded, text)

    def _score_actions(self, hidden, batch):
        logits = self.audio_head(hidden).reshape(-1, self.codebooks, self.audio_vocab).float()
        return (
            (logits / batch.temperatures[..., None]).log_softmax(-1).gather(-1, batch.targets[..., None]).squeeze(-1)
        )
