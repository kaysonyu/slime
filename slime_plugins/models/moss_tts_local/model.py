"""Megatron global Transformer with mossLite-compatible Local action replay."""

import torch
from torch import nn
from torch.nn import functional as F

from slime.backends.megatron_utils.model_provider import SpeechPolicyModel

from .local_transformer import LocalTransformer


class MossLocalModel(SpeechPolicyModel):
    def __init__(self, config, spec, model_config, args, *, pre_process=True, post_process=True, pg_collection=None):
        super().__init__(config=config, pg_collection=pg_collection)
        self._initialize_backbone(
            config, spec, args, pre_process=pre_process, post_process=post_process, pg_collection=pg_collection
        )
        self.model_config = model_config
        self.local_chunk_size = args.local_chunk_size
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(model_config.audio_vocab_size, config.hidden_size) for _ in range(model_config.n_vq)]
        )
        self.local_transformer = LocalTransformer(model_config)
        self.local_text_lm_head = nn.Linear(config.hidden_size, 2, bias=False)
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, model_config.audio_vocab_size, bias=False)
                for _ in range(model_config.n_vq)
            ]
        )
        self.action_modules = (self.local_transformer, self.local_text_lm_head, self.audio_lm_heads)
        for module in (self.audio_embeddings, self.local_transformer, self.local_text_lm_head, self.audio_lm_heads):
            module.to(dtype=config.params_dtype)
            for parameter in module.parameters():
                parameter.tensor_model_parallel = False
                parameter.partition_dim = -1
                parameter.partition_stride = 1
                parameter.average_gradients_across_tp_domain = True
        if args.train_scope == "audio":
            self.text_embedding.requires_grad_(False)
            self.decoder.requires_grad_(False)

    def _embed_rows(self, batch):
        rows = batch.input_rows
        hidden = self.text_embedding(rows[:, 0][None], batch.position_ids[None])
        for channel, embedding in enumerate(self.audio_embeddings):
            code = rows[:, channel + 1]
            valid = code.ne(self.model_config.audio_pad_id)
            hidden = hidden + (embedding(code.masked_fill(~valid, 0)) * valid[:, None])[:, None]
        return hidden

    def _score_actions(self, global_hidden, batch):
        outputs = []
        for start in range(0, len(global_hidden), self.local_chunk_size):
            end = min(start + self.local_chunk_size, len(global_hidden))
            targets = batch.targets[start:end]
            inputs = [global_hidden[start:end]]
            for depth in range(1, self.model_config.n_vq):
                inputs.append(self.audio_embeddings[depth - 1](targets[:, depth]))
            local = self.local_transformer(torch.stack(inputs, dim=1))
            logits = self.local_text_lm_head(local[:, 0]).float()
            parts = [
                (logits / batch.temperatures[start:end, :1]).log_softmax(-1).gather(-1, targets[:, :1]).squeeze(-1)
            ]
            for depth, head in enumerate(self.audio_lm_heads):
                logits = F.linear(local[:, depth], head.weight).float()
                logprobs = (logits / batch.temperatures[start:end, depth + 1, None]).log_softmax(-1)
                parts.append(logprobs.gather(-1, targets[:, depth + 1, None]).squeeze(-1))
            outputs.append(torch.stack(parts, dim=-1))
        return torch.cat(outputs)
