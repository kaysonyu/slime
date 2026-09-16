"""Differentiable NanoGPT2 Local decoder; math follows mossLite's Local model.

Frames are independent batch rows. The causal sequence dimension is RVQ depth,
so this module has no global-time CP communication.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LocalAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(hidden, 3 * hidden)
        self.c_proj = nn.Linear(hidden, hidden)


class LocalMLP(nn.Module):
    def __init__(self, hidden, inner):
        super().__init__()
        self.fc_in = nn.Linear(hidden, inner)
        self.fc_out = nn.Linear(inner, hidden)

    def forward(self, x):
        return self.fc_out(F.silu(self.fc_in(x)))


class LocalBlock(nn.Module):
    def __init__(self, hidden, heads, inner, eps):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden, eps=eps)
        self.attn = LocalAttention(hidden, heads)
        self.ln_2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = LocalMLP(hidden, inner)


class LocalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        local = config.local
        self.heads = local["num_attention_heads"]
        self.rope_base = config.local_rope_base
        self.h = nn.ModuleList(
            [
                LocalBlock(config.hidden_size, self.heads, local["intermediate_size"], local["layer_norm_eps"])
                for _ in range(local["num_hidden_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=local["layer_norm_eps"])

    def forward(self, x):
        rows, depth, hidden = x.shape
        head_dim = hidden // self.heads
        freq = torch.outer(
            torch.arange(depth, device=x.device, dtype=torch.float32),
            self.rope_base ** (-torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim),
        )
        cos = freq.cos().repeat_interleave(2, -1).to(x.dtype)[None, None]
        sin = freq.sin().repeat_interleave(2, -1).to(x.dtype)[None, None]

        def rotate(value):
            rotated = torch.stack((-value[..., 1::2], value[..., ::2]), dim=-1).flatten(-2)
            return value * cos + rotated * sin

        for block in self.h:
            q, k, v = block.attn.c_attn(block.ln_1(x)).chunk(3, dim=-1)
            q, k, v = [t.reshape(rows, depth, self.heads, head_dim).transpose(1, 2) for t in (q, k, v)]
            attention = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
            x = x + block.attn.c_proj(attention.transpose(1, 2).reshape(rows, depth, hidden))
            x = x + block.mlp(block.ln_2(x))
        return self.ln_f(x)
