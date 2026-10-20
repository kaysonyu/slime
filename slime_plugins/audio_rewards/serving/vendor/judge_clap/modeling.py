"""模型: Qwen2.5-Omni Thinker 双塔编码 + 投影覆盖率判决 (单通道 SigLIP, 零几何变换).

- 嵌入提取照抄 AuroLA: 音频特征 masked_scatter 进占位 token -> LLM -> 最后一层
  hidden state -> 取 <emb> token 前一个位置的向量, 原始隐状态直接用, 无任何包装
  (无 per-sample LN, 无校准仿射, 无 softplus 原子化 —— 全部已被证伪或退役).
- 判决量 (投影余量): M(t,a) = <a, t> - ||t||^2.
  符号即真值: 把 a 分解为 alpha*\\hat t + u (u ⊥ t) 则 M = ||t||(alpha - ||t||),
  alpha > ||t|| 供过于求为正, alpha = ||t|| 恰好满足在边界, 不足为负.
  与覆盖率同族 (M = ||t||^2 (C - 1)) 但不做除法: 两项对 t 同为二次齐次 =>
  "缩 t 涨分"的气泵结构性不存在 (cov 版实测 t_norm 泄漏到地板, 已证伪);
  共模 mu 在 <a,t> 与 ||t||^2 两侧同现, 期望自动抵消, 无任何语料平移/补丁.
- 初始化: logit = e^s * M + b. s = ln(10 / sigma_M) (sigma_M = 校准批上无关对 M 的 std,
  把初始 logits 压到 ±10 量级; M 的量纲是 ||h||^2, 这一步是纯单位换算), b = 0.
- 梯度体检: dM/dt = a - 2t 有界; dM/da = t 有界; 无除法无临界域.
- 判定 SigLIP 外层/标量 fp32 纪律不变; 集合嵌套 z 判定逻辑不变 (phase 1).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

EMB_TOKEN = "<emb>"


def _hidden_size(config) -> int:
    for attr in ("text_config", "thinker_config"):
        c = getattr(config, attr, None)
        if c is not None and hasattr(c, "hidden_size"):
            return c.hidden_size
    return config.hidden_size


class SiglipHead(nn.Module):
    """SigLIP 可学温度与偏差: 生效 logit = exp(logit_scale + scale_shift) * x + logit_bias + bias_shift.

    判定通道传 x=-S (缺口) 或 x=margin M, 对齐通道传 x=cos.
    scale_shift/bias_shift 是 buffers, 存首个 step 的运行时校准值而非直接改参数:
    DeepSpeed ZeRO 为可训参数建 fp32 主权重, optimizer step 用主权重覆盖工作副本,
    compute_loss 里 fill_ 进去的校准会被静默回滚 (maskedcos_res / cal_bal 两代 run 实测
    main_bias 恒 -10 即此病). buffers 不受 ZeRO 主权重管理, 且随 modules_to_save 一起
    进 adapter/checkpoint, resume 后依然有效.
    另: 两个标量的"学习"不走主优化器 —— ZeRO bf16 会把全 module 的工作副本钉成 bf16,
    小更新被 ULP 舍入吞掉 (fixw run 实测 bias fp32 主权重 2000 步挪 +2.9e-4, bf16 本体
    48.25 纹丝不动). 现由 JudgeClapTrainer 的 fp32 旁路 AdamW 直接在参数本体上步进."""

    def __init__(self, init_scale: float = 10.0, init_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.register_buffer("scale_shift", torch.tensor(0.0))
        self.register_buffer("bias_shift", torch.tensor(0.0))

    def effective(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(scale, bias) 生效值."""
        scale = (self.logit_scale + self.scale_shift).exp()
        bias = self.logit_bias + self.bias_shift
        return scale, bias


class JudgeClapModel(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("m_std", torch.tensor(1.0))   # 校准: 无关对 M 的 std (e^s = 10/m_std)
        self.register_buffer("m_mean", torch.tensor(0.0))  # 校准: 无关对 M 的均值 (b = -e^s*m_mean, 吸收双塔共模常数差)
        self.siglip_margin = SiglipHead(init_scale=1.0, init_bias=0.0)  # reset 里 s=ln(10/m_std)
        self.siglip_res = SiglipHead(init_scale=1.0, init_bias=0.0)     # 残差通道头 (q3/q4), 同纪律

    def load_atom_stats(self, npz_path: str) -> None:
        """读 scripts/compute_atom_stats.py 的产出 (m_std; 旧文件缺键则回落 1.0)."""
        data = np.load(npz_path)
        self._margin_stats = {"m_std": float(data["m_std"]) if "m_std" in data.files else 1.0,
                              "m_mean": float(data["m_mean"]) if "m_mean" in data.files else 0.0}

    def reset_custom_params(self) -> None:
        """from_pretrained 会按 transformers 的缺键逻辑重初始化 checkpoint 没有的键
        (覆盖 __init__ 与 load_atom_stats 灌入的值), 因此加载权重后必须显式重放.
        校准 buffer 与 siglip 标量强制 fp32, 且必须在 cast 之后再填值."""
        st = getattr(self, "_margin_stats", None) or {}
        m_std = max(float(st.get("m_std", 1.0)), 1e-12)
        m_mean = float(st.get("m_mean", 0.0))
        with torch.no_grad():
            for name, val in (("m_std", m_std), ("m_mean", m_mean)):
                buf = getattr(self, name)
                if buf.dtype != torch.float32:
                    buf.data = buf.data.float()
                buf.fill_(val)
        self.siglip_margin.to(torch.float32)
        self.siglip_res.to(torch.float32)
        with torch.no_grad():
            scale = 10.0 / m_std
            self.siglip_margin.logit_scale.fill_(math.log(scale))
            # b 吸收无关对的平均余量 (双塔共模失配的常数项): 无关对初始 logits = 0
            self.siglip_margin.logit_bias.fill_(-scale * m_mean)
        with torch.no_grad():
            # 残差头沿用同尺度纪律, logits 在批内自校准 (trainer 首个 step) 覆盖更精
            self.siglip_res.logit_scale.fill_(math.log(scale))
            self.siglip_res.logit_bias.fill_(0.0)
        with torch.no_grad():
            for head in (self.siglip_margin, self.siglip_res):
                head.scale_shift.zero_()   # 运行时校准位移清零 (from_pretrained 会覆盖缺键)
                head.bias_shift.zero_()

    # ---- 以下 get_placeholder_mask / encode 复刻 AuroLA model/qwen_omni.py 的前向路径 ----

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
            special_audio_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            ).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask, special_video_mask, special_audio_mask

    @torch.no_grad()
    def _rope_position_ids(self, input_ids, attention_mask, audio_feature_lengths):
        position_ids, rope_deltas = self.get_rope_index(
            input_ids, None, None, attention_mask, None, audio_feature_lengths, None
        )
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        rope_deltas = rope_deltas - delta0
        self.rope_deltas = rope_deltas
        return position_ids

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        embedding_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回每条序列 <emb> 前一个位置的最后一层 hidden state (未归一化)."""
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
            )
            # transformers 5.x 返回 BaseModelOutputWithPooling, 4.57 返回 tensor
            if hasattr(audio_features, "last_hidden_state"):
                audio_features = audio_features.last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        audio_feature_lengths = (
            torch.sum(feature_attention_mask, dim=1) if feature_attention_mask is not None else None
        )
        if position_ids is None:
            position_ids = self._rope_position_ids(input_ids, attention_mask, audio_feature_lengths)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
        )
        hidden_states = outputs[0]

        if embedding_positions is not None:
            return hidden_states[torch.arange(len(embedding_positions), device=hidden_states.device), embedding_positions]

        emb_id = self.config.emb_token_ids[0]
        if not bool((labels == emb_id).any(dim=1).all()):
            raise ValueError("部分序列缺少 <emb> token (可能被 max_length 截断)")
        emb_pos = torch.argmax((labels == emb_id).int(), dim=1)
        return hidden_states[torch.arange(len(emb_pos), device=hidden_states.device), emb_pos - 1]

    # ---- 投影余量判决 ----

    def pairwise_margin(self, audio_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """M(t,a) = <a, t> - ||t||^2. (N, d) x (M, d) -> (N, M) fp32. 一层 matmul, 无需分块."""
        a = audio_emb.float()
        t = text_emb.float()
        tn2 = t.pow(2).sum(-1)  # (M,)
        return (a @ t.t()) - tn2.unsqueeze(0)


# ---- 加载 / LoRA / 审计 ----

def add_embed_token(tokenizer, model, emb_token: str = EMB_TOKEN) -> list[int]:
    """加 <emb> token 并 resize; 校验新行非零 (transformers mean-resizing), 否则手动设为词表均值."""
    if emb_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([emb_token])
        model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.convert_tokens_to_ids([emb_token])
    model.config.emb_token_ids = ids
    emb = model.get_input_embeddings().weight
    new_id = ids[0]
    with torch.no_grad():
        if emb[new_id].abs().sum().item() == 0.0:
            emb[new_id] = emb[:new_id].mean(dim=0)
    return ids

