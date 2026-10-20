"""Frozen ReDimNet2 and JudgeCLAP inference; imports are local and pinned."""

import contextlib
import hashlib
import json
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .audio import SAMPLE_RATE, InvalidAudio


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_for(config, files):
    identity = {"config": config, "files": [(Path(p).name, file_digest(p)) for p in files]}
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class RedimBackend:
    def __init__(
        self,
        checkpoint,
        *,
        device="cuda",
        precision="float16",
        compile_model=False,
        attention="sdpa",
        batching="exact",
    ):
        from .vendor.redimnet2.redimnet2 import ReDimNet2Wrap

        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.precision = precision
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = data["model_config"]
        if (
            config.get("feat_type") != "tf"
            or config.get("F") != 72
            or config.get("hop_length", 160) != 160
            or not config.get("spec_params", {}).get("norm_signal")
            or config.get("emb_bn", False)
            or not config.get("spec_params", {}).get("do_preemph", True)
            or not config.get("return_2d_output")
            or config.get("embed_dim", 192) != 192
            or config.get("pooling_func", "ASTP") != "ASTP"
            or not config.get("global_context_att", True)
        ):
            raise ValueError("optimized frontend requires the pinned B6 TF-Mel checkpoint configuration")
        self.model = ReDimNet2Wrap(**data["model_config"]).eval()
        if self.model.backbone.time_stride != 4:
            raise ValueError("optimized backend requires time_stride=4")
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.to(self.device)
        self._thread = threading.local()
        self.batching = batching
        self.masks = None
        if batching == "masked":
            from .redim_masked import LengthMasks

            if attention != "sdpa":
                raise ValueError("masked batches require SDPA attention")
            self.masks = LengthMasks(self._thread)
            self.masks.install(self.model)
        if attention == "sdpa":
            self._enable_sdpa()
        self.core = self._core
        if compile_model:
            from torch._dynamo import config as dynamo_config

            # The finite time/batch grid has more variants than the default
            # eight-entry limit; do not silently fall back halfway through it.
            limit = "recompile_limit" if hasattr(dynamo_config, "recompile_limit") else "cache_size_limit"
            setattr(dynamo_config, limit, max(64, getattr(dynamo_config, limit)))
            self.core = torch.compile(self.core, dynamic=True, mode="default")
        source_files = sorted((Path(__file__).parent / "vendor/redimnet2").rglob("*.py"))
        version = version_for(
            {
                "model": "redimnet2",
                "precision": precision,
                "attention": attention,
                "batching": batching,
                "torch": torch.__version__,
                "frontend": "masked_full_audio_v1",
                "compile": compile_model,
            },
            [
                checkpoint,
                Path(__file__),
                Path(__file__).with_name("audio.py"),
                Path(__file__).with_name("redim_masked.py"),
                *source_files,
            ],
        )
        self.info = {
            "model": "redimnet2",
            "service_task": "speaker_embedding",
            "dimension": 192,
            "embedding_version": version,
            "dtype": "float32",
            "normalized": True,
            "audio_policy": {"sample_rate": SAMPLE_RATE, "channels": 1, "min_samples": 1600, "long_audio": "full"},
            "runtime": {
                "precision": precision,
                "attention": attention,
                "compile": compile_model,
                "batching": batching,
                "torch": torch.__version__,
            },
        }

    def _enable_sdpa(self):
        import types

        from .vendor.redimnet2.layers.attention import MultiHeadAttention

        masks = self.masks

        def forward(layer, hidden_states):
            batch, length, _ = hidden_states.shape
            if layer.qk_norm or layer.qk_rope:
                raise ValueError("this optimized checkpoint path expects no QK normalization or RoPE")
            q = layer._shape(layer.q_proj(hidden_states), length, batch)
            k = layer._shape(layer.k_proj(hidden_states), length, batch)
            v = layer._shape(layer.v_proj(hidden_states), length, batch)
            mask = masks.mask(length) if masks else None
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask[:, None, None, :] if mask is not None else None, dropout_p=0.0, is_causal=False
            )
            return layer.out_proj(out.transpose(1, 2).reshape(batch, length, layer.embed_dim))

        for module in self.model.modules():
            if isinstance(module, MultiHeadAttention):
                module.forward = types.MethodType(forward, module)

    def prepare_audio(self, wave):
        if len(wave) < 1600:
            raise InvalidAudio("too_short", num_samples=len(wave), min_samples=1600, sample_rate=SAMPLE_RATE)
        frames = ((len(wave) + 160 - 400) // 160 + 1) // 4 * 4
        key = 1 << (frames - 1).bit_length() if self.batching == "masked" else frames
        return (
            wave,
            key,
            key,
            {
                "input_seconds": len(wave) / SAMPLE_RATE,
                "processed_seconds": len(wave) / SAMPLE_RATE,
                "padded": False,
                "truncated": False,
            },
        )

    def _frontend(self, waves):
        """Batch only equal post-crop frame counts, preserving per-audio statistics.

        Zero-filled storage is masked BEFORE pre-emphasis/DFT, and mean removal
        sees each original feature length. The backbone never sees time padding.
        """
        lengths = torch.tensor([len(w) for w in waves], device=self.device)
        width = (self._thread.max_frames + 4) * 160 + 80 if getattr(self, "masks", None) else max(map(len, waves))
        array = np.zeros((len(waves), width), dtype=np.float32)
        for i, wave in enumerate(waves):
            array[i, : len(wave)] = wave
        x = torch.from_numpy(array).to(self.device)
        mask = torch.arange(x.shape[1], device=self.device)[None, :] < lengths[:, None]
        mean = x.sum(-1, keepdim=True) / lengths[:, None]
        centered = (x - mean) * mask
        std = (centered.square().sum(-1, keepdim=True) / lengths[:, None]).sqrt()
        x = centered / (std + self.model.spec.eps)
        previous = torch.cat((x[:, 1:2], x[:, :-1]), dim=-1)
        x = (x - 0.97 * previous) * mask
        spec = self.model.spec.torchfbank[2](x)
        spec = (spec + self.model.spec.eps).log()
        frame_lengths = (lengths + 160 - 400) // 160 + 1
        frame_mask = torch.arange(spec.shape[-1], device=self.device)[None, :] < frame_lengths[:, None]
        mean = (spec * frame_mask[:, None, :]).sum(-1, keepdim=True) / frame_lengths[:, None, None]
        spec = spec - mean
        # Keys are calculated on CPU to avoid a device synchronization here.
        frames = (
            self._thread.max_frames
            if getattr(self, "masks", None)
            else max(((len(w) + 160 - 400) // 160 + 1) // 4 * 4 for w in waves)
        )
        return spec[:, None, :, :frames]

    def _core(self, features):
        out = self.model.backbone(features).flatten(1, 2)
        # ASTP variance and the embedding head retain FP32 accumulation.
        with torch.autocast(self.device.type, enabled=False):
            out = self.model.linear(self.model.bn(self.model.pool(out.float())))
            return F.normalize(out.float(), p=2, dim=-1)

    def warmup(self, max_batch, max_cost):
        if self.batching != "masked":
            return
        # cuDNN execution-plan caches are thread-local. Warm the actual worker,
        # not a separate validation process, for the finite common shape grid.
        for frames in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            wave = np.random.default_rng(frames).normal(0, 0.03, max(1600, frames * 160 + 80)).astype(np.float32)
            for batch in [1, 2] if frames < 128 else [1, 2, 4, 8, 16, 32]:
                if batch <= max_batch and batch * frames <= max_cost:
                    self.encode_audio([wave] * batch)
            logging.info("ReDimNet2 warmup frames=%s complete", frames)

    def encode_audio(self, waves):
        stream_context = contextlib.nullcontext()
        if self.device.type == "cuda":
            if not hasattr(self._thread, "stream"):
                self._thread.stream = torch.cuda.Stream(device=self.device)
            stream_context = torch.cuda.stream(self._thread.stream)
        with torch.inference_mode(), stream_context:
            real_batch = len(waves)
            if self.masks:
                padded_batch = 1 << (real_batch - 1).bit_length()
                waves = list(waves) + [np.zeros(1600, dtype=np.float32)] * (padded_batch - real_batch)
                lengths = [((len(w) + 160 - 400) // 160 + 1) // 4 * 4 for w in waves]
                self._thread.frame_lengths = torch.tensor(lengths, device=self.device)
                self._thread.max_frames = 1 << (max(lengths) - 1).bit_length()
                self._thread.masks = {}
            try:
                features = self._frontend(waves)
                with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.precision != "float32"):
                    # Rare short/long inputs must not trigger minutes of online
                    # compilation and block every queued reward request.
                    core = self._core if self.masks and not 128 <= self._thread.max_frames <= 4096 else self.core
                    embeddings = core(features)
                return embeddings[:real_batch].cpu().numpy()
            finally:
                if self.masks:
                    del self._thread.frame_lengths, self._thread.masks


class JudgeBackend:
    def __init__(
        self,
        base,
        adapter,
        atom_stats,
        *,
        device="cuda",
        merge_lora=False,
        compile_model=False,
        feature_device="cpu",
        adapter_precision="float32",
        preprocessing="batch",
    ):
        import peft
        import transformers
        from peft import PeftModel
        from transformers import AutoProcessor

        from .vendor.judge_clap.collator import build_audio_batch, build_text_batch
        from .vendor.judge_clap.modeling import JudgeClapModel, add_embed_token

        self.device = torch.device(device)
        self._gpu_lock = threading.Lock()
        self._processor_lock = threading.Lock()
        self.feature_device = feature_device
        if preprocessing == "per_audio" and feature_device != "cpu":
            raise ValueError("per_audio preprocessing requires CPU features")
        self.preprocessing = preprocessing
        self.base = base
        self._preprocessors = threading.local()
        self.timings = {"audio_batches": 0, "preprocess_ms": 0.0, "encode_ms": 0.0}
        self.build_audio_batch, self.build_text_batch = build_audio_batch, build_text_batch
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)

        class EmbeddingPromptFilter(logging.Filter):
            def filter(self, record):
                return "System prompt modified, audio output may not work" not in record.getMessage()

        logging.getLogger().addFilter(EmbeddingPromptFilter())
        attention = "flash_attention_2"
        flash_version = None
        try:
            import flash_attn
            from flash_attn.bert_padding import unpad_input  # noqa: F401

            flash_version = flash_attn.__version__
        except ImportError:
            attention = "sdpa"
        model = JudgeClapModel.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True
        )
        add_embed_token(self.processor.tokenizer, model)
        model.load_atom_stats(atom_stats)
        model.reset_custom_params()
        model.config.use_cache = False
        peft_model = PeftModel.from_pretrained(
            model, adapter, is_trainable=False, autocast_adapter_dtype=adapter_precision == "float32"
        )
        self.model = (
            (peft_model.merge_and_unload(safe_merge=True) if merge_lora else peft_model.get_base_model())
            .eval()
            .to(device)
        )
        self.model.siglip_margin.float()
        self.scale, self.bias = (float(x.item()) for x in self.model.siglip_margin.effective())
        if compile_model:
            self.model.model = torch.compile(self.model.model, dynamic=True, mode="default")
        categories = json.loads((Path(__file__).parent / "vendor/judge_clap/categories.json").read_text())
        base_files = [
            Path(base) / name
            for name in [
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ]
            if (Path(base) / name).exists()
        ]
        base_files += sorted(Path(base).glob("*.safetensors")) + sorted(Path(base).glob("pytorch_model*.bin"))
        adapter_files = sorted(Path(adapter).glob("*.safetensors")) + [
            Path(adapter) / "adapter_config.json",
            Path(atom_stats),
        ]
        source_files = sorted((Path(__file__).parent / "vendor/judge_clap").glob("*.py"))
        version = version_for(
            {
                "model": "judge-clap",
                "base": str(Path(base).resolve()),
                "merge_lora": merge_lora,
                "compile": compile_model,
                "attention": attention,
                "categories": categories,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "flash_attn": flash_version,
                "feature_device": feature_device,
                "adapter_precision": adapter_precision,
                "preprocessing": preprocessing,
            },
            [*base_files, *adapter_files, Path(__file__), Path(__file__).with_name("audio.py"), *source_files],
        )
        self.info = {
            "model": "judge-clap",
            "service_task": "judge_clap",
            "dimension": 3584,
            "embedding_version": version,
            "dtype": "float32",
            "normalized": False,
            "scoring": {"type": "siglip_margin_v1", "scale": self.scale, "bias": self.bias},
            "supported_categories": categories,
            "query_versions": ["moss_tts_v2_revision6_instruction_v1"],
            "audio_policy": {
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "short_audio": "right_pad_zeros_to_2s",
                "long_audio": "keep_first_30s",
            },
            "runtime": {
                "precision": "bfloat16",
                "attention": attention,
                "merge_lora": merge_lora,
                "compile": compile_model,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "flash_attn": flash_version,
                "feature_device": feature_device,
                "adapter_precision": adapter_precision,
                "preprocessing": preprocessing,
            },
        }

    def prepare_audio(self, wave):
        from .vendor.judge_clap.collator import pad_crop

        size = len(wave)
        wave = pad_crop(wave)
        prepared = wave
        if self.preprocessing == "per_audio":
            from transformers import AutoProcessor

            if not hasattr(self._preprocessors, "processor"):
                processor = AutoProcessor.from_pretrained(self.base, local_files_only=True)
                processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<emb>"]})
                if processor.tokenizer.convert_tokens_to_ids("<emb>") != self.model.config.emb_token_ids[0]:
                    raise RuntimeError("preprocessor embedding token differs")
                self._preprocessors.processor = processor
            prepared = self.build_audio_batch(self._preprocessors.processor, ["input.wav"], [wave])
        return (
            prepared,
            math.ceil(len(wave) / (5 * SAMPLE_RATE)),
            math.ceil(len(wave) / 640) + 64,
            {
                "input_seconds": size / SAMPLE_RATE,
                "processed_seconds": len(wave) / SAMPLE_RATE,
                "padded": size < 32000,
                "truncated": size > 480000,
            },
        )

    def _encode(self, batch):
        with self._gpu_lock, torch.inference_mode():
            # Position construction and validation run on the processor's CPU
            # tensors, avoiding repeated GPU-to-CPU synchronizations in Python.
            labels = batch["labels"]
            token = self.model.config.emb_token_ids[0]
            matches = labels == token
            if not bool(matches.any(dim=1).all()):
                raise ValueError("text truncation removed the embedding marker")
            lengths = batch["feature_attention_mask"].sum(-1) if "feature_attention_mask" in batch else None
            batch["position_ids"] = self.model._rope_position_ids(batch["input_ids"], batch["attention_mask"], lengths)
            batch["embedding_positions"] = matches.int().argmax(dim=1) - 1
            batch = {k: v.to(self.device) for k, v in batch.items()}
            return self.model.encode(**batch).float().cpu().numpy()

    def encode_audio(self, waves):
        start = time.perf_counter()
        if self.preprocessing == "per_audio":
            batch = collate_audio_features(waves, self.processor.tokenizer.pad_token_id)
        else:
            with self._processor_lock:
                batch = self.build_audio_batch(
                    self.processor, ["input.wav"] * len(waves), waves, feature_device=self.feature_device
                )
        prepared = time.perf_counter()
        result = self._encode(batch)
        self.timings["audio_batches"] += 1
        self.timings["preprocess_ms"] += (prepared - start) * 1000
        self.timings["encode_ms"] += (time.perf_counter() - prepared) * 1000
        return result

    def validate_text(self, text):
        """Reject overlong complete prompts before they enter a shared GPU batch."""
        from .vendor.judge_clap.collator import MAX_TEXT_TOKENS, text_message

        with self._processor_lock:
            rendered = self.processor.apply_chat_template(
                [text_message(text)], tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(text=rendered, text_kwargs={"truncation": False, "padding": False})
        if len(inputs["input_ids"][0]) > MAX_TEXT_TOKENS:
            raise ValueError(f"text exceeds {MAX_TEXT_TOKENS} tokens including the chat template")

    def encode_text(self, texts):
        with self._processor_lock:
            batch = self.build_text_batch(self.processor, texts)
        return self._encode(batch)


def collate_audio_features(items, pad_token_id):
    """Left-pad token sequences; each CPU fbank already has the same 30 s width."""
    width = max(item["input_ids"].shape[-1] for item in items)
    batch = {}
    for key, padding in [("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)]:
        batch[key] = torch.cat([F.pad(item[key], (width - item[key].shape[-1], 0), value=padding) for item in items])
    for key in ["input_features", "feature_attention_mask"]:
        batch[key] = torch.cat([item[key] for item in items])
    return batch
