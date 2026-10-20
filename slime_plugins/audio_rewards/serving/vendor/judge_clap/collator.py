"""批构建: 音频加载 (torchaudio ffmpeg) + AuroLA chat 模板 + processor 分批.

音频批与文本批分开过 processor (避免文本被 padding 到音频序列长度).
音频放 user 内容最前; assistant 固定 "<emb>."; labels 中 pad 位置置 -100.
"""
import os

import numpy as np
import torch

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = 480000   # 30s @ 16kHz
MIN_AUDIO_SAMPLES = 32000    # 2s
MAX_TEXT_TOKENS = 1024


def audio_message(audio_path: str) -> list[dict]:
    return [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "\nSummarize above audio in one word: "},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


def text_message(text: str) -> list[dict]:
    return [
        {"role": "user", "content": [
            {"type": "text", "text": f"{text}\nSummarize above sentence in one word: "},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


def load_audio_mono(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """解码 -> 单声道 float32 -> 重采样到 16k. 失败抛异常由调用方兜底.

    首选 torchaudio ffmpeg 后端 (qz 镜像可用); 失败回退 ffmpeg 子进程.
    """
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path), backend="ffmpeg")
        wav = wav.to(dtype=torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != sample_rate:
            wav = torchaudio.functional.resample(wav, int(sr), sample_rate)
        a = wav.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        a = _load_audio_ffmpeg_subprocess(path, sample_rate)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{path}: empty waveform")
    return a


def _load_audio_ffmpeg_subprocess(path: str, sample_rate: int) -> np.ndarray:
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


def pad_crop(a: np.ndarray) -> np.ndarray:
    if len(a) < MIN_AUDIO_SAMPLES:
        out = np.zeros(MIN_AUDIO_SAMPLES, dtype=np.float32)
        out[: len(a)] = a
        return out
    if len(a) > MAX_AUDIO_SAMPLES:
        return a[:MAX_AUDIO_SAMPLES]
    return a


def _make_labels(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return labels


def build_audio_batch(processor, audio_paths: list[str], waveforms: list[np.ndarray], feature_device: str = "cpu") -> dict:
    msgs = [audio_message(p) for p in audio_paths]
    texts = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=texts, audio=waveforms, return_tensors="pt", padding=True,
                       audio_kwargs={"device": feature_device})
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "input_features": inputs["input_features"],
        "feature_attention_mask": inputs["feature_attention_mask"],
        "labels": _make_labels(inputs["input_ids"], processor.tokenizer.pad_token_id),
    }


def build_text_batch(processor, texts_raw: list[str]) -> dict:
    msgs = [text_message(t) for t in texts_raw]
    texts = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=texts, return_tensors="pt",
        text_kwargs={"max_length": MAX_TEXT_TOKENS, "truncation": True, "padding": True},
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": _make_labels(inputs["input_ids"], processor.tokenizer.pad_token_id),
    }

