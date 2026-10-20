"""CPU protocol, batching and preprocessing invariants for frozen services."""

import asyncio
import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.prism.reward_services.bench.benchmark import summarize

from slime_plugins.audio_rewards.serving.audio import InvalidAudio
from slime_plugins.audio_rewards.serving.batching import Batcher
from slime_plugins.audio_rewards.serving.server import PROTOCOL, create_app

NUM_GPUS = 0


class Stub:
    def __init__(self):
        self.info = {
            "model": "stub",
            "service_task": "judge_clap",
            "dimension": 4,
            "embedding_version": "test-v1",
            "dtype": "float32",
            "normalized": False,
            "supported_categories": ["Speed", "Gender"],
            "query_versions": ["moss_tts_v2_revision6_instruction_v1"],
            "scoring": {"type": "siglip_margin_v1", "scale": 0.3, "bias": 0.2},
        }
        self.text_calls = []

    def prepare_audio(self, wave):
        if len(wave) < 10:
            raise InvalidAudio("too_short")
        return wave, len(wave), 1, {"input_seconds": len(wave) / 16000, "processed_seconds": len(wave) / 16000}

    def encode_audio(self, waves):
        return np.array([[float(w.mean()), 0.2, 0.3, 0.4] for w in waves], dtype=np.float32)

    def validate_text(self, text):
        pass

    def encode_text(self, texts):
        self.text_calls.extend(texts)
        return np.array([[len(t) / 100, 0.1, 0.3, 0.2] for t in texts], dtype=np.float32)


def wav_payload(seconds=1):
    stream = io.BytesIO()
    sf.write(stream, np.zeros(int(seconds * 16000), dtype=np.float32), 16000, format="WAV")
    return {"data_b64": base64.b64encode(stream.getvalue()).decode(), "suffix": ".wav"}


def request(inputs, **extra):
    return {"request_id": "r1", "expected_embedding_version": "test-v1", "inputs": inputs, **extra}


def test_audio_batch_order_upload_and_invalid_items():
    with TestClient(create_app(Stub)) as client:
        payload = request(
            [
                {"id": "long", "audio": wav_payload(1)},
                {"id": "empty", "audio": wav_payload(0)},
                {"id": "short", "audio": wav_payload(0.5)},
            ]
        )
        result = client.post("/embed_audio", json=payload)
        assert result.status_code == 200
        body = result.json()
        assert body["protocol"] == PROTOCOL and body["request_id"] == "r1"
        assert [x["id"] for x in body["results"]] == ["long", "empty", "short"]
        assert body["results"][1] == {
            "id": "empty",
            "status": "invalid_audio",
            "reason": "empty_audio",
            "num_samples": 0,
            "sample_rate": 16000,
        }
        assert len(body["results"][0]["embedding"]) == 4


def test_version_lock_and_text_categories():
    with TestClient(create_app(Stub)) as client:
        assert client.get("/health").json()["status"] == "ready"
        assert client.get("/info").json()["normalized"] is False
        assert (
            client.post(
                "/embed_text", json=request([{"id": "1", "text": "Speed: fast"}], expected_embedding_version="old")
            ).status_code
            == 409
        )
        for text in ["Loudness: high", "Speed:", "Speed: <emb>", "unknown"]:
            assert client.post("/embed_text", json=request([{"id": "1", "text": text}])).status_code == 422
        response = client.post("/embed_text", json=request([{"id": "1", "text": "Speed: fast"}]))
        assert response.status_code == 200 and len(response.json()["results"][0]["embedding"]) == 4


@pytest.fixture
def judge_text_backend():
    """Exercise the real text collator/guard with a local tokenizer and no model weights."""
    import threading

    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from slime_plugins.audio_rewards.serving.backends import JudgeBackend
    from slime_plugins.audio_rewards.serving.vendor.judge_clap.collator import build_text_batch

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "x": 2, "<emb>": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=["<emb>"],
        chat_template=(
            "{% for message in messages %}{% for part in message['content'] %}"
            "{{ part['text'] }} {% endfor %}{% endfor %}"
        ),
    )

    class TextProcessor:
        def __init__(self):
            self.tokenizer = tokenizer

        def apply_chat_template(self, *args, **kwargs):
            return self.tokenizer.apply_chat_template(*args, **kwargs)

        def __call__(self, *, text, text_kwargs, return_tensors=None):
            return self.tokenizer(text, return_tensors=return_tensors, **text_kwargs)

    class TextBackend(Stub):
        validate_text = JudgeBackend.validate_text

        def encode_text(self, texts):
            self.text_calls.extend(texts)
            return JudgeBackend._encode(self, build_text_batch(self.processor, texts))

    backend = TextBackend()
    backend.processor = TextProcessor()
    backend._processor_lock = threading.Lock()
    backend._gpu_lock = threading.Lock()
    backend.device = torch.device("cpu")
    backend.model = SimpleNamespace(
        config=SimpleNamespace(emb_token_ids=[tokenizer.convert_tokens_to_ids("<emb>")]),
        _rope_position_ids=lambda ids, mask, lengths: torch.zeros_like(ids),
        encode=lambda **batch: torch.zeros((len(batch["input_ids"]), 4)),
    )
    return backend


@pytest.mark.parametrize(("words", "expected_status"), [(1014, 200), (1015, 200), (1016, 422)])
def test_judge_text_limit_includes_template_and_accepts_exact_boundary(judge_text_backend, words, expected_status):
    # The query has words + 1 tokens; the full collator prompt adds 8 more.
    # All three raw queries fit 1024 tokens and 4096 characters on their own.
    text = "Speed:" + " x" * words
    with TestClient(create_app(lambda: judge_text_backend), raise_server_exceptions=False) as client:
        response = client.post("/embed_text", json=request([{"id": "boundary", "text": text}]))
        assert response.status_code == expected_status
        if expected_status == 422:
            assert "1024 tokens including the chat template" in response.json()["detail"]
            assert text not in judge_text_backend.text_calls
        else:
            assert response.json()["results"][0]["status"] == "success"


@pytest.mark.parametrize("mixed_request", [False, True])
def test_overlong_text_cannot_fail_concurrent_requests_in_shared_batch(judge_text_backend, mixed_request):
    import httpx

    overlong = "Speed:" + " x" * 1500
    bad_inputs = [{"id": "long", "text": overlong}]
    if mixed_request:
        bad_inputs.append({"id": "same-request", "text": "Speed: slow"})
    app = create_app(lambda: judge_text_backend, window_ms=100)

    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                good, bad, other = await asyncio.gather(
                    client.post("/embed_text", json=request([{"id": "good", "text": "Speed: fast"}])),
                    client.post("/embed_text", json=request(bad_inputs)),
                    client.post("/embed_text", json=request([{"id": "other", "text": "Gender: female"}])),
                )
                assert [good.status_code, bad.status_code, other.status_code] == [200, 422, 200]
                assert "1024 tokens including the chat template" in bad.json()["detail"]
                assert overlong not in judge_text_backend.text_calls
                assert app.state.text.stats["failed_items"] == 0
                assert app.state.active == 0

    asyncio.run(run())


def test_bad_audio_transport_is_a_request_error():
    with TestClient(create_app(Stub)) as client:
        for audio in [{}, {"path": "/missing", "data_b64": "eA=="}, {"data_b64": "!"}]:
            response = client.post("/embed_audio", json=request([{"id": "1", "audio": audio}]))
            assert response.status_code == 400


def test_batcher_never_combines_incompatible_shapes_or_exceeds_budget():
    batches = []

    def encode(values):
        batches.append(values)
        assert len({x[0] for x in values}) == 1
        assert len(values) <= 2
        return values

    async def run():
        batcher = Batcher(encode, max_batch=4, max_cost=4, window_ms=5)
        batcher.start()
        values = [(key, i) for i, key in enumerate([1, 2, 1, 1, 2, 1])]
        results = await asyncio.gather(*(batcher.submit(value, value[0], 2) for value in values))
        await batcher.close()
        assert results == values
        assert sum(map(len, batches)) == len(values)

    asyncio.run(run())


def test_redim_frontend_matches_independent_upstream_features():
    import torch

    pytest.importorskip("scipy")
    pytest.importorskip("torchaudio")
    from slime_plugins.audio_rewards.serving.backends import RedimBackend
    from slime_plugins.audio_rewards.serving.vendor.redimnet2.layers.features_tf import TFMelBanks

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        backend = RedimBackend.__new__(RedimBackend)
        backend.device = torch.device("cpu")
        backend.model = SimpleNamespace(spec=TFMelBanks(n_mels=72, norm_signal=True, do_preemph=True).eval())
        rng = np.random.default_rng(4)
        waves = [
            (rng.standard_normal(size) * scale + offset).astype(np.float32)
            for size, scale, offset in [(16000, 0.03, 0.1), (15840, 0.1, -0.2), (15680, 0.5, 0.3)]
        ]
        with torch.inference_mode():
            actual = backend._frontend(waves)
            expected = torch.cat([backend.model.spec(torch.from_numpy(x)[None])[..., :96] for x in waves])[:, None]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)
        # Last possible waveform length in a 256-frame bucket includes 79
        # samples beyond (256 + 4) * hop. Keep them in waveform normalization.
        backend.masks = object()
        backend._thread = SimpleNamespace(max_frames=256)
        boundary = (rng.standard_normal(41679) * 0.03).astype(np.float32)
        with torch.inference_mode():
            actual = backend._frontend([boundary])
            expected = backend.model.spec(torch.from_numpy(boundary)[None])[None, ..., :256]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)
    finally:
        torch.set_num_threads(previous)


def test_benchmark_excludes_drain_from_throughput_but_keeps_tail_latency():
    records = [
        {"start": 1.0, "end": 2.0, "success_items": 2, "success_audio_seconds": 10, "invalid_items": 0, "error": None},
        {
            "start": 2.0,
            "end": 12.0,
            "success_items": 2,
            "success_audio_seconds": 10,
            "invalid_items": 0,
            "error": None,
        },
    ]
    result = summarize(records, 1, 11)
    assert result["successful_audio_per_s"] == 0.2
    assert result["successful_audio_items"] == 2
    assert result["completed_after_window"] == 1
    assert result["latency_ms"]["p99"] > 9000


def test_judge_client_uses_raw_embeddings_and_reuses_text_cache(tmp_path, monkeypatch):
    from slime_plugins.audio_rewards import client as audio_client
    from slime_plugins.prism import reward_adapter

    backend = Stub()
    path = tmp_path / "one.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    with TestClient(create_app(lambda: backend)) as client:

        async def get(url, **kwargs):
            return client.get("/info").json()

        async def post(url, payload, **kwargs):
            response = client.post("/" + url.rsplit("/", 1)[-1], json=payload)
            response.raise_for_status()
            return response.json()

        monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(get))
        monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(post))
        scorer = reward_adapter.PrismRewardAdapter(
            {"judge_clap": {"endpoint": "http://test", "api": "voxrl_embedding"}}
        )
        sample = SimpleNamespace(
            metadata={
                "omni_frames": 1,
                "omni_audio_path": str(path),
                "omni_prompt": {"global_instruction": {"Speed": "fast | slow", "Appearance": "tall"}},
            }
        )

        async def run():
            first = (await scorer.score_components(sample, ["judge_clap"]))["judge_clap"]
            calls = len(backend.text_calls)
            second = (await scorer.score_components(sample, ["judge_clap"]))["judge_clap"]
            assert len(backend.text_calls) == calls
            assert first["raw"] == second["raw"]
            assert first["details"]["queries"] == ["Speed: fast", "Speed: slow"]
            assert first["details"]["dropped_categories"] == ["Appearance"]
            a, t = backend.encode_audio([np.zeros(16000, dtype=np.float32)])[0], backend.encode_text(
                ["Speed: fast", "Speed: slow"]
            )
            expected = 1 / (1 + np.exp(-(0.3 * (t @ a - np.square(t).sum(-1)) + 0.2)))
            np.testing.assert_allclose(first["details"]["attribute_scores"], expected, rtol=1e-6)

        asyncio.run(run())


def test_masked_network_matches_full_utterances_without_time_padding():
    import threading

    import torch

    pytest.importorskip("scipy")
    pytest.importorskip("torchaudio")
    from slime_plugins.audio_rewards.serving.backends import RedimBackend
    from slime_plugins.audio_rewards.serving.redim_masked import LengthMasks
    from slime_plugins.audio_rewards.serving.vendor.redimnet2.redimnet2 import ReDimNet2Wrap

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(7)
        backend = RedimBackend.__new__(RedimBackend)
        backend.device = torch.device("cpu")
        backend._thread = threading.local()
        backend.model = ReDimNet2Wrap(
            F=8,
            C=8,
            causal="none",
            agg_gnorm=True,
            out_channels=8,
            return_2d_output=True,
            stages_setup=[((1, 1), 1, 1, [(3, 3)], 4), ((2, 2), 1, 1, [(3, 3)], 4)],
            embed_dim=8,
        ).eval()
        backend.masks = LengthMasks(backend._thread)
        backend.masks.install(backend.model)
        backend._enable_sdpa()
        short, long = torch.randn(1, 1, 8, 32), torch.randn(1, 1, 8, 64)
        with torch.inference_mode():
            expected = torch.cat([backend._core(short), backend._core(long)])
            backend._thread.frame_lengths = torch.tensor([32, 64])
            backend._thread.max_frames = 64
            backend._thread.masks = {}
            actual = backend._core(torch.cat([torch.nn.functional.pad(short, (0, 32)), long]))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    finally:
        torch.set_num_threads(previous)


def test_precomputed_cache_is_versioned_and_read_only(tmp_path):
    import json
    import sqlite3

    from slime_plugins.audio_rewards.cache import read_embeddings

    path = tmp_path / "texts.sqlite"
    identity = {"embedding_version": "v1", "dimension": 4, "normalized": False}
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, vector BLOB)")
        db.execute("INSERT INTO metadata VALUES ('identity', ?)", (json.dumps(identity),))
        db.execute("INSERT INTO embeddings VALUES (?, ?)", ("Speed: fast", np.arange(4, dtype="<f4").tobytes()))
    before = path.read_bytes()
    assert read_embeddings(path, identity, ["Speed: fast", "Speed: slow"]) == {"Speed: fast": [0.0, 1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match="embedding_version"):
        read_embeddings(path, {**identity, "embedding_version": "v2"}, ["Speed: fast"])
    assert path.read_bytes() == before


def test_judge_prepared_audio_collation_preserves_masks_and_labels():
    import torch

    from slime_plugins.audio_rewards.serving.backends import collate_audio_features

    items = []
    for length in [3, 5]:
        ids = torch.arange(10, 10 + length)[None]
        items.append(
            {
                "input_ids": ids,
                "labels": ids.clone(),
                "attention_mask": torch.ones_like(ids),
                "input_features": torch.full((1, 2, 8), float(length)),
                "feature_attention_mask": torch.ones(1, 8, dtype=torch.long),
            }
        )
    result = collate_audio_features(items, 7)
    assert result["input_ids"][0].tolist() == [7, 7, 10, 11, 12]
    assert result["labels"][0].tolist() == [-100, -100, 10, 11, 12]
    assert result["attention_mask"][0].tolist() == [0, 0, 1, 1, 1]
    torch.testing.assert_close(result["input_features"], torch.cat([x["input_features"] for x in items]))


def test_audio_client_accepts_explicit_queries_without_prism_metadata(tmp_path, monkeypatch):
    from slime_plugins.audio_rewards import client as audio_client

    backend = Stub()
    backend.info["supported_categories"] = ["Custom attribute"]
    backend.info["query_versions"] = ["custom-v1"]
    path = tmp_path / "explicit.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    with TestClient(create_app(lambda: backend)) as service:

        async def get(url, **kwargs):
            return service.get("/info").json()

        async def post(url, payload, **kwargs):
            result = service.post("/" + url.rsplit("/", 1)[-1], json=payload)
            result.raise_for_status()
            return result.json()

        monkeypatch.setattr(audio_client.RewardHTTPClient, "get", staticmethod(get))
        monkeypatch.setattr(audio_client.RewardHTTPClient, "post", staticmethod(post))
        client = audio_client.AudioRewardClient({"judge_clap": {"api": "voxrl_embedding", "endpoint": "http://test"}})
        result = asyncio.run(
            client.judge_clap(path, ["Custom attribute: warm"], "explicit", query_version="custom-v1")
        )
        assert result["status"] == "success" and 0 <= result["raw"] <= 1
        assert result["details"]["queries"] == ["Custom attribute: warm"]
