"""Audio-to-embedding orchestration and cosine similarity business logic."""

from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .audio import PREPROCESS_VERSION, AudioInputError, AudioLoader, FileSignature, LoadedAudio
from .backend import EmbeddingBackend
from .buckets import BucketPolicy
from .cache import CacheKey, CacheValue, ReferenceEmbeddingCache
from .config import MAX_REQUEST_ITEMS


@dataclass(frozen=True, slots=True)
class ItemError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    item_id: str
    similarity: float | None
    reward_similarity: float | None
    reference_cache_hit: bool
    reference_bucket_seconds: int | None
    candidate_bucket_seconds: int | None
    error: ItemError | None


@dataclass(slots=True)
class _Embedded:
    embedding: torch.Tensor | None = None
    bucket_seconds: int | None = None
    effective_samples: int | None = None
    truncated: bool | None = None
    cache_hit: bool = False
    error: ItemError | None = None


def _input_error(exc: AudioInputError | FileNotFoundError | PermissionError) -> ItemError:
    if isinstance(exc, AudioInputError):
        messages = {
            "path_access_denied": "audio path is not permitted",
            "audio_decode_failed": "audio could not be decoded",
            "audio_too_short": "audio is shorter than the minimum duration",
        }
        return ItemError(exc.code, messages.get(exc.code, "audio input is invalid"))
    if isinstance(exc, FileNotFoundError):
        return ItemError("file_not_found", "audio file was not found")
    return ItemError("permission_denied", "audio file is not readable")


class EmbeddingService:
    """Coordinate audio loading, cache fills, embedding, and cosine scoring."""

    def __init__(
        self,
        loader: AudioLoader,
        backend: EmbeddingBackend,
        bucket_policy: BucketPolicy | None = None,
        reference_cache: ReferenceEmbeddingCache | None = None,
        max_items_per_request: int = MAX_REQUEST_ITEMS,
        audio_workers: int = 16,
    ) -> None:
        if not 1 <= max_items_per_request <= MAX_REQUEST_ITEMS:
            raise ValueError(f"max_items_per_request must be between 1 and {MAX_REQUEST_ITEMS}")
        if audio_workers <= 0:
            raise ValueError("audio_workers must be positive")
        self.loader = loader
        self.backend = backend
        self.bucket_policy = bucket_policy or BucketPolicy()
        self.reference_cache = (
            reference_cache if reference_cache is not None else ReferenceEmbeddingCache()
        )
        self.max_items_per_request = max_items_per_request
        self._audio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=audio_workers, thread_name_prefix="audio-frontend"
        )

    @property
    def is_ready(self) -> bool:
        return bool(self.backend.is_ready)

    def close(self) -> None:
        self._audio_executor.shutdown(wait=True, cancel_futures=False)
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    def _cache_key(self, signature: FileSignature) -> CacheKey:
        return CacheKey(
            signature=signature,
            backend_fingerprint=self.backend.fingerprint,
            preprocess_version=PREPROCESS_VERSION,
            bucket_version=self.bucket_policy.version,
        )

    def _infer_loaded(self, loaded: Sequence[LoadedAudio]) -> list[_Embedded]:
        outputs = [_Embedded() for _ in loaded]
        groups = self.bucket_policy.group_indices([item.num_samples for item in loaded])
        for bucket, indices in groups.items():
            for start in range(0, len(indices), bucket.max_batch):
                chunk = indices[start : start + bucket.max_batch]
                batch, lengths = self.bucket_policy.pad(
                    [loaded[index].waveform for index in chunk], bucket
                )
                embeddings = self.backend.embed(batch, lengths)
                if embeddings.shape[0] != len(chunk):
                    raise RuntimeError("backend result count does not match batch")
                for index, embedding in zip(chunk, embeddings, strict=True):
                    item = loaded[index]
                    outputs[index] = _Embedded(
                        embedding=embedding,
                        bucket_seconds=bucket.seconds,
                        effective_samples=item.num_samples,
                        truncated=item.metadata.truncated,
                    )
        return outputs

    @staticmethod
    def _cached_result(value: CacheValue) -> _Embedded:
        return _Embedded(
            embedding=value.embedding,
            bucket_seconds=value.bucket_seconds,
            effective_samples=value.effective_samples,
            truncated=value.truncated,
            cache_hit=True,
        )

    @staticmethod
    def _cache_value(result: _Embedded) -> CacheValue:
        if (
            result.embedding is None
            or result.bucket_seconds is None
            or result.effective_samples is None
            or result.truncated is None
        ):
            raise RuntimeError("cannot cache an incomplete embedding result")
        return CacheValue(
            embedding=result.embedding,
            bucket_seconds=result.bucket_seconds,
            effective_samples=result.effective_samples,
            truncated=result.truncated,
        )

    def _prepare_path(
        self, path: str, cache_reference: bool
    ) -> tuple[
        _Embedded | None,
        LoadedAudio | None,
        CacheKey | None,
        concurrent.futures.Future[CacheValue] | None,
    ]:
        cache_key: CacheKey | None = None
        cache_future: concurrent.futures.Future[CacheValue] | None = None
        cache_owner = False
        try:
            resolved = self.loader.resolve_path(path)
            if cache_reference:
                cache_key = self._cache_key(self.loader.file_signature(resolved))
                cached, cache_future, cache_owner = self.reference_cache.reserve(cache_key)
                if cached is not None:
                    return self._cached_result(cached), None, None, None
                if not cache_owner:
                    if cache_future is None:
                        raise RuntimeError("reference cache reservation is incomplete")
                    shared = self.reference_cache.wait(cache_future)
                    return self._cached_result(shared), None, None, None
            loaded = self.loader.load_path(resolved)
        except (AudioInputError, FileNotFoundError, PermissionError) as exc:
            if cache_owner and cache_key is not None and cache_future is not None:
                self.reference_cache.fail(cache_key, cache_future, exc)
            return _Embedded(error=_input_error(exc)), None, None, None
        return None, loaded, cache_key, cache_future

    def _embed_paths(self, paths: Sequence[str], *, cache_references: bool) -> list[_Embedded]:
        if not 1 <= len(paths) <= self.max_items_per_request:
            raise ValueError(
                f"request item count must be between 1 and {self.max_items_per_request}"
            )

        unique_paths = list(dict.fromkeys(paths))
        path_indices = {path: index for index, path in enumerate(unique_paths)}
        outputs = [_Embedded() for _ in unique_paths]
        pending_loaded: list[LoadedAudio] = []
        pending_indices: list[int] = []
        pending_keys: list[CacheKey | None] = []
        pending_futures: list[concurrent.futures.Future[CacheValue] | None] = []

        prepared = self._audio_executor.map(
            lambda path: self._prepare_path(path, cache_references), unique_paths
        )
        for index, (completed, loaded, cache_key, cache_future) in enumerate(prepared):
            if completed is not None:
                outputs[index] = completed
            elif loaded is not None:
                pending_loaded.append(loaded)
                pending_indices.append(index)
                pending_keys.append(cache_key)
                pending_futures.append(cache_future)

        if pending_loaded:
            inference_completed = False
            try:
                inferred = self._infer_loaded(pending_loaded)
                for index, cache_key, cache_future, result in zip(
                    pending_indices,
                    pending_keys,
                    pending_futures,
                    inferred,
                    strict=True,
                ):
                    outputs[index] = result
                    if cache_key is not None and cache_future is not None:
                        self.reference_cache.publish(
                            cache_key, cache_future, self._cache_value(result)
                        )
                inference_completed = True
            finally:
                if not inference_completed:
                    failure = RuntimeError("reference embedding generation failed")
                    for cache_key, cache_future in zip(pending_keys, pending_futures, strict=True):
                        if cache_key is not None and cache_future is not None:
                            self.reference_cache.fail(cache_key, cache_future, failure)

        return [outputs[path_indices[path]] for path in paths]

    def similarities(
        self,
        item_ids: Sequence[str],
        reference_paths: Sequence[str],
        candidate_paths: Sequence[str],
    ) -> list[SimilarityResult]:
        """Return per-item similarity while retaining input order and errors."""
        if not (len(item_ids) == len(reference_paths) == len(candidate_paths)):
            raise ValueError("ids and audio paths must have equal length")
        references = self._embed_paths(reference_paths, cache_references=True)
        candidates = self._embed_paths(candidate_paths, cache_references=False)

        results: list[SimilarityResult] = []
        for item_id, reference, candidate in zip(item_ids, references, candidates, strict=True):
            error = reference.error or candidate.error
            if error is not None:
                results.append(
                    SimilarityResult(
                        item_id=item_id,
                        similarity=None,
                        reward_similarity=None,
                        reference_cache_hit=reference.cache_hit,
                        reference_bucket_seconds=reference.bucket_seconds,
                        candidate_bucket_seconds=candidate.bucket_seconds,
                        error=error,
                    )
                )
                continue
            if reference.embedding is None or candidate.embedding is None:
                raise RuntimeError("similarity input is missing an embedding")
            score = float(
                F.cosine_similarity(
                    reference.embedding.reshape(1, -1),
                    candidate.embedding.reshape(1, -1),
                    dim=-1,
                ).item()
            )
            if not math.isfinite(score):
                raise RuntimeError("similarity result is not finite")
            results.append(
                SimilarityResult(
                    item_id=item_id,
                    similarity=score,
                    reward_similarity=max(0.0, min(1.0, score)),
                    reference_cache_hit=reference.cache_hit,
                    reference_bucket_seconds=reference.bucket_seconds,
                    candidate_bucket_seconds=candidate.bucket_seconds,
                    error=None,
                )
            )
        return results
