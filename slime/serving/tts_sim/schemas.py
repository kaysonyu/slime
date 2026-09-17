"""Strict FastAPI request and response models for SIM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import MAX_REQUEST_ITEMS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class HealthResponse(StrictModel):
    status: str
    model: str | None = None


class BackendDetailsResponse(StrictModel):
    type: str
    precision: str
    tf32_matmul: bool
    compile: bool


class DynamicBatchingResponse(StrictModel):
    queue_depth: int | None
    batches: int | None
    items: int | None
    max_observed_batch: int | None
    batch_size_counts: dict[int, int] | None
    shape_counts: dict[str, int] | None


class ReferenceCacheResponse(StrictModel):
    items: int
    hits: int
    misses: int
    evictions: int
    singleflight_waits: int


class ModelResponse(StrictModel):
    name: str
    embedding_dimension: int
    sample_rate: int
    max_audio_seconds: int
    preprocess_version: str
    checkpoint_sha256: str
    expected_checkpoint_sha256: str
    execution_fingerprint: str
    allow_tf32: bool
    backend_precision: str
    backend_details: BackendDetailsResponse
    backend: str
    execution_backend: str
    devices: list[str]
    parallelism: int
    per_device_batch: int
    global_batch: int
    source_tree: str
    buckets_seconds: list[int]
    dynamic_batching: DynamicBatchingResponse
    reference_cache: ReferenceCacheResponse


class SimilarityPathItem(StrictModel):
    id: str = Field(min_length=1, max_length=512)
    reference_path: str = Field(min_length=1, max_length=16_384)
    candidate_path: str = Field(min_length=1, max_length=16_384)


class SimilarityPathRequest(StrictModel):
    """One request batch; a shared reference enables cache-friendly batching."""

    items: list[SimilarityPathItem] = Field(min_length=1, max_length=MAX_REQUEST_ITEMS)

    @model_validator(mode="after")
    def validate_batch_identity(self) -> SimilarityPathRequest:
        item_ids = [item.id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item ids must be unique within a request")
        reference_paths = {item.reference_path for item in self.items}
        if len(reference_paths) != 1:
            raise ValueError("all items in a request must use the same reference path")
        return self


class ItemErrorResponse(StrictModel):
    code: str
    message: str


class SimilarityItemResponse(StrictModel):
    id: str
    similarity: float | None
    reward_similarity: float | None
    reference_cache_hit: bool
    reference_bucket_seconds: int | None
    candidate_bucket_seconds: int | None
    error: ItemErrorResponse | None


class SimilarityResponse(StrictModel):
    model: str
    backend: str
    elapsed_ms: float
    results: list[SimilarityItemResponse]
