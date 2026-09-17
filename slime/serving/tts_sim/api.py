"""FastAPI lifecycle, admission control, health, similarity, and metrics routes."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .assets import (
    CANONICAL_CHECKPOINT_SHA256,
    EMBEDDING_DIMENSION,
    MAX_AUDIO_SECONDS,
    MODEL_NAME,
    TARGET_SAMPLE_RATE,
)
from .audio import PREPROCESS_VERSION, AudioLoader
from .backend import BackendPool, PyTorchEmbeddingBackend
from .buckets import BucketPolicy, BucketSpec
from .cache import ReferenceEmbeddingCache
from .config import Settings
from .scheduler import DynamicBatchingBackend, QueueFullError
from .schemas import (
    BackendDetailsResponse,
    DynamicBatchingResponse,
    HealthResponse,
    ItemErrorResponse,
    ModelResponse,
    ReferenceCacheResponse,
    SimilarityItemResponse,
    SimilarityPathRequest,
    SimilarityResponse,
)
from .service import EmbeddingService, SimilarityResult


class AdmissionGate:
    """Bound admitted HTTP work independently from the embedding queue."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0
        self._lock = threading.Lock()

    @asynccontextmanager
    async def enter(self) -> AsyncIterator[None]:
        with self._lock:
            if self.active >= self.limit:
                raise HTTPException(status_code=429, detail="too many in-flight requests")
            self.active += 1
        try:
            yield
        finally:
            with self._lock:
                self.active -= 1


def _similarity_item(item: SimilarityResult) -> SimilarityItemResponse:
    error = (
        ItemErrorResponse(code=item.error.code, message=item.error.message)
        if item.error is not None
        else None
    )
    return SimilarityItemResponse(
        id=item.item_id,
        similarity=item.similarity,
        reward_similarity=item.reward_similarity,
        reference_cache_hit=item.reference_cache_hit,
        reference_bucket_seconds=item.reference_bucket_seconds,
        candidate_bucket_seconds=item.candidate_bucket_seconds,
        error=error,
    )


def build_service(settings: Settings) -> EmbeddingService:
    """Assemble the strict backend, dynamic batcher, buckets, and cache."""
    backends = tuple(
        PyTorchEmbeddingBackend(
            checkpoint=settings.checkpoint,
            device=device,
            verify_hash=index == 0,
            allow_tf32=settings.allow_tf32,
        )
        for index, device in enumerate(settings.devices)
    )
    backend = BackendPool(backends)
    dynamic = DynamicBatchingBackend(
        backend,
        max_batch=settings.global_batch,
        max_delay_ms=settings.dynamic_delay_ms,
        max_queue_items=settings.max_queue_items,
    )
    buckets = BucketPolicy(
        tuple(
            BucketSpec(seconds, max_batch=settings.global_batch)
            for seconds in (4, 8, 12, 16, 20, 24, 30)
        )
    )
    return EmbeddingService(
        AudioLoader(settings.allowed_roots),
        dynamic,
        buckets,
        ReferenceEmbeddingCache(settings.reference_cache_items),
        max_items_per_request=settings.max_items_per_request,
        audio_workers=settings.audio_workers,
    )


def create_app(
    service: EmbeddingService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the FastAPI surface and own service shutdown through lifespan."""
    effective_settings = settings or Settings.from_env()
    effective_service = service or build_service(effective_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        effective_service.close()

    app = FastAPI(
        title="MOSS-TTS Torch SIM service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.embedding_service = effective_service
    app.state.settings = effective_settings
    gate = AdmissionGate(effective_settings.max_inflight_requests)

    registry = CollectorRegistry()
    request_count = Counter(
        "wavlm_sim_http_requests_total",
        "HTTP requests",
        ("route", "method", "status"),
        registry=registry,
    )
    request_latency = Histogram(
        "wavlm_sim_http_request_seconds",
        "HTTP request duration",
        ("route", "method"),
        registry=registry,
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    )
    cache_size = Gauge("wavlm_sim_reference_cache_items", "Reference cache size", registry=registry)
    cache_hits = Gauge(
        "wavlm_sim_reference_cache_hits_total",
        "Reference embedding cache hits",
        registry=registry,
    )
    cache_misses = Gauge(
        "wavlm_sim_reference_cache_misses_total",
        "Reference embedding cache owner misses",
        registry=registry,
    )
    cache_evictions = Gauge(
        "wavlm_sim_reference_cache_evictions_total",
        "Reference embedding cache evictions",
        registry=registry,
    )
    cache_singleflight_waits = Gauge(
        "wavlm_sim_reference_cache_singleflight_waits_total",
        "Requests that reused an in-flight reference embedding fill",
        registry=registry,
    )
    inflight = Gauge(
        "wavlm_sim_inflight_requests", "Admitted in-flight requests", registry=registry
    )

    health_routes = {"/health/live", "/health/ready"}

    @app.middleware("http")
    async def controls_and_metrics(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            # Health probes stay public; every other route uses constant-time
            # API-key comparison when authentication is configured.
            if effective_settings.api_key and request.url.path not in health_routes:
                authorization = request.headers.get("authorization", "")
                bearer = authorization[7:] if authorization.lower().startswith("bearer ") else ""
                supplied = bearer or request.headers.get("x-api-key", "")
                if not supplied or not secrets.compare_digest(supplied, effective_settings.api_key):
                    status = 401
                    return JSONResponse({"detail": "invalid API key"}, status_code=status)

            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", request.url.path)
            request_count.labels(route, request.method, str(status)).inc()
            request_latency.labels(route, request.method).observe(time.perf_counter() - started)

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(_request: Request, exc: QueueFullError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.get("/health/live")
    async def live() -> HealthResponse:
        return HealthResponse(status="live")

    @app.get("/health/ready")
    async def ready() -> HealthResponse:
        if not effective_service.is_ready:
            raise HTTPException(status_code=503, detail="SIM model is not ready")
        return HealthResponse(status="ready", model=MODEL_NAME)

    @app.get("/v1/model")
    async def model_info() -> ModelResponse:
        backend = effective_service.backend
        details = BackendDetailsResponse(
            type=backend.details["type"],
            precision=backend.details["precision"],
            tf32_matmul=backend.details["tf32_matmul"],
            compile=backend.details["compile"],
        )
        return ModelResponse(
            name=MODEL_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
            sample_rate=TARGET_SAMPLE_RATE,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            preprocess_version=PREPROCESS_VERSION,
            checkpoint_sha256=backend.checkpoint_sha256,
            expected_checkpoint_sha256=CANONICAL_CHECKPOINT_SHA256,
            execution_fingerprint=backend.fingerprint,
            allow_tf32=backend.allow_tf32,
            backend_precision=backend.precision,
            backend_details=details,
            backend="pytorch",
            execution_backend=backend.name,
            devices=list(effective_settings.devices),
            parallelism=len(effective_settings.devices),
            per_device_batch=effective_settings.per_device_batch,
            global_batch=effective_settings.global_batch,
            source_tree=effective_settings.source_tree,
            buckets_seconds=[bucket.seconds for bucket in effective_service.bucket_policy.buckets],
            dynamic_batching=DynamicBatchingResponse(
                queue_depth=getattr(backend, "queue_depth", None),
                batches=getattr(backend, "batches", None),
                items=getattr(backend, "items", None),
                max_observed_batch=getattr(backend, "max_observed_batch", None),
                batch_size_counts=getattr(backend, "batch_size_counts", None),
                shape_counts=getattr(backend, "shape_counts", None),
            ),
            reference_cache=ReferenceCacheResponse(
                items=len(effective_service.reference_cache),
                hits=effective_service.reference_cache.hits,
                misses=effective_service.reference_cache.misses,
                evictions=effective_service.reference_cache.evictions,
                singleflight_waits=effective_service.reference_cache.singleflight_waits,
            ),
        )

    @app.post("/v1/similarities", response_model=SimilarityResponse)
    async def similarities(payload: SimilarityPathRequest) -> SimilarityResponse:
        started = time.perf_counter()
        try:
            async with gate.enter():
                inflight.set(gate.active)
                results = await anyio.to_thread.run_sync(
                    effective_service.similarities,
                    [item.id for item in payload.items],
                    [item.reference_path for item in payload.items],
                    [item.candidate_path for item in payload.items],
                )
        finally:
            inflight.set(gate.active)
        return SimilarityResponse(
            model=MODEL_NAME,
            backend="pytorch",
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            results=[_similarity_item(item) for item in results],
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        cache_size.set(len(effective_service.reference_cache))
        cache_hits.set(effective_service.reference_cache.hits)
        cache_misses.set(effective_service.reference_cache.misses)
        cache_evictions.set(effective_service.reference_cache.evictions)
        cache_singleflight_waits.set(effective_service.reference_cache.singleflight_waits)
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    return app
