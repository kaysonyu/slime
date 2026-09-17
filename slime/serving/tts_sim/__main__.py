"""Standalone uvicorn entry point for the Torch SIM service."""

from __future__ import annotations

import json

import uvicorn

from .api import create_app
from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    app = create_app(settings=settings)
    backend = app.state.embedding_service.backend
    print(
        json.dumps(
            {
                "allow_tf32": bool(getattr(backend, "allow_tf32", False)),
                "backend_precision": getattr(backend, "precision", "unknown"),
                "checkpoint_sha256": getattr(backend, "checkpoint_sha256", backend.fingerprint),
                "devices": settings.devices,
                "event": "wavlm_sim_service_configured",
                "execution_backend": backend.name,
                "execution_fingerprint": backend.fingerprint,
                "global_batch": settings.global_batch,
                "parallelism": len(settings.devices),
                "per_device_batch": settings.per_device_batch,
                "source_tree": settings.source_tree,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
        proxy_headers=True,
        access_log=settings.access_log,
    )


if __name__ == "__main__":
    main()
