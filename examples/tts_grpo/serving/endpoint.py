from __future__ import annotations

import argparse
import fcntl
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ENDPOINT_KEYS = (
    "TTS_ASR_URL",
    "TTS_SIM_URL",
    "TTS_JUDGE_URL",
)
_ASSIGNMENT = re.compile(r"^([A-Z0-9_]+)=(\S+)$")


# Result details let delete report ownership mismatches without exposing URLs.
@dataclass(frozen=True)
class RemoveResult:
    state_removed: bool
    bundle_removed: bool
    warning: str | None = None


def _url(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = value.strip()
    if any(char.isspace() for char in value):
        raise ValueError(f"{label} must not contain whitespace")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must use HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain credentials")
    return value


def _endpoint_url(detail: Mapping[str, object], api_path: str) -> str:
    status = detail.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("serving.status must be a non-empty string")
    if status.strip() != "RUNNING":
        raise ValueError(f"serving is {status.strip()}, not RUNNING")
    extra = detail.get("extra_info")
    if not isinstance(extra, Mapping):
        raise ValueError("serving.extra_info must be an object")
    base_url = _url(extra.get("service"), "serving.extra_info.service").rstrip("/")
    if not api_path.startswith("/"):
        raise ValueError("API path must start with '/'")
    return _url(f"{base_url}{api_path}", "serving endpoint")


def _detail(workspace: str, name: str) -> Mapping[str, object]:
    # Inspire intentionally hides service URLs from public CLI JSON. These
    # optional imports must therefore run in the installed Inspire runtime.
    from inspire.config.workspaces import select_workspace_id
    from inspire.platform.web.browser_api import servings
    from inspire.platform.web.session import get_web_session

    session = get_web_session()
    workspace_id = select_workspace_id(
        explicit_workspace_name=workspace,
        session=session,
    )
    items, _ = servings.list_servings(
        workspace_id=workspace_id,
        keyword=name,
        page_size=100,
        session=session,
    )
    matches = [item for item in items if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one serving named {name!r}, found {len(matches)}")
    detail = servings.get_serving_detail(
        matches[0].inference_serving_id,
        session=session,
    )
    if not isinstance(detail, Mapping):
        raise ValueError("serving detail must be an object")
    return detail


def get_url(workspace: str, name: str, api_path: str) -> str:
    return _endpoint_url(_detail(workspace, name), api_path)


# Endpoint files use a strict, non-executable KEY=URL format.
def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    if not path.is_file():
        raise ValueError(f"endpoint file is not a regular file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            raise ValueError("endpoint file contains a malformed assignment")
        key, value = match.groups()
        if key not in ENDPOINT_KEYS:
            raise ValueError(f"endpoint file contains unsupported key: {key}")
        if key in values:
            raise ValueError(f"endpoint file contains duplicate key: {key}")
        values[key] = _url(value, key)
    return values


def _write_env(path: Path, values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            for key in ENDPOINT_KEYS:
                if key in values:
                    stream.write(f"{key}={values[key]}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(bundle: Path) -> Iterator[None]:
    lock_path = bundle.with_name(f"{bundle.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def save_endpoint(key: str, url: str, state_file: Path, bundle: Path | None) -> None:
    if key not in ENDPOINT_KEYS:
        raise ValueError(f"unsupported endpoint key: {key}")
    url = _url(url, key)
    if bundle is None:
        _write_env(state_file, {key: url})
        return
    with _lock(bundle):
        values = _read_env(bundle)
        values[key] = url
        _write_env(state_file, {key: url})
        _write_env(bundle, values)


def remove_endpoint(key: str, state_file: Path, bundle: Path | None) -> RemoveResult:
    if key not in ENDPOINT_KEYS:
        raise ValueError(f"unsupported endpoint key: {key}")
    if bundle is None:
        existed = state_file.exists()
        state_file.unlink(missing_ok=True)
        return RemoveResult(state_removed=existed, bundle_removed=False)

    with _lock(bundle):
        state_values = _read_env(state_file)
        if state_values and set(state_values) != {key}:
            raise ValueError(f"service state file must contain only {key}")
        expected_url = state_values.get(key)
        bundle_values = _read_env(bundle)
        bundle_url = bundle_values.get(key)
        removed = expected_url is not None and bundle_url == expected_url
        warning = None
        if removed:
            del bundle_values[key]
            _write_env(bundle, bundle_values)
        elif bundle_url is not None and expected_url is None:
            warning = f"kept {key} in the endpoint bundle because the service state file is missing"
        elif bundle_url is not None:
            warning = f"kept {key} in the endpoint bundle because the service state file has a different URL"
        existed = state_file.exists()
        state_file.unlink(missing_ok=True)
        return RemoveResult(
            state_removed=existed,
            bundle_removed=removed,
            warning=warning,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Read an authenticated Inspire Serving URL.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--api-path", required=True)
    args = parser.parse_args()
    print(get_url(args.workspace, args.name, args.api_path))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
