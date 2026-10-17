"""Redacted snapshots of resolved driver configuration."""

import dataclasses
import enum
import json
import os
import re
import uuid
from pathlib import Path


def redact(value):
    """Redact common credential fields, retaining environment variable names."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if re.search(
                    r"password|secret|(?:api|admin|access|wandb)[_-]?key|auth[_-]?token|authorization", str(key), re.I
                )
                and not str(key).endswith("_env")
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.dumps(redact(json.loads(value)), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return value


def _json_value(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def write_resolved_configuration(args):
    destination = getattr(args, "resolved_config_path", None)
    if destination is None:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize dataclass values before redacting their nested credential keys.
    configuration = json.loads(json.dumps(vars(args), default=_json_value))
    record = {"schema_version": 1, "options": redact(configuration)}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
