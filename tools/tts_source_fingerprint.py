"""Content identity for shared source assets, including uncommitted development files."""

import hashlib
from pathlib import Path
import sys


def source_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
    if not files:
        raise ValueError("Source fingerprint requires existing source files")
    for path in files:
        digest.update((str(path.relative_to(root)) if root.is_dir() else path.name).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    print(source_fingerprint(sys.argv[1]))
