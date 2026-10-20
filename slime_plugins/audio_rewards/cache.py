"""Read-only, versioned embedding tables shared by rollout workers."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote


def read_embeddings(path, identity, keys):
    import numpy as np

    uri = "file:" + quote(str(Path(path).resolve())) + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        row = database.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row is None:
            raise ValueError("precomputed embedding cache lacks identity metadata")
        metadata = json.loads(row[0])
        for field in ["embedding_version", "dimension", "normalized"]:
            if metadata.get(field) != identity.get(field):
                raise ValueError(f"precomputed embedding cache has a different {field}")
        result = {}
        for key in keys:
            row = database.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
            if row is None:
                continue
            vector = np.frombuffer(row[0], dtype="<f4")
            if len(vector) != identity["dimension"] or not np.isfinite(vector).all():
                raise ValueError("invalid vector in precomputed embedding cache")
            result[key] = vector.tolist()
        return result
