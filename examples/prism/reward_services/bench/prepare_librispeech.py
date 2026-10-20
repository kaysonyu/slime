"""Download the official LibriSpeech test sets and freeze a shared audio manifest.

Uses the upstream MD5s for transport integrity, records SHA256 for provenance,
and retains original complete utterances. No model-specific cropping is applied.
"""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

CHECKSUMS = {
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(root, subsets, mirror):
    root.mkdir(parents=True, exist_ok=True)
    provenance = []
    records = []
    for subset in subsets:
        archive = root / f"{subset}.tar.gz"
        source = f"{mirror.rstrip('/')}/{archive.name}"
        if not archive.is_file():
            partial = archive.with_suffix(".partial")
            print(f"Downloading {source}", flush=True)
            urllib.request.urlretrieve(source, partial)
            partial.replace(archive)
        if digest(archive, "md5") != CHECKSUMS[subset]:
            raise ValueError(f"Upstream MD5 differs: {archive}")
        destination = root / "LibriSpeech" / subset
        marker = root / f".{subset}.extracted"
        if not marker.exists():
            with tarfile.open(archive) as stream:
                # All members must remain under the chosen data directory.
                for member in stream.getmembers():
                    target = (root / member.name).resolve()
                    if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():
                        raise ValueError(f"Unexpected archive member: {member.name}")
                stream.extractall(root, filter="data")
            marker.touch()
        transcripts = {}
        for path in sorted(destination.rglob("*.trans.txt")):
            for line in path.read_text().splitlines():
                key, text = line.split(" ", 1)
                transcripts[key] = text
        for path in sorted(destination.rglob("*.flac")):
            info = sf.info(path)
            records.append(
                {
                    "id": path.stem,
                    "audio": str(path.resolve()),
                    "source": subset,
                    "speaker_id": path.stem.split("-")[0],
                    "duration_s": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "text": transcripts[path.stem],
                    "language": "en",
                }
            )
        provenance.append({"subset": subset, "source": source, "sha256": digest(archive), "md5": CHECKSUMS[subset]})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    summary = {
        "dataset": "LibriSpeech",
        "license": "CC BY 4.0",
        "source": "https://www.openslr.org/12/",
        "archives": provenance,
        "samples": len(records),
        "audio_hours": sum(x["duration_s"] for x in records) / 3600,
        "manifest_sha256": digest(manifest),
    }
    (root / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=CHECKSUMS, default=list(CHECKSUMS))
    parser.add_argument("--mirror", default="https://openslr.magicdatatech.com/resources/12")
    args = parser.parse_args()
    prepare(args.output, args.subsets, args.mirror)
