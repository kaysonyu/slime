"""Speech manifest preflight and identity; rendering stays with the model processor."""

import hashlib
import json
from pathlib import Path

import soundfile as sf

from slime.rollout.rm_hub.language import resolve_language
from slime.rollout.tts_schema import parse_manifest_row_json, parse_reference_audios_metadata
from slime.utils.types import Sample


def audio_identity(path: Path) -> dict:
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Reference audio must be an existing absolute regular file")
    resolved = path.resolve(strict=True)
    info = sf.info(resolved)
    if info.format != "WAV" or info.frames <= 0 or info.channels < 1 or info.samplerate <= 0:
        raise ValueError("Reference audio must be a nonempty decodable WAV")
    # Verify the stream, not just a header claiming a longer file.
    with sf.SoundFile(resolved) as stream:
        frames = sum(len(block) for block in stream.blocks(blocksize=65536))
    if frames != info.frames:
        raise ValueError("Reference WAV is truncated")
    stat = resolved.stat()
    return dict(
        path=str(resolved),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        frames=info.frames,
        sample_rate=info.samplerate,
        channels=info.channels,
    )


def validate_reward_sample(sample, config):
    if config is None:
        return
    if config.weight("wer") > 0:
        resolve_language(sample.metadata.get("language"))
    if config.weight("judge") > 0:
        from slime.rollout.rm_hub.judge import parse_sample_rubrics

        parse_sample_rubrics(sample)
    if config.weight("reference_similarity") > 0:
        references = parse_reference_audios_metadata(sample.metadata.get("reference_audios"))
        if not references:
            raise ValueError("reference_similarity requires reference_audios")
        if any("timbre" in reference.uses for reference in references) and "timbre_sim" not in config.services:
            raise ValueError("Timbre reference requires services.timbre_sim")


def load_speech_manifest(
    path, *, language="en", reward_config=None, hf_checkpoint=None, metadata_overrides=None, teacher_domains=None
):
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw)
    digest.update(b"speech-manifest-v2")
    contract = dict(
        language=language,
        reward=reward_config.identity if reward_config else None,
        metadata_overrides=metadata_overrides or {},
        teacher_domains=sorted(teacher_domains or ()),
    )
    digest.update(json.dumps(contract, sort_keys=True).encode())
    if hf_checkpoint:
        root = Path(hf_checkpoint)
        names = {
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "preprocessor_config.json",
        }
        names.update(path.name for path in root.glob("*.py"))
        for name in sorted(names):
            artifact = root / name
            if artifact.is_file():
                digest.update(name.encode())
                digest.update(artifact.read_bytes())
    samples, seen, identities = [], set(), {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = parse_manifest_row_json(line, line_number=number)
        sample_id = row.id or row.metadata.get("id") or f"line-{number}"
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Invalid sample id at line {number}")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample id at line {number}")
        seen.add(sample_id)
        text = row.text if row.text is not None else row.script
        if isinstance(text, tuple):
            if any(item.local_instruction is not None for item in text):
                raise ValueError(
                    "Local generation does not support per-script local_instruction; provide global instructions"
                )
            text = "\n".join(item.text for item in text)
        metadata = dict(row.metadata)
        fields = {
            "id": sample_id,
            "domain": row.domain,
            "language": row.language,
            "instructions": row.instructions or row.global_instruction,
            "ref_audio": str(row.ref_audio) if row.ref_audio else None,
            "ref_text": row.ref_text,
        }
        for key, value in fields.items():
            if value is not None:
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Conflicting top-level and metadata field {key} on line {number}")
                metadata[key] = value
        if row.reference_audios:
            metadata["reference_audios"] = [reference.model_dump(mode="json") for reference in row.reference_audios]
        if row.rubric:
            metadata["rubric"] = [item.model_dump(mode="json") for item in row.rubric]
        metadata.update(metadata_overrides or {})
        metadata["language"] = resolve_language(metadata.get("language") or language).canonical
        references = parse_reference_audios_metadata(metadata.get("reference_audios", []))
        if not references and metadata.get("ref_audio"):
            metadata["reference_audios"] = [{"id": "reference", "path": metadata["ref_audio"], "uses": ["timbre"]}]
            references = parse_reference_audios_metadata(metadata["reference_audios"])
        if references and not metadata.get("ref_audio"):
            if len(references) != 1:
                raise ValueError("Multiple scoring references require explicit ref_audio for Local generation")
            metadata["ref_audio"] = str(references[0].path)
        paths = [reference.path for reference in references]
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError(f"Duplicate resolved reference audio on line {number}")
        if metadata.get("ref_audio"):
            paths.append(Path(metadata["ref_audio"]))
        for audio in paths:
            resolved = audio.resolve()
            if resolved not in identities:
                identities[resolved] = audio_identity(audio)
            digest.update(json.dumps(identities[resolved], sort_keys=True).encode())
        sample = Sample(prompt=text, label=row.target_text or text, metadata=metadata)
        validate_reward_sample(sample, reward_config)
        if teacher_domains is not None and metadata.get("domain") not in teacher_domains:
            raise ValueError(f"Sample at line {number} does not match exactly one teacher domain")
        samples.append(sample)
    if not samples:
        raise ValueError("TTS prompt dataset is empty")
    return samples, digest.hexdigest(), hashlib.sha256(raw).hexdigest(), bool(identities)
