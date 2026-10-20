"""Offline parity audit against a local moss-eval tree and Prism prompt protocol.

No checkpoint weights or GPU are loaded. This validates rendered prompt text and
normalization only, not model inference or service audio preprocessing parity.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from examples.prism.eval.build import digest, read_rows
from examples.prism.eval.metrics import units


def audit(root, moss_eval, protocol_path):
    sys.path.insert(0, str(moss_eval / "src"))
    from adapters.models.moss_tts import runtime as rt
    from runtime.metrics.asr_backends import strip_wer_punctuation
    from runtime.metrics.dialect_accent.wer_hooks import normalize_pair as dialect_pair
    from runtime.metrics.instruction_tts_eval.qwen_wer import normalize_wer_units
    from runtime.metrics.voice_taxonomy.wer_hooks import normalize_pair as taxonomy_pair

    spec = importlib.util.spec_from_file_location("prism_eval_audit_protocol", protocol_path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    rows = read_rows(root / "all.jsonl")
    normalizations, prompts = 0, 0
    for row in rows:
        metadata, recipe = row["metadata"], row["metadata"]["prism_eval"]
        prompt = metadata["omni_prompt"]
        language = metadata["language"] if metadata["show_language"] else None
        request = {
            **prompt,
            "global_instruction": {**({"Language": language} if language else {}), **prompt["global_instruction"]},
        }
        actual = protocol.render_prompt_request(request)
        script, instruction = prompt["script"], prompt["global_instruction"]
        if isinstance(script, list):
            instruction = {
                "global instruction": instruction,
                "local instruction": [
                    {"text": s["text"], "local instruction": s["local_instruction"]}
                    for s in script
                    if s.get("local_instruction")
                ],
            }
            script = "".join(s["text"] for s in script)
        expected = rt.render_moss_tts_v2_unified_prompt(
            script=script,
            reference_indices=(1,) if prompt["references"] else (),
            global_instruction=json.dumps(instruction, ensure_ascii=False) if instruction else None,
            task_type=metadata["task_type"],
            language=language,
            sample_id=recipe["id"],
            prompt_revision=6,
            modality_override="sing" if recipe["domain"] == "sing" and not prompt["references"] else None,
        )
        if actual != expected:
            raise ValueError(f"Prompt mismatch: {recipe['id']}\nACTUAL:\n{actual}\nEXPECTED:\n{expected}")
        prompts += 1
        # Exercise both real references and a synthetic edge-case transcription.
        for text in (metadata["transcript_target"], "Don't ‘STOP’ 審訊係幺1Ａ [pause 12]。 hello,world"):
            profile, lang = recipe["normalization"], recipe["language"]
            if profile == "instruction-v1":
                _, expected_units = normalize_wer_units(text, lang)
            elif profile in {"dialect-t2s-v1", "taxonomy-v1"}:
                normalized, _ = (dialect_pair if profile == "dialect-t2s-v1" else taxonomy_pair)(lang, text, text)
                expected_units = normalized.split()
            else:
                normalized = " ".join(strip_wer_punctuation(text).split())
                expected_units = (
                    list("".join(normalized.split())) if lang in {"zh", "ja", "ko"} else normalized.lower().split()
                )
            if list(expected_units) != units(text, lang, profile):
                raise ValueError(f"Normalization mismatch: {recipe['id']} {text!r}")
            normalizations += 1
    return {
        "status": "passed",
        "rendered_prompts": prompts,
        "normalization_pairs": normalizations,
        "protocol_path": str(protocol_path),
        "protocol_sha256": digest(protocol_path),
        "manifest_sha256": digest(root / "manifest.json"),
        "scope": "CPU prompt rendering and text normalization; no GPU/audio-service parity claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--moss-eval", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.moss_eval, args.protocol)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
