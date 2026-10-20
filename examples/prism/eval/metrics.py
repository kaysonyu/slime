"""Versioned text normalization and failure-aware evaluation aggregation.

Normalization ports the named moss-eval helpers recorded in manifest.json.
ASR backend/chunking and SIM preprocessing can differ: these are proxy metrics.
"""

import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from slime_plugins.audio_rewards.scoring import edit_distance

PROFILES = {"instruction-v1", "pair-v2", "dialect-t2s-v1", "taxonomy-v1"}
CHAR_LANGUAGES = {"zh", "ja", "ko"}
_PAUSE = re.compile(r"\[\s*pause\s+(?:\d+(?:\.\d+)?|\.\d+)\s*s?\s*\]", re.I)
_APOSTROPHES = {"'", "‘", "’", "ʻ", "ʼ", "＇"}


@lru_cache(maxsize=1)
def t2s():
    from opencc import OpenCC

    return OpenCC("t2s")


def units(text, language, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown normalization profile: {profile}")
    if not isinstance(text, str):
        raise TypeError("ASR/reference text must be a string")
    if profile in {"instruction-v1", "taxonomy-v1"}:
        text = unicodedata.normalize("NFKC", _PAUSE.sub(" ", text))
    if profile == "instruction-v1" and language == "zh":
        import zhconv

        text = zhconv.convert(text, "zh-cn")
    if profile == "dialect-t2s-v1" and language in CHAR_LANGUAGES:
        text = t2s().convert(text)
    chars = []
    for i, char in enumerate(text):
        if char in _APOSTROPHES:
            if 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
                chars.append("'")
            elif profile == "taxonomy-v1":
                chars.append(" ")
        elif unicodedata.category(char)[0] in {"P", "S"}:
            if profile == "taxonomy-v1":
                chars.append(" ")
        else:
            chars.append(char)
    text = "".join(chars)
    if language not in CHAR_LANGUAGES or profile in {"instruction-v1", "taxonomy-v1"}:
        text = text.lower()
    if profile == "taxonomy-v1":
        if language in CHAR_LANGUAGES:
            text = text.replace("幺", "一").replace("1", "一")
        else:
            text = re.sub(r"([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])", r" \1 ", text)
    return [c for c in text if not c.isspace()] if language in CHAR_LANGUAGES else text.split()


def wer_record(reference, hypothesis, language, profile):
    expected, actual = units(reference, language, profile), units(hypothesis, language, profile)
    if not expected:
        raise ValueError("Reference is empty after WER normalization")
    # Empty successful ASR output represents full deletions, not a service error.
    errors = edit_distance(expected, actual)
    return {
        "status": "success",
        "value": errors / len(expected),
        "errors": errors,
        "reference_units": len(expected),
        "hypothesis_units": len(actual),
        "unit": "character" if language in CHAR_LANGUAGES else "word",
        "normalization": profile,
        "reference_normalized": " ".join(expected),
        "hypothesis_normalized": " ".join(actual),
        "transcript": hypothesis,
        "empty_transcript": not actual,
        "asr_language_mode": "auto",
    }


def judge_record(queries, scores, gt_scores=None):
    import math

    if len(queries) != len(scores) or (gt_scores is not None and len(gt_scores) != len(scores)):
        raise ValueError("Judge query/score count mismatch")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*scores, *(gt_scores or [])]):
        raise ValueError("Judge probabilities must be finite within [0,1]")
    kept = [i for i in range(len(scores)) if gt_scores is None or gt_scores[i] > 0.5]
    passed = sum(scores[i] > 0.5 for i in kept)
    return {
        "status": "success" if kept else "not_applicable",
        "value": passed / len(kept) if kept else None,
        "passed": passed,
        "total": len(kept),
        "queries": queries,
        "scores": scores,
        "gt_scores": gt_scores,
        "kept_indices": kept,
        "threshold": 0.5,
        "threshold_operator": ">",
        "gate_excluded": len(scores) - len(kept),
    }


def summarize(records):
    """Keep WER profiles/units and tasks separate; no seven-domain total score."""
    groups = defaultdict(list)
    for record in records:
        recipe = record["recipe"]
        prefix = recipe["domain"]
        for key in (
            prefix,
            f"{prefix}/benchmark/{recipe['benchmark']}",
            f"{prefix}/language/{recipe['language']}",
            f"{prefix}/arm/{recipe['arm']}",
            f"{prefix}/task/{recipe['task']}",
        ):
            groups[key].append(record)
        for field in ("class", "category"):
            value = recipe.get("source_meta", {}).get(field)
            if value:
                groups[f"{prefix}/{field}/{value}"].append(record)
    output = {}
    for key, rows in sorted(groups.items()):
        values = {
            "samples": len(rows),
            "generation_returned": sum(r.get("generation_status") == "success" for r in rows),
            "generation_success": sum(
                r.get("generation_status") == "success" and bool(r.get("audio_path")) and (r.get("frames") or 0) > 0
                for r in rows
            ),
            "zero_frames": sum(r.get("frames") == 0 for r in rows),
            "truncated": sum(r.get("truncated", False) for r in rows),
        }
        for metric in ("wer", "sim", "judge_clap"):
            expected = [r for r in rows if metric in r["recipe"]["metrics"]]
            outcomes = [r.get("metrics", {}).get(metric, {"status": "missing"}) for r in expected]
            valid = [v for v in outcomes if v["status"] == "success"]
            values[f"{metric}/expected"] = len(expected)
            values[f"{metric}/valid"] = len(valid)
            values[f"{metric}/not_applicable"] = sum(v["status"] == "not_applicable" for v in outcomes)
            values[f"{metric}/failed"] = sum(v["status"] not in {"success", "not_applicable"} for v in outcomes)
            if expected:
                values[f"{metric}/coverage"] = len(valid) / len(expected)
            if not valid:
                continue
            if metric == "wer":
                buckets = defaultdict(list)
                for v in valid:
                    buckets[(v["normalization"], v["unit"])].append(v)
                for (profile, unit), items in buckets.items():
                    name = "cer" if unit == "character" else "wer"
                    values[f"{name}/{profile}/micro"] = sum(v["errors"] for v in items) / sum(
                        v["reference_units"] for v in items
                    )
                    values[f"{name}/{profile}/macro"] = sum(v["value"] for v in items) / len(items)
                    values[f"{name}/{profile}/empty_asr"] = sum(v["empty_transcript"] for v in items)
            elif metric == "judge_clap":
                values["judge_clap/pass_micro"] = sum(v["passed"] for v in valid) / sum(v["total"] for v in valid)
                values["judge_clap/pass_macro"] = sum(v["value"] for v in valid) / len(valid)
                values["judge_clap/attributes"] = sum(v["total"] for v in valid)
                values["judge_clap/gt_excluded"] = sum(v.get("gate_excluded", 0) for v in outcomes)
                categories = defaultdict(lambda: [0, 0])
                for value in valid:
                    for index in value["kept_indices"]:
                        category = value["queries"][index].partition(":")[0].strip().lower().replace(" ", "_")
                        categories[category][0] += value["scores"][index] > 0.5
                        categories[category][1] += 1
                for category, (passed, total) in categories.items():
                    values[f"judge_clap/category/{category}/pass"] = passed / total
                    values[f"judge_clap/category/{category}/attributes"] = total
            else:
                values["sim/mean"] = sum(v["value"] for v in valid) / len(valid)
        durations = [r["audio_seconds"] for r in rows if r.get("audio_seconds") is not None]
        if durations:
            values["audio_seconds/mean"] = sum(durations) / len(durations)
        output.update((f"{key}/{name}", value) for name, value in values.items())
    return output
