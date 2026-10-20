"""Select and freeze seven-domain small eval from full canonical moss-eval exports.

Only this build tool imports moss-eval. Runtime consumes the frozen package.
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from examples.prism.eval.materialize import SOURCES

SEED = 20261020
LANGUAGES = dict(
    zh="Chinese",
    en="English",
    ja="Japanese",
    ko="Korean",
    de="German",
    es="Spanish (Spain)",
    fr="French",
    it="Italian",
    ru="Russian",
)
DOMAINS = {
    "base": 128,
    "instruction": 160,
    "clone_w_instruction": 96,
    "dialect": 200,
    "asmr": 48,
    "sing": 48,
    "event": 160,
}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def rank(row):
    return hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()


def duration(row):
    estimate = row.get("meta", {}).get("est_secs")
    if estimate is not None:
        return float(estimate)
    text = row["text"]
    return len(text) / 4 if row["language"] in {"zh", "ja", "ko"} else len(text.split()) / 2.5


def choose(rows, n, *, feature=None):
    """Deterministic round-robin coverage of category and short/medium/long bins.

    Exclude estimated >60s texts; allocation fails loudly instead of shrinking.
    Maximum one 30–60s sentinel per stratum. Estimates are not audio durations.
    """
    groups = defaultdict(list)
    for row in rows:
        seconds = duration(row)
        if seconds <= 60:
            bucket = "short" if seconds <= 10 else "medium" if seconds <= 30 else "long"
            groups[(str(feature(row)) if feature else "all", bucket)].append(row)
    for values in groups.values():
        values.sort(key=rank)
    chosen, bucket_counts, feature_counts = [], Counter(), Counter()
    available = {
        bucket: sum(len(v) for (_, b), v in groups.items() if b == bucket) for bucket in ("short", "medium", "long")
    }
    long_target = int(n >= 4 and available["long"] > 0)
    targets = {"long": long_target, "short": round((n - long_target) * 0.6)}
    targets["medium"] = n - targets["long"] - targets["short"]
    while len(chosen) < n:
        candidates = [
            key
            for key, values in groups.items()
            if values and (key[1] != "long" or bucket_counts["long"] < long_target)
        ]
        if not candidates:
            raise ValueError(f"Insufficient eligible rows: requested {n}, got {len(chosen)}")
        # Meet duration quotas first, diversify features within each bucket.
        key = min(
            candidates,
            key=lambda key: (
                bucket_counts[key[1]] / max(targets[key[1]], 0.01),
                feature_counts[key[0]],
                rank(groups[key][0]),
            ),
        )
        chosen.append(groups[key].pop(0))
        bucket_counts[key[1]] += 1
        feature_counts[key[0]] += 1
    return chosen


def select(sources):
    selected = []

    def add(domain, bench, rows):
        selected.extend((domain, bench, row) for row in rows)

    def bytask(bench):
        result = defaultdict(list)
        for row in sources[bench]:
            result[row["task"]].append(row)
        return result

    for lang in ("zh", "en"):
        add("base", "seed-tts", choose(bytask("seed-tts")[f"seed-tts-zeroshot-{lang}"], 32))
    for lang in ("ja", "ko", "de", "es", "fr", "it", "ru"):
        add("base", "cv3", choose(bytask("cv3")[f"cv3-zeroshot-{lang}"], 4))
    taxonomy = defaultdict(list)
    for row in sources["moss-tts-voice-taxonomy"]:
        taxonomy[(row["language"], row["meta"]["source_info"]["category"])].append(row)
    if len(taxonomy) != 12:
        raise ValueError("Expected six voice taxonomy categories in each language")
    for rows in taxonomy.values():
        add("base", "moss-tts-voice-taxonomy", choose(rows, 3))
    for rows in bytask("instruction-tts").values():
        add("instruction", "instruction-tts", choose(rows, 8))
    tasks = bytask("instruction-tts-v2")
    for lang in ("zh", "en"):
        for category, n in (("global", 8), ("special", 6), ("combined", 4), ("boundary", 4)):
            add("instruction", "instruction-tts-v2", choose(tasks[f"instruction-tts-v2-{lang}-{category}"], n))
        task = f"instruction-tts-v2-{lang}-local"
        paired = {r["sample_name"]: r for r in tasks[task + "-loc"]}
        originals = choose([r for r in tasks[task] if r["sample_name"] in paired], 4)
        add("instruction", "instruction-tts-v2", originals + [paired[r["sample_name"]] for r in originals])
    for rows in bytask("mint-bench").values():
        add(
            "instruction",
            "mint-bench",
            choose(rows, 12, feature=lambda r: "/".join(r["meta"]["taxonomy_node"].split("/")[:2])),
        )
    for rows in bytask("instruction-tts-mossv2").values():
        add("instruction", "instruction-tts-mossv2", choose(rows, 5))
    add("instruction", "breezeblue", choose(sources["breezeblue"], 8))
    for task, rows in bytask("instruction-voiceclone").items():
        n = 16 if task in {"ivc-mossv1-zh", "ivc-breezeblue-en"} else 8
        add("clone_w_instruction", "instruction-voiceclone", choose(rows, n))
    for domain, prefix in (("asmr", "whisper"), ("sing", "sing")):
        tasks = bytask("whisper-sing")
        for lang in ("zh", "en"):
            task = f"{prefix}-{lang}"
            rows = choose(tasks[task], 12, feature=lambda r: r["meta"].get("scene", r["meta"].get("style")))
            clones = {r["meta"]["source_id"]: r for r in tasks[task + "-clone"]}
            add(domain, "whisper-sing", rows + [clones[r["meta"]["source_id"]] for r in rows])
    tasks = bytask("sound-event")
    for lang in ("zh", "en"):
        single = tasks[f"event-{lang}"]
        events = defaultdict(list)
        for row in single:
            events[row["meta"]["events"][0]].append(row)
        if len(events) != 34:
            raise ValueError("Expected 34 single event tags")
        rows = [choose(v, 1)[0] for v in events.values()]
        multi_task = next(t for t in tasks if "multi" in t and t.endswith(lang))
        multi = choose(tasks[multi_task], 6, feature=lambda r: len(r["meta"]["events"]))
        for task, plain in ((f"event-{lang}", rows), (multi_task, multi)):
            clones = {r["meta"]["source_id"]: r for r in tasks[task + "-clone"]}
            add("event", "sound-event", plain + [clones[r["meta"]["source_id"]] for r in plain])
    classes = defaultdict(list)
    for row in sources["dialect-accent"]:
        classes[row["meta"]["class"]].append(row)
    if len(classes) != 50:
        raise ValueError("Expected 50 dialect/accent classes")
    for rows in classes.values():
        plain = {r["meta"]["source_id"]: r for r in rows if not r.get("prompt_audio")}
        clones = [r for r in rows if r.get("prompt_audio")]
        same = choose([r for r in clones if r["meta"]["case"].startswith("bench_")], 1)[0]
        # Plain names differ from clone names only in the task portion of the ID.
        add("dialect", "dialect-accent", [plain[same["meta"]["source_id"]], same])
        for prefix in ("zhref_", "xref_far_"):
            add("dialect", "dialect-accent", choose([r for r in clones if r["meta"]["case"].startswith(prefix)], 1))
    counts = Counter(d for d, _, _ in selected)
    if dict(counts) != DOMAINS:
        raise ValueError(f"Quota mismatch: {counts}")
    if len({(b, r["id"]) for _, b, r in selected}) != sum(DOMAINS.values()):
        raise ValueError("Duplicate selected IDs")
    return sorted(selected, key=lambda item: (item[0], item[1], item[2]["id"]))


class Freezer:
    def __init__(self, args):
        self.args = args
        self.assets = {}
        self.source_files = {}
        sys.path.insert(0, str(args.moss_eval / "src"))
        from adapters.models.moss_tts import runtime
        from runtime.metrics.instruction_tts_eval import _rev4_scope

        self.runtime, self.scope = runtime, _rev4_scope
        self.queries = {}
        for bench, folder in SOURCES.items():
            path = args.datasets / folder / "judge_values_all.jsonl"
            if path.is_file():
                self.source_files[str(path)] = digest(path)
                self.queries[bench] = {r["id"]: r["queries"] for r in read_rows(path)}

    def asset(self, source):
        source = Path(source).resolve(strict=True)
        sha = digest(source)
        relative = f"assets/{sha}{source.suffix.lower()}"
        destination = self.args.output / relative
        if not destination.exists():
            shutil.copyfile(source, destination)
        self.assets[relative] = {"sha256": sha, "bytes": destination.stat().st_size, "source": str(source)}
        return str(destination)

    @staticmethod
    def attributes(lines):
        return {key: value.strip() for line in lines for key, _, value in [line.partition(":")]}

    def convert(self, domain, bench, row):
        from adapters.models.moss_tts.text_tn import apply_text_tn_to_script, text_tn_language_ok
        from benchmarks.definitions.sound_event.data import strip_event_tags

        rt = self.runtime
        sid, lang = row["id"], row["language"]
        script = row.get("script", row["text"])
        instruction = row.get("global_instruction")
        reference = row.get("prompt_audio")
        task = row.get("moss_task_type") or rt._rev4_task_type(int(bool(reference)), bool(instruction), sid)
        if task == "Local Instruction":
            nested = rt._rev4_decode_local_instruction_payload(instruction, sid)
            script = [
                {"text": text, "local_instruction": self.attributes(lines)}
                for text, lines in rt._rev4_local_script(script, nested["local instruction"], sid)
            ]
            global_attributes = self.attributes(rt._rev4_instruction_lines_from_obj(nested["global instruction"], sid))
        else:
            global_attributes = self.attributes(rt._rev4_instruction_lines(instruction, sid))
        original_script = script
        if text_tn_language_ok(lang):
            script, _ = apply_text_tn_to_script(script)
        tn_changed = script != original_script
        # The Omni composable prompt API accepts literal text. Freeze the platform
        # robustness/control-tag cleanup too, so no live moss-eval import is needed.
        if isinstance(script, str):
            script = rt.normalize_tts_text(script)
        else:
            script = [
                {**segment, "text": rt.normalize_tts_text(segment["text"]) or segment["text"]} for segment in script
            ]
        prompt = {
            "modality": "sing" if domain == "sing" else "speech",
            "script": script,
            "global_instruction": global_attributes,
            "references": {},
            "reference_attributes": {},
        }
        if reference:
            prompt["references"] = {"audio1": self.asset(reference)}
            prompt["reference_attributes"] = rt._REV4_REFERENCE_RULES[rt._rev4_task_key(task)]
        queries, excluded = [], []
        official = self.queries.get(bench, {}).get(sid, [])
        if domain in {"instruction", "clone_w_instruction"} and not official:
            raise ValueError(f"Missing official queries for {sid}")
        # Deployed JudgeCLAP has 27 categories. Keep original atomic query strings;
        # category mapping is for eligibility only, never translate query values.
        supported = set(rt.REV4_INSTRUCTION_FIELDS) - {"Language", "Appearance", "Listener context", "Physical scene"}
        keep, _ = self.scope.partition_queries(official, allow_local=bench == "instruction-tts-v2")
        for i, query in enumerate(official):
            category = self.scope.canonical_rev4_field(self.scope.query_category(query), allow_local=True)
            if i in keep and category in supported:
                queries.append(query)
            else:
                excluded.append(
                    {
                        "index": i,
                        "query": query,
                        "reason": "renderer_scope" if i not in keep else "judge_category_unsupported",
                    }
                )
        gt = None
        if bench == "instruction-tts-mossv2":
            subset = row["task"].removeprefix(bench + "-")
            root = self.args.datasets / SOURCES[bench] / subset
            mapping = {r["id"]: r["file"] for r in read_rows(root / "metadata.jsonl")}
            gt = self.asset(root / mapping[row["sample_name"]])
        meta = row.get("meta", {})
        arm = "clone" if reference else "plain"
        if domain == "dialect" and reference:
            arm = (
                "same"
                if meta["case"].startswith("bench_")
                else "mandarin" if meta["case"].startswith("zhref_") else "far"
            )
        metric_names = ["wer"] + (["sim"] if reference else []) + (["judge_clap"] if queries else [])
        profile = (
            "dialect-t2s-v1"
            if domain == "dialect"
            else (
                "taxonomy-v1"
                if bench == "moss-tts-voice-taxonomy"
                else ("instruction-v1" if domain in {"instruction", "clone_w_instruction"} else "pair-v2")
            )
        )
        stable_id = f"{bench}/{sid}"
        recipe = {
            "id": stable_id,
            "domain": domain,
            "benchmark": bench,
            "source_id": sid,
            "task": row["task"],
            "arm": arm,
            "language": lang,
            "metrics": metric_names,
            "normalization": profile,
            "queries": queries,
            "official_queries": official,
            "excluded_queries": excluded,
            "gt_audio": gt,
            "source_meta": meta,
            "estimated_seconds": duration(row),
            "prompt_text": row.get("prompt_text"),
            "generation_seed": int(rank({"id": stable_id})[:8], 16) & 0x7FFFFFFF,
            "tn": {
                "version": "v1.16",
                "policy": "segment-before-tn",
                "changed": tn_changed,
                "cleanup": "platform-normalize_tts_text",
            },
            "original_script": original_script,
        }
        target = strip_event_tags(row["text"]) if domain == "event" else row["text"]
        return {
            "text": row["text"],
            "metadata": {
                "omni_prompt": prompt,
                "task_type": task,
                "language": LANGUAGES[lang],
                "show_language": row.get("moss_prompt_language") != "off",
                "transcript_target": target,
                "prism_eval": recipe,
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("canonical", "moss-eval", "datasets", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if (args.output / "manifest.json").exists():
        raise FileExistsError(f"Use a new version directory: {args.output}")
    sources = {f.stem: read_rows(f) for f in sorted(args.canonical.glob("*.jsonl"))}
    selected = select(sources)
    (args.output / "assets").mkdir(parents=True, exist_ok=True)
    freezer = Freezer(args)
    rows = [freezer.convert(*item) for item in selected]
    package_fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for row in rows:
        row["metadata"]["prism_eval"]["package_fingerprint"] = package_fingerprint
    write_rows(args.output / "all.jsonl", rows)
    for domain in DOMAINS:
        write_rows(
            args.output / f"{domain}.jsonl", [r for r in rows if r["metadata"]["prism_eval"]["domain"] == domain]
        )
    # Smoke covers every domain and the structurally distinct local/GT paths.
    smoke = []
    for domain in DOMAINS:
        pool = [r for r in rows if r["metadata"]["prism_eval"]["domain"] == domain]
        if domain == "instruction":
            chosen = [
                next(r for r in pool if r["metadata"]["task_type"] == "Local Instruction"),
                next(r for r in pool if r["metadata"]["prism_eval"]["gt_audio"]),
            ]
        else:
            chosen = []
        while len(chosen) < 4:
            coverage = Counter(
                (r["metadata"]["prism_eval"]["language"], r["metadata"]["prism_eval"]["arm"]) for r in chosen
            )
            remaining = [r for r in pool if r not in chosen]
            r = min(
                remaining,
                key=lambda r: (
                    coverage[(r["metadata"]["prism_eval"]["language"], r["metadata"]["prism_eval"]["arm"])],
                    r["metadata"]["prism_eval"]["estimated_seconds"],
                    r["metadata"]["prism_eval"]["id"],
                ),
            )
            chosen.append(r)
        smoke.extend(chosen)
    write_rows(args.output / "smoke.jsonl", smoke)
    for kind, names in (("small", list(DOMAINS)), ("smoke", ["smoke"])):
        config = {
            "eval": {
                "defaults": {
                    "input_key": "text",
                    "metadata_key": "metadata",
                    "apply_chat_template": False,
                    "label_key": None,
                    "multimodal_keys": {},
                    "n_samples_per_eval_prompt": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_response_len": 4096,
                    "custom_rm_path": "examples.prism.eval.runtime.score",
                    "custom_generate_function_path": "examples.prism.eval.runtime.generate",
                },
                "datasets": [{"name": "prism-" + n, "path": str(args.output / f"{n}.jsonl")} for n in names],
            }
        }
        # JSON is valid YAML and avoids a build-only YAML dependency.
        write_json(args.output / f"{kind}.yaml", config)
    source_files = {str(p): digest(p) for p in sorted(args.canonical.glob("*.jsonl"))}
    source_files.update(freezer.source_files)
    code_files = {str(p): digest(p) for p in sorted((args.moss_eval / "src").rglob("*.py"))}
    provenance = {
        "schema": "prism-small-eval.v1",
        "seed": SEED,
        "base_commit": "0486696e0aa75d10b3a517df5f23e814f289d2f2",
        "package_fingerprint": package_fingerprint,
        "counts": DOMAINS,
        "total": len(rows),
        "smoke": len(smoke),
        "assets": freezer.assets,
        "source_files": source_files,
        "platform_code_sha256": code_files,
        "selection_code_sha256": digest(__file__),
        "service_policy": ["Qwen3-ASR-1.7B", "ReDimNet2", "JudgeCLAP-v1012"],
        "metric_scope": "three-service proxy; no same-WAV platform parity claimed",
        "gt_gate": "same-service p>0.5 at evaluation time; never use an unversioned historic cache",
        "files": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
    }
    write_json(args.output / "manifest.json", provenance)
    print(
        json.dumps({"output": str(args.output), "rows": len(rows), "assets": len(freezer.assets)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
