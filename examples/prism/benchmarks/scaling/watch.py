"""CPU-only lifecycle controller: poll the queue and satisfy ASR scale requests.

No language-model polling is involved. Only the named campaign ASR is changed.
"""

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from campaign import atomic_json


def cli(*arguments):
    result = subprocess.run(["inspire", "--json", *arguments], text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"inspire {arguments[:2]} failed: {result.stderr[-500:]}")
    value = json.loads(result.stdout)
    if not value.get("success"):
        raise RuntimeError(f"inspire {arguments[:2]} returned failure")
    return value.get("data", {})


def smoke(manifest):
    token = Path(manifest["auth_file"]).read_text().strip()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(manifest["asr_endpoint"] + "/info", headers=headers)
    with opener.open(request, timeout=20) as response:
        info = json.load(response)
    if info.get("protocol") != "voxrl.scoring.v1":
        raise ValueError("Incorrect ASR protocol")
    with (Path(manifest["root"]) / "data/train.jsonl").open() as stream:
        row = json.loads(next(stream))
    audio = next(iter(row["metadata"]["omni_prompt"]["references"].values()))
    payload = {"request_id": f"scale-smoke-{time.time_ns()}", "audio": {"path": audio}}
    request = urllib.request.Request(
        manifest["asr_endpoint"] + "/score", headers=headers, data=json.dumps(payload).encode()
    )
    with opener.open(request, timeout=90) as response:
        result = json.load(response)
    if result.get("request_id") != payload["request_id"] or not isinstance(result.get("result", {}).get("text"), str):
        raise ValueError("ASR business smoke failed")
    return {"info": info, "score": result, "time": time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    workspace, service = manifest["workspace"], manifest["asr_name"]
    started = time.monotonic()
    acknowledged = None
    initial_smoke = (root / "services/initial-smoke.json").exists()
    while True:
        try:
            job = cli("job", "status", manifest["job_name"], "--workspace", workspace)
            serving = cli("serving", "status", service, "--workspace", workspace)
            status = job.get("status", "UNKNOWN")
            atomic_json(root / "platform/monitor-state.json", {"time": time.time(), "job": job, "serving": serving})
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                if serving.get("status") != "STOPPED":
                    cli("serving", "stop", service, "--workspace", workspace)
                from analyze import analyze

                analyze(root)
                atomic_json(root / "platform/monitor-finished.json", {"time": time.time(), "job_status": status})
                return
            if not initial_smoke and serving.get("status") == "RUNNING":
                atomic_json(root / "services/initial-smoke.json", smoke(manifest))
                initial_smoke = True
            if status == "RUNNING":
                if serving.get("status") == "STOPPED":
                    cli("serving", "start", service, "--workspace", workspace)
                    acknowledged = None
                request_file = root / "services/request.json"
                if request_file.exists():
                    request = json.loads(request_file.read_text())
                    if request["case"] != acknowledged:
                        replicas = request["replicas"]
                        if replicas not in {4, 8, 16}:
                            raise ValueError("ASR request exceeds approved matrix")
                        if serving.get("replicas") != replicas:
                            cli("serving", "scale", service, "--workspace", workspace, "--replicas", str(replicas))
                        instances = cli("serving", "instances", service, "--workspace", workspace).get("items", [])
                        if len(instances) == replicas and all(
                            item["status"] == "INSTANCE_RUNNING" for item in instances
                        ):
                            receipt = smoke(manifest)
                            receipt.update(case=request["case"], replicas=replicas, instances=instances)
                            atomic_json(root / "runs" / request["case"] / "services-ready.json", receipt)
                            acknowledged = request["case"]
            elif initial_smoke and time.monotonic() - started > 900 and serving.get("status") == "RUNNING":
                # Retain the tested deployment configuration while waiting for eight nodes.
                cli("serving", "stop", service, "--workspace", workspace)
            print(
                json.dumps(
                    {"time": time.time(), "job": status, "serving": serving.get("status"), "acknowledged": acknowledged}
                ),
                flush=True,
            )
        except Exception as error:
            atomic_json(root / "platform/monitor-error.json", {"time": time.time(), "error": str(error)})
            print(json.dumps({"time": time.time(), "error": str(error)}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
