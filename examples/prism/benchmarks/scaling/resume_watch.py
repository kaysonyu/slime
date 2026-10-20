"""CPU polling that wakes only the existing authorized experiment thread."""

import argparse
import fcntl
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from campaign import atomic_json


def collect_events(root: Path, now: float) -> list[str]:
    manifest = json.loads((root / "manifest.json").read_text())
    events = []
    for failure in sorted((root / "platform/lifecycle-extension").glob("failed-*.json")):
        events.append(f"recovery-controller-failed:{failure.stem}:{int(failure.stat().st_mtime)}")
    if (root / "campaign-finished.json").exists() or (root / "platform/monitor-finished.json").exists():
        events.append("terminal")
    for case in manifest["cases"]:
        path = root / "runs" / case["name"] / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            events.append(("measured:" if summary.get("valid") else "failure:") + case["name"])
    if len(list((root / "platform").glob("node-*.json"))) == manifest["nodes"]:
        events.append("allocated-128")
    heartbeat = root / "platform/monitor-state.json"
    if heartbeat.exists() and now - heartbeat.stat().st_mtime > 600:
        events.append(f"monitor-stale:{int(now // 1800)}")
    request_path = root / "services/request.json"
    if request_path.exists():
        request = json.loads(request_path.read_text())
        run = root / "runs" / request["case"]
        waiting = now - request_path.stat().st_mtime
        # Wake before the GPU parents' two-hour barrier expires, not every poll.
        if waiting >= 900 and not (run / "services-ready.json").exists() and not (run / "summary.json").exists():
            events.append(f"asr-readiness-stalled:{request['case']}:{int((waiting - 900) // 1800)}")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--rpc-library", type=Path, required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.root
    spec = importlib.util.spec_from_file_location("capacity_rpc", args.rpc_library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.launch:
        with module.RPC(args.socket) as client:
            thread = client.call("thread/read", {"threadId": args.thread_id, "includeTurns": False})["thread"]
            client.call("thread/queue/list", {"threadId": args.thread_id, "limit": 100})
            assert thread["id"] == args.thread_id
        if (root / "platform/controller-pids.json").exists():
            raise FileExistsError("Inspect existing controllers before restarting")
        commands = {
            "watch": [sys.executable, "-u", str(Path(__file__).with_name("watch.py")), str(root)],
            "resume": [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *[a for a in sys.argv[1:] if a != "--launch"],
            ],
        }
        processes = {}
        for name, command in commands.items():
            with (root / "platform" / f"{name}.log").open("a") as log:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            processes[name] = {"pid": process.pid, "command": command}
        atomic_json(
            root / "platform/controller-pids.json",
            {"time": time.time(), "thread_id": args.thread_id, "processes": processes},
        )
        print(json.dumps(processes))
        return
    lock = (root / "platform/resume.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "platform/resume-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"handled": []}
    while True:
        try:
            events = collect_events(root, time.time())
            event = next((value for value in events if value not in state["handled"]), None)
            if event:
                with module.RPC(args.socket) as client:
                    thread = client.call("thread/read", {"threadId": args.thread_id, "includeTurns": False})["thread"]
                    queued = client.call("thread/queue/list", {"threadId": args.thread_id, "limit": 100})["data"]
                    identity = str(uuid.uuid5(uuid.NAMESPACE_URL, f"prism-grpo128:{root}:{event}"))
                    own = next((entry for entry in queued if entry.get("clientUserMessageId") == identity), None)
                    if thread["status"]["type"] == "idle" and (not queued or own is not None):
                        if own is None:
                            prompt = (
                                f"定时监控触发：Prism GRPO 128 卡扩展实验 {event}。继续原任务，不另开任务或 agent。"
                                f"实验目录 {root}；分支 perf/prism-grpo-128gpu-20261020；工作树 {args.worktree}。"
                                "先读 manifest、platform/monitor-state、runs/*/summary 与日志。等待 128 H200 实际就绪后跑完全部实验；"
                                "范围为 Prism 1.7B、fullclone_chinese、真实 Qwen3-ASR WER、128 张训推共卡、ASR 另计最多32张。"
                                "检查同轮64/B4096与128/B8192扩展效率、100步稳定性和checkpoint恢复；失败时优先在保留的同轮资源内诊断修复补测。"
                                "不得用提交成功、短跑或平台SUCCEEDED替代实际验收。只有结果充分才总结并推送；排队继续CPU监控，不持续轮询耗token。"
                            )
                            own = client.call(
                                "thread/queue/add",
                                {
                                    "threadId": args.thread_id,
                                    "clientUserMessageId": identity,
                                    "input": [{"type": "text", "text": prompt, "text_elements": []}],
                                },
                            )["queuedSubmission"]
                        state["handled"].append(event)
                        state["last_submission_id"] = own["id"]
                        atomic_json(state_path, state)
                        try:
                            client.call(
                                "thread/queue/start", {"threadId": args.thread_id, "queuedSubmissionId": own["id"]}
                            )
                        except RuntimeError as error:
                            if "queued submission not found" not in str(error):
                                raise
                        print(json.dumps({"time": time.time(), "triggered": event}), flush=True)
                        if event == "terminal":
                            return
            state["last_check"] = time.time()
            state.pop("error", None)
            atomic_json(state_path, state)
        except Exception as error:
            state.update(last_check=time.time(), error=str(error))
            atomic_json(state_path, state)
            print(json.dumps({"error": str(error)}), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
