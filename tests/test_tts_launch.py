"""Job planning, allocation validation and complete-checkpoint recovery."""

import json
import os
from pathlib import Path
import subprocess
import socket
import sys

import pytest

from examples.tts_grpo.launch import check_ports, checkpoint_for_round, environment, training_command

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]


def test_gpu_layout_reserves_services_and_checks_multi_node(monkeypatch):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        with pytest.raises(OSError):
            check_ports([occupied.getsockname()[1]])
    with pytest.raises(ValueError, match="distinct"):
        check_ports([18410, 18410])
    monkeypatch.setattr(os, "environ", {"PROMPT_DATA": "/train", "EVAL_DATA": "/eval"})
    env = environment()
    assert env["TRAIN_GPUS"] == "2" and env["START_LOCAL_ASR"] == "1"
    monkeypatch.setenv("ASR_ENDPOINT", "http://asr.invalid/transcribe")
    env = environment()
    assert env["TRAIN_GPUS"] == "3" and env["ROLLOUT_BATCH_SIZE"] == "3"
    monkeypatch.setenv("INSPIRE_NODES", "2")
    with pytest.raises(ValueError, match="8 GPUs"):
        environment()
    monkeypatch.setenv("INSPIRE_QUOTA", "8,120,1600")
    monkeypatch.setenv("ASR_ENDPOINT", "http://asr.invalid/transcribe")
    monkeypatch.setenv("ROLLOUT_BATCH_SIZE", "7")
    env = environment()
    assert env["TRAIN_GPUS"] == "7" and env["START_LOCAL_ASR"] == "0"
    command = training_command(env, Path("/output"), "/ckpt", ["http://a", "http://b"], "head:6379")
    assert command[command.index("--global-batch-size") + 1] == "28"
    assert command[command.index("--actor-num-nodes") + 1] == "2"
    assert command[command.index("--ray-address") + 1] == "head:6379"


def test_resume_selects_last_complete_checkpoint_including_rollout_zero(tmp_path):
    checkpoint = tmp_path / "running_round_0/checkpoints"
    iteration = checkpoint / "iter_0000000"
    iteration.mkdir(parents=True)
    (iteration / ".metadata").write_bytes(b"complete")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0")
    (checkpoint / "rollout").mkdir()
    (checkpoint / "rollout/global_dataset_state_dict_0.pt").write_bytes(b"sampler")
    incomplete = tmp_path / "running_round_1/checkpoints"
    incomplete.mkdir(parents=True)
    (incomplete / "latest_checkpointed_iteration.txt").write_text("7")
    assert checkpoint_for_round(tmp_path, 2, "/init") == str(checkpoint)
    assert checkpoint_for_round(tmp_path, 0, "/init") == "/init"
    (checkpoint / "rollout/global_dataset_state_dict_0.pt").unlink()
    with pytest.raises(ValueError, match="no complete"):
        checkpoint_for_round(tmp_path, 2, "/init")


@pytest.mark.parametrize("bundle", ["UNSUPPORTED=http://host\n", "TTS_ASR_URL=$(echo bad)\n", "TTS_ASR_URL=http://a\nTTS_ASR_URL=http://b\n"])
def test_endpoint_bundle_rejects_commands_unknown_keys_and_duplicates(tmp_path, monkeypatch, bundle):
    path = tmp_path / "endpoints.env"
    path.write_text(bundle)
    monkeypatch.setattr(os, "environ", {"PROMPT_DATA": "/train", "EVAL_DATA": "/eval", "ENDPOINT_BUNDLE": str(path)})
    with pytest.raises(ValueError, match="bundle"):
        environment()


def test_job_plan_runs_preflight_without_submitting_or_leaking_secrets(tmp_path):
    for name in ("model", "checkpoint", "omni", "megatron", "codec"):
        (tmp_path / name).mkdir()
    (tmp_path / "model/config.json").write_text(json.dumps({"sglang_omni_compat": {"local_layout": "gpt2_qkv_interleaved_v1"}}))
    (tmp_path / "checkpoint/latest_checkpointed_iteration.txt").write_text("0")
    (tmp_path / "checkpoint/slime_checkpoint.json").write_text('{"kind":"initialization"}')
    for name in ("train", "eval"):
        (tmp_path / f"{name}.jsonl").write_text('{"text":"hello"}\n')
    secret = tmp_path / "secret.env"
    secret.write_text("INSPIRE_API_KEY=must-not-appear\n")
    binary = tmp_path / "inspire"
    record = tmp_path / "command.json"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(record)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    env = os.environ | {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "MODEL_DIR": str(tmp_path / "model"),
        "TRAIN_CHECKPOINT": str(tmp_path / "checkpoint"),
        "OMNI_DIR": str(tmp_path / "omni"),
        "MEGATRON_DIR": str(tmp_path / "megatron"),
        "CODEC_DIR": str(tmp_path / "codec"),
        "PROMPT_DATA": str(tmp_path / "train.jsonl"),
        "EVAL_DATA": str(tmp_path / "eval.jsonl"),
        "REWARD_SECRET_ENV": str(secret),
        "ASR_ENDPOINT": "http://asr.invalid/v1/audio/transcriptions",
        "TRAIN_GPUS": "2",
        "JOB_NAME": "unit-plan",
        "INSPIRE_NODES": "1",
        "INSPIRE_QUOTA": "4,60,800",
    }
    result = subprocess.run([sys.executable, str(ROOT / "examples/tts_grpo/launch.py"), "plan"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = json.loads(record.read_text())
    assert "--dry-run" in command and "--no-auto-fault-tolerance" in command
    assert "must-not-appear" not in result.stdout + result.stderr + record.read_text()
    assert command[command.index("--quota") + 1] == "4,60,800"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
