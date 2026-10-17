"""CPU contracts for Prism recipes, output directories, and owned Ray lifecycle."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NUM_GPUS = 0


def command_options(command):
    options = {}
    for index, argument in enumerate(command):
        if argument.startswith("--"):
            key, separator, value = argument[2:].partition("=")
            if not separator:
                value = command[index + 1] if index + 1 < len(command) else True
                if isinstance(value, str) and value.startswith("--"):
                    value = True
            options[key.replace("-", "_")] = value
    return options


@pytest.fixture
def runtime(tmp_path):
    from test_prism_argument_configuration import _synthetic_prism_config

    hf = tmp_path / "student with spaces"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_synthetic_prism_config()))
    custom = tmp_path / "custom.yaml"
    custom.write_text("omni_return_audio: false\ncustom_rm_path: user.reward\n")
    interpreter = tmp_path / "training python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
        import json, os, sys, time
        from pathlib import Path
        root = Path(os.environ['TEST_ROOT'])
        rank = os.environ.get('PET_NODE_RANK', os.environ.get('RANK', '0'))
        if sys.argv[1:3] == ['-m', 'ray.scripts.scripts']:
            with (root / f'ray-{rank}.jsonl').open('a') as f:
                f.write(json.dumps(sys.argv[3:]) + '\\n')
            if sys.argv[3] == 'start':
                env = {key: value for key, value in os.environ.items()
                       if key.startswith(('NCCL_', 'GLOO_'))}
                (root / f'network-{rank}.json').write_text(json.dumps(env))
                if rank == os.environ.get('TEST_FAIL_RAY_RANK'): sys.exit(9)
        elif len(sys.argv) > 1 and sys.argv[1].endswith('/train.py'):
            state = Path(os.environ['TEST_ROUND_PATH']) / '.ray'
            nodes = int(os.environ.get('PET_NNODES', os.environ.get('WORLD_SIZE', '1')))
            assert all((state / f'node-{i}.ready').is_file() for i in range(nodes))
            (root / 'train.json').write_text(json.dumps(sys.argv[1:]))
            with (root / 'driver.jsonl').open('a') as f:
                f.write(json.dumps({'rank': rank, 'pid': os.getpid(),
                                    'address': os.environ['RAY_ADDRESS']}) + '\\n')
            print('training complete', flush=True)
            while os.environ.get('TEST_WAIT_DRIVER'): time.sleep(0.1)
            sys.exit(int(os.environ.get('TEST_TRAIN_STATUS', '0')))
        else:
            with (root / 'python-calls.jsonl').open('a') as f:
                f.write(json.dumps(sys.argv[1:]) + '\\n')
        """
        )
        + f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    interpreter.chmod(0o755)
    output_root = tmp_path / "outputs with spaces"
    round_path = output_root / "prism-tests" / "trial_job001" / "running_round_0"
    environment = dict(os.environ)
    for name in ("MASTER_ADDR", "MASTER_PORT", "PET_MASTER_ADDR", "PET_MASTER_PORT", "RANK", "WORLD_SIZE"):
        environment.pop(name, None)
    environment.update(
        TRAIN_PYTHON=str(interpreter),
        TEST_ROOT=str(tmp_path),
        OUTPUT_ROOT=str(output_root),
        RUN_NAME="prism-tests",
        JOB_NAME="trial",
        TRAIN_JOB_ID="job001",
        RUNNING_ROUND="0",
        TEST_ROUND_PATH=str(round_path),
        PET_NNODES="1",
        PET_NODE_RANK="0",
        PET_NPROC_PER_NODE="1",
        RAY_START_TIMEOUT="5",
        RAY_JOB_TIMEOUT="10",
    )
    return {
        "env": environment,
        "args": ["--hf-checkpoint", str(hf), "--custom-config-path", str(custom)],
        "root": tmp_path,
        "round": round_path,
    }


def run_recipe(runtime, mode, extra=()):
    return subprocess.run(
        ["bash", str(REPO / f"examples/prism/run-{mode}.sh"), *runtime["args"], *extra],
        env=runtime["env"],
        cwd=runtime["root"],
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("mode,samples,gpus", [("grpo", "4", "1"), ("mopd", "1", "1")])
def test_recipes_execute_train_directly_with_native_parameters(runtime, mode, samples, gpus):
    result = run_recipe(runtime, mode)
    assert result.returncode == 0, result.stderr
    command = json.loads((runtime["root"] / "train.json").read_text())
    assert command[0] == str(REPO / "train.py")
    options = command_options(command)
    assert options["n_samples_per_prompt"] == samples
    assert options["rollout_num_gpus"] == gpus
    assert options["rollout_backend"] == "sglang-omni"
    assert options.get("no_omni_return_audio" if mode == "mopd" else "omni_return_audio") is True
    assert options["advantage_estimator"] == "grpo"
    assert ("use_opd" in options) == (mode == "mopd")
    assert options["hf_checkpoint"] == runtime["args"][1]
    assert options["custom_model_provider_path"] == "slime_plugins.models.prism.model_provider"
    if mode == "grpo":
        assert options["custom_rm_path"] == "examples.prism.rewards.base.reward"
    else:
        assert "custom_rm_path" not in options
        assert "opd_teacher_config" in options
        assert "rollout_sample_hook_path" not in options
    assert int(options["rollout_batch_size"]) * int(samples) == int(options["global_batch_size"])
    # Besides train.py, the shell calls only the existing model configuration module.
    calls = [json.loads(line) for line in (runtime["root"] / "python-calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][:2] == ["-m", "slime_plugins.models.prism_configuration"]
    assert ray_calls(runtime, 0)[-1] == ["stop", "--force"]
    assert options["save"] == str(runtime["round"] / "checkpoint")
    assert options["tensorboard_dir"] == str(runtime["round"] / "tensorboard")
    assert options["omni_audio_dir"] == str(runtime["round"] / "rollout_audio")
    assert options["resolved_config_path"] == str(runtime["round"] / "config/resolved-args.json")
    assert options["phase_trace_path"] == str(runtime["round"] / "logs/phases.jsonl")
    assert "training complete" in (runtime["round"] / "logs/rank0.log").read_text()
    assert Path(options["prompt_data"]).is_file()
    assert options["omni_config"] == str(REPO / "examples/prism/configs/student.yaml")
    assert options["omni_update_stage"] == "tts_engine"
    capabilities = command.index("--omni-worker-capabilities")
    assert command[capabilities + 1 : capabilities + 4] == ["chat", "audio_input", "audio_output"]


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
def test_native_overrides_and_custom_config_survive_without_rewriting(runtime, mode):
    custom = Path(runtime["args"][-1])
    content = custom.read_text()
    runtime["env"].update(LEARNING_RATE="100", GLOBAL_BATCH_SIZE="999", RL_LOAD="/missing")
    result = run_recipe(
        runtime,
        mode,
        [
            "--lr",
            "5e-7",
            "--global-batch-size",
            "4",
            "--rollout-batch-size",
            "8",
            "--omni-admin-key",
            "test-secret",
            "--user-hook-option",
            "literal $(command) `value`",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert "test-secret" not in result.stdout + result.stderr
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["lr"] == "5e-7"
    assert options["global_batch_size"] == "4" and options["rollout_batch_size"] == "8"
    assert options["custom_config_path"] == str(custom)
    assert options["omni_admin_key"] == "test-secret"
    assert options["user_hook_option"] == "literal $(command) `value`"
    assert "load" not in options
    assert custom.read_text() == content


def test_architecture_uses_last_native_hf_checkpoint_including_equals_form(runtime):
    other = runtime["root"] / "second student"
    other.mkdir()
    config = json.loads((Path(runtime["args"][1]) / "config.json").read_text())
    config["language_config"]["hidden_size"] *= 2
    (other / "config.json").write_text(json.dumps(config))
    result = run_recipe(runtime, "grpo", [f"--hf-checkpoint={other}"])
    assert result.returncode == 0, result.stderr
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["hf_checkpoint"] == str(other)
    assert options["hidden_size"] == str(config["language_config"]["hidden_size"])


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
def test_invalid_model_config_stops_before_training(runtime, mode):
    (Path(runtime["args"][1]) / "config.json").write_text("{}")
    result = run_recipe(runtime, mode)
    assert result.returncode != 0 and "moss_tts_prism" in result.stdout + result.stderr
    assert not (runtime["root"] / "train.json").exists()


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
def test_removed_dry_run_is_rejected_instead_of_starting_training(runtime, mode):
    result = run_recipe(runtime, mode, ["--dry-run"])
    assert result.returncode == 2 and "--dry-run was removed" in result.stderr
    assert not (runtime["root"] / "train.json").exists()
    assert not (runtime["root"] / "python-calls.jsonl").exists()


def test_copied_recipe_uses_explicit_source_root(runtime):
    copied = runtime["root"] / "my-experiment.sh"
    copied.write_text((REPO / "examples/prism/run-grpo.sh").read_text())
    arguments = ["--hf-checkpoint", runtime["args"][1]]
    result = subprocess.run(
        ["bash", str(copied), *arguments],
        env={**runtime["env"], "SLIME_ROOT": str(REPO)},
        cwd=runtime["root"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    command = json.loads((runtime["root"] / "train.json").read_text())
    assert command[0] == str(REPO / "train.py")
    assert command_options(command)["omni_return_audio"] is True
    assert "custom_config_path" not in command_options(command)


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
def test_recipe_passes_native_resume_and_propagates_training_failure(runtime, mode):
    runtime["env"]["TEST_TRAIN_STATUS"] = "7"
    result = run_recipe(runtime, mode, ["--load", "/previous/checkpoint", "--use-checkpoint-opt-param-scheduler"])
    assert result.returncode == 7
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["load"] == "/previous/checkpoint"
    assert options["use_checkpoint_opt_param_scheduler"] is True
    # Native Slime gives explicit --load precedence over the fresh-run initializer.
    assert options["custom_megatron_model_initializer_path"].endswith("load_prism_pretrained")


def ray_calls(runtime, rank):
    path = runtime["root"] / f"ray-{rank}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def finish_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
@pytest.mark.parametrize("nodes,status", [(1, 0), (1, 7), (2, 0), (2, 7)])
def test_recipes_start_each_node_run_only_head_and_propagate_status(runtime, mode, nodes, status):
    network = {
        "NCCL_IB_QPS_PER_CONNECTION": "4",
        "NCCL_GDR_LEVEL": "2",
        "NCCL_IB_PCI_RELAXED_ORDERING": "1",
        "NCCL_IB_TC": "160",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "GLOO_SOCKET_IFNAME": "eth0",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_HCA": "mlx5_0,mlx5_1",
    }
    runtime["env"].update(network)
    runtime["env"].update(
        PET_NNODES=str(nodes),
        PET_NPROC_PER_NODE="2",
        MASTER_ADDR="head.internal",
        MASTER_PORT="7123",
        PET_MASTER_ADDR="unused.internal",
        PET_MASTER_PORT="9999",
        TEST_TRAIN_STATUS=str(status),
        # PET values describe nodes and take precedence over generic ranks/counts.
        WORLD_SIZE="64",
        RANK="63",
    )
    processes = []
    try:
        for rank in reversed(range(nodes)):
            processes.append(
                subprocess.Popen(
                    [
                        "bash",
                        str(REPO / f"examples/prism/run-{mode}.sh"),
                        *runtime["args"],
                        "--omni-admin-key",
                        "test-secret",
                    ],
                    env={**runtime["env"], "PET_NODE_RANK": str(rank)},
                    cwd=runtime["root"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == status, stdout + stderr
            assert "test-secret" not in stdout + stderr
    finally:
        for process in processes:
            finish_process(process)
    calls = [json.loads(line) for line in (runtime["root"] / "driver.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["rank"] == "0"
    assert calls[0]["address"] == "head.internal:7123"
    assert (runtime["round"] / ".ray/exit-code").read_text().strip() == str(status)
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["actor_num_nodes"] == str(nodes)
    assert options["actor_num_gpus_per_node"] == "2"
    assert options["rollout_num_gpus"] == str(nodes * 2)
    assert int(options["rollout_batch_size"]) * int(options["n_samples_per_prompt"]) == int(
        options["global_batch_size"]
    )
    for rank in range(nodes):
        start, stop = ray_calls(runtime, rank)
        assert start[0] == "start" and stop == ["stop", "--force"]
        options = command_options(start)
        assert options["num_gpus"] == "2"
        assert "num_cpus" not in options
        assert ("head" in options) == (rank == 0)
        assert options["address" if rank else "port"] == ("head.internal:7123" if rank else "7123")
        inherited = json.loads((runtime["root"] / f"network-{rank}.json").read_text())
        assert all(inherited[key] == value for key, value in network.items())
        assert (runtime["round"] / f"logs/rank{rank}.log").is_file()


def test_generic_rank_and_pet_master_fallback(runtime):
    runtime["env"].pop("PET_NODE_RANK")
    runtime["env"].pop("PET_NNODES")
    runtime["env"].update(RANK="0", WORLD_SIZE="1", PET_MASTER_ADDR="localhost", PET_MASTER_PORT="7234")
    result = run_recipe(runtime, "grpo")
    assert result.returncode == 0, result.stdout + result.stderr
    assert command_options(ray_calls(runtime, 0)[0])["port"] == "7234"
    call = json.loads((runtime["root"] / "driver.jsonl").read_text())
    assert call["address"] == "localhost:7234"


@pytest.mark.parametrize("mode", ["grpo", "mopd"])
def test_local_defaults_and_new_round_keep_previous_outputs(runtime, mode):
    for key in (
        "RUN_NAME",
        "JOB_NAME",
        "TRAIN_JOB_ID",
        "RUNNING_ROUND",
        "PET_NODE_RANK",
        "PET_NNODES",
        "PET_NPROC_PER_NODE",
    ):
        runtime["env"].pop(key, None)
    job = Path(runtime["env"]["OUTPUT_ROOT"]) / f"prism-{mode}" / "manual_local"
    runtime["env"]["TEST_ROUND_PATH"] = str(job / "running_round_0")
    result = run_recipe(runtime, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    previous = job / "running_round_0"
    log = (previous / "logs/rank0.log").read_bytes()
    runtime["env"].update(RUNNING_ROUND="1", TEST_ROUND_PATH=str(job / "running_round_1"))
    result = run_recipe(runtime, mode, ["--load", str(previous / "checkpoint")])
    assert result.returncode == 0, result.stdout + result.stderr
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["load"] == str(previous / "checkpoint")
    assert options["save"] == str(job / "running_round_1/checkpoint")
    assert (previous / "logs/rank0.log").read_bytes() == log


def test_mixrl_uses_the_grpo_recipe_and_explicit_category_reward(runtime):
    result = run_recipe(
        runtime,
        "grpo",
        [
            "--prompt-data",
            str(REPO / "examples/prism/data/mixrl.example.jsonl"),
            "--custom-rm-path",
            "examples.prism.rewards.reward",
        ],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    options = command_options(json.loads((runtime["root"] / "train.json").read_text()))
    assert options["custom_rm_path"] == "examples.prism.rewards.reward"
    assert options["prompt_data"].endswith("mixrl.example.jsonl")


@pytest.mark.parametrize("failure", ["ray_start", "missing_worker"])
def test_startup_failure_cleans_ray_without_starting_training(runtime, failure):
    if failure == "ray_start":
        runtime["env"]["TEST_FAIL_RAY_RANK"] = "0"
    else:
        runtime["env"].update(PET_NNODES="2", MASTER_ADDR="head.internal", RAY_START_TIMEOUT="1")
    result = run_recipe(runtime, "grpo")
    assert result.returncode == (9 if failure == "ray_start" else 1), result.stdout + result.stderr
    assert not (runtime["root"] / "train.json").exists()
    assert ray_calls(runtime, 0)[-1] == ["stop", "--force"]
    assert (runtime["round"] / ".ray/exit-code").read_text().strip() == str(result.returncode)


@pytest.mark.parametrize("rank", [0, 1])
def test_completed_round_is_not_reused(runtime, rank):
    directory = runtime["round"] / ".ray"
    directory.mkdir(parents=True)
    (directory / "exit-code").write_text("0\n")
    runtime["env"].update(PET_NODE_RANK=str(rank), PET_NNODES="2", MASTER_ADDR="head.internal")
    result = run_recipe(runtime, "grpo")
    assert result.returncode == 2
    assert not ray_calls(runtime, rank)
    assert not (runtime["root"] / "train.json").exists()
    assert (directory / "exit-code").read_text() == "0\n"


def test_termination_reaches_training_and_stops_ray(runtime):
    runtime["env"]["TEST_WAIT_DRIVER"] = "1"
    process = subprocess.Popen(
        ["bash", str(REPO / "examples/prism/run-grpo.sh"), *runtime["args"]],
        env=runtime["env"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        record = runtime["root"] / "driver.jsonl"
        deadline = time.monotonic() + 10
        while not record.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert record.exists(), "training did not start"
        driver_pid = json.loads(record.read_text())["pid"]
        process.terminate()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 143, stdout + stderr
        with pytest.raises(ProcessLookupError):
            os.kill(driver_pid, 0)
        assert ray_calls(runtime, 0)[-1] == ["stop", "--force"]
        assert (runtime["round"] / ".ray/exit-code").read_text().strip() == "143"
    finally:
        finish_process(process)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
