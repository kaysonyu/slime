"""Installed-package boundaries and resources for clients and GPU serving."""

import ast
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed_package(tmp_path_factory):
    target = tmp_path_factory.mktemp("audio-reward-wheel")
    (target / "metadata").mkdir()
    command = [
        sys.executable,
        "setup.py",
        "build",
        "--build-base",
        str(target / "build"),
        "egg_info",
        "--egg-base",
        str(target / "metadata"),
        "bdist_wheel",
        "--dist-dir",
        str(target / "dist"),
        "--bdist-dir",
        str(target / "bdist"),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    wheel = next((target / "dist").glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(target / "site"),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return target, wheel


def test_client_imports_without_training_or_serving_dependencies(installed_package):
    target, _ = installed_package
    script = r"""
import importlib.abc, importlib.util, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
blocked = {"torch", "torchaudio", "torchvision", "transformers", "peft", "fastapi", "uvicorn", "numpy", "ray", "examples", "slime"}
class DenyHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("client imported optional dependency: " + fullname)
sys.meta_path.insert(0, DenyHeavyImports())
from slime_plugins.audio_rewards import client, scoring, serving
from slime_plugins.prism.reward_adapter import PrismRewardAdapter
assert pathlib.Path(client.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
assert importlib.util.find_spec("slime_plugins.tts_reward") is None
assert importlib.util.find_spec("slime_plugins.embedding_cache") is None
assert not any(name.startswith("slime_plugins.audio_rewards.serving.") for name in sys.modules)
assert scoring.transcription_score("hello", "hello", "en")["transformed"] == 1
print(json.dumps({"client": client.__file__, "light_import": True}))
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(target / "site")],
        cwd=target,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["light_import"]


def test_wheel_contains_model_definitions_and_resources(installed_package):
    target, wheel = installed_package
    prefix = "slime_plugins/audio_rewards/serving/vendor/"
    with zipfile.ZipFile(wheel) as archive:
        # This repository's custom wheel class uses a .data/purelib section.
        names = {name.split(".data/purelib/", 1)[-1]: name for name in archive.namelist()}
        assert not any(name.startswith("examples/") for name in names)
        for name in [
            "SOURCES.md",
            "judge_clap/modeling.py",
            "judge_clap/collator.py",
            "judge_clap/categories.json",
            "redimnet2/redimnet2.py",
            "redimnet2/layers/features_tf.py",
            "redimnet2/layers/attention.py",
            "redimnet2/LICENSE",
            "redimnet2/LICENSE-APACHE-2.0",
        ]:
            assert prefix + name in names
            assert (target / "site" / prefix / name).is_file()
        categories = json.loads(archive.read(names[prefix + "judge_clap/categories.json"]))
        assert len(categories) == 27 and "Loudness" not in categories
        stats_path = "slime_plugins/audio_rewards/prosody_stats.json"
        assert set(json.loads(archive.read(names[stats_path]))) == {"zh", "en"}
        assert (target / "site" / stats_path).is_file()


def test_installed_serving_cli_does_not_need_examples_checkout(installed_package):
    pytest.importorskip("fastapi")
    pytest.importorskip("soundfile")
    target, _ = installed_package
    script = """
import pathlib, runpy, sys
sys.path.insert(0, sys.argv[1])
import slime_plugins.audio_rewards.serving.server as server
assert pathlib.Path(server.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
sys.argv = ["server", "--help"]
server.main()
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script, str(target / "site")],
        cwd=target,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--model" in result.stdout and "judge_clap" in result.stdout


def test_package_dependency_direction():
    package = ROOT / "slime_plugins/audio_rewards"
    for path in package.rglob("*.py"):
        if "vendor" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            for module in modules:
                assert not module.startswith(("examples", "slime_plugins.prism")), (path, module)
                if "serving" in path.relative_to(package).parts:
                    assert module.split(".")[0] != "slime", (path, module)
                else:
                    assert "serving" not in module.split("."), (path, module)
