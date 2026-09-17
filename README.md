# slime — TTS RL cleanup baseline

[中文版](README_zh.md)

This branch starts from [slime v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2)
(`3778dbf6`) and prepares the repository for TTS reinforcement learning.
Native TTS models, rollout, rewards and launch scripts are introduced in
[the implementation PR](https://github.com/kaysonyu/slime/pull/3).

The cleanup retains the existing `train.py` / Ray / Megatron / SGLang runtime,
weight conversion libraries, training utilities and their tests. It removes
unrelated examples and model plugins, the upstream documentation site, legacy
Docker/Conda build recipes and patches, and standalone conversion/profiling tools.
Runtime changes remain with the implementation that replaces them.

The cleanup baseline expects an already prepared training environment. The
legacy image-building recipes are available in
[upstream v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2/docker).
The TTS implementation supplies its own environment requirements and launch instructions.

The [CI template](.github/workflows/pr-test.yml.j2) lists the retained CPU tests;
GPU and SGLang environment checks remain separate. Run repository formatting checks with:

```bash
pre-commit run --all-files --show-diff-on-failure
```

Historical documentation is linked from [docs/README.md](docs/README.md).
The project retains its upstream [Apache 2.0 license](LICENSE).
