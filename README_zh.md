# slime — TTS RL 精简基线

[English](README.md)

本分支从 [slime v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2)
（`3778dbf6`）开始整理。原生 TTS 模型、rollout、奖励与训练脚本由
[功能 PR](https://github.com/kaysonyu/slime/pull/3) 加入。

当前保留 `train.py`、Ray、Megatron、SGLang 运行时、权重转换库、训练公共模块及其测试。
精简范围包括无关示例和模型插件、上游文档站、旧 Docker/Conda 构建与补丁、独立转换和分析工具。
需要随新后端替换的运行时代码留在功能 PR。

此基线使用已经准备好的训练环境。旧镜像构建方式保留在
[上游 v0.3.2](https://github.com/THUDM/slime/tree/v0.3.2/docker)，
TTS 功能 PR 会提供对应的环境依赖和启动说明。

保留的 CPU 测试登记在 [CI 模板](.github/workflows/pr-test.yml.j2)，GPU 和 SGLang 环境检查另行运行。
格式检查命令：

```bash
pre-commit run --all-files --show-diff-on-failure
```

历史文档入口见 [docs/README.md](docs/README.md)。项目保留上游的 [Apache 2.0 许可证](LICENSE)。
