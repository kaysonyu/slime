slime 文档
====================

.. note::

   当前分支是 TTS 精简基线，已移除模型插件和模型专用示例，保留的训练核心仍与 v0.3.2 一致。TTS 功能将在后续 PR 中加入。

slime 是一个面向 RL Scaling 的 LLM 后训练框架，提供两大核心能力：

- 高性能训练：通过连接 Megatron 与 SGLang，支持多种模式下的高效训练；
- 灵活的数据生成：通过自定义数据生成接口与基于服务器的引擎，实现任意训练数据生成流程。

slime 的设计目标，是让这两大能力彼此强化，同时避免把系统变成一组割裂的 trainer、rollout service 和 agent framework。Megatron training、SGLang rollout、custom data generation、reward computation、verifier feedback 和 environment interaction 都流经同一条 training / rollout / Data Buffer 路径。

这让 slime 成为最经受实战验证的开源 RL post-training 框架之一：它足够轻量、清晰、易扩展，同时也经过了 SOTA 级模型发布背后的完整训练闭环验证。

为什么这个设计重要
------------------

- **经过 frontier model 训练验证**：slime 是 `GLM-5.2 <https://z.ai/blog/glm-5.2>`_、`GLM-5.1 <https://z.ai/blog/glm-5.1>`_、`GLM-5 <https://z.ai/blog/glm-5>`_、`GLM-4.7 <https://z.ai/blog/glm-4.7>`_、`GLM-4.6 <https://z.ai/blog/glm-4.6>`_、`GLM-4.5 <https://z.ai/blog/glm-4.5>`_ 背后的 RL 训练框架。
- **从设计开始就是 native**：slime 直接透传 Megatron 参数，并通过 ``--sglang-`` 前缀暴露当前安装版本 SGLang 支持的参数。新的上游训练和 serving 优化可以直接使用，不需要在 slime 里再加一层 wrapper。
- **专注 SGLang rollout**：slime 有意选择单一 rollout backend，避免为了同时兼容多个 inference engine 而被迫抽象成 lowest-common-denominator 的公共能力子集，从而可以直接发挥 SGLang-specific 的 serving、routing、caching、disaggregation 和 weight-sync 能力。
- **Agentic workflow 就是数据生成**：tool use、sandbox interaction、verifier reward、environment feedback、multi-agent loop 和 long-horizon agentic workflow 都接入同一条 training / rollout / Data Buffer 路径，而不是 fork training kernel。
- **BF16 训练 + FP8 rollout**：大规模 MoE recipe 使用 Megatron BF16 training state 搭配 SGLang FP8 rollout/inference；long-context rollout 还可以通过 ``--sglang-kv-cache-dtype fp8_e4m3`` 提升有效 KV cache 容量。
- **核心验证**：CPU correctness tests 默认运行，通用 GPU log-probability/entropy 数值测试仍由 label 触发。详见 :doc:`developer_guide/ci`。

按使用场景开始
--------------

- 配置 training 和 rollout 参数：:doc:`get_started/usage`
- 添加 custom generation、reward 或 rollout function：:doc:`get_started/customization`
- 配置生产级 SGLang rollout topology：:doc:`advanced/sglang-config`
- 接入 external rollout engines：:doc:`advanced/external-rollout-engines`
- 以字节级 delta 同步权重：:doc:`advanced/delta-weight-sync`
- 使用 PD disaggregation：:doc:`advanced/pd-disaggregation`
- 使用 BF16 训练 + FP8 rollout 或 FP8 KV cache：:doc:`advanced/low-precision`
- 了解 CI 和可靠性覆盖：:doc:`developer_guide/ci`
- 调试、trace 和 profiling 长时间任务：:doc:`developer_guide/debug`、:doc:`developer_guide/trace`、:doc:`developer_guide/profiling`

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

   get_started/usage.md
   get_started/customization.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: 高级特性

   advanced/on-policy-distillation.md
   advanced/speculative-decoding.md
   advanced/low-precision.md
   advanced/reproducibility.md
   advanced/fault-tolerance.md
   advanced/observability.md
   advanced/pd-disaggregation.md
   advanced/external-rollout-engines.md
   advanced/delta-weight-sync.md
   advanced/sglang-config.md
   advanced/megatron-config.md

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md

.. toctree::
   :maxdepth: 1
   :caption: 博客

   blogs/release_v0.1.0.md
   blogs/introducing_slime.md
