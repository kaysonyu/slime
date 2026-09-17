# MOSS-TTS 奖励服务的 Inspire 部署

这个目录用一个入口管理 ASR、SIM 和 Judge 三套独立的 Inspire Custom Serving：

```text
serving/
├── manage.py             # 本地配置校验和 Serving 生命周期
├── config.toml           # 三套服务的独立部署配置
├── endpoint.py           # 认证 URL、endpoint state 和 bundle
├── runtime/launch_sim.sh # 远端 SIM 进程入口
└── README.md
```

用户只需要操作 `manage.py` 和一份 TOML 配置。`endpoint.py` 使用 Inspire CLI 自己的
Python 环境，`launch_sim.sh` 运行在远端 Serving 容器内，都不是用户入口。

## 1. 准备配置

默认 [`config.toml`](config.toml) 固定了当前项目的调度、镜像和模型默认值，但 service
name、生成音频根和 reference 根仍包含 `REPLACE_ME`，不能直接部署。推荐复制到已被
Git 忽略的本地文件：

```bash
SERVING=examples/tts_grpo/serving
cp "$SERVING/config.toml" "$SERVING/config.local.toml"
```

编辑 `config.local.toml` 中所有 `REPLACE_ME`。配置分为 `[asr]`、`[sim]`、`[judge]`
三个完整 section；即使调度值相同也不共享隐式 fallback，修改一套服务不会改变另外两套。

配置中不能放 API key、endpoint URL 或其他凭据。训练密钥继续保存在项目内、未跟踪且
权限为 `0600` 的 `examples/tts_grpo/reward-secret.env`。

## 2. 检查和部署

入口格式只有一种：

```text
manage.py ACTION TARGET [--config PATH]
```

旧的 `asr.sh`、`sim.sh`、`judge.sh`、`deployment.env` 和三个
`*_SERVING_CONFIG` 入口已移除；部署配置统一通过 `--config` 传入。

先检查单个服务或全部服务：

```bash
"$SERVING/manage.py" plan asr --config "$SERVING/config.local.toml"
"$SERVING/manage.py" plan sim --config "$SERVING/config.local.toml"
"$SERVING/manage.py" plan judge --config "$SERVING/config.local.toml"

"$SERVING/manage.py" plan all --config "$SERVING/config.local.toml"
```

确认 dry-run 中每套服务自己的 quota、replicas、image 和启动命令后，逐个部署：

```bash
"$SERVING/manage.py" deploy asr --config "$SERVING/config.local.toml"
"$SERVING/manage.py" deploy sim --config "$SERVING/config.local.toml"
"$SERVING/manage.py" deploy judge --config "$SERVING/config.local.toml"
```

`deploy all` 被明确禁止。三套服务相互独立，逐个部署可以避免后一套失败时产生隐式回滚、
重复创建或误删已经运行的服务。

每次 `deploy` 会完成：

1. 配置、模型和共享路径校验；
2. `inspire serving create --dry-run`；
3. 精确 service name 不存在检查；
4. 一次真实 create，写操作不自动重试；
5. 等待 `RUNNING`；
6. 读取 canonical URL 并写 endpoint 文件。

只读 list、status、endpoint 查询和 dry-run 使用有限指数退避重试。

## 3. Endpoint 文件

每套服务保留一个按 service name 隔离、权限为 `0600` 的状态文件：

```text
.state/asr/<service-name>.env
.state/sim/<service-name>.env
.state/judge/<service-name>.env
```

同时原子更新项目内未跟踪的：

```text
examples/tts_grpo/endpoints.env
```

该 bundle 只允许：

```text
TTS_ASR_URL
TTS_SIM_URL
TTS_JUDGE_URL
```

单独部署或刷新一个服务时，其他已有条目会保留，因此部分部署可以正常生成部分 bundle：

```bash
"$SERVING/manage.py" endpoint sim --config "$SERVING/config.local.toml"
```

三套服务都在运行时可以一次刷新：

```bash
"$SERVING/manage.py" endpoint all --config "$SERVING/config.local.toml"
```

`endpoint all` 是严格的顺序刷新；任一服务不存在或未达到 `RUNNING` 时命令会失败，不会
把失败服务写入 bundle。bundle 更新带进程锁，多个独立部署并发完成时不会相互覆盖。

删除服务时只会在 aggregate URL 与该 service name 自己的 state 文件一致时删除 bundle
条目。删除旧 service 不会清掉同类型新 service 的 URL。

## 4. 状态、停止和删除

状态和等待：

```bash
"$SERVING/manage.py" status all --config "$SERVING/config.local.toml"
"$SERVING/manage.py" wait sim --config "$SERVING/config.local.toml"
```

停止只影响选中的服务：

```bash
"$SERVING/manage.py" stop sim --config "$SERVING/config.local.toml"
```

删除必须显式传入配置中的完整 service name：

```bash
"$SERVING/manage.py" delete sim \
  --config "$SERVING/config.local.toml" \
  --confirm-delete <SIM_SERVICE_NAME>
```

运行中的服务会先 stop 并等待 `STOPPED`，然后 delete 并等待 `ABSENT`。只有确认平台对象
已经不存在后，才清理对应 endpoint state。

## 5. 资源语义

每套服务的 `quota` 描述一个 Inspire replica 内的 GPU、CPU 和内存，`replicas` 描述
平台 replica 数；总卡数为 `quota.gpus × replicas`。

| 服务 | 一个平台 replica 内部 |
| --- | --- |
| ASR | 一个 vLLM 服务，TP=1、DP=`quota.gpus` |
| SIM | 一个 Python 进程、一个共享 reference cache、每卡一个 Torch 模型 |
| Judge | 一个 vLLM 服务，TP=1、DP=`quota.gpus` |

GPU 数只接受 `1`、`4`、`8`。SIM 客户端继续通过 reference 路径散列设置请求亲和 header，
让相同 reference 的请求落到同一个平台 replica，复用该 replica 的 cache。

ASR/Judge 直接使用镜像内的 `vllm` 和配置中的 model path，不依赖额外 RM 源码仓库。
SIM 使用 Slime 内的 `slime/serving/tts_sim`。提交时记录实际源码内容的 SHA256 和 launcher blob，
启动时重新检查，支持未提交的开发文件；提交后改动这些文件会使启动校验失败。

## 6. 训练接入

Serving bundle 和 secret 就绪后，显式传给训练 submitter：

```bash
export ENDPOINT_BUNDLE=/inspire/你的目录/endpoints.env
export REWARD_SECRET_ENV=/inspire/你的目录/reward-secret.env
export REWARD_CONFIG="$PWD/examples/tts_grpo/reward_wer_sim.yaml"
bash examples/tts_grpo/submit_inspire.sh --dry-run
bash examples/tts_grpo/submit_inspire.sh --submit
```

Submitter 只要求 bundle 包含实际 reward 配置引用的 endpoint。例如 SIM-only 配置只要求
`TTS_SIM_URL`。提交前检查数据与配置；实际请求由运行时进行有限重试与协议校验。部署状态和 endpoint 是否可用可先用本管理器的 wait/status 命令核实。

## 7. 高级运行参数

正常使用不需要设置以下变量；它们主要用于状态目录覆盖、受控重试和测试：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `SERVING_STATE_DIR` | `serving/.state` | 单服务 endpoint state 目录 |
| `SERVING_ENDPOINT_BUNDLE` | `examples/tts_grpo/endpoints.env` | aggregate bundle；空字符串关闭 |
| `SERVING_INSPIRE_CWD` | `/tmp` | Inspire CLI 的隔离工作目录 |
| `SERVING_READONLY_ATTEMPTS` | `3` | 只读操作最大尝试次数 |
| `SERVING_RETRY_DELAY_SECONDS` | `2` | 指数退避初始秒数 |
| `SERVING_POLL_SECONDS` | `10` | 状态轮询秒数 |
| `SERVING_WAIT_TIMEOUT_SECONDS` | `3600` | 等待 RUNNING 超时秒数 |
| `SERVING_STOP_TIMEOUT_SECONDS` | `600` | 等待 STOPPED 超时秒数 |
| `SERVING_DELETE_TIMEOUT_SECONDS` | `600` | 等待 ABSENT 超时秒数 |

部署机要求 Python 3.11 或更新版本，以及 Inspire CLI 7.0.1 或更新版本。Inspire 的公开
`serving status --json` 有意隐藏 URL，所以 `endpoint.py` 只在已安装 Inspire 工具自己的
Python 环境中读取认证 detail；URL 仅进入 `0600` 文件，不输出到终端日志。

正式发布前仍需在目标 GPU、镜像和模型上记录显存、启动时间、batch 形状、QPS、p95/p99、
队列等待和错误率。CPU 测试只验证编排和配置契约，不能替代 Serving 实测。
