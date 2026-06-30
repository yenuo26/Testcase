# vLLM KVC Benchmark 测试

基于 pytest 的 vLLM 长上下文 serving benchmark 脚本，用于对比 **KVC hot-score 缓存策略** 与 **Baseline** 在不同上下文长度下的性能。

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.8+ | 运行 pytest |
| pytest | 测试框架 |
| requests | 健康检查 |
| vLLM | 需支持 `--cache-policy`、`--sparse-topk`、`--copy-method` 等 KVC 参数 |
| GPU | 默认 `tensor_parallel=4`，需 4 卡 |
| Linux | 脚本使用 `pkill`、`os.setsid`，建议在 Linux 容器/服务器中运行 |

### 安装 Python 依赖

```bash
pip install pytest requests
```

### 目录结构（默认路径）

```
/workspace/models/Qwen2.5-7B-Instruct-1M/   # 模型
/vllm-workspace/vllm/                        # vLLM 源码（含 benchmarks/benchmark_serving.py）
/vllm-workspace/LongBench/data.json          # 自定义 JSONL 数据集
```

## 环境变量

所有路径与服务参数均可通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_DIR` | `/workspace/models` | 模型根目录 |
| `MODEL_NAME` | `Qwen2.5-7B-Instruct-1M` | 模型子目录名 |
| `WORK_DIR` | `/vllm-workspace` | 工作根目录 |
| `BENCHMARK_DIR` | `$WORK_DIR/vllm` | vLLM benchmark 目录 |
| `DATASET_DIR` | `$WORK_DIR/LongBench` | 数据集目录 |
| `DATASET_PATH` | `$DATASET_DIR/data.json` | 数据集文件 |
| `PORT` | `8001` | vLLM 服务端口 |
| `BASE_URL` | `http://127.0.0.1:$PORT` | benchmark 请求地址 |
| `VLLM_LOG` | `$WORK_DIR/vllm_run.log` | vLLM 服务日志 |
| `BENCHMARK_LOG` | `$WORK_DIR/run_benchmark.log` | benchmark 运行日志 |
| `PRED_LOG` | `$WORK_DIR/run_pred.log` | pred.py 运行日志 |
| `SERVICE_TIMEOUT` | `600` | 等待服务就绪超时（秒） |
| `HEALTH_CHECK_INTERVAL` | `2` | 健康检查间隔（秒） |
| `AUTO_CLEANUP` | `true` | 测试结束后是否自动停止 vLLM 进程 |

示例：

```bash
export MODEL_DIR=/data/models
export PORT=8002
export AUTO_CLEANUP=false   # 测试后保留 vLLM 进程便于调试
```

## 测试用例

测试函数名须与 `TEST_CONFIGS` 中的 key **完全一致**，fixture 才会加载对应配置。

| 测试函数 | 说明 | vLLM 特点 | 输入长度 |
|----------|------|-----------|----------|
| `test_kvc_hot_perf_1M` | KVC hot-score，100 万上下文 | `--cache-policy hot-score --sparse-topk 2048` | 700k ~ 1M |
| `test_kvc_hot_perf_100k_2048` | KVC hot-score，10 万上下文 | `sparse_topk=2048` | 32k ~ 100k |
| `test_kvc_hot_perf_100k_4096` | KVC hot-score，10 万上下文 | `sparse_topk=4096` | 32k ~ 100k |
| `test_kvc_baseline_perf_1M` | Baseline，100 万上下文 | 不启用 cache-policy | 700k ~ 1M |
| `test_kvc_baseline_perf_100k` | Baseline，10 万上下文 | 不启用 cache-policy | 32k ~ 100k |
| `test_kvc_baseline_acc` | Baseline，LongBench 精度评估 | 不启用 cache-policy | 运行 `pred.py` |
| `test_kvc_hot_acc_2048` | KVC hot-score 精度评估 | `sparse_topk=2048` | 运行 `pred.py` |
| `test_kvc_hot_acc_4096` | KVC hot-score 精度评估 | `sparse_topk=4096` | 运行 `pred.py` |

各用例的 vLLM / benchmark / pred 参数定义在 `test_kvc.py` 中的 `TEST_CONFIGS` 字典，修改配置请编辑该字典并保持 key 与测试函数名一致。

Pred 类用例在服务就绪后执行：

```bash
python {DATASET_DIR}/pred.py --model {MODEL_NAME} --n_proc 3
```

## 使用方法

### 运行全部测试

```bash
pytest test_kvc.py -v
```

### 运行单个测试

```bash
# KVC 100k，topk=2048
pytest test_kvc.py::test_kvc_hot_perf_100k_2048 -v

# KVC 100k，topk=4096
pytest test_kvc.py::test_kvc_hot_perf_100k_4096 -v

# KVC 1M（耗时长、显存/内存需求高）
pytest test_kvc.py::test_kvc_hot_perf_1M -v

# Baseline 对比
pytest test_kvc.py::test_kvc_baseline_perf_100k -v
pytest test_kvc.py::test_kvc_baseline_perf_1M -v

# LongBench pred 精度评估
pytest test_kvc.py::test_kvc_baseline_acc -v
pytest test_kvc.py::test_kvc_hot_acc_2048 -v
pytest test_kvc.py::test_kvc_hot_acc_4096 -v
```

### 查看实时输出

```bash
pytest test_kvc.py -v -s
```

`-s` 可显示 benchmark 进程的 stdout。

## 执行流程

1. 检查模型、b benchmark 脚本、数据集路径是否存在
2. 按 `TEST_CONFIGS` 启动 `vllm serve`
3. 轮询 `/health` 或日志中的 `Uvicorn running on`，等待服务就绪
4. 在 `BENCHMARK_DIR` 下执行 `benchmarks/benchmark_serving.py`
5. 断言退出码为 0，并检查 `benchmark_results.json`
6. 若 `AUTO_CLEANUP=true`，通过 `pkill` 停止 vLLM 进程

## 输出文件

| 文件 | 位置 |
|------|------|
| vLLM 服务日志 | `$VLLM_LOG`（默认 `/vllm-workspace/vllm_run.log`） |
| Benchmark 日志 | `$BENCHMARK_LOG`（默认 `/vllm-workspace/run_benchmark.log`） |
| Pred 日志 | `$PRED_LOG`（默认 `/vllm-workspace/run_pred.log`） |
| Benchmark 结果 | `$BENCHMARK_DIR/benchmark_results.json` |

## 常见问题

**服务启动超时**

查看 `$VLLM_LOG` 最后 30 行；可适当增大 `SERVICE_TIMEOUT`。

**Benchmark 连接失败**

确认 `BASE_URL` 使用 `127.0.0.1` 而非 `0.0.0.0`（客户端无法连接 `0.0.0.0`）。

**Baseline 测试误带 `--sparse-topk`**

已修复：仅当 `sparse_topk` 显式配置时才会追加该参数。

**Windows 本地无法运行**

清理逻辑依赖 `pkill`，进程组管理依赖 `setsid`；请在 Linux 环境或 Docker 中执行。
