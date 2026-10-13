# 简化 RCST

方法只有两步：

1. 用通过审核、有正迁移分数且观察到奖励差异的子题，与无提示原题混合训练。
2. 最后只训练无提示原题，检验是否真正学会。

移除 guided-root 阶段和 Bridge 奖励。默认前 40 步每批 28 个原题、4 个子题，后 10 步每批 32 个原题；这些只是验证起点，不是调优后的最佳参数。子题单独作答，原题输入不附加子题、答案或计划。

**状态：实验性实现，未运行新训练，未证明优于 Vanilla。** 旧 1866 题实验的 RCST/Vanilla 为 37.96/37.60，独立 1000 题实验为 35.07/38.32；二者数据池和训练条件不同，不能拼成稳定提升。

## 数据准备

在已有服务器环境中执行。需要 pandas/pyarrow；训练沿用项目已有 Scaf-GRPO 环境。

- `source.parquet`：答案经过审核的源题目，保留标准 system + user prompt。
- `selected.jsonl`：`aggregate_rcst_probes.py --policy lcb_positive` 的输出，每个根题一条，包含拒绝项。污染候选须隔离或重生成，并重新测量；不能只改 ID 或旧分数。
- `learnability.jsonl`：`probe_selected_subproblem_learnability.py` 生成的 `subproblem_learnability.jsonl`，需要 `num_samples`、`correct_samples`、`p_sub`。

先生成待审核清单（所有核验项默认为 false）：

```bash
python prepare_simple_rcst.py --source-data source.parquet \
  --candidates selected.jsonl --learnability learnability.jsonl \
  --review-template reviews.jsonl
```

逐条审核 `reviews.jsonl`：根题答案、子题及完整奖励契约、probe/可学习性是否使用修正后的数据；通过对应检查才设为 true，并填写 `evidence`。哈希绑定题面、答案和测量记录，数据变化后必须重新审核。清单不自动证明数学正确，也不能靠批量改 true 代替审核。

所有根题必须通过答案审核。未通过子题审核、上游拒绝、没有正 `gain_lcb`、少于两轮 probe 或未观察到正确/错误两种子题回答时，保留原题训练，不启用该子题。有限采样全错/全对只表示本轮缺少奖励差异，不表示真实成功率就是 0/1。

```bash
python prepare_simple_rcst.py --source-data source.parquet \
  --candidates selected.jsonl --learnability learnability.jsonl \
  --reviews reviews.jsonl --output-dir outputs/simple_rcst_data
```

输出方法/Vanilla 两份有序训练数据和 `summary.json`，包括接受/拒绝原因、实际子题槽位与文件哈希。无合格子题时，两臂数据完全相同。源题按完整题面唯一匹配，缺失或歧义直接报错；不会回退到可能重复的 ID。另有 `extract_selected_roots_with_source.py` 可按此规则重新关联生成资料，关联成功仍需数学审核。

## 训练与评测

先检查执行目标资源，再显式启动。以下命令会训练一条实验臂；设 `DRY_RUN=1` 只检查数据与参数。

```bash
DATA_DIR=/path/to/simple_rcst_data \
SCAF_REPO=/path/to/Scaf-GRPO MODEL_PATH=/path/to/Qwen2.5-Math-1.5B \
RUN_DIR=/path/to/new_method_run TRAIN_GPU=0 ARM=method \
  bash scripts/run_simple_rcst_1gpu.sh
```

Vanilla 用同一数据目录和基模型，改 `ARM=vanilla` 与新 `RUN_DIR`。训练器关闭 shuffle，按固定槽位执行两阶段；不中断、不重置优化器。长 prompt 报错，禁止静默过滤破坏阶段边界。每个样本拥有独立 ID，子题继承原题的 system prompt 和奖励分发来源。

训练结束的权重在 `RUN_DIR/checkpoints/global_step_50/actor`（默认配置）；按现有合并工具合并，再使用 `scripts/run_qwen_math_1_5b_paper_eval_2h100.sh` 做七基准无提示 greedy pass@1。该启动器不自动评测、不使用测试集选择阶段或参数，首次版本要求新运行目录。

两臂保持相同全局步数、batch 和 rollout 设置，但方法替换了一部分原题曝光；生成 token 与候选/probe 成本仍需单独报告。`--seed` 只控制数据顺序，不能当作多训练种子结果。RCST 的 LoRA 探针依然只是待验证代理，`gain_lcb` 不构成安全迁移保证。

下一轮先比较 Vanilla、简单子题选择和此版本；有稳定收益后再扩大数据。
