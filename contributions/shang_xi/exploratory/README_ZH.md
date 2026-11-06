# Shang Xi：验证集驱动的 IMP 停止规则（探索性追加分析）

本包在 2026-11-05 使用小组原有 **350 份 clean IMP 日志**完成，没有重新训练，也没有修改共享 Overleaf 或 GitHub。本工作由 AI 协助编写和核验；若进入论文，应如实写明数据由小组原实验生成、此项为 Shang Xi 的独立日志分析。

## 研究问题及规则

能否仅依据已经完成的轮次的验证集准确率，提前停止 IMP 搜索，并减少实际执行前缀的优化更新数？

- 每个数据集/种子各自从 round 0 开始；验证准确率**严格提高**时将未改善计数清零，持平也视为未改善。
- 主规则 `patience_3`：最早从第 3 轮开始，若连续 3 轮没有验证准确率新高就停止；如未触发，最多运行到第 15 轮。敏感性规则为 `patience_2` 与 `patience_4`。
- 在实际停止轮数以内，再按验证准确率选最佳候选；并列时选更早轮次。**停止和选择都不看测试集**，测试集仅用于最终评价。
- 与完整 15 轮及固定 5 轮前缀比较。执行费用按实际停止轮数核算 $C(b)=T+b(T-k)$，包括寻找 mask 所需的先前训练。该费用排除 controls、超参搜索、数据处理和测评。
- 数据集先平均五种子，再以 35 个数据集为独立重采样单位做 10,000 次逐点 percentile bootstrap；种子固定为 20261105。

注意：先前已经看过固定预算的汇总数据，这套停止规则是**事后提出的探索性规则**，不能表述为预注册、在线部署实测或验证了某个优越性假设。准确率是离散值，验证集的平台期可能过早触发停止。

## 可核查结果

以下差值均为该策略减去完整 15 轮策略的平均测试准确率，单位百分点。区间为逐点 95% bootstrap 区间。平均费用单位为一次 dense 训练的优化更新数 $T$。

| 设置 | 策略 | 平均费用 $/T$ | 相对完整预算节省更新数 | 测试准确率差 [95% 区间] | 平均停止轮 |
| --- | --- | ---: | ---: | ---: | ---: |
| Tuned | 固定 5 轮 | 5.9009 | 62.42% | −0.2458 [−0.4449, −0.0835] | 5.00 |
| Tuned | 连续 3 轮无改善 | 5.5647 | 64.56% | −0.2198 [−0.4151, −0.0502] | 4.66 |
| Fixed | 固定 5 轮 | 5.9000 | 62.42% | −0.0976 [−0.2130, −0.0028] | 5.00 |
| Fixed | 连续 3 轮无改善 | 5.7600 | 63.31% | −0.1266 [−0.2477, −0.0308] | 4.86 |

同一个固定 5 轮数据集均值与先前分析一致；上述区间端点略有变化，是此脚本按自己的固定随机种子重采样。与**固定 5 轮**直接配对比较时，tuned 主规则的准确率差为 +0.0260 pp，95% 区间 [−0.0003, +0.0565]；更新费用差为 −0.3362T，区间 [−0.7169T, +0.1119T]。fixed 主规则的准确率差为 −0.0291 pp，区间 [−0.0787, +0.0146]；费用差 −0.1400T，区间 [−0.4984T, +0.2296T]。这些区间跨过零，**不能宣布自适应规则优于固定 5 轮**，也不能说两者等价。敏感性规则的完整输出在 CSV 中。

主规则有 40.6% tuned 与 37.1% fixed 的运行最终选回未剪枝 dense round 0。它是搜索/选择策略的结果，不可把所有结果称为 winning sparse ticket；不同策略选中的平均密度也不同。

## 复现与文件

| 文件 | 内容 |
| --- | --- |
| `analysis/adaptive_stop.py` | 验证集停止规则、严格原始运行核验、配对及 bootstrap |
| `analysis/plot_adaptive.py` | 由摘要 CSV 画黑白图 |
| `results/adaptive_seed.csv` | 每运行、每规则的 1,400 行回放 |
| `results/adaptive_dataset.csv` | 每数据集、每规则的 280 行聚合 |
| `results/adaptive_summary.csv` | 8 行总体结果 |
| `results/paired_vs_fixed5.csv` | 6 种规则/设置组合、两项指标的配对比较 |
| `results/adaptive_scatter.pdf` | 策略的更新费用与性能差；误差线是准确率区间，不是费用区间 |
| `results/manifest.json` | 原始 350 文件的 SHA-256 与预先写入代码的规则参数 |

将脚本加入原有**独立分析包**（不覆盖小组项目文件），从该包根目录运行：

```bash
python3 -m analysis.adaptive_stop \
  --results-root /path/to/350_original_runs \
  --out results/adaptive_shang_new
python3 -m analysis.plot_adaptive \
  results/adaptive_shang_new/adaptive_summary.csv \
  results/adaptive_shang_new/adaptive_scatter.pdf
```

需要 Python、NumPy、PyYAML、matplotlib；数据读取不需要 Torch、OpenML、UCloud、GPU。分析脚本拒绝已有输出目录和缺失/无效运行。与原预算分析的 350 个输入哈希完全一致；固定 5 轮的 350 行选择、费用和测试值逐一与旧分析核对；另以人为构造序列检查停止规则不读取测试准确率。

## 论文位置与限制

这张图及完整表适合补充材料或 GitHub 结果页。第三次作业主文四页紧张，且队友还要加入实测时间图，因此**先不要把本图再塞入主文**。可以在现有 Q4 的 Discussion 增添一句由小组审核后的探索性说明，但在没有版面时保留为代码和附属结果也有可核查贡献。

Suggested short sentence (review before use): "In a retrospective validation-only stopping replay, a three-round patience rule used 5.56 rather than 15.70 dense-run update equivalents for tuned MLPs, with a mean accuracy difference of -0.220 percentage points; it did not establish an advantage over a fixed five-round budget."

这项分析不解决 one-shot 同密度对照，也没有实际硬件时间、独立数据 split 或超参数搜索费用。不能把不同策略的结果解释为同密度 pruning 方法的因果差异。
