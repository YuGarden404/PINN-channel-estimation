# DeepMIMO 多基站 RSS Baseline 阶段总结

## 实验目标

本阶段实验的目标是验证：在 DeepMIMO 场景下，多基站 RSS 指纹是否能够作为额外先验，提升神经网络信道估计性能，尤其是在 LS-like 信道观测较弱的低 SNR 条件下。

第一阶段复现实验的主要发现是：在原始 Boston/Wireless-InSite 流程中，正常 RSS 相比 zero RSS 或 constant RSS 没有带来稳定的 NMSE 增益。因此第二阶段不再只讨论单基站 RSS，而是进一步测试多基站 RSS 是否能够提供更强的位置/环境相关信息。

本阶段核心问题可以概括为：

> 当 LS-like 信道输入质量下降时，多基站 RSS 指纹是否能够帮助模型恢复更准确的目标基站信道？

## 数据与实验设置

- 场景：`o1_60`
- 用户数量：`5000`
- 目标基站：`TX 10`
- 目标 pair index：`0`
- 多基站 RSS 来源：`TX 10 / TX 11 / TX 12`
- 多基站 RSS 使用同一 RX set 下的用户样本
- 目标信道张量形状：`[5000, 1, 8, 1]`
- 展平后的实数信道维度：`16`
- LS-like 输入：目标信道叠加复高斯白噪声
- 测试 SNR：`0 dB`、`-5 dB`、`-10 dB`
- 训练轮数：`100 epochs`
- batch size：`256`
- 随机种子：`42`

需要注意：当前 LS-like 输入是通过给目标信道加入复 AWGN 得到的简化代理输入，并不是完整的 LS-OFDM 估计器。因此本阶段结果应被理解为 RSS 融合 baseline，而不是最终通信系统级 benchmark。

## 对比方法

本阶段共比较五种输入模式：

- `ls_only`：只使用 noisy LS-like 信道输入。
- `single_rss`：使用 LS-like 输入和目标基站单个 RSS。
- `multibs_rss`：使用 LS-like 输入和 3 个基站的 RSS 指纹。
- `zero_rss`：使用 LS-like 输入和全零 RSS 向量。
- `shuffled_rss`：使用 LS-like 输入和样本顺序打乱后的多基站 RSS。

其中 `zero_rss` 和 `shuffled_rss` 是关键对照组。如果真实多基站 RSS 被模型有效利用，那么 `multibs_rss` 应该明显优于这两个对照组，尤其应该在低 SNR 下体现优势。

## 实验结果

### Test NMSE

| SNR | `ls_only` | `single_rss` | `multibs_rss` | `zero_rss` | `shuffled_rss` |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.418038 | 0.432860 | 0.419044 | **0.410742** | 0.434593 |
| -5 dB | 0.754223 | 0.793107 | 0.794015 | **0.734087** | 0.774373 |
| -10 dB | **1.070395** | 1.105507 | 1.133112 | 1.112689 | 1.113992 |

### Test NMSE in dB

| SNR | `ls_only` | `single_rss` | `multibs_rss` | `zero_rss` | `shuffled_rss` |
|---:|---:|---:|---:|---:|---:|
| 0 dB | -3.79 | -3.64 | -3.78 | **-3.86** | -3.62 |
| -5 dB | -1.22 | -1.01 | -1.00 | **-1.34** | -1.11 |
| -10 dB | **0.30** | 0.44 | 0.54 | 0.46 | 0.47 |

## 主要观察

1. `multibs_rss` 没有稳定优于 `ls_only`。

   在 `0 dB` 下，`multibs_rss` 与 `ls_only` 基本持平；在 `-5 dB` 和 `-10 dB` 下，`multibs_rss` 反而劣于 `ls_only`。这说明当前多基站 RSS 融合方式没有带来稳定的信道估计增益。

2. 降低 LS-like 输入 SNR 后，多基站 RSS 优势仍未显现。

   原本的假设是：当 LS-like 输入变弱时，RSS 这种低维环境/位置先验可能会变得更有帮助。但从 `0/-5/-10 dB` 三组结果来看，当前模型并没有表现出这种趋势。

3. `zero_rss` 表现异常强。

   `zero_rss` 在 `0 dB` 和 `-5 dB` 下都是最优，在 `-10 dB` 下也与其他 RSS 模式接近。这说明当前网络主要仍在做 LS-like 输入的黑盒去噪或残差修正，而不是稳定利用 RSS 分支中的有效信息。

4. `shuffled_rss` 没有稳定显著差于真实 `multibs_rss`。

   如果模型真正利用了 RSS 与用户样本之间的对应关系，那么打乱样本顺序后的 `shuffled_rss` 应明显退化。但当前实验中，`shuffled_rss` 并没有稳定显著差于 `multibs_rss`，这进一步说明 RSS 样本关联没有被当前模型有效学习。

5. 当前简单拼接式 MLP 融合结构很可能不足。

   这个负结果不能直接说明 RSS 本身没有价值。更准确的解释是：当前使用的轻量 MLP 和 feature concatenation 融合方式不足以把多基站 RSS 指纹转化为稳定的信道估计增益。

## 阶段性结论

在当前 DeepMIMO 设置下，naive multi-BS RSS fusion 没有带来稳定的 NMSE 改善。这个结论在 `0 dB`、`-5 dB` 和 `-10 dB` 三种 LS-like 输入质量下都成立。

因此，本阶段不能得出“多基站 RSS 能够提升信道估计性能”的正向结论。更稳妥的表述是：

> 多基站 RSS 指纹可能包含与用户位置和传播环境相关的信息，但当前轻量拼接式 MLP 没有有效地将这些信息转化为信道估计增益。

这与第一阶段 ablation 的结论一致：在当前模型框架中，RSS 相关输入还没有形成稳定、可解释、可复现的性能贡献。

## 下一步建议

不建议继续在同一个 naive MLP 融合结构上单纯增加更多 SNR 点。当前 `0/-5/-10 dB` 三组结果已经足够说明：简单拼接 RSS 不足以带来稳定收益。

下一步更应该修改模型结构，重点测试更强的 RSS 条件化融合方式，例如：

- RSS-guided residual refinement：让 RSS 预测 LS 残差的修正方向或调制因子，而不是简单拼接输入。
- FiLM / gating fusion：用 RSS 生成隐藏特征的 scale 和 bias，对信道特征进行条件调制。
- Attention fusion：将 RSS 或位置特征作为上下文，对 LS-like 信道特征进行加权修正。
- 显式加入 UE position：比较位置输入、RSS 输入和多基站 RSS 输入之间的差异。
- 增加信道维度和任务复杂度：当前目标信道只有 `1 x 8 x 1` 个复数维度，后续可在融合机制有效后扩展到更丰富的天线/子载波设置。

## 复现实验命令

`SNR = 0 dB`：

```powershell
python deepmimo_multibs/run_five_baselines.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel `
  --epochs 100 `
  --batch-size 256
```

`SNR = -5 dB`：

```powershell
python deepmimo_multibs/run_five_baselines.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-5 `
  --epochs 100 `
  --batch-size 256
```

`SNR = -10 dB`：

```powershell
python deepmimo_multibs/run_five_baselines.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10 `
  --epochs 100 `
  --batch-size 256
```

结果文件：

- `deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel/logs/summary_ep100.csv`
- `deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-5/logs/summary_ep100.csv`
- `deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/logs/summary_ep100.csv`
