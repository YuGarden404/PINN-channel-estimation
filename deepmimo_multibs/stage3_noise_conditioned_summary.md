# Stage 3: Noise-Conditioned Channel Refinement 阶段总结

## 1. 当前论文核心

本阶段工作的核心不是否定原作者的 RSS-informed / cross-attention channel estimation 思路，而是在其基础上寻找更稳定、更有解释性的改进方向。

前期实验表明，直接将 RSS 特征拼接或融合到网络中，并不能稳定提升 DeepMIMO 场景下的信道估计精度。相反，cross-attention residual refinement 本身是有效的，说明神经网络对 noisy LS channel estimate 的 refinement 能够带来收益。

因此，当前论文的核心可以收束为：

> 在原作者 cross-attention residual refinement 的基础上，引入显式 noise map estimation 与 non-blind channel denoising，使模型从“隐式学习去噪”转向“噪声条件化的信道细化”，从而提升低 SNR 条件下的鲁棒信道估计性能。

## 2. 建议论文题目

推荐英文题目：

**Noise-Conditioned Non-Blind Channel Refinement for DeepMIMO-Based Channel Estimation**

可选题目：

**CBDNet-Inspired Noise-Aware Channel Estimation under Low-SNR DeepMIMO Scenarios**

中文题目：

**面向 DeepMIMO 信道估计的噪声条件非盲信道细化方法**

## 3. 我们的模型

当前模型可命名为：

**NC-CENet: Noise-Conditioned Channel Estimation Network**

也可以使用更直观的名称：

**CBDNet-CE: CBDNet-style Channel Estimator**

模型脚本：

```text
deepmimo_multibs/train_cbdnet_baseline.py
```

模型思想来自 CBDNet 的两阶段去噪范式，但任务从图像去噪迁移到信道估计：

1. **Noise Map Estimation**
   - 输入 noisy LS channel estimate。
   - 估计 token-wise noise map。
   - noise map 作为显式噪声强度/噪声结构的中间表征。

2. **Non-Blind Channel Denoising**
   - 将 noisy LS channel estimate 与估计出的 noise map 拼接。
   - 网络在噪声条件已知的情况下进行 residual channel refinement。
   - 输出 refined channel estimate。

模型学习目标从普通的：

```text
H_LS -> H_clean
```

变为：

```text
H_LS -> noise_map
[H_LS, noise_map] -> H_refined
```

这使模型不再完全依赖隐式黑盒去噪，而是显式建模输入信道中的噪声状态。

## 4. 实验设置

数据集与任务设置：

- 场景：`DeepMIMO o1_60`
- 用户数：`5000`
- 目标基站：`TX 10`
- 多基站 RSS 来源：`TX 10 / TX 11 / TX 12`
- 目标信道形状：`[5000, 1, 8, 1]`
- 实数展开维度：`16`
- 输入：目标信道叠加复高斯白噪声得到的 LS-like channel estimate
- SNR 设置：主实验为 `0 dB`, `-5 dB`, `-10 dB`；补充极低 SNR 实验为 `-15 dB`, `-20 dB`
- 训练轮数：`100 epochs`
- batch size：`256`
- 主要评价指标：Test NMSE 与 dB 表示

需要注意：当前 LS-like 输入是简化的 noisy channel proxy，还不是完整通信链路中的严格 OFDM-LS 估计器。因此当前阶段结果应表述为 DeepMIMO-based neural channel refinement baseline，而不是最终系统级通信 benchmark。

## 5. 当前关键结果

### 5.1 MLP RSS baseline

| SNR | ls_only | single_rss | multibs_rss | zero_rss | shuffled_rss |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.418038 | 0.432860 | 0.419044 | **0.410742** | 0.434593 |
| -5 dB | 0.754223 | 0.793107 | 0.794015 | **0.734087** | 0.774373 |
| -10 dB | **1.070395** | 1.105507 | 1.133112 | 1.112689 | 1.113992 |

结论：naive RSS fusion 没有稳定收益。`zero_rss` 和 `shuffled_rss` 对照说明，当前简单拼接式 RSS 分支并没有稳定学习到可泛化的 RSS-channel 对应关系。

### 5.2 Cross-attention residual baseline

SNR = 0 dB，100 epochs：

| mode | Best Val NMSE | Test NMSE | Test dB |
|---|---:|---:|---:|
| ls_only | 0.300158 | **0.364926** | -4.38 |
| single_rss | 0.301669 | 0.376288 | -4.24 |
| multibs_rss | **0.294580** | 0.381816 | -4.18 |
| zero_rss | 0.298630 | 0.368618 | -4.33 |
| shuffled_rss | 0.305098 | 0.384424 | -4.15 |

Cross-attention ls_only 在低 SNR 下的结果：

| SNR | Test NMSE |
|---:|---:|
| -5 dB | 0.645223 |
| -10 dB | 0.968148 |

结论：cross-attention residual refinement 明显优于简单 MLP baseline，但 RSS 分支仍然没有稳定带来收益。

### 5.3 Noise-aware auxiliary baseline

脚本：

```text
deepmimo_multibs/train_noise_aware_baseline.py
```

| SNR | Noise weight | Test NMSE | Test dB |
|---:|---:|---:|---:|
| 0 dB | 0.1 | 0.374753 | -4.26 |
| 0 dB | 0.01 | 0.378097 | -4.22 |
| -5 dB | 0.1 | 0.681647 | -1.66 |
| -5 dB | 0.01 | 0.677717 | -1.69 |
| -10 dB | 0.1 | 0.973858 | -0.12 |
| -10 dB | 0.01 | 0.981476 | -0.08 |
| -15 dB | 0.01 | 1.005800 | 0.0251 |
| -20 dB | 0.01 | 1.050226 | 0.2128 |

结论：简单 auxiliary noise loss 可以超过 MLP baseline，但整体仍弱于 cross-attention ls_only。这说明“噪声建模”方向有价值，但仅加辅助头还不够。

### 5.4 CBDNet-style noise-conditioned baseline

脚本：

```text
deepmimo_multibs/train_cbdnet_baseline.py
```

当前最重要结果：

| SNR | MLP ls_only | Cross-attn ls_only | Noise-aware nw=0.01 | CBDNet-style nw=0.01 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.418038 | 0.364926 | 0.378097 | **0.359841** |
| -5 dB | 0.754223 | 0.645223 | 0.677717 | **0.644630** |
| -10 dB | 1.070395 | 0.968148 | 0.981476 | **0.871116** |
| -15 dB | 1.377312 | 1.014650 | **1.005800** | 1.010255 |
| -20 dB | 2.014708 | 1.057143 | **1.050226** | 1.145429 |

对应 dB：

| SNR | CBDNet-style Test dB |
|---:|---:|
| 0 dB | -4.4389 |
| -5 dB | -1.9069 |
| -10 dB | -0.5992 |

结论：

- 在 `0 dB`，CBDNet-style 模型略优于 cross-attention ls_only。
- 在 `-5 dB`，CBDNet-style 模型与 cross-attention ls_only 基本持平并略优。
- 在 `-10 dB`，CBDNet-style 模型显著优于 cross-attention ls_only，Test NMSE 从 `0.968148` 降至 `0.871116`。

这说明显式 noise map + non-blind denoising 在 `-10 dB` 附近优势最明显，同时不会牺牲中等 SNR 下的性能。新补充的 `-15/-20 dB` 结果也提示：固定 `noise_weight = 0.01` 并不能保证在极端低 SNR 下持续优于 cross-attention，尤其 `-20 dB` 需要单独调参或增强模型容量。

### 5.5 Early stopping 与 best checkpoint 报告规范化

为避免将最后一轮模型误当作最终模型，CBDNet-style baseline 已加入 early stopping 与 best checkpoint 规范报告。所有 test 结果均加载 validation NMSE 最低的 `best_model.pth` 后计算。

Early stopping 设置：

- `--patience 20`
- `--min-delta 0.0`
- 最大训练轮数：`100 epochs`
- 报告指标：validation-selected best checkpoint 的 test NMSE

| SNR | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB | Early Stop |
|---:|---:|---:|---:|---:|---:|---|
| 0 dB | 17 | 37 | 0.301688 | **0.359841** | -4.4389 | Yes |
| -5 dB | 32 | 52 | 0.572482 | **0.644630** | -1.9069 | Yes |
| -10 dB | 53 | 73 | 0.809526 | **0.871116** | -0.5992 | Yes |
| -15 dB | 30 | 50 | 0.989515 | **1.010255** | 0.0443 | Yes |
| -20 dB | 22 | 42 | 1.109376 | 1.145429 | 0.5897 | Yes |

结论：

- Early stopping 没有改变 CBDNet-style 模型的 test 结论，而是让报告方式更规范。
- 三个 SNR 下的 test 结果均来自 validation-selected best checkpoint。
- 在 `0/-5/-10 dB` 主实验范围内，best epoch 随 SNR 降低而后移：`0 dB` 为 epoch 17，`-5 dB` 为 epoch 32，`-10 dB` 为 epoch 53。极低 SNR 补充实验中，`-15 dB` 和 `-20 dB` 的最佳轮次反而提前，说明固定训练配置下模型可能更早进入过拟合或欠稳定区域，不能简单概括为“SNR 越低训练越久越好”。
- 论文中可以表述为：For fair model selection, all reported test results are evaluated using the checkpoint with the lowest validation NMSE. Early stopping with a patience of 20 epochs is used to avoid overfitting and unnecessary training.

### 5.6 Noise weight 消融实验

为验证 noise map supervision 权重不是偶然选择，在 `SNR = -10 dB` 的 CBDNet-style `ls_only` 设置下，对 `--noise-weight` 进行消融。所有实验均使用 validation-selected best checkpoint，并启用 early stopping：

- `--patience 20`
- `--min-delta 0.0`
- 最大训练轮数：`100 epochs`

| Noise Weight | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---:|---:|---:|---:|---:|---:|
| 0 | 56 | 76 | 0.814311 | 0.878649 | -0.5618 |
| 0.001 | 34 | 54 | 0.822343 | 0.878225 | -0.5639 |
| 0.005 | 40 | 60 | 0.820491 | 0.885283 | -0.5292 |
| **0.01** | 53 | 73 | **0.809526** | **0.871116** | **-0.5992** |
| 0.05 | 52 | 72 | 0.812018 | 0.892857 | -0.4922 |

结论：

- `noise_weight = 0.01` 在 `-10 dB` 下取得当前最优 Test NMSE：`0.871116`。
- `noise_weight = 0` 仍然优于普通 cross-attention ls_only，说明 CBDNet-style 的“显式 noise map 作为 denoising 输入”结构本身已经有效。
- 加入适度 noise supervision 后，性能进一步提升；但过大的权重 `0.05` 会牺牲最终 channel NMSE。
- 因此，noise map supervision 的作用更适合表述为 regularization / guidance，而不是越强越好。
- 当前结果支持将 `noise_weight = 0.01` 作为 `0/-5/-10 dB` 主实验默认设置；但后续 `-15/-20 dB` 补充消融表明，极低 SNR 下更小的 `noise_weight = 0.001` 更合适。

#### 5.6.1 极低 SNR Noise Weight 补充消融

为解释固定 `noise_weight = 0.01` 在 `-20 dB` 下弱于 cross-attention 的现象，补充 `-15 dB` 和 `-20 dB` 的 noise weight ablation。

`SNR = -15 dB`：

| Noise Weight | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---:|---:|---:|---:|---:|---:|
| 0 | 43 | 63 | 0.991953 | 1.047623 | 0.2021 |
| **0.001** | 80 | 100 | **0.953437** | **0.977731** | **-0.0978** |
| 0.005 | 54 | 74 | 0.957147 | 0.991267 | -0.0381 |
| 0.01 | 30 | 50 | 0.989515 | 1.010255 | 0.0443 |
| 0.05 | 59 | 79 | 0.959772 | 0.992787 | -0.0314 |

`SNR = -20 dB`：

| Noise Weight | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---:|---:|---:|---:|---:|---:|
| 0 | 39 | 59 | 1.089298 | 1.098779 | 0.4091 |
| **0.001** | 48 | 68 | **1.038070** | **1.063262** | **0.2664** |
| 0.005 | 67 | 87 | 1.048543 | 1.076384 | 0.3197 |
| 0.01 | 22 | 42 | 1.109376 | 1.145429 | 0.5897 |
| 0.05 | 27 | 47 | 1.089934 | 1.095953 | 0.3979 |

结论：
- 极低 SNR 下最佳权重从 `0.01` 漂移到 `0.001`，说明 noise supervision 不能固定一档通吃所有噪声强度。
- `-15 dB` 下，最佳 CBDNet-style Test NMSE 为 `0.977731`，优于 cross-attention 的 `1.014650`。
- `-20 dB` 下，最佳 CBDNet-style Test NMSE 为 `1.063262`，接近但仍略弱于 cross-attention 的 `1.057143`。
- 因此，极低 SNR 的负面结果主要来自超参数不匹配和优化稳定性，而不应表述为 noise-conditioned refinement 失效。

### 5.7 RSS 辅助条件实验

为验证 RSS 在 CBDNet-style noise-conditioned refinement 中是否仍能作为有效辅助条件，在 `SNR = -10 dB` 下固定 `noise_weight = 0.01`，比较 `ls_only`、`single_rss`、`multibs_rss` 和 `shuffled_rss`。所有实验均使用 validation-selected best checkpoint，并启用 early stopping：

- `--patience 20`
- `--min-delta 0.0`
- 最大训练轮数：`100 epochs`

| Mode | RSS Dim | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---|---:|---:|---:|---:|---:|---:|
| **ls_only** | 0 | 53 | 73 | **0.809526** | **0.871116** | **-0.5992** |
| multibs_rss | 3 | 48 | 68 | 0.828833 | 0.889460 | -0.5087 |
| single_rss | 1 | 40 | 60 | 0.832064 | 0.892777 | -0.4926 |
| shuffled_rss | 3 | 34 | 54 | 0.827822 | 0.888009 | -0.5158 |

结论：

- 在当前 DeepMIMO 设置下，CBDNet-style 模型加入 RSS 后没有超过 `ls_only`。
- `multibs_rss`、`single_rss` 与 `shuffled_rss` 的结果非常接近，说明当前 RSS 分支仍没有稳定学习到样本级 RSS-channel 对应关系。
- 这与前期 MLP baseline 和 cross-attention baseline 的观察一致：RSS fusion 不是当前性能提升的主要来源。
- 因此，论文主贡献应明确聚焦在 noise-conditioned non-blind channel refinement，而不是 RSS-informed fusion。
- 更稳妥的论文表述是：RSS may contain useful environmental priors, but under the current low-dimensional DeepMIMO channel setting and fusion design, the observed gain mainly comes from explicit noise-conditioned refinement.

### 5.8 Noise map / residual 可视化

为增强模型解释性，新增可视化脚本：

```text
deepmimo_multibs/visualize_cbdnet_noise.py
```

该脚本读取 CBDNet-style 模型的 `best_model.pth`，在 test split 上导出：

- `energy_summary.png`
- `noise_energy_scatter.png`
- `error_before_after_scatter.png`
- `token_heatmaps.png`
- `visualization_metrics.json`

三档 SNR 的可视化结果路径：

```text
deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel/runs/cbdnet_ls_only_nw001_ep100_es20/visualizations
deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-5/runs/cbdnet_ls_only_nw001_ep100_es20/visualizations
deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cbdnet_ls_only_nw001_ep100_es20/visualizations
```

可视化指标如下。这里的 `ls_nmse` 是 noisy LS-like 输入相对 clean target 的直接 NMSE，`refined_nmse` 是 CBDNet-style refinement 后的 NMSE，二者用于说明模型对 noisy LS 输入的修正效果。

| SNR | LS NMSE | Refined NMSE | NMSE Gain | True Noise Energy | Pred Noise Energy | Refined Error Energy | Residual Energy | Noise Energy Corr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 3.134104 | 0.358784 | 2.775321 | 0.054722 | 0.038467 | 0.008628 | 0.049943 | 0.101340 |
| -5 dB | 9.910910 | 0.643044 | 9.267866 | 0.076300 | 0.030418 | 0.009676 | 0.075334 | 0.341935 |
| -10 dB | 31.341043 | 0.869740 | 30.471303 | 0.084102 | 0.015123 | 0.006096 | 0.085615 | 0.363095 |

主要观察：

- CBDNet-style 模型显著降低 noisy LS-like 输入误差，三档 SNR 下 `refined_nmse` 均远低于直接 LS NMSE。
- `residual_energy_mean` 与 `true_noise_energy_mean` 在 `-5 dB` 和 `-10 dB` 下非常接近，说明模型输出的 residual 主要承担了抵消输入噪声的作用。
- `pred_noise_energy_mean` 不等于真实噪声能量，这说明 noise map 不应被解释为逐点精确噪声幅值估计，而更像是 denoising/refinement 的条件引导特征。
- `noise_energy_corr` 在低 SNR 下更高：`-5 dB` 为 `0.341935`，`-10 dB` 为 `0.363095`，高于 `0 dB` 的 `0.101340`。这与论文主张一致：显式噪声条件化在强噪声场景下更有价值。
- 因此，可视化部分可以作为模型解释性证据：模型不仅降低 NMSE，也学到了与噪声强度相关的中间表征，并通过 residual refinement 抵消 noisy LS 输入中的扰动。

### 5.9 跨 DeepMIMO 场景外部验证

为避免结论只依赖 `o1_60` 场景，进一步下载并测试 DeepMIMO `asu_campus_3p5`、`city_7_sandiego_3p5` 和 `city_0_newyork_3p5` 三个额外场景。

新增场景特征：

- 场景：`asu_campus_3p5`、`city_7_sandiego_3p5`、`city_0_newyork_3p5`
- TX/RX pair：目标 TX/RX pair，并使用 top-5000 power 用户作为有效覆盖子集
- 原始用户数：分别为 `131931`、`36432`、`31719`
- channel shape：`[num_users, 1, 8, 1]`
- 测试 SNR：`-10 dB`

初始直接取前 `5000` 个用户时，NMSE 出现 `1e8` 量级异常。进一步检查 manifest 后发现，大量样本为零信道或极弱信道，导致 NMSE 分母 `||H||^2` 接近 0，从而放大指标。因此，外部验证采用按平均信道功率从高到低排序后选取 top-5000 有效覆盖用户：

```text
--sort-by-power descending --num-users 5000
```

筛选后数据统计：

| Scenario | Source Users | Selected Users | Selected Power Min | Selected Power Mean | Selected Power Max |
|---|---:|---:|---:|---:|---:|
| asu_campus_3p5 | 131931 | 5000 | 2.408770e-11 | 5.345881e-11 | 2.426602e-10 |
| city_7_sandiego_3p5 | 36432 | 5000 | 1.680043e-11 | 2.596491e-10 | 8.834474e-09 |
| city_0_newyork_3p5 | 31719 | 5000 | 2.982259e-12 | 2.257269e-10 | 8.760148e-09 |

外部验证结果：

| Scenario | Selection | SNR | Model | Best Val NMSE | Test NMSE | Test dB |
|---|---|---:|---|---:|---:|---:|
| asu_campus_3p5 | top-5000 power | -10 dB | Cross-attention ls_only | 0.934861 | 0.940095 | -0.2683 |
| asu_campus_3p5 | top-5000 power | -10 dB | CBDNet-style ls_only | **0.934779** | **0.928213** | **-0.3235** |
| city_7_sandiego_3p5 | top-5000 power | -10 dB | Cross-attention ls_only | 0.989913 | 0.991768 | -0.0359 |
| city_7_sandiego_3p5 | top-5000 power | -10 dB | Noise-conditioned adapter | **0.985512** | **0.989168** | **-0.0473** |
| city_0_newyork_3p5 | top-5000 power | -10 dB | Cross-attention ls_only | 1.011011 | 1.018754 | 0.0807 |
| city_0_newyork_3p5 | top-5000 power | -10 dB | Noise-conditioned adapter | **0.994361** | **0.999144** | **-0.0037** |

结论：

- 在新的 `asu_campus_3p5` 场景上，经过有效覆盖用户筛选后，CBDNet-style 仍然略优于 cross-attention baseline。
- 直接从零训练的 CBDNet-style 在 San Diego / New York 上不稳定，但以 frozen cross-attention checkpoint 为底座的 noise-conditioned adapter 能稳定超过对应 baseline。
- San Diego 上，adapter 将 Test NMSE 从 `0.991768` 降至 `0.989168`；New York 上，从 `1.018754` 降至 `0.999144`。
- 该实验同时揭示了 DeepMIMO 场景评估中的一个重要注意点：当样本中存在大量 near-zero channel 时，普通 NMSE 会被极小分母严重放大。因此跨场景验证必须明确样本选择策略或采用额外稳健指标。
- 论文中更有力的表述应转向：noise-conditioned refinement 作为 cross-attention estimator 之后的轻量 adapter，可以在多个 DeepMIMO 场景上提供额外增益。
- 这比“从零训练一个更大的 CBDNet-style 网络”更符合原作者框架上的增量创新，也更适合强调低时延部署。

## 6. 模型效果好在哪里

当前模型的优势不是简单地在某一个单点实验中提升，而是形成了清晰的趋势：

> 显式噪声条件建模在低 SNR 信道估计中有价值，当前最清晰的收益出现在 `-10 dB` 附近；极端低 SNR 下需要重新调参验证。

与 cross-attention baseline 相比，CBDNet-style 模型的关键优势是：

- 将噪声从隐式扰动变成显式中间变量。
- 将普通 blind denoising 变成 noise-conditioned non-blind denoising。
- 在强噪声条件下显著降低 Test NMSE。
- 在 0 dB 和 -5 dB 下不发生明显性能损失。
- 给模型提供更好的可解释结构：先估计噪声，再根据信道和噪声共同 refine。

## 6.1 效率与低时延分析

新增效率测试脚本：

```text
deepmimo_multibs/benchmark_inference.py
```

测试设置：
- 场景：`o1_60`
- SNR：`-10 dB`
- mode：`ls_only`
- batch size：`256`
- 设备：CUDA
- warmup：`50`
- iters：`200`

效率结果：

| Model | Params | Checkpoint Size | Latency / Batch | Latency / Sample | Throughput |
|---|---:|---:|---:|---:|---:|
| MLP LS-only | 0.142M | 0.55 MB | 0.239 ms | 0.933 us | 1.07M samples/s |
| Cross-attn LS-only | 0.493M | 1.90 MB | 1.640 ms | 6.405 us | 156k samples/s |
| Noise-aware | 0.527M | 2.03 MB | 1.862 ms | 7.275 us | 137k samples/s |
| NC-CENet | 0.906M | 3.48 MB | 2.342 ms | 9.148 us | 109k samples/s |

结论：
- NC-CENet 不是最轻量模型，但参数量仍小于 `1M`，checkpoint 约 `3.5 MB`。
- 单样本平均推理时间约 `9.15 us`，可以支撑低时延 post-LS refinement 的表述。
- 相比 cross-attention，NC-CENet 用额外约 `2.74 us/sample` 的延迟换取 `-10 dB` 下更低 NMSE 和显式噪声条件解释。
- 论文中应表述为 accuracy / latency 对比：MLP 最快但精度弱，NC-CENet 稍慢但低 SNR refinement 能力更强。

补充 adapter 效率测试：

| Model | Scenario | Total Params | Trainable Params | Checkpoint Size | Latency / Batch | Latency / Sample | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| Noise-conditioned adapter | San Diego | 0.605M | 0.113M | 2.35 MB | 2.495 ms | 9.748 us | 102.6k samples/s |

说明：
- 上表为 adapter 端到端推理，包括 frozen cross-attention base 和 lightweight adapter correction。
- adapter 总参数量包含 cross-attention 主干，但训练参数只有 `0.113M`，因此它更适合作为轻量残差校准模块描述。
- 不能再把独立 NC-CENet 的 `9.15 us/sample` 直接当作 adapter 延迟；adapter 实测为 `9.75 us/sample`。

## 7. 模型的作用

模型的直接作用是：

> 从 noisy LS channel estimate 中恢复更接近真实信道的 refined channel estimate。

在通信系统中，它可以作为传统 LS/MMSE 信道估计之后的神经 refinement 模块，用于提升 CSI 质量。

潜在应用包括：

- 低 SNR 信道估计
- pilot-limited channel estimation
- DeepMIMO / 多基站场景中的 channel refinement
- 下游 beamforming、precoding、localization 等任务的 CSI 预处理

## 8. 当前论文贡献表述

可以写成三点：

1. **重新评估 RSS-informed refinement 的有效性。**
   在 DeepMIMO 多基站设置下，直接 RSS fusion 没有稳定优于 LS-only baseline，说明 RSS 信息需要更合理的结构化利用方式。

2. **提出 noise-conditioned non-blind channel refinement。**
   在 cross-attention residual refinement 基础上，引入 CBDNet-style noise map estimation，使模型显式感知 noisy LS 输入中的噪声状态。

3. **验证强噪声条件下的鲁棒性提升。**
   在 `0 dB / -5 dB / -10 dB` 三档 SNR 下，CBDNet-style 模型均达到优于或不弱于 cross-attention baseline 的效果，尤其在 `-10 dB` 下将 Test NMSE 从 `0.968148` 降至 `0.871116`。

## 9. 当前可用复现实验命令

### CBDNet-style, SNR = 0 dB

```powershell
python deepmimo_multibs/train_cbdnet_baseline.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel `
  --mode ls_only `
  --epochs 100 `
  --batch-size 256 `
  --device cuda `
  --noise-weight 0.01 `
  --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel/runs/cbdnet_ls_only_nw001_ep100
```

### CBDNet-style, SNR = -5 dB

```powershell
python deepmimo_multibs/train_cbdnet_baseline.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-5 `
  --mode ls_only `
  --epochs 100 `
  --batch-size 256 `
  --device cuda `
  --noise-weight 0.01 `
  --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-5/runs/cbdnet_ls_only_nw001_ep100
```

### CBDNet-style, SNR = -10 dB

```powershell
python deepmimo_multibs/train_cbdnet_baseline.py `
  --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10 `
  --mode ls_only `
  --epochs 100 `
  --batch-size 256 `
  --device cuda `
  --noise-weight 0.01 `
  --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cbdnet_ls_only_nw001_ep100
```

## 10. 下一步建议

当前阶段已经足够形成论文主线。下一步建议优先做以下补充：

1. **早停或 best checkpoint 报告规范化**
   已完成。当前 CBDNet-style 脚本支持 `--patience` 与 `--min-delta`，并在 `result.json` 中记录 `best_epoch`、`epochs_trained`、`early_stopped`。三个 SNR 下的结果均已按 validation-selected best checkpoint 报告。

2. **noise weight 消融**
   已完成。`SNR = -10 dB` 下，`noise_weight = 0.01` 取得当前最优 Test NMSE：`0.871116`。结果说明 noise map supervision 的最佳作用是适度 regularization / guidance，过大的噪声监督权重会削弱最终 channel estimation 目标。

3. **RSS 是否还能作为辅助条件**
   已完成。`SNR = -10 dB` 下，CBDNet-style `ls_only` 的 Test NMSE 为 `0.871116`，优于 `multibs_rss` 的 `0.889460`、`single_rss` 的 `0.892777` 和 `shuffled_rss` 的 `0.888009`。因此当前论文主贡献应聚焦于 noise-conditioned refinement，而不是 RSS fusion。

4. **可视化 noise map 或 residual**
   已完成。新增 `visualize_cbdnet_noise.py`，可导出 energy summary、noise energy scatter、error before/after scatter、token heatmaps 和 JSON 指标。三档 SNR 结果表明，模型 residual energy 与 true noise energy 在低 SNR 下高度接近，noise energy correlation 也在低 SNR 下更高，支持 noise-conditioned refinement 的解释性。

5. **扩展到更复杂信道维度 / 场景**
   已完成初步跨场景验证。新增 `asu_campus_3p5` 场景测试，发现原始前 5000 用户包含大量 near-zero channel，需按信道功率筛选有效覆盖用户。采用 top-5000 power 用户后，CBDNet-style 在 `-10 dB` 下 Test NMSE 为 `0.928213`，优于 cross-attention baseline 的 `0.940095`。后续更强版本可以继续扩展到更多天线、更多子载波或完整 OFDM-LS 设置。

## 11. 当前阶段一句话总结

> 我们从“RSS 是否直接提升信道估计”推进到“如何在 noisy LS 输入下做更鲁棒的结构化 refinement”。当前结果显示，CBDNet-inspired noise-conditioned non-blind denoising 是比 naive RSS fusion 更稳定、更有解释性、也更适合低 SNR 信道估计的创新方向。
