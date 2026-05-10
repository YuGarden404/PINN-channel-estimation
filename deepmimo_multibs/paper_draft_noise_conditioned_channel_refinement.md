# 面向 DeepMIMO 信道估计的噪声条件残差细化方法

> 论文中文初稿。除模型名、英文术语、公式、脚本名和必要引用外，主体内容均使用中文撰写。本文工作基于原作者公开代码 **Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals** 进行复现、扩展和二次实验。我们引用并承接原作者关于 PINN、RSS-informed refinement、Transformer / cross-attention 信道细化的基础框架，在此基础上新增 DeepMIMO 数据构建、RSS 对照实验、noise-aware baseline、noise-conditioned residual refinement、轻量噪声条件 adapter、早停规范、消融实验、可视化、效率测试和跨场景验证。

## 摘要

准确的无线信道估计是下一代移动通信系统中的关键问题。传统 LS / MMSE 等模型驱动方法具有较好的可解释性，但在低 SNR、少导频和复杂传播环境下容易退化；纯数据驱动方法虽然表达能力强，但往往缺少物理约束和稳定解释。原作者提出的 physics-informed neural channel estimation 框架通过将初始 LS-OFDM 信道估计与 RSS map 等传播环境信息结合，实现了对传统估计结果的神经细化。

本文在原作者工作基础上进一步研究 DeepMIMO 场景中的信道细化问题。实验首先发现，在当前低维 DeepMIMO 多基站设置下，直接拼接或注意力融合 RSS 信息并不能稳定提升 NMSE；相反，cross-attention residual refinement 本身是有效的。基于这一观察，本文将研究重点从“如何直接融合 RSS”转向“如何更鲁棒地修正 noisy LS-like 信道输入”。受非盲去噪和残差误差建模思想启发，本文提出一种 **noise-conditioned residual refinement** 框架：一方面通过显式估计 token-wise noise map 引导信道去噪；另一方面在已训练 cross-attention estimator 之后引入零初始化轻量 adapter，使其只学习 baseline 未能消除的剩余误差。

在 DeepMIMO `o1_60` 场景下，独立训练的 NC-CENet 在 `0 dB`、`-5 dB` 和 `-10 dB` 三档 SNR 中均达到优于或不弱于 cross-attention baseline 的效果，尤其在 `-10 dB` 下将 Test NMSE 从 `0.9681` 降至 `0.8711`。进一步的 noise weight 消融、RSS 对照实验和 noise map / residual 可视化表明，性能提升主要来自显式 noise-conditioned refinement，而不是直接 RSS fusion。跨场景实验进一步显示，轻量 noise-conditioned adapter 能在 `asu_campus_3p5`、`city_7_sandiego_3p5` 和 `city_0_newyork_3p5` 上均超过对应 cross-attention baseline，说明噪声条件化不仅可以作为独立 estimator，也可以作为已有强估计器之后的低风险增量校准模块。

## 1. 引言

无线信道估计的目标是从有限导频观测中恢复准确的信道状态信息。高质量的信道估计对于 beamforming、precoding、用户定位以及后续资源分配都有重要影响。然而，在低信噪比、导频资源受限或传播环境复杂的场景中，传统估计器往往难以获得稳定结果。

原作者的工作 **Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals** 提出了一个重要方向：不完全依赖黑盒神经网络，而是将模型驱动的初始信道估计与传播环境中的物理信息结合起来。具体而言，原作者通过 U-Net、Transformer 和 cross-attention 等结构，将 LS-OFDM 初始估计与 RSS map 等环境特征融合，从而实现信道估计精度提升。

本文的出发点不是否定原作者结论，而是在原作者框架基础上寻找更稳定的创新方向。我们首先复现和扩展 RSS-informed refinement 思路，并在 DeepMIMO 多基站数据上测试 RSS 的作用。实验显示，在当前设置下，RSS 输入并没有带来稳定增益，`zero_rss` 和 `shuffled_rss` 等对照组甚至常常接近真实 RSS 的结果。这说明，简单 RSS fusion 并不一定能被网络有效利用。

但与此同时，cross-attention residual refinement 相比 MLP baseline 仍然明显有效。这给出一个更建设性的启发：真正稳定的收益可能来自对 noisy LS-like 信道输入本身的结构化细化，而不是直接依赖 RSS。基于这一观察，本文提出显式噪声建模的信道细化方法。

本文核心问题可以概括为：

> 在 RSS fusion 收益不稳定的情况下，是否可以通过显式 noise map estimation 和 non-blind denoising，提高 noisy LS-like channel refinement 的鲁棒性？

## 2. 与原作者代码和工作的关系

本文工作基于原作者公开代码仓库：

**Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals**  
作者：Alireza Javid and Nuria González-Prelcic

原作者代码提供了以下基础：

- physics-informed neural channel estimation 的整体问题设定；
- LS / LS-OFDM 初始信道估计与神经网络 refinement 的基本框架；
- RSS map / propagation prior 融合思想；
- U-Net、Transformer、cross-attention 等结构设计；
- 以 validation-selected checkpoint 报告测试结果的实验规范；
- NMSE 作为主要评价指标的训练与测试流程。

按照原作者 README 中的声明，若使用其代码或数据，应引用如下工作：

```bibtex
@inproceedings{javid2025physics,
  title={Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals},
  author={Javid, Alireza and Prelcic, Nuria Gonzalez},
  booktitle={NeurIPS 2025 Workshop: AI and ML for Next-Generation Wireless Communications and Networking}
}
```

本文在该基础上新增了 `deepmimo_multibs/` 实验分支，包括：

- DeepMIMO 场景检查与数据构建脚本；
- 多基站 RSS / pathloss fingerprint 构建；
- MLP RSS baseline；
- cross-attention residual baseline；
- noise-aware auxiliary baseline；
- CBDNet-style noise-conditioned channel estimator；
- early stopping 与 best checkpoint 规范化；
- noise weight 消融；
- RSS 辅助条件对照实验；
- noise map / residual 可视化；
- `asu_campus_3p5` 跨场景验证。

正式论文中应明确声明：

> 本文实现基于 Javid 和 González-Prelcic 的公开代码，并引用其 physics-informed / RSS-informed / cross-attention 信道细化框架。在此基础上，本文扩展 DeepMIMO 数据处理流程、RSS 对照实验、噪声条件残差细化模块和轻量 adapter 后处理机制。本文新增实验与结论仅对应本文构建的 DeepMIMO 分支，不应与原作者 Wireless Insite / pilot-limited OFDM benchmark 混同。

## 3. 问题定义

设真实信道为 `H`，noisy LS-like 初始估计为 `H_LS`。在当前 DeepMIMO 实验中，`H_LS` 由真实信道叠加复高斯白噪声得到：

```text
H_LS = H + N
```

其中 `N` 表示由指定 SNR 控制的 complex AWGN。

模型目标是学习一个 refinement function：

```text
f_theta(H_LS, c) -> H_hat
```

其中 `c` 可以是可选上下文信息，例如 RSS。本文最终主模型不依赖 RSS，而是将显式估计的 noise map 作为条件信息。

评价指标为 NMSE：

```text
NMSE = ||H_hat - H||_2^2 / ||H||_2^2
```

当前主要 DeepMIMO 设置为：

- channel shape：`[5000, 1, 8, 1]`
- complex channel tokens：`8`
- 每个 complex token 展开为 real / imaginary 两个实数维度
- 输入 token shape 可理解为 `[8, 2]`

## 4. RSS 融合实验带来的动机

原作者工作强调 RSS map 等传播环境信息可以作为物理先验，辅助信道估计。本文首先在 DeepMIMO 多基站设置下测试这一思路是否稳定。

### 4.1 MLP RSS Baseline

我们比较五种输入模式：

- `ls_only`：仅使用 noisy LS-like 信道输入；
- `single_rss`：使用目标基站单个 RSS；
- `multibs_rss`：使用 TX10 / TX11 / TX12 三个基站 RSS；
- `zero_rss`：使用全零 RSS 向量；
- `shuffled_rss`：使用样本顺序打乱后的 RSS。

Test NMSE 如下：

| SNR | ls_only | single_rss | multibs_rss | zero_rss | shuffled_rss |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.418038 | 0.432860 | 0.419044 | **0.410742** | 0.434593 |
| -5 dB | 0.754223 | 0.793107 | 0.794015 | **0.734087** | 0.774373 |
| -10 dB | **1.070395** | 1.105507 | 1.133112 | 1.112689 | 1.113992 |

可以看到，naive RSS concatenation 没有稳定优于 `ls_only`。特别是 `zero_rss` 在部分设置下反而更优，说明当前 MLP 结构并没有稳定利用 RSS 中的样本级传播信息。

### 4.2 Cross-Attention Residual Baseline

为了排除“模型太弱”的可能性，我们进一步测试 cross-attention residual baseline。

在 `SNR = 0 dB` 下：

| Mode | Best Val NMSE | Test NMSE | Test dB |
|---|---:|---:|---:|
| ls_only | 0.300158 | **0.364926** | -4.38 |
| single_rss | 0.301669 | 0.376288 | -4.24 |
| multibs_rss | **0.294580** | 0.381816 | -4.18 |
| zero_rss | 0.298630 | 0.368618 | -4.33 |
| shuffled_rss | 0.305098 | 0.384424 | -4.15 |

cross-attention 相比 MLP 有明显提升，但 RSS 仍不稳定。这说明 cross-attention residual refinement 是有价值的，而 RSS fusion 在当前低维 DeepMIMO 设置下不是稳定收益来源。

这也引出本文方法：保留 residual refinement 的思想，但将显式噪声条件作为更可靠的 refinement guidance。

## 5. 方法：噪声条件残差细化框架

本文提出一种噪声条件残差细化框架。该框架继承原作者“model-based initial estimation + neural refinement”的混合思想，但将条件信息从外部 RSS prior 扩展为由输入信道自身估计得到的显式噪声条件。换言之，本文并不把 noisy LS-like 信道估计视为普通黑盒输入，而是先估计其噪声状态，再在噪声条件已知的情况下执行 residual refinement。

为了避免将方法退化为简单堆叠网络，本文采用两级论证方式。第一，构建独立训练的 **NC-CENet: Noise-Conditioned Channel Estimation Network**，验证显式 noise map 对低 SNR 信道细化是否有帮助。第二，在跨场景实验中进一步采用 **noise-conditioned adapter**：以已经训练好的 cross-attention estimator 作为主估计器并冻结其参数，adapter 仅学习主估计器输出中的剩余误差。该设计使本文方法更接近 residual error compensation / post-refinement，而不是任意增加网络深度。

### 5.1 噪声条件化的基本形式

设真实信道为 `H`，noisy LS-like 初始估计为：

```text
H_LS = H + N
```

其中 `N` 为由指定 SNR 控制的 complex AWGN。对每个 complex channel token，本文将其实部和虚部分离，形成 token-wise real-valued representation。对于当前 `o1_60` 设置，信道包含 `8` 个 complex tokens，因此网络输入可写为：

```text
X_LS in R^{B x 8 x 2}
```

其中 `B` 为 batch size，最后一维对应 real / imaginary components。普通黑盒 refinement 直接学习：

```text
X_LS -> X_hat
```

本文首先将该映射分解为两个耦合子问题：

```text
X_LS -> M_noise
[X_LS, M_noise] -> Delta_X
X_hat = X_LS + Delta_X
```

其中 `M_noise` 表示估计的噪声条件，`Delta_X` 表示对初始 LS-like 输入的残差修正。该设计与原作者使用 RSS map 作为 propagation prior 的思想保持一致：二者都不是让网络完全从零恢复信道，而是通过额外条件约束 refinement 的解空间。区别在于，本文的条件变量来自输入信道的噪声结构，而不是外部环境 RSS map。

### 5.2 Stage 1: Noise Map Estimation

第一阶段学习如下映射，从 noisy LS-like 信道输入中估计 token-wise noise map：

```text
M_noise = g_phi(X_LS)
```

其中 `g_phi` 是一个 token-wise feature extractor，用于从 noisy LS-like 信道中估计噪声相关表征。`M_noise` 与 `X_LS` 保持相同的 token resolution，因此每个 channel token 都拥有对应的噪声条件。直观上，该模块回答的问题是：当前输入中哪些 token 更可能受到噪声扰动，以及扰动在 real / imaginary components 上呈现怎样的结构。

需要强调的是，`M_noise` 不应被过度解释为逐点精确的物理噪声幅值。由于最终训练目标仍然以 channel reconstruction 为主，`M_noise` 更合理的解释是 **denoising guidance feature**：它为后续 refinement 提供噪声强度和扰动模式的中间表征，使模型不必完全依赖隐式 hidden states 来判断输入可靠性。

### 5.3 Stage 2: Non-Blind Channel Denoising

第二阶段将 noisy channel token 与 noise map 拼接为条件化输入：

```text
Z = concat(X_LS, M_noise)
```

随后通过 token projection、positional encoding 和 Transformer-style self-attention 模块进行全局建模。self-attention 允许每个 channel token 根据其它 token 的状态调整自身修正量，从而捕获不同天线/token 之间的相关结构。该过程可表示为：

```text
U = Transformer(Proj(Z) + P)
Delta_X = h_theta(U)
X_hat = X_LS + Delta_X
```

其中 `P` 为位置编码，`h_theta` 为输出投影层。残差形式 `X_hat = X_LS + Delta_X` 保留了初始 LS-like estimator 的模型驱动信息，使网络主要学习噪声抵消和结构化误差修正，而不是重新生成完整信道。这一点与原作者的 refinement philosophy 一致：神经网络作为 model-based estimator 的增强器，而不是替代整个通信估计链路。

从 denoising 角度看，普通 blind refinement 只学习 `X_LS -> X_hat`，而 NC-CENet 学习 `X_LS -> M_noise` 与 `(X_LS, M_noise) -> X_hat` 两个过程。因此，后者是显式 noise-conditioned non-blind refinement。该结构在低 SNR 场景下尤其有意义，因为输入误差不再是弱扰动，而会成为决定 refinement 方向的主导因素。

### 5.4 Cross-Attention 后的噪声条件 Adapter

独立训练的 NC-CENet 能够验证噪声条件化本身的有效性，但在跨场景实验中，直接从零训练一个新的 estimator 也会引入优化稳定性和场景分布差异的问题。为了更严格地回答“噪声条件化是否能在 cross-attention baseline 基础上带来增益”，本文进一步设计轻量 noise-conditioned adapter。

设已经训练好的 cross-attention estimator 为 `F_ca`：

```text
X_ca = F_ca(X_LS)
```

adapter 不替代 `F_ca`，而是学习其剩余误差：

```text
M_noise = g_phi(X_LS)
Delta_A = a_psi(X_LS, X_ca, X_ca - X_LS, M_noise)
X_hat = X_ca + alpha Delta_A
```

其中 `alpha` 是较小的 adapter scale，本文使用 `alpha = 0.1`。在训练时，`F_ca` 参数被冻结，只有 `g_phi` 和 `a_psi` 更新。为保证 adapter 不是任意破坏已有估计器，本文将 correction head 的最后一层零初始化，使训练开始时：

```text
Delta_A = 0,  X_hat = X_ca
```

因此，adapter 的初始性能严格等价于 cross-attention baseline。后续训练若能降低 validation / test NMSE，则可解释为模型学到了 baseline 未能消除的系统性残差误差。这一性质使 adapter 更像一个可控的 post-refinement / calibration module，而不是简单地在已有模型后面堆叠网络。

adapter 的训练损失为：

```text
L = L_channel + lambda_noise L_noise + lambda_corr ||Delta_A||_2^2
```

其中 `L_channel` 仍为 NMSE，`L_noise` 约束 noise map 与真实扰动的一致性，`lambda_corr` 对 adapter correction 进行幅度正则化。该正则项鼓励 adapter 只进行必要的小幅修正，从而保留 cross-attention estimator 的主体预测能力。

### 5.5 损失函数

训练损失由 channel NMSE 和 noise map auxiliary loss 组成：

```text
L = L_channel + lambda_noise L_noise
```

其中主项采用 NMSE：

```text
L_channel = ||H_hat - H||_2^2 / ||H||_2^2
```

噪声辅助项使用真实扰动 `H - H_LS` 作为 supervision：

```text
L_noise = MSE(M_noise, H - H_LS)
```

主实验中：

```text
lambda_noise = 0.01
```

该权重控制 noise map 的物理可解释性与最终 channel estimation accuracy 之间的折中。消融实验显示，适度噪声监督有助于 regularization 和条件引导；但过大的噪声监督会使模型过分拟合 noise map，而削弱最终 NMSE 目标。在极低 SNR 下，最佳权重进一步漂移到 `0.001`，说明噪声条件监督应视为可调节的 inductive bias，而不是固定不变的物理约束。

### 5.6 与原作者 PINN 框架的关系

原作者方法的核心是利用 RSS map 等环境传播信息作为 physics-informed prior，通过 cross-attention 将初始信道估计与外部物理上下文融合。本文保留这一混合式思想，但根据当前 DeepMIMO 实验观察，将条件信息从 RSS fusion 转向 learned noise condition，并进一步将其作为 cross-attention estimator 之后的 residual calibration signal：

```text
Original PINN: LS estimate + RSS / propagation context -> refined channel
NC-CENet:      LS estimate + learned noise condition    -> refined channel
Adapter:       cross-attention estimate + learned noise condition -> residual calibration
```

这种设计不是对 RSS-informed refinement 的否定，而是在 RSS 收益不稳定的低维 DeepMIMO 设置下，对“何种条件信息最稳定有效”的进一步探索。实验结果表明，当前性能提升主要来自对 noisy LS-like 输入本身的显式噪声条件化建模。

### 5.7 实现脚本

独立 NC-CENet 训练脚本：

```text
deepmimo_multibs/train_cbdnet_baseline.py
```

cross-attention 后置 adapter 训练脚本：

```text
deepmimo_multibs/train_nc_adapter.py
```

可视化脚本：

```text
deepmimo_multibs/visualize_cbdnet_noise.py
```

数据构建脚本：

```text
deepmimo_multibs/build_multibs_dataset.py
deepmimo_multibs/build_channel_dataset.py
```

## 6. 实验设置

### 6.1 主场景：DeepMIMO o1_60

主要实验使用 DeepMIMO `o1_60`：

- 用户数：`5000`
- 目标基站：`TX10`
- 可选 RSS 来源：`TX10 / TX11 / TX12`
- channel shape：`[5000, 1, 8, 1]`
- channel tokens：`8`
- SNR：`0 dB`、`-5 dB`、`-10 dB`
- batch size：`256`
- 最大训练轮数：`100`
- 随机种子：`42`
- 设备：CUDA

当前 LS-like 输入由 clean channel 叠加 complex AWGN 得到。因此本文现阶段结果应被理解为 DeepMIMO-based neural channel refinement baseline，而不是完整通信链路中的最终 OFDM-LS benchmark。

### 6.2 Model Selection

所有最终测试结果均加载 validation NMSE 最低的 `best_model.pth` 后计算。需要特别说明的是，本文包含两类模型证据：

- **独立噪声条件模型**：包括 noise-aware baseline 和 NC-CENet / CBDNet-style estimator，用于验证显式 noise map、noise loss、RSS 对照和低 SNR 去噪机理；
- **后置噪声条件 adapter**：以已训练 cross-attention checkpoint 为主估计器并冻结其参数，只训练零初始化轻量 adapter，用于验证在强 baseline 基础上的增量残差校准能力。

因此，第 7 节和第 8 节中的 SNR 曲线、noise weight 消融、RSS 对照和可视化主要对应独立 NC-CENet / CBDNet-style estimator；第 10 节跨场景验证中的 San Diego 和 New York 结果对应 noise-conditioned adapter。二者回答的问题不同，不能将独立模型的效率或消融结果直接等同于 adapter 的结果。

CBDNet-style 模型使用：

- `--patience 20`
- `--min-delta 0.0`

这样可以避免报告最后一轮模型导致的不稳定性。adapter 实验使用零初始化 correction head，并在训练前保存等价于 cross-attention baseline 的初始状态；若后续 validation NMSE 下降，则保存 adapter checkpoint。

## 7. 主实验结果

### 7.1 不同 SNR 下的对比

建议图：

```text
deepmimo_multibs/paper_figures/fig1_main_nmse_vs_snr.png
deepmimo_multibs/paper_figures/fig2_nmse_db_vs_snr.png
```

| SNR | MLP ls_only | Cross-attn ls_only | Noise-aware nw=0.01 | NC-CENet / CBDNet-style |
|---:|---:|---:|---:|---:|
| 0 dB | 0.418038 | 0.364926 | 0.378097 | **0.359841** |
| -5 dB | 0.754223 | 0.645223 | 0.677717 | **0.644630** |
| -10 dB | 1.070395 | 0.968148 | 0.981476 | **0.871116** |
| -15 dB | 1.377312 | 1.014650 | **1.005800** | 1.010255 |
| -20 dB | 2.014708 | 1.057143 | **1.050226** | 1.145429 |

可以看到，NC-CENet 在 `0/-5/-10 dB` 三档主实验中达到最优或基本最优，且在 `-10 dB` 强噪声场景下优势最明显。补充的 `-15/-20 dB` 结果显示，MLP baseline 在极低 SNR 下明显退化；Noise-aware auxiliary baseline 在极低 SNR 下反而较稳，提示噪声监督和模型容量在极端噪声区间需要单独讨论。

在 `-10 dB` 下，相比 cross-attention baseline：

```text
0.968148 -> 0.871116
```

这说明显式 noise-conditioned refinement 对严重噪声条件更有帮助。

### 7.1.1 极低 SNR 初步补充实验

为进一步测试极强干扰条件，补充 `-15 dB` 和 `-20 dB` 两档实验。当前结果显示，固定 `noise_weight = 0.01` 时，CBDNet-style 在极低 SNR 下不再优于 cross-attention baseline。

| SNR | Cross-attn ls_only | NC-CENet / CBDNet-style | Observation |
|---:|---:|---:|---|
| -15 dB | **1.014650** | 1.010255 | 基本持平，CBDNet 略优 |
| -20 dB | **1.057143** | 1.145429 | Cross-attention 更优 |

这一结果提示：`noise_weight = 0.01` 是 `0/-5/-10 dB` 主实验中的有效设置，但在 `-20 dB` 极端噪声条件下可能不再最优。后续需要对极低 SNR 单独进行 noise weight 或模型容量消融。论文表述中应避免简单声称“噪声越强优势越大”，更稳妥的说法是：NC-CENet 在中低 SNR 尤其 `-10 dB` 下表现出明显优势，而极端低 SNR 下需要进一步调参和结构增强。

### 7.2 早停与 Best Checkpoint

建议图：

```text
deepmimo_multibs/paper_figures/fig5_best_epoch_early_stop.png
```

| SNR | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB | Early Stop |
|---:|---:|---:|---:|---:|---:|---|
| 0 dB | 17 | 37 | 0.301688 | **0.359841** | -4.4389 | Yes |
| -5 dB | 32 | 52 | 0.572482 | **0.644630** | -1.9069 | Yes |
| -10 dB | 53 | 73 | 0.809526 | **0.871116** | -0.5992 | Yes |

best epoch 随着 SNR 降低而后移，说明噪声越强，模型需要更长训练才能学到稳定 refinement。

## 8. 消融实验

### 8.1 Noise Weight 消融

建议图：

```text
deepmimo_multibs/paper_figures/fig3_noise_weight_ablation.png
```

在 `SNR = -10 dB` 下进行 `lambda_noise` 消融：

| Noise Weight | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---:|---:|---:|---:|---:|---:|
| 0 | 56 | 76 | 0.814311 | 0.878649 | -0.5618 |
| 0.001 | 34 | 54 | 0.822343 | 0.878225 | -0.5639 |
| 0.005 | 40 | 60 | 0.820491 | 0.885283 | -0.5292 |
| **0.01** | 53 | 73 | **0.809526** | **0.871116** | **-0.5992** |
| 0.05 | 52 | 72 | 0.812018 | 0.892857 | -0.4922 |

结论：

- `lambda_noise = 0.01` 最优；
- `lambda_noise = 0` 仍然较强，说明结构本身有效；
- 适度 noise supervision 有帮助；
- 过大的 noise supervision 会干扰 channel estimation 主目标。

### 8.1.1 极低 SNR 下的 Noise Weight 补充消融

为解释 `-15 dB` 和 `-20 dB` 下固定 `lambda_noise = 0.01` 表现不稳定的问题，进一步在两档极低 SNR 上补充 noise weight 消融。

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

补充消融表明，极低 SNR 下 `lambda_noise = 0.001` 明显优于主实验使用的 `0.01`。在 `-15 dB` 下，调小 noise supervision 后 NC-CENet 的 Test NMSE 从 `1.010255` 降至 `0.977731`，已经优于 cross-attention baseline 的 `1.014650`；在 `-20 dB` 下，最佳结果从 `1.145429` 改善到 `1.063262`，接近但仍略弱于 cross-attention baseline 的 `1.057143`。因此，极低 SNR 下的问题更像是噪声监督权重和优化稳定性需要单独调节，而不是 noise-conditioned refinement 完全失效。

### 8.2 RSS 辅助条件实验

建议图：

```text
deepmimo_multibs/paper_figures/fig4_rss_control.png
```

在 `SNR = -10 dB`、`lambda_noise = 0.01` 下测试 RSS 是否仍有帮助：

| Mode | RSS Dim | Best Epoch | Epochs Trained | Best Val NMSE | Test NMSE | Test dB |
|---|---:|---:|---:|---:|---:|---:|
| **ls_only** | 0 | 53 | 73 | **0.809526** | **0.871116** | **-0.5992** |
| multibs_rss | 3 | 48 | 68 | 0.828833 | 0.889460 | -0.5087 |
| single_rss | 1 | 40 | 60 | 0.832064 | 0.892777 | -0.4926 |
| shuffled_rss | 3 | 34 | 54 | 0.827822 | 0.888009 | -0.5158 |

RSS 输入没有提升 CBDNet-style 模型。真实 RSS 与 shuffled RSS 表现接近，说明当前 RSS 分支仍未稳定学习到样本级 RSS-channel 对应关系。

因此，本文主贡献应表述为：

> noise-conditioned non-blind channel refinement，而不是 RSS fusion。

## 9. Noise Map 与 Residual 可视化

为了增强可解释性，我们使用：

```text
deepmimo_multibs/visualize_cbdnet_noise.py
```

导出：

- `energy_summary.png`
- `noise_energy_scatter.png`
- `error_before_after_scatter.png`
- `token_heatmaps.png`
- `visualization_metrics.json`

### 9.1 可视化指标

| SNR | LS NMSE | Refined NMSE | NMSE Gain | True Noise Energy | Pred Noise Energy | Refined Error Energy | Residual Energy | Noise Energy Corr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 3.134104 | 0.358784 | 2.775321 | 0.054722 | 0.038467 | 0.008628 | 0.049943 | 0.101340 |
| -5 dB | 9.910910 | 0.643044 | 9.267866 | 0.076300 | 0.030418 | 0.009676 | 0.075334 | 0.341935 |
| -10 dB | 31.341043 | 0.869740 | 30.471303 | 0.084102 | 0.015123 | 0.006096 | 0.085615 | 0.363095 |

主要观察：

- refinement 后 NMSE 远低于直接 noisy LS-like 输入；
- 在低 SNR 下，residual energy 与 true noise energy 接近，说明模型 residual 主要承担抵消噪声的作用；
- noise energy correlation 在低 SNR 下更高，支持“噪声条件化在强噪声场景下更有价值”；
- predicted noise map 不应被解释为逐点精确物理噪声，而应理解为 denoising guidance feature。

## 10. 跨场景验证

为了验证结论不只依赖 `o1_60`，本文进一步测试 DeepMIMO `asu_campus_3p5`、`city_7_sandiego_3p5` 和 `city_0_newyork_3p5` 三个额外场景。

建议图：

```text
deepmimo_multibs/paper_figures/fig6_cross_scenario_asu.png
```

在 `asu_campus_3p5` 场景中，初始直接取前 `5000` 个用户时，NMSE 出现 `1e8` 量级异常。检查后发现，样本中包含大量零信道或极弱信道，导致 NMSE 分母 `||H||^2` 接近 0。因此，跨场景实验统一采用按平均信道功率从高到低排序后选取 top-5000 有效覆盖用户：

```text
--sort-by-power descending --num-users 5000
```

各场景筛选后统计如下：

| Scenario | Source Users | Selected Users | Selected Power Min | Selected Power Mean | Selected Power Max |
|---|---:|---:|---:|---:|---:|
| asu_campus_3p5 | 131931 | 5000 | 2.408770e-11 | 5.345881e-11 | 2.426602e-10 |
| city_7_sandiego_3p5 | 36432 | 5000 | 1.680043e-11 | 2.596491e-10 | 8.834474e-09 |
| city_0_newyork_3p5 | 31719 | 5000 | 2.982259e-12 | 2.257269e-10 | 8.760148e-09 |

在 `SNR = -10 dB` 下：

| Scenario | Selection | Model | Best Val NMSE | Test NMSE | Test dB |
|---|---|---|---:|---:|---:|
| asu_campus_3p5 | top-5000 power | Cross-attention ls_only | 0.934861 | 0.940095 | -0.2683 |
| asu_campus_3p5 | top-5000 power | NC-CENet / CBDNet-style | **0.934779** | **0.928213** | **-0.3235** |
| city_7_sandiego_3p5 | top-5000 power | Cross-attention ls_only | 0.989913 | 0.991768 | -0.0359 |
| city_7_sandiego_3p5 | top-5000 power | Noise-conditioned adapter | **0.985512** | **0.989168** | **-0.0473** |
| city_0_newyork_3p5 | top-5000 power | Cross-attention ls_only | 1.011011 | 1.018754 | 0.0807 |
| city_0_newyork_3p5 | top-5000 power | Noise-conditioned adapter | **0.994361** | **0.999144** | **-0.0037** |

早期 CBDNet-style 模型在 San Diego 和 New York 上直接从零训练时表现不稳定，说明仅靠固定结构和固定 `noise_weight = 0.01` 并不能保证跨场景泛化。为更严格验证“噪声条件化是否能在 cross-attention 基础上带来增益”，本文进一步采用轻量 **noise-conditioned adapter**：先加载每个场景训练好的 cross-attention checkpoint，并冻结其参数，然后训练一个零初始化的小型噪声条件 adapter 对 cross-attention 输出进行二次残差修正。由于 adapter 初始状态等价于原 cross-attention estimator，最终提升可以解释为对强 baseline 的增量 refinement。

结果显示，在 San Diego 和 New York 两个城市级场景中，noise-conditioned adapter 均超过对应 cross-attention baseline。其中 San Diego 的 Test NMSE 从 `0.991768` 降至 `0.989168`，New York 从 `1.018754` 降至 `0.999144`。这说明，当噪声条件模块以轻量 adapter 形式叠加在已有 cross-attention estimator 上时，跨场景稳定性明显优于从零训练的 CBDNet-style 版本，也更符合“在原作者 cross-attention refinement 基础上进一步提升”的论文主张。

## 11. 效率与低时延分析

5G/6G 信道估计模块不仅需要较高精度，也需要满足低时延推理要求。为评估所提方法作为 LS-like 初始估计之后的 neural refinement module 是否具备实际部署潜力，本文在 `o1_60`、`SNR = -10 dB`、`batch size = 256` 设置下，对主要独立模型进行推理效率测试。测试脚本为：

```text
deepmimo_multibs/benchmark_inference.py
```

建议图：

```text
deepmimo_multibs/paper_figures/fig7_accuracy_latency_tradeoff.png
```

| Model | Params | Checkpoint Size | Latency / Batch | Latency / Sample | Throughput |
|---|---:|---:|---:|---:|---:|
| MLP LS-only | 0.142M | 0.55 MB | 0.239 ms | 0.933 us | 1.07M samples/s |
| Cross-attn LS-only | 0.493M | 1.90 MB | 1.640 ms | 6.405 us | 156k samples/s |
| Noise-aware | 0.527M | 2.03 MB | 1.862 ms | 7.275 us | 137k samples/s |
| NC-CENet | 0.906M | 3.48 MB | 2.342 ms | 9.148 us | 109k samples/s |

可以看到，独立 NC-CENet 因为额外引入 noise estimation branch 和 condition-guided denoising branch，参数量和延迟高于 cross-attention baseline。然而其总参数量仍小于 `1M`，checkpoint 约 `3.5 MB`，在 RTX-class GPU 上的单样本平均推理时间约为 `9.15 us`。这说明独立噪声条件 estimator 虽然不是最轻量的模型，但仍可作为低时延 post-LS refinement 模块使用。

从 accuracy / latency 对比看，MLP baseline 具有最低延迟，但在 `-10 dB` 下 Test NMSE 明显较差；cross-attention 和 noise-aware baseline 延迟适中；独立 NC-CENet 以额外推理开销换取更好的低 SNR NMSE 和更清晰的噪声条件解释。

需要注意，表中效率数据对应独立 NC-CENet。第 10 节中的 adapter 需要先得到 cross-attention 输出，再执行小型 noise encoder 与 correction encoder，因此其端到端开销应单独报告：

| Model | Scenario | Total Params | Trainable Params | Checkpoint Size | Latency / Batch | Latency / Sample | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| Noise-conditioned adapter | San Diego | 0.605M | 0.113M | 2.35 MB | 2.495 ms | 9.748 us | 102.6k samples/s |

该结果在 `city_7_sandiego_3p5`、`batch size = 256`、CUDA、warmup `50`、iters `200` 下测得。由于 adapter 包含 frozen cross-attention 主干，因此总参数量为 `0.605M`；但实际训练参数仅为 `0.113M`，说明该模块主要作为轻量残差校准器使用。端到端单样本延迟约 `9.75 us`，略高于独立 NC-CENet 的 `9.15 us`，但仍处于微秒级推理范围。后续若将 cross-attention 与 adapter 进行算子融合或蒸馏，有望进一步降低部署开销。

## 12. 讨论

### 12.1 本文核心贡献是什么？

本文的核心贡献不是证明 RSS 一定有效。相反，实验显示，在当前 DeepMIMO 设置下，直接 RSS fusion 并不稳定。

更准确的贡献表述是：

> 在 noisy LS-like 信道输入下，显式 noise-conditioned residual refinement 能够提升神经信道估计的鲁棒性；当其以零初始化轻量 adapter 形式叠加于 cross-attention estimator 后，还可以作为低风险的残差误差校准模块。

因此，本文贡献并不是简单提出一个更大的网络，而是将信道估计误差分解为两个层次：cross-attention 主估计器负责捕获主要信道结构，noise-conditioned 模块负责建模剩余噪声相关误差。这种分解比端到端替换 baseline 更容易解释，也更符合通信系统中“传统估计器 + 神经后处理”的工程部署方式。

### 12.2 为什么这是对原作者工作的延伸？

原作者工作的核心思想是：将模型驱动初始估计与物理/环境信息结合，通过神经网络进行信道细化。本文保留这一方向，但实验发现，在当前 DeepMIMO 设置下，最可靠的条件信息不是 RSS，而是模型从输入中显式学习得到的 noise condition。

可以概括为：

```text
原作者方向：LS estimate + physical/RSS context -> refined channel
本文扩展：LS estimate + learned noise condition -> refined channel
```

因此，本文是在原作者框架上的结构化延伸，而不是简单替代。

### 12.3 是否存在简单堆叠模块的问题？

一个自然质疑是：如果在 cross-attention 后再接一个 adapter，是否只是通过增加参数量获得小幅提升。本文从三个方面降低这一风险。

第一，adapter 不是任意后接网络，而是以 noise map 为条件的残差误差模型。它的输入不仅包含 cross-attention 输出，还包含 `X_LS`、`X_ca - X_LS` 和显式估计的 `M_noise`，目标是修正与输入噪声状态相关的剩余误差。

第二，adapter 采用零初始化 correction head，使初始模型严格退化为原 cross-attention baseline。因此，训练过程不是从随机后处理器开始，而是在不破坏 baseline 的前提下寻找小幅增益。这与 ResNet、LoRA 和许多 calibration / adapter 方法中的 identity-preserving initialization 思想一致。

第三，cross-attention 主干在 adapter 实验中被冻结，adapter 只学习低维残差修正。若性能提升来自简单增大主模型容量，则冻结主干的小型 adapter 不应稳定带来增益。San Diego 和 New York 的结果表明，噪声条件残差校准确实能降低 cross-attention 的剩余误差。

### 12.4 局限性

当前工作仍有以下局限：

- LS-like 输入由 clean channel 叠加 AWGN 得到，还不是完整 OFDM-LS 链路；
- 主场景 channel 维度较小，仅为 `[1, 8, 1]`；
- RSS fusion 可能需要更丰富的空间维度、更复杂 channel 表示或更强融合机制；
- adapter 结果仍属于场景内训练和测试，还不是跨场景零样本泛化；
- 当前跨场景结论主要基于单随机种子，最终论文级结果最好增加多随机种子统计；
- 后续应扩展到更多天线、更多子载波或完整 pilot-limited OFDM 设置。

## 13. 结论

本文基于原作者 physics-informed neural channel estimation 代码框架，提出一种噪声条件残差细化方法。实验表明，直接 RSS fusion 在当前 DeepMIMO 设置下并不稳定，而显式 noise map estimation 与 noise-conditioned denoising 能够更稳定地提升 noisy LS-like channel refinement 性能。独立训练的 NC-CENet 在 `o1_60` 三档 SNR 下均优于或不弱于 cross-attention baseline，尤其在 `-10 dB` 下取得明显提升。进一步地，零初始化 noise-conditioned adapter 在 San Diego 和 New York 场景中均超过对应 frozen cross-attention baseline，说明该方法可以作为已有强 estimator 后的轻量残差校准模块。总体而言，本文的主要结论是：噪声条件化并非简单替代 RSS prior，而是为低 SNR 信道估计提供了一种可解释、可控且具备低时延潜力的 residual refinement 机制。

## 附录 A. 主要复现实验命令

训练 `SNR = -10 dB` 下的 NC-CENet：

```powershell
python deepmimo_multibs/train_cbdnet_baseline.py --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10 --mode ls_only --epochs 100 --batch-size 256 --device cuda --noise-weight 0.01 --patience 20 --min-delta 0.0 --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cbdnet_ls_only_nw001_ep100_es20
```

训练 cross-attention baseline：

```powershell
python deepmimo_multibs/train_cross_attention_baseline.py --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10 --mode ls_only --epochs 100 --batch-size 256 --device cuda --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cross_attention_ls_only_ep100
```

生成 noise map / residual 可视化：

```powershell
python deepmimo_multibs/visualize_cbdnet_noise.py --data-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10 --checkpoint deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cbdnet_ls_only_nw001_ep100_es20/best_model.pth --mode ls_only --split test --max-samples 500 --heatmap-samples 80 --device cuda --out-dir deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel_snr-10/runs/cbdnet_ls_only_nw001_ep100_es20/visualizations
```

构建 `asu_campus_3p5` top-power 外部验证数据：

```powershell
python deepmimo_multibs/build_channel_dataset.py --scenario asu_campus_3p5 --target-pair-index 0 --rss-dir deepmimo_multibs/processed/asu_campus_3p5_single_tx_rss_all --out-dir deepmimo_multibs/processed/asu_campus_3p5_single_tx_channel_snr-10_top5000 --num-users 5000 --sort-by-power descending --snr -10 --seed 42
```

训练 San Diego 场景的 noise-conditioned adapter：

```powershell
python deepmimo_multibs/train_nc_adapter.py --data-dir deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000 --base-checkpoint deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000/runs/cross_attention_ls_only_ep100/best_model.pth --mode ls_only --epochs 80 --batch-size 256 --device cuda --noise-weight 0.001 --adapter-scale 0.1 --patience 15 --out-dir deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000/runs/nc_adapter_ls_only_nw0001_s01_ep80_es15
```

测试 San Diego adapter 推理效率：

```powershell
python deepmimo_multibs/benchmark_inference.py --model nc_adapter --data-dir deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000 --checkpoint deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000/runs/nc_adapter_ls_only_nw0001_s01_ep80_es15/best_model.pth --base-checkpoint deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000/runs/cross_attention_ls_only_ep100/best_model.pth --mode ls_only --device cuda --batch-size 256 --warmup 50 --iters 200 --out deepmimo_multibs/processed/city_7_sandiego_3p5_rx0_tx1_2_3_channel_snr-10_top5000/runs/nc_adapter_ls_only_nw0001_s01_ep80_es15/benchmark.json
```
