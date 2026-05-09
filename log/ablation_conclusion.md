# 消融实验阶段性结论

实验设置：Boston 15GHz 数据的前 1000 个样本，SNR = 0 dB，Np = 256，训练/验证/测试划分为 800/100/100，训练 100 epochs。UE noisy position 由 ray-tracing endpoint 提取 clean position 后加入水平高斯噪声生成。当前结果用于第一轮攻击性消融；后续还需要在 500 epochs、全量 9877 样本和多随机种子上继续验证。

## 结果汇总

| Experiment | Test NMSE | Test NMSE(dB) |
|---|---:|---:|
| LS input | 0.075315 | -11.23 |
| PINN normal RSS + noisy pos, alpha=0.01（first） | 0.021161 | -16.74 |
| PINN normal RSS + noisy pos, alpha=0.01（rerun） | 0.022283 | -16.52 |
| PINN normal RSS + noisy pos, alpha=0 | 0.021351 | -16.71 |
| PINN zero RSS, alpha=0.01 | 0.022897 | -16.40 |
| PINN constant RSS, alpha=0.01 | 0.022483 | -16.48 |
| PINN wrong RSS 8GHz, alpha=0.01 | 0.022495 | -16.48 |
| PINN wrong RSS canyon, alpha=0.01 | 0.022876 | -16.41 |
| PINN normal RSS + clean pos, alpha=0.01 | 0.022573 | -16.46 |

## 中文结论

第一，移除 physics-informed loss 后，模型测试 NMSE 为 -16.71 dB，与第一次 normal RSS 结果 -16.74 dB 仅相差约 0.03 dB。这说明当前代码中的物理损失项对最终性能贡献极小，原模型的主要增益并不依赖该项物理约束。

第二，将 RSS 输入置零后，模型仍达到 -16.40 dB；将 RSS 输入替换为常量图后，模型达到 -16.48 dB。二者相比 LS 初始估计的 -11.23 dB 仍有约 4.8 到 5.2 dB 的提升。这说明即使没有有效 RSS 地图内容，网络仍能获得接近论文声称幅度的增益。

第三，将 15GHz Boston 信道估计任务中的 RSS 输入替换为错误频段的 8GHz RSS map，测试 NMSE 为 -16.48 dB；替换为错误场景的 canyon RSS map，测试 NMSE 为 -16.41 dB。二者与 normal RSS rerun 的 -16.52 dB 非常接近，说明模型对 RSS 地图的频段一致性和场景一致性并不敏感。

第四，将 noisy UE position 替换为从 ray-tracing endpoint 提取的 clean position 后，测试 NMSE 为 -16.46 dB，未优于 noisy position 设置。这表明 RSS 裁剪位置精度并不是当前模型性能提升的主要来源。

综合来看，在该 1000 样本复现实验中，原作者模型的主要性能收益更可能来自 U-Net/Transformer 对 LS 初始信道估计的监督式去噪或残差修正，而不是来自 RSS 地图先验或 physics-informed loss。后续 500 epoch 实验应优先复查 normal、alpha=0、zero RSS、constant RSS 四组，以确认该现象是否稳定。

## 500 epoch 复查结果

在相同 1000 样本、SNR = 0 dB、Np = 256 的设置下，进一步对四个核心方向训练 500 epochs。为避免 checkpoint 占用过大，本轮仅保存 best validation 的 `state_dict`，不再保存包含 optimizer/scheduler 的 train checkpoint。

| Experiment | Test NMSE | Test NMSE(dB) |
|---|---:|---:|
| PINN normal RSS + noisy pos, alpha=0.01 | 0.017340 | -17.61 |
| PINN normal RSS + noisy pos, alpha=0 | 0.017270 | -17.63 |
| PINN zero RSS, alpha=0.01 | 0.017329 | -17.61 |
| PINN constant RSS, alpha=0.01 | 0.017352 | -17.61 |

500 epoch 结果进一步强化了 100 epoch 阶段的判断：normal RSS、zero RSS、constant RSS 三组的测试性能几乎完全一致，最大差异约 0.02 dB；同时 alpha=0 的结果反而略好于 alpha=0.01，差异约 0.02 dB。也就是说，在更长训练后，RSS 图像内容和 physics-informed loss 仍未表现出稳定、可分辨的贡献。

因此，在当前复现实验设置中，模型性能提升主要来自网络对 LS 初始估计的监督式非线性修正，而不是来自 RSS 地图先验或物理损失项。若论文声称 RSS/physics 是关键贡献，则这些 500 epoch 消融结果构成了更强的反证：即使删除物理损失或将 RSS 替换为无信息输入，模型仍达到与完整设置相同的测试精度。

## 全量 9877 样本第一阶段消融总结

在完整 9877 样本、SNR = 0 dB、Np = 256、训练 500 epochs 的设置下，完成了四个核心方向的第一阶段消融：

| Experiment | Test NMSE | Test NMSE(dB) |
|---|---:|---:|
| PINN normal RSS + noisy pos, alpha=0.01 | 0.018861 | -17.24 |
| PINN normal RSS + noisy pos, alpha=0 | 0.018541 | -17.32 |
| PINN zero RSS, alpha=0.01 | 0.017240 | -17.63 |
| PINN constant RSS, alpha=0.01 | 0.018793 | -17.26 |

全量结果与 1000 样本实验的趋势一致：移除 physics loss 后，测试性能没有下降，反而由 -17.24 dB 小幅提升到 -17.32 dB；将 RSS 输入替换为常量图后，结果为 -17.26 dB，几乎与 normal RSS 一致；将 RSS 输入置零后，测试结果达到 -17.63 dB，是当前全量四组实验中最好的结果。

需要谨慎解释 zero RSS 最优这一现象。它不能直接说明“zero RSS 更符合物理模型”，也不能说明 RSS 先验在理论上没有价值。更合理的解释是：在当前实现、当前数据和当前 NMSE 指标下，模型主要学习的是从 LS 初始估计到真实信道的监督式修正；有效 RSS 图像和当前 physics loss 并没有为该监督任务提供稳定收益，反而可能引入额外噪声、尺度不匹配或弱相关先验，使优化和泛化略受影响。

同时，zero RSS 模型虽然在 NMSE 指标上最好，但它更接近一个黑盒 LS refinement 网络。该模型可能很好地拟合当前数据分布，却不一定满足更强的物理一致性要求，例如路径损耗一致性、空间平滑性、delay-domain 稀疏性、AoA/AoD 几何一致性或跨场景传播规律。因此，第一阶段结果应表述为：当前代码中的 RSS/physics 设计没有在 NMSE 指标上体现出有效贡献，而不是否定所有 RSS/physics-informed 方法。

由此得到的第一阶段结论是：原模型在该复现实验中的主要性能来源更可能是深度网络对 LS 初始信道估计的去噪与残差修正，而不是 RSS 地图内容或当前形式的 physics-informed loss。后续研究应从“RSS 是否有用”进一步转向“在什么信息受限条件下 RSS 才有用，以及怎样设计真正能利用 RSS/物理先验的模型”。


我们不是否定 RSS 和 physics-informed 方法，而是指出原方法没有证明模型真正利用 RSS/physics。基于这一发现，我们转向 DeepMIMO 开源数据，研究 RSS 在低导频、低 SNR、多基站条件下的真实贡献，并提出显式利用 multi-BS RSS 的信道估计模型。
