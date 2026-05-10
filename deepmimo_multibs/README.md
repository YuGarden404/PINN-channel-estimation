# DeepMIMO Multi-BS RSS Study

This directory starts the second-stage project: testing whether multi-BS RSS
fingerprints provide a useful prior for neural channel estimation.

Stage-1 finding: on the reproduced Boston/Wireless-InSite pipeline, normal RSS
did not improve NMSE over zero/constant RSS. Stage-2 asks a more constructive
question: when LS observations are weak, can multi-BS RSS become useful?

## Planned Pipeline

1. Install DeepMIMO:

```powershell
pip install DeepMIMO
```

2. Inspect an installed/downloaded DeepMIMO scenario:

```powershell
python deepmimo_multibs/inspect_deepmimo.py --scenario o1_60
```

3. Build a multi-BS dataset manifest and arrays:

```powershell
python deepmimo_multibs/build_multibs_dataset.py `
  --scenario o1_60 `
  --bs-ids 1 2 3 `
  --target-bs 1 `
  --out-dir deepmimo_multibs/processed/o1_60_bs123 `
  --num-users 5000 `
  --snr 0 `
  --pilot-spacing 4
```

The first implementation writes:

- `channels_target.npy`: target-BS channel tensor.
- `ls_target.npy`: LS/OFDM initial estimate for target BS.
- `rss_multibs.npy`: multi-BS RSS/pathloss fingerprint.
- `user_positions.npy`: UE positions.
- `manifest.json`: reproducibility metadata.

## Ablation Plan

Compare these inputs under matched train/val/test splits:

- LS only.
- LS + single-BS RSS.
- LS + multi-BS RSS.
- LS + zero RSS.
- LS + shuffled multi-BS RSS.

Primary stress settings:

- `Np = 32, 64, 128, 256`
- `SNR = -10, -5, 0, 5 dB`

The key criterion is not only final NMSE, but whether wrong/zero/shuffled RSS
causes a measurable drop. If not, the model is probably still behaving as a
black-box LS refiner.


已经完成了 DeepMIMO 多基站 RSS baseline 的第一轮完整实验，数据设置是：

场景：o1_60
用户数：5000
目标 BS：TX 10
多 BS RSS：TX 10 / 11 / 12
LS-like 输入：目标信道加复高斯噪声，SNR = 0 dB
训练：五组 baseline，每组 100 epochs
结果如下：

模式	Test NMSE	Test dB
ls_only	0.418038	-3.79 dB
single_rss	0.432860	-3.64 dB
multibs_rss	0.419044	-3.78 dB
zero_rss	0.410742	-3.86 dB
shuffled_rss	0.434593	-3.62 dB
目前最重要的观察是：

multibs_rss 和 ls_only 几乎一样，没有明显优于 LS-only。
zero_rss 反而略好，这说明当前模型主要还是在做 LS-like 输入的黑盒去噪。
shuffled_rss 最差，说明 RSS 并不是完全没信息，但当前简单 MLP 融合方式没有稳定把 RSS 信息转化成增益。
这个结论和第一阶段 ablation 一致：当前“RSS/物理信息”不是稳定有效的贡献项，不能过度宣称 RSS 已经提升性能。
目前可以写成阶段性结论：
在 SNR=0 dB、5000 用户、轻量 MLP 融合设置下，naive multi-BS RSS fusion 没有带来明确 NMSE 增益。