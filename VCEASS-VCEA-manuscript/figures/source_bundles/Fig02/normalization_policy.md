# Normalization Policy

- In-tile：使用同 tile 训练窗口的位移均值/标准差与静态特征通道统计量。
- Cross-site：zero-shot / transfer 使用 source tile 的归一化统计量，但目标 tile 的辅助特征仍按目标历史时间序列计算。
- Local calibration：当前实现限制的是监督微调窗口数量，而不是预先可得的目标历史位移长度；因此 E5 中目标辅助特征与归一化统计仍基于目标 pre-test history window 计算。