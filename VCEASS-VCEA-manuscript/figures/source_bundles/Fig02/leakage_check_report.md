# Leakage-safe Pipeline Check

- 辅助特征 velocity / acceleration / seasonal amplitude 统一由各 tile 训练窗口拟合得到，用于模型输入。
- 像元级 regime typing 为支持 E6/E8 的诊断分析，采用 full-series coarse grid descriptor 计算，不进入模型输入。
- 时序切分采用 70/10/20 chronological split；窗口按目标时刻归属 split，输入窗口允许跨 split 边界但不允许访问目标之后的时间帧。
- 模型训练与评估统一使用 valid-mask 加权指标；空单元采用最近邻补值，仅用于密集张量构造，不用于 valid-pixel 指标计数。