# V7 Final Model Definition

最终冻结名称：**COEA: Causal Online Expert Aggregation**。

建议正文简称：**COEA**。

## Expert set

- `E = {Persistence, Linear-Seasonal, M1, C3_opt}`。

## Layer 1: Causal online expert aggregation

对每个 `tile × horizon` 维护一组在线权重：

```text
ŷ_(t+h) = Σ_k w_(k,t,h) · ŷ^(k)_(t+h)
```

本轮最终采用的在线更新为 EWA，相对 persistence 归一化损失：

```text
L_(k,t,h) = MSE(ŷ^(k)_(t+h), y_(t+h)) / max(MSE(ŷ^pers_(t+h), y_(t+h)), ε)
w_(k,t+1,h) ∝ w_(k,t,h) · exp(-η · L_(k,t,h))
```

- 最终配置：`update=ewa`, `eta=4.0`, `warm_start=True`, `m1_floor=0.5`, `granularity=tile_horizon`。
- warm start 仅来自 validation replay 的末端权重，不读取 test truth。

## Optional deployment wrapper: CRCF risk fallback

```text
ŷ^final_(t+h) = {
  ŷ^COEA_(t+h),            if r_(t,h) <= τ_h
  ŷ^M1_(t+h),              if r_(t,h) >  τ_h
}
```

其中风险分数定义为：

```text
r_(t,h) = |ŷ^COEA_(t+h) - ŷ^M1_(t+h)| / max(s_(t,h), 0.1)
```

这里 `s_(t,h)` 取 `max(C3 local scale, residual_std)`；阈值 `τ_h` 只能由 validation calibration split 标定。

## 解释

- COEA 负责在 `Persistence / Linear-Seasonal / M1 / C3_opt` 之间做因果在线聚合。
- CRCF 是可选部署层：在高风险区域把输出回退到 `M1`。但在 audited full-six 中，`CRCF` 的平均 skill 略低于 plain `COEA`，因此正文主模型优先冻结为 `COEA` 本体。

## 论文写法建议

- 标题友好名称：**Causal Online Expert Aggregation for EGMS Ground-Deformation Nowcasting**。
- 若需要保留 CRCF，可在 supplementary 或 deployment discussion 中写成：validation-calibrated risk fallback wrapper over COEA。

