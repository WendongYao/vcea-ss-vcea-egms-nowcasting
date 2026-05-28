# Methods Static/Online Aggregation Draft

## Expert set

- `E = {Persistence, Linear-Seasonal, M1, C3_opt}`。

## Validation-Calibrated Expert Aggregation (VCEA)

```text
y_hat_(t+h) = sum_k w_(k,h,tile)^val * y_hat_k(t+h)
```

- 本轮最终 static 配置：`StaticVal_inverse_temp4p0_sh0p1_floor0p0`。
- `w^val` 仅由 validation split 估计，并固定迁移到 test。

## Causal Online Expert Aggregation (COEA)

```text
w_(k,t+1,h) ∝ w_(k,t,h) * exp(-eta * L_(k,t,h))
```

- delayed feedback 约束：只有 `target_frame <= current_frame` 时才能更新。
- audited COEA 配置：`COEA_ewa_eta4p0_warm_floor50_tile_horizon`。

## Relationship

- 最终论文建议将 VCEA 与 COEA 视为同一 expert aggregation family 的 static / online 两个变体。
- 由于当前环境没有 `python-docx` 依赖，本草稿以 Markdown 形式落盘。
