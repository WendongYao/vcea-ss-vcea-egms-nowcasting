# Final Aggregation Decision

数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。

- final VCEA config = `StaticVal_inverse_temp4p0_sh0p1_floor0p0`。
- audited COEA config = `COEA_ewa_eta4p0_warm_floor50_tile_horizon`。
- final verdict = `Strong-Baseline-Aware Expert Aggregation`。

## Comparison

- `VCEA`: Mean Skill_MSE=0.1776, gain vs M1=0.0993, better tiles=6/6, stress ratio=1.0463, std=0.0105.
- `COEA`: Mean Skill_MSE=0.1322, gain vs M1=0.0539, better tiles=6/6, stress ratio=0.9960, std=0.0207.
- `CRCF-COEA`: Mean Skill_MSE=0.1273, gain vs M1=0.0490, better tiles=6/6, stress ratio=1.0023, std=0.0217.
- `M1`: Mean Skill_MSE=0.0783, gain vs M1=-0.0000, better tiles=4/6, stress ratio=1.0023, std=0.0207.
- `Persistence`: Mean Skill_MSE=0.0000, gain vs M1=-0.0783, better tiles=0/6, stress ratio=1.0000, std=0.0000.

## Verdict rationale

- StaticVal 与 COEA 差距有限，建议将论文主线冻结为同一 aggregation family，同时报告 static 和 online 变体。
