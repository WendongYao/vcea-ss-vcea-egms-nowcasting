# V7 Finalization Report

数据声明：本轮 finalization 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。

## Action 1: Causal replay audit

- 审计文档：`${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/v7_causal_replay_audit.md`。
- 原始 V7 quick 存在 search-test tuning；本轮已用 validation-only nested rerun 修正。

## Action 2: Final model freeze

- 基于 audited full-six 的 V7 family 胜出者：`COEA_ewa_eta4p0_warm_floor50_tile_horizon`。
- 因为 `COEA` 在 audited full-six 中优于 `CRCF` 包装，所以正文主模型先冻结为 `COEA`，`CRCF` 作为可选部署安全包装保留。
- 数学定义文档：`${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/v7_final_model_definition.md`。

## Action 3: Minimal ablation

- `Ablation_static_val_weighted`: Mean Skill_MSE=0.1478, gain vs M1=0.0695, stress ratio=0.9965.
- `COEA_audited`: Mean Skill_MSE=0.1322, gain vs M1=0.0539, stress ratio=0.9960.
- `CRCF_COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.1273, gain vs M1=0.0490, stress ratio=1.0023.
- `Ablation_coea_floor_probe`: Mean Skill_MSE=0.1164, gain vs M1=0.0381, stress ratio=0.9925.
- `Ablation_coea_no_linear`: Mean Skill_MSE=0.1145, gain vs M1=0.0361, stress ratio=0.9960.
- `Ablation_coea_no_c3`: Mean Skill_MSE=0.1127, gain vs M1=0.0343, stress ratio=1.0004.
- `Ablation_coea_tile_only`: Mean Skill_MSE=0.1087, gain vs M1=0.0304, stress ratio=0.9955.
- `Ablation_coea_no_warm`: Mean Skill_MSE=0.0981, gain vs M1=0.0197, stress ratio=1.0977.
- `M1`: Mean Skill_MSE=0.0783, gain vs M1=-0.0000, stress ratio=1.0023.
- `CRCFpers_COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.0751, gain vs M1=-0.0032, stress ratio=1.0000.
- `Persistence`: Mean Skill_MSE=0.0000, gain vs M1=-0.0783, stress ratio=1.0000.
- `Ablation_equal_weight`: Mean Skill_MSE=-1.8055, gain vs M1=-1.8838, stress ratio=4.1122.
- `Linear-Seasonal`: Mean Skill_MSE=-36.1495, gain vs M1=-36.2278, stress ratio=16.8447.

## Important caveat

- `Ablation_static_val_weighted` 的 Mean Skill_MSE=0.1478，高于 `COEA`。这说明当前数据上“online update 必要性”没有被完全证明；如果坚持使用 `COEA` 作为主模型，supplementary 必须如实报告这个更强的简单基线。

## Action 4: Interpretability figures

- Figure 1: `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/finalization/v7_figure1_expert_weight_trajectories.png`。
- Figure 2: `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/finalization/v7_figure2_expert_share_by_horizon_regime.png`。
- Figure 3: `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/finalization/v7_figure3_crcf_fallback_rate.png`。
- Figure 4: `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/finalization/v7_figure4_skill_heatmaps.png`。
- Figure 5: `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v7/finalization/v7_figure5_case_maps.png`。

## Audited quick summary

- COEA `COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.1548, gain vs M1=0.0799, better tiles=3/3, stress ratio=1.0366, gate=False.
- CRCF `CRCF_COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.1539, gain vs M1=0.0790, better tiles=2/3, stress ratio=1.0374, gate=False.

## Audited full-six summary

- COEA `COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.1322, gain vs M1=0.0539, better tiles=6/6, stress ratio=0.9960, std=0.0207, Practical GO=True.
- CRCF `CRCF_COEA_ewa_eta4p0_warm_floor50_tile_horizon`: Mean Skill_MSE=0.1273, gain vs M1=0.0490, better tiles=6/6, stress ratio=1.0023, std=0.0217, Practical GO=True.

## Final verdict

- 修正后的 audited pipeline 支持 `COEA` 作为 V7 family 的主候选；`CRCF` 更适合作为可选 safety wrapper，而不是默认主模型。
