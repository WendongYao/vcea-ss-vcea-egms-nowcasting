# ACT-Former v2 Go/No-Go 报告

数据声明：本轮 ACT-Former v2 改进实验继续统一使用 **EGMS 2019-2023 Ortho U**。这是因为当前数据获取限制下，仅该时间段相关 tile 已完成本地载入与处理。

## GO 门槛

- 三个搜索 tile：`E32N34 / E39N30 / E48N24`。
- 四个 horizon：`t+1/t+3/t+5/t+10`。
- 核心指标：`Skill_MSE = 1 - MSE_model / MSE_persistence`。
- 一轮 quick search 的最低目标：平均 `Skill_MSE > 0`，并且 `E32N34/E39N30` 多数长 horizon 为正 skill，`E48N24` 不比 persistence 差超过约 5%。
- 最终 GO 需要 3 seeds 稳定复现，且正 skill 不能完全被 std 覆盖。

## Master Skill Table

- 文件：`${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/master_skill_table.csv`
- 当前包含行数：48。

## Quick Search 结果

### M1

- 平均 Skill_MSE = 0.0473
- E48N24 最差 RMSE / persistence = 1.0014
- E32N34 正 skill 长 horizon 数 = 3
- E39N30 正 skill 长 horizon 数 = 3
- 是否通过单 seed GO：`是`
- 结果文件：`${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/model_search_m1.csv`

## 3-Seed 复验

- 选中模型：`M1`
- 平均 Skill_MSE = 0.0446
- 平均 Skill_MSE std = 0.0059
- E48N24 最差 RMSE / persistence = 1.0018
- E32N34 稳定正 skill 长 horizon 数 = 3
- E39N30 稳定正 skill 长 horizon 数 = 3
- 最终是否达到 GO：`是`
- 汇总文件：`${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/final_model_selection.csv`

## 停止原因

- `M1` 在 3 seeds 复验后达到 GO 门槛：平均 Skill_MSE=0.0446。

## 输出文件

- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/master_skill_table.csv`
- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/model_search_m0.csv`
- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/model_search_m1.csv`
- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/gate_weight_summary.csv`
- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/final_model_selection.csv`
- `${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/run_summary.json`

## 假设与说明

- 为避免未来信息泄漏，gate descriptor 仅使用训练窗口可得的 `velocity / amplitude / trend_R² / acceleration`，没有把 full-series `change_score` 直接输入 gate。
- `M3` 的 temporal patch 采用可执行近似实现：把更长历史切成 temporal groups 后再做空间 patch 编码，以满足长历史/局部时间片建模目标。
- 若某个候选模型在单 seed 上已明显不满足 GO 约束，则继续推进到下一个计划模型，而不做无意义的 3-seed 扩展。