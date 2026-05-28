# VCEA / SS-VCEA Protocol Audit Report

Audit date: 2026-05-28  
Scope: repository inspection only; no manuscript edits, no metric recomputation, no retraining, no figure/table regeneration.

## 1. Executive summary

### Safe claims

- The project is consistently built on **EGMS 2019--2023 Ortho U**; the strong-audit manifest, split files, and product manifests all point to that product family.
- The **primary main-six benchmark** supports the current headline that VCEA improves over M1:
  - `VCEA = 0.17763291690164576`
  - `M1 = 0.0783341224701473`
  - source: `outputs/tgrs_2019_2023/staticval_ea_study/final_aggregation_comparison.csv`
- The **14-tile geographically stratified pan-European frozen audit** supports the current broad result summary:
  - `VCEA mean Skill_MSE = 0.2601067725993061`
  - `M1 mean Skill_MSE = 0.11275046554026745`
  - `VCEA mean RMSE = 0.8659666717229856 mm`
  - `M1 mean RMSE = 1.0665808097415028 mm`
  - `VCEA better tiles vs M1 = 14`
  - source: `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_group_summary.csv`
- **SS-VCEA q97.5** is validation-calibrated and held fixed for test/external evaluation:
  - `SS-VCEA-COEA_q97.5 mean Skill_MSE = 0.2292810678048671`
  - `stress ratio = 0.9973025376578523`
  - source: same strong E7 group summary table
- The **COEA delayed-feedback implementation** appears causally correct: target labels are only used after the corresponding target frame becomes observable.

### Claims requiring wording changes

- The manuscript statement that original external tiles and 14-tile audit tiles are **"not used for ... aggregation calibration"** is **not technically accurate**.
  - The code and manifests show **same-tile historical validation calibration** for VCEA weights on:
    - six main development tiles,
    - original four external locked tiles,
    - strong 14-tile audit tiles.
- Any wording implying:
  - `calibration-free transfer`,
  - `zero-shot transfer`,
  - `held-out tiles completely unseen until test`,
  - `no audit-tile labels used in aggregation calibration`
  should be avoided.
- "Cross-tile robustness" / "transfer" wording is still usable, but only if bounded as:
  - **held-out tiles with tile-local historical validation calibration and frozen future evaluation**.

### Claims that cannot be fully verified from available files

- The strong 14-tile selection is clearly described as **post hoc metadata-only**, but the repository does **not** contain a dedicated:
  - `selection_tile_pool.csv`
  - `selection_notes.md`
  for the strong audit.
- Because those two files are missing, the metadata-only selection rationale is well supported but not maximally documented.

### High-risk leakage concerns

- **No direct future-label leakage** was found in:
  - VCEA weighting,
  - SS-VCEA threshold calibration,
  - COEA delayed feedback,
  - strong 14-tile audit evaluation.
- The **high-risk issue is wording, not code**:
  - held-out external/audit tiles are not "calibration-free";
  - they **do use their own historical validation labels** for aggregation calibration.

## 2. Calibration protocol verdict

### Verdict summary

- **Main-six**: A. within-tile historical validation calibration + future frozen test
- **Original external four tiles**: A. within-tile historical validation calibration + future frozen test
- **14-tile strong audit**: A. within-tile historical validation calibration + future frozen test

Confidence: **high**

### B1. Calibration source audit

| Evaluation set | Tile list | Calibration source | Uses same-tile validation labels? | Uses future test labels? | Uses external/audit labels for tuning? | Correct manuscript wording |
|---|---|---|---|---|---|---|
| Primary main-six benchmark | `E32N34, E32N35, E39N30, E44N23, E48N24, E58N17` | Per-tile validation cache on the same tile and seed; VCEA weights estimated from validation loss, then applied to future test windows | Yes | No | N/A | historical validation calibration followed by frozen future evaluation |
| Original four external locked tiles | `E30N35, E37N42, E42N21, E45N51` | Per-tile historical validation cache on each external tile; frozen expert pool; same-tile validation calibration for aggregation weights | Yes | No | Yes, for aggregation calibration only on historical validation segments; no use for expert search or threshold selection | held-out tiles with tile-local historical validation calibration and frozen future evaluation |
| Geographically stratified 14-tile pan-European frozen audit | `E39N23, E48N26, E40N31, E44N30, E48N32, E31N16, E55N14, E49N22, E57N25, E58N26, E44N43, E49N44, E27N24, E34N39` | Per-tile historical validation cache on each selected held-out tile; frozen expert pool/config/thresholds; same-tile validation calibration for aggregation weights | Yes | No | Yes, for aggregation calibration only on historical validation segments; no use for expert search or threshold selection | held-out tiles with tile-local historical validation calibration and frozen future evaluation |

### Evidence

- `tgrs_experiments/vcea_extra_experiments.py`
  - `run_e1_leakage_audit`: lines around `490-497`
  - `val_cache = ... "val"`
  - `test_cache = ... "test"`
  - `weights = estimate_static_weights(val_cache, FINAL_VCEA_CONFIG)`
  - `vcea_pred = predict_static(test_cache, weights)`
- `tgrs_experiments/vcea_extra_experiments.py`
  - `run_e2_expert_pool_ablation`: main validation/test caches created at lines `657-658`
  - `run_e3_calibration_ablation`: main validation/test caches created at lines `793-794`
- `tgrs_experiments/staticval_ea_study.py`
  - `estimate_static_weights(...)`: line `216`
  - validation loss computed from `cache["y_true_mm"]` vs expert predictions
- `splits/external4_locked_eval.json`
  - `"locked_after_mainline_freeze": true`
  - `"used_for_architecture_selection": false`
  - `"Historical exact VCEA external runs used frozen expert definitions with tile-local validation calibration, not architecture search."`
- `outputs/tgrs_2019_2023/confirmation/external_tiles/freeze_manifest.yaml`
  - `"external_eval_notes: exact VCEA uses frozen four-expert static validation weighting on each external tile"`
- `tgrs_experiments/vcea_e7_pan_europe_audit.py`
  - external exact evaluation uses `val_cache` and `test_cache` for each external tile
- `tgrs_experiments/vcea_e7_pan_europe_audit_strong.py`
  - `build_predictions(...)`: lines `511-515`
  - `val_cache = build_external_cache_from_runs(..., "val")`
  - `test_cache = build_external_cache_from_runs(..., "test")`
  - `vcea_pred, _weights = static_prediction_from_external(val_cache, test_cache)`

### Implication for manuscript wording

- Do **not** claim that external/audit tiles were completely unseen until test evaluation.
- Do **not** claim that external/audit tiles were excluded from aggregation calibration.
- Safe wording:
  - **"historical validation calibration followed by frozen future evaluation"**
  - **"held-out tiles with tile-local historical validation calibration and frozen future evaluation"**

## 3. Frozen audit verdict

Confidence: **high** for frozen expert/config/threshold/metric logic; **medium** for selection documentation completeness because two expected selection-support files are missing.

### B2. Strong E7 / 14-tile audit protocol

| Audit property | Verified? | Evidence path | Notes | Manuscript risk |
|---|---|---|---|---|
| Expert pool frozen before strong audit | Yes | `outputs/tgrs_2019_2023/vcea_extra_experiments/E7_pan_europe_locked_audit_strong/manifests/audit_E7_manifest.json` | `"expert_pool": ["Persistence","Linear-Seasonal","M1","C3_opt"]` | Low |
| VCEA calibration logic frozen before strong audit | Yes | same manifest | `"model_id": "StaticVal_inverse_temp4p0_sh0p1_floor0p0"` | Low |
| SS-VCEA q97.5 threshold frozen before strong audit | Yes | same manifest | `"ss_percentile": 97.5`, `"ss_thresholds": [...]` | Low |
| Preprocessing fixed before strong audit | Yes | same manifest + reuse of existing bundle/cache loaders | Uses pre-existing processing pipeline and bundle logic | Low |
| Metrics fixed before strong audit | Yes | strong audit result tables + shared evaluation code | Same metrics emitted into strong audit tables | Low |
| Tile selection metadata-only | Yes | same manifest + tile metadata/prefetch tables | `"chosen using only official API tile catalog metadata"` | Low |
| E7 labels used for selecting experts | No | manifest + frozen expert pool | Expert pool fixed before audit | Low |
| E7 labels used for selecting q97.5 | No | threshold calibration code + threshold files | Thresholds from main-six validation only | Low |
| E7 labels used for threshold tuning | No | `tgrs_experiments/tgrs_vcea_hardening.py`, `threshold_diag_main6_validation.csv` | No recomputation on strong E7 | Low |
| 14 tiles selected after seeing metric outcomes | No evidence found | manifest + tile metadata table | Selection described as post hoc metadata-only, not metric-driven | Medium because dedicated `selection_notes.md` / `selection_tile_pool.csv` are missing |
| 14 tiles preselected before any evaluation | Not strictly documented | manifest | Best supported description is **post hoc metadata-only after model/calibration/threshold freeze** | Medium |

### Evidence

- `outputs/tgrs_2019_2023/vcea_extra_experiments/E7_pan_europe_locked_audit_strong/manifests/audit_E7_manifest.json`
  - `"selection_policy": "post_hoc selected external locked audit; fixed after model/calibration/threshold freeze; chosen using only official API tile catalog metadata"`
  - `"selected_tile_count": 14`
  - `"frozen_objects": { ... }`
- `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_tile_metadata.csv`
- `outputs/tgrs_2019_2023/vcea_extra_experiments/E7_pan_europe_locked_audit_strong/manifests/table_E7_selected_tiles_prefetch.csv`

### Implication for manuscript wording

Safe wording:

- **"metadata-stratified 14-tile pan-European frozen audit"**
- **"finite, metadata-stratified audit set"**
- **"not exhaustive validation over all European deformation conditions"**

Avoid:

- "population-level European validation"
- "validated across Europe"
- "calibration-free transfer"

## 4. Original four external locked tiles verdict

Confidence: **high**

### B3. Original four external locked tiles

| External set | Tile IDs | Used for expert search? | Used for threshold selection? | Used for reporting only? | Evidence |
|---|---|---|---|---|---|
| Original frozen external locked evaluation | `E30N35, E37N42, E42N21, E45N51` | No | No | No; used for frozen external evaluation with tile-local historical validation calibration | `splits/external4_locked_eval.json`, `outputs/tgrs_2019_2023/confirmation/external_tiles/freeze_manifest.yaml`, `tgrs_experiments/vcea_e7_pan_europe_audit.py`, `outputs/tgrs_2019_2023/confirmation/external_tiles/external_summary.csv` |

### Key quoted fields

- `splits/external4_locked_eval.json`
  - `"locked_after_mainline_freeze": true`
  - `"used_for_architecture_selection": false`
  - `"Historical exact VCEA external runs used frozen expert definitions with tile-local validation calibration, not architecture search."`
- `outputs/tgrs_2019_2023/confirmation/external_tiles/freeze_manifest.yaml`
  - `"exact VCEA uses frozen four-expert static validation weighting on each external tile"`

### Implication for manuscript wording

- External four-tile evaluation is **not** reporting-only.
- It is a valid frozen external check, but **not** a calibration-free check.
- Safe wording:
  - **"original four-tile frozen external evaluation with tile-local historical validation calibration and frozen future test evaluation"**

## 5. SS-VCEA threshold and stress-ratio verdict

### B4. SS-VCEA q97.5 threshold source

Confidence: **high**

| Threshold item | Value / protocol | Evidence path | Risk of leakage? | Correct manuscript wording |
|---|---|---|---|---|
| Safe target for selected SS-VCEA variant | `COEA` | `outputs/tgrs_2019_2023/hardening_package/ss_vcea/ss_vcea_selected_external_summary.csv`, `ss_vcea_threshold_sweep.csv`, strong-audit manifest | Low | SS-VCEA q97.5 uses COEA as the frozen safe target |
| Percentile source | `97.5th percentile` of validation risk | `tgrs_experiments/tgrs_vcea_hardening.py`, `paper_assets_vcea_extras/tables_csv_original/E7_strong/threshold_diag_main6_validation.csv` | Low | q97.5 is selected from the validation risk distribution and then held fixed |
| Calibration split | main-six validation only | `tgrs_experiments/tgrs_vcea_hardening.py` | Low | main-six validation risk distribution |
| Pooling granularity | per horizon, pooled across main-six validation caches | same code + threshold CSV | Low | per-horizon pooled main-six validation risk distribution |
| Recomputed on original external tiles? | No | threshold code + external eval scripts | Low | thresholds held fixed on external evaluation |
| Recomputed on 14-tile strong audit? | No | strong audit manifest + strong audit script | Low | thresholds held fixed on strong audit |
| Selected because of external/test performance? | No evidence found | threshold sweep files + hardening code | Low | selected from validation-only sweep as preferred conservative operating point |
| Selected because it maximizes validation skill? | No | `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E6_ssvcea_threshold_sweep.csv` shows `q99` slightly higher skill than `q97p5` | Low | preferred conservative operating point, not highest-skill point |

### Exact threshold values found

Source: `outputs/tgrs_2019_2023/hardening_package/ss_vcea/ss_vcea_thresholds.csv`

- `COEA,97.5,t+1,0.7517598271369934`
- `COEA,97.5,t+3,0.8297779560089111`
- `COEA,97.5,t+5,0.8662713170051575`
- `COEA,97.5,t+10,0.9856411814689636`

### Supporting selection-sweep evidence

Source: `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E6_ssvcea_threshold_sweep.csv`

- `q97p5, Skill MSE = 0.1776167232477429, Stress ratio = 0.9916542573534443`
- `q99, Skill MSE = 0.17874124506204678, Stress ratio = 0.9967612822280874`

### Implication for manuscript wording

Safe wording:

- **"q97.5 is selected from the pooled main-six validation risk distribution on a per-horizon basis and then held fixed for test and external audit evaluation."**
- **"q97.5 is the preferred conservative operating point, not simply the highest-skill threshold."**

## 6. COEA delayed-feedback verdict

Confidence: **high**

### B5. COEA delayed-feedback protocol

| COEA component | Verified behavior | Evidence path | Leakage risk | Manuscript wording |
|---|---|---|---|---|
| Update schedule | COEA maintains pending outcomes per horizon and only releases them when the target frame becomes observable | `tgrs_experiments/act_v7.py` | Low | delayed feedback is implemented causally |
| Forecast at `t+h` uses label `y_{t+h}`? | No | same file | Low | no target label is used before forecast production |
| Feedback delay implemented correctly? | Yes; pending outcomes update only when `target_frame <= current_frame` | same file, lines around `327-337` | Low | safe to describe as simulated release schedule |
| Test labels available before prediction? | No evidence found | same file | Low | no pre-update on target label |
| Warm start source | validation cache only | same file, `warm_start_weights(...)` | Low | validation-only warm start |
| SS-VCEA fallback to COEA leaks future labels? | No evidence found | `tgrs_experiments/vcea_e7_pan_europe_audit_strong.py` + `act_v7.py` | Low | safe fallback wording supported |

### Key code evidence

- `tgrs_experiments/act_v7.py`
  - `replay_coea(...)`: line `290`
  - `pending` queue: line `320`
  - delayed update condition: `if target_frame <= current_frame:` at line `328`
  - `warm_start_weights(...)`: line `376`

### Recommended wording

- **"COEA updates are allowed only after the corresponding target frame becomes observable under the simulated release schedule; no target label is used before producing the forecast for that target."**

## 7. Leakage and split verdict

Confidence: **high**

### B6. Split protocol and leakage audit

| Protocol | Split rule | History crosses boundary? | Future target used? | Descriptor/scaler fit on | Evidence | Manuscript wording |
|---|---|---|---|---|---|---|
| Main online protocol (target-time assignment) | Sample assigned by target frame; history may reach backward across a boundary as long as target belongs to later split | Yes | No | train split only | `tgrs_experiments/pipeline.py`, `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E1_leakage_audit.csv`, main manuscript protocol text | target-time assignment under a leakage-safe online updating interpretation |
| Full-window containment audit | Entire input-target window must lie inside the split | No | No | recalibrated on stricter val subset; no retraining | `table_E1_leakage_audit.csv`, `tgrs_experiments/vcea_extra_experiments.py` | stricter split-protocol audit |
| 10-frame gap-buffer audit | Full-window containment plus a 10-frame exclusion buffer around split boundaries | No | No | recalibrated on stricter val subset; no retraining | same as above | gap-buffer split-protocol audit |
| Descriptor fitting | Seasonal/trend descriptors fit on training portion for main pipelines | N/A | No | training portion | `tgrs_experiments/pipeline.py`, `fit_descriptors(...)` | training-only descriptor fitting |
| Scaler/normalization | `disp_mu` / `disp_sigma` estimated on training portion | N/A | No | training portion | `tgrs_experiments/pipeline.py`, lines around `504-505`, `671-672` | train-only normalization |
| Missing-value fill | nearest-neighbor fill inside raster construction; valid masks retained for evaluation | N/A | No | not label-informed | `tgrs_experiments/pipeline.py`, `NearestNDInterpolator`, `fill_nearest(...)` | nearest-neighbor fill with mask-aware evaluation |

### Key evidence

- `tgrs_experiments/pipeline.py`
  - `build_split(...)`: line `240`
  - `fit_descriptors(...)`: line `254`
  - `fill_nearest(...)`: line `343`
  - train-only normalization stats: lines around `504-505`, `671-672`
- `data/processed/tgrs_2019_2023/E32N34/split.json`
- `splits/main6_train_val_test.json`
- `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E1_leakage_audit.csv`
- `tgrs_experiments/vcea_extra_experiments.py`
  - strong note: stricter leakage audit recalibrates fixed experts under stricter window filters rather than retraining all neural experts

### Implication for manuscript wording

Safe wording:

- **"Inputs may cross a split boundary under the target-time assignment protocol, but no future target frame is used before it becomes available."**
- **"Full-window containment and 10-frame gap-buffer audits provide stricter leakage-control alternatives."**
- **"These stricter audits recalibrate the aggregation stage on stricter validation subsets rather than retraining the underlying neural experts."**

## 8. Expert pool and baseline consistency verdict

Confidence: **high** for main-six/external/strong exact packages; **high** that 0.178 / 0.171 / 0.150 belong to different packages and must stay scoped.

### B7. Expert pool and baseline consistency

| Model/expert | Main-six | Original external | Strong E7 | Exact/proxy/transfer? | Evidence | Manuscript risk |
|---|---|---|---|---|---|---|
| Persistence | Yes | Yes | Yes | exact | `configs/infra_v2/data_config.yaml`, cache registry, result tables | Low |
| Linear-Seasonal | Yes | Yes | Yes | exact | same | Low |
| M1 | Yes | Yes | Yes | exact | same + exact/proxy flags | Low |
| C3_opt | Yes | Yes | Yes | exact at 64 | same + exact/proxy flags | Low |
| COEA_exact | Yes | Yes | Yes | exact aggregation baseline | exact/proxy flags + result tables | Low |
| VCEA_exact | Yes | Yes | Yes | exact at 64; 512 sensitivity uses transfer inference/cache reuse | exact/proxy flags, resolution table | Medium if high-resolution sensitivity is not properly scoped |
| SS-VCEA q97.5 | Yes | Yes | Yes | exact conservative wrapper at 64 | threshold sweep, selected summaries, strong audit tables | Low |
| StaticVal-PLM1 | resolution sensitivity only | No | No | proxy | exact/proxy flags | Low if kept out of primary claims |

### Exact scope audit for the three important VCEA numbers

| VCEA number | Exact value found | Package / scope | Evidence | Correct scope wording |
|---|---|---|---|---|
| Primary main-six VCEA | `0.17763291690164576` | primary exact main-six benchmark | `outputs/tgrs_2019_2023/staticval_ea_study/final_aggregation_comparison.csv` | primary main-six claim |
| Matched safety-oriented reference VCEA | `0.170500956671233` | matched safety-oriented recomputation used to compare VCEA/SS-VCEA under same package | `outputs/tgrs_2019_2023/hardening_package/ss_vcea/ss_vcea_threshold_sweep.csv` | matched safety-oriented reference, not primary claim |
| Resolution sensitivity VCEA exact 64 | `0.1504403447771985` | resolution-aware sensitivity package | `outputs/tgrs_2019_2023/hardening_package/resolution_aware/resolution_exact_proxy_table.csv` | sensitivity evidence only, not a replacement for primary main-six claim |

### Supporting quoted fields

- `configs/infra_v2/data_config.yaml`
  - four-expert pool includes persistence, linear-seasonal, M1, C3_opt
- `outputs/tgrs_2019_2023/hardening_package/reproducibility/exact_proxy_transfer_flags.csv`
  - `VCEA_exact,external_locked_eval,64,False,False,...`
  - `VCEA_exact,main6_aggregate,64,False,False,...`
  - `VCEA_exact,resolution_sensitivity,512,False,True,...`
  - `StaticVal-PLM1,...,is_proxy=True`

### Implication for manuscript wording

- The current three-number scope logic is technically supportable **if** the scope is always explicit.
- Do not let `0.171` or `0.150` appear as if they were substitutes for the primary `0.178`.

## 9. Strong E7 result integrity check

Confidence: **high**

### B8. Strong E7 result integrity

| Claim | Supported? | Exact value found | Source table/path | Notes |
|---|---|---|---|---|
| 14 held-out tiles | Yes | `14` | `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_group_summary.csv` | overall rows use 14 tiles |
| 56 tile-horizon evaluations | Yes | `56` | same table | overall rows use 56 rows |
| VCEA mean Skill_MSE | Yes | `0.2601067725993061` | same table | rounds to 0.260 |
| M1 mean Skill_MSE | Yes | `0.11275046554026745` | same table | rounds to 0.113 |
| VCEA mean RMSE [mm] | Yes | `0.8659666717229856` | same table | rounds to 0.866 |
| M1 mean RMSE [mm] | Yes | `1.0665808097415028` | same table | rounds to 1.067 |
| VCEA better on 14/14 held-out tiles | Yes | `14` | same table | `Better tiles vs M1 = 14` |
| SS-VCEA q97.5 mean Skill_MSE | Yes | `0.2292810678048671` | same table | rounds to 0.229 |
| SS-VCEA q97.5 stress ratio | Yes | `0.9973025376578523` | same table | rounds to 0.997 |
| VCEA stress ratio | Yes | `0.9977127233192455` | same table | rounds to 0.998 |
| Localized E44N43 short-temporal-span issue | Yes | `E44N43, t+10/t+5, SS-VCEA-COEA_q97.5` and `E44N43, t+1, VCEA` | `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_failure_cases.csv` | supported |
| Localized E49N22 low-coverage issue | Yes | `E49N22, t+10, VCEA` | same failure-case table | supported |

### Additional integrity check

Source: `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_locked_results_by_tile.csv`

- Each of the models `M1`, `VCEA`, and `SS-VCEA-COEA_q97.5` has:
  - `14` unique tiles
  - `56` tile-horizon rows

### Implication for manuscript wording

- The current strong-audit numerical summary is supported.
- Safe wording remains:
  - **"positive transfer across the tested broad-region, audit-regime, and coverage groups"**
  - **not** "validated across Europe"

## 10. Statistical significance readiness

Confidence: **high**

### B9. Readiness for significance testing

| Test | Required data | Available? | Source file | Can compute without rerunning models? |
|---|---|---|---|---|
| Tile-level sign test | by-tile paired outcomes for VCEA vs baseline | Yes | `outputs/tgrs_2019_2023/hardening_package/ss_vcea/ss_vcea_selected_tile_horizon.csv`, `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_locked_results_by_tile.csv` | Yes |
| Paired tile-level bootstrap CI | per-tile or per-tile-horizon paired gains | Yes | same files | Yes |
| Per-horizon mean gain and CI | horizon-level rows | Yes | same files + `ss_vcea_main6.csv` | Yes |
| Wilcoxon signed-rank test | paired tile-level or tile-horizon-level gains | Yes | same files | Yes |

### Note

- The repository appears to contain enough row-level material to compute these later, if desired, without rerunning any model.

## 11. Stress ratio / risk score definition verdict

Confidence: **high**

### B10. Stress ratio / risk score definition

| Quantity | Code definition | Manuscript definition | Match? | Evidence | Recommended wording |
|---|---|---|---|---|---|
| Disagreement `d` | `abs(vcea - safe) / scale` | described as disagreement between VCEA and safe prediction | Yes | `tgrs_experiments/tgrs_vcea_hardening.py` lines around `191` | disagreement component of equal-normalized operational risk |
| Correction `c` | `abs(vcea - persistence) / scale` | described as correction magnitude relative to persistence | Yes | same file line `192` | correction component relative to persistence |
| Volatility `v` | normalized local residual variability | described as local volatility component | Yes | same file line `194` | local volatility component |
| Aggregate risk `r` | `(d + c + local_vol) / 3.0` | equal-normalized average of three components | Yes | same file line `195` | equal-normalized operational risk score |
| Thresholding rule | per-horizon validation percentile | described as q97.5 validation threshold | Yes | threshold code + manuscript | validation-calibrated per-horizon threshold |
| Stress ratio | maximum stress-tile RMSE / persistence RMSE in summary function | described as stress / over-correction safety measure | Mostly yes | `tgrs_experiments/tgrs_vcea_hardening.py` line `259`; manuscript line around `581` | operational over-correction / reliability indicator relative to persistence |
| Geophysical hazard meaning | no hazard semantics in code | manuscript mostly uses reliability wording | Mostly yes | code + manuscript | explicitly state this is not a geophysical hazard metric |

### Important nuance

- In code, `stress_ratio` is a **relative operational error measure** against persistence on designated stress subsets; it is **not** a physical hazard score.

## 12. Reproducibility package readiness

Confidence: **medium-high** overall; main ingredients are present, but supplement hygiene still needs cleanup.

### B11. Reproducibility package readiness

| Artifact type | Present? | Path | Missing fields | Ready for anonymous supplement? |
|---|---|---|---|---|
| Main config files | Yes | `configs/infra_v2/data_config.yaml` | none obvious | Mostly yes |
| Product manifest | Yes | `data/manifests/product_manifest.csv` | none obvious | Yes |
| Tile manifest | Yes | `data/manifests/tile_manifest.csv` | none obvious | Yes |
| Resolution manifest | Yes | `data/manifests/resolution_manifest.csv` | none obvious | Yes |
| Split definitions | Yes | `splits/main6_train_val_test.json`, `splits/external4_locked_eval.json`, per-tile `split.json` | none obvious | Yes |
| Random seeds | Yes | `outputs/tgrs_2019_2023/infrastructure_v2/freeze_manifest.yaml` | none obvious | Yes |
| Software versions | Yes | same freeze manifest | no standalone env export found | Mostly yes |
| Hardware / environment info | Partial | same freeze manifest + reproducibility summary | dedicated environment export not found | Mostly yes |
| Expert cache registry | Yes | `cache/expert_cache_registry.yaml` | none obvious | Yes |
| Aggregation weight registry | Yes | `cache/aggregation_weight_registry.yaml` | none obvious | Yes |
| Threshold diagnostics | Yes | `paper_assets_vcea_extras/tables_csv_original/E7_strong/threshold_diag_main6_validation.csv` | none obvious | Yes |
| Result CSVs | Yes | `outputs/...`, `paper_assets_vcea_extras/tables_csv_original/...` | none obvious | Yes |
| Figure generation scripts | Yes | `tgrs_experiments/...` | none obvious | Mostly yes |
| Claim-to-evidence map | Yes | `outputs/tgrs_2019_2023/hardening_package/reproducibility/claim_to_result_map.md` | manuscript-number synchronization may need later checking | Mostly yes |
| Exact/proxy/transfer flags | Yes | `outputs/tgrs_2019_2023/hardening_package/reproducibility/exact_proxy_transfer_flags.csv` | none obvious | Yes |
| Strong-audit selection support docs | Partial | manifest + metadata tables present | `selection_tile_pool.csv`, `selection_notes.md` not found | Medium |

### Raw data redistribution check

- The audited materials appear to **reference raw EGMS files by tile/product metadata and local file path / hash**, not redistribute the raw EGMS data through the manuscript asset bundle.

### Supplement caution

- Many manifests still contain absolute local Windows paths such as:
  - `${PROJECT_ROOT}\...`
- Those paths should be scrubbed before any anonymous supplement release.

## 13. Manuscript wording risk list

Confidence: **high** for the main manuscript file; **medium** for archived support files because some are clearly legacy package outputs rather than active submission source.

### B12. Risky phrase search

Main manuscript inspected: `VCEASS-VCEA-manuscript/bare_jrnl_new_sample4.tex`

| Phrase | Found? | Location | Risk | Suggested replacement |
|---|---|---|---|---|
| validated across Europe | No | main manuscript not found | Low | none needed |
| Europe-wide validation | No | main manuscript not found | Low | none needed |
| exhaustive European validation | No | main manuscript not found | Low | none needed |
| pan-European validation | No exact risky phrase in main manuscript | main manuscript not found | Low | keep "geographically stratified 14-tile pan-European frozen audit" |
| generalizes across Europe | No | main manuscript not found | Low | none needed |
| zero-shot transfer | No | main manuscript not found | Low | none needed |
| calibration-free transfer | No | main manuscript not found | Low | none needed |
| geophysical hazard risk | No | main manuscript not found | Low | none needed |
| stress-safe in physical sense | No | main manuscript not found | Low | keep operational wording |
| Strong E7 | No | main manuscript not found | Low | none needed |
| SkillMSE | No | main manuscript not found | Low | none needed |
| Skill_MSE | No | main manuscript not found | Low | none needed |
| hardening package | No | main manuscript not found | Low | none needed |
| asset package | No | main manuscript not found | Low | none needed |
| final winner | No | main manuscript not found | Low | none needed |
| strongest route | No | main manuscript not found | Low | none needed |

### Additional wording risks found even though the exact banned phrases were mostly removed

| Phrase / sentence | Found? | Location | Risk | Suggested replacement |
|---|---|---|---|---|
| "Test labels, original external tiles, and the 14-tile ... tiles are not used for expert search, aggregation calibration, or threshold selection." | Yes | `bare_jrnl_new_sample4.tex:144` | High; inaccurate for aggregation calibration | say they are not used for expert search or threshold selection, but held-out tiles use tile-local historical validation segments for aggregation calibration |
| "supports cross-tile robustness" | Yes | `bare_jrnl_new_sample4.tex:408` | Medium; acceptable only if bounded | add "under tile-local historical validation calibration and frozen future evaluation" |
| "To test transfer ..." / "positive transfer" | Yes | `bare_jrnl_new_sample4.tex:411, 454, 461, 501, 517` | Medium; safe if not conflated with calibration-free transfer | retain but bound as tested held-out-tile transfer under the audited protocol |
| "External transfer holds on the 14-tile frozen audit" | Yes | `bare_jrnl_new_sample4.tex:1006` | Medium; should not imply calibration-free transfer | rename as held-out audit transfer under frozen protocol |

### Legacy support-file phrase residue outside the main compiled manuscript

| Phrase | Found? | Location | Risk | Suggested replacement |
|---|---|---|---|---|
| Strong E7 | Yes | `paper_assets_vcea_extras/vcea_extra_tables_and_figures.tex` | Low if not compiled; medium if bundled as active source | rename to 14-tile pan-European frozen audit in submission-facing support files |
| asset package | Yes | `tgrs_vcea_ssvcea_figures_tables_draft.tex` | Low if archival only | remove from submission-facing source bundle |
| Skill_MSE | Yes | multiple archival CSV/tex support assets | Low if tables are external data only | keep scoped as raw artifact field names or map to manuscript terminology in supplement |

## 14. Recommended manuscript wording

### VCEA calibration protocol

Recommended safe wording:

- **"VCEA uses tile- and horizon-aware weights estimated from historical validation losses and then frozen for future evaluation."**
- For held-out sets:
  - **"On held-out tiles, the expert pool and calibration rule are frozen, and aggregation weights are calibrated only from the tile's historical validation segment before future frozen test evaluation."**

### Original external four-tile audit

- **"The original four-tile external evaluation is a held-out frozen future test with tile-local historical validation calibration, not a calibration-free transfer test."**

### 14-tile strong audit

- **"The geographically stratified 14-tile pan-European frozen audit evaluates held-out EGMS tiles under frozen expert definitions, frozen calibration logic, frozen thresholds, and tile-local historical validation calibration followed by future frozen evaluation."**
- Boundary sentence:
  - **"This audit provides geographically stratified frozen external evidence over the tested held-out EGMS tiles; it should not be interpreted as exhaustive validation over all European deformation conditions."**

### SS-VCEA threshold

- **"SS-VCEA q97.5 uses per-horizon thresholds derived from the pooled main-six validation risk distribution and then held fixed for test and external audit evaluation."**
- If emphasizing choice:
  - **"q97.5 is the preferred conservative operating point from the validation-only threshold sweep."**

### Stress ratio

- **"Stress ratio is an operational over-correction / reliability indicator defined relative to persistence, not a geophysical hazard metric."**

### Pan-European claim boundary

- **"positive transfer across the tested held-out tiles and metadata strata"**
- **not**:
  - "validated across Europe"
  - "Europe-wide validation"
  - "calibration-free transfer across Europe"

## 15. Action list

### Must fix before submission

1. Correct the manuscript statement at `bare_jrnl_new_sample4.tex:144`:
   - held-out external/audit tiles are **not** excluded from aggregation calibration;
   - they **do** use tile-local historical validation labels for VCEA weight estimation.
2. Ensure all "transfer" language is bounded so it cannot be read as:
   - zero-shot transfer,
   - calibration-free transfer,
   - unseen-until-test aggregation.
3. Keep the three VCEA numbers explicitly scoped:
   - `0.17763291690164576` primary main-six,
   - `0.170500956671233` matched safety-oriented reference,
   - `0.1504403447771985` resolution sensitivity package.
4. If a reproducibility supplement is distributed, scrub absolute local paths from manifests and registry files.

### Should fix if easy

1. Clarify in manuscript prose that the strict split-protocol audit recalibrates aggregation on stricter validation subsets **without retraining the neural experts**.
2. Clarify that q97.5 is the **preferred conservative operating point**, not the highest-skill threshold.
3. Add one short sentence that stress ratio is an **operational reliability / over-correction measure relative to persistence**.
4. If submission includes strong-audit support files, add or regenerate explicit selection-support artifacts for the 14-tile metadata-only choice.

### Optional improvement

1. Add a supplement note summarizing the exact strong-audit selection workflow and missing pool-selection artifacts.
2. Add later significance testing from existing row-level tables:
   - sign test,
   - paired bootstrap CI,
   - Wilcoxon,
   without rerunning any model.

### No action needed

1. Main-six headline result (`0.17763291690164576` vs `0.0783341224701473`) is supported.
2. Strong 14-tile audit headline result (`0.2601067725993061` vs `0.11275046554026745`, RMSE `0.8659666717229856` vs `1.0665808097415028`) is supported.
3. SS-VCEA strong-audit summary (`0.2292810678048671`, stress ratio `0.9973025376578523`) is supported.
4. COEA delayed-feedback wording is supported by the current implementation.
5. Current non-Europe-wide claim boundary is mostly already in good shape.

## 16. Files explicitly inspected

Primary inspected set used for this audit: **55 files** across:

- manuscript source,
- split definitions,
- manifests,
- core experiment scripts,
- strong-audit scripts and manifests,
- threshold diagnostics,
- strong-audit result tables,
- reproducibility registries,
- exact/proxy/transfer tracking files.

Missing artifacts specifically looked for and not found:

- `outputs/tgrs_2019_2023/vcea_extra_experiments/E7_pan_europe_locked_audit_strong/manifests/selection_tile_pool.csv`
- `outputs/tgrs_2019_2023/vcea_extra_experiments/E7_pan_europe_locked_audit_strong/manifests/selection_notes.md`
- `outputs/tgrs_2019_2023/staticval_ea_study/paper_artifact_manifest.csv`

