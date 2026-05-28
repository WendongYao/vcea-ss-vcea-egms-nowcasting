# Claim-to-Result Map

| Claim | Supporting file | Figure/Table | Notes |
| --- | --- | --- | --- |
| VCEA outperforms M1 on the primary main-six benchmark | `outputs/tgrs_2019_2023/staticval_ea_study/final_aggregation_comparison.csv` | Table II | 64x64 primary main-six benchmark |
| Split-boundary leakage does not explain the gain | `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E1_leakage_audit.csv` | Table III / Fig. 4 | stricter split protocols recalibrate aggregation without retraining experts |
| Heterogeneous experts matter | `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E2_expert_ablation_main6.csv` | Table IV / Fig. 5 | removal tests are based on the frozen main expert pool |
| Tile+horizon validation calibration matters | `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E3_calibration_ablation.csv` | Table V / Fig. 6 | oracle row is diagnostic only |
| SS-VCEA q97.5 reduces over-correction pressure with limited skill loss | `paper_assets_vcea_extras/tables_csv_original/E1_E6/table_E6_ssvcea_threshold_sweep.csv` | Table VI / Fig. 7 | q97.5 is the selected conservative operating point |
| Original external evaluation is held out but still uses tile-local historical validation calibration | `outputs/tgrs_2019_2023/confirmation/external_tiles/external_summary.csv` | Appendix external table | future labels remain frozen during evaluation |
| The 14-tile European audit shows positive held-out future-evaluation performance | `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_group_summary.csv` | Table VII / Figs. 8-10 | finite metadata-stratified audit set |
| Statistical audit supports positive VCEA gain over M1 on the 14 audit tiles | `tables/table_E7_statistical_significance.csv` | Appendix statistical audit | descriptive robustness audit, not population-level inference |
| Operating-boundary cases are associated with short temporal span and low coverage | `paper_assets_vcea_extras/tables_csv_original/E7_strong/table_E7_pan_europe_failure_cases.csv` | Table VIII | localized operating boundary |
