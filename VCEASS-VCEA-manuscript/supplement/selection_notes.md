# 14-Tile Audit Selection Notes

The 14 audit tiles were selected after expert definitions, VCEA calibration logic, SS-VCEA q97.5 thresholds, preprocessing, and metrics were fixed. Selection used metadata-only criteria, including broad European region, audit-regime group, valid-cell coverage group, temporal availability, and EGMS product availability. No model performance metric from candidate tiles was used for selecting the 14 audit tiles.

The available reproducibility package provides:

- selected-tile metadata in `assets/results/table_E7_pan_europe_locked_results_by_tile.csv`;
- grouped audit metadata in `assets/results/table_E7_pan_europe_group_summary.csv`;
- selected-tile metadata annotations in `assets/manifests/audit_E7_manifest.json`;
- the selected-tile catalog in `selection_tile_pool.csv`.

The complete candidate pool was not archived with full regime and coverage annotations for every nonselected tile. The available package therefore provides:

- the API-derived candidate list used to reconstruct the archived selection frame;
- the selected-tile metadata table;
- the metadata-only selection rationale;
- a selected/nonselected indicator in `selection_tile_pool.csv`.

Accordingly, `selection_tile_pool.csv` should be interpreted as a reproducibility-oriented audit register rather than as a fully annotated census of all candidate tiles.
