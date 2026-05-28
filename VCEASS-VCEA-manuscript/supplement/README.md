# VCEA / SS-VCEA Reproducibility Supplement

This supplement packages the paper-facing reproducibility assets for the VCEA / SS-VCEA manuscript.

## Scope

The supplement supports:

- split definitions for the primary main-six benchmark and the original external locked evaluation;
- product and tile manifests;
- expert-cache and aggregation-weight registries;
- threshold diagnostics for SS-VCEA q97.5;
- compact result CSVs used by the manuscript;
- the statistical audit added for the 14-tile European frozen evaluation;
- metadata-only documentation for the 14-tile audit selection process;
- claim-to-evidence tracing and exact/proxy/transfer flags.

## Package layout

- `assets/configs/`: copied experiment and preprocessing configs used by the manuscript.
- `assets/splits/`: frozen split definitions.
- `assets/manifests/`: product, tile, audit, and asset manifests.
- `assets/registries/`: expert-cache lineage, aggregation-weight lineage, and exact/proxy/transfer flags.
- `assets/results/`: compact CSV and JSON result tables used by the manuscript.
- `assets/scripts/`: lightweight scripts used to regenerate audit-only tables from archived result CSVs.
- `selection_notes.md`: metadata-only selection rationale for the 14-tile European audit.
- `selection_tile_pool.csv`: reconstructed candidate-pool register with selected/nonselected flag.
- `claim_to_evidence_map.md`: compact mapping from manuscript claims to evidence files.
- `reproducibility/`: paper-facing claim/source/lineage maps referenced by the appendix reproducibility table.
- `environment.txt`: frozen software and hardware summary.
- `seeds.csv`: archived experiment and diagnostic seeds.
- `reproducibility_manifest.json`: top-level supplement manifest.

## Protocol notes

- VCEA uses tile-local historical validation calibration followed by frozen future evaluation.
- On the original external tiles and the 14-tile European audit, expert definitions, preprocessing, evaluation metrics, calibration logic, and SS-VCEA q97.5 thresholds are fixed before evaluation.
- Future test labels are not used for expert search, threshold selection, fallback tuning, or prediction before the corresponding target frame becomes observable.

## Raw data policy

Raw EGMS products are not redistributed in this package. The package provides product identifiers, tile metadata, preprocessing configuration, split definitions, and derived metrics; raw products can be obtained from the official EGMS/Copernicus service subject to their access conditions.

## Path policy

Absolute local paths have been sanitized to paper-facing placeholders such as `${PROJECT_ROOT}` and `${MANUSCRIPT_ROOT}` where applicable.

## Low-cost baseline note

See `baseline_extension_todo.md` for a record of inexpensive baseline variants that could be added in a future appendix-only extension without changing the current manuscript claims.
