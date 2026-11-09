# EGMS ground-deformation nowcasting: forecast-combination benchmark and pre-registered confirmatory test

Code for "Where does short-horizon skill come from? A pre-registered, multi-release evaluation of
validation-calibrated forecast combination for EGMS ground-deformation maps" (submitted to the
International Journal of Remote Sensing, 2026).

## What is new for the IJRS manuscript

- `tgrs_experiments/` — preprocessing, expert models (M1 gated residual patch Transformer,
  C3_opt multi-scale ConvNeXt-lite), training, and the E7 audit (unchanged from the earlier release).
- `analysis_vcea_comparators_20261104/` — forecast-combination comparator audit
  (`run_vcea_comparators.py`) and the pre-registered confirmatory protocol (`variant_protocol.py`).
  - `prereg/` holds the protocol, the 24 confirmatory tiles, the frozen variant, Amendment 01
    and their SHA-256 files. `variant_protocol.py eval` refuses to run if these or the method code change.
  - `results/` holds the exploratory comparator results (main-six, E7);
    `results/fresh/` holds the single locked confirmatory run (see `EVAL_DONE`).
  - `cache/fresh_download_manifest.csv` lists the SHA-256 of every EGMS 2020-2024 archive used.
- `paper_ijrs/` — `make_tables.py` and `manuscript/make_manuscript_figures.py` produce every table
  and figure in the paper; `paper_ijrs/tables/` holds the resulting CSVs.
- `outputs/tgrs_2019_2023/` — preprocessing config and E7 tile metadata / per-tile results read by those scripts.

Archived snapshot: https://doi.org/10.5281/zenodo.20434102 (version 2, matching the submitted
manuscript: https://doi.org/10.5281/zenodo.24040477).

Raw EGMS products are not redistributed. Download them from https://egms.land.copernicus.eu/
with your own EGMS service key (see `egms_service_key.example.json`; never commit a real key).
Cached expert predictions (~2 GB) are not included; the scripts regenerate them.

Licence: MIT (see `LICENSE`).

## Earlier VCEA / SS-VCEA release (kept for provenance)

The material below describes the earlier VCEA / SS-VCEA draft. Its manuscript sources and
SS-VCEA hardening scripts are not used by the IJRS manuscript.

This repository contains the main code, configs, manifests, and manuscript sources used for the VCEA / SS-VCEA study on EGMS regional ground-deformation nowcasting under strong persistence baselines.

### Scope

Included here:

- core experiment code in `tgrs_experiments/`
- experiment configs in `configs/`
- split definitions in `splits/`
- lightweight cache registries and manifests in `cache/` and `data/manifests/`
- paper asset sources in `paper_assets_vcea_extras/`
- manuscript source in `VCEASS-VCEA-manuscript/`
- wrapper scripts used to reproduce paper tables, figures, and audits

Not redistributed here:

- raw EGMS products
- extracted EGMS tiles
- processed tensor caches
- large training outputs
- model checkpoints / weights

Raw EGMS products are not redistributed in this package. This release provides product identifiers, tile metadata, preprocessing configuration, split definitions, and derived result manifests. Raw products must be obtained from the official EGMS / Copernicus service subject to their access conditions.

### Repository Layout

- `tgrs_experiments/`: main experiment implementations
- `configs/`: experiment and infrastructure configs
- `splits/`: chronological split definitions
- `data/manifests/`: dataset and tile metadata only
- `cache/`: expert-pool and aggregation registries
- `paper_assets_vcea_extras/`: manuscript-facing tables, figures, and traceability assets
- `VCEASS-VCEA-manuscript/`: LaTeX manuscript source

### Main Entry Points

The primary code lives in `tgrs_experiments/`. Common wrapper scripts include:

- `run_staticval_ea_study.py`
- `run_vcea_extra_experiments.py`
- `run_vcea_e7_pan_europe_audit.py`
- `run_vcea_e7_pan_europe_audit_strong.py`
- `run_tgrs_vcea_hardening.py`

Paper asset generation helpers:

- `prepare_tgrs_manuscript_assets.py`
- `generate_manuscript_final_figures.py`
- `generate_manuscript_panels_and_latex.py`
- `generate_paper_assets_vcea_extras.py`

Some helper scripts expect a local `outputs/tgrs_2019_2023/` archive that is not included in this release. Those scripts are kept for traceability and reproducibility of the paper build pipeline.

### Environment

A minimal environment file is provided:

- `environment.yml`

You can create it with:

```powershell
conda env create -f environment.yml
conda activate vcea-extra-experiments
```

### EGMS API Credentials

The strong E7 audit code expects EGMS service credentials through environment variables:

- `EGMS_SERVICE_KEY_JSON`
- `EGMS_SERVICE_KEY_PATH`

An example template is provided in:

- `egms_service_key.example.json`

Do not commit real credentials.

### Notes for Public GitHub Release

- Paths in manifests and registries have been sanitized to `${PROJECT_ROOT}` style placeholders.
- This repository intentionally excludes large data artifacts and model weights.
- Code in this repository is released under the MIT License (see `LICENSE`).
