# VCEA / SS-VCEA Main Code Release

This repository contains the main code, configs, manifests, and manuscript sources used for the VCEA / SS-VCEA study on EGMS regional ground-deformation nowcasting under strong persistence baselines.

## Scope

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

## Repository Layout

- `tgrs_experiments/`: main experiment implementations
- `configs/`: experiment and infrastructure configs
- `splits/`: chronological split definitions
- `data/manifests/`: dataset and tile metadata only
- `cache/`: expert-pool and aggregation registries
- `paper_assets_vcea_extras/`: manuscript-facing tables, figures, and traceability assets
- `VCEASS-VCEA-manuscript/`: LaTeX manuscript source

## Main Entry Points

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

## Environment

A minimal environment file is provided:

- `environment.yml`

You can create it with:

```powershell
conda env create -f environment.yml
conda activate vcea-extra-experiments
```

## EGMS API Credentials

The strong E7 audit code expects EGMS service credentials through environment variables:

- `EGMS_SERVICE_KEY_JSON`
- `EGMS_SERVICE_KEY_PATH`

An example template is provided in:

- `egms_service_key.example.json`

Do not commit real credentials.

## Notes for Public GitHub Release

- Paths in manifests and registries have been sanitized to `${PROJECT_ROOT}` style placeholders.
- This repository intentionally excludes large data artifacts and model weights.
- If you plan to make the repository public, choose an explicit license before publishing. No license file has been selected automatically in this release.
