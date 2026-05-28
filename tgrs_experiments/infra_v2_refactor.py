from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback only
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
PYTHON_PROJECT = ROOT
CONFIGS_DIR = ROOT / "configs" / "infra_v2"
DATA_MANIFEST_DIR = ROOT / "data" / "manifests"
LEGACY_PROCESSED_DIR = ROOT / "data" / "processed" / "tgrs_2019_2023"
EXTRACTED_DIR = ROOT / "data" / "extracted_2019_2023"
RAW_DIR = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "splits"
CACHE_ROOT = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs" / "tgrs_2019_2023" / "infrastructure_v2"

MAIN_TILES = ["E32N34", "E32N35", "E39N30", "E44N23", "E48N24", "E58N17"]
EXTERNAL_TILES = ["E30N35", "E37N42", "E42N21", "E45N51"]
RESOLUTION_FAMILY_TILES = ["E32N34", "E39N30", "E48N24"]
RESOLUTIONS = [64, 128, 512]
HORIZONS = ["t+1", "t+3", "t+5", "t+10"]


@dataclass(frozen=True)
class SourceFile:
    path: Path
    label: str
    granularity: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_model_name(name: str) -> str:
    mapping = {
        "C2_M1_scalar": "M1",
        "Persistence": "Persistence",
        "Linear-Seasonal": "Linear-Seasonal",
        "linear_seasonal": "Linear-Seasonal",
        "LinearSeasonal": "Linear-Seasonal",
        "M0": "M0",
        "C3_opt": "C3_opt",
        "VCEA_exact": "VCEA_exact",
        "COEA": "COEA_exact",
        "COEA_exact": "COEA_exact",
        "COEA_ewa_eta4p0_warm_floor50_tile_horizon": "COEA_exact",
        "CRCF_COEA_ewa_eta4p0_warm_floor50_tile_horizon": "CRCF-COEA",
        "VCEA": "VCEA_exact",
        "StaticVal-PLM1": "StaticVal-PLM1",
    }
    return mapping.get(name, name)


def model_family(name: str) -> str:
    norm = normalize_model_name(name)
    mapping = {
        "Persistence": "baseline",
        "Linear-Seasonal": "baseline",
        "M0": "single_backbone",
        "M1": "single_backbone",
        "C3_opt": "single_backbone",
        "StaticVal-PLM1": "aggregation_proxy",
        "VCEA_exact": "static_aggregation",
        "COEA_exact": "online_aggregation",
        "CRCF-COEA": "risk_controlled_online_aggregation",
    }
    return mapping.get(norm, "other")


def tile_role(tile_id: str, default: str = "main6") -> str:
    if tile_id in MAIN_TILES:
        return "main6"
    if tile_id in EXTERNAL_TILES:
        return "external_locked_eval"
    if tile_id == "ALL_MAIN6":
        return "main6_aggregate"
    return default


def aggregation_type(model: str) -> str:
    norm = normalize_model_name(model)
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA"}:
        return "exact"
    if norm == "StaticVal-PLM1":
        return "proxy"
    return "none"


def is_proxy(model: str) -> bool:
    return aggregation_type(model) == "proxy"


def uses_transfer(model: str, resolution: Any) -> bool:
    try:
        res = int(resolution)
    except Exception:
        return False
    norm = normalize_model_name(model)
    return res == 512 and norm in {"C3_opt", "VCEA_exact", "COEA_exact", "CRCF-COEA"}


def train_resolution(model: str, resolution: Any) -> Any:
    try:
        res = int(resolution)
    except Exception:
        return resolution
    if uses_transfer(model, res):
        return 128
    return res


def calibration_resolution(model: str, resolution: Any) -> Any:
    try:
        res = int(resolution)
    except Exception:
        return resolution
    norm = normalize_model_name(model)
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA", "StaticVal-PLM1"} and res == 512:
        return 128
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA", "StaticVal-PLM1"}:
        return res
    return train_resolution(model, res)


def expert_pool_id(model: str) -> str:
    norm = normalize_model_name(model)
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA"}:
        return "pool_persistence_linearseasonal_m1_c3_v1"
    if norm == "StaticVal-PLM1":
        return "pool_persistence_linearseasonal_m1_proxy_v1"
    return ""


def aggregation_weight_id(model: str, resolution: Any, tile_id: str = "") -> str:
    norm = normalize_model_name(model)
    try:
        res = int(resolution)
    except Exception:
        res = resolution
    if norm == "VCEA_exact":
        if tile_id in EXTERNAL_TILES:
            return f"agg_vcea_external_locked_r{res}"
        return f"agg_vcea_exact_r{res}"
    if norm == "COEA_exact":
        return f"agg_coea_exact_r{res}"
    if norm == "CRCF-COEA":
        return "agg_crcf_coea_r64"
    if norm == "StaticVal-PLM1":
        return f"agg_staticval_proxy_r{res}"
    return ""


def expert_cache_id(model: str, resolution: Any, tile_id: str = "") -> str:
    norm = normalize_model_name(model)
    try:
        res = int(resolution)
    except Exception:
        res = resolution
    if norm in {"Persistence", "Linear-Seasonal", "M1", "C3_opt"}:
        suffix = "transfer128to512" if uses_transfer(model, res) and norm == "C3_opt" else f"r{res}"
        scope = "external_locked" if tile_id in EXTERNAL_TILES else "main"
        return f"expert_{norm.lower().replace('-', '_')}_{scope}_{suffix}"
    if norm == "StaticVal-PLM1":
        return "pool_persistence_linearseasonal_m1_proxy_v1"
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA"}:
        return "pool_persistence_linearseasonal_m1_c3_v1"
    return ""


def notes_for_row(model: str, resolution: Any, tile_id: str = "", source_label: str = "") -> str:
    notes: list[str] = []
    norm = normalize_model_name(model)
    if tile_id in EXTERNAL_TILES:
        notes.append("external locked evaluation; not used for architecture selection")
    if uses_transfer(model, resolution):
        notes.append("512 uses 128->512 transfer inference for C3-derived branch")
    if norm in {"VCEA_exact", "COEA_exact", "CRCF-COEA"} and str(resolution) == "512":
        notes.append("512 aggregation reuses 128 validation cache for calibration")
    if norm == "StaticVal-PLM1":
        notes.append("proxy aggregation; not exact four-expert end-to-end result")
    if source_label == "main6_primary_summary" and norm == "VCEA_exact":
        notes.append("full-six primary VCEA available only as aggregate summary in historical artifact")
    return "; ".join(notes)


def parse_dates_from_csv_header(csv_path: Path) -> tuple[str, str, int]:
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    date_cols = [c for c in header if c.isdigit() and len(c) == 8]
    if not date_cols:
        return "", "", 0
    start = f"{date_cols[0][0:4]}-{date_cols[0][4:6]}-{date_cols[0][6:8]}"
    end = f"{date_cols[-1][0:4]}-{date_cols[-1][4:6]}-{date_cols[-1][6:8]}"
    return start, end, len(date_cols)


def parse_xml_metadata(xml_path: Path) -> tuple[str, str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    level = root.findtext("product_level", default="")
    production_date = root.findtext("production_date", default="")
    return level, production_date


def read_csv_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def dump_yaml(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_data_config() -> dict[str, Any]:
    return {
        "dataset": {
            "name": "EGMS 2019-2023 Ortho U",
            "statement": "Due to data access constraints, all frozen experiments continue to use EGMS 2019-2023 Ortho U.",
            "input_length": 16,
            "horizons": [1, 3, 5, 10],
        },
        "scope_decision": {
            "mainline": "64x64 regional nowcasting remains the primary paper setting",
            "high_resolution_role": "128/512 are robustness and resolution-sensitivity axes, not a task redefinition",
            "external_locked_eval": True,
        },
        "paths": {
            "raw_zip_dir": str(RAW_DIR),
            "extracted_dir": str(EXTRACTED_DIR),
            "legacy_processed_dir": str(LEGACY_PROCESSED_DIR),
            "manifest_dir": str(DATA_MANIFEST_DIR),
            "split_dir": str(SPLITS_DIR),
            "cache_root": str(CACHE_ROOT),
            "output_root": str(ROOT / "outputs" / "tgrs_2019_2023"),
        },
        "tile_groups": {
            "main6": MAIN_TILES,
            "external4_locked": EXTERNAL_TILES,
            "resolution_family_tiles": RESOLUTION_FAMILY_TILES,
        },
        "resolutions": RESOLUTIONS,
        "expert_pool": {
            "pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
            "experts": ["Persistence", "Linear-Seasonal", "M1", "C3_opt"],
        },
    }


def build_schema_files() -> None:
    ensure_dir(CONFIGS_DIR)
    dump_yaml(build_data_config(), CONFIGS_DIR / "data_config.yaml")

    result_schema = pd.DataFrame(
        [
            {"field": "experiment_id", "type": "string", "required": True, "description": "Unique migrated experiment row id"},
            {"field": "source_artifact", "type": "string", "required": True, "description": "Historical source artifact file"},
            {"field": "granularity", "type": "string", "required": True, "description": "raw_horizon_seed / summary_horizon / summary_model"},
            {"field": "seed_scope", "type": "string", "required": True, "description": "raw / aggregate_mean_std / aggregate_model"},
            {"field": "tile_id", "type": "string", "required": True, "description": "Tile id or ALL_MAIN6 aggregate token"},
            {"field": "tile_role", "type": "string", "required": True, "description": "main6 / external_locked_eval / resolution_sensitivity / main6_aggregate"},
            {"field": "resolution", "type": "integer", "required": True, "description": "64 / 128 / 512"},
            {"field": "model_name", "type": "string", "required": True, "description": "Normalized model name"},
            {"field": "source_model_name", "type": "string", "required": True, "description": "Original historical label before normalization"},
            {"field": "model_family", "type": "string", "required": True, "description": "single_backbone / static_aggregation / online_aggregation / baseline"},
            {"field": "expert_pool_id", "type": "string", "required": False, "description": "Referenced expert pool id for aggregation models"},
            {"field": "aggregation_type", "type": "string", "required": True, "description": "none / exact / proxy"},
            {"field": "is_proxy", "type": "boolean", "required": True, "description": "Whether row is based on a proxy aggregation"},
            {"field": "train_resolution", "type": "integer_or_blank", "required": True, "description": "Resolution used to train the relevant expert/model"},
            {"field": "calibration_resolution", "type": "integer_or_blank", "required": True, "description": "Resolution used for validation calibration"},
            {"field": "test_resolution", "type": "integer", "required": True, "description": "Resolution used at evaluation time"},
            {"field": "transfer_inference", "type": "boolean", "required": True, "description": "Whether test-time transfer was used, e.g. 128->512"},
            {"field": "phase", "type": "string", "required": False, "description": "test / confirmation_resolution / external_locked / aggregate"},
            {"field": "horizon", "type": "string", "required": True, "description": "t+1 / t+3 / t+5 / t+10 / all"},
            {"field": "seed", "type": "string", "required": True, "description": "Seed id or aggregate token"},
            {"field": "regime", "type": "string", "required": False, "description": "Dominant regime label where available"},
            {"field": "skill_mse", "type": "float", "required": True, "description": "Primary Skill_MSE metric"},
            {"field": "skill_rmse", "type": "float_or_blank", "required": False, "description": "Skill_RMSE if available"},
            {"field": "rmse_mm", "type": "float_or_blank", "required": False, "description": "RMSE in mm"},
            {"field": "mae_mm", "type": "float_or_blank", "required": False, "description": "MAE in mm"},
            {"field": "ssim", "type": "float_or_blank", "required": False, "description": "SSIM where available"},
            {"field": "acc_1mm", "type": "float_or_blank", "required": False, "description": "Accuracy within 1 mm"},
            {"field": "acc_0_5mm", "type": "float_or_blank", "required": False, "description": "Accuracy within 0.5 mm"},
            {"field": "stress_ratio", "type": "float_or_blank", "required": False, "description": "RMSE / persistence RMSE or worst stress ratio in summaries"},
            {"field": "persistence_rmse_mm", "type": "float_or_blank", "required": False, "description": "Persistence RMSE companion"},
            {"field": "linear_seasonal_rmse_mm", "type": "float_or_blank", "required": False, "description": "Linear-seasonal RMSE companion"},
            {"field": "hotspot_skill", "type": "float_or_blank", "required": False, "description": "Hotspot Skill_MSE diagnostic"},
            {"field": "boundary_error", "type": "float_or_blank", "required": False, "description": "Boundary gradient RMSE diagnostic"},
            {"field": "runtime_total_seconds", "type": "float_or_blank", "required": False, "description": "End-to-end runtime where available"},
            {"field": "runtime_infer_seconds", "type": "float_or_blank", "required": False, "description": "Inference runtime where available"},
            {"field": "runtime_aggregation_only_seconds", "type": "float_or_blank", "required": False, "description": "Aggregation-only runtime for aggregation models"},
            {"field": "peak_memory_mb", "type": "float_or_blank", "required": False, "description": "Peak GPU memory where available"},
            {"field": "param_count", "type": "float_or_blank", "required": False, "description": "Parameter count where available"},
            {"field": "cache_size_bytes", "type": "float_or_blank", "required": False, "description": "Reserved field for future cache accounting"},
            {"field": "calibration_source", "type": "string_or_blank", "required": False, "description": "Source artifact for validation calibration"},
            {"field": "expert_cache_id", "type": "string_or_blank", "required": False, "description": "Referenced expert cache registry id"},
            {"field": "aggregation_weight_id", "type": "string_or_blank", "required": False, "description": "Referenced aggregation weight registry id"},
            {"field": "notes", "type": "string_or_blank", "required": False, "description": "Caveats such as proxy, transfer, calibration reuse"},
        ]
    )
    write_csv_df(result_schema, CONFIGS_DIR / "result_schema_v2.csv")

    runtime_schema = pd.DataFrame(
        [
            {"field": "runtime_total_seconds", "unit": "seconds", "description": "End-to-end pipeline cost including expert generation when available"},
            {"field": "runtime_infer_seconds", "unit": "seconds", "description": "Per-run or per-eval inference cost from source artifact"},
            {"field": "runtime_aggregation_only_seconds", "unit": "seconds", "description": "Aggregation-only cost, excluding expert generation"},
            {"field": "peak_memory_mb", "unit": "MB", "description": "Peak GPU memory where historical artifact reported it"},
            {"field": "param_count", "unit": "count", "description": "Trainable parameter count or 0 for non-parametric aggregation"},
            {"field": "runtime_scope", "unit": "categorical", "description": "full_pipeline / aggregation_only / unavailable"},
        ]
    )
    write_csv_df(runtime_schema, CONFIGS_DIR / "runtime_schema.csv")

    metrics_schema = pd.DataFrame(
        [
            {"metric_group": "primary", "field": "skill_mse", "description": "Primary relative-to-persistence metric"},
            {"metric_group": "primary", "field": "rmse_mm", "description": "RMSE in mm"},
            {"metric_group": "primary", "field": "mae_mm", "description": "MAE in mm"},
            {"metric_group": "primary", "field": "ssim", "description": "Structural similarity"},
            {"metric_group": "safety", "field": "stress_ratio", "description": "Over-correction safety diagnostic"},
            {"metric_group": "resolution", "field": "hotspot_skill", "description": "Hotspot Skill_MSE"},
            {"metric_group": "resolution", "field": "boundary_error", "description": "Boundary gradient RMSE"},
            {"metric_group": "context", "field": "regime", "description": "Dominant or local regime label"},
        ]
    )
    write_csv_df(metrics_schema, CONFIGS_DIR / "metrics_schema_v2.csv")


def build_processed_resolution_views() -> None:
    processed_root = ROOT / "data" / "processed"
    for resolution in RESOLUTIONS:
        view_dir = processed_root / f"r{resolution:03d}"
        ensure_dir(view_dir)
        readme = (
            f"# r{resolution:03d} logical resolution view\n\n"
            "This refactor keeps legacy processed tensors in "
            "`data/processed/tgrs_2019_2023/<tile>/tensor_<resolution>.npz` when they already exist.\n\n"
            "For resolutions without persisted tensors, availability is tracked in "
            "`data/manifests/resolution_manifest.csv` and rows may be marked as on-demand or transfer-based.\n"
        )
        write_text(view_dir / "README.md", readme)


def build_manifests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(DATA_MANIFEST_DIR)

    main_manifest = read_csv_df(LEGACY_PROCESSED_DIR / "data_manifest.csv")
    main_manifest["tile_role"] = "main6"
    main_manifest["product_root"] = main_manifest["tile_id"].apply(lambda t: str(EXTRACTED_DIR / t))
    main_manifest["dataset_version"] = "EGMS_2019_2023_Ortho_U"

    external_manifest = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_tile_manifest.csv")
    ext_rows: list[dict[str, Any]] = []
    for _, row in external_manifest.iterrows():
        tile = row["Tile"]
        tile_dir = EXTRACTED_DIR / tile
        csv_path = next(tile_dir.glob("*.csv"))
        xml_path = next(tile_dir.glob("*.xml"))
        zip_path = next(RAW_DIR.glob(f"*{tile}*.zip"))
        start, end, num_epochs = parse_dates_from_csv_header(csv_path)
        level, prod_date = parse_xml_metadata(xml_path)
        ext_rows.append(
            {
                "tile_id": tile,
                "role": "external_locked_eval",
                "zip_name": zip_path.name,
                "csv_name": csv_path.name,
                "xml_name": xml_path.name,
                "product_level": level,
                "production_date": prod_date,
                "time_span": f"{start} – {end}" if start and end else "",
                "num_epochs": int(row["Epochs"]),
                "num_raw_points": int(row["Num points"]),
                "valid_grid_ratio_64": float(row["Valid64"]),
                "valid_grid_ratio_128": float(row["Valid128"]),
                "assigned_regime": row["Dominant regime"],
                "product_root": str(tile_dir),
                "dataset_version": "EGMS_2019_2023_Ortho_U",
            }
        )
    ext_product = pd.DataFrame(ext_rows)

    product_manifest = pd.concat(
        [
            main_manifest[
                [
                    "tile_id",
                    "role",
                    "zip_name",
                    "csv_name",
                    "xml_name",
                    "product_level",
                    "production_date",
                    "time_span",
                    "num_epochs",
                    "num_raw_points",
                    "valid_grid_ratio_64",
                    "valid_grid_ratio_128",
                    "assigned_regime",
                    "product_root",
                    "dataset_version",
                ]
            ],
            ext_product,
        ],
        ignore_index=True,
    )
    write_csv_df(product_manifest, DATA_MANIFEST_DIR / "product_manifest.csv")

    tile_rows: list[dict[str, Any]] = []
    for _, row in main_manifest.iterrows():
        tile_rows.append(
            {
                "tile_id": row["tile_id"],
                "tile_role": "main6",
                "dominant_regime": row["assigned_regime"],
                "num_points": int(row["num_raw_points"]),
                "num_epochs": int(row["num_epochs"]),
                "valid_grid_ratio_64": float(row["valid_grid_ratio_64"]),
                "valid_grid_ratio_128": float(row["valid_grid_ratio_128"]),
                "regime_stable": "",
                "regime_seasonal": "",
                "regime_subsidence": "",
                "regime_accelerating": "",
                "regime_coseismic": "",
                "regime_mixed": "",
                "legacy_manifest_source": str(LEGACY_PROCESSED_DIR / "data_manifest.csv"),
            }
        )
    for _, row in external_manifest.iterrows():
        tile_rows.append(
            {
                "tile_id": row["Tile"],
                "tile_role": "external_locked_eval",
                "dominant_regime": row["Dominant regime"],
                "num_points": int(row["Num points"]),
                "num_epochs": int(row["Epochs"]),
                "valid_grid_ratio_64": float(row["Valid64"]),
                "valid_grid_ratio_128": float(row["Valid128"]),
                "regime_stable": float(row["Regime stable"]),
                "regime_seasonal": float(row["Regime seasonal"]),
                "regime_subsidence": float(row["Regime subsidence"]),
                "regime_accelerating": float(row["Regime accelerating"]),
                "regime_coseismic": float(row["Regime coseismic"]),
                "regime_mixed": float(row["Regime mixed"]),
                "legacy_manifest_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_tile_manifest.csv"),
            }
        )
    tile_manifest = pd.DataFrame(tile_rows)
    write_csv_df(tile_manifest, DATA_MANIFEST_DIR / "tile_manifest.csv")

    resolution_rows: list[dict[str, Any]] = []
    for tile in MAIN_TILES + EXTERNAL_TILES:
        role = tile_role(tile)
        for resolution in RESOLUTIONS:
            tensor_path = LEGACY_PROCESSED_DIR / tile / f"tensor_{resolution}.npz"
            split_path = LEGACY_PROCESSED_DIR / tile / "split.json" if tile in MAIN_TILES else ""
            if tensor_path.exists():
                availability = "persisted_tensor"
                data_source = str(tensor_path)
            elif resolution == 512 and tile in MAIN_TILES:
                availability = "on_demand_from_raw_product"
                data_source = str(EXTRACTED_DIR / tile)
            elif tile in EXTERNAL_TILES and resolution == 64:
                availability = "locked_eval_from_raw_product"
                data_source = str(EXTRACTED_DIR / tile)
            else:
                availability = "not_standardized_in_current_scope"
                data_source = ""
            supports_training = role == "main6" and (resolution in {64, 128} or (resolution == 512 and tile in MAIN_TILES))
            supports_locked_eval = role == "external_locked_eval" and resolution == 64
            note = ""
            if resolution == 512 and tile in MAIN_TILES:
                note = "No persisted tensor cache in legacy layout; 512 experiments were generated on demand and may include transfer assumptions for C3-based experts."
            elif tile in EXTERNAL_TILES and resolution != 64:
                note = "External locked evaluation is frozen at 64x64 in the current paper scope."
            resolution_rows.append(
                {
                    "tile_id": tile,
                    "tile_role": role,
                    "resolution": resolution,
                    "availability": availability,
                    "data_source": data_source,
                    "split_path": str(split_path) if split_path else "",
                    "supports_training": supports_training,
                    "supports_locked_eval": supports_locked_eval,
                    "note": note,
                }
            )
    resolution_manifest = pd.DataFrame(resolution_rows)
    write_csv_df(resolution_manifest, DATA_MANIFEST_DIR / "resolution_manifest.csv")
    return product_manifest, tile_manifest, resolution_manifest


def build_split_specs() -> None:
    ensure_dir(SPLITS_DIR)
    main_split_payload: dict[str, Any] = {
        "dataset": "EGMS 2019-2023 Ortho U",
        "role": "main6_train_val_test",
        "tiles": {},
    }
    for tile in MAIN_TILES:
        split_path = LEGACY_PROCESSED_DIR / tile / "split.json"
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        main_split_payload["tiles"][tile] = split_data
    write_text(SPLITS_DIR / "main6_train_val_test.json", json.dumps(main_split_payload, ensure_ascii=False, indent=2))

    external_split = {
        "dataset": "EGMS 2019-2023 Ortho U",
        "role": "external_locked_eval",
        "locked_after_mainline_freeze": True,
        "used_for_architecture_selection": False,
        "tiles": EXTERNAL_TILES,
        "notes": [
            "External tiles serve as locked evaluation only.",
            "Historical exact VCEA external runs used frozen expert definitions with tile-local validation calibration, not architecture search.",
        ],
    }
    write_text(SPLITS_DIR / "external4_locked_eval.json", json.dumps(external_split, ensure_ascii=False, indent=2))

    resolution_split = {
        "dataset": "EGMS 2019-2023 Ortho U",
        "role": "resolution_sensitivity",
        "tiles": RESOLUTION_FAMILY_TILES,
        "resolutions": RESOLUTIONS,
        "notes": [
            "Resolution-family exact comparison is standardized on three diagnostic tiles.",
            "64x64 remains the primary full-six paper setting.",
            "512 exact aggregation rows may rely on 128 validation cache reuse and C3 transfer inference.",
        ],
    }
    write_text(SPLITS_DIR / "resolution_sensitivity_tiles.json", json.dumps(resolution_split, ensure_ascii=False, indent=2))


def build_cache_registries() -> None:
    for path in [
        CACHE_ROOT / "experts",
        CACHE_ROOT / "aggregation",
        CACHE_ROOT / "calibration" / "val_losses",
        CACHE_ROOT / "calibration" / "weights",
        CACHE_ROOT / "calibration" / "risk_thresholds",
    ]:
        ensure_dir(path)

    write_text(
        CACHE_ROOT / "experts" / "README.md",
        "# Expert cache registry\n\nHistorical expert predictions remain in legacy experiment outputs. "
        "This directory stores the normalized registry that points back to those artifacts.\n",
    )
    write_text(
        CACHE_ROOT / "aggregation" / "README.md",
        "# Aggregation cache registry\n\nThis directory standardizes VCEA/COEA/CRCF aggregation provenance without moving legacy artifacts.\n",
    )
    write_text(
        CACHE_ROOT / "calibration" / "README.md",
        "# Calibration registry\n\nValidation calibration traces, selected weights, and risk thresholds are registered here.\n",
    )

    expert_registry = {
        "version": "infra_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "cache_id": "expert_persistence_main_r064",
                "expert_name": "Persistence",
                "scope": "main6",
                "resolution": 64,
                "generation_type": "analytic_baseline",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_linear_seasonal_main_r064",
                "expert_name": "Linear-Seasonal",
                "scope": "main6",
                "resolution": 64,
                "generation_type": "kinematic_baseline",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_m1_main_r064",
                "expert_name": "M1",
                "scope": "main6",
                "resolution": 64,
                "generation_type": "trained_neural_backbone",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "act_v2" / "m1_suite" / "step1b_scalar_rows.csv"),
            },
            {
                "cache_id": "expert_c3_opt_main_r064",
                "expert_name": "C3_opt",
                "scope": "main6",
                "resolution": 64,
                "generation_type": "trained_neural_backbone",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_m1_main_r128",
                "expert_name": "M1",
                "scope": "resolution_family",
                "resolution": 128,
                "generation_type": "trained_neural_backbone",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_c3_opt_main_r128",
                "expert_name": "C3_opt",
                "scope": "resolution_family",
                "resolution": 128,
                "generation_type": "trained_neural_backbone",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_m1_main_r512",
                "expert_name": "M1",
                "scope": "resolution_family",
                "resolution": 512,
                "generation_type": "trained_neural_backbone",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"),
            },
            {
                "cache_id": "expert_c3_opt_main_transfer128to512",
                "expert_name": "C3_opt",
                "scope": "resolution_family",
                "resolution": 512,
                "generation_type": "transfer_inference",
                "train_resolution": 128,
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_report.md"),
                "notes": "512 C3 direct training hit bad allocation; 128->512 transfer inference was used.",
            },
            {
                "cache_id": "expert_pool_persistence_linearseasonal_m1_c3_v1",
                "expert_name": "pool_persistence_linearseasonal_m1_c3_v1",
                "scope": "aggregation",
                "resolution": "mixed",
                "generation_type": "logical_pool",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "final_vcea_weights.csv"),
            },
            {
                "cache_id": "expert_pool_persistence_linearseasonal_m1_proxy_v1",
                "expert_name": "pool_persistence_linearseasonal_m1_proxy_v1",
                "scope": "aggregation_proxy",
                "resolution": "mixed",
                "generation_type": "logical_pool",
                "source_artifact": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "resolution_sensitivity" / "resolution_64_128_512_summary.csv"),
                "notes": "Proxy pool without C3 expert, used by StaticVal-PLM1 sensitivity package.",
            },
        ],
    }
    dump_yaml(expert_registry, CACHE_ROOT / "expert_cache_registry.yaml")

    aggregation_registry = {
        "version": "infra_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "aggregation_weight_id": "agg_vcea_exact_r64",
                "model_name": "VCEA_exact",
                "aggregation_type": "exact",
                "resolution": 64,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "final_vcea_weights.csv"),
                "selection_protocol": "validation-calibrated static weighting",
            },
            {
                "aggregation_weight_id": "agg_vcea_external_locked_r64",
                "model_name": "VCEA_exact",
                "aggregation_type": "exact",
                "resolution": 64,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml"),
                "selection_protocol": "frozen four-expert static validation weighting on external tile",
            },
            {
                "aggregation_weight_id": "agg_coea_exact_r64",
                "model_name": "COEA_exact",
                "aggregation_type": "exact",
                "resolution": 64,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_quick_val_grid.csv"),
                "selection_protocol": "causal delayed-feedback online aggregation",
            },
            {
                "aggregation_weight_id": "agg_crcf_coea_r64",
                "model_name": "CRCF-COEA",
                "aggregation_type": "exact",
                "resolution": 64,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_crcf_thresholds.csv"),
                "selection_protocol": "COEA with validation-calibrated risk fallback",
            },
            {
                "aggregation_weight_id": "agg_vcea_exact_r128",
                "model_name": "VCEA_exact",
                "aggregation_type": "exact",
                "resolution": 128,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "aggregation_weight_summary.csv"),
                "selection_protocol": "static validation weighting",
            },
            {
                "aggregation_weight_id": "agg_coea_exact_r128",
                "model_name": "COEA_exact",
                "aggregation_type": "exact",
                "resolution": 128,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "aggregation_weight_summary.csv"),
                "selection_protocol": "online aggregation traced from family-resolution extension",
            },
            {
                "aggregation_weight_id": "agg_vcea_exact_r512",
                "model_name": "VCEA_exact",
                "aggregation_type": "exact",
                "resolution": 512,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_report.md"),
                "selection_protocol": "static validation weighting with 128 validation cache reuse",
            },
            {
                "aggregation_weight_id": "agg_coea_exact_r512",
                "model_name": "COEA_exact",
                "aggregation_type": "exact",
                "resolution": 512,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_c3_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_report.md"),
                "selection_protocol": "online aggregation with 128 validation cache reuse",
            },
            {
                "aggregation_weight_id": "agg_staticval_proxy_r64",
                "model_name": "StaticVal-PLM1",
                "aggregation_type": "proxy",
                "resolution": 64,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_proxy_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml"),
                "selection_protocol": "proxy static weighting",
            },
            {
                "aggregation_weight_id": "agg_staticval_proxy_r128",
                "model_name": "StaticVal-PLM1",
                "aggregation_type": "proxy",
                "resolution": 128,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_proxy_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml"),
                "selection_protocol": "proxy static weighting",
            },
            {
                "aggregation_weight_id": "agg_staticval_proxy_r512",
                "model_name": "StaticVal-PLM1",
                "aggregation_type": "proxy",
                "resolution": 512,
                "expert_pool_id": "pool_persistence_linearseasonal_m1_proxy_v1",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml"),
                "selection_protocol": "proxy static weighting",
            },
        ],
    }
    dump_yaml(aggregation_registry, CACHE_ROOT / "aggregation_weight_registry.yaml")

    static_variants = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "static_val_weight_variants.csv")
    static_variants = static_variants.rename(
        columns={
            "Model": "candidate_id",
            "Method": "method",
            "Temperature": "temperature",
            "Shrinkage": "shrinkage",
            "M1 floor": "m1_floor",
            "Nested score": "selection_score",
            "Nested val suffix Mean Skill_MSE": "selection_skill_mse",
            "Nested stress ratio": "selection_stress_ratio",
            "Test Mean Skill_MSE": "observed_test_skill_mse",
        }
    )
    static_variants["family"] = "static_validation_weighting"
    static_variants["selection_split"] = "nested_validation_suffix"
    static_variants["selected"] = static_variants["candidate_id"].eq("StaticVal_inverse_temp4p0_sh0p1_floor0p0")

    coea_grid = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_quick_val_grid.csv")
    coea_grid = coea_grid.rename(
        columns={
            "Model": "candidate_id",
            "Family": "family",
            "Eta": "eta",
            "Warm start": "warm_start",
            "M1 floor": "m1_floor",
            "Selection split": "selection_split",
            "Mean Skill_MSE": "selection_skill_mse",
            "Stress ratio": "selection_stress_ratio",
            "Score": "selection_score",
        }
    )
    coea_grid["selected"] = coea_grid["candidate_id"].eq("COEA_ewa_eta4p0_warm_floor50_tile_horizon")

    val_loss_cache = pd.concat(
        [
            static_variants[
                [
                    "family",
                    "candidate_id",
                    "selection_split",
                    "selection_skill_mse",
                    "selection_stress_ratio",
                    "selection_score",
                    "selected",
                ]
            ],
            coea_grid[
                [
                    "family",
                    "candidate_id",
                    "selection_split",
                    "selection_skill_mse",
                    "selection_stress_ratio",
                    "selection_score",
                    "selected",
                ]
            ],
        ],
        ignore_index=True,
    )
    write_csv_df(val_loss_cache, CACHE_ROOT / "calibration" / "val_losses" / "val_loss_cache.csv")


def try_float(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except Exception:
        return value


def safe_get(row: pd.Series, key: str, default: Any = "") -> Any:
    return row[key] if key in row.index else default


def first_matching_boundary(boundary_df: pd.DataFrame, tile: str, resolution: Any, model: str) -> Any:
    match = boundary_df[
        (boundary_df["Tile"].astype(str) == str(tile))
        & (boundary_df["Resolution"].astype(str) == str(resolution))
        & (boundary_df["Model"].astype(str) == str(model))
    ]
    if match.empty:
        return ""
    return try_float(match.iloc[0]["Gradient RMSE"])


def first_matching_hotspot(hotspot_df: pd.DataFrame, tile: str, resolution: Any, model: str) -> Any:
    match = hotspot_df[
        (hotspot_df["Tile"].astype(str) == str(tile))
        & (hotspot_df["Resolution"].astype(str) == str(resolution))
        & (hotspot_df["Model"].astype(str) == str(model))
    ]
    if match.empty:
        return ""
    return try_float(match.iloc[0]["Hotspot Skill_MSE"])


def tile_regime_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    main_df = read_csv_df(LEGACY_PROCESSED_DIR / "data_manifest.csv")
    for _, row in main_df.iterrows():
        lookup[str(row["tile_id"])] = str(row["assigned_regime"])
    ext_df = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_tile_manifest.csv")
    for _, row in ext_df.iterrows():
        lookup[str(row["Tile"])] = str(row["Dominant regime"])
    return lookup


def migrate_family_resolution_rows(boundary_df: pd.DataFrame, hotspot_df: pd.DataFrame, regime_lookup: dict[str, str]) -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_quick_rows.csv"
    df = read_csv_df(src)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        norm_model = normalize_model_name(model)
        resolution = int(row["Resolution"])
        tile = str(row["Tile"])
        rmse = float(row["RMSE [mm]"])
        persistence_rmse = float(row["Persistence RMSE [mm]"])
        rows.append(
            {
                "experiment_id": f"resolution_family::{norm_model}::{tile}::r{resolution}::{row['Seed']}::{row['Horizon']}",
                "source_artifact": str(src),
                "granularity": "raw_horizon_seed",
                "seed_scope": "raw",
                "tile_id": tile,
                "tile_role": "resolution_sensitivity",
                "resolution": resolution,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": model_family(model),
                "expert_pool_id": expert_pool_id(model),
                "aggregation_type": aggregation_type(model),
                "is_proxy": is_proxy(model),
                "train_resolution": train_resolution(model, resolution),
                "calibration_resolution": calibration_resolution(model, resolution),
                "test_resolution": resolution,
                "transfer_inference": uses_transfer(model, resolution),
                "phase": str(row["Phase"]),
                "horizon": str(row["Horizon"]),
                "seed": str(row["Seed"]),
                "regime": regime_lookup.get(tile, ""),
                "skill_mse": float(row["Skill_MSE"]),
                "skill_rmse": float(row["Skill_RMSE"]),
                "rmse_mm": rmse,
                "mae_mm": float(row["MAE [mm]"]),
                "ssim": float(row["SSIM"]),
                "acc_1mm": float(row["Acc@1mm [%]"]),
                "acc_0_5mm": float(row["Acc@0.5mm [%]"]),
                "stress_ratio": rmse / persistence_rmse if persistence_rmse else "",
                "persistence_rmse_mm": persistence_rmse,
                "linear_seasonal_rmse_mm": float(row["LinearSeasonal RMSE [mm]"]),
                "hotspot_skill": first_matching_hotspot(hotspot_df, tile, resolution, model),
                "boundary_error": first_matching_boundary(boundary_df, tile, resolution, model),
                "runtime_total_seconds": float(row["Elapsed seconds"]) if norm_model in {"M1", "M0", "C3_opt", "Persistence", "Linear-Seasonal"} else "",
                "runtime_infer_seconds": float(row["Infer seconds"]),
                "runtime_aggregation_only_seconds": float(row["Elapsed seconds"]) if aggregation_type(model) != "none" else "",
                "peak_memory_mb": "",
                "param_count": try_float(row["Param count"]),
                "cache_size_bytes": "",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "aggregation_weight_summary.csv")
                if aggregation_type(model) != "none"
                else "",
                "expert_cache_id": expert_cache_id(model, resolution, tile),
                "aggregation_weight_id": aggregation_weight_id(model, resolution, tile),
                "notes": notes_for_row(model, resolution, tile, "resolution_family"),
            }
        )
    return rows


def migrate_m1_main6_rows(boundary_df: pd.DataFrame, hotspot_df: pd.DataFrame, regime_lookup: dict[str, str]) -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "act_v2" / "m1_suite" / "step1b_scalar_rows.csv"
    df = read_csv_df(src)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        tile = str(row["Tile"])
        model = str(row["Model"])
        norm_model = "M1"
        rmse = float(row["RMSE [mm]"])
        persistence_rmse = float(row["Persistence RMSE [mm]"])
        rows.append(
            {
                "experiment_id": f"main6_primary::{norm_model}::{tile}::r64::{row['Seed']}::{row['Horizon']}",
                "source_artifact": str(src),
                "granularity": "raw_horizon_seed",
                "seed_scope": "raw",
                "tile_id": tile,
                "tile_role": "main6",
                "resolution": 64,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": "single_backbone",
                "expert_pool_id": "",
                "aggregation_type": "none",
                "is_proxy": False,
                "train_resolution": 64,
                "calibration_resolution": 64,
                "test_resolution": 64,
                "transfer_inference": False,
                "phase": "test",
                "horizon": str(row["Horizon"]),
                "seed": str(row["Seed"]),
                "regime": regime_lookup.get(tile, ""),
                "skill_mse": float(row["Skill_MSE"]),
                "skill_rmse": float(row["Skill_RMSE"]),
                "rmse_mm": rmse,
                "mae_mm": float(row["MAE [mm]"]),
                "ssim": float(row["SSIM"]),
                "acc_1mm": float(row["Acc@1mm [%]"]),
                "acc_0_5mm": float(row["Acc@0.5mm [%]"]),
                "stress_ratio": rmse / persistence_rmse if persistence_rmse else "",
                "persistence_rmse_mm": persistence_rmse,
                "linear_seasonal_rmse_mm": float(row["LinearSeasonal RMSE [mm]"]),
                "hotspot_skill": first_matching_hotspot(hotspot_df, tile, 64, norm_model),
                "boundary_error": first_matching_boundary(boundary_df, tile, 64, norm_model),
                "runtime_total_seconds": float(row["Elapsed seconds"]),
                "runtime_infer_seconds": try_float(safe_get(row, "Infer seconds", "")),
                "runtime_aggregation_only_seconds": "",
                "peak_memory_mb": "",
                "param_count": try_float(safe_get(row, "Param count", "")),
                "cache_size_bytes": "",
                "calibration_source": "",
                "expert_cache_id": expert_cache_id(norm_model, 64, tile),
                "aggregation_weight_id": "",
                "notes": "Primary full-six M1 row from act_v2 m1_suite.",
            }
        )
    return rows


def migrate_coea_crcf_rows(regime_lookup: dict[str, str]) -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_full6_raw.csv"
    df = read_csv_df(src)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        norm_model = normalize_model_name(model)
        tile = str(row["Tile"])
        rmse = float(row["RMSE [mm]"])
        persistence_rmse = float(row["Persistence RMSE [mm]"])
        rows.append(
            {
                "experiment_id": f"main6_online::{norm_model}::{tile}::r64::{row['Seed']}::{row['Horizon']}",
                "source_artifact": str(src),
                "granularity": "raw_horizon_seed",
                "seed_scope": "raw",
                "tile_id": tile,
                "tile_role": "main6",
                "resolution": 64,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": model_family(model),
                "expert_pool_id": expert_pool_id(model),
                "aggregation_type": "exact",
                "is_proxy": False,
                "train_resolution": 64,
                "calibration_resolution": 64,
                "test_resolution": 64,
                "transfer_inference": False,
                "phase": str(row["Phase"]),
                "horizon": str(row["Horizon"]),
                "seed": str(row["Seed"]),
                "regime": regime_lookup.get(tile, ""),
                "skill_mse": float(row["Skill_MSE"]),
                "skill_rmse": float(row["Skill_RMSE"]),
                "rmse_mm": rmse,
                "mae_mm": float(row["MAE [mm]"]),
                "ssim": float(row["SSIM"]),
                "acc_1mm": float(row["Acc@1mm [%]"]),
                "acc_0_5mm": float(row["Acc@0.5mm [%]"]),
                "stress_ratio": rmse / persistence_rmse if persistence_rmse else "",
                "persistence_rmse_mm": persistence_rmse,
                "linear_seasonal_rmse_mm": float(row["LinearSeasonal RMSE [mm]"]),
                "hotspot_skill": "",
                "boundary_error": "",
                "runtime_total_seconds": "",
                "runtime_infer_seconds": float(row["Infer seconds"]),
                "runtime_aggregation_only_seconds": float(row["Elapsed seconds"]),
                "peak_memory_mb": "",
                "param_count": try_float(row["Param count"]),
                "cache_size_bytes": "",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_quick_val_grid.csv"),
                "expert_cache_id": expert_cache_id(model, 64, tile),
                "aggregation_weight_id": aggregation_weight_id(model, 64, tile),
                "notes": "Aggregation-only runtime; expert generation cost excluded in historical audited_v7 artifact.",
            }
        )
    return rows


def migrate_main_aggregate_rows() -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "final_aggregation_comparison.csv"
    df = read_csv_df(src)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        norm_model = normalize_model_name(model)
        rows.append(
            {
                "experiment_id": f"main6_primary_summary::{norm_model}::r64",
                "source_artifact": str(src),
                "granularity": "summary_model",
                "seed_scope": "aggregate_model",
                "tile_id": "ALL_MAIN6",
                "tile_role": "main6_aggregate",
                "resolution": 64,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": model_family(model),
                "expert_pool_id": expert_pool_id(model),
                "aggregation_type": aggregation_type(model),
                "is_proxy": is_proxy(model),
                "train_resolution": 64,
                "calibration_resolution": calibration_resolution(model, 64),
                "test_resolution": 64,
                "transfer_inference": False,
                "phase": "aggregate",
                "horizon": "all",
                "seed": "aggregate_model",
                "regime": "",
                "skill_mse": float(row["Mean Skill_MSE"]),
                "skill_rmse": "",
                "rmse_mm": "",
                "mae_mm": "",
                "ssim": "",
                "acc_1mm": "",
                "acc_0_5mm": "",
                "stress_ratio": float(row["Stress ratio"]),
                "persistence_rmse_mm": "",
                "linear_seasonal_rmse_mm": "",
                "hotspot_skill": "",
                "boundary_error": "",
                "runtime_total_seconds": "",
                "runtime_infer_seconds": "",
                "runtime_aggregation_only_seconds": "",
                "peak_memory_mb": "",
                "param_count": "",
                "cache_size_bytes": "",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "final_vcea_weights.csv")
                if norm_model == "VCEA_exact"
                else str(ROOT / "outputs" / "tgrs_2019_2023" / "act_v7" / "finalization" / "audited_quick_val_grid.csv")
                if norm_model in {"COEA_exact", "CRCF-COEA"}
                else "",
                "expert_cache_id": expert_cache_id(model, 64, "ALL_MAIN6"),
                "aggregation_weight_id": aggregation_weight_id(model, 64, "ALL_MAIN6"),
                "notes": notes_for_row(model, 64, "ALL_MAIN6", "main6_primary_summary"),
            }
        )
    return rows


def migrate_external_summary_rows() -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_summary.csv"
    df = read_csv_df(src)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        norm_model = normalize_model_name(model)
        tile = str(row["Tile"])
        rmse = float(row["RMSE [mm] mean"])
        persistence_rmse = float(row["Persistence RMSE [mm] mean"])
        rows.append(
            {
                "experiment_id": f"external_locked_summary::{norm_model}::{tile}::r64::{row['Horizon']}",
                "source_artifact": str(src),
                "granularity": "summary_horizon",
                "seed_scope": "aggregate_mean_std",
                "tile_id": tile,
                "tile_role": "external_locked_eval",
                "resolution": 64,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": model_family(model),
                "expert_pool_id": expert_pool_id(model),
                "aggregation_type": aggregation_type(model),
                "is_proxy": is_proxy(model),
                "train_resolution": train_resolution(model, 64),
                "calibration_resolution": calibration_resolution(model, 64),
                "test_resolution": 64,
                "transfer_inference": False,
                "phase": "external_locked_summary",
                "horizon": str(row["Horizon"]),
                "seed": "aggregate_mean_std",
                "regime": "",
                "skill_mse": float(row["Skill_MSE mean"]),
                "skill_rmse": float(row["Skill_RMSE mean"]),
                "rmse_mm": rmse,
                "mae_mm": float(row["MAE [mm] mean"]),
                "ssim": float(row["SSIM mean"]),
                "acc_1mm": float(row["Acc@1mm [%] mean"]),
                "acc_0_5mm": float(row["Acc@0.5mm [%] mean"]),
                "stress_ratio": rmse / persistence_rmse if persistence_rmse else "",
                "persistence_rmse_mm": persistence_rmse,
                "linear_seasonal_rmse_mm": float(row["LinearSeasonal RMSE [mm] mean"]),
                "hotspot_skill": "",
                "boundary_error": "",
                "runtime_total_seconds": try_float(row["Elapsed seconds mean"]),
                "runtime_infer_seconds": try_float(row["Infer seconds mean"]),
                "runtime_aggregation_only_seconds": try_float(row["Elapsed seconds mean"]) if aggregation_type(model) != "none" else "",
                "peak_memory_mb": "",
                "param_count": try_float(row["Param count mean"]),
                "cache_size_bytes": "",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml")
                if aggregation_type(model) != "none"
                else "",
                "expert_cache_id": expert_cache_id(model, 64, tile),
                "aggregation_weight_id": aggregation_weight_id(model, 64, tile),
                "notes": notes_for_row(model, 64, tile, "external_locked_summary"),
            }
        )
    return rows


def migrate_proxy_summary_rows() -> list[dict[str, Any]]:
    src = ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "resolution_sensitivity" / "resolution_64_128_512_summary.csv"
    df = read_csv_df(src)
    df = df[df["Model"].astype(str) == "StaticVal-PLM1"].copy()
    hotspot_df = read_csv_df(OUTPUT_DIR / "hotspot_metrics_v2.csv")
    boundary_df = read_csv_df(OUTPUT_DIR / "boundary_metrics_v2.csv")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        model = str(row["Model"])
        norm_model = normalize_model_name(model)
        tile = str(row["Tile"])
        resolution = int(row["Resolution"])
        rmse = float(row["RMSE [mm] mean"])
        persistence_rmse = float(row["Persistence RMSE [mm] mean"])
        rows.append(
            {
                "experiment_id": f"resolution_proxy::{norm_model}::{tile}::r{resolution}::{row['Horizon']}",
                "source_artifact": str(src),
                "granularity": "summary_horizon",
                "seed_scope": "aggregate_mean_std",
                "tile_id": tile,
                "tile_role": "resolution_sensitivity",
                "resolution": resolution,
                "model_name": norm_model,
                "source_model_name": model,
                "model_family": model_family(model),
                "expert_pool_id": expert_pool_id(model),
                "aggregation_type": aggregation_type(model),
                "is_proxy": is_proxy(model),
                "train_resolution": train_resolution(model, resolution),
                "calibration_resolution": calibration_resolution(model, resolution),
                "test_resolution": resolution,
                "transfer_inference": uses_transfer(model, resolution),
                "phase": "resolution_confirmation_proxy",
                "horizon": str(row["Horizon"]),
                "seed": "aggregate_mean_std",
                "regime": "",
                "skill_mse": float(row["Skill_MSE mean"]),
                "skill_rmse": float(row["Skill_RMSE mean"]),
                "rmse_mm": rmse,
                "mae_mm": float(row["MAE [mm] mean"]),
                "ssim": float(row["SSIM mean"]),
                "acc_1mm": float(row["Acc@1mm [%] mean"]),
                "acc_0_5mm": float(row["Acc@0.5mm [%] mean"]),
                "stress_ratio": rmse / persistence_rmse if persistence_rmse else "",
                "persistence_rmse_mm": persistence_rmse,
                "linear_seasonal_rmse_mm": float(row["LinearSeasonal RMSE [mm] mean"]),
                "hotspot_skill": first_matching_hotspot(hotspot_df.rename(columns={"tile_id": "Tile", "resolution": "Resolution", "model_name": "Model", "hotspot_skill_mse": "Hotspot Skill_MSE"}), tile, resolution, model),
                "boundary_error": first_matching_boundary(boundary_df.rename(columns={"tile_id": "Tile", "resolution": "Resolution", "model_name": "Model", "boundary_gradient_rmse": "Gradient RMSE"}), tile, resolution, model),
                "runtime_total_seconds": "",
                "runtime_infer_seconds": try_float(row["Infer seconds mean"]),
                "runtime_aggregation_only_seconds": try_float(row["Elapsed seconds mean"]),
                "peak_memory_mb": "",
                "param_count": try_float(row["Param count mean"]),
                "cache_size_bytes": "",
                "calibration_source": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "freeze_manifest.yaml"),
                "expert_cache_id": expert_cache_id(model, resolution, tile),
                "aggregation_weight_id": aggregation_weight_id(model, resolution, tile),
                "notes": notes_for_row(model, resolution, tile, "resolution_proxy"),
            }
        )
    return rows


def build_metric_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_hotspot = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_hotspot.csv")
    family_hotspot = family_hotspot.rename(
        columns={
            "Tile": "tile_id",
            "Resolution": "resolution",
            "Model": "model_name",
            "Hotspot Skill_MSE": "hotspot_skill_mse",
            "Hotspot RMSE [mm]": "hotspot_rmse_mm",
            "Hotspot MAE [mm]": "hotspot_mae_mm",
        }
    )
    family_hotspot["scope"] = "resolution_family"
    family_hotspot["model_name"] = family_hotspot["model_name"].map(normalize_model_name)

    confirm_hotspot = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "resolution_sensitivity" / "hotspot_metrics.csv")
    confirm_hotspot = confirm_hotspot.rename(
        columns={
            "Tile": "tile_id",
            "Resolution": "resolution",
            "Model": "model_name",
            "Hotspot Skill_MSE": "hotspot_skill_mse",
            "Hotspot RMSE [mm]": "hotspot_rmse_mm",
            "Hotspot MAE [mm]": "hotspot_mae_mm",
        }
    )
    confirm_hotspot["scope"] = "resolution_confirmation_proxy"
    confirm_hotspot["model_name"] = confirm_hotspot["model_name"].map(normalize_model_name)
    hotspot_v2 = pd.concat([family_hotspot, confirm_hotspot], ignore_index=True)
    hotspot_v2 = hotspot_v2.drop_duplicates(subset=["tile_id", "resolution", "model_name"], keep="first")
    write_csv_df(hotspot_v2, OUTPUT_DIR / "hotspot_metrics_v2.csv")

    family_boundary = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_boundary.csv")
    family_boundary = family_boundary.rename(
        columns={
            "Tile": "tile_id",
            "Resolution": "resolution",
            "Model": "model_name",
            "Gradient RMSE": "boundary_gradient_rmse",
        }
    )
    family_boundary["scope"] = "resolution_family"
    family_boundary["model_name"] = family_boundary["model_name"].map(normalize_model_name)

    confirm_boundary = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "resolution_sensitivity" / "boundary_gradient_metrics.csv")
    confirm_boundary = confirm_boundary.rename(
        columns={
            "Tile": "tile_id",
            "Resolution": "resolution",
            "Model": "model_name",
            "Gradient RMSE": "boundary_gradient_rmse",
        }
    )
    confirm_boundary["scope"] = "resolution_confirmation_proxy"
    confirm_boundary["model_name"] = confirm_boundary["model_name"].map(normalize_model_name)
    boundary_v2 = pd.concat([family_boundary, confirm_boundary], ignore_index=True)
    boundary_v2 = boundary_v2.drop_duplicates(subset=["tile_id", "resolution", "model_name"], keep="first")
    write_csv_df(boundary_v2, OUTPUT_DIR / "boundary_metrics_v2.csv")

    regime_df = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_regime_breakdown.csv")
    regime_df = regime_df.rename(
        columns={
            "Model": "model_name",
            "Tile": "tile_id",
            "Seed": "seed",
            "Horizon": "horizon",
            "Regime": "regime",
            "Skill_MSE": "skill_mse",
            "RMSE [mm]": "rmse_mm",
            "MAE [mm]": "mae_mm",
            "SSIM": "ssim",
        }
    )
    regime_df["scope"] = "external_locked_eval"
    regime_df["model_name"] = regime_df["model_name"].map(normalize_model_name)
    write_csv_df(regime_df, OUTPUT_DIR / "regime_metrics_v2.csv")
    return hotspot_v2, boundary_v2, regime_df


def build_migrated_results() -> pd.DataFrame:
    hotspot_df, boundary_df, _ = build_metric_outputs()
    regime_lookup = tile_regime_lookup()
    hotspot_lookup = hotspot_df.rename(columns={"tile_id": "Tile", "resolution": "Resolution", "model_name": "Model", "hotspot_skill_mse": "Hotspot Skill_MSE"})
    boundary_lookup = boundary_df.rename(columns={"tile_id": "Tile", "resolution": "Resolution", "model_name": "Model", "boundary_gradient_rmse": "Gradient RMSE"})
    rows = []
    rows.extend(migrate_family_resolution_rows(boundary_lookup, hotspot_lookup, regime_lookup))
    rows.extend(migrate_m1_main6_rows(boundary_lookup, hotspot_lookup, regime_lookup))
    rows.extend(migrate_coea_crcf_rows(regime_lookup))
    rows.extend(migrate_main_aggregate_rows())
    rows.extend(migrate_external_summary_rows())
    rows.extend(migrate_proxy_summary_rows())
    migrated = pd.DataFrame(rows)
    migrated = migrated.sort_values(["tile_role", "tile_id", "resolution", "model_name", "seed", "horizon"]).reset_index(drop=True)
    write_csv_df(migrated, OUTPUT_DIR / "migrated_results_v2.csv")
    return migrated


def build_runtime_protocol() -> None:
    text = """# Runtime / Memory Protocol

本文件统一历史结果里的 runtime / memory 口径，避免把 aggregation-only 和 full-pipeline 成本混写。

## 口径

- `runtime_total_seconds`：端到端总成本。只有历史 artifact 明确给出时才填写。
- `runtime_infer_seconds`：对应 source artifact 报告的 inference 或评估耗时。
- `runtime_aggregation_only_seconds`：只包含 aggregation 本身，不包含 expert 训练或 expert prediction generation。
- `peak_memory_mb`：仅当历史 artifact 明确给出 GPU 峰值显存时填写。

## 当前明确规则

- `M1 / M0 / C3_opt`：来自训练/评估 artifact 的 elapsed seconds 视为 model-level runtime。
- `VCEA_exact / COEA_exact / CRCF-COEA`：历史 resolution-family 与 V7 audited artifact 报告的是 aggregation cost；不得把它们直接写成 end-to-end pipeline cost。
- `512` 分辨率下的 `VCEA_exact / COEA_exact` 还额外带有 `128` validation cache reuse；因此 runtime 与 calibration 假设必须一起报告。
- `external_locked_eval` 历史汇总只保留了 model/eval 级别耗时，不单独追溯 raw cache IO。

## 实施决策

- 本次 infra_v2 apply 不移动历史结果文件，也不改写旧 runtime 数值。
- 新 schema 通过 `runtime_total_seconds` 和 `runtime_aggregation_only_seconds` 拆口径。
- 论文或 rebuttal 中若比较 efficiency，必须显式说明是否包含 expert generation cost。
"""
    write_text(OUTPUT_DIR / "runtime_memory_protocol.md", text)


def build_output_scope_registry() -> None:
    rows = [
        {
            "scope_id": "main6_primary_64",
            "logical_role": "paper_mainline",
            "resolution_scope": "64",
            "source_path": str(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study"),
            "migration_mode": "registry_only",
            "adopted_in_paper": True,
            "note": "Primary main-six VCEA/M1/COEA/CRCF results remain in legacy directories and are indexed by infra_v2.",
        },
        {
            "scope_id": "external4_locked_64",
            "logical_role": "robustness_external_locked_eval",
            "resolution_scope": "64",
            "source_path": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles"),
            "migration_mode": "registry_only",
            "adopted_in_paper": True,
            "note": "External tiles are frozen robustness evidence, not architecture-selection data.",
        },
        {
            "scope_id": "resolution_family_64_128_512",
            "logical_role": "resolution_sensitivity",
            "resolution_scope": "64|128|512",
            "source_path": str(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension"),
            "migration_mode": "registry_only",
            "adopted_in_paper": True,
            "note": "Three-tile family-level exact/proxy/transfer sensitivity package.",
        },
        {
            "scope_id": "hotspot_boundary_diagnostics",
            "logical_role": "diagnostic_metrics",
            "resolution_scope": "64|128|512",
            "source_path": str(OUTPUT_DIR),
            "migration_mode": "normalized_copy",
            "adopted_in_paper": True,
            "note": "Hotspot/boundary/regime metrics are re-exported in normalized v2 schema.",
        },
        {
            "scope_id": "legacy_search_history",
            "logical_role": "supplementary_search_history",
            "resolution_scope": "mixed",
            "source_path": str(ROOT / "outputs" / "tgrs_2019_2023"),
            "migration_mode": "not_moved",
            "adopted_in_paper": False,
            "note": "Historical V1-V8 / DSCF search artifacts stay in place and are referenced by source path only.",
        },
    ]
    write_csv_df(pd.DataFrame(rows), OUTPUT_DIR / "output_scope_registry.csv")


def build_paper_ready_summary_tables(migrated: pd.DataFrame) -> None:
    main_comp = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "staticval_ea_study" / "final_aggregation_comparison.csv")
    main_rows = []
    for _, row in main_comp.iterrows():
        main_rows.append(
            {
                "table_id": "main6_primary_64",
                "section": "mainline",
                "model_name": normalize_model_name(row["Model"]),
                "resolution": 64,
                "mean_skill_mse": float(row["Mean Skill_MSE"]),
                "gain_vs_m1": float(row["Gain vs M1"]),
                "stress_ratio": float(row["Stress ratio"]),
                "better_tiles": int(row["Better tiles"]),
                "provenance": "staticval_ea_study/final_aggregation_comparison.csv",
                "note": "Primary full-six 64x64 result",
            }
        )

    external_summary = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_summary.csv")
    external_stress = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "external_tiles" / "external_stress_report.csv")
    ext_mean_skill = external_summary.groupby("Model")["Skill_MSE mean"].mean().to_dict()
    ext_worst_stress = external_stress.groupby("Model")["Worst RMSE/persistence"].max().to_dict()
    m1_ext = float(ext_mean_skill.get("M1", 0.0))
    external_rows = []
    for model, mean_skill in ext_mean_skill.items():
        external_rows.append(
            {
                "table_id": "external_locked_64",
                "section": "robustness",
                "model_name": normalize_model_name(model),
                "resolution": 64,
                "mean_skill_mse": float(mean_skill),
                "gain_vs_m1": float(mean_skill) - m1_ext,
                "stress_ratio": float(ext_worst_stress.get(model, "")),
                "better_tiles": "",
                "provenance": "confirmation/external_tiles",
                "note": "External locked evaluation aggregated over tile-horizon summaries",
            }
        )

    family_rank = read_csv_df(ROOT / "outputs" / "tgrs_2019_2023" / "confirmation" / "model_family_resolution_extension" / "family_resolution_ranking.csv")
    family_rows = []
    for _, row in family_rank.iterrows():
        family_rows.append(
            {
                "table_id": "resolution_family",
                "section": "resolution_sensitivity",
                "model_name": normalize_model_name(row["Model"]),
                "resolution": int(row["Resolution"]),
                "mean_skill_mse": float(row["Mean Skill_MSE"]),
                "gain_vs_m1": "",
                "stress_ratio": float(row["Stress ratio"]),
                "better_tiles": int(row["Better tiles vs M1"]),
                "provenance": "confirmation/model_family_resolution_extension/family_resolution_ranking.csv",
                "note": "Three-tile family-level exact resolution comparison",
            }
        )

    paper_df = pd.DataFrame(main_rows + external_rows + family_rows)
    write_csv_df(paper_df, OUTPUT_DIR / "paper_ready_summary_tables.csv")


def build_native_resolution_feasibility() -> None:
    text = """# Native Resolution Feasibility Note

本说明只记录当前 frozen scope 下为什么 **不** 把论文主线改成 native / full-resolution inspection。

## 已有信号

- `VCEA_exact` 在 `64 -> 128 -> 512` 上随分辨率提升而变强。
- `M1` 不随分辨率自然获益，说明“更高分辨率自动更好”并不成立。
- `C3_opt@512` 触发了 direct training 的 `bad allocation`，因此当前 `512` 结果对 C3 分支依赖 `128 -> 512 transfer inference`。
- `VCEA_exact / COEA_exact @512` 还复用了 `128` validation cache 做 calibration。

## 结论

- `512` 结果足以作为 **resolution-aware robustness / sensitivity evidence**。
- 但它还不足以支撑“改成 high-resolution inspection 主线”的方法与实验闭环。
- 因此 infra_v2 只把 `512` 固定为显式记录的 sensitivity 维度，不升级为当前论文主任务。
"""
    write_text(OUTPUT_DIR / "native_resolution_feasibility.md", text)


def build_migration_log(migrated: pd.DataFrame) -> None:
    by_source = migrated.groupby("source_artifact").size().sort_values(ascending=False)
    lines = [
        "# Migration Log",
        "",
        "本次 apply 采用 **non-destructive registry migration**：不移动旧输出目录，只新增 infra_v2 schema / registry / standardized result table。",
        "",
        "## Source Artifacts",
        "",
    ]
    for source, count in by_source.items():
        lines.append(f"- `{source}` -> {count} rows")
    lines.extend(
        [
            "",
            "## Mapping Rules",
            "",
            "- `C2_M1_scalar` 规范化为 `M1`。",
            "- `COEA_ewa_eta4p0_warm_floor50_tile_horizon` 规范化为 `COEA_exact`。",
            "- `CRCF_COEA_ewa_eta4p0_warm_floor50_tile_horizon` 规范化为 `CRCF-COEA`。",
            "- `StaticVal-PLM1` 保留为 `proxy`，不和 `VCEA_exact` 混写。",
            "- `512` 下的 `C3_opt`、`VCEA_exact`、`COEA_exact` 显式标注 `transfer_inference` 或 `128 validation cache reuse`。",
            "",
            "## Scope Decision Applied",
            "",
            "- 保留 `64x64` 作为主文主线。",
            "- `128/512` 进入 resolution robustness / sensitivity registry。",
            "- `external tiles` 保持 locked evaluation 标记。",
            "- 历史搜索输出目录不做破坏式搬迁，统一由 `output_scope_registry.csv` 编目。",
        ]
    )
    write_text(OUTPUT_DIR / "migration_log.md", "\n".join(lines))


def build_freeze_manifest() -> None:
    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = None

    torch_version = ""
    cuda_available = ""
    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", "")
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass

    manifest = {
        "dataset": "EGMS 2019-2023 Ortho U",
        "scope_decision": {
            "mainline": "64x64 regional nowcasting",
            "resolution_sensitivity": [64, 128, 512],
            "external_locked_eval": EXTERNAL_TILES,
        },
        "models": {
            "mainline": ["VCEA_exact", "M1", "COEA_exact", "CRCF-COEA", "Persistence"],
            "resolution_family": ["Persistence", "Linear-Seasonal", "M0", "M1", "C3_opt", "VCEA_exact", "COEA_exact", "StaticVal-PLM1"],
        },
        "seeds": [42, 43, 44],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": str(torch_version),
            "torch_cuda_available": bool(cuda_available) if cuda_available != "" else "",
        },
        "workspace": {
            "root": str(ROOT),
            "git_commit": git_commit,
            "git_note": "workspace not under git" if git_commit is None else "",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump_yaml(manifest, OUTPUT_DIR / "freeze_manifest.yaml")


def build_qa_report(migrated: pd.DataFrame) -> None:
    critical_fields = [
        "resolution",
        "tile_role",
        "aggregation_type",
        "is_proxy",
        "train_resolution",
        "calibration_resolution",
        "test_resolution",
    ]
    missing_counts = {}
    for field in critical_fields:
        series = migrated[field].astype(str)
        missing_counts[field] = int(series.isna().sum() + series.eq("").sum())

    exact_rows = migrated[migrated["aggregation_type"] == "exact"]
    proxy_rows = migrated[migrated["aggregation_type"] == "proxy"]
    transfer_rows = migrated[migrated["transfer_inference"] == True]  # noqa: E712
    external_rows = migrated[migrated["tile_role"] == "external_locked_eval"]

    lines = [
        "# Infra Refactor QA",
        "",
        f"- migrated row count: **{len(migrated)}**",
        f"- exact rows: **{len(exact_rows)}**",
        f"- proxy rows: **{len(proxy_rows)}**",
        f"- transfer rows: **{len(transfer_rows)}**",
        f"- external locked rows: **{len(external_rows)}**",
        "",
        "## Critical Field Missing Count",
        "",
    ]
    for field, count in missing_counts.items():
        lines.append(f"- `{field}`: {count}")
    transfer_required = migrated[migrated["transfer_inference"] == True]  # noqa: E712
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- all aggregation rows carry `expert_pool_id`: {bool((exact_rows['expert_pool_id'].astype(str) != '').all())}",
            f"- all aggregation rows carry `aggregation_weight_id`: {bool((exact_rows['aggregation_weight_id'].astype(str) != '').all())}",
            f"- external rows all marked locked eval: {bool((external_rows['tile_role'] == 'external_locked_eval').all())}",
            f"- transfer-marked rows carry explicit notes: {bool((transfer_required['notes'].astype(str).str.len() > 0).all())}",
            "",
            "## Non-destructive decision",
            "",
            "- Legacy output paths were preserved.",
            "- New logical organization is expressed through `output_scope_registry.csv`, `cache/*_registry.yaml`, and `migrated_results_v2.csv`.",
        ]
    )
    write_text(OUTPUT_DIR / "infra_refactor_QA.md", "\n".join(lines))


def build_scope_decision_apply_report(migrated: pd.DataFrame) -> None:
    counts = Counter(migrated["tile_role"])
    text = f"""# Scope Decision Apply Report

本次 apply 完整落实了 `VCEA_DSCF_infrastructure_refactor_plan_and_scope_decision_final` 的核心范围：

- 保留 `64×64 regional nowcasting` 为主文主线；
- 把 `128/512` 固定为 `resolution robustness / sensitivity`；
- 把 external tiles 固定为 `locked external validation`；
- 不把论文主线改成 high-resolution inspection；
- 不做破坏式目录搬迁，而是增加可追溯的 schema / manifest / registry / migrated result layer。

## Result Migration Summary

- total rows migrated: **{len(migrated)}**
- by tile_role: `{dict(counts)}`

## Delivered

- `configs/infra_v2/*`
- `data/manifests/*`
- `splits/*`
- `cache/*_registry.yaml`
- `outputs/tgrs_2019_2023/infrastructure_v2/*`

## Important Applied Caveats

- `proxy` 与 `exact` 已显式拆开；
- `512` 下的 `transfer inference` 与 `128 validation cache reuse` 已显式标注；
- `aggregation-only runtime` 与 `full-pipeline runtime` 已拆口径；
- `external locked eval` 已独立登记，不再与主线搜索混用。
"""
    write_text(OUTPUT_DIR / "scope_decision_apply_report.md", text)


def run() -> dict[str, Any]:
    ensure_dir(OUTPUT_DIR)
    build_schema_files()
    build_processed_resolution_views()
    build_manifests()
    build_split_specs()
    build_cache_registries()
    migrated = build_migrated_results()
    build_runtime_protocol()
    build_output_scope_registry()
    build_paper_ready_summary_tables(migrated)
    build_native_resolution_feasibility()
    build_migration_log(migrated)
    build_freeze_manifest()
    build_qa_report(migrated)
    build_scope_decision_apply_report(migrated)

    summary = {
        "status": "ok",
        "generated_root": str(OUTPUT_DIR),
        "migrated_rows": int(len(migrated)),
        "main6_rows": int((migrated["tile_role"] == "main6").sum()),
        "external_locked_rows": int((migrated["tile_role"] == "external_locked_eval").sum()),
        "resolution_rows": int((migrated["tile_role"] == "resolution_sensitivity").sum()),
    }
    write_text(OUTPUT_DIR / "run_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
