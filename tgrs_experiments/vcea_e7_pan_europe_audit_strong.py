from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

matplotlib.use("Agg")

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, train_eval_single_tile
from .act_v3 import summarize_seeded_rows
from .act_v4c_c3_candidate import c3_runtime_optimized_config, m1_scalar_config
from .act_v4c_efficiency import train_eval_rescue_single_tile
from .act_v5_selector import infer_c3_split, infer_m1_split
from .dscf_confirmation import build_bundle_from_tile_id, extract_raw_archives
from .pipeline import PROJECT_ROOT, RAW_DIR, TILE_ORDER, ensure_dir, preprocess_all_tiles
from .staticval_ea_study import rows_from_prediction
from .tgrs_vcea_hardening import (
    FINAL_COEA_CONFIG,
    FINAL_VCEA_CONFIG,
    apply_ss_wrapper,
    calibrate_thresholds,
    coea_prediction,
    risk_components,
    static_prediction_from_external,
)


EGMS_STATEMENT = "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U**，新增 external locked tiles 通过官方 CLMS/EGMS API 获取。"
API_ENDPOINT = "https://egms.land.copernicus.eu/insar-api/archive"
SAFE_TARGET = "COEA"
SAFE_PERCENTILE = 97.5
SS_LABEL = "SS-VCEA-COEA_q97.5"
AUDIT_MODELS = ["Persistence", "Linear-Seasonal", "M1", "C3_opt", "VCEA", SS_LABEL]
FALLBACK_HIGH_THRESHOLD = 0.35
DOWNLOAD_SKEW_SECONDS = 120
MIN_DOWNLOAD_BYTES = 1024
PHASE = "E7_pan_europe_locked_audit_strong"
ORIGINAL_EXTERNAL_TILES = ["E30N35", "E37N42", "E42N21", "E45N51"]

SELECTED_LOCKED_TILES: list[dict[str, str]] = [
    {
        "tile_id": "E34N39",
        "planned_region": "Northwest/Atlantic",
        "country_note": "Scotland / United Kingdom (centroid proxy)",
        "selection_role": "high-coverage positive control",
        "selection_note": "northwest high-coverage anchor chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E27N24",
        "planned_region": "Northwest/Atlantic",
        "country_note": "Atlantic Iberia / offshore fringe (centroid proxy)",
        "selection_role": "lower-coverage Atlantic audit",
        "selection_note": "northwest lower-support tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E40N31",
        "planned_region": "Central/Continental",
        "country_note": "Benelux–Germany corridor (centroid proxy)",
        "selection_role": "central dense urban-industrial",
        "selection_note": "central Europe dense tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E44N30",
        "planned_region": "Central/Continental",
        "country_note": "Germany–Czechia belt (centroid proxy)",
        "selection_role": "central medium-support transfer",
        "selection_note": "central Europe medium-support tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E48N32",
        "planned_region": "Central/Continental",
        "country_note": "Poland / east-central Europe (centroid proxy)",
        "selection_role": "east-central transfer extension",
        "selection_note": "east-central tile added to widen continental spread without using labels",
    },
    {
        "tile_id": "E44N43",
        "planned_region": "Northern/Nordic",
        "country_note": "Sweden–Norway border zone (centroid proxy)",
        "selection_role": "nordic dense support",
        "selection_note": "nordic dense tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E49N44",
        "planned_region": "Northern/Nordic",
        "country_note": "Finland / Nordic interior (centroid proxy)",
        "selection_role": "nordic moderate-support audit",
        "selection_note": "nordic moderate-support tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E31N16",
        "planned_region": "Mediterranean/Southern",
        "country_note": "Andalusia / southern Spain (centroid proxy)",
        "selection_role": "southern dense positive control",
        "selection_note": "southern dense tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E55N14",
        "planned_region": "Mediterranean/Southern",
        "country_note": "Crete / Greece (centroid proxy)",
        "selection_role": "southern island support audit",
        "selection_note": "southern island tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E39N23",
        "planned_region": "Alpine/Mountain",
        "country_note": "western Alps / southeast France (centroid proxy)",
        "selection_role": "alpine dense support",
        "selection_note": "alpine dense tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E48N26",
        "planned_region": "Alpine/Mountain",
        "country_note": "eastern Alps / Slovakia-Hungary corridor (centroid proxy)",
        "selection_role": "alpine moderate-support audit",
        "selection_note": "alpine moderate-support tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E49N22",
        "planned_region": "Mediterranean/Tectonic",
        "country_note": "western Balkans / Adriatic hinterland (centroid proxy)",
        "selection_role": "tectonic dense support",
        "selection_note": "tectonic dense tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E57N25",
        "planned_region": "Mediterranean/Tectonic",
        "country_note": "Romania–Black Sea corridor (centroid proxy)",
        "selection_role": "tectonic moderate-support transfer",
        "selection_note": "tectonic moderate-support tile chosen from API bbox/filesize metadata only",
    },
    {
        "tile_id": "E58N26",
        "planned_region": "Mediterranean/Tectonic",
        "country_note": "Moldova–Ukraine / Black Sea fringe (centroid proxy)",
        "selection_role": "tectonic lower-support failure-mode audit",
        "selection_note": "tectonic lower-support tile chosen from API bbox/filesize metadata only",
    },
]


STRONG_ROOT = ensure_dir(ensure_dir(PROJECT_ROOT / "outputs" / "tgrs_2019_2023") / "vcea_extra_experiments" / "E7_pan_europe_locked_audit_strong")
CONFIG_ROOT = ensure_dir(STRONG_ROOT / "configs")
MANIFEST_ROOT = ensure_dir(STRONG_ROOT / "manifests")
CSV_ROOT = ensure_dir(STRONG_ROOT / "csv")
FIG_ROOT = ensure_dir(STRONG_ROOT / "figures")
LOG_ROOT = ensure_dir(STRONG_ROOT / "logs")
RUN_ROOT = ensure_dir(STRONG_ROOT / "runs")

CONFIG_PATH = CONFIG_ROOT / "config_E7_frozen_audit.yaml"
MANIFEST_PATH = MANIFEST_ROOT / "audit_E7_manifest.json"
API_CATALOG_PATH = MANIFEST_ROOT / "table_E7_api_tile_catalog.csv"
SELECTED_PATH = MANIFEST_ROOT / "table_E7_selected_tiles_prefetch.csv"
METADATA_PATH = MANIFEST_ROOT / "table_E7_pan_europe_tile_metadata.csv"
RESULTS_PATH = CSV_ROOT / "table_E7_pan_europe_locked_results_by_tile.csv"
GROUP_SUMMARY_PATH = CSV_ROOT / "table_E7_pan_europe_group_summary.csv"
FAILURE_PATH = CSV_ROOT / "table_E7_pan_europe_failure_cases.csv"
README_PATH = STRONG_ROOT / "README_E7_outputs.md"
REPORT_PATH = STRONG_ROOT / "E7_pan_europe_locked_audit_report.md"
SUMMARY_JSON_PATH = STRONG_ROOT / "run_summary.json"
STDOUT_LOG_PATH = LOG_ROOT / "run_E7_stdout.log"
ENV_LOG_PATH = LOG_ROOT / "environment_E7.txt"
GIT_LOG_PATH = LOG_ROOT / "git_commit_E7.txt"

TILE_MAP_PATH = FIG_ROOT / "fig_E7_pan_europe_tile_map"
SKILL_REGION_PATH = FIG_ROOT / "fig_E7_skill_by_region"
SKILL_REGIME_PATH = FIG_ROOT / "fig_E7_skill_by_regime"
SKILL_COVERAGE_PATH = FIG_ROOT / "fig_E7_skill_vs_coverage"
STRESS_REGION_PATH = FIG_ROOT / "fig_E7_stress_by_region"
FAILURE_FIG_PATH = FIG_ROOT / "fig_E7_failure_case_overview"


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_val in value.items():
                lines.append(f"  {sub_key}: {json.dumps(sub_val, ensure_ascii=False)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {json.dumps(item, ensure_ascii=False)}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_command(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        merged = "\n".join(part for part in [out, err] if part)
        return merged if merged else f"returncode={result.returncode}"
    except Exception as exc:
        return f"unavailable: {exc}"


def save_fig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_service_key() -> dict[str, Any]:
    raw_json = os.environ.get("EGMS_SERVICE_KEY_JSON")
    if raw_json:
        return json.loads(raw_json)
    raw_path = os.environ.get("EGMS_SERVICE_KEY_PATH")
    if raw_path:
        return json.loads(Path(raw_path).read_text(encoding="utf-8-sig"))
    raise RuntimeError(
        "Missing EGMS credentials. Set EGMS_SERVICE_KEY_JSON or EGMS_SERVICE_KEY_PATH before running strong E7."
    )


def get_access_token(service_key: dict[str, Any], skew_seconds: int = DOWNLOAD_SKEW_SECONDS) -> str:
    now = int(time.time()) - int(skew_seconds)
    claim_set = {
        "iss": service_key["client_id"],
        "sub": service_key["user_id"],
        "aud": service_key["token_uri"],
        "iat": now,
        "exp": now + 3600,
    }
    grant = jwt.encode(claim_set, service_key["private_key"].encode("utf-8"), algorithm="RS256")
    result = requests.post(
        service_key["token_uri"],
        headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant},
        timeout=60,
    )
    result.raise_for_status()
    payload = result.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"CLMS token exchange failed: {payload}")
    return str(token)


def build_session(service_key: dict[str, Any]) -> requests.Session:
    access_token = get_access_token(service_key)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}", "Accept": "application/json"})
    return session


def fetch_api_catalog(session: requests.Session) -> pd.DataFrame:
    tile_ids_resp = session.get(f"{API_ENDPOINT}/tile_ids", timeout=120)
    tile_ids_resp.raise_for_status()
    tile_ids = tile_ids_resp.json()
    rows: list[dict[str, Any]] = []
    for idx, tile_id in enumerate(tile_ids, start=1):
        query = {
            "id": None,
            "tileId": tile_id,
            "releases": ["2019-2023"],
            "levels": ["L3"],
            "productType": "ORTHO-UP",
        }
        response = session.post(f"{API_ENDPOINT}/search", data=json.dumps(query), timeout=120)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("status") or not payload.get("hits"):
            continue
        hit = payload["hits"][0]
        bbox = hit["bbox"]
        min_lon, min_lat = bbox[0]
        max_lon, max_lat = bbox[1]
        rows.append(
            {
                "tile_id": tile_id,
                "centroid_lon": (float(min_lon) + float(max_lon)) / 2.0,
                "centroid_lat": (float(min_lat) + float(max_lat)) / 2.0,
                "bbox_min_lon": float(min_lon),
                "bbox_min_lat": float(min_lat),
                "bbox_max_lon": float(max_lon),
                "bbox_max_lat": float(max_lat),
                "filename": hit["filename"],
                "path": hit["path"],
                "filesize_mb": float(hit["filesize"]) / 1024.0 / 1024.0,
                "release": hit["release"],
                "version": hit["version"],
                "sub_release": int(hit.get("sub_release", 0)),
            }
        )
        if idx % 100 == 0:
            print(f"[E7 strong] catalog progress {idx}/{len(tile_ids)}")
    catalog_df = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    catalog_df.to_csv(API_CATALOG_PATH, index=False, encoding="utf-8-sig")
    return catalog_df


def build_selected_prefetch(catalog_df: pd.DataFrame) -> pd.DataFrame:
    selected_df = pd.DataFrame(SELECTED_LOCKED_TILES)
    merged = selected_df.merge(catalog_df, on="tile_id", how="left", validate="one_to_one")
    if merged["filename"].isna().any():
        missing = merged.loc[merged["filename"].isna(), "tile_id"].tolist()
        raise RuntimeError(f"Selected tiles missing from EGMS API catalog: {missing}")
    merged["selection_status"] = "post_hoc_selected_external_locked_audit"
    merged.to_csv(SELECTED_PATH, index=False, encoding="utf-8-sig")
    return merged


def search_tile_download(session: requests.Session, tile_id: str) -> tuple[dict[str, Any], str]:
    query = {
        "id": None,
        "tileId": tile_id,
        "releases": ["2019-2023"],
        "levels": ["L3"],
        "productType": "ORTHO-UP",
    }
    response = session.post(f"{API_ENDPOINT}/search", data=json.dumps(query), timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("status") or not payload.get("hits"):
        raise RuntimeError(f"EGMS search failed for tile {tile_id}: {payload}")
    return payload["hits"][0], str(payload["id"])


def ensure_downloaded(session: requests.Session, tile_id: str, expected_filename: str) -> Path:
    dest = RAW_DIR / expected_filename
    if dest.exists() and dest.stat().st_size > MIN_DOWNLOAD_BYTES:
        return dest
    hit, query_id = search_tile_download(session, tile_id)
    url = f"{API_ENDPOINT}/download/{hit['filename']}?id={query_id}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if tmp.stat().st_size <= MIN_DOWNLOAD_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file looks invalid for tile {tile_id}: {dest.name}")
    tmp.replace(dest)
    return dest


def compute_centroid_latlon(csv_path: Path) -> tuple[float, float]:
    frame = pd.read_csv(csv_path, usecols=["easting", "northing"])
    x = float(frame["easting"].mean())
    y = float(frame["northing"].mean())
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def derive_regime_group(row: pd.Series) -> str:
    valid = float(row["valid_grid_ratio_64"])
    if valid < 0.30:
        return "sparse-support"
    if float(row["regime_coseismic"]) >= 0.20:
        return "abrupt-mixed"
    if float(row["regime_subsidence"]) >= 0.20:
        return "subsidence-mixed"
    if float(row["regime_seasonal"]) >= 0.05:
        return "seasonal-mixed"
    return "mixed-general"


def coverage_group(values: pd.Series) -> pd.Series:
    q1 = float(values.quantile(1 / 3))
    q2 = float(values.quantile(2 / 3))

    def _label(v: float) -> str:
        if v <= q1:
            return "lower-coverage"
        if v <= q2:
            return "middle-coverage"
        return "higher-coverage"

    return values.apply(_label)


def build_tile_metadata(selected_df: pd.DataFrame, bundles: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, pre in selected_df.iterrows():
        tile = str(pre["tile_id"])
        bundle = bundles[tile]
        meta = bundle.metadata
        lat, lon = compute_centroid_latlon(Path(bundle.raw_csv_path))
        zip_path = RAW_DIR / str(pre["filename"])
        row = {
            "tile_id": tile,
            "tile_role": "post_hoc_selected_external_locked_audit",
            "country_note": pre["country_note"],
            "broad_region": pre["planned_region"],
            "selection_role": pre["selection_role"],
            "selection_note": pre["selection_note"],
            "selection_status": pre["selection_status"],
            "centroid_lat": lat,
            "centroid_lon": lon,
            "api_bbox_centroid_lat": float(pre["centroid_lat"]),
            "api_bbox_centroid_lon": float(pre["centroid_lon"]),
            "api_path": pre["path"],
            "api_filesize_mb": float(pre["filesize_mb"]),
            "dominant_regime": meta.assigned_regime,
            "audit_regime_group": "",
            "num_points": int(meta.num_raw_points),
            "frames": int(meta.num_epochs),
            "temporal_span": meta.time_span,
            "valid_grid_ratio_64": float(meta.valid_grid_ratio_64),
            "valid_grid_ratio_128": float(meta.valid_grid_ratio_128),
            "regime_stable": float((bundle.tensors64["regime_map"] == 0).mean()),
            "regime_seasonal": float((bundle.tensors64["regime_map"] == 1).mean()),
            "regime_subsidence": float((bundle.tensors64["regime_map"] == 2).mean()),
            "regime_accelerating": float((bundle.tensors64["regime_map"] == 3).mean()),
            "regime_coseismic": float((bundle.tensors64["regime_map"] == 4).mean()),
            "regime_mixed": float((bundle.tensors64["regime_map"] == 5).mean()),
            "low_coverage_flag": bool(meta.valid_grid_ratio_64 < 0.30),
            "raw_zip_path": str(zip_path),
            "raw_zip_sha256": hash_file(zip_path),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["audit_regime_group"] = df.apply(derive_regime_group, axis=1)
    df["coverage_group"] = coverage_group(df["valid_grid_ratio_64"])
    df = df.sort_values(["broad_region", "tile_id"]).reset_index(drop=True)
    df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")
    return df


def build_external_cache_from_runs(bundle: Any, seed: int, split_name: str) -> dict[str, np.ndarray]:
    m1_dir = RUN_ROOT / "M1" / bundle.tile_id / f"seed_{seed}"
    c3_dir = RUN_ROOT / "C3_opt" / bundle.tile_id / f"seed_{seed}"
    m1 = infer_m1_split(bundle, m1_dir, split_name)
    c3 = infer_c3_split(bundle, c3_dir, split_name)
    if not np.array_equal(m1["current_frames"].reshape(-1), c3["current_frames"].reshape(-1)):
        raise ValueError(f"Frame mismatch tile={bundle.tile_id} seed={seed} split={split_name}")
    if not np.array_equal(m1["target_indices"], c3["target_indices"]):
        raise ValueError(f"Target mismatch tile={bundle.tile_id} seed={seed} split={split_name}")
    tensors = bundle.tensors64
    return {
        "y_true_mm": m1["y_true_mm"].astype(np.float32),
        "persistence_mm": m1["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": m1["linear_seasonal_mm"].astype(np.float32),
        "m1_pred_mm": m1["y_pred_mm"].astype(np.float32),
        "c3_pred_mm": c3["y_pred_mm"].astype(np.float32),
        "m1_gate_mm": m1["gate_weights"][:, :, 2].astype(np.float32),
        "c3_scale_mm": c3["scale_mm"].astype(np.float32),
        "valid_mask": m1["valid_mask"].astype(np.float32),
        "regime_map": m1["regime_map"].astype(np.int16),
        "current_frames": m1["current_frames"].reshape(-1).astype(np.int64),
        "target_indices": m1["target_indices"].astype(np.int64),
        "last_disp_mm": m1["persistence_mm"][:, 0:1].astype(np.float32),
        "descriptor_amp": tensors["descriptor_amp"].astype(np.float32),
        "trend_r2": tensors["trend_r2"].astype(np.float32),
        "residual_std": tensors["residual_std"].astype(np.float32),
    }


def build_predictions(main_bundles: dict[str, Any], external_bundles: dict[str, Any], selected_tiles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    thresholds, threshold_diag = calibrate_thresholds(main_bundles, SAFE_TARGET, SAFE_PERCENTILE)
    threshold_diag.to_csv(MANIFEST_ROOT / "threshold_diag_main6_validation.csv", index=False, encoding="utf-8-sig")

    raw_frames: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for tile in selected_tiles:
        bundle = external_bundles[tile]
        for seed in FINAL_SEEDS:
            m1_cfg = replace(m1_scalar_config(), seed=seed)
            c3_cfg = replace(c3_runtime_optimized_config(), seed=seed)
            m1_dir = RUN_ROOT / "M1" / tile / f"seed_{seed}"
            c3_dir = RUN_ROOT / "C3_opt" / tile / f"seed_{seed}"

            train_eval_single_tile(bundle, m1_cfg, m1_dir, PHASE, resolution=64, reuse_if_exists=True)
            train_eval_rescue_single_tile(bundle, c3_cfg, c3_dir, PHASE, reuse_if_exists=True)

            val_cache = build_external_cache_from_runs(bundle, seed, "val")
            test_cache = build_external_cache_from_runs(bundle, seed, "test")
            vcea_pred, _weights = static_prediction_from_external(val_cache, test_cache)
            coea_pred, _coea_weights = coea_prediction(test_cache, val_cache)
            ss_pred, fallback_mask, risk = apply_ss_wrapper(test_cache, vcea_pred, coea_pred, thresholds)

            for model_name, pred in [
                ("Persistence", test_cache["persistence_mm"]),
                ("Linear-Seasonal", test_cache["linear_seasonal_mm"]),
                ("M1", test_cache["m1_pred_mm"]),
                ("C3_opt", test_cache["c3_pred_mm"]),
                ("VCEA", vcea_pred),
                (SS_LABEL, ss_pred),
            ]:
                raw_frames.append(rows_from_prediction(model_name, tile, seed, PHASE, test_cache, pred))

            valid = test_cache["valid_mask"] > 0.5
            safe_risk, disagreement, correction, local_vol = risk_components(test_cache, vcea_pred, coea_pred)
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                fallback_rows.append(
                    {
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Fallback rate": float(fallback_mask[:, h_idx][:, valid].mean()),
                        "Risk mean": float(safe_risk[:, h_idx][:, valid].mean()),
                        "Disagreement mean": float(disagreement[:, h_idx][:, valid].mean()),
                        "Correction mean": float(correction[:, h_idx][:, valid].mean()),
                        "Local volatility mean": float(local_vol[:, h_idx][:, valid].mean()),
                        "Risk threshold": float(thresholds[h_idx]),
                    }
                )
    raw_df = pd.concat(raw_frames, ignore_index=True)
    fallback_df = pd.DataFrame(fallback_rows)
    return raw_df, fallback_df, thresholds


def enrich_summary(summary_df: pd.DataFrame, metadata_df: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["Stress ratio"] = out["RMSE [mm] mean"] / out["Persistence RMSE [mm] mean"]
    out = out.merge(metadata_df, left_on="Tile", right_on="tile_id", how="left")
    fallback_summary = fallback_df.groupby(["Tile", "Horizon"], as_index=False).mean(numeric_only=True)
    out = out.merge(fallback_summary, on=["Tile", "Horizon"], how="left")

    m1 = out[out["Model"] == "M1"][["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean"]].rename(
        columns={"Skill_MSE mean": "M1 Skill_MSE mean", "RMSE [mm] mean": "M1 RMSE [mm] mean"}
    )
    vcea = out[out["Model"] == "VCEA"][["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean"]].rename(
        columns={"Skill_MSE mean": "VCEA Skill_MSE mean", "RMSE [mm] mean": "VCEA RMSE [mm] mean"}
    )
    out = out.merge(m1, on=["Tile", "Horizon"], how="left").merge(vcea, on=["Tile", "Horizon"], how="left")
    out["Gain vs M1"] = out["Skill_MSE mean"] - out["M1 Skill_MSE mean"]
    out["Gain/Loss vs VCEA"] = out["Skill_MSE mean"] - out["VCEA Skill_MSE mean"]
    return out


def build_group_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    focus = results_df[results_df["Model"].isin(["M1", "VCEA", SS_LABEL])].copy()
    group_defs = [
        ("overall", None),
        ("broad_region", "broad_region"),
        ("audit_regime_group", "audit_regime_group"),
        ("coverage_group", "coverage_group"),
    ]
    rows: list[dict[str, Any]] = []
    for group_type, col in group_defs:
        subsets = [("all", focus)] if col is None else [(str(name), part.copy()) for name, part in focus.groupby(col)]
        for group_name, part in subsets:
            tile_count = int(part["Tile"].nunique())
            if tile_count == 0:
                continue
            for model_name, model_part in part.groupby("Model"):
                tile_mean = model_part.groupby("Tile", as_index=False)[["Skill_MSE mean", "Gain vs M1", "Gain/Loss vs VCEA"]].mean(numeric_only=True)
                rows.append(
                    {
                        "Group type": group_type,
                        "Group name": group_name,
                        "Model": model_name,
                        "Num tiles": tile_count,
                        "Num tile-horizon rows": int(model_part.shape[0]),
                        "Mean Skill_MSE": float(model_part["Skill_MSE mean"].mean()),
                        "Mean Skill_MSE std": float(model_part["Skill_MSE std"].mean()),
                        "Mean RMSE [mm]": float(model_part["RMSE [mm] mean"].mean()),
                        "Mean MAE [mm]": float(model_part["MAE [mm] mean"].mean()),
                        "Mean Gain vs M1": float(model_part["Gain vs M1"].mean()),
                        "Mean Gain/Loss vs VCEA": float(model_part["Gain/Loss vs VCEA"].mean()),
                        "Better tiles vs M1": int((tile_mean["Gain vs M1"] > 0).sum()),
                        "Better tiles vs VCEA": int((tile_mean["Gain/Loss vs VCEA"] > 0).sum()),
                        "Stress ratio": float(model_part["Stress ratio"].max()),
                        "Mean fallback rate": float(model_part["Fallback rate"].mean()) if model_name == SS_LABEL else float("nan"),
                        "No severe horizon collapse": bool((model_part.groupby("Horizon")["Skill_MSE mean"].mean() > -0.25).all()),
                    }
                )
    group_df = pd.DataFrame(rows)
    group_df.to_csv(GROUP_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return group_df


def build_failure_table(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Tile",
        "Horizon",
        "Model",
        "Broad region",
        "Regime group",
        "Coverage group",
        "Skill_MSE mean",
        "Gain vs M1",
        "Gain/Loss vs VCEA",
        "Stress ratio",
        "Fallback rate",
        "Failure reason",
        "Likely cause note",
    ]
    focus = results_df[results_df["Model"].isin(["VCEA", SS_LABEL])].copy()
    failures: list[dict[str, Any]] = []
    for _, row in focus.iterrows():
        reasons: list[str] = []
        if row["Model"] == "VCEA" and float(row["Gain vs M1"]) <= 0:
            reasons.append("VCEA_not_better_than_M1")
        if float(row["Skill_MSE mean"]) < 0:
            reasons.append("negative_skill")
        if row["Model"] == SS_LABEL and float(row["Stress ratio"]) > 1.0:
            reasons.append("stress_ratio_gt_1")
        if row["Model"] == SS_LABEL and pd.notna(row["Fallback rate"]) and float(row["Fallback rate"]) > FALLBACK_HIGH_THRESHOLD:
            reasons.append("high_fallback_rate")
        if not reasons:
            continue
        support_notes: list[str] = []
        if bool(row["low_coverage_flag"]):
            support_notes.append("low_coverage")
        if float(row["regime_coseismic"]) >= 0.15:
            support_notes.append("abrupt_support")
        if int(row["frames"]) < 150:
            support_notes.append("short_temporal_span")
        if not support_notes:
            support_notes.append("mixed_regime_or_other")
        failures.append(
            {
                "Tile": row["Tile"],
                "Horizon": row["Horizon"],
                "Model": row["Model"],
                "Broad region": row["broad_region"],
                "Regime group": row["audit_regime_group"],
                "Coverage group": row["coverage_group"],
                "Skill_MSE mean": float(row["Skill_MSE mean"]),
                "Gain vs M1": float(row["Gain vs M1"]),
                "Gain/Loss vs VCEA": float(row["Gain/Loss vs VCEA"]),
                "Stress ratio": float(row["Stress ratio"]),
                "Fallback rate": float(row["Fallback rate"]) if pd.notna(row["Fallback rate"]) else float("nan"),
                "Failure reason": ";".join(reasons),
                "Likely cause note": ";".join(support_notes),
            }
        )
    failure_df = pd.DataFrame(failures, columns=columns)
    failure_df.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")
    return failure_df


def centroid_from_local_tile(tile_id: str) -> tuple[float, float]:
    raw_files = extract_raw_archives()
    csv_path = raw_files[tile_id]["csv"]
    lat, lon = compute_centroid_latlon(csv_path)
    return lat, lon


def plot_tile_map(metadata_df: pd.DataFrame) -> None:
    rows = []
    for tile in TILE_ORDER:
        lat, lon = centroid_from_local_tile(tile)
        rows.append({"tile_id": tile, "lat": lat, "lon": lon, "group": "main-six"})
    for tile in ORIGINAL_EXTERNAL_TILES:
        lat, lon = centroid_from_local_tile(tile)
        rows.append({"tile_id": tile, "lat": lat, "lon": lon, "group": "original external"})
    for _, row in metadata_df.iterrows():
        rows.append({"tile_id": row["tile_id"], "lat": row["centroid_lat"], "lon": row["centroid_lon"], "group": "new locked tiles"})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11.2, 6.9))
    color_map = {"main-six": "#4c78a8", "original external": "#72b7b2", "new locked tiles": "#f58518"}
    for group_name, part in df.groupby("group"):
        ax.scatter(part["lon"], part["lat"], s=85, label=group_name, color=color_map[group_name], edgecolor="black", linewidth=0.6)
        for _, row in part.iterrows():
            ax.text(row["lon"] + 0.18, row["lat"] + 0.12, row["tile_id"], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("Held-out tiles in the geographically stratified 14-tile audit", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    save_fig(fig, TILE_MAP_PATH)


def plot_skill_by_region(group_df: pd.DataFrame) -> None:
    vcea = group_df[(group_df["Group type"] == "broad_region") & (group_df["Model"] == "VCEA")].copy()
    ss = group_df[(group_df["Group type"] == "broad_region") & (group_df["Model"] == SS_LABEL)].copy()
    merged = vcea.merge(
        ss[["Group name", "Mean Skill_MSE", "Mean fallback rate"]],
        on="Group name",
        how="left",
        suffixes=(" (VCEA)", " (SS)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].bar(merged["Group name"], merged["Mean Gain vs M1"], color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("VCEA gain over M1 by broad region", fontsize=12)
    axes[0].set_ylabel(r"Mean Skill$_{\mathrm{MSE}}$ gain", fontsize=11)
    axes[0].tick_params(axis="x", rotation=20, labelsize=10)
    axes[0].tick_params(axis="y", labelsize=10)
    axes[1].bar(merged["Group name"], merged["Mean fallback rate (SS)"], color="#f58518")
    axes[1].set_title("SS-VCEA fallback rate by broad region", fontsize=12)
    axes[1].set_ylabel("Fallback rate", fontsize=11)
    axes[1].tick_params(axis="x", rotation=20, labelsize=10)
    axes[1].tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    save_fig(fig, SKILL_REGION_PATH)


def plot_skill_by_regime(group_df: pd.DataFrame) -> None:
    vcea = group_df[(group_df["Group type"] == "audit_regime_group") & (group_df["Model"] == "VCEA")].copy()
    ss = group_df[(group_df["Group type"] == "audit_regime_group") & (group_df["Model"] == SS_LABEL)].copy()
    merged = vcea.merge(
        ss[["Group name", "Stress ratio", "Mean fallback rate"]],
        on="Group name",
        how="left",
        suffixes=(" (VCEA)", " (SS)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(merged["Group name"], merged["Mean Gain vs M1"], color="#54a24b")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("VCEA gain vs M1 by regime group")
    axes[0].set_ylabel("Mean gain")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(merged["Group name"], merged["Stress ratio (SS)"], color="#e45756", alpha=0.85, label="Stress ratio")
    axes[1].plot(merged["Group name"], merged["Mean fallback rate (SS)"], color="#4c78a8", marker="o", label="Fallback rate")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("SS-VCEA stress/fallback by regime group")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_fig(fig, SKILL_REGIME_PATH)


def plot_skill_vs_coverage(results_df: pd.DataFrame) -> None:
    tile_level = (
        results_df[results_df["Model"] == "VCEA"]
        .groupby(["Tile", "broad_region", "valid_grid_ratio_64"], as_index=False)[["Gain vs M1"]]
        .mean(numeric_only=True)
    )
    colors = {
        "Northwest/Atlantic": "#4c78a8",
        "Central/Continental": "#54a24b",
        "Northern/Nordic": "#72b7b2",
        "Mediterranean/Southern": "#f58518",
        "Alpine/Mountain": "#b279a2",
        "Mediterranean/Tectonic": "#e45756",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for _, row in tile_level.iterrows():
        ax.scatter(
            row["valid_grid_ratio_64"],
            row["Gain vs M1"],
            s=80,
            color=colors.get(row["broad_region"], "#999999"),
            edgecolor="black",
            linewidth=0.6,
        )
        ax.text(row["valid_grid_ratio_64"] + 0.004, row["Gain vs M1"] + 0.002, row["Tile"], fontsize=8.5)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valid coverage (64x64)", fontsize=11)
    ax.set_ylabel(r"VCEA Skill$_{\mathrm{MSE}}$ gain over M1", fontsize=11)
    ax.set_title("VCEA gain versus valid coverage in the 14-tile pan-European frozen audit", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.25)
    save_fig(fig, SKILL_COVERAGE_PATH)


def plot_stress_by_region(group_df: pd.DataFrame) -> None:
    ss = group_df[(group_df["Group type"] == "broad_region") & (group_df["Model"] == SS_LABEL)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].bar(ss["Group name"], ss["Stress ratio"], color="#e45756")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("SS-VCEA stress ratio by region")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(ss["Group name"], ss["Mean fallback rate"], color="#72b7b2")
    axes[1].set_title("SS-VCEA fallback rate by region")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_fig(fig, STRESS_REGION_PATH)


def plot_failure_overview(results_df: pd.DataFrame) -> None:
    tile_order = metadata_order(results_df)
    horizon_order = [f"t+{h}" for h in SEARCH_HORIZONS]
    vcea = results_df[results_df["Model"] == "VCEA"].pivot(index="Tile", columns="Horizon", values="Gain vs M1").reindex(index=tile_order, columns=horizon_order)
    ss = results_df[results_df["Model"] == SS_LABEL].pivot(index="Tile", columns="Horizon", values="Stress ratio").reindex(index=tile_order, columns=horizon_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    im0 = axes[0].imshow(vcea.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-0.12, vmax=max(0.12, float(np.nanmax(vcea.to_numpy()))))
    axes[0].set_title(r"VCEA Skill$_{\mathrm{MSE}}$ gain vs M1")
    axes[0].set_xticks(range(len(horizon_order)), horizon_order)
    axes[0].set_yticks(range(len(tile_order)), tile_order)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ss.to_numpy(), cmap="magma", aspect="auto", vmin=0.95, vmax=max(1.08, float(np.nanmax(ss.to_numpy()))))
    axes[1].set_title("SS-VCEA stress ratio")
    axes[1].set_xticks(range(len(horizon_order)), horizon_order)
    axes[1].set_yticks(range(len(tile_order)), tile_order)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, FAILURE_FIG_PATH)


def metadata_order(results_df: pd.DataFrame) -> list[str]:
    order = (
        results_df[["Tile", "broad_region", "valid_grid_ratio_64"]]
        .drop_duplicates()
        .sort_values(["broad_region", "valid_grid_ratio_64", "Tile"], ascending=[True, False, True])["Tile"]
        .tolist()
    )
    return order


def build_report(metadata_df: pd.DataFrame, group_df: pd.DataFrame, failure_df: pd.DataFrame, thresholds: np.ndarray) -> None:
    overall_vcea = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == "VCEA")].iloc[0]
    overall_ss = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == SS_LABEL)].iloc[0]
    lines = [
        "# E7 Pan-European Locked Audit Strong Version 实验报告",
        "",
        EGMS_STATEMENT,
        "",
        "## 实验定位",
        "- 本轮按照 `VCEA_E7_Pan_European_Locked_Audit_Plan` 的强版本执行。新增 tile 集合在模型搜索、VCEA calibration 和 SS-VCEA q97.5 阈值全部冻结之后确定。",
        f"- 新增 locked external tiles 数量：`{metadata_df.shape[0]}`。",
        f"- 新增 tile ID：`{', '.join(metadata_df['tile_id'].tolist())}`。",
        "- 选择依据只使用官方 EGMS API 返回的 tile ID、bbox 位置和压缩文件大小代理，不使用任何新增 tile 的标签表现做二次筛选。",
        "",
        "## Frozen 设置",
        f"- VCEA config: `{FINAL_VCEA_CONFIG.model_id}`。",
        f"- COEA config: `{FINAL_COEA_CONFIG.model_id}`。",
        f"- SS-VCEA safe target: `{SAFE_TARGET}`；percentile: `{SAFE_PERCENTILE}`。",
        f"- Risk 定义: `equal-normalized risk = (disagreement + correction + local_vol) / 3`。",
        f"- threshold 向量: `{', '.join(f't+{h}={thr:.4f}' for h, thr in zip(SEARCH_HORIZONS, thresholds))}`。",
        "",
        "## Audit 覆盖",
        f"- broad region 覆盖: `{', '.join(sorted(metadata_df['broad_region'].unique()))}`。",
        f"- regime group 覆盖: `{', '.join(sorted(metadata_df['audit_regime_group'].unique()))}`。",
        f"- lower coverage tiles: `{int(metadata_df['low_coverage_flag'].sum())}` / `{int(metadata_df.shape[0])}`。",
        "",
        "## 主要结果",
        f"- VCEA overall Mean Skill_MSE = `{float(overall_vcea['Mean Skill_MSE']):.4f}`。",
        f"- VCEA gain vs M1 = `{float(overall_vcea['Mean Gain vs M1']):.4f}`，better tiles vs M1 = `{int(overall_vcea['Better tiles vs M1'])}/{int(metadata_df.shape[0])}`。",
        f"- SS-VCEA overall Mean Skill_MSE = `{float(overall_ss['Mean Skill_MSE']):.4f}`，gain/loss vs VCEA = `{float(overall_ss['Mean Gain/Loss vs VCEA']):.4f}`。",
        f"- SS-VCEA stress ratio = `{float(overall_ss['Stress ratio']):.4f}`，mean fallback rate = `{float(overall_ss['Mean fallback rate']):.4f}`。",
        f"- no severe horizon collapse: `VCEA={bool(overall_vcea['No severe horizon collapse'])}`，`SS-VCEA={bool(overall_ss['No severe horizon collapse'])}`。",
        "",
        "## 论文口径建议",
        "- 这轮结果可以写成 `Pan-European locked audit on additional post-hoc selected external tiles under frozen calibration and threshold settings`。",
        "- 仍然不建议写成 `validated across Europe`；更稳妥的表述是 `positive average transfer beyond the development and original external sets`，并保留 low-coverage / abrupt regime failure mode 讨论。",
        "",
        "## Failure cases",
        f"- failure rows: `{int(failure_df.shape[0])}`。",
        "- failure 表保留 `negative skill`、`VCEA not better than M1`、`stress ratio > 1` 和 `high fallback rate` 情形，不做有利筛选。",
        "",
        "## 关键产物",
        f"- metadata: `{METADATA_PATH.as_posix()}`",
        f"- by-tile results: `{RESULTS_PATH.as_posix()}`",
        f"- group summary: `{GROUP_SUMMARY_PATH.as_posix()}`",
        f"- failure table: `{FAILURE_PATH.as_posix()}`",
        f"- manifest: `{MANIFEST_PATH.as_posix()}`",
        f"- config: `{CONFIG_PATH.as_posix()}`",
    ]
    write_markdown(REPORT_PATH, lines)


def build_readme() -> None:
    lines = [
        "# README E7 strong outputs",
        "",
        EGMS_STATEMENT,
        "",
        "本目录为 `VCEA_E7_Pan_European_Locked_Audit_Plan` 的强版本执行结果。",
        "",
        "## 说明",
        "- 新增 tiles 通过官方 EGMS archive API 下载。",
        "- expert pool、VCEA calibration、SS-VCEA q97.5 阈值、预处理与评价脚本全部保持 frozen。",
        "- 新增 tiles 的确定只依赖官方 API 的 tile catalog 元信息，不依赖新增 tile 的标签表现。",
        "",
        "## 目录结构",
        "- `configs/config_E7_frozen_audit.yaml`: 冻结参数与审计设置。",
        "- `manifests/table_E7_api_tile_catalog.csv`: 官方 API 全 tile catalog。",
        "- `manifests/table_E7_selected_tiles_prefetch.csv`: 固定新增 tile 清单与选择说明。",
        "- `manifests/table_E7_pan_europe_tile_metadata.csv`: 下载并预处理后的 tile 元数据与 regime/coverage 审计。",
        "- `csv/table_E7_pan_europe_locked_results_by_tile.csv`: 全模型、全 tile、全 horizon 指标表。",
        "- `csv/table_E7_pan_europe_group_summary.csv`: region/regime/coverage 分组汇总。",
        "- `csv/table_E7_pan_europe_failure_cases.csv`: failure case 明细。",
        "- `figures/*.png|pdf`: 地图、分组图、coverage scatter、stress 图和 heatmap。",
        "- `runs/`: 新 external locked tiles 上的 M1 与 C3_opt 运行缓存。",
        "",
        "## 复现实验命令",
        "```powershell",
        "$env:EGMS_SERVICE_KEY_PATH='C:/path/to/token.json'",
        "python run_vcea_e7_pan_europe_audit_strong.py",
        "```",
    ]
    write_markdown(README_PATH, lines)


def write_logs(manifest: dict[str, Any]) -> None:
    env_lines = [
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        f"requests: {requests.__version__}",
        f"jwt: {jwt.__version__}",
    ]
    ENV_LOG_PATH.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    GIT_LOG_PATH.write_text(str(manifest.get("git_commit", "unavailable")) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dir(STRONG_ROOT)
    service_key = load_service_key()
    session = build_session(service_key)
    catalog_df = fetch_api_catalog(session)
    selected_prefetch_df = build_selected_prefetch(catalog_df)

    downloaded: list[Path] = []
    for _, row in selected_prefetch_df.iterrows():
        downloaded.append(ensure_downloaded(session, str(row["tile_id"]), str(row["filename"])))

    main_bundles = preprocess_all_tiles()
    raw_files = extract_raw_archives()
    selected_tiles = selected_prefetch_df["tile_id"].tolist()
    external_bundles = {tile: build_bundle_from_tile_id(tile, raw_files) for tile in selected_tiles}

    metadata_df = build_tile_metadata(selected_prefetch_df, external_bundles)
    raw_df, fallback_df, thresholds = build_predictions(main_bundles, external_bundles, selected_tiles)

    summary_df = summarize_seeded_rows(raw_df)
    results_df = enrich_summary(summary_df, metadata_df, fallback_df)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    group_df = build_group_summary(results_df)
    failure_df = build_failure_table(results_df)

    plot_tile_map(metadata_df)
    plot_skill_by_region(group_df)
    plot_skill_by_regime(group_df)
    plot_skill_vs_coverage(results_df)
    plot_stress_by_region(group_df)
    plot_failure_overview(results_df)

    git_commit = run_command(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    manifest = {
        "experiment_id": "E7_pan_europe_locked_audit_strong",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_strong_external_locked_audit",
        "egms_statement": EGMS_STATEMENT,
        "source_docs": {
            "clms_api_download_docs": "https://eea.github.io/clms-api-docs/download.html",
            "clms_api_auth_docs": "https://eea.github.io/clms-api-docs/authentication.html",
            "egms_product_page": "https://land.copernicus.eu/en/products/european-ground-motion-service/egms-ortho",
            "official_egms_api_notebook": "https://github.com/copernicus-land/egms-api",
        },
        "selection_policy": "post_hoc selected external locked audit; fixed after model/calibration/threshold freeze; chosen using only official API tile catalog metadata",
        "selected_tile_count": int(metadata_df.shape[0]),
        "selected_tiles": metadata_df["tile_id"].tolist(),
        "selected_regions": sorted(metadata_df["broad_region"].unique().tolist()),
        "frozen_objects": {
            "expert_pool": ["Persistence", "Linear-Seasonal", "M1", "C3_opt"],
            "vcea_config": asdict(FINAL_VCEA_CONFIG),
            "ss_safe_target": SAFE_TARGET,
            "ss_percentile": SAFE_PERCENTILE,
            "ss_thresholds": [float(x) for x in thresholds.tolist()],
            "coea_config": asdict(FINAL_COEA_CONFIG),
        },
        "data_hashes": metadata_df[["tile_id", "raw_zip_path", "raw_zip_sha256"]].to_dict(orient="records"),
        "git_commit": git_commit,
        "script_path": str(Path(__file__).resolve()),
        "output_root": str(STRONG_ROOT),
        "downloaded_files": [str(path) for path in downloaded],
    }
    write_json(MANIFEST_PATH, manifest)
    write_yaml(
        CONFIG_PATH,
        {
            "experiment_id": "E7_pan_europe_locked_audit_strong",
            "mode": "frozen_external_locked_audit_strong",
            "selected_tiles": selected_tiles,
            "seeds": FINAL_SEEDS,
            "horizons": SEARCH_HORIZONS,
            "vcea_config": asdict(FINAL_VCEA_CONFIG),
            "coea_config": asdict(FINAL_COEA_CONFIG),
            "ss_safe_target": SAFE_TARGET,
            "ss_percentile": SAFE_PERCENTILE,
            "ss_label": SS_LABEL,
            "risk_definition": "equal-normalized risk = (disagreement + correction + local_vol) / 3",
        },
    )

    write_logs(manifest)
    build_readme()
    build_report(metadata_df, group_df, failure_df, thresholds)

    overall_vcea = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == "VCEA")].iloc[0]
    overall_ss = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == SS_LABEL)].iloc[0]
    summary = {
        "status": "completed_strong_external_locked_audit",
        "selected_tile_count": int(metadata_df.shape[0]),
        "vcea_mean_skill_mse": float(overall_vcea["Mean Skill_MSE"]),
        "vcea_gain_vs_m1": float(overall_vcea["Mean Gain vs M1"]),
        "vcea_better_tiles_vs_m1": int(overall_vcea["Better tiles vs M1"]),
        "ss_mean_skill_mse": float(overall_ss["Mean Skill_MSE"]),
        "ss_gain_vs_vcea": float(overall_ss["Mean Gain/Loss vs VCEA"]),
        "ss_stress_ratio": float(overall_ss["Stress ratio"]),
        "failure_rows": int(failure_df.shape[0]),
        "downloaded_new_tiles": int(len(downloaded)),
    }
    write_json(SUMMARY_JSON_PATH, summary)
    STDOUT_LOG_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
