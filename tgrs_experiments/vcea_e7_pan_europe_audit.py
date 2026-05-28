from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS
from .act_v3 import summarize_seeded_rows
from .pipeline import OUTPUT_ROOT, TILE_ORDER, ensure_dir, preprocess_all_tiles
from .tgrs_vcea_hardening import (
    EGMS_STATEMENT,
    FINAL_COEA_CONFIG,
    FINAL_VCEA_CONFIG,
    SAFE_MODEL_LABEL,
    apply_ss_wrapper,
    build_external_cache,
    calibrate_thresholds,
    coea_prediction,
    load_external_bundles,
    risk_components,
    static_prediction_from_external,
)
from .staticval_ea_study import rows_from_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E7_ROOT = ensure_dir(OUTPUT_ROOT / "vcea_extra_experiments" / "E7_pan_europe_locked_audit")
CONFIG_ROOT = ensure_dir(E7_ROOT / "configs")
MANIFEST_ROOT = ensure_dir(E7_ROOT / "manifests")
CSV_ROOT = ensure_dir(E7_ROOT / "csv")
FIG_ROOT = ensure_dir(E7_ROOT / "figures")
LOG_ROOT = ensure_dir(E7_ROOT / "logs")

CONFIG_PATH = CONFIG_ROOT / "config_E7_frozen_audit.yaml"
MANIFEST_PATH = MANIFEST_ROOT / "audit_E7_manifest.json"
METADATA_PATH = MANIFEST_ROOT / "table_E7_pan_europe_tile_metadata.csv"
RESULTS_PATH = CSV_ROOT / "table_E7_pan_europe_locked_results_by_tile.csv"
GROUP_SUMMARY_PATH = CSV_ROOT / "table_E7_pan_europe_group_summary.csv"
FAILURE_PATH = CSV_ROOT / "table_E7_pan_europe_failure_cases.csv"
README_PATH = E7_ROOT / "README_E7_outputs.md"
REPORT_PATH = E7_ROOT / "E7_pan_europe_locked_audit_report.md"
SUMMARY_JSON_PATH = E7_ROOT / "run_summary.json"
STDOUT_LOG_PATH = LOG_ROOT / "run_E7_stdout.log"
ENV_LOG_PATH = LOG_ROOT / "environment_E7.txt"
GIT_LOG_PATH = LOG_ROOT / "git_commit_E7.txt"

TILE_MAP_PATH = FIG_ROOT / "fig_E7_pan_europe_tile_map"
SKILL_REGION_PATH = FIG_ROOT / "fig_E7_skill_by_region"
SKILL_REGIME_PATH = FIG_ROOT / "fig_E7_skill_by_regime"
SKILL_COVERAGE_PATH = FIG_ROOT / "fig_E7_skill_vs_coverage"
STRESS_REGION_PATH = FIG_ROOT / "fig_E7_stress_by_region"
FAILURE_FIG_PATH = FIG_ROOT / "fig_E7_failure_case_overview"

AUDIT_MODELS = ["Persistence", "Linear-Seasonal", "M1", "C3_opt", "VCEA", "SS-VCEA-COEA_q97.5"]
SS_LABEL = "SS-VCEA-COEA_q97.5"
SAFE_TARGET = "COEA"
SAFE_PERCENTILE = 97.5
FALLBACK_HIGH_THRESHOLD = 0.35
MIN_GAIN_STRONG = 0.0


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


def external_zip_lookup() -> dict[str, Path]:
    root = PROJECT_ROOT / "data" / "raw"
    lookup: dict[str, Path] = {}
    for zip_path in root.glob("EGMS_L3_*_100km_U_2019_2023_*.zip"):
        parts = zip_path.stem.split("_")
        if len(parts) >= 3:
            lookup[parts[2]] = zip_path
    return lookup


def compute_centroid_latlon(csv_path: str) -> tuple[float, float]:
    frame = pd.read_csv(csv_path, usecols=["easting", "northing"])
    x = float(frame["easting"].mean())
    y = float(frame["northing"].mean())
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def assign_region(lat: float, lon: float) -> str:
    if lat >= 65:
        return "Northern/Nordic"
    if lat < 45:
        return "Mediterranean/Southern"
    if lon < 0:
        return "Northwest/Atlantic"
    if 45 <= lat < 60 and 0 <= lon <= 18:
        return "Central/Continental"
    return "Other/Peripheral"


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


def save_fig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_tile_metadata(external_bundles: dict[str, Any]) -> pd.DataFrame:
    zip_lookup = external_zip_lookup()
    rows: list[dict[str, Any]] = []
    for tile, bundle in sorted(external_bundles.items()):
        meta = bundle.metadata
        lat, lon = compute_centroid_latlon(bundle.raw_csv_path)
        row = {
            "tile_id": tile,
            "tile_role": "external_locked_eval_available_local",
            "country_note": "not resolved from local metadata",
            "broad_region": assign_region(lat, lon),
            "centroid_lat": lat,
            "centroid_lon": lon,
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
            "selection_rationale": "all available local external locked tiles under EGMS 2019-2023 constraint",
            "locked_status": "frozen external reused; no additional local tiles available",
            "raw_zip_path": str(zip_lookup.get(tile, "")),
            "raw_zip_sha256": hash_file(zip_lookup[tile]) if tile in zip_lookup else "",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["audit_regime_group"] = df.apply(derive_regime_group, axis=1)
    df["coverage_group"] = coverage_group(df["valid_grid_ratio_64"])
    return df.sort_values("tile_id").reset_index(drop=True)


def build_predictions(main_bundles: dict[str, Any], external_bundles: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    thresholds, threshold_diag = calibrate_thresholds(main_bundles, SAFE_TARGET, SAFE_PERCENTILE)
    threshold_diag.to_csv(MANIFEST_ROOT / "threshold_diag_main6_validation.csv", index=False, encoding="utf-8-sig")

    raw_frames: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    for tile, bundle in sorted(external_bundles.items()):
        for seed in FINAL_SEEDS:
            val_cache = build_external_cache(bundle, seed, "val")
            test_cache = build_external_cache(bundle, seed, "test")

            vcea_pred, _weights = static_prediction_from_external(val_cache, test_cache)
            coea_pred, _coea_weights = coea_prediction(test_cache, val_cache)
            ss_pred, fallback_mask, risk = apply_ss_wrapper(test_cache, vcea_pred, coea_pred, thresholds)

            model_payload = [
                ("Persistence", test_cache["persistence_mm"]),
                ("Linear-Seasonal", test_cache["linear_seasonal_mm"]),
                ("M1", test_cache["m1_pred_mm"]),
                ("C3_opt", test_cache["c3_pred_mm"]),
                ("VCEA", vcea_pred),
                (SS_LABEL, ss_pred),
            ]
            for model_name, pred in model_payload:
                raw_frames.append(rows_from_prediction(model_name, tile, seed, "E7_pan_europe_locked_audit", test_cache, pred))

            valid = test_cache["valid_mask"] > 0.5
            safe_pred = coea_pred
            safe_risk, disagreement, correction, local_vol = risk_components(test_cache, vcea_pred, safe_pred)
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                fallback_rate = float(fallback_mask[:, h_idx][:, valid].mean())
                mean_risk = float(safe_risk[:, h_idx][:, valid].mean())
                fallback_rows.append(
                    {
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Fallback rate": fallback_rate,
                        "Risk mean": mean_risk,
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
        if col is None:
            subsets = [("all", focus)]
        else:
            subsets = [(str(name), part.copy()) for name, part in focus.groupby(col)]
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
    return pd.DataFrame(rows)


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
        if row["Model"] == "VCEA" and row["Gain vs M1"] <= 0:
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
    return pd.DataFrame(failures, columns=columns)


def plot_tile_map(metadata_df: pd.DataFrame) -> None:
    main_manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "tile_manifest.csv")
    main_manifest = main_manifest[main_manifest["tile_role"] == "main6"].copy()
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    rows = []
    extracted_root = PROJECT_ROOT / "data" / "extracted_2019_2023"
    for tile in main_manifest["tile_id"]:
        csv_path = next((extracted_root / tile).glob("*.csv"))
        frame = pd.read_csv(csv_path, usecols=["easting", "northing"])
        lon, lat = transformer.transform(float(frame["easting"].mean()), float(frame["northing"].mean()))
        rows.append({"tile_id": tile, "lat": lat, "lon": lon, "group": "main-six"})
    for _, row in metadata_df.iterrows():
        rows.append({"tile_id": row["tile_id"], "lat": row["centroid_lat"], "lon": row["centroid_lon"], "group": "frozen external audit"})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    color_map = {"main-six": "#4c78a8", "frozen external audit": "#f58518"}
    for group_name, part in df.groupby("group"):
        ax.scatter(part["lon"], part["lat"], s=90, label=group_name, color=color_map[group_name], edgecolor="black", linewidth=0.6)
        for _, row in part.iterrows():
            ax.text(row["lon"] + 0.3, row["lat"] + 0.15, row["tile_id"], fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("E7 frozen Europe-scale audit map (available local tiles)")
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(merged["Group name"], merged["Mean Gain vs M1"], color="#4c78a8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("VCEA gain vs M1 by region")
    axes[0].set_ylabel("Mean gain")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(merged["Group name"], merged["Mean fallback rate (SS)"], color="#f58518")
    axes[1].set_title("SS-VCEA fallback rate by region")
    axes[1].set_ylabel("Fallback rate")
    axes[1].tick_params(axis="x", rotation=20)
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
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
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {
        "Northwest/Atlantic": "#4c78a8",
        "Northern/Nordic": "#72b7b2",
        "Mediterranean/Southern": "#f58518",
        "Central/Continental": "#54a24b",
        "Other/Peripheral": "#b279a2",
    }
    for _, row in tile_level.iterrows():
        ax.scatter(row["valid_grid_ratio_64"], row["Gain vs M1"], s=80, color=colors.get(row["broad_region"], "#999999"), edgecolor="black", linewidth=0.6)
        ax.text(row["valid_grid_ratio_64"] + 0.005, row["Gain vs M1"] + 0.002, row["Tile"], fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Valid coverage (64x64)")
    ax.set_ylabel("VCEA gain vs M1")
    ax.set_title("E7 coverage sensitivity under frozen external audit")
    ax.grid(True, alpha=0.25)
    save_fig(fig, SKILL_COVERAGE_PATH)


def plot_stress_by_region(group_df: pd.DataFrame) -> None:
    ss = group_df[(group_df["Group type"] == "broad_region") & (group_df["Model"] == SS_LABEL)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
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
    vcea = results_df[results_df["Model"] == "VCEA"].pivot(index="Tile", columns="Horizon", values="Gain vs M1")
    ss = results_df[results_df["Model"] == SS_LABEL].pivot(index="Tile", columns="Horizon", values="Stress ratio")
    tile_order = sorted(results_df["Tile"].unique())
    horizon_order = [f"t+{h}" for h in SEARCH_HORIZONS]
    vcea = vcea.reindex(index=tile_order, columns=horizon_order)
    ss = ss.reindex(index=tile_order, columns=horizon_order)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].imshow(vcea.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-0.1, vmax=max(0.1, float(np.nanmax(vcea.to_numpy()))))
    axes[0].set_title("VCEA gain vs M1")
    axes[0].set_xticks(range(len(horizon_order)), horizon_order)
    axes[0].set_yticks(range(len(tile_order)), tile_order)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(ss.to_numpy(), cmap="magma", aspect="auto", vmin=0.95, vmax=max(1.05, float(np.nanmax(ss.to_numpy()))))
    axes[1].set_title("SS-VCEA stress ratio")
    axes[1].set_xticks(range(len(horizon_order)), horizon_order)
    axes[1].set_yticks(range(len(tile_order)), tile_order)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_fig(fig, FAILURE_FIG_PATH)


def build_report(metadata_df: pd.DataFrame, results_df: pd.DataFrame, group_df: pd.DataFrame, failure_df: pd.DataFrame, thresholds: np.ndarray) -> None:
    overall_vcea = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == "VCEA")].iloc[0]
    overall_ss = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == SS_LABEL)].iloc[0]
    mean_gain = float(overall_vcea["Mean Gain vs M1"])
    better_tiles = int(overall_vcea["Better tiles vs M1"])
    lines = [
        "# E7 Pan-European Locked Audit 实验报告",
        "",
        EGMS_STATEMENT,
        "",
        "## 实验定位",
        "- 本轮目标来自 `VCEA_E7_Pan_European_Locked_Audit_Plan.docx`。",
        "- 计划强版本要求新增 12-18 个从未参与开发的欧洲 locked tiles；但当前本地可用数据仅有 `main-six + original external 4` 共 10 个 tile。",
        "- 因此本轮执行的是 **受数据可得性约束的 frozen Europe-scale external audit**：不改 expert pool、不重调 VCEA 权重、不重调 SS-VCEA operating point，只对当前全部可用 external locked tiles 做完整冻结复算与分组审计。",
        "",
        "## Frozen 设置",
        f"- VCEA config: `{FINAL_VCEA_CONFIG.model_id}`。",
        f"- SS-VCEA safe target: `{SAFE_TARGET}`；percentile: `{SAFE_PERCENTILE}`。",
        f"- Risk 定义保持 `equal-normalized risk = (disagreement + correction + local_vol) / 3`。",
        f"- threshold 向量: `{', '.join(f't+{h}={thr:.4f}' for h, thr in zip(SEARCH_HORIZONS, thresholds))}`。",
        "",
        "## Audit 样本",
        f"- external frozen audit tile 数: `{metadata_df.shape[0]}`。",
        f"- broad region 覆盖: `{', '.join(sorted(metadata_df['broad_region'].unique()))}`。",
        f"- regime group 覆盖: `{', '.join(sorted(metadata_df['audit_regime_group'].unique()))}`。",
        "",
        "## 主要结果",
        f"- VCEA overall Mean Skill_MSE = `{float(overall_vcea['Mean Skill_MSE']):.4f}`，gain vs M1 = `{mean_gain:.4f}`，better tiles vs M1 = `{better_tiles}/{int(metadata_df.shape[0])}`。",
        f"- SS-VCEA overall Mean Skill_MSE = `{float(overall_ss['Mean Skill_MSE']):.4f}`，gain/loss vs VCEA = `{float(overall_ss['Mean Gain/Loss vs VCEA']):.4f}`，stress ratio = `{float(overall_ss['Stress ratio']):.4f}`。",
        f"- no severe horizon collapse: `VCEA={(group_df[(group_df['Group type']=='overall') & (group_df['Model']=='VCEA')]['No severe horizon collapse'].iloc[0])}`，`SS-VCEA={(group_df[(group_df['Group type']=='overall') & (group_df['Model']==SS_LABEL)]['No severe horizon collapse'].iloc[0])}`。",
        "",
        "## 论文口径建议",
        "- 当前证据支持把这轮结果写成 `Europe-scale frozen external audit under available local external tiles`，而不是 `validated across Europe`。",
        "- 若需要计划文档中的强版本 `Pan-European locked audit`，后续必须新增本地不存在的外部 tile 数据源。",
        "",
        "## Failure cases",
        f"- failure rows: `{int(failure_df.shape[0])}`。",
        "- failure 表中保留了 `VCEA not better than M1`、`negative skill`、`stress ratio > 1`、`high fallback rate` 等情形；没有删去不利 tile。",
        "",
        "## 产物目录",
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
        "# README E7 outputs",
        "",
        EGMS_STATEMENT,
        "",
        "本目录为 `VCEA_E7_Pan_European_Locked_Audit_Plan` 的受限版执行结果。",
        "限制说明：本地仅有 `main-six + original external 4` 共 10 个 tile，没有新增的欧洲 raw tile，因此本轮是 frozen external audit 的可执行最小版。",
        "",
        "## 目录结构",
        "- `configs/config_E7_frozen_audit.yaml`: 冻结参数与审计设置。",
        "- `manifests/audit_E7_manifest.json`: 数据 hash、脚本版本、冻结状态、时间戳。",
        "- `manifests/table_E7_pan_europe_tile_metadata.csv`: tile 元数据、region、coverage、regime group。",
        "- `csv/table_E7_pan_europe_locked_results_by_tile.csv`: 全模型、全 tile、全 horizon 指标表。",
        "- `csv/table_E7_pan_europe_group_summary.csv`: region/regime/coverage 分组汇总。",
        "- `csv/table_E7_pan_europe_failure_cases.csv`: failure case 明细。",
        "- `figures/*.png|pdf`: 地图、分组图、coverage scatter、stress 图。",
        "- `logs/*`: 运行日志、环境、git commit。",
        "",
        "## 复现实验命令",
        "```powershell",
        "python run_vcea_e7_pan_europe_audit.py",
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
    ]
    ENV_LOG_PATH.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    git_text = manifest.get("git_commit", "unavailable")
    GIT_LOG_PATH.write_text(str(git_text) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    ensure_dir(E7_ROOT)

    main_bundles = preprocess_all_tiles()
    external_bundles = load_external_bundles()
    metadata_df = build_tile_metadata(external_bundles)
    metadata_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    raw_df, fallback_df, thresholds = build_predictions(main_bundles, external_bundles)
    summary_df = summarize_seeded_rows(raw_df)
    results_df = enrich_summary(summary_df, metadata_df, fallback_df)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    group_df = build_group_summary(results_df)
    group_df.to_csv(GROUP_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    failure_df = build_failure_table(results_df)
    failure_df.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    plot_tile_map(metadata_df)
    plot_skill_by_region(group_df)
    plot_skill_by_regime(group_df)
    plot_skill_vs_coverage(results_df)
    plot_stress_by_region(group_df)
    plot_failure_overview(results_df)

    git_commit = run_command(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    manifest = {
        "experiment_id": "E7_pan_europe_locked_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_constrained_local_only",
        "egms_statement": EGMS_STATEMENT,
        "scope_note": "No additional local raw tiles beyond main-six + original external 4 were available; audit executed on all available frozen external tiles.",
        "frozen_objects": {
            "expert_pool": ["Persistence", "Linear-Seasonal", "M1", "C3_opt"],
            "vcea_config": asdict(FINAL_VCEA_CONFIG),
            "ss_safe_target": SAFE_TARGET,
            "ss_percentile": SAFE_PERCENTILE,
            "ss_thresholds": [float(x) for x in thresholds.tolist()],
            "coea_config": asdict(FINAL_COEA_CONFIG),
        },
        "external_tile_count": int(metadata_df.shape[0]),
        "external_tiles": metadata_df["tile_id"].tolist(),
        "external_regions": sorted(metadata_df["broad_region"].unique().tolist()),
        "data_hashes": metadata_df[["tile_id", "raw_zip_path", "raw_zip_sha256"]].to_dict(orient="records"),
        "git_commit": git_commit,
        "script_path": str(Path(__file__).resolve()),
        "output_root": str(E7_ROOT),
    }
    write_json(MANIFEST_PATH, manifest)
    write_yaml(
        CONFIG_PATH,
        {
            "experiment_id": "E7_pan_europe_locked_audit",
            "mode": "frozen_external_audit_constrained_local_only",
            "external_tiles": metadata_df["tile_id"].tolist(),
            "seeds": FINAL_SEEDS,
            "horizons": SEARCH_HORIZONS,
            "vcea_config": asdict(FINAL_VCEA_CONFIG),
            "ss_safe_target": SAFE_TARGET,
            "ss_percentile": SAFE_PERCENTILE,
            "ss_label": SS_LABEL,
            "risk_definition": "equal-normalized risk = (disagreement + correction + local_vol) / 3",
        },
    )

    write_logs(manifest)
    build_readme()
    build_report(metadata_df, results_df, group_df, failure_df, thresholds)

    overall_vcea = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == "VCEA")].iloc[0]
    overall_ss = group_df[(group_df["Group type"] == "overall") & (group_df["Group name"] == "all") & (group_df["Model"] == SS_LABEL)].iloc[0]
    summary = {
        "status": "completed_constrained_local_only",
        "external_tile_count": int(metadata_df.shape[0]),
        "vcea_mean_skill_mse": float(overall_vcea["Mean Skill_MSE"]),
        "vcea_gain_vs_m1": float(overall_vcea["Mean Gain vs M1"]),
        "vcea_better_tiles_vs_m1": int(overall_vcea["Better tiles vs M1"]),
        "ss_mean_skill_mse": float(overall_ss["Mean Skill_MSE"]),
        "ss_gain_vs_vcea": float(overall_ss["Mean Gain/Loss vs VCEA"]),
        "ss_stress_ratio": float(overall_ss["Stress ratio"]),
        "failure_rows": int(failure_df.shape[0]),
        "new_external_tiles_available_locally": 0,
    }
    write_json(SUMMARY_JSON_PATH, summary)
    STDOUT_LOG_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
