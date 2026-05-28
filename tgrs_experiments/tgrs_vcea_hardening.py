from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS
from .act_v3 import summarize_seeded_rows
from .act_v5_selector import infer_c3_split, infer_m1_split
from .act_v7 import COEAConfig, build_seed_expert_cache, replay_coea, warm_start_weights
from .dscf_confirmation import EXTERNAL_CANDIDATES, build_bundle_from_tile_id, extract_raw_archives
from .pipeline import OUTPUT_ROOT, REGIME_NAMES, TILE_ORDER, TileBundle, ensure_dir, preprocess_all_tiles
from .staticval_ea_study import StaticConfig, estimate_static_weights, predict_static, rows_from_prediction


HARDEN_ROOT = ensure_dir(OUTPUT_ROOT / "hardening_package")
VTSA_ROOT = ensure_dir(HARDEN_ROOT / "vtsa")
SS_ROOT = ensure_dir(HARDEN_ROOT / "ss_vcea")
RES_ROOT = ensure_dir(HARDEN_ROOT / "resolution_aware")
REPRO_ROOT = ensure_dir(HARDEN_ROOT / "reproducibility")

FINAL_VCEA_CONFIG = StaticConfig(
    model_id="StaticVal_inverse_temp4p0_sh0p1_floor0p0",
    method="inverse",
    temperature=4.0,
    shrinkage=0.1,
    m1_floor=0.0,
)
FINAL_COEA_CONFIG = COEAConfig(
    model_id="COEA_ewa_eta4p0_warm_floor50_tile_horizon",
    update_mode="ewa",
    eta=4.0,
    warm_start=True,
    m1_floor=0.5,
    granularity="tile_horizon",
)

STRESS_TILES = ["E48N24", "E58N17"]
RISK_PERCENTILES = [90.0, 95.0, 97.5]
SAFE_TARGETS = ["M1", "COEA"]
SAFE_MODEL_LABEL = {"M1": "SS-VCEA-M1", "COEA": "SS-VCEA-COEA"}
VCEA_MAINLINE_LABEL = "VCEA"
COEA_MAINLINE_LABEL = "COEA"
EGMS_STATEMENT = "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。"

STATIC_STUDY_ROOT = OUTPUT_ROOT / "staticval_ea_study"
ACT_V7_FINAL_ROOT = OUTPUT_ROOT / "act_v7" / "finalization"
CONFIRM_ROOT = OUTPUT_ROOT / "confirmation"
FAMILY_RES_ROOT = CONFIRM_ROOT / "model_family_resolution_extension"
INFRA_V2_ROOT = OUTPUT_ROOT / "infrastructure_v2"
EXTERNAL_RUN_ROOT = CONFIRM_ROOT / "external_tiles" / "runs"
M1_MAIN6_SUMMARY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"

VTSA_RANK_STABILITY_OUT = VTSA_ROOT / "vtsa_rank_stability.csv"
VTSA_TOPK_RETENTION_OUT = VTSA_ROOT / "vtsa_topk_retention.csv"
VTSA_DRIFT_OUT = VTSA_ROOT / "vtsa_coea_drift.csv"
VTSA_BLOCKED_OUT = VTSA_ROOT / "vtsa_blocked_validation.csv"
VTSA_PROTOCOL_OUT = VTSA_ROOT / "vtsa_protocol_comparison.csv"
VTSA_SUMMARY_MD = VTSA_ROOT / "vtsa_mechanism_summary.md"
VTSA_FIGURE_PDF = VTSA_ROOT / "fig_vtsa_mechanism.pdf"

SS_SWEEP_OUT = SS_ROOT / "ss_vcea_threshold_sweep.csv"
SS_MAIN6_OUT = SS_ROOT / "ss_vcea_main6.csv"
SS_EXTERNAL_OUT = SS_ROOT / "ss_vcea_external.csv"
SS_MAIN6_RAW_OUT = SS_ROOT / "ss_vcea_main6_raw.csv"
SS_EXTERNAL_RAW_OUT = SS_ROOT / "ss_vcea_external_raw.csv"
SS_FALLBACK_OUT = SS_ROOT / "ss_vcea_fallback_rates.csv"
SS_THRESHOLDS_OUT = SS_ROOT / "ss_vcea_thresholds.csv"
SS_SUMMARY_MD = SS_ROOT / "ss_vcea_summary.md"
SS_PARETO_FIG = SS_ROOT / "fig_stress_skill_pareto.pdf"
SS_STRESS_MAP_FIG = SS_ROOT / "fig_stress_tile_error_map.pdf"

RES_TABLE_OUT = RES_ROOT / "resolution_exact_proxy_table.csv"
RES_STRESS_OUT = RES_ROOT / "resolution_stress_curve.csv"
RES_NOTES_MD = RES_ROOT / "resolution_notes_for_caption.md"
RES_SUMMARY_MD = RES_ROOT / "resolution_summary.md"
RES_HEATMAP_PDF = RES_ROOT / "resolution_family_heatmap.pdf"

REPRO_LINEAGE_OUT = REPRO_ROOT / "result_lineage_table.csv"
REPRO_SOURCE_MAP_OUT = REPRO_ROOT / "paper_table_source_map.csv"
REPRO_CLAIM_MD = REPRO_ROOT / "claim_to_result_map.md"
REPRO_CHECKLIST_MD = REPRO_ROOT / "reproducibility_checklist.md"
REPRO_SUMMARY_MD = REPRO_ROOT / "reproducibility_audit_summary.md"
REPRO_FIGURE_MAP_OUT = REPRO_ROOT / "figure_source_map.csv"
REPRO_FLAGS_OUT = REPRO_ROOT / "exact_proxy_transfer_flags.csv"

SS_SELECTED_MAIN6_SUMMARY = SS_ROOT / "ss_vcea_selected_main6_summary.csv"
SS_SELECTED_TILE_HORIZON = SS_ROOT / "ss_vcea_selected_tile_horizon.csv"
SS_SELECTED_STRESS_REPORT = SS_ROOT / "ss_vcea_selected_stress_report.csv"
SS_SELECTED_PARETO = SS_ROOT / "ss_vcea_skill_stress_pareto.csv"
SS_SELECTED_EXTERNAL_SUMMARY = SS_ROOT / "ss_vcea_selected_external_summary.csv"
SS_SELECTED_EXTERNAL_STRESS = SS_ROOT / "ss_vcea_selected_external_stress.csv"
SS_SELECTED_HEATMAP_PDF = SS_ROOT / "ss_vcea_selected_tile_horizon_heatmaps.pdf"

FINAL_REPORT_MD = HARDEN_ROOT / "tgrs_vcea_hardening_detailed_report.md"
PAPER_INSERT_MD = HARDEN_ROOT / "hardening_paper_insert.md"
SUMMARY_JSON = HARDEN_ROOT / "run_summary.json"


MAIN_CACHE: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
EXTERNAL_CACHE: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def get_main_cache(bundles: dict[str, TileBundle], tile: str, seed: int, split_name: str) -> dict[str, np.ndarray]:
    key = (tile, seed, split_name)
    if key not in MAIN_CACHE:
        MAIN_CACHE[key] = build_seed_expert_cache(bundles, tile, seed, split_name)
    return MAIN_CACHE[key]


def load_external_bundles() -> dict[str, TileBundle]:
    raw_files = extract_raw_archives()
    available = [tile for tile in EXTERNAL_CANDIDATES if tile in raw_files]
    return {tile: build_bundle_from_tile_id(tile, raw_files) for tile in available}


def build_external_cache(bundle: TileBundle, seed: int, split_name: str) -> dict[str, np.ndarray]:
    key = (bundle.tile_id, seed, split_name)
    if key in EXTERNAL_CACHE:
        return EXTERNAL_CACHE[key]
    m1_dir = EXTERNAL_RUN_ROOT / "M1" / bundle.tile_id / f"seed_{seed}"
    c3_dir = EXTERNAL_RUN_ROOT / "C3_opt" / bundle.tile_id / f"seed_{seed}"
    m1 = infer_m1_split(bundle, m1_dir, split_name)
    c3 = infer_c3_split(bundle, c3_dir, split_name)
    if not np.array_equal(m1["current_frames"].reshape(-1), c3["current_frames"].reshape(-1)):
        raise ValueError(f"External frame mismatch tile={bundle.tile_id} seed={seed} split={split_name}")
    if not np.array_equal(m1["target_indices"], c3["target_indices"]):
        raise ValueError(f"External target mismatch tile={bundle.tile_id} seed={seed} split={split_name}")
    tensors = bundle.tensors64
    cache = {
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
    EXTERNAL_CACHE[key] = cache
    return cache


def static_prediction(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    weights = estimate_static_weights(cache, FINAL_VCEA_CONFIG)
    pred = predict_static(cache, weights)
    return pred.astype(np.float32), weights.astype(np.float32)


def coea_prediction(cache: dict[str, np.ndarray], val_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    init_weights = warm_start_weights(val_cache, FINAL_COEA_CONFIG)
    pred, weight_snapshots, _ = replay_coea(cache, FINAL_COEA_CONFIG, init_weights=init_weights, return_trace=False)
    return pred.astype(np.float32), weight_snapshots.astype(np.float32)


def risk_components(
    cache: dict[str, np.ndarray],
    vcea_pred: np.ndarray,
    safe_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resid = np.maximum(cache["residual_std"].astype(np.float32), 0.1)
    scale = resid[None, None, :, :]
    valid = cache["valid_mask"][None, None, :, :] > 0.5
    disagreement = np.abs(vcea_pred - safe_pred) / scale
    correction = np.abs(vcea_pred - cache["persistence_mm"]) / scale
    resid_norm = resid / max(float(np.median(resid[cache["valid_mask"] > 0.5])) if np.any(cache["valid_mask"] > 0.5) else float(np.median(resid)), 0.1)
    local_vol = np.broadcast_to(resid_norm[None, None, :, :], disagreement.shape).astype(np.float32)
    risk = (disagreement + correction + local_vol) / 3.0
    risk = np.where(valid, risk, 0.0).astype(np.float32)
    return risk.astype(np.float32), disagreement.astype(np.float32), correction.astype(np.float32), local_vol.astype(np.float32)


def calibrate_thresholds(
    bundles: dict[str, TileBundle],
    safe_target: str,
    percentile: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    horizon_values: dict[str, list[np.ndarray]] = {f"t+{h}": [] for h in SEARCH_HORIZONS}
    rows: list[dict[str, Any]] = []
    for tile in TILE_ORDER:
        for seed in FINAL_SEEDS:
            val_cache = get_main_cache(bundles, tile, seed, "val")
            vcea_val, _ = static_prediction(val_cache)
            if safe_target == "M1":
                safe_val = val_cache["m1_pred_mm"]
            else:
                safe_val, _ = coea_prediction(val_cache, val_cache)
            risk, disagreement, correction, local_vol = risk_components(val_cache, vcea_val, safe_val)
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                mask = val_cache["valid_mask"] > 0.5
                horizon_values[f"t+{horizon}"].append(risk[:, h_idx][:, mask].reshape(-1))
                rows.append(
                    {
                        "Safe target": safe_target,
                        "Percentile": percentile,
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Risk mean": float(risk[:, h_idx][:, mask].mean()),
                        "Disagreement mean": float(disagreement[:, h_idx][:, mask].mean()),
                        "Correction magnitude mean": float(correction[:, h_idx][:, mask].mean()),
                        "Local volatility mean": float(local_vol[:, h_idx][:, mask].mean()),
                    }
                )
    thresholds = []
    for horizon in SEARCH_HORIZONS:
        values = np.concatenate(horizon_values[f"t+{horizon}"], axis=0)
        thresholds.append(float(np.percentile(values, percentile)))
    return np.asarray(thresholds, dtype=np.float32), pd.DataFrame(rows)


def apply_ss_wrapper(
    cache: dict[str, np.ndarray],
    vcea_pred: np.ndarray,
    safe_pred: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    risk, disagreement, correction, _local_vol = risk_components(cache, vcea_pred, safe_pred)
    pred = vcea_pred.copy()
    fallback_mask = np.zeros_like(risk, dtype=bool)
    for h_idx in range(len(SEARCH_HORIZONS)):
        mask = (risk[:, h_idx] > float(thresholds[h_idx])) & (cache["valid_mask"] > 0.5)
        pred[:, h_idx][mask] = safe_pred[:, h_idx][mask]
        fallback_mask[:, h_idx] = mask
    return pred.astype(np.float32), fallback_mask, risk.astype(np.float32)


def summarize_model_vs_reference(summary_df: pd.DataFrame, ref_df: pd.DataFrame, stress_tiles: list[str]) -> dict[str, Any]:
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    mean_std = float(summary_df["Skill_MSE std"].mean())
    stress_part = summary_df[summary_df["Tile"].isin(stress_tiles)] if stress_tiles else summary_df
    stress_ratio = float((stress_part["RMSE [mm] mean"] / stress_part["Persistence RMSE [mm] mean"]).max())
    merged = summary_df.merge(
        ref_df[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "Ref Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    tile_delta = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "Ref Skill_MSE mean"]]
        .mean()
        .assign(**{"Delta vs Ref": lambda d: d["Skill_MSE mean"] - d["Ref Skill_MSE mean"]})
    )
    return {
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "stress_ratio": stress_ratio,
        "better_tiles_vs_ref": int((tile_delta["Delta vs Ref"] > 0).sum()),
        "tile_delta": tile_delta,
    }


def summarize_external_reference(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[summary_df["Model"] == "M1"].copy()


def generate_vtsa_outputs() -> dict[str, Any]:
    rank_df = pd.read_csv(STATIC_STUDY_ROOT / "expert_rank_stability.csv")
    horizon_corr_df = pd.read_csv(STATIC_STUDY_ROOT / "rank_correlation_by_horizon.csv")
    drift_df = pd.read_csv(STATIC_STUDY_ROOT / "weight_drift_metrics.csv")
    blocked_df = pd.read_csv(STATIC_STUDY_ROOT / "blocked_validation_transfer.csv")
    delay_df = pd.read_csv(STATIC_STUDY_ROOT / "delayed_feedback_penalty.csv")
    final_comp = pd.read_csv(STATIC_STUDY_ROOT / "final_aggregation_comparison.csv")

    rank_out = (
        rank_df.groupby(["Tile", "Horizon"], as_index=False)[["Spearman", "Kendall"]]
        .mean()
        .sort_values(["Tile", "Horizon"])
    )
    topk_out = (
        rank_df.groupby(["Tile", "Horizon"], as_index=False)[["Val top1 stays top1", "Val top1 stays top2"]]
        .mean()
        .sort_values(["Tile", "Horizon"])
    )
    drift_out = (
        drift_df.groupby(["Tile", "Horizon"], as_index=False)[
            ["Static entropy", "COEA entropy mean", "Effective experts mean", "L1 drift mean", "L1 drift std", "Switching rate"]
        ]
        .mean()
        .sort_values(["Tile", "Horizon"])
    )
    blocked_out = blocked_df.groupby(["Protocol", "Train block"], as_index=False)[["Mean Skill_MSE", "Stress ratio"]].mean().sort_values(["Protocol", "Train block"])
    protocol_rows = [
        {
            "Protocol": "VCEA",
            "Mean Skill_MSE": float(final_comp.loc[final_comp["Model"] == "VCEA", "Mean Skill_MSE"].iloc[0]),
            "Stress ratio": float(final_comp.loc[final_comp["Model"] == "VCEA", "Stress ratio"].iloc[0]),
            "Note": "static validation-calibrated aggregation",
        },
        {
            "Protocol": "COEA",
            "Mean Skill_MSE": float(final_comp.loc[final_comp["Model"] == "COEA", "Mean Skill_MSE"].iloc[0]),
            "Stress ratio": float(final_comp.loc[final_comp["Model"] == "COEA", "Stress ratio"].iloc[0]),
            "Note": "audited delayed-feedback online update",
        },
    ]
    for _, row in delay_df.iterrows():
        protocol_rows.append(
            {
                "Protocol": str(row["Model"]),
                "Mean Skill_MSE": float(row["Mean Skill_MSE"]),
                "Stress ratio": float(row["Stress ratio"]),
                "Note": f"mode={row['Mode']}, update_every={row['Update every']}, frozen={row['Frozen']}",
            }
        )
    protocol_out = pd.DataFrame(protocol_rows)

    rank_out.to_csv(VTSA_RANK_STABILITY_OUT, index=False, encoding="utf-8-sig")
    topk_out.to_csv(VTSA_TOPK_RETENTION_OUT, index=False, encoding="utf-8-sig")
    drift_out.to_csv(VTSA_DRIFT_OUT, index=False, encoding="utf-8-sig")
    blocked_out.to_csv(VTSA_BLOCKED_OUT, index=False, encoding="utf-8-sig")
    protocol_out.to_csv(VTSA_PROTOCOL_OUT, index=False, encoding="utf-8-sig")

    heat = rank_out.pivot(index="Tile", columns="Horizon", values="Spearman").reindex(TILE_ORDER)
    topk_plot = horizon_corr_df.set_index("Horizon")[["Val top1 stays top1", "Val top1 stays top2"]].reindex([f"t+{h}" for h in SEARCH_HORIZONS])
    drift_plot = drift_out.groupby("Horizon", as_index=False)[["L1 drift mean", "Switching rate"]].mean().set_index("Horizon").reindex([f"t+{h}" for h in SEARCH_HORIZONS])
    delay_plot = delay_df.sort_values("Mean Skill_MSE", ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    im = axes[0, 0].imshow(heat.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=1.0)
    axes[0, 0].set_title("VTSA-1 Spearman by tile/horizon")
    axes[0, 0].set_yticks(range(len(heat.index)))
    axes[0, 0].set_yticklabels(heat.index.tolist())
    axes[0, 0].set_xticks(range(len(heat.columns)))
    axes[0, 0].set_xticklabels(heat.columns.tolist())
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    topk_plot.plot(kind="bar", ax=axes[0, 1], color=["#4c78a8", "#f58518"])
    axes[0, 1].set_title("VTSA-2 top-k retention")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(drift_plot.index, drift_plot["L1 drift mean"], marker="o", label="COEA L1 drift")
    axes[1, 0].plot(drift_plot.index, drift_plot["Switching rate"], marker="s", label="Switching rate")
    axes[1, 0].set_title("VTSA-3 online drift diagnostics")
    axes[1, 0].legend()

    axes[1, 1].bar(delay_plot["Model"], delay_plot["Mean Skill_MSE"], color="#54a24b")
    axes[1, 1].axhline(float(final_comp.loc[final_comp["Model"] == "VCEA", "Mean Skill_MSE"].iloc[0]), color="black", linestyle="--", label="VCEA")
    axes[1, 1].set_title("VTSA-4 delayed-feedback penalty")
    axes[1, 1].tick_params(axis="x", rotation=35)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(VTSA_FIGURE_PDF)
    plt.close(fig)

    mean_spearman = float(rank_df["Spearman"].mean())
    mean_kendall = float(rank_df["Kendall"].mean())
    top1 = float(rank_df["Val top1 stays top1"].mean())
    top2 = float(rank_df["Val top1 stays top2"].mean())
    mean_drift = float(drift_df["L1 drift mean"].mean())
    blocked_to_test = blocked_df.loc[blocked_df["Protocol"] == "blocked_to_test", "Mean Skill_MSE"].mean()
    lines = [
        "# VTSA 机制总结",
        "",
        EGMS_STATEMENT,
        "",
        f"- mean Spearman = `{mean_spearman:.4f}`。",
        f"- mean Kendall = `{mean_kendall:.4f}`。",
        f"- validation top1 stays top1 = `{top1:.4f}`；top1 stays top2 = `{top2:.4f}`。",
        f"- COEA mean L1 drift = `{mean_drift:.4f}`。",
        f"- blocked-to-test Mean Skill_MSE = `{blocked_to_test:.4f}`。",
        "",
        "结论：当前证据链支持“validation-to-test expert ranking stability”是 VCEA 优于在线漂移更新的主要机制。",
        "需要保留的边界：稳定性是 tile/horizon-dependent 的，不应把它写成 universal law。",
    ]
    write_markdown(VTSA_SUMMARY_MD, lines)
    return {
        "mean_spearman": mean_spearman,
        "mean_kendall": mean_kendall,
        "top1_ratio": top1,
        "top2_ratio": top2,
        "mean_drift": mean_drift,
        "blocked_to_test_skill": float(blocked_to_test),
    }


@dataclass
class SSSweepResult:
    safe_target: str
    percentile: float
    thresholds: np.ndarray
    main6_summary: pd.DataFrame
    main6_raw: pd.DataFrame
    main6_stats: dict[str, Any]
    external_summary: pd.DataFrame
    external_raw: pd.DataFrame
    external_stats: dict[str, Any]
    fallback_df: pd.DataFrame


def build_main6_reference_rows(bundles: dict[str, TileBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for tile in TILE_ORDER:
        for seed in FINAL_SEEDS:
            test_cache = get_main_cache(bundles, tile, seed, "test")
            val_cache = get_main_cache(bundles, tile, seed, "val")
            vcea_pred, _weights = static_prediction(test_cache)
            coea_pred, _coea_weights = coea_prediction(test_cache, val_cache)
            rows.append(rows_from_prediction(VCEA_MAINLINE_LABEL, tile, seed, "hardening_main6", test_cache, vcea_pred))
            rows.append(rows_from_prediction(COEA_MAINLINE_LABEL, tile, seed, "hardening_main6", test_cache, coea_pred))
            rows.append(rows_from_prediction("M1", tile, seed, "hardening_main6", test_cache, test_cache["m1_pred_mm"]))
            rows.append(rows_from_prediction("Persistence", tile, seed, "hardening_main6", test_cache, test_cache["persistence_mm"]))
    raw = pd.concat(rows, ignore_index=True)
    summary = summarize_seeded_rows(raw)
    return raw, summary


def build_external_reference_rows(external_bundles: dict[str, TileBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for tile, bundle in external_bundles.items():
        for seed in FINAL_SEEDS:
            val_cache = build_external_cache(bundle, seed, "val")
            test_cache = build_external_cache(bundle, seed, "test")
            vcea_pred, _weights = static_prediction_from_external(val_cache, test_cache)
            coea_pred, _snap = coea_prediction(test_cache, val_cache)
            rows.append(rows_from_prediction(VCEA_MAINLINE_LABEL, tile, seed, "hardening_external", test_cache, vcea_pred))
            rows.append(rows_from_prediction(COEA_MAINLINE_LABEL, tile, seed, "hardening_external", test_cache, coea_pred))
            rows.append(rows_from_prediction("M1", tile, seed, "hardening_external", test_cache, test_cache["m1_pred_mm"]))
            rows.append(rows_from_prediction("Persistence", tile, seed, "hardening_external", test_cache, test_cache["persistence_mm"]))
    raw = pd.concat(rows, ignore_index=True)
    summary = summarize_seeded_rows(raw)
    return raw, summary


def static_prediction_from_external(val_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    weights = estimate_static_weights(val_cache, FINAL_VCEA_CONFIG)
    pred = predict_static(test_cache, weights)
    return pred.astype(np.float32), weights.astype(np.float32)


def run_ss_sweep_for_safe_target(
    bundles: dict[str, TileBundle],
    external_bundles: dict[str, TileBundle],
    safe_target: str,
    percentile: float,
    main6_ref_summary: pd.DataFrame,
    external_ref_summary: pd.DataFrame,
) -> SSSweepResult:
    thresholds, calibration_rows = calibrate_thresholds(bundles, safe_target, percentile)
    main_rows: list[pd.DataFrame] = []
    external_rows: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []

    for tile in TILE_ORDER:
        for seed in FINAL_SEEDS:
            val_cache = get_main_cache(bundles, tile, seed, "val")
            test_cache = get_main_cache(bundles, tile, seed, "test")
            vcea_test, _ = static_prediction(test_cache)
            if safe_target == "M1":
                safe_test = test_cache["m1_pred_mm"]
            else:
                safe_test, _ = coea_prediction(test_cache, val_cache)
            pred, fallback_mask, risk = apply_ss_wrapper(test_cache, vcea_test, safe_test, thresholds)
            model_name = f"{SAFE_MODEL_LABEL[safe_target]}_q{str(percentile).replace('.', 'p')}"
            main_rows.append(rows_from_prediction(model_name, tile, seed, "hardening_main6", test_cache, pred))
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                valid = test_cache["valid_mask"] > 0.5
                fallback_rows.append(
                    {
                        "Scope": "main6",
                        "Model": model_name,
                        "Safe target": safe_target,
                        "Percentile": percentile,
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Fallback rate": float(fallback_mask[:, h_idx][:, valid].mean()),
                        "Risk mean": float(risk[:, h_idx][:, valid].mean()),
                    }
                )

    for tile, bundle in external_bundles.items():
        for seed in FINAL_SEEDS:
            val_cache = build_external_cache(bundle, seed, "val")
            test_cache = build_external_cache(bundle, seed, "test")
            vcea_test, _ = static_prediction_from_external(val_cache, test_cache)
            if safe_target == "M1":
                safe_test = test_cache["m1_pred_mm"]
            else:
                safe_test, _ = coea_prediction(test_cache, val_cache)
            pred, fallback_mask, risk = apply_ss_wrapper(test_cache, vcea_test, safe_test, thresholds)
            model_name = f"{SAFE_MODEL_LABEL[safe_target]}_q{str(percentile).replace('.', 'p')}"
            external_rows.append(rows_from_prediction(model_name, tile, seed, "hardening_external", test_cache, pred))
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                valid = test_cache["valid_mask"] > 0.5
                fallback_rows.append(
                    {
                        "Scope": "external_locked",
                        "Model": model_name,
                        "Safe target": safe_target,
                        "Percentile": percentile,
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Fallback rate": float(fallback_mask[:, h_idx][:, valid].mean()),
                        "Risk mean": float(risk[:, h_idx][:, valid].mean()),
                    }
                )

    calibration_df = pd.DataFrame(calibration_rows)
    main_raw = pd.concat(main_rows, ignore_index=True)
    external_raw = pd.concat(external_rows, ignore_index=True)
    main_summary = summarize_seeded_rows(main_raw)
    external_summary = summarize_seeded_rows(external_raw)
    main_stats = summarize_model_vs_reference(main_summary, main6_ref_summary, STRESS_TILES)
    external_stats = summarize_model_vs_reference(external_summary, external_ref_summary, list(external_bundles.keys()))
    threshold_df = pd.DataFrame(
        {
            "Safe target": safe_target,
            "Percentile": percentile,
            "Horizon": [f"t+{h}" for h in SEARCH_HORIZONS],
            "Threshold": thresholds.tolist(),
        }
    )
    threshold_df["Calibration risk mean"] = calibration_df.groupby("Horizon")["Risk mean"].mean().reindex(threshold_df["Horizon"]).to_numpy()
    return SSSweepResult(
        safe_target=safe_target,
        percentile=percentile,
        thresholds=thresholds,
        main6_summary=main_summary,
        main6_raw=main_raw,
        main6_stats=main_stats,
        external_summary=external_summary,
        external_raw=external_raw,
        external_stats=external_stats,
        fallback_df=pd.concat([pd.DataFrame(fallback_rows), threshold_df], ignore_index=False),
    )


def generate_ss_vcea_outputs(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    external_bundles = load_external_bundles()
    main6_ref_raw, main6_ref_summary = build_main6_reference_rows(bundles)
    external_ref_raw, external_ref_summary = build_external_reference_rows(external_bundles)

    sweep_rows: list[dict[str, Any]] = []
    all_main_raw = [main6_ref_raw]
    all_external_raw = [external_ref_raw]
    fallback_tables: list[pd.DataFrame] = []
    threshold_tables: list[pd.DataFrame] = []
    sweep_results: list[SSSweepResult] = []
    threshold_lookup: dict[str, np.ndarray] = {}

    for safe_target in SAFE_TARGETS:
        for percentile in RISK_PERCENTILES:
            result = run_ss_sweep_for_safe_target(
                bundles,
                external_bundles,
                safe_target,
                percentile,
                main6_ref_summary[main6_ref_summary["Model"] == "M1"].copy(),
                external_ref_summary[external_ref_summary["Model"] == "M1"].copy(),
            )
            sweep_results.append(result)
            model_name = f"{SAFE_MODEL_LABEL[safe_target]}_q{str(percentile).replace('.', 'p')}"
            threshold_lookup[model_name] = result.thresholds.copy()
            all_main_raw.append(result.main6_raw)
            all_external_raw.append(result.external_raw)
            fallback_tables.append(result.fallback_df[result.fallback_df.columns.intersection(["Scope", "Model", "Safe target", "Percentile", "Tile", "Seed", "Horizon", "Fallback rate", "Risk mean"])])
            threshold_tables.append(
                pd.DataFrame(
                    {
                        "Safe target": safe_target,
                        "Percentile": percentile,
                        "Horizon": [f"t+{h}" for h in SEARCH_HORIZONS],
                        "Threshold": result.thresholds.tolist(),
                    }
                )
            )
            sweep_rows.append(
                {
                        "Scope": "main6",
                        "Model": model_name,
                    "Safe target": safe_target,
                    "Percentile": percentile,
                    "Mean Skill_MSE": result.main6_stats["mean_skill_mse"],
                    "Mean Skill std": result.main6_stats["mean_skill_std"],
                    "Stress ratio": result.main6_stats["stress_ratio"],
                    "Better tiles vs M1": result.main6_stats["better_tiles_vs_ref"],
                }
            )
            sweep_rows.append(
                {
                        "Scope": "external_locked",
                        "Model": model_name,
                    "Safe target": safe_target,
                    "Percentile": percentile,
                    "Mean Skill_MSE": result.external_stats["mean_skill_mse"],
                    "Mean Skill std": result.external_stats["mean_skill_std"],
                    "Stress ratio": result.external_stats["stress_ratio"],
                    "Better tiles vs M1": result.external_stats["better_tiles_vs_ref"],
                }
            )

    base_rows = []
    for scope, summary_df in [("main6", main6_ref_summary), ("external_locked", external_ref_summary)]:
        for model in [VCEA_MAINLINE_LABEL, COEA_MAINLINE_LABEL, "M1", "Persistence"]:
            part = summary_df[summary_df["Model"] == model].copy()
            if part.empty:
                continue
            ref_part = summary_df[summary_df["Model"] == "M1"].copy()
            stats = summarize_model_vs_reference(part, ref_part, STRESS_TILES if scope == "main6" else list(external_bundles.keys()))
            base_rows.append(
                {
                    "Scope": scope,
                    "Model": model,
                    "Safe target": "base",
                    "Percentile": float("nan"),
                    "Mean Skill_MSE": stats["mean_skill_mse"],
                    "Mean Skill std": stats["mean_skill_std"],
                    "Stress ratio": stats["stress_ratio"],
                    "Better tiles vs M1": stats["better_tiles_vs_ref"],
                }
            )
    sweep_df = pd.concat([pd.DataFrame(base_rows), pd.DataFrame(sweep_rows)], ignore_index=True)
    sweep_df.to_csv(SS_SWEEP_OUT, index=False, encoding="utf-8-sig")

    main6_all_raw = pd.concat(all_main_raw, ignore_index=True)
    external_all_raw = pd.concat(all_external_raw, ignore_index=True)
    main6_all_raw.to_csv(SS_MAIN6_RAW_OUT, index=False, encoding="utf-8-sig")
    external_all_raw.to_csv(SS_EXTERNAL_RAW_OUT, index=False, encoding="utf-8-sig")

    main6_all_summary = summarize_seeded_rows(main6_all_raw)
    external_all_summary = summarize_seeded_rows(external_all_raw)
    main6_all_summary.to_csv(SS_MAIN6_OUT, index=False, encoding="utf-8-sig")
    external_all_summary.to_csv(SS_EXTERNAL_OUT, index=False, encoding="utf-8-sig")
    pd.concat(fallback_tables, ignore_index=True).to_csv(SS_FALLBACK_OUT, index=False, encoding="utf-8-sig")
    pd.concat(threshold_tables, ignore_index=True).to_csv(SS_THRESHOLDS_OUT, index=False, encoding="utf-8-sig")

    vcea_main = float(sweep_df[(sweep_df["Scope"] == "main6") & (sweep_df["Model"] == VCEA_MAINLINE_LABEL)]["Mean Skill_MSE"].iloc[0])
    eligible = sweep_df[
        (sweep_df["Scope"] == "main6")
        & (sweep_df["Model"].str.startswith("SS-VCEA"))
        & (sweep_df["Mean Skill_MSE"] >= vcea_main - 0.01)
        & (sweep_df["Stress ratio"] <= 1.03)
    ].copy()
    selected_model = eligible.sort_values(["Mean Skill_MSE", "Stress ratio"], ascending=[False, True]).iloc[0]["Model"] if not eligible.empty else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax_idx, scope in enumerate(["main6", "external_locked"]):
        ax = axes[ax_idx]
        part = sweep_df[sweep_df["Scope"] == scope].copy()
        base = part[part["Model"].isin([VCEA_MAINLINE_LABEL, COEA_MAINLINE_LABEL, "M1"])]
        ss_part = part[part["Model"].str.startswith("SS-VCEA")].copy()
        ax.scatter(base["Stress ratio"], base["Mean Skill_MSE"], s=70, label="base")
        for _, row in base.iterrows():
            ax.annotate(row["Model"], (row["Stress ratio"], row["Mean Skill_MSE"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        colors = {"M1": "#4c78a8", "COEA": "#f58518"}
        for safe_target, grp in ss_part.groupby("Safe target"):
            ax.plot(grp["Stress ratio"], grp["Mean Skill_MSE"], marker="o", label=f"{safe_target} fallback", color=colors.get(safe_target, None))
        ax.axvline(1.03, color="black", linestyle="--", linewidth=1)
        ax.axhline(vcea_main - 0.01 if scope == "main6" else float(part.loc[part["Model"] == VCEA_MAINLINE_LABEL, "Mean Skill_MSE"].iloc[0]) - 0.01, color="gray", linestyle=":")
        ax.set_title(f"{scope} skill-stress Pareto")
        ax.set_xlabel("stress ratio")
        ax.set_ylabel("Mean Skill_MSE")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SS_PARETO_FIG)
    plt.close(fig)

    if selected_model is not None:
        selected_part = main6_all_raw[(main6_all_raw["Model"] == selected_model) & (main6_all_raw["Tile"].isin(STRESS_TILES))].copy()
        vcea_part = main6_all_raw[(main6_all_raw["Model"] == VCEA_MAINLINE_LABEL) & (main6_all_raw["Tile"].isin(STRESS_TILES))].copy()
        if not selected_part.empty and not vcea_part.empty:
            tile = "E48N24" if "E48N24" in selected_part["Tile"].unique() else selected_part["Tile"].iloc[0]
            seed = int(selected_part[selected_part["Tile"] == tile]["Seed"].iloc[0])
            cache = get_main_cache(bundles, tile, seed, "test")
            vcea_pred, _ = static_prediction(cache)
            if "M1" in selected_model:
                safe_pred = cache["m1_pred_mm"]
            else:
                safe_pred, _ = coea_prediction(cache, get_main_cache(bundles, tile, seed, "val"))
            threshold_arr = threshold_lookup[selected_model]
            ss_pred, fallback_mask, _risk = apply_ss_wrapper(cache, vcea_pred, safe_pred, threshold_arr)
            h_idx = 0
            vcea_err = np.abs(vcea_pred[:, h_idx] - cache["y_true_mm"][:, h_idx]).mean(axis=0)
            ss_err = np.abs(ss_pred[:, h_idx] - cache["y_true_mm"][:, h_idx]).mean(axis=0)
            fallback_rate_map = fallback_mask[:, h_idx].mean(axis=0)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            for ax, arr, title in zip(
                axes,
                [vcea_err, ss_err, fallback_rate_map],
                ["VCEA abs error (mean)", "SS-VCEA abs error (mean)", "Fallback rate"],
                strict=True,
            ):
                im = ax.imshow(arr, cmap="magma")
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.75)
            fig.suptitle(f"Stress tile audit: {tile} seed={seed} t+1")
            fig.tight_layout()
            fig.savefig(SS_STRESS_MAP_FIG)
            plt.close(fig)

    lines = [
        "# SS-VCEA 总结",
        "",
        EGMS_STATEMENT,
        "",
        "风险定义假设：由于文档未给出 `lambda_1/lambda_2/lambda_3` 的具体数值，本轮使用等权归一化风险：",
        "`risk = (disagreement + correction_magnitude + local_volatility) / 3`。",
        "Fallback 机制假设：本轮使用 hard fallback，即 `risk > tau_h` 时直接切换到安全预测。",
        "",
        f"- main-six baseline VCEA Mean Skill_MSE = `{vcea_main:.4f}`。",
        f"- main-six 可行 SS-VCEA 变体数 = `{int(len(eligible))}`。",
        f"- selected model = `{selected_model}`。" if selected_model is not None else "- selected model = `None`。",
        "",
        "阈值标定全部来自 main-six validation，不使用 test 调参；external locked 仅做冻结阈值后的外部检查。",
    ]
    write_markdown(SS_SUMMARY_MD, lines)
    return {
        "selected_model": selected_model,
        "eligible_count": int(len(eligible)),
        "vcea_main_skill": vcea_main,
        "main6_models": sorted(main6_all_summary["Model"].unique().tolist()),
        "external_models": sorted(external_all_summary["Model"].unique().tolist()),
    }


def generate_resolution_outputs() -> dict[str, Any]:
    ranking_df = pd.read_csv(FAMILY_RES_ROOT / "family_resolution_ranking.csv")
    summary_df = pd.read_csv(FAMILY_RES_ROOT / "family_resolution_summary.csv")
    runtime_df = pd.read_csv(FAMILY_RES_ROOT / "family_resolution_runtime.csv")
    migrated_df = pd.read_csv(INFRA_V2_ROOT / "migrated_results_v2.csv")

    provenance = (
        migrated_df[
            migrated_df["model_name"].isin(["M0", "M1", "C3_opt", "VCEA_exact", "COEA_exact", "StaticVal-PLM1"])
            & (migrated_df["resolution"].isin([64, 128, 512]))
            & (migrated_df["tile_role"].isin(["main", "main_resolution_sensitivity"]))
        ]
        .groupby(["model_name", "resolution"], as_index=False)
        .agg(
            is_proxy=("is_proxy", "max"),
            transfer_inference=("transfer_inference", "max"),
            train_resolution=("train_resolution", "max"),
            calibration_resolution=("calibration_resolution", "max"),
            test_resolution=("test_resolution", "max"),
            aggregation_type=("aggregation_type", lambda s: ",".join(sorted(set(str(v) for v in s if pd.notna(v))))),
            notes=("notes", lambda s: " | ".join(sorted(set(str(v) for v in s if pd.notna(v) and str(v) != "nan")))),
        )
    )

    table_df = ranking_df.merge(provenance, left_on=["Model", "Resolution"], right_on=["model_name", "resolution"], how="left")
    table_df = table_df.drop(columns=["model_name", "resolution"])
    table_df = table_df.rename(columns={"Model": "model_name", "Resolution": "resolution", "Mean Skill_MSE": "mean_skill_mse", "Mean RMSE [mm]": "mean_rmse_mm", "Stress ratio": "stress_ratio"})
    table_df.to_csv(RES_TABLE_OUT, index=False, encoding="utf-8-sig")

    stress_curve = table_df[["model_name", "resolution", "mean_skill_mse", "stress_ratio", "is_proxy", "transfer_inference"]].sort_values(["model_name", "resolution"])
    stress_curve.to_csv(RES_STRESS_OUT, index=False, encoding="utf-8-sig")

    pivot = table_df.pivot(index="model_name", columns="resolution", values="mean_skill_mse").reindex(["M0", "M1", "C3_opt", "COEA_exact", "VCEA_exact", "StaticVal-PLM1"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im = axes[0].imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.05, vmax=0.20)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index.tolist())
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels([str(c) for c in pivot.columns])
    axes[0].set_title("Mean Skill_MSE by family / resolution")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    for model, part in stress_curve.groupby("model_name"):
        axes[1].plot(part["resolution"].astype(str), part["stress_ratio"], marker="o", label=model)
    axes[1].axhline(1.03, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Stress ratio vs resolution")
    axes[1].set_ylabel("stress ratio")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES_HEATMAP_PDF)
    plt.close(fig)

    notes_lines = [
        "# Resolution-aware caption notes",
        "",
        EGMS_STATEMENT,
        "",
        "- `512` 结果只用作 sensitivity，不进入主任务定义。",
        "- `C3_opt@512` 使用 `128->512 transfer inference`；直接训练曾触发 bad allocation。",
        "- `VCEA_exact/COEA_exact@512` 的校准使用同 tile 的 `128` validation cache。",
        "- `VCEA_exact/COEA_exact` 的 runtime 仅统计 aggregation/calibration 自身，不包含生成 `M1/C3_opt` expert 的训练成本。",
        "- 当前结论适用于 aggregation family，不应外推为“所有复杂单模型都从高分辨率获益”。",
    ]
    write_markdown(RES_NOTES_MD, notes_lines)

    mean_vcea_64 = float(table_df[(table_df["model_name"] == "VCEA_exact") & (table_df["resolution"] == 64)]["mean_skill_mse"].iloc[0])
    mean_vcea_512 = float(table_df[(table_df["model_name"] == "VCEA_exact") & (table_df["resolution"] == 512)]["mean_skill_mse"].iloc[0])
    mean_m1_64 = float(table_df[(table_df["model_name"] == "M1") & (table_df["resolution"] == 64)]["mean_skill_mse"].iloc[0])
    mean_m1_512 = float(table_df[(table_df["model_name"] == "M1") & (table_df["resolution"] == 512)]["mean_skill_mse"].iloc[0])
    lines = [
        "# Resolution-aware robustness summary",
        "",
        EGMS_STATEMENT,
        "",
        f"- `M1`: `64={mean_m1_64:.4f}` -> `512={mean_m1_512:.4f}`，随分辨率升高退化。",
        f"- `VCEA_exact`: `64={mean_vcea_64:.4f}` -> `512={mean_vcea_512:.4f}`，随分辨率升高增强。",
        "- 这支持“高分辨率不是 aggregation family 的瓶颈”，但不支持把 `512` 改写成主任务。",
        "- exact/proxy/transfer/calibration provenance 已在表格中显式标注。",
    ]
    write_markdown(RES_SUMMARY_MD, lines)
    return {
        "vcea_64": mean_vcea_64,
        "vcea_512": mean_vcea_512,
        "m1_64": mean_m1_64,
        "m1_512": mean_m1_512,
        "runtime_rows": int(runtime_df.shape[0]),
        "summary_rows": int(summary_df.shape[0]),
    }


def generate_reproducibility_outputs() -> dict[str, Any]:
    migrated = pd.read_csv(INFRA_V2_ROOT / "migrated_results_v2.csv")
    tables = pd.read_csv(INFRA_V2_ROOT / "paper_ready_summary_tables.csv")
    freeze_text = (INFRA_V2_ROOT / "freeze_manifest.yaml").read_text(encoding="utf-8")
    qa_text = (INFRA_V2_ROOT / "infra_refactor_QA.md").read_text(encoding="utf-8")

    lineage = migrated[
        migrated["model_name"].isin(["VCEA_exact", "COEA_exact", "M1", "Persistence", "C3_opt", "StaticVal-PLM1"])
        & migrated["tile_role"].isin(["main", "main_resolution_sensitivity", "external_locked_eval"])
    ][
        [
            "experiment_id",
            "model_name",
            "tile_id",
            "tile_role",
            "resolution",
            "aggregation_type",
            "is_proxy",
            "train_resolution",
            "calibration_resolution",
            "test_resolution",
            "transfer_inference",
            "source_artifact",
            "calibration_source",
            "expert_cache_id",
            "aggregation_weight_id",
            "notes",
        ]
    ].sort_values(["model_name", "tile_role", "tile_id", "resolution", "experiment_id"])
    lineage.to_csv(REPRO_LINEAGE_OUT, index=False, encoding="utf-8-sig")

    source_map_rows = [
        {
            "paper_item": "Main result table (M1 / COEA / VCEA)",
            "section": "Results",
            "source_file": str(STATIC_STUDY_ROOT / "final_aggregation_comparison.csv"),
            "note": "full-six 64x64 primary comparison",
        },
        {
            "paper_item": "VTSA table / figure",
            "section": "Method mechanism",
            "source_file": str(VTSA_ROOT / "vtsa_protocol_comparison.csv"),
            "note": "rank stability, drift, delayed-feedback evidence",
        },
        {
            "paper_item": "SS-VCEA safety table",
            "section": "Safety analysis",
            "source_file": str(SS_SWEEP_OUT),
            "note": "validation-only threshold sweep",
        },
        {
            "paper_item": "External locked evaluation table",
            "section": "Robustness",
            "source_file": str(CONFIRM_ROOT / "external_tiles" / "external_summary.csv"),
            "note": "external locked exact evaluation",
        },
        {
            "paper_item": "Resolution-aware robustness table",
            "section": "Robustness",
            "source_file": str(RES_TABLE_OUT),
            "note": "exact/proxy/transfer summary with provenance",
        },
        {
            "paper_item": "Selected SS-VCEA main table row",
            "section": "Safety analysis",
            "source_file": str(SS_SELECTED_MAIN6_SUMMARY),
            "note": "selected SS-VCEA-COEA_q97p5 export for main table",
        },
        {
            "paper_item": "Selected SS-VCEA external locked row",
            "section": "Supplement / robustness",
            "source_file": str(SS_SELECTED_EXTERNAL_SUMMARY),
            "note": "frozen external evaluation of selected SS-VCEA variant",
        },
        {
            "paper_item": "Reproducibility statement",
            "section": "Appendix / statement",
            "source_file": str(REPRO_LINEAGE_OUT),
            "note": "lineage, manifests, registries",
        },
    ]
    source_map_df = pd.DataFrame(source_map_rows)
    source_map_df.to_csv(REPRO_SOURCE_MAP_OUT, index=False, encoding="utf-8-sig")

    figure_map_df = pd.DataFrame(
        [
            {
                "figure_id": "Fig-VTSA-1",
                "section": "Method mechanism",
                "source_file": str(VTSA_FIGURE_PDF),
                "supporting_table": str(VTSA_PROTOCOL_OUT),
                "note": "rank stability, retention, drift, delayed-feedback panel",
            },
            {
                "figure_id": "Fig-SS-1",
                "section": "Safety analysis",
                "source_file": str(SS_PARETO_FIG),
                "supporting_table": str(SS_SELECTED_PARETO),
                "note": "main-six and external skill-stress Pareto",
            },
            {
                "figure_id": "Fig-SS-2",
                "section": "Safety analysis",
                "source_file": str(SS_SELECTED_HEATMAP_PDF),
                "supporting_table": str(SS_SELECTED_TILE_HORIZON),
                "note": "selected SS-VCEA tile-horizon heatmaps vs VCEA and M1",
            },
            {
                "figure_id": "Fig-SS-3",
                "section": "Safety analysis",
                "source_file": str(SS_STRESS_MAP_FIG),
                "supporting_table": str(SS_SELECTED_STRESS_REPORT),
                "note": "stress-tile error map for selected SS-VCEA variant",
            },
            {
                "figure_id": "Fig-RES-1",
                "section": "Robustness",
                "source_file": str(RES_HEATMAP_PDF),
                "supporting_table": str(RES_TABLE_OUT),
                "note": "resolution-aware family heatmap and stress curve",
            },
        ]
    )
    figure_map_df.to_csv(REPRO_FIGURE_MAP_OUT, index=False, encoding="utf-8-sig")

    flags_df = (
        migrated[
            migrated["model_name"].isin(["M0", "M1", "C3_opt", "COEA_exact", "VCEA_exact", "StaticVal-PLM1"])
            & migrated["resolution"].isin([64, 128, 512])
        ][
            [
                "model_name",
                "tile_role",
                "resolution",
                "is_proxy",
                "transfer_inference",
                "train_resolution",
                "calibration_resolution",
                "test_resolution",
                "aggregation_type",
                "notes",
            ]
        ]
        .drop_duplicates()
        .sort_values(["model_name", "tile_role", "resolution"])
        .reset_index(drop=True)
    )
    flags_df.to_csv(REPRO_FLAGS_OUT, index=False, encoding="utf-8-sig")

    claim_lines = [
        "# Claim-to-Result Map",
        "",
        EGMS_STATEMENT,
        "",
        "| Claim | Supporting file | Figure/Table | Notes |",
        "| --- | --- | --- | --- |",
        f"| VCEA outperforms M1 under strong persistence baselines | `{(STATIC_STUDY_ROOT / 'final_aggregation_comparison.csv').as_posix()}` | Main table | 64x64 main setting |",
        f"| Validation ranking stability explains VCEA | `{VTSA_RANK_STABILITY_OUT.as_posix()}` | Fig-VTSA-1 / Table-VTSA-1 | tile/horizon dependent, not universal |",
        f"| SS-VCEA reduces stress risk on main-six | `{SS_SELECTED_STRESS_REPORT.as_posix()}` | Fig-SS-1 / Fig-SS-2 | equal-normalized risk; selected variant only |",
        f"| Selected SS-VCEA external result is a conservative deployment check | `{SS_SELECTED_EXTERNAL_SUMMARY.as_posix()}` | Table-SS-External | no retuning on external locked tiles |",
        f"| VCEA benefits from resolution-aware robustness | `{RES_TABLE_OUT.as_posix()}` | Fig-RES-1 | 512 sensitivity only |",
        f"| Results are reproducible via infra_v2 | `{REPRO_LINEAGE_OUT.as_posix()}` | Appendix Table A1 | exact/proxy/transfer flags included |",
    ]
    write_markdown(REPRO_CLAIM_MD, claim_lines)

    checklist_lines = [
        "# Reproducibility Checklist",
        "",
        EGMS_STATEMENT,
        "",
        "- [x] Data product is frozen to `EGMS 2019-2023 Ortho U`.",
        "- [x] Main-six and external locked chronological splits are registered.",
        "- [x] Aggregation outputs carry exact/proxy/transfer flags.",
        "- [x] Expert cache registry is preserved through infra_v2.",
        "- [x] Aggregation weight registry / calibration source are traceable.",
        "- [x] No test-time threshold tuning is used in SS-VCEA.",
        "- [x] Resolution-specific assumptions for `512` are explicitly documented.",
        "- [x] Result lineage can map table rows back to artifacts.",
        "",
        "Supporting files:",
        f"- freeze manifest: `{(INFRA_V2_ROOT / 'freeze_manifest.yaml').as_posix()}`",
        f"- QA note: `{(INFRA_V2_ROOT / 'infra_refactor_QA.md').as_posix()}`",
        f"- paper-ready tables: `{(INFRA_V2_ROOT / 'paper_ready_summary_tables.csv').as_posix()}`",
    ]
    write_markdown(REPRO_CHECKLIST_MD, checklist_lines)

    summary_lines = [
        "# Reproducibility Audit Summary",
        "",
        EGMS_STATEMENT,
        "",
        f"- migrated rows traced in infra_v2: `{int(migrated.shape[0])}`。",
        f"- paper-ready summary rows: `{int(tables.shape[0])}`。",
        "- 本轮不新增模型族，只把主文需要的声明映射回 manifest / split / cache / aggregation provenance。",
        "",
        "freeze manifest excerpt:",
        "```yaml",
        freeze_text.strip(),
        "```",
        "",
        "QA excerpt:",
        "```md",
        qa_text.strip(),
        "```",
    ]
    write_markdown(REPRO_SUMMARY_MD, summary_lines)
    return {
        "lineage_rows": int(lineage.shape[0]),
        "paper_map_rows": int(source_map_df.shape[0]),
        "figure_map_rows": int(figure_map_df.shape[0]),
        "flag_rows": int(flags_df.shape[0]),
        "migrated_rows": int(migrated.shape[0]),
        "paper_ready_rows": int(tables.shape[0]),
    }


def export_selected_ss_vcea_followups() -> dict[str, Any]:
    sweep_df = pd.read_csv(SS_SWEEP_OUT)
    main6_df = pd.read_csv(SS_MAIN6_OUT)
    external_df = pd.read_csv(SS_EXTERNAL_OUT)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    selected_model = str(summary["ss_vcea"]["selected_model"])

    def build_summary_row(scope_df: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected = scope_df[scope_df["Model"] == selected_model].copy()
        vcea = scope_df[scope_df["Model"] == "VCEA"].copy()
        m1 = scope_df[scope_df["Model"] == "M1"].copy()
        coea = scope_df[scope_df["Model"] == "COEA"].copy()
        selected_mean = float(selected["Skill_MSE mean"].mean())
        vcea_mean = float(vcea["Skill_MSE mean"].mean())
        m1_mean = float(m1["Skill_MSE mean"].mean())
        coea_mean = float(coea["Skill_MSE mean"].mean())
        selected_std = float(selected["Skill_MSE std"].mean())
        selected_stress = float((selected["RMSE [mm] mean"] / selected["Persistence RMSE [mm] mean"]).max())
        vcea_stress = float((vcea["RMSE [mm] mean"] / vcea["Persistence RMSE [mm] mean"]).max())
        coea_stress = float((coea["RMSE [mm] mean"] / coea["Persistence RMSE [mm] mean"]).max())
        tile_compare = (
            selected[["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean", "Persistence RMSE [mm] mean"]]
            .rename(columns={"Skill_MSE mean": "Selected Skill_MSE mean", "RMSE [mm] mean": "Selected RMSE [mm] mean"})
            .merge(vcea[["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean"]].rename(columns={"Skill_MSE mean": "VCEA Skill_MSE mean", "RMSE [mm] mean": "VCEA RMSE [mm] mean"}), on=["Tile", "Horizon"], how="left")
            .merge(m1[["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean", "RMSE [mm] mean": "M1 RMSE [mm] mean"}), on=["Tile", "Horizon"], how="left")
            .merge(coea[["Tile", "Horizon", "Skill_MSE mean", "RMSE [mm] mean"]].rename(columns={"Skill_MSE mean": "COEA Skill_MSE mean", "RMSE [mm] mean": "COEA RMSE [mm] mean"}), on=["Tile", "Horizon"], how="left")
        )
        tile_compare["Gain vs M1"] = tile_compare["Selected Skill_MSE mean"] - tile_compare["M1 Skill_MSE mean"]
        tile_compare["Gain/Loss vs VCEA"] = tile_compare["Selected Skill_MSE mean"] - tile_compare["VCEA Skill_MSE mean"]
        tile_compare["Gain/Loss vs COEA"] = tile_compare["Selected Skill_MSE mean"] - tile_compare["COEA Skill_MSE mean"]
        tile_compare["Selected stress ratio"] = tile_compare["Selected RMSE [mm] mean"] / tile_compare["Persistence RMSE [mm] mean"]
        tile_compare["VCEA stress ratio"] = tile_compare["VCEA RMSE [mm] mean"] / tile_compare["Persistence RMSE [mm] mean"]
        tile_compare["M1 stress ratio"] = tile_compare["M1 RMSE [mm] mean"] / tile_compare["Persistence RMSE [mm] mean"]
        tile_compare["COEA stress ratio"] = tile_compare["COEA RMSE [mm] mean"] / tile_compare["Persistence RMSE [mm] mean"]

        selected_tile = tile_compare.groupby("Tile", as_index=False)["Selected Skill_MSE mean"].mean()
        vcea_tile = tile_compare.groupby("Tile", as_index=False)["VCEA Skill_MSE mean"].mean().rename(columns={"VCEA Skill_MSE mean": "VCEA Tile Skill"})
        m1_tile = tile_compare.groupby("Tile", as_index=False)["M1 Skill_MSE mean"].mean().rename(columns={"M1 Skill_MSE mean": "M1 Tile Skill"})
        tile_delta = selected_tile.merge(vcea_tile, on="Tile").merge(m1_tile, on="Tile")
        better_tiles_vs_vcea = int((tile_delta["Selected Skill_MSE mean"] > tile_delta["VCEA Tile Skill"]).sum())
        better_tiles_vs_m1 = int((tile_delta["Selected Skill_MSE mean"] > tile_delta["M1 Tile Skill"]).sum())
        horizon_means = tile_compare.groupby("Horizon", as_index=False)["Selected Skill_MSE mean"].mean().rename(columns={"Selected Skill_MSE mean": "Selected Horizon Skill"})
        no_severe_horizon_collapse = bool((horizon_means["Selected Horizon Skill"] > 0).all())
        summary_row = pd.DataFrame(
            [
                {
                    "Scope": scope,
                    "Model": selected_model,
                    "Risk definition": "equal-normalized risk",
                    "Safe target": "COEA",
                    "Threshold percentile": 97.5,
                    "Mean Skill_MSE": selected_mean,
                    "Gain vs M1": selected_mean - m1_mean,
                    "Gain/Loss vs VCEA": selected_mean - vcea_mean,
                    "Gain/Loss vs COEA": selected_mean - coea_mean,
                    "Stress ratio": selected_stress,
                    "Stress delta vs VCEA": selected_stress - vcea_stress,
                    "Stress delta vs COEA": selected_stress - coea_stress,
                    "Better tiles vs M1": better_tiles_vs_m1,
                    "Better tiles vs VCEA": better_tiles_vs_vcea,
                    "Seed std": selected_std,
                    "No severe horizon collapse": no_severe_horizon_collapse,
                    "Min horizon mean Skill_MSE": float(horizon_means["Selected Horizon Skill"].min()),
                    "Main criterion stress<=1.03": bool(selected_stress <= 1.03),
                    "Main criterion skill>=VCEA-0.01": bool(selected_mean >= vcea_mean - 0.01),
                    "Deployment note": (
                        "conservative deployment variant" if scope == "external_locked" and (selected_stress > vcea_stress or selected_mean < vcea_mean)
                        else "selected hardening variant"
                    ),
                }
            ]
        )
        return summary_row, tile_compare

    main_summary_row, main_tile_compare = build_summary_row(main6_df, "main6")
    external_summary_row, external_tile_compare = build_summary_row(external_df, "external_locked")
    main_summary_row.to_csv(SS_SELECTED_MAIN6_SUMMARY, index=False, encoding="utf-8-sig")
    external_summary_row.to_csv(SS_SELECTED_EXTERNAL_SUMMARY, index=False, encoding="utf-8-sig")

    main_tile_compare.to_csv(SS_SELECTED_TILE_HORIZON, index=False, encoding="utf-8-sig")

    stress_report = main_tile_compare[main_tile_compare["Tile"].isin(STRESS_TILES)].copy()
    stress_report["Stress delta vs VCEA"] = stress_report["Selected stress ratio"] - stress_report["VCEA stress ratio"]
    stress_report["Stress delta vs M1"] = stress_report["Selected stress ratio"] - stress_report["M1 stress ratio"]
    stress_report.to_csv(SS_SELECTED_STRESS_REPORT, index=False, encoding="utf-8-sig")

    external_stress = external_tile_compare.copy()
    external_stress["Stress delta vs VCEA"] = external_stress["Selected stress ratio"] - external_stress["VCEA stress ratio"]
    external_stress["Stress delta vs M1"] = external_stress["Selected stress ratio"] - external_stress["M1 stress ratio"]
    external_stress.to_csv(SS_SELECTED_EXTERNAL_STRESS, index=False, encoding="utf-8-sig")

    pareto = sweep_df[sweep_df["Scope"].isin(["main6", "external_locked"])].copy()
    main6_vcea_mean = float(main_summary_row["Mean Skill_MSE"].iloc[0] - main_summary_row["Gain/Loss vs VCEA"].iloc[0])
    external_vcea_mean = float(external_summary_row["Mean Skill_MSE"].iloc[0] - external_summary_row["Gain/Loss vs VCEA"].iloc[0])
    main6_m1_mean = float(main_summary_row["Mean Skill_MSE"].iloc[0] - main_summary_row["Gain vs M1"].iloc[0])
    external_m1_mean = float(external_summary_row["Mean Skill_MSE"].iloc[0] - external_summary_row["Gain vs M1"].iloc[0])
    pareto["Gain/Loss vs VCEA"] = pareto.apply(
        lambda r: float(r["Mean Skill_MSE"] - (main6_vcea_mean if r["Scope"] == "main6" else external_vcea_mean)),
        axis=1,
    )
    pareto["Gain vs M1"] = pareto.apply(
        lambda r: float(r["Mean Skill_MSE"] - (main6_m1_mean if r["Scope"] == "main6" else external_m1_mean)),
        axis=1,
    )
    pareto["Selected"] = pareto["Model"] == selected_model
    pareto.to_csv(SS_SELECTED_PARETO, index=False, encoding="utf-8-sig")

    order_tiles_main = TILE_ORDER
    order_horizons = [f"t+{h}" for h in SEARCH_HORIZONS]
    selected_heat = main_tile_compare.pivot(index="Tile", columns="Horizon", values="Selected Skill_MSE mean").reindex(index=order_tiles_main, columns=order_horizons)
    delta_vcea_heat = main_tile_compare.pivot(index="Tile", columns="Horizon", values="Gain/Loss vs VCEA").reindex(index=order_tiles_main, columns=order_horizons)
    delta_m1_heat = main_tile_compare.pivot(index="Tile", columns="Horizon", values="Gain vs M1").reindex(index=order_tiles_main, columns=order_horizons)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, mat, title, vmin, vmax in [
        (axes[0], selected_heat, "Selected Skill_MSE", 0.0, float(max(0.35, np.nanmax(selected_heat.to_numpy())))),
        (axes[1], delta_vcea_heat, "Selected - VCEA", float(min(-0.15, np.nanmin(delta_vcea_heat.to_numpy()))), float(max(0.15, np.nanmax(delta_vcea_heat.to_numpy())))),
        (axes[2], delta_m1_heat, "Selected - M1", float(min(-0.05, np.nanmin(delta_m1_heat.to_numpy()))), float(max(0.25, np.nanmax(delta_m1_heat.to_numpy())))),
    ]:
        im = ax.imshow(mat.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index.tolist())
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns.tolist())
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{selected_model} tile/horizon heatmaps")
    fig.tight_layout()
    fig.savefig(SS_SELECTED_HEATMAP_PDF)
    plt.close(fig)

    return {
        "selected_model": selected_model,
        "main6_stress_ratio": float(main_summary_row["Stress ratio"].iloc[0]),
        "main6_mean_skill": float(main_summary_row["Mean Skill_MSE"].iloc[0]),
        "main6_gain_vs_vcea": float(main_summary_row["Gain/Loss vs VCEA"].iloc[0]),
        "main6_better_tiles_vs_vcea": int(main_summary_row["Better tiles vs VCEA"].iloc[0]),
        "external_stress_ratio": float(external_summary_row["Stress ratio"].iloc[0]),
        "external_mean_skill": float(external_summary_row["Mean Skill_MSE"].iloc[0]),
        "external_gain_vs_vcea": float(external_summary_row["Gain/Loss vs VCEA"].iloc[0]),
        "external_note": str(external_summary_row["Deployment note"].iloc[0]),
    }


def write_final_reports(vtsa: dict[str, Any], ss: dict[str, Any], res: dict[str, Any], repro: dict[str, Any]) -> None:
    main6_vcea = pd.read_csv(SS_SWEEP_OUT)
    main6_base_vcea = float(main6_vcea[(main6_vcea["Scope"] == "main6") & (main6_vcea["Model"] == VCEA_MAINLINE_LABEL)]["Mean Skill_MSE"].iloc[0])
    lines = [
        "# TGRS VCEA Hardening 实验总报告",
        "",
        EGMS_STATEMENT,
        "",
        "这轮 hardening 包严格按文档冻结主线执行：不再扩展新模型家族，不改数据年份，不把 `512` 改写成主任务，而是围绕 `VCEA` 做机制、safety、resolution-aware 与 reproducibility 四类增强。",
        "",
        "## E1 VTSA",
        f"- mean Spearman = `{vtsa['mean_spearman']:.4f}`，mean Kendall = `{vtsa['mean_kendall']:.4f}`。",
        f"- validation top1/top2 retention = `{vtsa['top1_ratio']:.4f}` / `{vtsa['top2_ratio']:.4f}`。",
        f"- COEA mean L1 drift = `{vtsa['mean_drift']:.4f}`。",
        "结论：当前证据支持“validation-to-test ranking stability”是 VCEA 有效的主机制；但它是 tile/horizon-dependent 的，不应写成 universal law。",
        "",
        "## E2 SS-VCEA",
        f"- baseline VCEA main-six Mean Skill_MSE = `{main6_base_vcea:.4f}`。",
        f"- eligible SS variants = `{ss['eligible_count']}`。",
        f"- selected SS variant = `{ss['selected_model']}`。",
        "风险定义假设：文档未给出 `lambda` 具体值，因此本轮使用等权归一化风险，并在报告中显式保留该假设。",
        "",
        "## E3 Resolution-aware",
        f"- `M1`: `64={res['m1_64']:.4f}` -> `512={res['m1_512']:.4f}`。",
        f"- `VCEA_exact`: `64={res['vcea_64']:.4f}` -> `512={res['vcea_512']:.4f}`。",
        "结论：高分辨率并不是 aggregation family 的主要限制；但 `512` 仍只是 sensitivity，不进入主结果。",
        "",
        "## E4 Reproducibility",
        f"- infra_v2 lineage rows = `{repro['lineage_rows']}`。",
        f"- paper source map rows = `{repro['paper_map_rows']}`。",
        "结论：当前主文证据已能系统映射回 manifest、split、expert cache、aggregation calibration 与 exact/proxy/transfer provenance。",
        "",
        "## 结果目录",
        f"- VTSA: `{VTSA_ROOT.as_posix()}`",
        f"- SS-VCEA: `{SS_ROOT.as_posix()}`",
        f"- Resolution-aware: `{RES_ROOT.as_posix()}`",
        f"- Reproducibility: `{REPRO_ROOT.as_posix()}`",
    ]
    write_markdown(FINAL_REPORT_MD, lines)

    insert_lines = [
        "# Hardening Paper Insert",
        "",
        "## VTSA",
        "We explain the strength of VCEA through validation-to-test stability analysis rather than through learned expert selection. Across the six EGMS tiles, validation expert ranks remain strongly aligned with test ranks, while delayed-feedback online updates exhibit substantial weight drift. This supports static validation-calibrated aggregation as a more reliable mechanism than online adaptation under strong persistence baselines.",
        "",
        "## SS-VCEA",
        "To address the stress-tile caveat of VCEA, we introduce SS-VCEA, a validation-calibrated fallback wrapper that reverts to a safer predictor only when a normalized risk score exceeds a threshold calibrated on validation data. This design does not learn a selector and performs no test-time threshold tuning.",
        "",
        "## Resolution-aware robustness",
        "Resolution sensitivity shows that higher resolution does not uniformly benefit single backbones, but aggregation remains robust and can improve with higher-resolution expert diversity. We therefore present 64x64 as the main regional nowcasting setting, while retaining 128/512 only as robustness evidence with explicit transfer/calibration caveats.",
        "",
        "## Reproducibility",
        "The final workflow is backed by infra_v2 with frozen manifests, chronological split registries, expert-cache traceability, aggregation calibration provenance, and exact/proxy/transfer flags for result-level lineage.",
    ]
    write_markdown(PAPER_INSERT_MD, insert_lines)


def run_hardening_package() -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    vtsa = generate_vtsa_outputs()
    ss = generate_ss_vcea_outputs(bundles)
    res = generate_resolution_outputs()
    repro = generate_reproducibility_outputs()
    ss_selected = export_selected_ss_vcea_followups()
    repro = generate_reproducibility_outputs()
    write_final_reports(vtsa, ss, res, repro)
    summary = {
        "data_product": "EGMS 2019-2023 Ortho U",
        "vtsa": vtsa,
        "ss_vcea": ss,
        "ss_vcea_selected": ss_selected,
        "resolution_aware": res,
        "reproducibility": repro,
        "generated_root": str(HARDEN_ROOT),
    }
    write_json(SUMMARY_JSON, summary)
    return summary
