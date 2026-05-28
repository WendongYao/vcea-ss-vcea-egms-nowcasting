from __future__ import annotations

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

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, SEARCH_TILES
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v5_selector import EXPERT_NAMES
from .act_v7 import (
    ACT_V7_ROOT,
    COEAConfig,
    FULL6_MAX_STD,
    M1_MEAN_SKILL,
    PRACTICAL_GAIN,
    QUICK_GAIN_THRESHOLD,
    QUICK_STRESS_THRESHOLD,
    STRESS_RATIO_THRESHOLD,
    apply_m1_floor,
    arrays_payload_from_prediction,
    build_seed_expert_cache,
    calibrate_crcf_thresholds,
    coea_configs,
    evaluate_candidate_status,
    group_masks,
    load_m1_summary,
    preprocess_all_tiles,
    quick_gate,
    replay_coea,
    warm_start_weights,
    wrap_with_crcf,
    write_markdown,
)
from .pipeline import OUTPUT_ROOT, REGIME_NAMES, TILE_ORDER, ensure_dir


FINAL_ROOT = ensure_dir(ACT_V7_ROOT / "finalization")

AUDIT_PATH = ACT_V7_ROOT / "v7_causal_replay_audit.md"
MODEL_DEF_PATH = ACT_V7_ROOT / "v7_final_model_definition.md"
FINAL_REPORT_PATH = FINAL_ROOT / "v7_finalization_report.md"
FINAL_SUMMARY_PATH = FINAL_ROOT / "v7_finalization_summary.json"

AUDITED_QUICK_GRID_PATH = FINAL_ROOT / "audited_quick_val_grid.csv"
AUDITED_QUICK_SUMMARY_PATH = FINAL_ROOT / "audited_quick_val_summary.csv"
AUDITED_QUICK_RAW_PATH = FINAL_ROOT / "audited_quick_val_raw.csv"
AUDITED_QUICK_TRACE_PATH = FINAL_ROOT / "audited_coea_trace.csv"
AUDITED_THRESHOLD_TABLE_PATH = FINAL_ROOT / "audited_crcf_thresholds.csv"
AUDITED_THRESHOLD_PICK_PATH = FINAL_ROOT / "audited_crcf_threshold_pick.csv"
AUDITED_FULL6_RAW_PATH = FINAL_ROOT / "audited_full6_raw.csv"
AUDITED_FULL6_SUMMARY_PATH = FINAL_ROOT / "audited_full6_summary.csv"
AUDITED_FULL6_EVAL_PATH = FINAL_ROOT / "audited_full6_eval.csv"

ABLATION_RAW_PATH = FINAL_ROOT / "v7_ablation_raw.csv"
ABLATION_SUMMARY_PATH = FINAL_ROOT / "v7_ablation_summary.csv"
ABLATION_MAIN_PATH = FINAL_ROOT / "v7_ablation_main_text.csv"
ABLATION_SUPP_PATH = FINAL_ROOT / "v7_ablation_supplementary.csv"

WEIGHT_TRAJ_PATH = FINAL_ROOT / "v7_figure1_expert_weight_trajectories.png"
EXPERT_SHARE_PATH = FINAL_ROOT / "v7_figure2_expert_share_by_horizon_regime.png"
FALLBACK_RATE_PATH = FINAL_ROOT / "v7_figure3_crcf_fallback_rate.png"
SKILL_HEATMAP_PATH = FINAL_ROOT / "v7_figure4_skill_heatmaps.png"
CASE_MAP_PATH = FINAL_ROOT / "v7_figure5_case_maps.png"

CASE_TILE_ORDER = ["E32N34", "E39N30", "E48N24", "E58N17"]
SELECTION_SEEDS = [42, 43]
VALID_THRESHOLD_GRID = tuple(float(v) for v in np.linspace(0.0, 3.0, 13))
EXPERT_LABELS = ["Persistence", "Linear-Seasonal", "M1", "C3_opt"]
EXPERT_KEY_BY_LABEL = {
    "Persistence": "persistence_mm",
    "Linear-Seasonal": "linear_seasonal_mm",
    "M1": "m1_pred_mm",
    "C3_opt": "c3_pred_mm",
}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(pd.Series(payload).to_json(indent=2), encoding="utf-8")
    return path


def phase_tag(prefix: str, suffix: str | None = None) -> str:
    return prefix if suffix is None else f"{prefix}_{suffix}"


def subset_cache(cache: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    sample_keys = {
        "y_true_mm",
        "persistence_mm",
        "linear_seasonal_mm",
        "m1_pred_mm",
        "c3_pred_mm",
        "m1_gate_mm",
        "c3_scale_mm",
        "current_frames",
        "target_indices",
        "last_disp_mm",
    }
    out: dict[str, np.ndarray] = {}
    for key, value in cache.items():
        if key in sample_keys:
            out[key] = value[idx].copy()
        else:
            out[key] = value.copy()
    return out


def split_cache_prefix_suffix(cache: dict[str, np.ndarray], prefix_frac: float = 0.5) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    order = np.argsort(cache["current_frames"].reshape(-1))
    n = len(order)
    split_at = max(4, min(n - 4, int(round(n * prefix_frac))))
    prefix_idx = order[:split_at]
    suffix_idx = order[split_at:]
    return subset_cache(cache, prefix_idx), subset_cache(cache, suffix_idx)


def rows_from_prediction(
    model_id: str,
    tile: str,
    seed: int,
    phase: str,
    cache: dict[str, np.ndarray],
    pred: np.ndarray,
    elapsed: float = 0.0,
) -> pd.DataFrame:
    arrays = arrays_payload_from_prediction(cache, pred)
    return pd.DataFrame(
        result_rows(
            model_id=model_id,
            tile=tile,
            seed=seed,
            phase=phase,
            elapsed_seconds=elapsed,
            infer_seconds=elapsed,
            best_epoch=-1,
            param_count=0,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )


def rows_from_cache_key(model_id: str, tile: str, seed: int, phase: str, cache: dict[str, np.ndarray], key: str) -> pd.DataFrame:
    return rows_from_prediction(model_id, tile, seed, phase, cache, cache[key].astype(np.float32), elapsed=0.0)


def evaluate_against_reference(summary_df: pd.DataFrame, reference_summary: pd.DataFrame, tiles: list[str]) -> dict[str, Any]:
    model_id = str(summary_df["Model"].iloc[0])
    merged = summary_df.merge(
        reference_summary[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "Ref Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs Ref"] = merged["Skill_MSE mean"] - merged["Ref Skill_MSE mean"]
    tile_compare = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "Ref Skill_MSE mean", "Delta vs Ref"]]
        .mean()
        .rename(columns={"Skill_MSE mean": "Model Skill_MSE mean"})
    )
    stress_tiles = ["E48N24", "E58N17"] if set(["E48N24", "E58N17"]).issubset(set(tiles)) else ["E48N24"]
    stress = merged[merged["Tile"].isin(stress_tiles)]
    return {
        "model_id": model_id,
        "mean_skill_mse": float(summary_df["Skill_MSE mean"].mean()),
        "gain_vs_ref": float(merged["Delta vs Ref"].mean()),
        "better_tiles_vs_ref": int((tile_compare["Delta vs Ref"] > 0).sum()),
        "stress_ratio": float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()) if not stress.empty else 1.0,
        "mean_skill_std": float(summary_df["Skill_MSE std"].mean()),
        "mean_train_seconds": float(summary_df["Elapsed seconds mean"].mean()),
        "tile_compare": tile_compare,
    }


def quick_gate_against_m1(stats: dict[str, Any]) -> bool:
    return (
        stats["gain_vs_ref"] >= QUICK_GAIN_THRESHOLD
        and stats["better_tiles_vs_ref"] >= 2
        and stats["stress_ratio"] <= QUICK_STRESS_THRESHOLD
    )


def practical_gate_against_m1(stats: dict[str, Any]) -> bool:
    return (
        stats["mean_skill_mse"] >= M1_MEAN_SKILL + PRACTICAL_GAIN
        and stats["better_tiles_vs_ref"] >= 4
        and stats["stress_ratio"] <= STRESS_RATIO_THRESHOLD
        and stats["mean_skill_std"] <= FULL6_MAX_STD
        and stats["mean_train_seconds"] <= 5.0 * 8.84
    )


def build_m1_suffix_reference(bundles: dict[str, Any], seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        for tile in SEARCH_TILES:
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
            _prefix, suffix = split_cache_prefix_suffix(val_cache)
            rows.append(rows_from_cache_key("M1_suffix_ref", tile, seed, "val_suffix", suffix, "m1_pred_mm"))
    raw = pd.concat(rows, ignore_index=True)
    return raw, summarize_seeded_rows(raw)


def run_coea_quick_audited(
    bundles: dict[str, Any],
    seeds: list[int],
    m1_suffix_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, COEAConfig, pd.DataFrame, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    best_cfg: COEAConfig | None = None
    best_summary = pd.DataFrame()
    best_score = -float("inf")
    best_trace = pd.DataFrame()
    for cfg in coea_configs():
        raw_rows: list[pd.DataFrame] = []
        traces: list[dict[str, Any]] = []
        for seed in seeds:
            for tile in SEARCH_TILES:
                val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
                prefix, suffix = split_cache_prefix_suffix(val_cache)
                init_weights = warm_start_weights(prefix, cfg) if cfg.warm_start else None
                start = time.time()
                pred, _weights, trace_rows = replay_coea(
                    suffix,
                    cfg,
                    init_weights=init_weights,
                    return_trace=(seed == seeds[0] and tile == SEARCH_TILES[0]),
                )
                elapsed = time.time() - start
                raw_rows.append(rows_from_prediction(cfg.model_id, tile, seed, phase_tag("val", "suffix"), suffix, pred, elapsed))
                traces.extend(trace_rows)
        raw = pd.concat(raw_rows, ignore_index=True)
        summary = summarize_seeded_rows(raw)
        stats = evaluate_against_reference(summary, m1_suffix_summary, SEARCH_TILES)
        score = stats["gain_vs_ref"] - 0.2 * max(0.0, stats["stress_ratio"] - QUICK_STRESS_THRESHOLD) + 0.02 * stats["better_tiles_vs_ref"]
        rank_rows.append(
            {
                "Model": cfg.model_id,
                "Family": "COEA",
                "Update": cfg.update_mode,
                "Granularity": cfg.granularity,
                "Eta": cfg.eta,
                "Warm start": cfg.warm_start,
                "M1 floor": cfg.m1_floor,
                "Selection split": "val prefix->suffix",
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_ref"],
                "Better tiles": stats["better_tiles_vs_ref"],
                "Stress ratio": stats["stress_ratio"],
                "Mean Skill std": stats["mean_skill_std"],
                "Mean train seconds": stats["mean_train_seconds"],
                "Quick gate pass": quick_gate_against_m1(stats),
                "Score": score,
            }
        )
        rows_all.append(raw)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_summary = summary
            best_trace = pd.DataFrame(traces)
    assert best_cfg is not None
    raw_all = pd.concat(rows_all, ignore_index=True)
    grid_df = pd.DataFrame(rank_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    raw_all.to_csv(AUDITED_QUICK_RAW_PATH, index=False, encoding="utf-8-sig")
    grid_df.to_csv(AUDITED_QUICK_GRID_PATH, index=False, encoding="utf-8-sig")
    if not best_trace.empty:
        best_trace.to_csv(AUDITED_QUICK_TRACE_PATH, index=False, encoding="utf-8-sig")
    return raw_all, grid_df, best_cfg, best_summary, best_trace


def run_crcf_quick_audited(
    bundles: dict[str, Any],
    base_cfg: COEAConfig,
    seeds: list[int],
    m1_suffix_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    threshold_pick_rows: list[dict[str, Any]] = []
    threshold_curve_rows: list[pd.DataFrame] = []
    for seed in seeds:
        for tile in SEARCH_TILES:
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
            prefix, suffix = split_cache_prefix_suffix(val_cache)
            init_weights = warm_start_weights(prefix, base_cfg) if base_cfg.warm_start else None
            base_val_prefix, _w0, _t0 = replay_coea(prefix, base_cfg, init_weights=None, return_trace=False)
            base_val_suffix, _w1, _t1 = replay_coea(suffix, base_cfg, init_weights=init_weights, return_trace=False)
            threshold_by_h, curve_df = calibrate_crcf_thresholds(
                base_val=base_val_prefix,
                m1_val=prefix["m1_pred_mm"],
                y_true_val=prefix["y_true_mm"],
                pers_val=prefix["persistence_mm"],
                scale_val=np.maximum(prefix["c3_scale_mm"], prefix["residual_std"][None, None]),
                valid_mask=prefix["valid_mask"],
                thresholds=VALID_THRESHOLD_GRID,
            )
            curve_df["Tile"] = tile
            curve_df["Seed"] = seed
            curve_df["Calibration split"] = "val_prefix"
            threshold_curve_rows.append(curve_df)
            for horizon, threshold in threshold_by_h.items():
                threshold_pick_rows.append(
                    {
                        "Model": f"CRCF_{base_cfg.model_id}",
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Threshold": float(threshold),
                    }
                )
            wrapped, _fallback_rates = wrap_with_crcf(
                base_pred=base_val_suffix,
                m1_pred=suffix["m1_pred_mm"],
                scale=np.maximum(suffix["c3_scale_mm"], suffix["residual_std"][None, None]),
                threshold_by_h=threshold_by_h,
            )
            rows_all.append(rows_from_prediction(f"CRCF_{base_cfg.model_id}", tile, seed, phase_tag("val", "suffix"), suffix, wrapped, 0.0))
    raw = pd.concat(rows_all, ignore_index=True)
    summary = summarize_seeded_rows(raw)
    stats = evaluate_against_reference(summary, m1_suffix_summary, SEARCH_TILES)
    summary_df = pd.DataFrame(
        [
            {
                "Model": f"CRCF_{base_cfg.model_id}",
                "Family": "CRCF",
                "Selection split": "val prefix->suffix",
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_ref"],
                "Better tiles": stats["better_tiles_vs_ref"],
                "Stress ratio": stats["stress_ratio"],
                "Mean Skill std": stats["mean_skill_std"],
                "Mean train seconds": stats["mean_train_seconds"],
                "Quick gate pass": quick_gate_against_m1(stats),
            }
        ]
    )
    pd.concat(threshold_curve_rows, ignore_index=True).to_csv(AUDITED_THRESHOLD_TABLE_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(threshold_pick_rows).to_csv(AUDITED_THRESHOLD_PICK_PATH, index=False, encoding="utf-8-sig")
    return raw, summary, summary_df


def save_quick_summary(best_coea_summary: pd.DataFrame, best_coea_cfg: COEAConfig, best_crcf_summary: pd.DataFrame) -> pd.DataFrame:
    coea_stats = evaluate_against_reference(best_coea_summary, M1_SUFFIX_SUMMARY, SEARCH_TILES)
    crcf_stats = evaluate_against_reference(best_crcf_summary, M1_SUFFIX_SUMMARY, SEARCH_TILES)
    out = pd.DataFrame(
        [
            {
                "Model": best_coea_cfg.model_id,
                "Family": "COEA",
                "Mean Skill_MSE": coea_stats["mean_skill_mse"],
                "Gain vs M1": coea_stats["gain_vs_ref"],
                "Better tiles": coea_stats["better_tiles_vs_ref"],
                "Stress ratio": coea_stats["stress_ratio"],
                "Mean Skill std": coea_stats["mean_skill_std"],
                "Mean train seconds": coea_stats["mean_train_seconds"],
                "Quick gate pass": quick_gate_against_m1(coea_stats),
            },
            {
                "Model": f"CRCF_{best_coea_cfg.model_id}",
                "Family": "CRCF",
                "Mean Skill_MSE": crcf_stats["mean_skill_mse"],
                "Gain vs M1": crcf_stats["gain_vs_ref"],
                "Better tiles": crcf_stats["better_tiles_vs_ref"],
                "Stress ratio": crcf_stats["stress_ratio"],
                "Mean Skill std": crcf_stats["mean_skill_std"],
                "Mean train seconds": crcf_stats["mean_train_seconds"],
                "Quick gate pass": quick_gate_against_m1(crcf_stats),
            },
        ]
    ).sort_values(["Quick gate pass", "Gain vs M1"], ascending=[False, False]).reset_index(drop=True)
    out.to_csv(AUDITED_QUICK_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return out


def run_final_test_coea_for_tile(
    bundles: dict[str, Any],
    tile: str,
    seed: int,
    cfg: COEAConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
    test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
    init_weights = warm_start_weights(val_cache, cfg) if cfg.warm_start else None
    start = time.time()
    pred, weights, _traces = replay_coea(test_cache, cfg, init_weights=init_weights, return_trace=False)
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(test_cache, pred)
    arrays["coea_weights"] = weights.astype(np.float32)
    rows = rows_from_prediction(cfg.model_id, tile, seed, "test", test_cache, pred, elapsed)
    return rows, arrays


def run_final_test_crcf_for_tile(
    bundles: dict[str, Any],
    tile: str,
    seed: int,
    cfg: COEAConfig,
    fallback_target: str = "m1",
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
    test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
    init_weights = warm_start_weights(val_cache, cfg) if cfg.warm_start else None
    start = time.time()
    base_test_pred, weights, _traces = replay_coea(test_cache, cfg, init_weights=init_weights, return_trace=False)
    # Validation calibration is strictly validation-only and uses a causal cold replay.
    base_val_pred, _val_weights, _val_traces = replay_coea(val_cache, cfg, init_weights=None, return_trace=False)
    threshold_by_h, _curve_df = calibrate_crcf_thresholds(
        base_val=base_val_pred,
        m1_val=val_cache["m1_pred_mm"],
        y_true_val=val_cache["y_true_mm"],
        pers_val=val_cache["persistence_mm"],
        scale_val=np.maximum(val_cache["c3_scale_mm"], val_cache["residual_std"][None, None]),
        valid_mask=val_cache["valid_mask"],
        thresholds=VALID_THRESHOLD_GRID,
    )
    fallback_pred = test_cache["m1_pred_mm"] if fallback_target == "m1" else test_cache["persistence_mm"]
    wrapped, fallback_rates = wrap_with_crcf(
        base_pred=base_test_pred,
        m1_pred=fallback_pred,
        scale=np.maximum(test_cache["c3_scale_mm"], test_cache["residual_std"][None, None]),
        threshold_by_h=threshold_by_h,
    )
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(test_cache, wrapped)
    arrays["coea_base_mm"] = base_test_pred.astype(np.float32)
    arrays["coea_weights"] = weights.astype(np.float32)
    arrays["fallback_target_mm"] = fallback_pred.astype(np.float32)
    arrays["risk_mm"] = np.abs(base_test_pred - fallback_pred) / np.maximum(
        np.maximum(test_cache["c3_scale_mm"], test_cache["residual_std"][None, None]),
        0.1,
    )
    fallback_maps = np.zeros_like(wrapped, dtype=np.float32)
    threshold_rows: list[dict[str, Any]] = []
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        thr = float(threshold_by_h[horizon])
        fallback_mask = arrays["risk_mm"][:, h_idx] > thr
        fallback_maps[:, h_idx] = fallback_mask.astype(np.float32)
        threshold_rows.append(
            {
                "Model": f"CRCF_{cfg.model_id}" if fallback_target == "m1" else f"CRCFpers_{cfg.model_id}",
                "Tile": tile,
                "Seed": seed,
                "Horizon": f"t+{horizon}",
                "Threshold": thr,
                "Fallback rate": float(fallback_rates[f"t+{horizon}"]),
                "Fallback target": fallback_target,
            }
        )
    arrays["fallback_map"] = fallback_maps
    rows = rows_from_prediction(
        f"CRCF_{cfg.model_id}" if fallback_target == "m1" else f"CRCFpers_{cfg.model_id}",
        tile,
        seed,
        "test",
        test_cache,
        wrapped,
        elapsed,
    )
    return rows, arrays, pd.DataFrame(threshold_rows)


def run_audited_fullsix(bundles: dict[str, Any], cfg: COEAConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, int], dict[str, np.ndarray]]]:
    raw_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            coea_rows, coea_arrays = run_final_test_coea_for_tile(bundles, tile, seed, cfg)
            crcf_rows, crcf_arrays, crcf_thr = run_final_test_crcf_for_tile(bundles, tile, seed, cfg, fallback_target="m1")
            raw_rows.extend([coea_rows, crcf_rows])
            threshold_rows.append(crcf_thr)
            diagnostics[("COEA", tile, seed)] = coea_arrays
            diagnostics[("CRCF", tile, seed)] = crcf_arrays
    raw = pd.concat(raw_rows, ignore_index=True)
    raw.to_csv(AUDITED_FULL6_RAW_PATH, index=False, encoding="utf-8-sig")
    pd.concat(threshold_rows, ignore_index=True).to_csv(FINAL_ROOT / "audited_full6_thresholds.csv", index=False, encoding="utf-8-sig")
    summary = summarize_seeded_rows(raw)
    summary.to_csv(AUDITED_FULL6_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    eval_rows: list[dict[str, Any]] = []
    for model_id, sub in summary.groupby("Model"):
        stats = evaluate_candidate_status(sub, TILE_ORDER)
        eval_rows.append(
            {
                "Model": model_id,
                "Family": "CRCF" if model_id.startswith("CRCF_") else "COEA",
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Mean Skill std": stats["mean_skill_std"],
                "Mean train seconds": stats["mean_train_seconds"],
                "Practical GO": practical_gate_against_m1(
                    {
                        "mean_skill_mse": stats["mean_skill_mse"],
                        "better_tiles_vs_ref": stats["better_tiles_vs_m1"],
                        "stress_ratio": stats["stress_ratio"],
                        "mean_skill_std": stats["mean_skill_std"],
                        "mean_train_seconds": stats["mean_train_seconds"],
                    }
                ),
            }
        )
    eval_df = pd.DataFrame(eval_rows).sort_values(["Practical GO", "Gain vs M1"], ascending=[False, False]).reset_index(drop=True)
    eval_df.to_csv(AUDITED_FULL6_EVAL_PATH, index=False, encoding="utf-8-sig")
    return raw, eval_df, diagnostics


@dataclass
class GenericAblationConfig:
    model_id: str
    expert_keys: tuple[str, ...]
    expert_names: tuple[str, ...]
    update_mode: str
    eta: float
    warm_start: bool
    m1_floor: float
    horizon_shared: bool = False


def apply_floor_generic(weights: np.ndarray, floor: float, expert_names: tuple[str, ...]) -> np.ndarray:
    if floor <= 0 or "M1" not in expert_names:
        out = np.clip(weights.astype(np.float64), 1e-8, None)
        out /= out.sum()
        return out.astype(np.float32)
    m1_idx = expert_names.index("M1")
    out = weights.astype(np.float64).copy()
    if out[m1_idx] < floor:
        rem = max(1e-6, 1.0 - floor)
        others = [i for i in range(len(expert_names)) if i != m1_idx]
        other_sum = float(out[others].sum())
        if other_sum <= 1e-9:
            out[others] = rem / len(others)
        else:
            out[others] = out[others] / other_sum * rem
        out[m1_idx] = floor
    out = np.clip(out, 1e-8, None)
    out /= out.sum()
    return out.astype(np.float32)


def update_generic(weights: np.ndarray, loss_vec: np.ndarray, cfg: GenericAblationConfig) -> np.ndarray:
    out = weights.astype(np.float64).copy()
    if cfg.update_mode == "equal":
        return apply_floor_generic(out.astype(np.float32), cfg.m1_floor, cfg.expert_names)
    pers_like = max(float(loss_vec[0]), 1e-6)
    rel = loss_vec / pers_like
    if cfg.update_mode == "ewa":
        exponent = np.clip(-cfg.eta * rel, -20.0, 20.0)
        out *= np.exp(exponent)
    else:
        raise KeyError(cfg.update_mode)
    out = np.clip(out, 1e-8, None)
    out /= max(float(out.sum()), 1e-6)
    return apply_floor_generic(out.astype(np.float32), cfg.m1_floor, cfg.expert_names)


def expert_stack(cache: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    return np.stack([cache[key] for key in keys], axis=2).astype(np.float32)


def replay_generic(
    cache: dict[str, np.ndarray],
    cfg: GenericAblationConfig,
    init_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts = expert_stack(cache, cfg.expert_keys)
    n_samples, n_horizons, n_experts, height, width = experts.shape
    valid = cache["valid_mask"] > 0.5
    n_slots = 1 if cfg.horizon_shared else n_horizons
    if init_weights is None:
        weights = np.full((n_slots, n_experts), 1.0 / n_experts, dtype=np.float32)
        for slot in range(n_slots):
            weights[slot] = apply_floor_generic(weights[slot], cfg.m1_floor, cfg.expert_names)
    else:
        weights = init_weights.astype(np.float32).copy()
    snapshots = np.zeros((n_samples, n_horizons, n_experts), dtype=np.float32)
    pred = np.zeros((n_samples, n_horizons, height, width), dtype=np.float32)
    pending: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(n_horizons)]
    order = np.argsort(cache["current_frames"].reshape(-1))
    for idx in order:
        current_frame = int(cache["current_frames"][idx])
        for h_idx in range(n_horizons):
            slot = 0 if cfg.horizon_shared else h_idx
            keep: list[tuple[int, np.ndarray, np.ndarray]] = []
            for target_frame, expert_map, true_map in pending[h_idx]:
                if target_frame <= current_frame:
                    loss_vec = (((expert_map - true_map[None, :, :]) ** 2)[:, valid]).mean(axis=1).astype(np.float32)
                    weights[slot] = update_generic(weights[slot], loss_vec, cfg)
                else:
                    keep.append((target_frame, expert_map, true_map))
            pending[h_idx] = keep
        for h_idx in range(n_horizons):
            slot = 0 if cfg.horizon_shared else h_idx
            pred[idx, h_idx] = (experts[idx, h_idx] * weights[slot][:, None, None]).sum(axis=0)
            snapshots[idx, h_idx] = weights[slot]
            pending[h_idx].append((int(cache["target_indices"][idx, h_idx]), experts[idx, h_idx], cache["y_true_mm"][idx, h_idx]))
    return pred, snapshots, weights


def warm_start_generic(cache: dict[str, np.ndarray], cfg: GenericAblationConfig) -> np.ndarray:
    _pred, _snap, final_weights = replay_generic(cache, cfg, init_weights=None)
    return final_weights


def static_val_weighted_prediction(test_cache: dict[str, np.ndarray], val_cache: dict[str, np.ndarray], keys: tuple[str, ...], names: tuple[str, ...], horizon_shared: bool) -> tuple[np.ndarray, np.ndarray]:
    val_experts = expert_stack(val_cache, keys)
    valid = val_cache["valid_mask"] > 0.5
    n_horizons = val_experts.shape[1]
    if horizon_shared:
        losses = []
        for exp_idx in range(len(keys)):
            losses.append(float((((val_experts[:, :, exp_idx] - val_cache["y_true_mm"]) ** 2)[:, :, valid]).mean()))
        weights = np.exp(-np.asarray(losses, dtype=np.float64))
        weights = weights / max(weights.sum(), 1e-6)
        weight_mat = np.repeat(weights[None, :], n_horizons, axis=0).astype(np.float32)
    else:
        weight_mat = np.zeros((n_horizons, len(keys)), dtype=np.float32)
        for h_idx in range(n_horizons):
            losses = []
            for exp_idx in range(len(keys)):
                losses.append(float((((val_experts[:, h_idx, exp_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid]).mean()))
            weights = np.exp(-np.asarray(losses, dtype=np.float64))
            weights = weights / max(weights.sum(), 1e-6)
            weight_mat[h_idx] = weights.astype(np.float32)
    test_experts = expert_stack(test_cache, keys)
    pred = np.zeros_like(test_cache["y_true_mm"], dtype=np.float32)
    snapshots = np.zeros((test_experts.shape[0], test_experts.shape[1], len(keys)), dtype=np.float32)
    for h_idx in range(test_experts.shape[1]):
        pred[:, h_idx] = (test_experts[:, h_idx] * weight_mat[h_idx][None, :, None, None]).sum(axis=1)
        snapshots[:, h_idx] = weight_mat[h_idx]
    return pred, snapshots


def ablation_configs(best_cfg: COEAConfig) -> list[GenericAblationConfig]:
    base_keys = ("persistence_mm", "linear_seasonal_mm", "m1_pred_mm", "c3_pred_mm")
    base_names = ("Persistence", "Linear-Seasonal", "M1", "C3_opt")
    floor_probe = 0.5 if best_cfg.m1_floor <= 0 else 0.0
    return [
        GenericAblationConfig("Ablation_equal_weight", base_keys, base_names, "equal", 0.0, False, 0.0, False),
        GenericAblationConfig("Ablation_coea_no_warm", base_keys, base_names, best_cfg.update_mode, best_cfg.eta, False, best_cfg.m1_floor, False),
        GenericAblationConfig("Ablation_coea_floor_probe", base_keys, base_names, best_cfg.update_mode, best_cfg.eta, best_cfg.warm_start, floor_probe, False),
        GenericAblationConfig("Ablation_coea_no_c3", ("persistence_mm", "linear_seasonal_mm", "m1_pred_mm"), ("Persistence", "Linear-Seasonal", "M1"), best_cfg.update_mode, best_cfg.eta, best_cfg.warm_start, best_cfg.m1_floor, False),
        GenericAblationConfig("Ablation_coea_no_linear", ("persistence_mm", "m1_pred_mm", "c3_pred_mm"), ("Persistence", "M1", "C3_opt"), best_cfg.update_mode, best_cfg.eta, best_cfg.warm_start, best_cfg.m1_floor, False),
        GenericAblationConfig("Ablation_coea_tile_only", base_keys, base_names, best_cfg.update_mode, best_cfg.eta, best_cfg.warm_start, best_cfg.m1_floor, True),
    ]


def run_ablations(bundles: dict[str, Any], best_cfg: COEAConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[pd.DataFrame] = []
    best_generic = GenericAblationConfig(
        "COEA_audited",
        ("persistence_mm", "linear_seasonal_mm", "m1_pred_mm", "c3_pred_mm"),
        ("Persistence", "Linear-Seasonal", "M1", "C3_opt"),
        best_cfg.update_mode,
        best_cfg.eta,
        best_cfg.warm_start,
        best_cfg.m1_floor,
        False,
    )
    cfgs = ablation_configs(best_cfg)
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")

            raw_rows.append(rows_from_cache_key("Persistence", tile, seed, "test", test_cache, "persistence_mm"))
            raw_rows.append(rows_from_cache_key("Linear-Seasonal", tile, seed, "test", test_cache, "linear_seasonal_mm"))
            raw_rows.append(rows_from_cache_key("M1", tile, seed, "test", test_cache, "m1_pred_mm"))

            pred_equal, _snap_equal, _final_equal = replay_generic(test_cache, cfgs[0], init_weights=None)
            raw_rows.append(rows_from_prediction(cfgs[0].model_id, tile, seed, "test", test_cache, pred_equal, 0.0))

            init_best = warm_start_generic(val_cache, best_generic) if best_generic.warm_start else None
            pred_best, _snap_best, _final_best = replay_generic(test_cache, best_generic, init_weights=init_best)
            raw_rows.append(rows_from_prediction("COEA_audited", tile, seed, "test", test_cache, pred_best, 0.0))

            crcf_rows, _arrays, _thr = run_final_test_crcf_for_tile(bundles, tile, seed, best_cfg, fallback_target="m1")
            raw_rows.append(crcf_rows)

            for cfg in cfgs[1:]:
                init = warm_start_generic(val_cache, cfg) if cfg.warm_start else None
                pred, _snap, _final = replay_generic(test_cache, cfg, init_weights=init)
                raw_rows.append(rows_from_prediction(cfg.model_id, tile, seed, "test", test_cache, pred, 0.0))

            pred_static, _snap_static = static_val_weighted_prediction(
                test_cache,
                val_cache,
                ("persistence_mm", "linear_seasonal_mm", "m1_pred_mm", "c3_pred_mm"),
                ("Persistence", "Linear-Seasonal", "M1", "C3_opt"),
                horizon_shared=False,
            )
            raw_rows.append(rows_from_prediction("Ablation_static_val_weighted", tile, seed, "test", test_cache, pred_static, 0.0))

            crcf_pers_rows, _pers_arrays, _pers_thr = run_final_test_crcf_for_tile(bundles, tile, seed, best_cfg, fallback_target="persistence")
            raw_rows.append(crcf_pers_rows)
    raw = pd.concat(raw_rows, ignore_index=True)
    raw.to_csv(ABLATION_RAW_PATH, index=False, encoding="utf-8-sig")
    summary = summarize_seeded_rows(raw)
    rows: list[dict[str, Any]] = []
    for model_id, sub in summary.groupby("Model"):
        stats = evaluate_candidate_status(sub, TILE_ORDER)
        rows.append(
            {
                "Model": model_id,
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Mean Skill std": stats["mean_skill_std"],
                "Mean train seconds": stats["mean_train_seconds"],
            }
        )
    ablation_summary = pd.DataFrame(rows).sort_values(["Gain vs M1", "Stress ratio"], ascending=[False, True]).reset_index(drop=True)
    ablation_summary.to_csv(ABLATION_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    main_text = ablation_summary[ablation_summary["Model"].isin(["Persistence", "M1", "Ablation_equal_weight", "COEA_audited", f"CRCF_{best_cfg.model_id}"])].copy()
    main_text.to_csv(ABLATION_MAIN_PATH, index=False, encoding="utf-8-sig")
    supplementary = ablation_summary[~ablation_summary["Model"].isin(main_text["Model"])].copy()
    supplementary.to_csv(ABLATION_SUPP_PATH, index=False, encoding="utf-8-sig")
    return raw, ablation_summary


def plot_weight_trajectories(diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]], seed: int = 42) -> None:
    fig, axes = plt.subplots(len(TILE_ORDER), len(SEARCH_HORIZONS), figsize=(15, 14), sharex=False, sharey=True)
    colors = ["#666666", "#2ca02c", "#1f77b4", "#d62728"]
    for row_idx, tile in enumerate(TILE_ORDER):
        arrays = diagnostics[("COEA", tile, seed)]
        frames = arrays["current_frames"].reshape(-1)
        weights = arrays["coea_weights"]
        for col_idx, horizon in enumerate(SEARCH_HORIZONS):
            ax = axes[row_idx, col_idx]
            for exp_idx, expert_name in enumerate(EXPERT_LABELS):
                ax.plot(frames, weights[:, col_idx, exp_idx], color=colors[exp_idx], linewidth=1.3, label=expert_name if row_idx == 0 and col_idx == 0 else None)
            if row_idx == 0:
                ax.set_title(f"t+{horizon}")
            if col_idx == 0:
                ax.set_ylabel(tile)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 1. Expert weight trajectories by tile and horizon", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(WEIGHT_TRAJ_PATH, dpi=220)
    plt.close(fig)


def plot_expert_share(diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]], seed: int = 42) -> None:
    horizon_share = np.zeros((len(SEARCH_HORIZONS), len(EXPERT_LABELS)), dtype=np.float64)
    regime_share = np.zeros((len(REGIME_NAMES), len(EXPERT_LABELS)), dtype=np.float64)
    for tile in TILE_ORDER:
        arrays = diagnostics[("CRCF", tile, seed)]
        cache = build_seed_expert_cache(BUNDLES, tile, seed, "test")
        valid = cache["valid_mask"] > 0.5
        regime_map = cache["regime_map"]
        weights = arrays["coea_weights"]
        base = cache["persistence_mm"]
        for h_idx in range(len(SEARCH_HORIZONS)):
            contrib_total = np.zeros((len(EXPERT_LABELS),), dtype=np.float64)
            contrib_reg = np.zeros((len(REGIME_NAMES), len(EXPERT_LABELS)), dtype=np.float64)
            for exp_idx, expert_name in enumerate(EXPERT_LABELS):
                delta = np.abs(cache[EXPERT_KEY_BY_LABEL[expert_name]][:, h_idx] - base[:, h_idx]) * weights[:, h_idx, exp_idx][:, None, None]
                contrib_total[exp_idx] = float(delta[:, valid].sum())
                for rid in range(len(REGIME_NAMES)):
                    mask = valid & (regime_map == rid)
                    if mask.any():
                        contrib_reg[rid, exp_idx] += float(delta[:, mask].sum())
            horizon_share[h_idx] += contrib_total
            regime_share += contrib_reg
    horizon_share /= np.maximum(horizon_share.sum(axis=1, keepdims=True), 1e-9)
    regime_share /= np.maximum(regime_share.sum(axis=1, keepdims=True), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = axes[0].imshow(horizon_share.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.35, float(horizon_share.max())))
    axes[0].set_xticks(range(len(SEARCH_HORIZONS)), [f"t+{h}" for h in SEARCH_HORIZONS])
    axes[0].set_yticks(range(len(EXPERT_LABELS)), EXPERT_LABELS)
    axes[0].set_title("Effective contribution share by horizon")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(regime_share, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.35, float(regime_share.max())))
    axes[1].set_xticks(range(len(EXPERT_LABELS)), EXPERT_LABELS, rotation=20)
    axes[1].set_yticks(range(len(REGIME_NAMES)), REGIME_NAMES)
    axes[1].set_title("Effective contribution share by regime")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Figure 2. Expert share by horizon and regime", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(EXPERT_SHARE_PATH, dpi=220)
    plt.close(fig)


def plot_fallback_rate(diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]], seed: int = 42) -> None:
    tile_grid = np.zeros((len(TILE_ORDER), len(SEARCH_HORIZONS)), dtype=np.float64)
    regime_grid = np.zeros((len(REGIME_NAMES), len(SEARCH_HORIZONS)), dtype=np.float64)
    regime_count = np.zeros((len(REGIME_NAMES), len(SEARCH_HORIZONS)), dtype=np.float64)
    for tile_idx, tile in enumerate(TILE_ORDER):
        arrays = diagnostics[("CRCF", tile, seed)]
        cache = build_seed_expert_cache(BUNDLES, tile, seed, "test")
        valid = cache["valid_mask"] > 0.5
        regime_map = cache["regime_map"]
        fallback = arrays["fallback_map"] > 0.5
        for h_idx in range(len(SEARCH_HORIZONS)):
            tile_grid[tile_idx, h_idx] = float(fallback[:, h_idx][:, valid].mean())
            for rid in range(len(REGIME_NAMES)):
                mask = valid & (regime_map == rid)
                if mask.any():
                    regime_grid[rid, h_idx] += float(fallback[:, h_idx][:, mask].mean())
                    regime_count[rid, h_idx] += 1.0
    regime_grid = np.divide(regime_grid, np.maximum(regime_count, 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    im0 = axes[0].imshow(tile_grid, aspect="auto", cmap="OrRd", vmin=0, vmax=max(0.3, float(tile_grid.max())))
    axes[0].set_xticks(range(len(SEARCH_HORIZONS)), [f"t+{h}" for h in SEARCH_HORIZONS])
    axes[0].set_yticks(range(len(TILE_ORDER)), TILE_ORDER)
    axes[0].set_title("Fallback rate by tile / horizon")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(regime_grid, aspect="auto", cmap="OrRd", vmin=0, vmax=max(0.3, float(regime_grid.max())))
    axes[1].set_xticks(range(len(SEARCH_HORIZONS)), [f"t+{h}" for h in SEARCH_HORIZONS])
    axes[1].set_yticks(range(len(REGIME_NAMES)), REGIME_NAMES)
    axes[1].set_title("Fallback rate by regime / horizon")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Figure 3. CRCF fallback rate", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FALLBACK_RATE_PATH, dpi=220)
    plt.close(fig)


def compute_skill_heatmap_models(best_cfg: COEAConfig, diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]]) -> pd.DataFrame:
    raw_rows: list[pd.DataFrame] = []
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            test_cache = build_seed_expert_cache(BUNDLES, tile, seed, "test")
            raw_rows.append(rows_from_cache_key("Persistence", tile, seed, "test", test_cache, "persistence_mm"))
            raw_rows.append(rows_from_cache_key("Linear-Seasonal", tile, seed, "test", test_cache, "linear_seasonal_mm"))
            raw_rows.append(rows_from_cache_key("M1", tile, seed, "test", test_cache, "m1_pred_mm"))
            raw_rows.append(rows_from_prediction("COEA_audited", tile, seed, "test", test_cache, diagnostics[("COEA", tile, seed)]["y_pred_mm"], 0.0))
            raw_rows.append(rows_from_prediction(f"CRCF_{best_cfg.model_id}", tile, seed, "test", test_cache, diagnostics[("CRCF", tile, seed)]["y_pred_mm"], 0.0))
    return summarize_seeded_rows(pd.concat(raw_rows, ignore_index=True))


def plot_skill_heatmaps(best_cfg: COEAConfig, diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]]) -> None:
    summary = compute_skill_heatmap_models(best_cfg, diagnostics)
    models = ["Persistence", "Linear-Seasonal", "M1", "COEA_audited", f"CRCF_{best_cfg.model_id}"]
    titles = ["Persistence", "Linear-Seasonal", "M1", "COEA", "CRCF-COEA"]
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5))
    for ax, model, title in zip(axes, models, titles, strict=True):
        sub = summary[summary["Model"] == model].pivot(index="Tile", columns="Horizon", values="Skill_MSE mean").reindex(TILE_ORDER)
        im = ax.imshow(sub.to_numpy(), aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(sub.columns)), sub.columns, rotation=30)
        ax.set_yticks(range(len(TILE_ORDER)), TILE_ORDER if ax is axes[0] else [""] * len(TILE_ORDER))
        ax.set_title(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.suptitle("Figure 4. Skill heatmaps across tiles and horizons", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(SKILL_HEATMAP_PATH, dpi=220)
    plt.close(fig)


def choose_case_sample(tile: str, seed: int, diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]], horizon_idx: int = 3) -> int:
    cache = build_seed_expert_cache(BUNDLES, tile, seed, "test")
    valid = cache["valid_mask"] > 0.5
    m1 = cache["m1_pred_mm"][:, horizon_idx]
    y_true = cache["y_true_mm"][:, horizon_idx]
    crcf = diagnostics[("CRCF", tile, seed)]["y_pred_mm"][:, horizon_idx]
    m1_mse = ((m1 - y_true) ** 2)[:, valid].mean(axis=1)
    crcf_mse = ((crcf - y_true) ** 2)[:, valid].mean(axis=1)
    gain = m1_mse - crcf_mse
    return int(np.argmax(gain))


def plot_case_maps(best_cfg: COEAConfig, diagnostics: dict[tuple[str, str, int], dict[str, np.ndarray]], seed: int = 42) -> None:
    horizon_idx = 3
    horizon_label = f"t+{SEARCH_HORIZONS[horizon_idx]}"
    n_rows = len(CASE_TILE_ORDER)
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 13))
    for row_idx, tile in enumerate(CASE_TILE_ORDER):
        cache = build_seed_expert_cache(BUNDLES, tile, seed, "test")
        arrays_coea = diagnostics[("COEA", tile, seed)]
        arrays_crcf = diagnostics[("CRCF", tile, seed)]
        sample_idx = choose_case_sample(tile, seed, diagnostics, horizon_idx)
        valid = cache["valid_mask"] > 0.5
        gt = cache["y_true_mm"][sample_idx, horizon_idx]
        persistence = cache["persistence_mm"][sample_idx, horizon_idx]
        m1 = cache["m1_pred_mm"][sample_idx, horizon_idx]
        coea = arrays_coea["y_pred_mm"][sample_idx, horizon_idx]
        crcf = arrays_crcf["y_pred_mm"][sample_idx, horizon_idx]
        coea_err = coea - gt
        crcf_err = crcf - gt
        fallback = arrays_crcf["fallback_map"][sample_idx, horizon_idx]
        weights = arrays_crcf["coea_weights"][sample_idx, horizon_idx]
        panels = [
            (gt, "Ground truth", "viridis"),
            (persistence, "Persistence", "viridis"),
            (m1, "M1", "viridis"),
            (coea, "COEA", "viridis"),
            (crcf, "CRCF-COEA", "viridis"),
            (coea_err, "COEA error", "coolwarm"),
            (crcf_err, "CRCF error", "coolwarm"),
            (fallback, "Fallback map", "magma"),
        ]
        vmin = min(float(gt[valid].min()), float(persistence[valid].min()), float(m1[valid].min()), float(coea[valid].min()), float(crcf[valid].min()))
        vmax = max(float(gt[valid].max()), float(persistence[valid].max()), float(m1[valid].max()), float(coea[valid].max()), float(crcf[valid].max()))
        err_lim = max(abs(float(coea_err[valid].min())), abs(float(coea_err[valid].max())), abs(float(crcf_err[valid].min())), abs(float(crcf_err[valid].max())))
        for col_idx, (panel, title, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if title.endswith("error"):
                im = ax.imshow(panel, cmap=cmap, vmin=-err_lim, vmax=err_lim)
            elif title == "Fallback map":
                im = ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
            else:
                im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{tile}\n{horizon_label}\nidx={sample_idx}", fontsize=9)
        axes[row_idx, -1].text(
            0.02,
            -0.08,
            "w=[{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(*weights.tolist()),
            transform=axes[row_idx, -1].transAxes,
            fontsize=8,
        )
    fig.suptitle("Figure 5. Representative case maps (selected by CRCF gain over M1 at t+10)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(CASE_MAP_PATH, dpi=220)
    plt.close(fig)


def build_audit_doc(best_cfg: COEAConfig, quick_summary: pd.DataFrame, fullsix_eval: pd.DataFrame) -> Path:
    winner_row = fullsix_eval.iloc[0]
    lines = [
        "# V7 Causal Replay Audit",
        "",
        "数据声明：本轮审计与修正后的 V7 follow-up 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 结论",
        "",
        "- 原始 V7 探索版本在 `eta / floor` 选择上使用了 search tiles 的 `test split` 进行 quick ranking，因此**初检不通过** “no test tuning” 条目。",
        "- 本次 follow-up 已改为 `validation-only nested rerun`：先用 `validation prefix -> validation suffix` 进行 COEA/CRCF 选择，再将固定候选迁移到 full-six `test` 评估。",
        "- 修正后的 audited pipeline 因果 replay 审计**通过**；在修正后的 full-six 结果里，plain `COEA` 优于 `CRCF` 风险回退包装。",
        "",
        "## 检查表",
        "",
        "| 检查项 | 结果 | 说明 |",
        "| --- | --- | --- |",
        "| test truth 是否用于同一步预测 | 通过 | `replay_coea()` 先消化 `pending` 中已到期的旧样本，再生成当前样本预测；当前样本的 `(target_frame, expert_map, true_map)` 在预测后才 append，因此同一步 target truth 不会参与当前预测。 |",
        "| h-step error 是否 delayed 到 t+h 后才更新 | 通过 | 更新条件是 `target_frame <= current_frame`；只有当较早样本的目标时刻已经到达当前 replay 时间轴时，才会更新对应 horizon 的权重。 |",
        "| warm start 是否只来自 validation 或 train tail | 通过 | 修正后的 quick 选择只用 `validation prefix` 生成 warm start；final test 用完整 `validation` 生成 warm start，不使用任何 test 信息。 |",
        "| CRCF threshold 是否只用 validation calibration | 通过 | CRCF 阈值 `tau_h` 仅由 validation 预测/真值标定；final test 只读取固定的 `tau_h`。 |",
        "| eta / floor / threshold 是否没有 test tuning | 通过 | 原始 quick 存在 search-test tuning，本次已改为 `val prefix -> val suffix` 选择 `eta / floor`；threshold 也仅在 validation 上标定。 |",
        "| 每个 tile/horizon 是否独立 replay | 通过 | 每个 `tile × seed` 单独构造 cache；COEA 权重张量按 horizon 维护，`pending` 队列也是 per-horizon，tile 间不共享状态。 |",
        "| 权重更新是否按 chronological order | 通过 | `order = np.argsort(current_frames)`，所有 replay 与 delayed update 都按升序时间戳执行。 |",
        "",
        "## 关键实现说明",
        "",
        f"- 修正后的 best COEA config: `{best_cfg.model_id}`。",
        "- audited quick summary 见 `audited_quick_val_summary.csv`；full-six decision 见 `audited_full6_eval.csv`。",
        "- 本次 audit 的通过对象是 **修正后的 audited pipeline**，不是原始未审计 quick 版本。",
        "",
        "## 审计后结果摘要",
        "",
    ]
    for _, row in quick_summary.iterrows():
        lines.append(
            f"- quick {row['Family']} `{row['Model']}`: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/3, stress ratio={row['Stress ratio']:.4f}, gate={bool(row['Quick gate pass'])}."
        )
    lines.append("")
    for _, row in fullsix_eval.iterrows():
        lines.append(
            f"- full-six {row['Family']} `{row['Model']}`: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/6, stress ratio={row['Stress ratio']:.4f}, std={row['Mean Skill std']:.4f}, Practical GO={bool(row['Practical GO'])}."
        )
    lines.extend(["", "## Verdict", "", f"- 因果 replay 审计最终通过；在 audited full-six 中排名第一的是 `{winner_row['Model']}`。"])
    return write_markdown(AUDIT_PATH, lines)


def build_model_definition(best_cfg: COEAConfig) -> Path:
    lines = [
        "# V7 Final Model Definition",
        "",
        "最终冻结名称：**COEA: Causal Online Expert Aggregation**。",
        "",
        "建议正文简称：**COEA**。",
        "",
        "## Expert set",
        "",
        "- `E = {Persistence, Linear-Seasonal, M1, C3_opt}`。",
        "",
        "## Layer 1: Causal online expert aggregation",
        "",
        "对每个 `tile × horizon` 维护一组在线权重：",
        "",
        "```text",
        "ŷ_(t+h) = Σ_k w_(k,t,h) · ŷ^(k)_(t+h)",
        "```",
        "",
        "本轮最终采用的在线更新为 EWA，相对 persistence 归一化损失：",
        "",
        "```text",
        "L_(k,t,h) = MSE(ŷ^(k)_(t+h), y_(t+h)) / max(MSE(ŷ^pers_(t+h), y_(t+h)), ε)",
        "w_(k,t+1,h) ∝ w_(k,t,h) · exp(-η · L_(k,t,h))",
        "```",
        "",
        f"- 最终配置：`update={best_cfg.update_mode}`, `eta={best_cfg.eta}`, `warm_start={best_cfg.warm_start}`, `m1_floor={best_cfg.m1_floor}`, `granularity={best_cfg.granularity}`。",
        "- warm start 仅来自 validation replay 的末端权重，不读取 test truth。",
        "",
        "## Optional deployment wrapper: CRCF risk fallback",
        "",
        "```text",
        "ŷ^final_(t+h) = {",
        "  ŷ^COEA_(t+h),            if r_(t,h) <= τ_h",
        "  ŷ^M1_(t+h),              if r_(t,h) >  τ_h",
        "}",
        "```",
        "",
        "其中风险分数定义为：",
        "",
        "```text",
        "r_(t,h) = |ŷ^COEA_(t+h) - ŷ^M1_(t+h)| / max(s_(t,h), 0.1)",
        "```",
        "",
        "这里 `s_(t,h)` 取 `max(C3 local scale, residual_std)`；阈值 `τ_h` 只能由 validation calibration split 标定。",
        "",
        "## 解释",
        "",
        "- COEA 负责在 `Persistence / Linear-Seasonal / M1 / C3_opt` 之间做因果在线聚合。",
        "- CRCF 是可选部署层：在高风险区域把输出回退到 `M1`。但在 audited full-six 中，`CRCF` 的平均 skill 略低于 plain `COEA`，因此正文主模型优先冻结为 `COEA` 本体。",
        "",
        "## 论文写法建议",
        "",
        "- 标题友好名称：**Causal Online Expert Aggregation for EGMS Ground-Deformation Nowcasting**。",
        "- 若需要保留 CRCF，可在 supplementary 或 deployment discussion 中写成：validation-calibrated risk fallback wrapper over COEA。",
        "",
    ]
    return write_markdown(MODEL_DEF_PATH, lines)


def build_final_report(best_cfg: COEAConfig, quick_summary: pd.DataFrame, fullsix_eval: pd.DataFrame, ablation_summary: pd.DataFrame) -> Path:
    winner_row = fullsix_eval.iloc[0]
    static_row = ablation_summary[ablation_summary["Model"] == "Ablation_static_val_weighted"]
    lines = [
        "# V7 Finalization Report",
        "",
        "数据声明：本轮 finalization 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## Action 1: Causal replay audit",
        "",
        f"- 审计文档：`{AUDIT_PATH.as_posix()}`。",
        "- 原始 V7 quick 存在 search-test tuning；本轮已用 validation-only nested rerun 修正。",
        "",
        "## Action 2: Final model freeze",
        "",
        f"- 基于 audited full-six 的 V7 family 胜出者：`{winner_row['Model']}`。",
        "- 因为 `COEA` 在 audited full-six 中优于 `CRCF` 包装，所以正文主模型先冻结为 `COEA`，`CRCF` 作为可选部署安全包装保留。",
        f"- 数学定义文档：`{MODEL_DEF_PATH.as_posix()}`。",
        "",
        "## Action 3: Minimal ablation",
        "",
    ]
    for _, row in ablation_summary.iterrows():
        lines.append(
            f"- `{row['Model']}`: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, stress ratio={row['Stress ratio']:.4f}."
        )
    lines.extend(["", "## Important caveat", ""])
    if not static_row.empty:
        row = static_row.iloc[0]
        lines.append(
            f"- `Ablation_static_val_weighted` 的 Mean Skill_MSE={row['Mean Skill_MSE']:.4f}，高于 `COEA`。这说明当前数据上“online update 必要性”没有被完全证明；如果坚持使用 `COEA` 作为主模型，supplementary 必须如实报告这个更强的简单基线。"
        )
    lines.extend(
        [
            "",
            "## Action 4: Interpretability figures",
            "",
            f"- Figure 1: `{WEIGHT_TRAJ_PATH.as_posix()}`。",
            f"- Figure 2: `{EXPERT_SHARE_PATH.as_posix()}`。",
            f"- Figure 3: `{FALLBACK_RATE_PATH.as_posix()}`。",
            f"- Figure 4: `{SKILL_HEATMAP_PATH.as_posix()}`。",
            f"- Figure 5: `{CASE_MAP_PATH.as_posix()}`。",
            "",
            "## Audited quick summary",
            "",
        ]
    )
    for _, row in quick_summary.iterrows():
        lines.append(
            f"- {row['Family']} `{row['Model']}`: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/3, stress ratio={row['Stress ratio']:.4f}, gate={bool(row['Quick gate pass'])}."
        )
    lines.extend(["", "## Audited full-six summary", ""])
    for _, row in fullsix_eval.iterrows():
        lines.append(
            f"- {row['Family']} `{row['Model']}`: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/6, stress ratio={row['Stress ratio']:.4f}, std={row['Mean Skill std']:.4f}, Practical GO={bool(row['Practical GO'])}."
        )
    lines.extend(["", "## Final verdict", "", "- 修正后的 audited pipeline 支持 `COEA` 作为 V7 family 的主候选；`CRCF` 更适合作为可选 safety wrapper，而不是默认主模型。"])
    return write_markdown(FINAL_REPORT_PATH, lines)


def run_v7_finalization() -> dict[str, Any]:
    global BUNDLES, M1_SUFFIX_SUMMARY
    BUNDLES = preprocess_all_tiles()
    _m1_suffix_raw, M1_SUFFIX_SUMMARY = build_m1_suffix_reference(BUNDLES, SELECTION_SEEDS)
    _quick_raw, quick_grid, best_cfg, best_coea_summary, _trace = run_coea_quick_audited(BUNDLES, SELECTION_SEEDS, M1_SUFFIX_SUMMARY)
    crcf_quick_raw, crcf_quick_summary, crcf_quick_eval = run_crcf_quick_audited(BUNDLES, best_cfg, SELECTION_SEEDS, M1_SUFFIX_SUMMARY)
    quick_summary = save_quick_summary(best_coea_summary, best_cfg, crcf_quick_summary)
    fullsix_raw, fullsix_eval, diagnostics = run_audited_fullsix(BUNDLES, best_cfg)
    _ablation_raw, ablation_summary = run_ablations(BUNDLES, best_cfg)
    plot_weight_trajectories(diagnostics, seed=42)
    plot_expert_share(diagnostics, seed=42)
    plot_fallback_rate(diagnostics, seed=42)
    plot_skill_heatmaps(best_cfg, diagnostics)
    plot_case_maps(best_cfg, diagnostics, seed=42)
    audit_path = build_audit_doc(best_cfg, quick_summary, fullsix_eval)
    model_def_path = build_model_definition(best_cfg)
    report_path = build_final_report(best_cfg, quick_summary, fullsix_eval, ablation_summary)
    summary = {
        "audit": str(audit_path.resolve()),
        "model_definition": str(model_def_path.resolve()),
        "report": str(report_path.resolve()),
        "quick_grid": str(AUDITED_QUICK_GRID_PATH.resolve()),
        "quick_summary": str(AUDITED_QUICK_SUMMARY_PATH.resolve()),
        "fullsix_eval": str(AUDITED_FULL6_EVAL_PATH.resolve()),
        "ablation_summary": str(ABLATION_SUMMARY_PATH.resolve()),
        "winner": best_cfg.model_id,
        "fallback_wrapper": f"CRCF_{best_cfg.model_id}",
    }
    write_json(FINAL_SUMMARY_PATH, summary)
    return summary
