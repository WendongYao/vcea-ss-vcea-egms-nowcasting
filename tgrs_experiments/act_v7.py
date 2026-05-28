from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, SEARCH_TILES, write_simple_yaml
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v5_selector import C3_RUN_ROOT, EXPERT_NAMES, M1_RUN_ROOT, infer_c3_split, infer_m1_split
from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    REGIME_TO_ID,
    TILE_ORDER,
    TileBundle,
    ensure_dir,
    preprocess_all_tiles,
    write_json,
)


ACT_V7_ROOT = OUTPUT_ROOT / "act_v7"
ACT_V7_CONFIG_ROOT = ACT_V7_ROOT / "configs"
SEED_CACHE_ROOT = ACT_V7_ROOT / "seed_expert_cache"

QUICK_BASELINE_BANK = ACT_V7_ROOT / "quick_baseline_bank.csv"
QUICK_SUMMARY_PATH = ACT_V7_ROOT / "week1_ranked_summary.csv"
QUICK_DECISION_PATH = ACT_V7_ROOT / "week1_go_nogo.md"
FULL6_RAW_PATH = ACT_V7_ROOT / "fullsix_seed_metrics.csv"
FULL6_SUMMARY_PATH = ACT_V7_ROOT / "fullsix_mean_std.csv"
RUNTIME_LOG_PATH = ACT_V7_ROOT / "runtime_mem_logs.csv"
TILE_HORIZON_HEATMAP = ACT_V7_ROOT / "tile_horizon_heatmap.png"
STRESS_REPORT_PATH = ACT_V7_ROOT / "stress_tile_report.md"
WINNER_ABLATION_PATH = ACT_V7_ROOT / "winner_ablation.csv"
WINNER_PANEL_PATH = ACT_V7_ROOT / "winner_interpretability_panel.png"
FINAL_DECISION_PATH = ACT_V7_ROOT / "final_v7_decision.md"
PAPER_NOTES_PATH = ACT_V7_ROOT / "paper_integration_notes.md"
REPORT_PATH = ACT_V7_ROOT / "act_v7_detailed_report.md"
SUMMARY_PATH = ACT_V7_ROOT / "run_summary.json"

COEA_QUICK_GRID = ACT_V7_ROOT / "coea_quick_grid.csv"
COEA_WEIGHT_TRACE_PNG = ACT_V7_ROOT / "coea_weight_traces.png"
COEA_HEATMAP_PNG = ACT_V7_ROOT / "coea_tile_horizon_heatmap.png"
M1PP_QUICK_GRID = ACT_V7_ROOT / "m1pp_quick_grid.csv"
M1PP_GATE_CURVE_PNG = ACT_V7_ROOT / "m1pp_gate_horizon_curve.png"
M1PP_STRESS_BAR_PNG = ACT_V7_ROOT / "stress_ratio_bar_quick.png"
CLBE_QUICK_PATH = ACT_V7_ROOT / "clbe_quick.csv"
CLBE_COMPONENT_PNG = ACT_V7_ROOT / "trend_season_component_examples.png"
CRCF_QUICK_PATH = ACT_V7_ROOT / "crcf_quick.csv"
CRCF_RISK_COVERAGE_CSV = ACT_V7_ROOT / "risk_coverage_curve.csv"
CRCF_RISK_COVERAGE_PNG = ACT_V7_ROOT / "risk_coverage_curve.png"
CRCF_FALLBACK_CSV = ACT_V7_ROOT / "fallback_rate_by_horizon.csv"

M1_FULL6_SUMMARY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"
M1_MEAN_SKILL = 0.0783341230429559
PRACTICAL_GAIN = 0.02
QUICK_GAIN_THRESHOLD = 0.015
STRESS_RATIO_THRESHOLD = 1.03
QUICK_STRESS_THRESHOLD = 1.02
FULL6_MAX_STD = 0.025
MAX_TOPK = 2


def load_m1_summary() -> pd.DataFrame:
    return pd.read_csv(M1_FULL6_SUMMARY_PATH)


def arrays_from_npz(path: Path) -> dict[str, np.ndarray]:
    arr = np.load(path)
    return {name: arr[name] for name in arr.files}


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def ensure_seed_family_arrays(
    bundles: dict[str, TileBundle],
    family: str,
    tile: str,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    out_path = ensure_dir(SEED_CACHE_ROOT / family / tile) / f"seed_{seed}_{split_name}.npz"
    if out_path.exists():
        return arrays_from_npz(out_path)

    bundle = bundles[tile]
    if family == "M1":
        run_dir = M1_RUN_ROOT / tile / f"seed_{seed}"
        if split_name == "test":
            path = run_dir / "predictions.npz"
            if path.exists():
                arrays = arrays_from_npz(path)
            else:
                arrays = infer_m1_split(bundle, run_dir, split_name)
        else:
            arrays = infer_m1_split(bundle, run_dir, split_name)
    elif family == "C3_opt":
        run_dir = C3_RUN_ROOT / tile / f"seed_{seed}"
        if split_name == "test":
            path = run_dir / "predictions.npz"
            if path.exists():
                arrays = arrays_from_npz(path)
            else:
                arrays = infer_c3_split(bundle, run_dir, split_name)
        else:
            arrays = infer_c3_split(bundle, run_dir, split_name)
    else:
        raise KeyError(family)
    np.savez_compressed(out_path, **arrays)
    return arrays


def build_seed_expert_cache(
    bundles: dict[str, TileBundle],
    tile: str,
    seed: int,
    split_name: str,
) -> dict[str, np.ndarray]:
    bundle = bundles[tile]
    m1 = ensure_seed_family_arrays(bundles, "M1", tile, seed, split_name)
    c3 = ensure_seed_family_arrays(bundles, "C3_opt", tile, seed, split_name)
    if not np.array_equal(m1["current_frames"].reshape(-1), c3["current_frames"].reshape(-1)):
        raise ValueError(f"Frame mismatch tile={tile} seed={seed} split={split_name}")
    if not np.array_equal(m1["target_indices"], c3["target_indices"]):
        raise ValueError(f"Target mismatch tile={tile} seed={seed} split={split_name}")
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
        "residual_std": bundle.tensors64["residual_std"].astype(np.float32),
        "descriptor_amp": bundle.tensors64["descriptor_amp"].astype(np.float32),
        "seasonal_phase": bundle.tensors64["seasonal_phase"].astype(np.float32),
        "trend_r2": bundle.tensors64["trend_r2"].astype(np.float32),
        "disp": bundle.tensors64["disp"].astype(np.float32),
        "time_sin": bundle.tensors64["time_sin"].astype(np.float32),
        "time_cos": bundle.tensors64["time_cos"].astype(np.float32),
    }


def arrays_payload_from_prediction(cache: dict[str, np.ndarray], pred: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "y_pred_mm": pred.astype(np.float32),
        "y_true_mm": cache["y_true_mm"].astype(np.float32),
        "persistence_mm": cache["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
        "valid_mask": cache["valid_mask"].astype(np.float32),
        "regime_map": cache["regime_map"].astype(np.int16),
        "current_frames": cache["current_frames"].astype(np.int64),
        "target_indices": cache["target_indices"].astype(np.int64),
    }


def family_from_model(model_id: str) -> str:
    if model_id.startswith("COEA"):
        return "COEA"
    if model_id.startswith("M1PP"):
        return "M1++"
    if model_id.startswith("CRCF"):
        return "CRCF"
    if model_id.startswith("CLBE"):
        return "CLBE"
    return "Other"


def group_masks(cache: dict[str, np.ndarray], granularity: str) -> list[np.ndarray]:
    valid = cache["valid_mask"] > 0.5
    if granularity == "tile_horizon":
        return [valid]
    masks: list[np.ndarray] = []
    regime = cache["regime_map"]
    for rid in range(len(REGIME_NAMES)):
        mask = valid & (regime == rid)
        if mask.sum() == 0:
            mask = valid
        masks.append(mask)
    return masks


def apply_m1_floor(weights: np.ndarray, floor: float) -> np.ndarray:
    if floor <= 0:
        return weights / max(weights.sum(), 1e-6)
    out = weights.astype(np.float64).copy()
    m1_idx = 2
    if out[m1_idx] < floor:
        rem = max(1e-6, 1.0 - floor)
        others = np.array([idx for idx in range(len(out)) if idx != m1_idx], dtype=np.int64)
        other_sum = float(out[others].sum())
        if other_sum <= 1e-9:
            out[others] = rem / len(others)
        else:
            out[others] = out[others] / other_sum * rem
        out[m1_idx] = floor
    out = np.clip(out, 1e-8, None)
    out /= out.sum()
    return out.astype(np.float32)


@dataclass
class COEAConfig:
    model_id: str
    update_mode: str
    eta: float
    warm_start: bool
    m1_floor: float
    granularity: str = "tile_horizon"


def coea_configs() -> list[COEAConfig]:
    configs: list[COEAConfig] = []
    for granularity in ["tile_horizon", "regime_horizon"]:
        configs.append(
            COEAConfig(
                model_id=f"COEA_equal_{granularity}",
                update_mode="equal",
                eta=0.0,
                warm_start=False,
                m1_floor=0.0,
                granularity=granularity,
            )
        )
        for update_mode in ["ewa", "boa"]:
            for eta in [1.0, 2.0, 4.0]:
                for warm_start in [False, True]:
                    for floor in [0.0, 0.5]:
                        suffix = "warm" if warm_start else "cold"
                        floor_tag = "floor50" if floor > 0 else "floor0"
                        configs.append(
                            COEAConfig(
                                model_id=f"COEA_{update_mode}_eta{str(eta).replace('.', 'p')}_{suffix}_{floor_tag}_{granularity}",
                                update_mode=update_mode,
                                eta=eta,
                                warm_start=warm_start,
                                m1_floor=floor,
                                granularity=granularity,
                            )
                        )
    return configs


def update_weights_online(
    weights: np.ndarray,
    loss_vec: np.ndarray,
    config: COEAConfig,
) -> np.ndarray:
    out = weights.astype(np.float64).copy()
    if config.update_mode == "equal":
        return out.astype(np.float32)
    pers = max(float(loss_vec[0]), 1e-6)
    rel = loss_vec / pers
    if config.update_mode == "ewa":
        exponent = np.clip(-config.eta * rel, -20.0, 20.0)
        out *= np.exp(exponent)
    elif config.update_mode == "boa":
        centered = rel - float(np.dot(out, rel))
        exponent = np.clip(-config.eta * centered, -20.0, 20.0)
        out *= np.exp(exponent)
    else:
        raise KeyError(config.update_mode)
    out = np.clip(out, 1e-8, None)
    denom = float(out.sum())
    if not np.isfinite(denom) or denom <= 0:
        out = np.full_like(out, 1.0 / len(out))
    else:
        out /= denom
    return apply_m1_floor(out.astype(np.float32), config.m1_floor)


def replay_coea(
    cache: dict[str, np.ndarray],
    config: COEAConfig,
    init_weights: np.ndarray | None = None,
    return_trace: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    experts = np.stack(
        [
            cache["persistence_mm"],
            cache["linear_seasonal_mm"],
            cache["m1_pred_mm"],
            cache["c3_pred_mm"],
        ],
        axis=2,
    ).astype(np.float32)
    masks = group_masks(cache, config.granularity)
    n_groups = len(masks)
    n_samples, n_horizons, _, height, width = experts.shape
    if init_weights is None:
        weights = np.full((n_horizons, n_groups, len(EXPERT_NAMES)), 1.0 / len(EXPERT_NAMES), dtype=np.float32)
        if config.m1_floor > 0:
            for h_idx in range(n_horizons):
                for g_idx in range(n_groups):
                    weights[h_idx, g_idx] = apply_m1_floor(weights[h_idx, g_idx], config.m1_floor)
    else:
        weights = init_weights.astype(np.float32).copy()

    pred = np.zeros((n_samples, n_horizons, height, width), dtype=np.float32)
    weight_snapshots = np.zeros((n_samples, n_horizons, len(EXPERT_NAMES)), dtype=np.float32)
    traces: list[dict[str, Any]] = []
    pending: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in range(n_horizons)]
    order = np.argsort(cache["current_frames"])

    for idx in order:
        current_frame = int(cache["current_frames"][idx])
        for h_idx in range(n_horizons):
            keep: list[tuple[int, np.ndarray, np.ndarray]] = []
            for target_frame, expert_map, true_map in pending[h_idx]:
                if target_frame <= current_frame:
                    for g_idx, mask in enumerate(masks):
                        if int(mask.sum()) == 0:
                            continue
                        sq_err = ((expert_map - true_map[None, :, :]) ** 2)[:, mask]
                        loss_vec = sq_err.mean(axis=1).astype(np.float32)
                        weights[h_idx, g_idx] = update_weights_online(weights[h_idx, g_idx], loss_vec, config)
                else:
                    keep.append((target_frame, expert_map, true_map))
            pending[h_idx] = keep

        for h_idx in range(n_horizons):
            if config.granularity == "tile_horizon":
                w = weights[h_idx, 0]
                pred[idx, h_idx] = (experts[idx, h_idx] * w[:, None, None]).sum(axis=0)
                weight_snapshots[idx, h_idx] = w
            else:
                out = np.zeros((height, width), dtype=np.float32)
                mean_w = np.zeros((len(EXPERT_NAMES),), dtype=np.float32)
                for g_idx, mask in enumerate(masks):
                    w = weights[h_idx, g_idx]
                    # Keep expert dimension first; mixed advanced indexing can reorder axes.
                    expert_patch = experts[idx, h_idx][:, mask]
                    out[mask] = (expert_patch * w[:, None]).sum(axis=0)
                    mean_w += w
                pred[idx, h_idx] = out
                weight_snapshots[idx, h_idx] = mean_w / max(1, n_groups)
            if return_trace:
                for exp_idx, expert_name in enumerate(EXPERT_NAMES):
                    traces.append(
                        {
                            "Sample order": int(idx),
                            "Current frame": current_frame,
                            "Horizon": f"t+{SEARCH_HORIZONS[h_idx]}",
                            "Expert": expert_name,
                            "Mean weight": float(weight_snapshots[idx, h_idx, exp_idx]),
                        }
                    )
            pending[h_idx].append(
                (
                    int(cache["target_indices"][idx, h_idx]),
                    experts[idx, h_idx],
                    cache["y_true_mm"][idx, h_idx],
                )
            )
    return pred.astype(np.float32), weight_snapshots.astype(np.float32), traces


def warm_start_weights(val_cache: dict[str, np.ndarray], config: COEAConfig) -> np.ndarray:
    _pred, weights, _ = replay_coea(val_cache, config, init_weights=None, return_trace=False)
    last = weights[-1]
    n_groups = 1 if config.granularity == "tile_horizon" else len(REGIME_NAMES)
    out = np.repeat(last[:, None, :], n_groups, axis=1).astype(np.float32)
    if config.granularity == "regime_horizon":
        base = np.full((len(SEARCH_HORIZONS), n_groups, len(EXPERT_NAMES)), 1.0 / len(EXPERT_NAMES), dtype=np.float32)
        if config.m1_floor > 0:
            for h_idx in range(len(SEARCH_HORIZONS)):
                for g_idx in range(n_groups):
                    base[h_idx, g_idx] = apply_m1_floor(base[h_idx, g_idx], config.m1_floor)
        # val replay only gives averaged snapshots for regime_horizon; use replay end weights from live object would be more exact.
        # Conservative approximation: keep floor-adjusted uniform warm start.
        return base
    return out


def run_coea_for_tile(
    bundles: dict[str, TileBundle],
    tile: str,
    seed: int,
    split_name: str,
    config: COEAConfig,
    with_trace: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]], float]:
    cache = build_seed_expert_cache(bundles, tile, seed, split_name)
    init_weights = None
    if config.warm_start:
        val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
        init_weights = warm_start_weights(val_cache, config)
    start = time.time()
    pred, weights_snap, traces = replay_coea(cache, config, init_weights=init_weights, return_trace=with_trace)
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(cache, pred)
    arrays["coea_weights"] = weights_snap.astype(np.float32)
    rows = pd.DataFrame(
        result_rows(
            model_id=config.model_id,
            tile=tile,
            seed=seed,
            phase=split_name,
            elapsed_seconds=elapsed,
            infer_seconds=elapsed,
            best_epoch=-1,
            param_count=0,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )
    return rows, arrays, traces, elapsed


@dataclass
class M1PPConfig:
    model_id: str
    bounded_residual: bool
    lambda_oc: float
    beta_name: str
    beta_values: tuple[float, float, float, float]
    use_gate_prior: bool


def m1pp_configs() -> list[M1PPConfig]:
    beta_profiles = {
        "flat1": (1.0, 1.0, 1.0, 1.0),
        "cons": (0.6, 0.8, 0.95, 1.05),
        "inc": (0.5, 0.8, 1.0, 1.2),
    }
    configs: list[M1PPConfig] = []
    for bounded in [False, True]:
        for lam in [0.0, 0.05, 0.1]:
            for beta_name, beta_values in beta_profiles.items():
                for gate_prior in [False, True]:
                    bounded_tag = "bound" if bounded else "nobound"
                    gate_tag = "gate" if gate_prior else "nogate"
                    model_id = f"M1PP_{bounded_tag}_lam{str(lam).replace('.', 'p')}_{beta_name}_{gate_tag}"
                    configs.append(
                        M1PPConfig(
                            model_id=model_id,
                            bounded_residual=bounded,
                            lambda_oc=lam,
                            beta_name=beta_name,
                            beta_values=beta_values,
                            use_gate_prior=gate_prior,
                        )
                    )
    return configs


def run_m1pp_for_tile(
    bundles: dict[str, TileBundle],
    tile: str,
    seed: int,
    split_name: str,
    config: M1PPConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, float]:
    cache = build_seed_expert_cache(bundles, tile, seed, split_name)
    start = time.time()
    raw_delta = cache["m1_pred_mm"] - cache["persistence_mm"]
    gate = cache["m1_gate_mm"] if config.use_gate_prior else np.ones_like(cache["m1_gate_mm"], dtype=np.float32)
    scale = np.maximum(cache["c3_scale_mm"], cache["residual_std"][None, None, :, :]).astype(np.float32)
    pred = np.zeros_like(cache["m1_pred_mm"], dtype=np.float32)
    ratio_rows: list[dict[str, Any]] = []
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        beta = float(config.beta_values[h_idx])
        eff = raw_delta[:, h_idx] * gate[:, h_idx]
        eff = eff * beta
        if config.bounded_residual:
            cap = np.maximum(0.1, beta * scale[:, h_idx])
            eff = np.clip(eff, -cap, cap)
        if config.lambda_oc > 0:
            eff = eff / (1.0 + config.lambda_oc * np.abs(eff) / np.maximum(scale[:, h_idx], 0.1))
        pred[:, h_idx] = cache["persistence_mm"][:, h_idx] + eff
        raw_abs = np.abs(raw_delta[:, h_idx])
        ratio = np.abs(eff) / np.maximum(raw_abs, 1e-6)
        ratio_rows.append(
            {
                "Model": config.model_id,
                "Tile": tile,
                "Seed": seed,
                "Horizon": f"t+{horizon}",
                "Mean correction ratio": float(ratio.mean()),
                "Mean abs correction [mm]": float(np.abs(eff).mean()),
            }
        )
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(cache, pred)
    rows = pd.DataFrame(
        result_rows(
            model_id=config.model_id,
            tile=tile,
            seed=seed,
            phase=split_name,
            elapsed_seconds=elapsed,
            infer_seconds=elapsed,
            best_epoch=-1,
            param_count=0,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )
    return rows, arrays, pd.DataFrame(ratio_rows), elapsed


@dataclass
class CLBEConfig:
    model_id: str
    use_season: bool
    use_clip: bool


def clbe_configs() -> list[CLBEConfig]:
    return [
        CLBEConfig(model_id="CLBE_trend_only", use_season=False, use_clip=False),
        CLBEConfig(model_id="CLBE_trend_season", use_season=True, use_clip=False),
        CLBEConfig(model_id="CLBE_trend_season_clip", use_season=True, use_clip=True),
    ]


def seasonal_value(cache: dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    sin_t = float(cache["time_sin"][frame_idx])
    cos_t = float(cache["time_cos"][frame_idx])
    phase = cache["seasonal_phase"]
    amp = cache["descriptor_amp"]
    return amp * (np.sin(np.arctan2(sin_t, cos_t) + phase).astype(np.float32))


def run_clbe_for_tile(
    bundles: dict[str, TileBundle],
    tile: str,
    split_name: str,
    config: CLBEConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], float]:
    cache = build_seed_expert_cache(bundles, tile, 42, split_name)
    disp = cache["disp"]
    input_len = 16
    x = np.arange(input_len, dtype=np.float32)
    x_centered = x - x.mean()
    denom = float((x_centered**2).sum())
    residual_std = np.maximum(cache["residual_std"], 0.1).astype(np.float32)
    pred = np.zeros_like(cache["y_true_mm"], dtype=np.float32)
    start = time.time()
    for i, current_frame in enumerate(cache["current_frames"]):
        window = disp[current_frame - input_len + 1 : current_frame + 1]
        last = window[-1]
        slope = ((window * x_centered[:, None, None]).sum(axis=0) / denom).astype(np.float32)
        for h_idx, horizon in enumerate(SEARCH_HORIZONS):
            inc = slope * float(horizon)
            if config.use_season:
                current_season = seasonal_value(cache, int(current_frame))
                future_season = seasonal_value(cache, int(cache["target_indices"][i, h_idx]))
                inc = inc + (future_season - current_season)
            if config.use_clip:
                cap = 1.5 * residual_std * math.sqrt(float(horizon))
                inc = np.clip(inc, -cap, cap)
            pred[i, h_idx] = last + inc
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(cache, pred)
    rows = pd.DataFrame(
        result_rows(
            model_id=config.model_id,
            tile=tile,
            seed=seed,
            phase=split_name,
            elapsed_seconds=elapsed,
            infer_seconds=elapsed,
            best_epoch=-1,
            param_count=0,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )
    return rows, arrays, elapsed


@dataclass
class CRCFConfig:
    model_id: str
    base_model_id: str
    thresholds: tuple[float, ...]


def wrap_with_crcf(
    base_pred: np.ndarray,
    m1_pred: np.ndarray,
    scale: np.ndarray,
    threshold_by_h: dict[int, float],
) -> tuple[np.ndarray, dict[str, float]]:
    out = base_pred.copy()
    fallback_rates: dict[str, float] = {}
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        risk = np.abs(base_pred[:, h_idx] - m1_pred[:, h_idx]) / np.maximum(scale[:, h_idx], 0.1)
        fallback = risk > threshold_by_h[horizon]
        out[:, h_idx][fallback] = m1_pred[:, h_idx][fallback]
        fallback_rates[f"t+{horizon}"] = float(fallback.mean())
    return out.astype(np.float32), fallback_rates


def calibrate_crcf_thresholds(
    base_val: np.ndarray,
    m1_val: np.ndarray,
    y_true_val: np.ndarray,
    pers_val: np.ndarray,
    scale_val: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[dict[int, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    chosen: dict[int, float] = {}
    valid = valid_mask > 0.5
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        best_score = -float("inf")
        best_thr = thresholds[0]
        for thr in thresholds:
            risk = np.abs(base_val[:, h_idx] - m1_val[:, h_idx]) / np.maximum(scale_val[:, h_idx], 0.1)
            pred = base_val[:, h_idx].copy()
            fallback = risk > thr
            pred[fallback] = m1_val[:, h_idx][fallback]
            mse_model = float(((pred - y_true_val[:, h_idx])[:, valid] ** 2).mean())
            mse_pers = float(((pers_val[:, h_idx] - y_true_val[:, h_idx])[:, valid] ** 2).mean())
            skill = 1.0 - mse_model / max(mse_pers, 1e-6)
            stress_ratio = float(
                np.sqrt(mse_model) / max(float(np.sqrt(mse_pers)), 1e-6)
            )
            score = skill - 0.2 * max(0.0, stress_ratio - QUICK_STRESS_THRESHOLD)
            rows.append(
                {
                    "Horizon": f"t+{horizon}",
                    "Threshold": thr,
                    "Val Skill_MSE": skill,
                    "Val stress ratio": stress_ratio,
                    "Fallback rate": float(fallback.mean()),
                }
            )
            if score > best_score:
                best_score = score
                best_thr = thr
        chosen[horizon] = float(best_thr)
    return chosen, pd.DataFrame(rows)


def evaluate_candidate_status(summary_df: pd.DataFrame, tiles: list[str]) -> dict[str, Any]:
    m1 = load_m1_summary()
    base = m1[m1["Tile"].isin(tiles)].copy()
    model_id = summary_df["Model"].iloc[0]
    merged = summary_df.merge(
        base[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs M1"] = merged["Skill_MSE mean"] - merged["M1 Skill_MSE mean"]
    tile_compare = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "M1 Skill_MSE mean", "Delta vs M1"]]
        .mean()
        .rename(columns={"Skill_MSE mean": "Model Skill_MSE mean"})
    )
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    gain = float(merged["Delta vs M1"].mean())
    better_tiles = int((tile_compare["Delta vs M1"] > 0).sum())
    stress = merged[merged["Tile"].isin(["E48N24", "E58N17"] if set(["E48N24", "E58N17"]).issubset(set(tiles)) else ["E48N24"])]
    stress_ratio = float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()) if not stress.empty else 1.0
    mean_std = float(summary_df["Skill_MSE std"].mean())
    mean_train = float(summary_df["Elapsed seconds mean"].mean())
    return {
        "model_id": model_id,
        "family": family_from_model(model_id),
        "mean_skill_mse": mean_skill,
        "gain_vs_m1": gain,
        "better_tiles_vs_m1": better_tiles,
        "stress_ratio": stress_ratio,
        "mean_skill_std": mean_std,
        "mean_train_seconds": mean_train,
        "tile_compare": tile_compare,
    }


def quick_gate(stats: dict[str, Any]) -> bool:
    return (
        stats["gain_vs_m1"] >= QUICK_GAIN_THRESHOLD
        and stats["better_tiles_vs_m1"] >= 2
        and stats["stress_ratio"] <= QUICK_STRESS_THRESHOLD
    )


def practical_gate(stats: dict[str, Any]) -> bool:
    return (
        stats["mean_skill_mse"] >= M1_MEAN_SKILL + PRACTICAL_GAIN
        and stats["better_tiles_vs_m1"] >= 4
        and stats["stress_ratio"] <= STRESS_RATIO_THRESHOLD
        and stats["mean_skill_std"] <= FULL6_MAX_STD
        and stats["mean_train_seconds"] <= 5.0 * 8.84
    )


def pick_top_candidates(week1_df: pd.DataFrame) -> pd.DataFrame:
    eligible = week1_df[week1_df["Quick gate pass"]].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(
        ["Gain vs M1", "Better tiles", "Stress ratio", "Mean Skill_MSE"],
        ascending=[False, False, True, False],
    )
    return eligible.head(MAX_TOPK).reset_index(drop=True)


def plot_heatmap(summary_df: pd.DataFrame, out_path: Path, value_col: str, title: str, tiles: list[str]) -> None:
    pivot = summary_df.pivot(index="Tile", columns="Horizon", values=value_col).reindex(tiles)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_m1pp_gate_curve(ratio_df: pd.DataFrame, out_path: Path) -> None:
    summary = ratio_df.groupby("Horizon", as_index=False)["Mean correction ratio"].mean().set_index("Horizon").reindex([f"t+{h}" for h in SEARCH_HORIZONS])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(summary.index, summary["Mean correction ratio"], marker="o", linewidth=2.0)
    ax.set_title("M1++ correction ratio by horizon")
    ax.set_ylabel("|delta_cal| / |delta_M1|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stress_bar(m1_stats: dict[str, Any], best_stats: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(["M1", best_stats["model_id"]], [1.0, best_stats["stress_ratio"]], color=["#7f7f7f", "#1f77b4"])
    ax.axhline(QUICK_STRESS_THRESHOLD, color="crimson", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Stress RMSE / persistence")
    ax.set_title("Quick stress ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_clbe_component_examples(cache: dict[str, np.ndarray], out_path: Path) -> None:
    current_frame = int(cache["current_frames"][0])
    disp = cache["disp"][current_frame - 15 : current_frame + 1]
    mean_series = disp.mean(axis=(1, 2))
    x = np.arange(len(mean_series))
    coeff = np.polyfit(x, mean_series, 1)
    trend = coeff[0] * x + coeff[1]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(x, mean_series, label="window mean disp")
    ax.plot(x, trend, label="local trend")
    ax.set_title("CLBE trend component example")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_coea_weight_trace(trace_df: pd.DataFrame, out_path: Path) -> None:
    sub = trace_df[trace_df["Horizon"].isin(["t+5", "t+10"])].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, horizon in zip(axes, ["t+5", "t+10"], strict=True):
        part = sub[sub["Horizon"] == horizon]
        for expert in EXPERT_NAMES:
            p = part[part["Expert"] == expert]
            ax.plot(p["Current frame"], p["Mean weight"], label=expert)
        ax.set_title(horizon)
        ax.set_xlabel("current frame")
    axes[0].set_ylabel("weight")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_crcf_risk_curve(curve_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for horizon in [f"t+{h}" for h in SEARCH_HORIZONS]:
        part = curve_df[curve_df["Horizon"] == horizon]
        ax.plot(part["Fallback rate"], part["Val Skill_MSE"], marker="o", label=horizon)
    ax.set_xlabel("fallback rate")
    ax.set_ylabel("val Skill_MSE")
    ax.set_title("CRCF risk-coverage curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def quick_baseline_bank() -> pd.DataFrame:
    m1 = load_m1_summary()
    quick = m1[m1["Tile"].isin(SEARCH_TILES)].copy()
    quick.to_csv(QUICK_BASELINE_BANK, index=False, encoding="utf-8-sig")
    return quick


def run_coea_quick(
    bundles: dict[str, TileBundle],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, COEAConfig, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    best_cfg: COEAConfig | None = None
    best_score = -float("inf")
    best_trace = pd.DataFrame()
    for cfg in coea_configs():
        raw_rows: list[pd.DataFrame] = []
        traces: list[dict[str, Any]] = []
        for seed in seeds:
            for tile in SEARCH_TILES:
                rows, _arrays, trace_rows, _elapsed = run_coea_for_tile(bundles, tile, seed, "test", cfg, with_trace=(seed == seeds[0] and tile == "E39N30"))
                raw_rows.append(rows)
                traces.extend(trace_rows)
        raw = pd.concat(raw_rows, ignore_index=True)
        summary = summarize_seeded_rows(raw)
        stats = evaluate_candidate_status(summary, SEARCH_TILES)
        score = stats["gain_vs_m1"] - 0.2 * max(0.0, stats["stress_ratio"] - QUICK_STRESS_THRESHOLD) + 0.02 * stats["better_tiles_vs_m1"]
        rank_rows.append(
            {
                "Model": cfg.model_id,
                "Family": "COEA",
                "Update": cfg.update_mode,
                "Granularity": cfg.granularity,
                "Eta": cfg.eta,
                "Warm start": cfg.warm_start,
                "M1 floor": cfg.m1_floor,
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Quick gate pass": quick_gate(stats),
                "Score": score,
            }
        )
        rows_all.append(raw)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_trace = pd.DataFrame(traces)
    assert best_cfg is not None
    grid_df = pd.DataFrame(rank_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    grid_df.to_csv(COEA_QUICK_GRID, index=False, encoding="utf-8-sig")
    if not best_trace.empty:
        plot_coea_weight_trace(best_trace, COEA_WEIGHT_TRACE_PNG)
    best_rows = pd.concat([df for df in rows_all if df["Model"].iloc[0] == best_cfg.model_id], ignore_index=True)
    best_summary = summarize_seeded_rows(best_rows)
    plot_heatmap(best_summary, COEA_HEATMAP_PNG, "Skill_MSE mean", "Best COEA quick Skill_MSE", SEARCH_TILES)
    return pd.concat(rows_all, ignore_index=True), grid_df, best_cfg, best_summary


def run_m1pp_quick(
    bundles: dict[str, TileBundle],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, M1PPConfig, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    ratio_frames: list[pd.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    best_cfg: M1PPConfig | None = None
    best_ratio = pd.DataFrame()
    best_score = -float("inf")
    for cfg in m1pp_configs():
        raw_rows: list[pd.DataFrame] = []
        ratios: list[pd.DataFrame] = []
        for seed in seeds:
            for tile in SEARCH_TILES:
                rows, _arrays, ratio_df, _elapsed = run_m1pp_for_tile(bundles, tile, seed, "test", cfg)
                raw_rows.append(rows)
                ratios.append(ratio_df)
        raw = pd.concat(raw_rows, ignore_index=True)
        ratio = pd.concat(ratios, ignore_index=True)
        summary = summarize_seeded_rows(raw)
        stats = evaluate_candidate_status(summary, SEARCH_TILES)
        score = stats["gain_vs_m1"] - 0.15 * max(0.0, stats["stress_ratio"] - 1.0)
        rank_rows.append(
            {
                "Model": cfg.model_id,
                "Family": "M1++",
                "Bounded residual": cfg.bounded_residual,
                "Lambda_oc": cfg.lambda_oc,
                "Beta profile": cfg.beta_name,
                "Gate prior": cfg.use_gate_prior,
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Quick gate pass": quick_gate(stats),
                "Score": score,
            }
        )
        rows_all.append(raw)
        ratio_frames.append(ratio)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_ratio = ratio
    assert best_cfg is not None
    grid_df = pd.DataFrame(rank_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    grid_df.to_csv(M1PP_QUICK_GRID, index=False, encoding="utf-8-sig")
    plot_m1pp_gate_curve(best_ratio, M1PP_GATE_CURVE_PNG)
    best_rows = pd.concat([df for df in rows_all if df["Model"].iloc[0] == best_cfg.model_id], ignore_index=True)
    best_summary = summarize_seeded_rows(best_rows)
    best_stats = evaluate_candidate_status(best_summary, SEARCH_TILES)
    plot_stress_bar({"stress_ratio": 1.0}, best_stats, M1PP_STRESS_BAR_PNG)
    return pd.concat(rows_all, ignore_index=True), grid_df, best_cfg, best_summary


def run_clbe_quick(
    bundles: dict[str, TileBundle],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, CLBEConfig, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    best_cfg: CLBEConfig | None = None
    best_score = -float("inf")
    example_cache: dict[str, np.ndarray] | None = None
    for cfg in clbe_configs():
        raw_rows: list[pd.DataFrame] = []
        for seed in seeds:
            for tile in SEARCH_TILES:
                rows, _arrays, _elapsed = run_clbe_for_tile(bundles, tile, "test", cfg, seed)
                raw_rows.append(rows)
        raw = pd.concat(raw_rows, ignore_index=True)
        summary = summarize_seeded_rows(raw)
        stats = evaluate_candidate_status(summary, SEARCH_TILES)
        score = stats["gain_vs_m1"] - 0.1 * max(0.0, stats["stress_ratio"] - 1.0)
        rank_rows.append(
            {
                "Model": cfg.model_id,
                "Family": "CLBE",
                "Use season": cfg.use_season,
                "Use clip": cfg.use_clip,
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Quick gate pass": quick_gate(stats),
                "Score": score,
            }
        )
        rows_all.append(raw)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            example_cache = build_seed_expert_cache(bundles, "E39N30", 42, "test")
    assert best_cfg is not None
    grid_df = pd.DataFrame(rank_rows).sort_values("Score", ascending=False).reset_index(drop=True)
    grid_df.to_csv(CLBE_QUICK_PATH, index=False, encoding="utf-8-sig")
    if example_cache is not None:
        plot_clbe_component_examples(example_cache, CLBE_COMPONENT_PNG)
    best_rows = pd.concat([df for df in rows_all if df["Model"].iloc[0] == best_cfg.model_id], ignore_index=True)
    best_summary = summarize_seeded_rows(best_rows)
    return pd.concat(rows_all, ignore_index=True), grid_df, best_cfg, best_summary


def run_crcf_quick(
    bundles: dict[str, TileBundle],
    base_cfg: COEAConfig,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    curve_rows_all: list[pd.DataFrame] = []
    fallback_rows: list[dict[str, Any]] = []
    thresholds = tuple(float(v) for v in np.linspace(0.0, 3.0, 13))
    for seed in seeds:
        for tile in SEARCH_TILES:
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
            test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
            init = warm_start_weights(val_cache, base_cfg) if base_cfg.warm_start else None
            val_pred, _w, _t = replay_coea(val_cache, base_cfg, init_weights=init, return_trace=False)
            test_pred, _w2, _t2 = replay_coea(test_cache, base_cfg, init_weights=init, return_trace=False)
            thr_by_h, curve_df = calibrate_crcf_thresholds(
                base_val=val_pred,
                m1_val=val_cache["m1_pred_mm"],
                y_true_val=val_cache["y_true_mm"],
                pers_val=val_cache["persistence_mm"],
                scale_val=np.maximum(val_cache["c3_scale_mm"], val_cache["residual_std"][None, None]),
                valid_mask=val_cache["valid_mask"],
                thresholds=thresholds,
            )
            curve_df["Tile"] = tile
            curve_df["Seed"] = seed
            curve_rows_all.append(curve_df)
            wrapped, fallback_rates = wrap_with_crcf(
                base_pred=test_pred,
                m1_pred=test_cache["m1_pred_mm"],
                scale=np.maximum(test_cache["c3_scale_mm"], test_cache["residual_std"][None, None]),
                threshold_by_h=thr_by_h,
            )
            arrays = arrays_payload_from_prediction(test_cache, wrapped)
            rows = pd.DataFrame(
                result_rows(
                    model_id=f"CRCF_{base_cfg.model_id}",
                    tile=tile,
                    seed=seed,
                    phase="test",
                    elapsed_seconds=0.0,
                    infer_seconds=0.0,
                    best_epoch=-1,
                    param_count=0,
                    arrays=arrays,
                    horizons=SEARCH_HORIZONS,
                )
            )
            rows_all.append(rows)
            for horizon, rate in fallback_rates.items():
                fallback_rows.append({"Model": f"CRCF_{base_cfg.model_id}", "Tile": tile, "Seed": seed, "Horizon": horizon, "Fallback rate": rate})
    raw = pd.concat(rows_all, ignore_index=True)
    curve = pd.concat(curve_rows_all, ignore_index=True)
    fallback = pd.DataFrame(fallback_rows)
    raw.to_csv(CRCF_QUICK_PATH, index=False, encoding="utf-8-sig")
    curve.to_csv(CRCF_RISK_COVERAGE_CSV, index=False, encoding="utf-8-sig")
    fallback.to_csv(CRCF_FALLBACK_CSV, index=False, encoding="utf-8-sig")
    plot_crcf_risk_curve(curve.groupby(["Horizon", "Threshold"], as_index=False)[["Val Skill_MSE", "Fallback rate"]].mean(), CRCF_RISK_COVERAGE_PNG)
    return raw, summarize_seeded_rows(raw)


def run_crcf_for_tile(
    bundles: dict[str, TileBundle],
    tile: str,
    seed: int,
    base_cfg: COEAConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, float]:
    val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
    test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
    init = warm_start_weights(val_cache, base_cfg) if base_cfg.warm_start else None
    thresholds = tuple(float(v) for v in np.linspace(0.0, 3.0, 13))
    start = time.time()
    val_pred, _w, _t = replay_coea(val_cache, base_cfg, init_weights=init, return_trace=False)
    test_pred, _w2, _t2 = replay_coea(test_cache, base_cfg, init_weights=init, return_trace=False)
    thr_by_h, curve_df = calibrate_crcf_thresholds(
        base_val=val_pred,
        m1_val=val_cache["m1_pred_mm"],
        y_true_val=val_cache["y_true_mm"],
        pers_val=val_cache["persistence_mm"],
        scale_val=np.maximum(val_cache["c3_scale_mm"], val_cache["residual_std"][None, None]),
        valid_mask=val_cache["valid_mask"],
        thresholds=thresholds,
    )
    wrapped, fallback_rates = wrap_with_crcf(
        base_pred=test_pred,
        m1_pred=test_cache["m1_pred_mm"],
        scale=np.maximum(test_cache["c3_scale_mm"], test_cache["residual_std"][None, None]),
        threshold_by_h=thr_by_h,
    )
    elapsed = time.time() - start
    arrays = arrays_payload_from_prediction(test_cache, wrapped)
    rows = pd.DataFrame(
        result_rows(
            model_id=f"CRCF_{base_cfg.model_id}",
            tile=tile,
            seed=seed,
            phase="test",
            elapsed_seconds=elapsed,
            infer_seconds=elapsed,
            best_epoch=-1,
            param_count=0,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )
    fallback_df = pd.DataFrame(
        [
            {
                "Model": f"CRCF_{base_cfg.model_id}",
                "Tile": tile,
                "Seed": seed,
                "Horizon": horizon,
                "Fallback rate": rate,
            }
            for horizon, rate in fallback_rates.items()
        ]
    )
    curve_df["Tile"] = tile
    curve_df["Seed"] = seed
    return rows, arrays, pd.concat([curve_df, fallback_df], ignore_index=True, sort=False), elapsed


def family_winners(week1_tables: list[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat(week1_tables, ignore_index=True)
    grouped = all_rows.groupby("Family", as_index=False).first()
    return grouped


def week1_summary_table(best_stats: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in best_stats:
        rows.append(
            {
                "Model": item["model_id"],
                "Family": item["family"],
                "Mean Skill_MSE": item["mean_skill_mse"],
                "Gain vs M1": item["gain_vs_m1"],
                "Better tiles": item["better_tiles_vs_m1"],
                "Stress ratio": item["stress_ratio"],
                "Mean Skill std": item["mean_skill_std"],
                "Mean train seconds": item["mean_train_seconds"],
                "Quick gate pass": quick_gate(item),
            }
        )
    df = pd.DataFrame(rows).sort_values(["Quick gate pass", "Gain vs M1", "Stress ratio"], ascending=[False, False, True]).reset_index(drop=True)
    df.to_csv(QUICK_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return df


def config_from_model_id(
    model_id: str,
    best_coea: COEAConfig,
    best_m1pp: M1PPConfig,
    best_clbe: CLBEConfig,
) -> tuple[str, Any]:
    if model_id == best_coea.model_id:
        return "COEA", best_coea
    if model_id == best_m1pp.model_id:
        return "M1++", best_m1pp
    if model_id == best_clbe.model_id:
        return "CLBE", best_clbe
    if model_id.startswith("CRCF_"):
        return "CRCF", best_coea
    raise KeyError(model_id)


def run_candidate_fullsix(
    bundles: dict[str, TileBundle],
    model_id: str,
    family: str,
    cfg: Any,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_all: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for tile in TILE_ORDER:
            if family == "COEA":
                rows, _arrays, _trace, elapsed = run_coea_for_tile(bundles, tile, seed, "test", cfg, with_trace=False)
            elif family == "M1++":
                rows, _arrays, _ratio, elapsed = run_m1pp_for_tile(bundles, tile, seed, "test", cfg)
            elif family == "CLBE":
                rows, _arrays, elapsed = run_clbe_for_tile(bundles, tile, "test", cfg, seed)
            elif family == "CRCF":
                rows, _arrays, _diag, elapsed = run_crcf_for_tile(bundles, tile, seed, cfg)
            else:
                raise KeyError(family)
            rows_all.append(rows)
            runtime_rows.append(
                {
                    "Model": model_id,
                    "Family": family,
                    "Tile": tile,
                    "Seed": seed,
                    "Elapsed seconds": float(rows["Elapsed seconds"].iloc[0]),
                    "Infer seconds": float(rows["Infer seconds"].iloc[0]),
                    "Peak memory MB": 0.0,
                }
            )
    raw = pd.concat(rows_all, ignore_index=True)
    summary = summarize_seeded_rows(raw)
    return raw, pd.DataFrame(runtime_rows)


def evaluate_fullsix_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, sub in summary_df.groupby("Model"):
        stats = evaluate_candidate_status(sub, TILE_ORDER)
        stats["Practical GO"] = practical_gate(stats)
        rows.append(
            {
                "Model": model_id,
                "Family": stats["family"],
                "Mean Skill_MSE": stats["mean_skill_mse"],
                "Gain vs M1": stats["gain_vs_m1"],
                "Better tiles": stats["better_tiles_vs_m1"],
                "Stress ratio": stats["stress_ratio"],
                "Mean Skill std": stats["mean_skill_std"],
                "Mean train seconds": stats["mean_train_seconds"],
                "Practical GO": stats["Practical GO"],
            }
        )
    return pd.DataFrame(rows).sort_values(["Practical GO", "Gain vs M1", "Stress ratio"], ascending=[False, False, True]).reset_index(drop=True)


def choose_winner(full_eval: pd.DataFrame) -> str | None:
    eligible = full_eval[full_eval["Practical GO"]].copy()
    if eligible.empty:
        return None
    families = eligible["Family"].tolist()
    if "M1++" in families and "COEA" in families:
        m1pp = eligible[eligible["Family"] == "M1++"].iloc[0]
        coea = eligible[eligible["Family"] == "COEA"].iloc[0]
        if coea["Mean Skill_MSE"] > m1pp["Mean Skill_MSE"] + 0.005 and coea["Stress ratio"] <= m1pp["Stress ratio"] + 0.01:
            return str(coea["Model"])
        return str(m1pp["Model"])
    return str(eligible.iloc[0]["Model"])


def winner_ablation(
    bundles: dict[str, TileBundle],
    winner_family: str,
    winner_cfg: Any,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if winner_family == "M1++":
        candidates = [
            M1PPConfig("M1PP_ablate_M1_like", False, 0.0, "flat1", (1.0, 1.0, 1.0, 1.0), False),
            M1PPConfig("M1PP_ablate_bounded", True, 0.0, winner_cfg.beta_name, winner_cfg.beta_values, False),
            M1PPConfig("M1PP_ablate_gate", True, 0.0, winner_cfg.beta_name, winner_cfg.beta_values, True),
            M1PPConfig("M1PP_ablate_full", winner_cfg.bounded_residual, winner_cfg.lambda_oc, winner_cfg.beta_name, winner_cfg.beta_values, winner_cfg.use_gate_prior),
        ]
        for cfg in candidates:
            raw, _runtime = run_candidate_fullsix(bundles, cfg.model_id, "M1++", cfg, FINAL_SEEDS)
            rows.append(raw)
    elif winner_family == "COEA":
        candidates = [
            COEAConfig("COEA_ablate_equal", "equal", 0.0, False, 0.0, winner_cfg.granularity),
            COEAConfig("COEA_ablate_ewa", "ewa", winner_cfg.eta, winner_cfg.warm_start, 0.0, winner_cfg.granularity),
            COEAConfig("COEA_ablate_boa", "boa", winner_cfg.eta, winner_cfg.warm_start, 0.0, winner_cfg.granularity),
            COEAConfig("COEA_ablate_full", winner_cfg.update_mode, winner_cfg.eta, winner_cfg.warm_start, winner_cfg.m1_floor, winner_cfg.granularity),
        ]
        for cfg in candidates:
            raw, _runtime = run_candidate_fullsix(bundles, cfg.model_id, "COEA", cfg, FINAL_SEEDS)
            rows.append(raw)
    elif winner_family == "CLBE":
        for cfg in clbe_configs():
            raw, _runtime = run_candidate_fullsix(bundles, cfg.model_id, "CLBE", cfg, FINAL_SEEDS)
            rows.append(raw)
    else:
        return pd.DataFrame()
    raw_all = pd.concat(rows, ignore_index=True)
    summary = summarize_seeded_rows(raw_all)
    summary.to_csv(WINNER_ABLATION_PATH, index=False, encoding="utf-8-sig")
    plot_heatmap(summary[summary["Model"] == summary["Model"].iloc[-1]], WINNER_PANEL_PATH, "Skill_MSE mean", "Winner ablation panel", TILE_ORDER)
    return summary


def write_week1_decision(week1_df: pd.DataFrame, top_df: pd.DataFrame) -> Path:
    lines = [
        "# V7 Week-1 Go/No-Go",
        "",
        "数据声明：本轮 V7 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## Quick summary",
        "",
    ]
    for _, row in week1_df.iterrows():
        lines.append(
            f"- {row['Model']} ({row['Family']}): gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/3, stress ratio={row['Stress ratio']:.4f}, quick gate={bool(row['Quick gate pass'])}."
        )
    lines.extend(["", "## Decision", ""])
    if top_df.empty:
        lines.append("- 没有候选达到 quick-set 推进门槛，按 V7 文档硬条件立即停止后续模型探索。")
    else:
        joined = ", ".join(str(v) for v in top_df["Model"].tolist())
        lines.append(f"- 进入 full-six 的候选：{joined}。")
    return write_markdown(QUICK_DECISION_PATH, lines)


def write_stress_report(summary_df: pd.DataFrame) -> Path:
    lines = [
        "# Stress Tile Report",
        "",
    ]
    for model_id, sub in summary_df.groupby("Model"):
        stress = sub[sub["Tile"].isin(["E48N24", "E58N17"])]
        if stress.empty:
            continue
        lines.append(f"## {model_id}")
        lines.append("")
        for _, row in stress.iterrows():
            ratio = row["RMSE [mm] mean"] / row["Persistence RMSE [mm] mean"]
            lines.append(f"- {row['Tile']} {row['Horizon']}: RMSE/persistence = {ratio:.4f}, Skill_MSE = {row['Skill_MSE mean']:.4f}.")
        lines.append("")
    return write_markdown(STRESS_REPORT_PATH, lines)


def write_final_report(
    week1_df: pd.DataFrame,
    top_df: pd.DataFrame,
    full_eval: pd.DataFrame | None,
    winner: str | None,
    ablation_df: pd.DataFrame | None,
) -> tuple[Path, Path, Path]:
    lines = [
        "# ACT-Former V7 完整探索报告",
        "",
        "数据声明：本轮所有 V7 实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 候选族",
        "",
        "- COEA：因果在线专家聚合，专家集合固定为 persistence / linear-seasonal / M1 / C3_opt。",
        "- M1++：围绕 M1 的 horizon-aware conservative residual calibration。",
        "- CRCF：对 best COEA quick 进行基于 validation 的风险阈值回退。",
        "- CLBE：纯因果运动学 baseline（trend-only / trend+season / trend+season+clip）。",
        "",
        "## Week-1 quick search",
        "",
    ]
    for _, row in week1_df.iterrows():
        lines.append(
            f"- {row['Model']} ({row['Family']}): Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/3, stress ratio={row['Stress ratio']:.4f}, quick gate={bool(row['Quick gate pass'])}."
        )
    lines.extend(["", "## Week-1 decision", ""])
    if top_df.empty:
        lines.append("- 没有候选达到 quick-set 推进门槛，因此按文档硬停止条件直接冻结 M1，不进入 full-six。")
    else:
        lines.append(f"- 进入 full-six 的候选：{', '.join(top_df['Model'].tolist())}。")
    if full_eval is not None and not full_eval.empty:
        lines.extend(["", "## Full-six 3-seed", ""])
        for _, row in full_eval.iterrows():
            lines.append(
                f"- {row['Model']} ({row['Family']}): Mean Skill_MSE={row['Mean Skill_MSE']:.4f}, gain vs M1={row['Gain vs M1']:.4f}, better tiles={int(row['Better tiles'])}/6, stress ratio={row['Stress ratio']:.4f}, std={row['Mean Skill std']:.4f}, Practical GO={bool(row['Practical GO'])}."
            )
    lines.extend(["", "## Final decision", ""])
    if winner is None:
        lines.append("- 没有候选达到 V7 Practical GO，主模型保持 M1，V7 结果转入 supplementary / architecture search appendix。")
    else:
        lines.append(f"- 胜出者为 `{winner}`。")
    if ablation_df is not None and not ablation_df.empty:
        lines.extend(["", "## Winner ablation", "", f"- 结果文件：`{WINNER_ABLATION_PATH.as_posix()}`。"])
    report = write_markdown(REPORT_PATH, lines)

    decision_lines = [
        "# V7 Final Decision",
        "",
        f"- winner = `{winner}`" if winner is not None else "- winner = `None`",
        "- 结论：如果 V7 候选不能在 full-six × four-horizon × 3-seed 下稳定超过 M1，就停止所有模型探索并回到 M1。",
        "",
    ]
    final_decision = write_markdown(FINAL_DECISION_PATH, decision_lines)

    notes_lines = [
        "# Paper Integration Notes",
        "",
        "- 若 V7 失败：将 COEA / M1++ / CRCF / CLBE 写为最后一轮有限 architecture search，强调复杂 selector 与更重替代结构未能稳定超过 M1。",
        "- 若 V7 成功且 winner=M1++：主线改写为 conservative residual calibration around a strong persistence-centered ACT backbone。",
        "- 若 V7 成功且 winner=COEA：主线改写为 deployment-layer causal online aggregation over neural and kinematic experts。",
        "",
    ]
    paper_notes = write_markdown(PAPER_NOTES_PATH, notes_lines)
    return report, final_decision, paper_notes


def run_act_v7_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(ACT_V7_ROOT)
    ensure_dir(ACT_V7_CONFIG_ROOT)
    ensure_dir(SEED_CACHE_ROOT)
    bundles = preprocess_all_tiles()

    quick_baseline_bank()
    quick_seeds = [42, 43]

    coea_raw, coea_grid, best_coea_cfg, best_coea_summary = run_coea_quick(bundles, quick_seeds)
    m1pp_raw, m1pp_grid, best_m1pp_cfg, best_m1pp_summary = run_m1pp_quick(bundles, quick_seeds)
    clbe_raw, clbe_grid, best_clbe_cfg, best_clbe_summary = run_clbe_quick(bundles, quick_seeds)
    crcf_raw, crcf_summary = run_crcf_quick(bundles, best_coea_cfg, quick_seeds)

    family_stats = [
        evaluate_candidate_status(best_coea_summary, SEARCH_TILES),
        evaluate_candidate_status(best_m1pp_summary, SEARCH_TILES),
        evaluate_candidate_status(best_clbe_summary, SEARCH_TILES),
        evaluate_candidate_status(crcf_summary, SEARCH_TILES),
    ]
    week1_df = week1_summary_table(family_stats)
    top_df = pick_top_candidates(week1_df)
    write_week1_decision(week1_df, top_df)

    full_raw_frames: list[pd.DataFrame] = []
    runtime_frames: list[pd.DataFrame] = []
    full_eval: pd.DataFrame | None = None
    winner: str | None = None
    ablation_df: pd.DataFrame | None = None

    if not top_df.empty:
        for _, row in top_df.iterrows():
            family, cfg = config_from_model_id(str(row["Model"]), best_coea_cfg, best_m1pp_cfg, best_clbe_cfg)
            raw, runtime = run_candidate_fullsix(bundles, str(row["Model"]), family, cfg, FINAL_SEEDS)
            full_raw_frames.append(raw)
            runtime_frames.append(runtime)
        full_raw = pd.concat(full_raw_frames, ignore_index=True)
        full_summary = summarize_seeded_rows(full_raw)
        full_summary.to_csv(FULL6_SUMMARY_PATH, index=False, encoding="utf-8-sig")
        full_raw.to_csv(FULL6_RAW_PATH, index=False, encoding="utf-8-sig")
        runtime_df = pd.concat(runtime_frames, ignore_index=True)
        runtime_df.to_csv(RUNTIME_LOG_PATH, index=False, encoding="utf-8-sig")
        plot_heatmap(full_summary[full_summary["Model"] == top_df.iloc[0]["Model"]], TILE_HORIZON_HEATMAP, "Skill_MSE mean", "Top candidate full-six heatmap", TILE_ORDER)
        write_stress_report(full_summary)
        full_eval = evaluate_fullsix_candidates(full_summary)
        winner = choose_winner(full_eval)
        if winner is not None:
            family, cfg = config_from_model_id(winner, best_coea_cfg, best_m1pp_cfg, best_clbe_cfg)
            ablation_df = winner_ablation(bundles, family, cfg)
    else:
        pd.DataFrame().to_csv(FULL6_RAW_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(FULL6_SUMMARY_PATH, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(RUNTIME_LOG_PATH, index=False, encoding="utf-8-sig")

    report, final_decision, paper_notes = write_final_report(week1_df, top_df, full_eval, winner, ablation_df)

    summary = {
        "report": str(report.resolve()),
        "final_decision": str(final_decision.resolve()),
        "paper_notes": str(paper_notes.resolve()),
        "week1_summary": str(QUICK_SUMMARY_PATH.resolve()),
        "top_candidates": top_df["Model"].tolist(),
        "winner": winner,
        "fullsix_summary": str(FULL6_SUMMARY_PATH.resolve()),
        "coea_best_model": best_coea_cfg.model_id,
        "m1pp_best_model": best_m1pp_cfg.model_id,
        "clbe_best_model": best_clbe_cfg.model_id,
    }
    write_json(SUMMARY_PATH, summary)
    return summary
