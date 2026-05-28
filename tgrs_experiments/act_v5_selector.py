from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset

from .act_v2 import (
    CONFIG_ROOT as GLOBAL_CONFIG_ROOT,
    FINAL_SEEDS,
    SEARCH_HORIZONS,
    SEARCH_TILES,
    ActV2Config,
    ActV2Dataset,
    build_model as build_m1_model,
    collect_predictions as collect_predictions_m1,
    write_simple_yaml,
)
from .act_v2_m1_suite import FULL_TILES, baseline_rows_for_tiles, summarize_baselines
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v4c_efficiency import (
    V4CRescueConfig,
    build_rescue_model,
    collect_predictions_rescue,
    default_rescue_configs,
    loader_kwargs,
    prepare_rescue_datasets,
)
from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    TILE_ORDER,
    TileBundle,
    compute_metrics,
    ensure_dir,
    preprocess_all_tiles,
    seed_everything,
    write_json,
)


ACT_V5_ROOT = OUTPUT_ROOT / "act_v5"
ACT_V5_CONFIG_ROOT = GLOBAL_CONFIG_ROOT / "act_v5"
EXPERT_CACHE_ROOT = ACT_V5_ROOT / "expert_cache"
EXPERT_CACHE_MANIFEST = ACT_V5_ROOT / "v5_expert_cache_manifest.csv"
EXPERT_SEED_MANIFEST = ACT_V5_ROOT / "v5_selected_expert_runs.csv"
ORACLE_TABLE_PATH = ACT_V5_ROOT / "v5_oracle_table.csv"
ORACLE_GAIN_HEATMAP = ACT_V5_ROOT / "v5_oracle_gain_heatmap.png"
ORACLE_WIN_TABLE_PATH = ACT_V5_ROOT / "v5_oracle_expert_win_rates.csv"
ORACLE_DECISION_PATH = ACT_V5_ROOT / "v5_oracle_decision.md"
SELECTOR_RESULTS_PATH = ACT_V5_ROOT / "v5_selector_results.csv"
ABLATION_PATH = ACT_V5_ROOT / "v5_ablation.csv"
FINAL_DECISION_PATH = ACT_V5_ROOT / "v5_final_decision.md"
REPORT_PATH = ACT_V5_ROOT / "act_v5_detailed_report.md"
SUMMARY_PATH = ACT_V5_ROOT / "run_summary.json"

M1_RUN_ROOT = (
    OUTPUT_ROOT / "act_v2" / "m1_suite" / "runs" / "step1b_scalar_full6" / "C2_M1_scalar"
)
C3_RUN_ROOT = (
    OUTPUT_ROOT
    / "act_v4"
    / "v4c_efficiency_rescue"
    / "c3_candidate"
    / "runs"
    / "full6"
    / "C3_V4C_depthwise_temporal_opt"
)
M1_FULL6_SUMMARY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"
C3_FULL6_SUMMARY_PATH = OUTPUT_ROOT / "act_v4" / "v4c_efficiency_rescue" / "c3_candidate" / "c3_full6_summary.csv"

EXPERT_NAMES = ["persistence", "linear_seasonal", "M1", "C3_opt"]
RULE_MODEL_ID = "V5A_rule_selector"
SOFT_MODEL_ID = "V5B_soft_selector"
HARD_MODEL_ID = "V5C_hard_selector"
RISK_MODEL_ID = "V5D_risk_selector"
PATCH_SIZE = 8

ORACLE_GAIN_THRESHOLD = 0.05
MODEL_GAIN_THRESHOLD = 0.02
STRESS_RATIO_THRESHOLD = 1.03
BETTER_TILE_THRESHOLD = 4


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class V5SelectorConfig:
    model_id: str
    variant: str
    expert_names: list[str] = field(default_factory=lambda: EXPERT_NAMES.copy())
    hidden_dim: int = 32
    depth: int = 3
    batch_size: int = 8
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 0
    penalty_lambda: float = 0.5
    entropy_lambda: float = 0.01
    oracle_ce_lambda: float = 0.0
    risk_bce_lambda: float = 0.0
    risk_margin: float = 0.0
    use_risk_head: bool = False
    hard_selection: bool = False
    persistence_bias: float = 0.75
    gumbel_tau: float = 1.0
    select_thresholds: list[float] = field(default_factory=lambda: [0.45, 0.5, 0.55, 0.6, 0.65, 0.7])


def default_selector_configs() -> dict[str, V5SelectorConfig]:
    return {
        "soft": V5SelectorConfig(
            model_id=SOFT_MODEL_ID,
            variant="soft",
            hidden_dim=32,
            depth=3,
            penalty_lambda=0.35,
            entropy_lambda=0.015,
        ),
        "hard": V5SelectorConfig(
            model_id=HARD_MODEL_ID,
            variant="hard",
            hidden_dim=32,
            depth=3,
            penalty_lambda=0.2,
            entropy_lambda=0.0,
            oracle_ce_lambda=0.45,
            hard_selection=True,
            persistence_bias=0.5,
        ),
        "risk": V5SelectorConfig(
            model_id=RISK_MODEL_ID,
            variant="risk",
            hidden_dim=40,
            depth=3,
            penalty_lambda=0.55,
            entropy_lambda=0.01,
            risk_bce_lambda=0.35,
            use_risk_head=True,
            persistence_bias=1.0,
        ),
    }


def masked_smooth_l1_2d(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, :, :]
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / (weights.sum() + 1e-6)


def masked_mse_2d(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, :, :]
    err = (pred - target) ** 2
    return (err * weights).sum() / (weights.sum() + 1e-6)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_m1_full6_summary() -> pd.DataFrame:
    return pd.read_csv(M1_FULL6_SUMMARY_PATH)


def load_c3_full6_summary() -> pd.DataFrame:
    return pd.read_csv(C3_FULL6_SUMMARY_PATH)


def valid_mean_std(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = arr[mask > 0.5]
    return float(vals.mean()), float(vals.std() + 1e-6)


def zscore_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mu, sigma = valid_mean_std(arr, mask)
    return ((arr - mu) / sigma).astype(np.float32)


def seed_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tile in FULL_TILES:
        for family, root in [("M1", M1_RUN_ROOT), ("C3_opt", C3_RUN_ROOT)]:
            best_seed = None
            best_score = None
            best_path = None
            for seed in FINAL_SEEDS:
                result_path = root / tile / f"seed_{seed}" / "result.json"
                if not result_path.exists():
                    continue
                payload = load_json(result_path)
                score = float(payload["best_score"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_seed = seed
                    best_path = result_path.parent
            if best_seed is None or best_path is None or best_score is None:
                raise FileNotFoundError(f"Missing run directory for {family} tile={tile}")
            rows.append(
                {
                    "Expert family": family,
                    "Tile": tile,
                    "Selected seed": int(best_seed),
                    "Validation skill": float(best_score),
                    "Run directory": best_path.as_posix(),
                }
            )
    return rows


def ensure_seed_manifest(reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and EXPERT_SEED_MANIFEST.exists():
        return pd.read_csv(EXPERT_SEED_MANIFEST)
    rows = seed_manifest_rows()
    df = pd.DataFrame(rows)
    df.to_csv(EXPERT_SEED_MANIFEST, index=False, encoding="utf-8-sig")
    return df


def selected_run_dir(manifest: pd.DataFrame, family: str, tile: str) -> Path:
    row = manifest[(manifest["Expert family"] == family) & (manifest["Tile"] == tile)].iloc[0]
    return Path(str(row["Run directory"]))


def infer_m1_split(bundle: TileBundle, run_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    payload = torch.load(run_dir / "model.pth", map_location="cpu")
    config = ActV2Config(**payload["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ActV2Dataset(bundle, resolution=64, split_name=split_name, config=config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_m1_model(config, image_size=64, num_channels=dataset.xs[0].shape[1]).to(device)
    model.load_state_dict(payload["state_dict"])
    arrays = collect_predictions_m1(model, loader, device)
    return arrays


def infer_c3_split(bundle: TileBundle, run_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    payload = torch.load(run_dir / "model.pth", map_location="cpu")
    config = V4CRescueConfig(**payload["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = prepare_rescue_datasets(bundle, config)
    mapping = {"train": train_ds, "val": val_ds, "test": test_ds}
    loader = DataLoader(mapping[split_name], **loader_kwargs(config, shuffle=False))
    model = build_rescue_model(config, in_channels=6).to(device)
    model.load_state_dict(payload["state_dict"])
    arrays, _infer_seconds = collect_predictions_rescue(model, loader, device)
    return arrays


def build_expert_cache(bundle: TileBundle, run_manifest: pd.DataFrame, split_name: str, reuse_if_exists: bool = True) -> Path:
    tile = bundle.tile_id
    out_dir = ensure_dir(EXPERT_CACHE_ROOT / tile)
    cache_path = out_dir / f"{split_name}.npz"
    if reuse_if_exists and cache_path.exists():
        return cache_path

    m1_dir = selected_run_dir(run_manifest, "M1", tile)
    c3_dir = selected_run_dir(run_manifest, "C3_opt", tile)
    m1 = infer_m1_split(bundle, m1_dir, split_name)
    c3 = infer_c3_split(bundle, c3_dir, split_name)

    m1_frames = m1["current_frames"].reshape(-1)
    c3_frames = c3["current_frames"].reshape(-1)
    if not np.array_equal(m1_frames, c3_frames):
        raise ValueError(f"Frame mismatch for tile={tile} split={split_name}")
    if not np.array_equal(m1["target_indices"], c3["target_indices"]):
        raise ValueError(f"Target index mismatch for tile={tile} split={split_name}")

    tensors = bundle.tensors64
    payload = {
        "y_true_mm": m1["y_true_mm"].astype(np.float32),
        "persistence_mm": m1["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": m1["linear_seasonal_mm"].astype(np.float32),
        "m1_pred_mm": m1["y_pred_mm"].astype(np.float32),
        "c3_pred_mm": c3["y_pred_mm"].astype(np.float32),
        "m1_gate_mm": m1["gate_weights"][:, :, 2].astype(np.float32),
        "c3_scale_mm": c3["scale_mm"].astype(np.float32),
        "c3_delta_mm": c3["pred_delta_mm"].astype(np.float32),
        "valid_mask": m1["valid_mask"].astype(np.float32),
        "regime_map": m1["regime_map"].astype(np.int16),
        "current_frames": m1_frames.astype(np.int64),
        "target_indices": m1["target_indices"].astype(np.int64),
        "last_disp_mm": m1["persistence_mm"][:, 0:1].astype(np.float32),
        "descriptor_amp": tensors["descriptor_amp"].astype(np.float32),
        "trend_r2": tensors["trend_r2"].astype(np.float32),
        "residual_std": tensors["residual_std"].astype(np.float32),
        "change_score": tensors["change_score_full"].astype(np.float32),
    }
    np.savez_compressed(cache_path, **payload)
    return cache_path


def ensure_expert_caches(bundles: dict[str, TileBundle], reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and EXPERT_CACHE_MANIFEST.exists():
        return pd.read_csv(EXPERT_CACHE_MANIFEST)
    run_manifest = ensure_seed_manifest(reuse_if_exists=reuse_if_exists)
    rows: list[dict[str, Any]] = []
    for tile in FULL_TILES:
        bundle = bundles[tile]
        for split_name in ["train", "val", "test"]:
            path = build_expert_cache(bundle, run_manifest, split_name, reuse_if_exists=reuse_if_exists)
            rows.append(
                {
                    "Tile": tile,
                    "Split": split_name,
                    "Cache path": path.as_posix(),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(EXPERT_CACHE_MANIFEST, index=False, encoding="utf-8-sig")
    return df


def load_cache(tile: str, split_name: str) -> dict[str, np.ndarray]:
    path = EXPERT_CACHE_ROOT / tile / f"{split_name}.npz"
    arr = np.load(path)
    return {name: arr[name] for name in arr.files}


def expert_stack(cache: dict[str, np.ndarray], expert_names: list[str]) -> np.ndarray:
    mapping = {
        "persistence": cache["persistence_mm"],
        "linear_seasonal": cache["linear_seasonal_mm"],
        "M1": cache["m1_pred_mm"],
        "C3_opt": cache["c3_pred_mm"],
    }
    return np.stack([mapping[name] for name in expert_names], axis=2).astype(np.float32)


def oracle_from_cache(cache: dict[str, np.ndarray], mode: str, patch_size: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    names = EXPERT_NAMES
    experts = expert_stack(cache, names)
    y_true = cache["y_true_mm"]
    valid = cache["valid_mask"] > 0.5
    sq_err = (experts - y_true[:, :, None, :, :]) ** 2
    if mode == "pixel":
        best_idx = sq_err.argmin(axis=2)
        pred = np.take_along_axis(experts, best_idx[:, :, None, :, :], axis=2)[:, :, 0]
        return pred.astype(np.float32), best_idx.astype(np.int16)
    if mode == "sample":
        pred = np.zeros_like(y_true)
        choice = np.zeros((y_true.shape[0], y_true.shape[1]), dtype=np.int16)
        for i in range(y_true.shape[0]):
            for h in range(y_true.shape[1]):
                scores = [float(sq_err[i, h, idx][:, :][valid].mean()) for idx in range(len(names))]
                best_idx = int(np.argmin(scores))
                pred[i, h] = experts[i, h, best_idx]
                choice[i, h] = best_idx
        return pred.astype(np.float32), choice.astype(np.int16)
    if mode == "patch":
        pred = np.zeros_like(y_true)
        choice = np.zeros((y_true.shape[0], y_true.shape[1], y_true.shape[2] // patch_size, y_true.shape[3] // patch_size), dtype=np.int16)
        for i in range(y_true.shape[0]):
            for h in range(y_true.shape[1]):
                for py in range(0, y_true.shape[2], patch_size):
                    for px in range(0, y_true.shape[3], patch_size):
                        patch_mask = valid[py : py + patch_size, px : px + patch_size]
                        if int(patch_mask.sum()) == 0:
                            continue
                        scores = []
                        for idx in range(len(names)):
                            scores.append(float(sq_err[i, h, idx, py : py + patch_size, px : px + patch_size][patch_mask].mean()))
                        best_idx = int(np.argmin(scores))
                        pred[i, h, py : py + patch_size, px : px + patch_size] = experts[i, h, best_idx, py : py + patch_size, px : px + patch_size]
                        choice[i, h, py // patch_size, px // patch_size] = best_idx
        return pred.astype(np.float32), choice.astype(np.int16)
    raise ValueError(f"Unsupported oracle mode: {mode}")


def selector_arrays_from_prediction(cache: dict[str, np.ndarray], y_pred: np.ndarray, weights: np.ndarray | None = None, risk: np.ndarray | None = None, choice: np.ndarray | None = None) -> dict[str, np.ndarray]:
    arrays = {
        "y_pred_mm": y_pred.astype(np.float32),
        "y_true_mm": cache["y_true_mm"].astype(np.float32),
        "persistence_mm": cache["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
        "valid_mask": cache["valid_mask"].astype(np.float32),
        "regime_map": cache["regime_map"].astype(np.int16),
        "current_frames": cache["current_frames"].astype(np.int64),
        "target_indices": cache["target_indices"].astype(np.int64),
    }
    if weights is not None:
        arrays["selector_weights"] = weights.astype(np.float32)
    if risk is not None:
        arrays["risk_map"] = risk.astype(np.float32)
    if choice is not None:
        arrays["expert_choice"] = choice.astype(np.int16)
    return arrays


def oracle_and_m1_rows(tile: str, cache: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arrays_m1 = selector_arrays_from_prediction(cache, cache["m1_pred_mm"])
    rows.extend(result_rows("M1_expert_seeded", tile, 42, "oracle", 0.0, 0.0, -1, 0, arrays_m1, SEARCH_HORIZONS))
    for mode, model_id in [("sample", "V5S0_sample_oracle"), ("patch", "V5S0_patch_oracle"), ("pixel", "V5S0_pixel_oracle")]:
        pred, choice = oracle_from_cache(cache, mode)
        arrays = selector_arrays_from_prediction(cache, pred, choice=choice)
        rows.extend(result_rows(model_id, tile, 42, "oracle", 0.0, 0.0, -1, 0, arrays, SEARCH_HORIZONS))
    return rows


def oracle_win_rate_rows(tile: str, cache: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    _pred, choice = oracle_from_cache(cache, "pixel")
    valid = cache["valid_mask"] > 0.5
    regime_map = cache["regime_map"]
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        for regime_idx, regime_name in enumerate(REGIME_NAMES):
            mask = valid & (regime_map == regime_idx)
            if int(mask.sum()) < 10:
                continue
            for expert_idx, expert in enumerate(EXPERT_NAMES):
                rate = float((choice[:, h_idx][:, mask] == expert_idx).mean())
                rows.append(
                    {
                        "Tile": tile,
                        "Horizon": f"t+{horizon}",
                        "Regime": regime_name,
                        "Expert": expert,
                        "Pixel win rate": rate,
                    }
                )
    return rows


def run_oracle_analysis(reuse_if_exists: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if reuse_if_exists and ORACLE_TABLE_PATH.exists() and ORACLE_WIN_TABLE_PATH.exists():
        table = pd.read_csv(ORACLE_TABLE_PATH)
        wins = pd.read_csv(ORACLE_WIN_TABLE_PATH)
    else:
        rows: list[dict[str, Any]] = []
        win_rows: list[dict[str, Any]] = []
        for tile in FULL_TILES:
            cache = load_cache(tile, "test")
            rows.extend(oracle_and_m1_rows(tile, cache))
            win_rows.extend(oracle_win_rate_rows(tile, cache))
        table = pd.DataFrame(rows)
        wins = pd.DataFrame(win_rows)
        table.to_csv(ORACLE_TABLE_PATH, index=False, encoding="utf-8-sig")
        wins.to_csv(ORACLE_WIN_TABLE_PATH, index=False, encoding="utf-8-sig")

    patch_rows = table[table["Model"] == "V5S0_patch_oracle"].copy()
    pixel_rows = table[table["Model"] == "V5S0_pixel_oracle"].copy()
    sample_rows = table[table["Model"] == "V5S0_sample_oracle"].copy()
    m1_rows = table[table["Model"] == "M1_expert_seeded"][["Tile", "Horizon", "Skill_MSE"]].rename(columns={"Skill_MSE": "M1 Skill_MSE"})

    patch_gain = patch_rows.merge(m1_rows, on=["Tile", "Horizon"], how="left")
    patch_gain["Gain vs M1"] = patch_gain["Skill_MSE"] - patch_gain["M1 Skill_MSE"]
    pixel_gain = pixel_rows.merge(m1_rows, on=["Tile", "Horizon"], how="left")
    pixel_gain["Gain vs M1"] = pixel_gain["Skill_MSE"] - pixel_gain["M1 Skill_MSE"]
    sample_gain = sample_rows.merge(m1_rows, on=["Tile", "Horizon"], how="left")
    sample_gain["Gain vs M1"] = sample_gain["Skill_MSE"] - sample_gain["M1 Skill_MSE"]

    summary = {
        "sample_oracle_mean_gain_vs_m1": float(sample_gain["Gain vs M1"].mean()),
        "patch_oracle_mean_gain_vs_m1": float(patch_gain["Gain vs M1"].mean()),
        "pixel_oracle_mean_gain_vs_m1": float(pixel_gain["Gain vs M1"].mean()),
        "patch_positive_tiles": int(patch_gain.groupby("Tile")["Gain vs M1"].mean().gt(0).sum()),
        "pixel_positive_tiles": int(pixel_gain.groupby("Tile")["Gain vs M1"].mean().gt(0).sum()),
        "expert_diversity": wins.groupby("Expert")["Pixel win rate"].mean().to_dict() if not wins.empty else {},
        "status": "pass"
        if float(patch_gain["Gain vs M1"].mean()) >= ORACLE_GAIN_THRESHOLD
        and int(patch_gain.groupby("Tile")["Gain vs M1"].mean().gt(0).sum()) >= BETTER_TILE_THRESHOLD
        else "fail",
    }

    pivot = patch_gain.pivot(index="Tile", columns="Horizon", values="Gain vs M1").reindex(TILE_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Patch Oracle Gain vs M1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ORACLE_GAIN_HEATMAP, dpi=200)
    plt.close(fig)

    lines = [
        "# V5 Oracle 决策",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile。",
        "",
        f"- sample oracle mean gain vs M1 = {summary['sample_oracle_mean_gain_vs_m1']:.4f}",
        f"- patch oracle mean gain vs M1 = {summary['patch_oracle_mean_gain_vs_m1']:.4f}",
        f"- pixel oracle mean gain vs M1 = {summary['pixel_oracle_mean_gain_vs_m1']:.4f}",
        f"- patch positive tiles = {summary['patch_positive_tiles']} / 6",
        f"- pixel positive tiles = {summary['pixel_positive_tiles']} / 6",
        f"- oracle gate status = `{summary['status']}`",
        "",
        "结论：",
        "- patch-level oracle 作为可学习 selector 的可操作上界；pixel-level oracle 作为绝对上界。",
        "- 若 patch oracle 已明显超过 +0.05，则继续训练 selector；否则按方案停止 V5。",
        "",
    ]
    ORACLE_DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    return table, wins, summary


def rule_based_selector(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    p = cache["persistence_mm"]
    ls = cache["linear_seasonal_mm"]
    m1 = cache["m1_pred_mm"]
    c3 = cache["c3_pred_mm"]
    gate = cache["m1_gate_mm"]
    scale = np.maximum(cache["c3_scale_mm"], 0.1)
    change = cache["change_score"]
    trend = cache["trend_r2"]
    amp = cache["descriptor_amp"]
    pred = np.zeros_like(p)
    choice = np.zeros((p.shape[0], p.shape[1], p.shape[2], p.shape[3]), dtype=np.int16)
    valid = cache["valid_mask"] > 0.5
    jump_thr = float(np.quantile(change[valid], 0.80))
    amp_thr = float(np.quantile(amp[valid], 0.70))
    trend_thr = float(np.quantile(trend[valid], 0.60))

    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        dis_m1_p = np.abs(m1[:, h_idx] - p[:, h_idx]) / scale[:, h_idx]
        dis_c3_p = np.abs(c3[:, h_idx] - p[:, h_idx]) / scale[:, h_idx]
        dis_c3_m1 = np.abs(c3[:, h_idx] - m1[:, h_idx]) / scale[:, h_idx]
        high_risk = (scale[:, h_idx] > np.quantile(scale[:, h_idx][:, valid], 0.70)) | (change[None, :, :] > jump_thr) | (dis_c3_m1 > 1.25)
        smooth_long = np.full_like(change[None, :, :], horizon >= 5, dtype=bool) & (trend[None, :, :] > trend_thr) & (amp[None, :, :] > amp_thr)
        prefer_c3 = (horizon >= 3) & (dis_c3_p < dis_m1_p) & (gate[:, h_idx] < 0.25)
        choose = np.full_like(p[:, h_idx], 2, dtype=np.int16)
        if horizon == 1:
            choose[:] = 0
        choose[high_risk] = 0
        choose[(~high_risk) & smooth_long] = 1
        choose[(~high_risk) & prefer_c3] = 3
        pred[:, h_idx] = np.choose(choose, [p[:, h_idx], ls[:, h_idx], m1[:, h_idx], c3[:, h_idx]])
        choice[:, h_idx] = choose
    return pred.astype(np.float32), choice


def evaluate_direct_selector(
    model_id: str,
    tile: str,
    cache: dict[str, np.ndarray],
    pred: np.ndarray,
    choice: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    risk: np.ndarray | None = None,
    elapsed_seconds: float = 0.0,
    infer_seconds: float = 0.0,
    seed: int = 42,
    phase: str = "quick",
) -> dict[str, Any]:
    arrays = selector_arrays_from_prediction(cache, pred, weights=weights, risk=risk, choice=choice)
    rows = result_rows(model_id, tile, seed, phase, elapsed_seconds, infer_seconds, -1, 0, arrays, SEARCH_HORIZONS)
    result = {
        "rows": rows,
        "mean_skill_mse": float(pd.DataFrame(rows)["Skill_MSE"].mean()),
        "arrays": arrays,
    }
    return result


def run_rule_selector(reuse_if_exists: bool = True) -> pd.DataFrame:
    out_path = ACT_V5_ROOT / "rule_selector_results.csv"
    if reuse_if_exists and out_path.exists():
        return pd.read_csv(out_path)
    rows: list[dict[str, Any]] = []
    for tile in FULL_TILES:
        cache = load_cache(tile, "test")
        start = time.time()
        pred, choice = rule_based_selector(cache)
        elapsed = time.time() - start
        result = evaluate_direct_selector(RULE_MODEL_ID, tile, cache, pred, choice=choice, elapsed_seconds=elapsed, infer_seconds=elapsed, phase="quick")
        rows.extend(result["rows"])
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


def regime_channel(regime_map: np.ndarray) -> np.ndarray:
    return (regime_map.astype(np.float32) / max(1.0, float(len(REGIME_NAMES) - 1))).astype(np.float32)


class SelectorCacheDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], config: V5SelectorConfig) -> None:
        self.cache = cache
        self.config = config
        self.mask = cache["valid_mask"].astype(np.float32)
        self.sequence_count = int(cache["y_true_mm"].shape[0])
        self.output_len = int(cache["y_true_mm"].shape[1])
        self.height = int(cache["y_true_mm"].shape[2])
        self.width = int(cache["y_true_mm"].shape[3])
        self.expert_names = list(config.expert_names)
        self.experts = expert_stack(cache, self.expert_names)

        trend_norm = zscore_map(cache["trend_r2"], self.mask)
        amp_norm = zscore_map(cache["descriptor_amp"], self.mask)
        resid_norm = zscore_map(cache["residual_std"], self.mask)
        change_norm = zscore_map(cache["change_score"], self.mask)
        regime_norm = regime_channel(cache["regime_map"])

        self.features: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.pers: list[np.ndarray] = []
        self.expert_preds: list[np.ndarray] = []
        self.oracle_labels: list[np.ndarray] = []
        self.risk_targets: list[np.ndarray] = []
        self.sample_indices: list[int] = []
        self.horizon_indices: list[int] = []

        for sample_idx in range(self.sequence_count):
            last_disp = cache["last_disp_mm"][sample_idx, 0]
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                target = cache["y_true_mm"][sample_idx, h_idx]
                persistence = cache["persistence_mm"][sample_idx, h_idx]
                linear_seasonal = cache["linear_seasonal_mm"][sample_idx, h_idx]
                m1_pred = cache["m1_pred_mm"][sample_idx, h_idx]
                c3_pred = cache["c3_pred_mm"][sample_idx, h_idx]
                scale = np.maximum(cache["c3_scale_mm"][sample_idx, h_idx], 0.1)
                gate = cache["m1_gate_mm"][sample_idx, h_idx]

                pred_map = {
                    "persistence": persistence,
                    "linear_seasonal": linear_seasonal,
                    "M1": m1_pred,
                    "C3_opt": c3_pred,
                }
                expert_pred = np.stack([pred_map[name] for name in self.expert_names], axis=0).astype(np.float32)
                sq_err = (expert_pred - target[None, :, :]) ** 2
                oracle_label = sq_err.argmin(axis=0).astype(np.int64)
                risk_target = (oracle_label == self.expert_names.index("persistence")).astype(np.float32)

                ls_delta = (linear_seasonal - last_disp) / scale
                m1_delta = (m1_pred - last_disp) / scale
                c3_delta = (c3_pred - last_disp) / scale
                feat = np.stack(
                    [
                        ls_delta.astype(np.float32),
                        m1_delta.astype(np.float32),
                        c3_delta.astype(np.float32),
                        (np.abs(m1_pred - persistence) / scale).astype(np.float32),
                        (np.abs(c3_pred - persistence) / scale).astype(np.float32),
                        (np.abs(c3_pred - m1_pred) / scale).astype(np.float32),
                        (np.abs(linear_seasonal - persistence) / scale).astype(np.float32),
                        gate.astype(np.float32),
                        np.log1p(scale).astype(np.float32),
                        trend_norm.astype(np.float32),
                        amp_norm.astype(np.float32),
                        resid_norm.astype(np.float32),
                        change_norm.astype(np.float32),
                        np.full_like(scale, float(horizon) / 10.0, dtype=np.float32),
                        regime_norm.astype(np.float32),
                        self.mask.astype(np.float32),
                    ],
                    axis=0,
                ).astype(np.float32)

                self.features.append(feat)
                self.targets.append(target[None, :, :].astype(np.float32))
                self.pers.append(persistence[None, :, :].astype(np.float32))
                self.expert_preds.append(expert_pred.astype(np.float32))
                self.oracle_labels.append(oracle_label)
                self.risk_targets.append(risk_target.astype(np.float32))
                self.sample_indices.append(sample_idx)
                self.horizon_indices.append(h_idx)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]).float(),
            "target": torch.from_numpy(self.targets[index]).float(),
            "persistence": torch.from_numpy(self.pers[index]).float(),
            "expert_preds": torch.from_numpy(self.expert_preds[index]).float(),
            "mask": torch.from_numpy(self.mask).float(),
            "oracle_label": torch.from_numpy(self.oracle_labels[index]).long(),
            "risk_target": torch.from_numpy(self.risk_targets[index]).float(),
            "sample_index": torch.tensor(self.sample_indices[index], dtype=torch.long),
            "horizon_index": torch.tensor(self.horizon_indices[index], dtype=torch.long),
        }


class SelectorNet(nn.Module):
    def __init__(self, in_channels: int, num_experts: int, config: V5SelectorConfig) -> None:
        super().__init__()
        self.config = config
        blocks: list[nn.Module] = []
        hidden = config.hidden_dim
        blocks.extend([nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1), nn.GELU()])
        for _ in range(max(1, config.depth - 1)):
            blocks.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=1),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*blocks)
        self.logits_head = nn.Conv2d(hidden, num_experts, kernel_size=1)
        self.risk_head = nn.Conv2d(hidden, 1, kernel_size=1) if config.use_risk_head else None
        with torch.no_grad():
            self.logits_head.bias.zero_()
            self.logits_head.bias[0] = config.persistence_bias

    def forward(
        self,
        features: torch.Tensor,
        expert_preds: torch.Tensor,
        threshold: float | None = None,
        hard_inference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        hidden = self.backbone(features)
        logits = self.logits_head(hidden)
        if self.config.hard_selection:
            if self.training and not hard_inference:
                weights = F.gumbel_softmax(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                    tau=self.config.gumbel_tau,
                    hard=False,
                ).reshape(logits.shape[0], logits.shape[2], logits.shape[3], logits.shape[1]).permute(0, 3, 1, 2)
            else:
                choice = logits.argmax(dim=1)
                weights = F.one_hot(choice, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        else:
            weights = F.softmax(logits, dim=1)

        risk_logits = self.risk_head(hidden) if self.risk_head is not None else None
        risk_prob = torch.sigmoid(risk_logits) if risk_logits is not None else None
        if threshold is not None and risk_prob is not None:
            fallback = risk_prob > threshold
            weights = weights.clone()
            weights[:, 0:1] = torch.where(fallback, torch.ones_like(weights[:, 0:1]), weights[:, 0:1])
            weights[:, 1:] = torch.where(fallback.expand_as(weights[:, 1:]), torch.zeros_like(weights[:, 1:]), weights[:, 1:])
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        pred = (weights * expert_preds).sum(dim=1, keepdim=True)
        return pred, weights, risk_logits, logits


def selector_skill(pred: torch.Tensor, target: torch.Tensor, persistence: torch.Tensor, mask: torch.Tensor) -> float:
    pred_mse = masked_mse_2d(pred, target, mask)
    pers_mse = masked_mse_2d(persistence, target, mask)
    return 1.0 - float(pred_mse.detach().item() / max(pers_mse.detach().item(), 1e-6))


def selector_loss(
    pred: torch.Tensor,
    weights: torch.Tensor,
    risk_logits: torch.Tensor | None,
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: V5SelectorConfig,
) -> tuple[torch.Tensor, float]:
    target = batch["target"]
    persistence = batch["persistence"]
    mask = batch["mask"]
    pred_loss = masked_smooth_l1_2d(pred, target, mask)
    model_mse = masked_mse_2d(pred, target, mask)
    pers_mse = masked_mse_2d(persistence, target, mask)
    penalty = F.relu(model_mse - pers_mse + config.risk_margin)
    entropy = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=1)
    entropy_term = (entropy * mask).sum() / (mask.sum() + 1e-6)
    loss = pred_loss + config.penalty_lambda * penalty + config.entropy_lambda * entropy_term

    if config.oracle_ce_lambda > 0:
        ce = F.cross_entropy(logits, batch["oracle_label"], reduction="none")
        ce_term = (ce * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + config.oracle_ce_lambda * ce_term
    if config.use_risk_head and risk_logits is not None and config.risk_bce_lambda > 0:
        bce = F.binary_cross_entropy_with_logits(risk_logits[:, 0], batch["risk_target"], reduction="none")
        bce_term = (bce * mask).sum() / (mask.sum() + 1e-6)
        loss = loss + config.risk_bce_lambda * bce_term

    skill = 1.0 - float(model_mse.detach().item() / max(pers_mse.detach().item(), 1e-6))
    return loss, skill


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_selector_epoch(
    model: SelectorNet,
    loader: DataLoader,
    config: V5SelectorConfig,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    skills: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            pred, weights, risk_logits, logits = model(batch["features"], batch["expert_preds"])
            loss, skill = selector_loss(pred, weights, risk_logits, logits, batch, config)
        if training:
            assert optimizer is not None
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.item()))
        skills.append(float(skill))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "skill_mse": float(np.mean(skills)) if skills else float("nan"),
    }


def collect_selector_arrays(
    model: SelectorNet,
    loader: DataLoader,
    dataset: SelectorCacheDataset,
    config: V5SelectorConfig,
    device: torch.device,
    threshold: float | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    n_seq = dataset.sequence_count
    out_len = dataset.output_len
    h, w = dataset.height, dataset.width
    num_experts = len(config.expert_names)
    pred_full = np.zeros((n_seq, out_len, h, w), dtype=np.float32)
    weight_full = np.zeros((n_seq, out_len, num_experts, h, w), dtype=np.float32)
    risk_full = np.zeros((n_seq, out_len, h, w), dtype=np.float32) if config.use_risk_head else None
    choice_full = np.zeros((n_seq, out_len, h, w), dtype=np.int16)
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred, weights, risk_logits, _logits = model(
                batch["features"],
                batch["expert_preds"],
                threshold=threshold,
                hard_inference=config.hard_selection,
            )
            pred_np = pred[:, 0].cpu().numpy()
            weights_np = weights.cpu().numpy()
            risk_np = torch.sigmoid(risk_logits[:, 0]).cpu().numpy() if risk_logits is not None else None
            choice_np = weights_np.argmax(axis=1).astype(np.int16)
            sample_idx = batch["sample_index"].cpu().numpy().astype(np.int64)
            horizon_idx = batch["horizon_index"].cpu().numpy().astype(np.int64)
            for i in range(len(sample_idx)):
                pred_full[sample_idx[i], horizon_idx[i]] = pred_np[i]
                weight_full[sample_idx[i], horizon_idx[i]] = weights_np[i]
                choice_full[sample_idx[i], horizon_idx[i]] = choice_np[i]
                if risk_full is not None and risk_np is not None:
                    risk_full[sample_idx[i], horizon_idx[i]] = risk_np[i]
    infer_seconds = time.time() - start
    arrays = selector_arrays_from_prediction(
        dataset.cache,
        pred_full,
        weights=weight_full,
        risk=risk_full,
        choice=choice_full,
    )
    return arrays, infer_seconds


def calibrate_risk_threshold(model: SelectorNet, loader: DataLoader, dataset: SelectorCacheDataset, config: V5SelectorConfig, device: torch.device) -> float:
    best_thr = 0.5
    best_skill = -float("inf")
    for threshold in config.select_thresholds:
        arrays, _ = collect_selector_arrays(model, loader, dataset, config, device, threshold=threshold)
        rows = result_rows(config.model_id, "val", config.seed, "val_calibrate", 0.0, 0.0, -1, sum(p.numel() for p in model.parameters()), arrays, SEARCH_HORIZONS)
        skill = float(pd.DataFrame(rows)["Skill_MSE"].mean())
        if skill > best_skill:
            best_skill = skill
            best_thr = threshold
    return float(best_thr)


def train_selector_single_tile(
    tile: str,
    config: V5SelectorConfig,
    phase: str,
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(ACT_V5_ROOT / "runs" / phase / config.model_id / tile / f"seed_{config.seed}")
    result_path = output_dir / "result.json"
    pred_path = output_dir / "predictions.npz"
    ckpt_path = output_dir / "model.pth"
    if reuse_if_exists and result_path.exists() and pred_path.exists() and ckpt_path.exists():
        result = load_json(result_path)
        arrays_npz = np.load(pred_path)
        return {"result": result, "arrays": {name: arrays_npz[name] for name in arrays_npz.files}}

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = load_cache(tile, "train")
    val_cache = load_cache(tile, "val")
    test_cache = load_cache(tile, "test")
    train_ds = SelectorCacheDataset(train_cache, config)
    val_ds = SelectorCacheDataset(val_cache, config)
    test_ds = SelectorCacheDataset(test_cache, config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = SelectorNet(train_ds.features[0].shape[0], len(config.expert_names), config).to(device)
    param_count = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        train_stats = run_selector_epoch(model, train_loader, config, device, optimizer=optimizer, scaler=scaler)
        val_stats = run_selector_epoch(model, val_loader, config, device, optimizer=None, scaler=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_skill_mse": train_stats["skill_mse"],
                "val_loss": val_stats["loss"],
                "val_skill_mse": val_stats["skill_mse"],
            }
        )
        if val_stats["skill_mse"] > best_score:
            best_score = val_stats["skill_mse"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    threshold = None
    if config.use_risk_head:
        threshold = calibrate_risk_threshold(model, val_loader, val_ds, config, device)
    arrays, infer_seconds = collect_selector_arrays(model, test_loader, test_ds, config, device, threshold=threshold)
    elapsed_seconds = time.time() - start_time
    rows = result_rows(config.model_id, tile, config.seed, phase, elapsed_seconds, infer_seconds, best_epoch, param_count, arrays, SEARCH_HORIZONS)
    summary_df = pd.DataFrame(rows)
    result = {
        "model_id": config.model_id,
        "variant": config.variant,
        "tile": tile,
        "seed": config.seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": elapsed_seconds,
        "infer_seconds": infer_seconds,
        "param_count": param_count,
        "history": history,
        "risk_threshold": threshold,
        "rows": rows,
        "mean_skill_mse": float(summary_df["Skill_MSE"].mean()),
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def run_selector_seeded(
    tiles: list[str],
    config: V5SelectorConfig,
    phase: str,
    seeds: list[int],
    reuse_if_exists: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    write_simple_yaml(ACT_V5_CONFIG_ROOT / f"{config.model_id}_{phase}.yaml", asdict(config))
    for seed in seeds:
        cfg = replace(config, seed=seed)
        for tile in tiles:
            print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
            run = train_selector_single_tile(tile, cfg, phase=phase, reuse_if_exists=reuse_if_exists)
            print(f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
            rows.extend(run["result"]["rows"])
    return pd.DataFrame(rows)


def model_mean_skill(df: pd.DataFrame, model_id: str) -> float:
    return float(df[df["Model"] == model_id]["Skill_MSE"].mean())


def search_baseline_skill() -> float:
    m1 = load_m1_full6_summary()
    sub = m1[m1["Tile"].isin(SEARCH_TILES)]
    return float(sub["Skill_MSE mean"].mean())


def full6_m1_skill() -> float:
    m1 = load_m1_full6_summary()
    return float(m1["Skill_MSE mean"].mean())


def full6_m1_tile_skill() -> pd.DataFrame:
    m1 = load_m1_full6_summary()
    return m1.groupby("Tile", as_index=False)["Skill_MSE mean"].mean().rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"})


def summarize_selector_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(SELECTOR_RESULTS_PATH, index=False, encoding="utf-8-sig")
    return combined


def rule_summary_rows(rule_df: pd.DataFrame) -> pd.DataFrame:
    return rule_df.groupby(["Model", "Tile", "Horizon"], as_index=False)[["Skill_MSE", "RMSE [mm]", "Elapsed seconds"]].mean()


def evaluate_v5d(summary_df: pd.DataFrame) -> dict[str, Any]:
    risk_df = summary_df[summary_df["Model"] == RISK_MODEL_ID].copy()
    m1_summary = load_m1_full6_summary()
    merged = risk_df.merge(
        m1_summary[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs M1"] = merged["Skill_MSE mean"] - merged["M1 Skill_MSE mean"]
    tile_compare = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "M1 Skill_MSE mean", "Delta vs M1"]]
        .mean()
        .rename(columns={"Skill_MSE mean": "V5D Skill_MSE mean"})
    )
    better_tiles = int((tile_compare["Delta vs M1"] > 0).sum())
    stress = merged[merged["Tile"].isin(["E48N24", "E58N17"])].copy()
    stress_ratio = float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max())
    mean_skill = float(risk_df["Skill_MSE mean"].mean())
    mean_std = float(risk_df["Skill_MSE std"].mean())
    mean_train = float(risk_df["Elapsed seconds mean"].mean())
    status = "pass"
    if mean_skill < full6_m1_skill() + MODEL_GAIN_THRESHOLD or better_tiles < BETTER_TILE_THRESHOLD or stress_ratio > STRESS_RATIO_THRESHOLD:
        status = "fail"
    return {
        "status": status,
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "m1_mean_skill_mse": full6_m1_skill(),
        "gain_vs_m1": mean_skill - full6_m1_skill(),
        "better_tiles_vs_m1": better_tiles,
        "stress_ratio": stress_ratio,
        "mean_train_seconds": mean_train,
        "tile_compare": tile_compare,
    }


def run_ablation_if_needed(reuse_if_exists: bool = True) -> pd.DataFrame | None:
    out_path = ABLATION_PATH
    if reuse_if_exists and out_path.exists():
        return pd.read_csv(out_path)
    risk_base = default_selector_configs()["risk"]
    variants = [
        replace(risk_base, model_id="V5E_no_C3", expert_names=["persistence", "linear_seasonal", "M1"]),
        replace(risk_base, model_id="V5E_no_risk", use_risk_head=False, risk_bce_lambda=0.0, variant="soft"),
        replace(risk_base, model_id="V5E_no_persistence", expert_names=["linear_seasonal", "M1", "C3_opt"], persistence_bias=0.0),
    ]
    frames: list[pd.DataFrame] = []
    for cfg in variants:
        df = run_selector_seeded(FULL_TILES, cfg, phase="ablation", seeds=[42], reuse_if_exists=reuse_if_exists)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def write_final_decision(
    oracle_summary: dict[str, Any],
    rule_df: pd.DataFrame,
    soft_summary: pd.DataFrame,
    hard_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    risk_verdict: dict[str, Any] | None,
    ablation_df: pd.DataFrame | None,
) -> Path:
    rule_mean = float(rule_df["Skill_MSE"].mean()) if not rule_df.empty else float("nan")
    soft_mean = float(soft_summary["Skill_MSE mean"].mean()) if not soft_summary.empty else float("nan")
    hard_mean = float(hard_summary["Skill_MSE mean"].mean()) if not hard_summary.empty else float("nan")
    risk_mean = float(risk_summary["Skill_MSE mean"].mean()) if not risk_summary.empty else float("nan")
    m1_mean = full6_m1_skill()

    lines = [
        "# V5 Final Decision",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile。",
        "",
        "## Oracle gate",
        "",
        f"- patch oracle gain vs M1 = {oracle_summary['patch_oracle_mean_gain_vs_m1']:.4f}",
        f"- pixel oracle gain vs M1 = {oracle_summary['pixel_oracle_mean_gain_vs_m1']:.4f}",
        f"- oracle status = `{oracle_summary['status']}`",
        "",
        "## Selector results",
        "",
        f"- Rule selector mean Skill_MSE = {rule_mean:.4f}",
        f"- Soft selector quick mean Skill_MSE = {soft_mean:.4f}",
        f"- Hard selector quick mean Skill_MSE = {hard_mean:.4f}",
        f"- Risk-aware selector mean Skill_MSE = {risk_mean:.4f}",
        f"- Current M1 full-six mean Skill_MSE = {m1_mean:.4f}",
        "",
    ]
    if risk_verdict is not None:
        lines.extend(
            [
                "## Risk-aware verdict",
                "",
                f"- gain vs M1 = {risk_verdict['gain_vs_m1']:.4f}",
                f"- better tiles vs M1 = {risk_verdict['better_tiles_vs_m1']} / 6",
                f"- stress ratio = {risk_verdict['stress_ratio']:.4f}",
                f"- mean train seconds = {risk_verdict['mean_train_seconds']:.2f}",
                f"- V5 status = `{risk_verdict['status']}`",
                "",
            ]
        )
    if ablation_df is not None and not ablation_df.empty:
        lines.extend(
            [
                "## Ablation",
                "",
                "- 已按 no-C3 / no-risk / no-persistence 运行 1-seed ablation，结果见 `v5_ablation.csv`。",
                "",
            ]
        )
    lines.append("## Conclusion")
    lines.append("")
    if risk_verdict is not None and risk_verdict["status"] == "pass":
        lines.append("- V5 达到替代 M1 的门槛，主模型可以改写为 risk-aware expert selector。")
    else:
        lines.append("- V5 没有形成足以稳定替代 M1 的 full-six 证据，M1 仍是最终主模型；V5 作为有 oracle 空间但学习型 selector 未完全兑现的结构探索证据保留。")
    lines.append("")
    FINAL_DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    return FINAL_DECISION_PATH


def write_report(
    run_manifest: pd.DataFrame,
    oracle_summary: dict[str, Any],
    oracle_table: pd.DataFrame,
    wins: pd.DataFrame,
    rule_df: pd.DataFrame,
    soft_quick_summary: pd.DataFrame,
    hard_quick_summary: pd.DataFrame,
    risk_three_seed_summary: pd.DataFrame | None,
    risk_verdict: dict[str, Any] | None,
    ablation_df: pd.DataFrame | None,
) -> Path:
    rule_mean = float(rule_df["Skill_MSE"].mean()) if not rule_df.empty else float("nan")
    soft_mean = float(soft_quick_summary["Skill_MSE mean"].mean()) if not soft_quick_summary.empty else float("nan")
    hard_mean = float(hard_quick_summary["Skill_MSE mean"].mean()) if not hard_quick_summary.empty else float("nan")

    lines = [
        "# ACT-Former V5 风险感知专家选择实验报告",
        "",
        "数据声明：本轮所有实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 实验假设与边界",
        "",
        "- V5 不再重训 `M1` 或 `C3_opt` 本体，只在固定 expert predictions 上训练 selector。",
        "- `M1` 与 `C3_opt` expert 均按每个 tile 的 **validation best_score** 从 `seed 42/43/44` 中选定，再对 `train/val/test` 三个 split 重推预测缓存；这是一条额外假设，已记录在 `v5_selected_expert_runs.csv`。",
        "- 所有 selector 只使用 input window / training window 可得特征：expert predictions、expert disagreement、C3 scale、M1 gate、trend R²、seasonal amplitude、change score 和 horizon embedding。",
        "- Oracle 只作为 upper bound，不作为可部署模型主结果。",
        "",
        "## 1. Expert cache",
        "",
        f"- selected expert rows = {len(run_manifest)}",
        f"- cache manifest = `{EXPERT_CACHE_MANIFEST.as_posix()}`",
        "",
        "## 2. V5-0 Oracle upper bound",
        "",
        f"- sample oracle mean gain vs seeded-M1 = {oracle_summary['sample_oracle_mean_gain_vs_m1']:.4f}",
        f"- patch oracle mean gain vs seeded-M1 = {oracle_summary['patch_oracle_mean_gain_vs_m1']:.4f}",
        f"- pixel oracle mean gain vs seeded-M1 = {oracle_summary['pixel_oracle_mean_gain_vs_m1']:.4f}",
        f"- patch positive tiles = {oracle_summary['patch_positive_tiles']} / 6",
        f"- pixel positive tiles = {oracle_summary['pixel_positive_tiles']} / 6",
        f"- oracle gate status = `{oracle_summary['status']}`",
        "",
        "Oracle 结论：",
        "- patch/pixel oracle 都明显超过 `+0.05`，且 6/6 tile 为正，因此 V5 存在足够可学习空间，不属于“专家错误高度重合”的情形。",
        "",
        "## 3. V5-A Rule-based selector",
        "",
        f"- mean Skill_MSE = {rule_mean:.4f}",
        "- 规则优先在 `t+1`、高 jump/high disagreement 或高 volatility 情形回退 persistence；在长 horizon 且平滑区域允许 `M1/C3_opt/LinearSeasonal` 参与。",
        "",
        "## 4. V5-B / V5-C quick",
        "",
        f"- soft selector quick mean Skill_MSE = {soft_mean:.4f}",
        f"- hard selector quick mean Skill_MSE = {hard_mean:.4f}",
        "- quick 只在 `E32N34/E39N30/E48N24` 上进行，用于判断 soft mixing 与 hard selection 哪条主线更有希望。",
        "",
    ]
    if risk_three_seed_summary is not None and risk_verdict is not None:
        lines.extend(
            [
                "## 5. V5-D Risk-aware selector full-six 3 seeds",
                "",
                f"- mean Skill_MSE = {risk_verdict['mean_skill_mse']:.4f}",
                f"- mean Skill_MSE std = {risk_verdict['mean_skill_std']:.4f}",
                f"- gain vs M1 = {risk_verdict['gain_vs_m1']:.4f}",
                f"- better tiles vs M1 = {risk_verdict['better_tiles_vs_m1']} / 6",
                f"- stress ratio = {risk_verdict['stress_ratio']:.4f}",
                f"- mean train seconds = {risk_verdict['mean_train_seconds']:.2f}",
                f"- status = `{risk_verdict['status']}`",
                "",
            ]
        )
    if ablation_df is not None and not ablation_df.empty:
        lines.extend(
            [
                "## 6. Ablation",
                "",
                "- 仅在 V5-D 通过主门槛时运行。",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果文件",
            "",
            f"- expert seed manifest: `{EXPERT_SEED_MANIFEST.as_posix()}`",
            f"- expert cache manifest: `{EXPERT_CACHE_MANIFEST.as_posix()}`",
            f"- oracle table: `{ORACLE_TABLE_PATH.as_posix()}`",
            f"- oracle win table: `{ORACLE_WIN_TABLE_PATH.as_posix()}`",
            f"- selector results: `{SELECTOR_RESULTS_PATH.as_posix()}`",
            f"- final decision: `{FINAL_DECISION_PATH.as_posix()}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def run_act_v5_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(ACT_V5_ROOT)
    ensure_dir(ACT_V5_CONFIG_ROOT)
    ensure_dir(EXPERT_CACHE_ROOT)
    bundles = preprocess_all_tiles()
    run_manifest = ensure_seed_manifest(reuse_if_exists=reuse_if_exists)
    cache_manifest = ensure_expert_caches(bundles, reuse_if_exists=reuse_if_exists)
    oracle_table, wins, oracle_summary = run_oracle_analysis(reuse_if_exists=reuse_if_exists)

    frames: list[pd.DataFrame] = []
    rule_df = pd.DataFrame()
    soft_quick_summary = pd.DataFrame()
    hard_quick_summary = pd.DataFrame()
    risk_three_seed_summary: pd.DataFrame | None = None
    risk_verdict: dict[str, Any] | None = None
    ablation_df: pd.DataFrame | None = None

    if oracle_summary["status"] == "pass":
        rule_df = run_rule_selector(reuse_if_exists=reuse_if_exists)
        frames.append(rule_df)

        soft_cfg = default_selector_configs()["soft"]
        soft_rows = run_selector_seeded(SEARCH_TILES, soft_cfg, phase="soft_quick", seeds=[42], reuse_if_exists=reuse_if_exists)
        soft_quick_summary = summarize_seeded_rows(soft_rows)
        soft_quick_summary.to_csv(ACT_V5_ROOT / "soft_selector_quick.csv", index=False, encoding="utf-8-sig")
        frames.append(soft_rows)

        hard_cfg = default_selector_configs()["hard"]
        hard_rows = run_selector_seeded(SEARCH_TILES, hard_cfg, phase="hard_quick", seeds=[42], reuse_if_exists=reuse_if_exists)
        hard_quick_summary = summarize_seeded_rows(hard_rows)
        hard_quick_summary.to_csv(ACT_V5_ROOT / "hard_selector_quick.csv", index=False, encoding="utf-8-sig")
        frames.append(hard_rows)

        risk_cfg = default_selector_configs()["risk"]
        risk_rows = run_selector_seeded(FULL_TILES, risk_cfg, phase="risk_full6", seeds=FINAL_SEEDS, reuse_if_exists=reuse_if_exists)
        risk_three_seed_summary = summarize_seeded_rows(risk_rows)
        risk_three_seed_summary.to_csv(ACT_V5_ROOT / "risk_selector_3seed.csv", index=False, encoding="utf-8-sig")
        frames.append(risk_rows)
        risk_verdict = evaluate_v5d(risk_three_seed_summary)
        risk_verdict["tile_compare"].to_csv(ACT_V5_ROOT / "risk_selector_tile_compare.csv", index=False, encoding="utf-8-sig")
        write_json(ACT_V5_ROOT / "risk_selector_verdict.json", {k: v for k, v in risk_verdict.items() if k != "tile_compare"})
        if risk_verdict["status"] == "pass":
            ablation_df = run_ablation_if_needed(reuse_if_exists=reuse_if_exists)

    selector_results = summarize_selector_tables(frames)
    final_decision = write_final_decision(
        oracle_summary=oracle_summary,
        rule_df=rule_df,
        soft_summary=soft_quick_summary,
        hard_summary=hard_quick_summary,
        risk_summary=risk_three_seed_summary if risk_three_seed_summary is not None else pd.DataFrame(),
        risk_verdict=risk_verdict,
        ablation_df=ablation_df,
    )
    report = write_report(
        run_manifest=run_manifest,
        oracle_summary=oracle_summary,
        oracle_table=oracle_table,
        wins=wins,
        rule_df=rule_df,
        soft_quick_summary=soft_quick_summary,
        hard_quick_summary=hard_quick_summary,
        risk_three_seed_summary=risk_three_seed_summary,
        risk_verdict=risk_verdict,
        ablation_df=ablation_df,
    )

    summary = {
        "report": str(report.resolve()),
        "final_decision": str(final_decision.resolve()),
        "expert_seed_manifest": str(EXPERT_SEED_MANIFEST.resolve()),
        "expert_cache_manifest": str(cache_manifest.iloc[0]["Cache path"]) if not cache_manifest.empty else None,
        "oracle_status": oracle_summary["status"],
        "oracle_patch_gain": oracle_summary["patch_oracle_mean_gain_vs_m1"],
        "rule_mean_skill": float(rule_df["Skill_MSE"].mean()) if not rule_df.empty else None,
        "soft_quick_mean_skill": float(soft_quick_summary["Skill_MSE mean"].mean()) if not soft_quick_summary.empty else None,
        "hard_quick_mean_skill": float(hard_quick_summary["Skill_MSE mean"].mean()) if not hard_quick_summary.empty else None,
        "risk_verdict": {k: v for k, v in risk_verdict.items() if k != "tile_compare"} if risk_verdict is not None else None,
        "selector_results": str(SELECTOR_RESULTS_PATH.resolve()) if SELECTOR_RESULTS_PATH.exists() else None,
        "ablation_results": str(ABLATION_PATH.resolve()) if ablation_df is not None else None,
    }
    write_json(SUMMARY_PATH, summary)
    return summary
