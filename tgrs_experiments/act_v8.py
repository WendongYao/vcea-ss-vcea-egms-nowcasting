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

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS
from .act_v3 import (
    ActV3Config,
    ActV3Dataset,
    count_parameters,
    result_rows,
    summarize_seeded_rows,
)
from .act_v4c_efficiency import ConvNeXtLiteEncoder, build_temporal_mixer as build_proxy_temporal_mixer, maybe_compile
from .act_v7 import build_seed_expert_cache
from .pipeline import OUTPUT_ROOT, REGIME_NAMES, TILE_ORDER, TileBundle, ensure_dir, preprocess_all_tiles, seed_everything

try:
    from mamba_ssm.modules.mamba_simple import Mamba as RealMamba

    MAMBA_SSM_AVAILABLE = True
except Exception:
    RealMamba = None
    MAMBA_SSM_AVAILABLE = False


ACT_V8_ROOT = ensure_dir(OUTPUT_ROOT / "act_v8")
V8_RUN_ROOT = ensure_dir(ACT_V8_ROOT / "runs")
QUICK_ROOT = ensure_dir(ACT_V8_ROOT / "quick")
FULL6_ROOT = ensure_dir(ACT_V8_ROOT / "fullsix")
FIG_ROOT = ensure_dir(ACT_V8_ROOT / "figures")

SEARCH_TILES = ["E32N34", "E39N30", "E48N24"]
STRESS_TILES = ["E48N24", "E58N17"]
BEST_STATIC_MODEL_ID = "StaticVal_inverse_temp4p0_sh0p1_floor0p0"

M1_FULL6_SUMMARY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"
VCEA_FINAL_COMPARISON_PATH = OUTPUT_ROOT / "staticval_ea_study" / "final_aggregation_comparison.csv"

COMMON_METRICS = [
    "RMSE [mm]",
    "MAE [mm]",
    "Acc@1mm [%]",
    "Acc@0.5mm [%]",
    "SSIM",
    "Skill_MSE",
    "Skill_RMSE",
    "Persistence RMSE [mm]",
    "LinearSeasonal RMSE [mm]",
    "Elapsed seconds",
    "Infer seconds",
    "Param count",
    "Peak GPU memory [MB]",
]


@dataclass(frozen=True)
class StaticAggConfig:
    method: str = "inverse"
    temperature: float = 4.0
    shrinkage: float = 0.1
    m1_floor: float = 0.0


@dataclass
class V8Config:
    model_id: str
    family: str
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    dim: int = 64
    patch_size: int = 8
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 3
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    compile_model: bool = False
    # shared bounded head
    bound_eps_mm: float = 0.05
    bound_scale_lambda: float = 0.80
    bound_acc_lambda: float = 0.20
    bound_amp_lambda: float = 0.15
    no_bounded_head: bool = False
    # STR-Mamba
    scan_four_directions: bool = True
    no_spatial_scan: bool = False
    no_temporal_mixer: bool = False
    no_regime_modulation: bool = False
    temporal_depth: int = 1
    # KINO
    kino_modes: int = 8
    kino_operator_only: bool = False
    kino_no_decomposition: bool = False
    kino_no_low_operator: bool = False
    kino_no_high_corrector: bool = False
    kino_no_sign_consistency: bool = False
    kino_recent_lambda: float = 0.70
    kino_sign_lambda: float = 0.03
    # CoMEx
    comex_hard_lambda: float = 0.08
    comex_div_lambda: float = 0.01
    comex_no_hard_weight: bool = False
    comex_no_diversity: bool = False
    # bookkeeping
    phase_tag: str = "quick"

    @property
    def output_len(self) -> int:
        return len(self.horizons)


def best_static_cfg() -> StaticAggConfig:
    return StaticAggConfig()


def to_v3_cfg(config: V8Config) -> ActV3Config:
    return ActV3Config(
        model_id=config.model_id,
        variant="v3a",
        input_len=config.input_len,
        horizons=config.horizons,
        dim=config.dim,
        depth=1,
        heads=4,
        mlp_dim=128,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        patience=config.patience,
        seed=config.seed,
        use_amp=config.use_amp,
        select_by_skill=True,
        num_workers=config.num_workers,
    )


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def apply_m1_floor_generic(weights: np.ndarray, floor: float, m1_idx: int = 2) -> np.ndarray:
    out = np.clip(weights.astype(np.float64), 1e-8, None)
    out /= max(float(out.sum()), 1e-8)
    if floor <= 0:
        return out.astype(np.float32)
    if out[m1_idx] < floor:
        remain = max(1e-6, 1.0 - floor)
        others = np.array([idx for idx in range(len(out)) if idx != m1_idx], dtype=np.int64)
        other_sum = float(out[others].sum())
        if other_sum <= 1e-8:
            out[others] = remain / max(1, len(others))
        else:
            out[others] = out[others] / other_sum * remain
        out[m1_idx] = floor
    out = np.clip(out, 1e-8, None)
    out /= max(float(out.sum()), 1e-8)
    return out.astype(np.float32)


def estimate_static_weights_generic(
    y_true: np.ndarray,
    expert_preds: np.ndarray,
    valid_mask: np.ndarray,
    cfg: StaticAggConfig,
    persistence_idx: int = 0,
    m1_idx: int = 2,
) -> np.ndarray:
    valid = valid_mask > 0.5
    n_horizons = expert_preds.shape[1]
    n_experts = expert_preds.shape[2]
    weights = np.zeros((n_horizons, n_experts), dtype=np.float32)
    for h_idx in range(n_horizons):
        losses = []
        for exp_idx in range(n_experts):
            sq = (expert_preds[:, h_idx, exp_idx] - y_true[:, h_idx]) ** 2
            losses.append(float(sq[:, valid].mean()))
        loss_vec = np.asarray(losses, dtype=np.float64)
        rel = loss_vec / max(float(loss_vec[persistence_idx]), 1e-8)
        if cfg.method == "softmax":
            rel_shift = rel - rel.min()
            raw = np.exp(-cfg.temperature * rel_shift)
        elif cfg.method == "inverse":
            raw = 1.0 / np.power(np.maximum(rel, 1e-8), cfg.temperature)
        else:
            raise KeyError(cfg.method)
        raw = raw / max(float(raw.sum()), 1e-8)
        raw = (1.0 - cfg.shrinkage) * raw + cfg.shrinkage * (1.0 / n_experts)
        weights[h_idx] = apply_m1_floor_generic(raw.astype(np.float32), cfg.m1_floor, m1_idx=m1_idx)
    return weights


def predict_static_generic(expert_preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(expert_preds[:, :, 0], dtype=np.float32)
    for h_idx in range(expert_preds.shape[1]):
        # expert stack layout is [sample, horizon, expert, H, W]
        pred[:, h_idx] = (expert_preds[:, h_idx] * weights[h_idx, :, None, None]).sum(axis=1)
    return pred.astype(np.float32)


def stack_base_experts(cache: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [
            cache["persistence_mm"],
            cache["linear_seasonal_mm"],
            cache["m1_pred_mm"],
            cache["c3_pred_mm"],
        ],
        axis=2,
    ).astype(np.float32)


def normalize_horizon_maps(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    for h_idx in range(arr.shape[0]):
        scale = float(np.percentile(np.abs(arr[h_idx]), 95))
        out[h_idx] = np.clip(arr[h_idx] / max(scale, 1e-6), -2.0, 2.0).astype(np.float32)
    return out


def build_comex_feature_pack(bundles: dict[str, TileBundle], tile: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    train_cache = build_seed_expert_cache(bundles, tile, seed, "train")
    train_experts = stack_base_experts(train_cache)
    # Assumption: in absence of inner-split OOF expert errors, use train-only expert disagreement
    # and M1/C3 spread as a causal hard-case prior proxy.
    hard_prior = train_experts.std(axis=2).mean(axis=0) + 0.5 * np.abs(train_cache["c3_pred_mm"] - train_cache["m1_pred_mm"]).mean(axis=0)
    hard_prior = np.clip(normalize_horizon_maps(hard_prior), 0.0, 2.0).astype(np.float32)

    packs: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ["train", "val", "test"]:
        cache = build_seed_expert_cache(bundles, tile, seed, split_name)
        experts = stack_base_experts(cache)
        expert_mean = experts.mean(axis=2).astype(np.float32)
        expert_std = experts.std(axis=2).astype(np.float32)
        expert_gap = (expert_mean - cache["m1_pred_mm"]).astype(np.float32)
        expert_spread = np.abs(cache["c3_pred_mm"] - cache["m1_pred_mm"]).astype(np.float32)
        packs[split_name] = {
            "expert_mean_mm": expert_mean,
            "expert_std_mm": expert_std,
            "expert_gap_mm": expert_gap,
            "expert_spread_mm": expert_spread,
            "m1_expert_mm": cache["m1_pred_mm"].astype(np.float32),
            "c3_expert_mm": cache["c3_pred_mm"].astype(np.float32),
            "hard_case_prior": np.repeat(hard_prior[None], expert_mean.shape[0], axis=0).astype(np.float32),
            "current_frames": cache["current_frames"].astype(np.int64),
            "target_indices": cache["target_indices"].astype(np.int64),
        }
    return packs


def calibrate_safe_thresholds(
    val_pred: np.ndarray,
    val_m1: np.ndarray,
    val_true: np.ndarray,
    disagreement: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00),
) -> dict[int, float]:
    valid = valid_mask > 0.5
    out: dict[int, float] = {}
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        best_tau = float(thresholds[0])
        best_score = -float("inf")
        for tau in thresholds:
            pred = np.where(disagreement[:, h_idx] > tau, val_m1[:, h_idx], val_pred[:, h_idx]).astype(np.float32)
            model_mse = float(np.mean((pred[:, valid] - val_true[:, h_idx][:, valid]) ** 2))
            pers_mse = float(np.mean((val_m1[:, h_idx][:, valid] - val_true[:, h_idx][:, valid]) ** 2))
            score = 1.0 - model_mse / max(pers_mse, 1e-6)
            if score > best_score:
                best_score = score
                best_tau = float(tau)
        out[horizon] = best_tau
    return out


def apply_safe_wrapper(pred: np.ndarray, m1_pred: np.ndarray, disagreement: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    out = pred.copy().astype(np.float32)
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        tau = float(thresholds[horizon])
        risky = disagreement[:, h_idx] > tau
        out[:, h_idx][risky] = m1_pred[:, h_idx][risky]
    return out


def rows_from_arrays(model_id: str, tile: str, seed: int, phase: str, arrays: dict[str, np.ndarray], elapsed: float, infer_seconds: float, param_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        result_rows(
            model_id=model_id,
            tile=tile,
            seed=seed,
            phase=phase,
            elapsed_seconds=elapsed,
            infer_seconds=infer_seconds,
            best_epoch=-1,
            param_count=param_count,
            arrays=arrays,
            horizons=SEARCH_HORIZONS,
        )
    )


def build_reference_rows(model_id: str, tile: str, seed: int, phase: str, cache: dict[str, np.ndarray], pred: np.ndarray) -> pd.DataFrame:
    arrays = {
        "y_pred_mm": pred.astype(np.float32),
        "y_true_mm": cache["y_true_mm"].astype(np.float32),
        "persistence_mm": cache["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
        "valid_mask": cache["valid_mask"].astype(np.float32),
        "regime_map": cache["regime_map"].astype(np.int16),
        "current_frames": cache["current_frames"].astype(np.int64),
        "target_indices": cache["target_indices"].astype(np.int64),
    }
    return rows_from_arrays(model_id, tile, seed, phase, arrays, elapsed=0.0, infer_seconds=0.0, param_count=0)


def compare_summary_to_reference(summary_df: pd.DataFrame, ref_summary: pd.DataFrame, tiles: list[str]) -> dict[str, Any]:
    model_id = str(summary_df["Model"].iloc[0])
    merged = summary_df.merge(
        ref_summary[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "Ref Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs Ref"] = merged["Skill_MSE mean"] - merged["Ref Skill_MSE mean"]
    tile_compare = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "Ref Skill_MSE mean", "Delta vs Ref"]]
        .mean()
        .rename(columns={"Skill_MSE mean": "Model Skill_MSE mean"})
    )
    stress_tiles = ["E48N24", "E58N17"] if set(STRESS_TILES).issubset(set(tiles)) else ["E48N24"]
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


def reference_quick_m1_vcea(bundles: dict[str, TileBundle], seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m1_rows: list[pd.DataFrame] = []
    vcea_rows: list[pd.DataFrame] = []
    vcea_safe_rows: list[pd.DataFrame] = []
    cfg = best_static_cfg()
    for tile in SEARCH_TILES:
        val_cache = build_seed_expert_cache(bundles, tile, seed, "val")
        test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
        m1_rows.append(build_reference_rows("M1", tile, seed, "test", test_cache, test_cache["m1_pred_mm"]))
        expert_val = np.stack(
            [
                val_cache["persistence_mm"],
                val_cache["linear_seasonal_mm"],
                val_cache["m1_pred_mm"],
                val_cache["c3_pred_mm"],
            ],
            axis=2,
        ).astype(np.float32)
        expert_test = np.stack(
            [
                test_cache["persistence_mm"],
                test_cache["linear_seasonal_mm"],
                test_cache["m1_pred_mm"],
                test_cache["c3_pred_mm"],
            ],
            axis=2,
        ).astype(np.float32)
        weights = estimate_static_weights_generic(val_cache["y_true_mm"], expert_val, val_cache["valid_mask"], cfg)
        vcea_pred = predict_static_generic(expert_test, weights)
        vcea_rows.append(build_reference_rows("VCEA", tile, seed, "test", test_cache, vcea_pred))
        val_pred = predict_static_generic(expert_val, weights)
        val_disagree = expert_val.std(axis=2).astype(np.float32)
        test_disagree = expert_test.std(axis=2).astype(np.float32)
        safe_tau = calibrate_safe_thresholds(val_pred, val_cache["m1_pred_mm"], val_cache["y_true_mm"], val_disagree, val_cache["valid_mask"])
        vcea_safe = apply_safe_wrapper(vcea_pred, test_cache["m1_pred_mm"], test_disagree, safe_tau)
        vcea_safe_rows.append(build_reference_rows("VCEA-safe", tile, seed, "test", test_cache, vcea_safe))
    return (
        summarize_seeded_rows(pd.concat(m1_rows, ignore_index=True)),
        summarize_seeded_rows(pd.concat(vcea_rows, ignore_index=True)),
        summarize_seeded_rows(pd.concat(vcea_safe_rows, ignore_index=True)),
    )


class V8Dataset(Dataset):
    def __init__(
        self,
        bundle: TileBundle,
        split_name: str,
        config: V8Config,
        expert_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.bundle = bundle
        self.base = ActV3Dataset(bundle, resolution=64, split_name=split_name, config=to_v3_cfg(config))
        self.expert_features = expert_features
        tensors = bundle.tensors64
        self.disp = tensors["disp"].astype(np.float32)
        self.velocity_map = tensors["descriptor_velocity"].astype(np.float32)
        self.acc_map = tensors["descriptor_acc"].astype(np.float32)
        self.amp_map = tensors["descriptor_amp"].astype(np.float32)
        self.phase_map = tensors["seasonal_phase"].astype(np.float32)
        self.trend_r2_map = tensors["trend_r2"].astype(np.float32)
        self.regime_map = tensors["regime_map"].astype(np.int16)
        desc = np.stack(
            [
                self.velocity_map,
                self.acc_map,
                self.amp_map,
                self.trend_r2_map,
            ],
            axis=0,
        ).astype(np.float32)
        mu = desc.mean(axis=(1, 2), keepdims=True)
        sigma = desc.std(axis=(1, 2), keepdims=True) + 1e-6
        self.desc_norm = ((desc - mu) / sigma).astype(np.float32)
        if self.expert_features is not None:
            sample_frames = np.asarray([int(sample["current_frame"][0]) for sample in self.base.samples], dtype=np.int64)
            sample_targets = np.stack([np.asarray(sample["target_indices"], dtype=np.int64) for sample in self.base.samples], axis=0)
            if not np.array_equal(sample_frames, self.expert_features["current_frames"].reshape(-1)):
                raise ValueError(f"CoMEx feature current-frame mismatch tile={bundle.tile_id} split={split_name}")
            if not np.array_equal(sample_targets, self.expert_features["target_indices"]):
                raise ValueError(f"CoMEx feature target-index mismatch tile={bundle.tile_id} split={split_name}")

    def __len__(self) -> int:
        return len(self.base.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base.samples[index]
        current = int(sample["current_frame"][0])
        start = current - self.base.config.input_len + 1
        disp_hist = self.disp[start : current + 1].astype(np.float32)
        recent_mean = disp_hist.mean(axis=0, keepdims=True).astype(np.float32)
        if disp_hist.shape[0] >= 4:
            recent_velocity = np.diff(disp_hist[-4:], axis=0).mean(axis=0, keepdims=True).astype(np.float32)
        else:
            recent_velocity = np.diff(disp_hist, axis=0).mean(axis=0, keepdims=True).astype(np.float32) if disp_hist.shape[0] > 1 else np.zeros_like(recent_mean)
        out: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            tensor = torch.from_numpy(value)
            out[key] = tensor.float() if value.dtype != np.int64 else tensor.long()
        out["x_main"] = out["x"][:, [0, 1, 2, 4, 5, 6], :, :].contiguous()
        out["desc_stack"] = torch.from_numpy(self.desc_norm).float()
        out["velocity_map"] = torch.from_numpy(self.velocity_map[None]).float()
        out["acc_map"] = torch.from_numpy(self.acc_map[None]).float()
        out["amp_map"] = torch.from_numpy(self.amp_map[None]).float()
        out["phase_map"] = torch.from_numpy(self.phase_map[None]).float()
        out["trend_r2_map"] = torch.from_numpy(self.trend_r2_map[None]).float()
        out["recent_mean_mm"] = torch.from_numpy(recent_mean).float()
        out["recent_velocity_mm"] = torch.from_numpy(recent_velocity).float()
        out["current_time"] = out["time_history"][-1].clone()
        if self.expert_features is not None:
            for key in [
                "expert_mean_mm",
                "expert_std_mm",
                "expert_gap_mm",
                "expert_spread_mm",
                "m1_expert_mm",
                "c3_expert_mm",
                "hard_case_prior",
            ]:
                out[key] = torch.from_numpy(self.expert_features[key][index][:, None]).float()
        return out


class DirectionalPatchScan(nn.Module):
    def __init__(self, dim: int, side: int, four_direction: bool) -> None:
        super().__init__()
        self.side = side
        self.four_direction = four_direction
        self.use_real_mamba = MAMBA_SSM_AVAILABLE
        if self.use_real_mamba:
            n_dir = 4 if four_direction else 1
            self.mixers = nn.ModuleList([ResidualMambaBlock(dim) for _ in range(n_dir)])
            self.gru = None
        else:
            self.mixers = None
            self.gru = nn.GRU(dim, dim, batch_first=True)
        self.register_buffer("orders", self._build_orders(side), persistent=False)
        self.register_buffer("inverse_orders", self._build_inverse(self._build_orders(side)), persistent=False)

    @staticmethod
    def _build_orders(side: int) -> torch.Tensor:
        row = []
        col = []
        for y in range(side):
            for x in range(side):
                row.append(y * side + x)
                col.append(x * side + y)
        row = np.asarray(row, dtype=np.int64)
        row_rev = row[::-1].copy()
        col = np.asarray(col, dtype=np.int64)
        col_rev = col[::-1].copy()
        return torch.from_numpy(np.stack([row, row_rev, col, col_rev], axis=0))

    @staticmethod
    def _build_inverse(orders: torch.Tensor) -> torch.Tensor:
        inv = []
        for order in orders.cpu().numpy():
            arr = np.zeros_like(order)
            arr[order] = np.arange(len(order))
            inv.append(arr)
        return torch.from_numpy(np.stack(inv, axis=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        n_dir = 4 if self.four_direction else 1
        outs = []
        for dir_idx in range(n_dir):
            order = self.orders[dir_idx]
            inv = self.inverse_orders[dir_idx]
            seq = tokens[:, order]
            if self.use_real_mamba:
                assert self.mixers is not None
                out = self.mixers[dir_idx](seq)
            else:
                assert self.gru is not None
                out, _ = self.gru(seq)
            restored = out[:, inv]
            outs.append(restored)
        fused = torch.stack(outs, dim=0).mean(dim=0)
        return fused.transpose(1, 2).reshape(b, c, h, w)


class ResidualMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        if RealMamba is None:
            raise RuntimeError("RealMamba is not available in the current environment")
        self.norm = nn.LayerNorm(dim)
        self.mamba = RealMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class TemporalMambaMixer2d(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualMambaBlock(dim) for _ in range(max(1, depth))])

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = seq.shape
        x = seq.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c).contiguous()
        for layer in self.layers:
            x = layer(x)
        final = x[:, -1].reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        norm = final.pow(2).mean(dim=(1, 2, 3)).sqrt()
        return final, norm


class FiLMHorizonDecoder2d(nn.Module):
    def __init__(self, dim: int, output_len: int) -> None:
        super().__init__()
        self.output_len = output_len
        self.h_embed = nn.Parameter(torch.randn(output_len, dim) * 0.02)
        self.future_proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gamma_head = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.Conv2d(dim, dim, kernel_size=1))
        self.beta_head = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.Conv2d(dim, dim, kernel_size=1))
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor, future_time: torch.Tensor) -> torch.Tensor:
        b, c, h, w = state.shape
        future_embed = self.future_proj(future_time.reshape(b * self.output_len, 2)).reshape(b, self.output_len, c)
        cond = context[:, None, :, :, :] + future_embed[:, :, :, None, None] + self.h_embed[None, :, :, None, None]
        cond_flat = cond.reshape(b * self.output_len, c, h, w)
        gamma = self.gamma_head(cond_flat).reshape(b, self.output_len, c, h, w)
        beta = self.beta_head(cond_flat).reshape(b, self.output_len, c, h, w)
        feat = state[:, None, :, :, :] * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * beta + future_embed[:, :, :, None, None]
        raw = self.out(feat.reshape(b * self.output_len, c, h, w))
        return raw.reshape(b, self.output_len, 1, h, w)


class STRMambaModel(nn.Module):
    def __init__(self, config: V8Config) -> None:
        super().__init__()
        self.config = config
        self.side = 64 // config.patch_size
        spatial_in = 8
        temporal_in = 6
        self.spatial_patch = nn.Conv2d(spatial_in, config.dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.temporal_patch = nn.Conv2d(temporal_in, config.dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.scan = DirectionalPatchScan(config.dim, self.side, config.scan_four_directions)
        if MAMBA_SSM_AVAILABLE and not config.no_temporal_mixer:
            self.temporal = TemporalMambaMixer2d(config.dim, config.temporal_depth)
        else:
            self.temporal = build_proxy_temporal_mixer(config.dim, config.temporal_depth, "none" if config.no_temporal_mixer else "ssm")
        self.desc_proj = nn.Conv2d(4 + len(REGIME_NAMES), config.dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * config.dim, config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(config.dim, config.dim, kernel_size=config.patch_size, stride=config.patch_size),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = FiLMHorizonDecoder2d(config.dim, config.output_len)
        self.bound_rho = nn.Parameter(torch.full((config.output_len,), 0.8))

    def _regime_onehot(self, regime: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(regime.long(), num_classes=len(REGIME_NAMES)).permute(0, 3, 1, 2).float()
        return oh

    def _bounded(self, batch: dict[str, torch.Tensor], raw: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.no_bounded_head:
            pred = anchor[:, None, :, :, :] + raw
            scale = torch.ones_like(raw)
            return pred, scale
        rho = F.softplus(self.bound_rho)[None, :, None, None, None]
        acc = batch["acc_map"][:, None, :, :, :].abs()
        amp_map = batch["amp_map"][:, None, :, :, :].abs()
        scale = rho * (
            self.config.bound_eps_mm
            + self.config.bound_scale_lambda * batch["scale_mm"]
            + self.config.bound_acc_lambda * acc
            + self.config.bound_amp_lambda * amp_map
        )
        pred_delta = scale * torch.tanh(raw)
        pred = anchor[:, None, :, :, :] + pred_delta
        return pred, scale

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = batch["x_main"]
        spatial_feat = torch.cat(
            [
                x[:, -1, 0:1],
                batch["recent_mean_mm"] / 10.0,
                x[:, -1, 1:2],
                x[:, -1, 3:4],
                batch["desc_stack"],
            ],
            dim=1,
        )
        spatial_patch = self.spatial_patch(spatial_feat)
        if self.config.no_spatial_scan:
            spatial_state = spatial_patch
        else:
            spatial_state = self.scan(spatial_patch)

        b, t, _, _, _ = x.shape
        temporal_in = x[:, :, [0, 1, 3, 4, 5, 5], :, :].clone()
        temporal_in[:, :, 5] = torch.abs(x[:, :, 1])
        temporal_tokens = self.temporal_patch(temporal_in.reshape(b * t, 6, 64, 64)).reshape(b, t, self.config.dim, self.side, self.side)
        temporal_state, temporal_norm = self.temporal(temporal_tokens)

        desc_ctx = self.desc_proj(
            torch.cat(
                [
                    F.avg_pool2d(batch["desc_stack"], kernel_size=self.config.patch_size, stride=self.config.patch_size),
                    F.avg_pool2d(self._regime_onehot(batch["regime"]), kernel_size=self.config.patch_size, stride=self.config.patch_size),
                ],
                dim=1,
            )
        )
        context_patch = desc_ctx if not self.config.no_regime_modulation else torch.zeros_like(desc_ctx)
        fused = self.fuse(torch.cat([spatial_state, temporal_state], dim=1))
        fused_64 = self.upsample(fused)
        context_64 = F.interpolate(context_patch, size=fused_64.shape[-2:], mode="nearest")
        raw = self.decoder(fused_64, context_64, batch["future_time"])
        anchor = batch["last_disp_mm"]
        pred, scale = self._bounded(batch, raw, anchor)
        aux = {
            "scale_mm": scale,
            "pred_delta_mm": pred - anchor[:, None, :, :, :],
            "anchor_mm": anchor[:, None, :, :, :],
            "state_norm": temporal_norm,
            "spatial_norm": spatial_state.pow(2).mean(dim=(1, 2, 3)).sqrt(),
        }
        if return_aux:
            return pred, aux
        return pred


class LowModeOperator2d(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, modes: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.in_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.weight_real = nn.Parameter(torch.randn(hidden_dim, hidden_dim, modes, modes) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(hidden_dim, hidden_dim, modes, modes) * 0.02)

    def compl_mul2d(self, x: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        real = torch.einsum("bixy,ioxy->boxy", x.real, wr) - torch.einsum("bixy,ioxy->boxy", x.imag, wi)
        imag = torch.einsum("bixy,ioxy->boxy", x.real, wi) + torch.einsum("bixy,ioxy->boxy", x.imag, wr)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros((b, c, h, w // 2 + 1), dtype=torch.complex64, device=x.device)
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weight_real[:, :, :mh, :mw], self.weight_imag[:, :, :mh, :mw])
        out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return self.out_proj(out)


def seasonal_signal(amp: torch.Tensor, phase: torch.Tensor, sin_val: torch.Tensor, cos_val: torch.Tensor) -> torch.Tensor:
    sin_phase = torch.sin(phase)
    cos_phase = torch.cos(phase)
    return amp * (sin_val * cos_phase + cos_val * sin_phase)


class KINOModel(nn.Module):
    def __init__(self, config: V8Config) -> None:
        super().__init__()
        self.config = config
        self.recent_mix_logit = nn.Parameter(torch.tensor(math.log(config.kino_recent_lambda / max(1e-6, 1.0 - config.kino_recent_lambda)), dtype=torch.float32))
        self.low_operator = None if config.kino_no_low_operator else LowModeOperator2d(in_channels=5, hidden_dim=config.dim, modes=config.kino_modes)
        self.high_encoder = None
        if not config.kino_operator_only and not config.kino_no_high_corrector:
            self.high_encoder = ConvNeXtLiteEncoder(in_channels=1 + 1 + 1 + 4 + len(REGIME_NAMES), dim=config.dim, blocks=2)
        self.context_proj = nn.Sequential(
            nn.Conv2d(4, config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=1),
        )
        self.decoder = FiLMHorizonDecoder2d(config.dim, config.output_len)
        self.bound_rho = nn.Parameter(torch.full((config.output_len,), 0.8))

    def _regime_onehot(self, regime: torch.Tensor) -> torch.Tensor:
        return F.one_hot(regime.long(), num_classes=len(REGIME_NAMES)).permute(0, 3, 1, 2).float()

    def _causal_baseline(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        last_disp = batch["last_disp_mm"][:, 0]
        if self.config.kino_no_decomposition:
            return last_disp[:, None, :, :]
        lam = torch.sigmoid(self.recent_mix_logit)
        v_recent = batch["recent_velocity_mm"][:, 0]
        v_train = batch["velocity_map"][:, 0]
        tau = lam * v_recent + (1.0 - lam) * v_train
        amp_map = batch["amp_map"][:, 0]
        phase_map = batch["phase_map"][:, 0]
        cur_sin = batch["current_time"][:, 0][:, None, None]
        cur_cos = batch["current_time"][:, 1][:, None, None]
        cur_season = seasonal_signal(amp_map, phase_map, cur_sin, cur_cos)
        future_sin = batch["future_time"][:, :, 0][:, :, None, None]
        future_cos = batch["future_time"][:, :, 1][:, :, None, None]
        future_season = seasonal_signal(amp_map[:, None], phase_map[:, None], future_sin, future_cos)
        season_inc = future_season - cur_season[:, None]
        step = torch.tensor(self.config.horizons, device=last_disp.device, dtype=last_disp.dtype)[None, :, None, None]
        baseline = last_disp[:, None, :, :] + step * tau[:, None, :, :] + season_inc
        return baseline

    def _bounded(self, batch: dict[str, torch.Tensor], raw: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.no_bounded_head:
            pred = anchor[:, :, None, :, :] + raw
            scale = torch.ones_like(raw)
            return pred, scale
        rho = F.softplus(self.bound_rho)[None, :, None, None, None]
        acc = batch["acc_map"][:, None, :, :, :].abs()
        amp_map = batch["amp_map"][:, None, :, :, :].abs()
        scale = rho * (
            self.config.bound_eps_mm
            + self.config.bound_scale_lambda * batch["scale_mm"]
            + self.config.bound_acc_lambda * acc
            + self.config.bound_amp_lambda * amp_map
        )
        pred_delta = scale * torch.tanh(raw)
        pred = anchor[:, :, None, :, :] + pred_delta
        return pred, scale

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        baseline = self._causal_baseline(batch)
        low_inputs = torch.cat(
            [
                batch["x_main"][:, -1, 0:1],
                batch["recent_mean_mm"] / 10.0,
                batch["recent_velocity_mm"] / 2.0,
                batch["desc_stack"][:, 0:1],
                batch["desc_stack"][:, 2:3],
            ],
            dim=1,
        )
        state = torch.zeros((baseline.shape[0], self.config.dim, 64, 64), device=baseline.device, dtype=baseline.dtype)
        if self.low_operator is not None:
            state = state + self.low_operator(low_inputs)
        if self.high_encoder is not None:
            high_inputs = torch.cat(
                [
                    batch["x_main"][:, -1, 0:1],
                    batch["x_main"][:, -1, 1:2],
                    batch["x_main"][:, -1, 3:4],
                    batch["desc_stack"],
                    self._regime_onehot(batch["regime"]),
                ],
                dim=1,
            )
            state = state + self.high_encoder(high_inputs)
        context = self.context_proj(batch["desc_stack"][:, 0:4])
        raw = self.decoder(state, context, batch["future_time"])
        pred, scale = self._bounded(batch, raw, baseline)
        aux = {
            "scale_mm": scale,
            "pred_delta_mm": pred - baseline[:, :, None, :, :],
            "anchor_mm": baseline[:, :, None, :, :],
            "state_norm": state.pow(2).mean(dim=(1, 2, 3)).sqrt(),
            "spatial_norm": state.pow(2).mean(dim=(1, 2, 3)).sqrt(),
        }
        if return_aux:
            return pred, aux
        return pred


class CoMExModel(nn.Module):
    def __init__(self, config: V8Config) -> None:
        super().__init__()
        self.config = config
        raw_in = 1 + 1 + 1 + 4 + len(REGIME_NAMES)
        self.raw_encoder = nn.Sequential(
            nn.Conv2d(raw_in, config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(4 + len(REGIME_NAMES), config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=1),
        )
        self.cons_decoder = FiLMHorizonDecoder2d(config.dim, config.output_len)
        expert_in = 7
        self.ortho_encoder = nn.Sequential(
            nn.Conv2d(expert_in, config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.ortho_head = nn.Sequential(
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.dim, 1, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(3, max(8, config.dim // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(8, config.dim // 4), 1, kernel_size=1),
        )
        self.bound_rho = nn.Parameter(torch.full((config.output_len,), 0.7))

    def _regime_onehot(self, regime: torch.Tensor) -> torch.Tensor:
        return F.one_hot(regime.long(), num_classes=len(REGIME_NAMES)).permute(0, 3, 1, 2).float()

    def _bounded(self, batch: dict[str, torch.Tensor], raw: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.no_bounded_head:
            pred = anchor[:, :, None, :, :] + raw
            scale = torch.ones_like(raw)
            return pred, scale
        rho = F.softplus(self.bound_rho)[None, :, None, None, None]
        acc = batch["acc_map"][:, None, :, :, :].abs()
        amp_map = batch["amp_map"][:, None, :, :, :].abs()
        expert_std = batch["expert_std_mm"].abs()
        scale = rho * (
            self.config.bound_eps_mm
            + self.config.bound_scale_lambda * (batch["scale_mm"] + 0.25 * expert_std)
            + self.config.bound_acc_lambda * acc
            + self.config.bound_amp_lambda * amp_map
        )
        pred_delta = scale * torch.tanh(raw)
        pred = anchor[:, :, None, :, :] + pred_delta
        return pred, scale

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        regime_onehot = self._regime_onehot(batch["regime"])
        raw_inputs = torch.cat(
            [
                batch["x_main"][:, -1, 0:1],
                batch["recent_mean_mm"] / 10.0,
                batch["recent_velocity_mm"] / 2.0,
                batch["desc_stack"],
                regime_onehot,
            ],
            dim=1,
        )
        raw_feat = self.raw_encoder(raw_inputs)
        context = self.context_proj(torch.cat([batch["desc_stack"], regime_onehot], dim=1))
        cons_raw = self.cons_decoder(raw_feat, context, batch["future_time"])

        b, h = batch["expert_mean_mm"].shape[:2]
        ortho_inputs = torch.cat(
            [
                batch["expert_mean_mm"],
                batch["expert_std_mm"],
                batch["expert_gap_mm"],
                batch["expert_spread_mm"],
                batch["linear_seasonal_mm"],
                batch["persistence_mm"],
                batch["hard_case_prior"],
            ],
            dim=2,
        )
        ortho_state = self.ortho_encoder(ortho_inputs.reshape(b * h, ortho_inputs.shape[2], 64, 64))
        ortho_raw = self.ortho_head(ortho_state).reshape(b, h, 1, 64, 64)

        gate_inputs = torch.cat(
            [
                batch["expert_std_mm"],
                batch["expert_spread_mm"],
                batch["hard_case_prior"],
            ],
            dim=2,
        )
        q = torch.sigmoid(self.gate_head(gate_inputs.reshape(b * h, gate_inputs.shape[2], 64, 64))).reshape(b, h, 1, 64, 64)
        raw = (1.0 - q) * cons_raw + q * ortho_raw
        anchor = batch["m1_expert_mm"][:, :, 0]
        pred, scale = self._bounded(batch, raw, anchor)
        aux = {
            "scale_mm": scale,
            "pred_delta_mm": pred - anchor[:, :, None, :, :],
            "anchor_mm": anchor[:, :, None, :, :],
            "state_norm": raw_feat.pow(2).mean(dim=(1, 2, 3)).sqrt(),
            "spatial_norm": ortho_state.pow(2).mean(dim=(1, 2, 3)).reshape(b, h).mean(dim=1),
            "selector_q": q,
        }
        if return_aux:
            return pred, aux
        return pred


def loader_kwargs(config: V8Config, shuffle: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_model(config: V8Config) -> nn.Module:
    if config.family == "STR-Mamba":
        return STRMambaModel(config)
    if config.family == "KINO":
        return KINOModel(config)
    if config.family == "CoMEx":
        return CoMExModel(config)
    raise KeyError(config.family)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None, :, :]
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None, :, :]
    err = (pred - target) ** 2
    return (err * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def compute_loss_v8(pred: torch.Tensor, batch: dict[str, torch.Tensor], aux: dict[str, torch.Tensor], config: V8Config) -> tuple[torch.Tensor, float]:
    target = batch["target_mm"]
    mask = batch["mask"]
    loss = masked_smooth_l1(pred, target, mask)
    if config.family == "KINO" and not config.kino_no_sign_consistency:
        stable_mask = ((batch["regime"] == 0) | (batch["regime"] == 2)).float() * batch["mask"]
        if float(stable_mask.sum().item()) > 1.0:
            true_delta = target - aux["anchor_mm"]
            pred_delta = pred - aux["anchor_mm"]
            sign_pen = F.relu(-(pred_delta * true_delta))
            weights = stable_mask[:, None, None, :, :]
            loss = loss + config.kino_sign_lambda * (sign_pen * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)
    if config.family == "CoMEx":
        if not config.comex_no_hard_weight:
            hard_weights = mask[:, None, None, :, :] * (1.0 + batch["hard_case_prior"])
            hard_term = F.smooth_l1_loss(pred, target, reduction="none")
            loss = loss + config.comex_hard_lambda * (hard_term * hard_weights).sum() / (hard_weights.sum() * pred.shape[1] + 1e-6)
        if not config.comex_no_diversity:
            pred_err = (pred - target).reshape(pred.shape[0], pred.shape[1], -1)
            m1_err = (batch["m1_expert_mm"] - target).reshape(pred.shape[0], pred.shape[1], -1)
            c3_err = (batch["c3_expert_mm"] - target).reshape(pred.shape[0], pred.shape[1], -1)
            div_pen = F.relu(F.cosine_similarity(pred_err, m1_err, dim=-1)).mean() + F.relu(F.cosine_similarity(pred_err, c3_err, dim=-1)).mean()
            loss = loss + config.comex_div_lambda * div_pen
    model_mse = masked_mse(pred, target, mask)
    pers_mse = masked_mse(batch["persistence_mm"], target, mask)
    skill_mse = 1.0 - float(model_mse.detach().item() / max(pers_mse.detach().item(), 1e-6))
    return loss, skill_mse


def run_epoch_v8(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer | None, scaler: amp.GradScaler | None, config: V8Config, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    skills: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            pred, aux = model(batch, return_aux=True)
            loss, skill_mse = compute_loss_v8(pred, batch, aux, config)
        if training:
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
        skills.append(float(skill_mse))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "skill_mse": float(np.mean(skills)) if skills else float("nan"),
    }


def collect_predictions_v8(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    pers_list: list[np.ndarray] = []
    ls_list: list[np.ndarray] = []
    scale_list: list[np.ndarray] = []
    delta_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    regime_list: list[np.ndarray] = []
    current_frame_list: list[np.ndarray] = []
    target_index_list: list[np.ndarray] = []
    state_norm_list: list[np.ndarray] = []
    spatial_norm_list: list[np.ndarray] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred, aux = model(batch, return_aux=True)
            pred_list.append(pred.cpu().numpy()[:, :, 0].astype(np.float32))
            target_list.append(batch["target_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            pers_list.append(batch["persistence_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            ls_list.append(batch["linear_seasonal_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            scale_list.append(aux["scale_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            delta_list.append(aux["pred_delta_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            mask_list.append(batch["mask"].cpu().numpy().astype(np.float32))
            regime_list.append(batch["regime"].cpu().numpy().astype(np.int16))
            current_frame_list.append(batch["current_frame"].cpu().numpy().astype(np.int64))
            target_index_list.append(batch["target_indices"].cpu().numpy().astype(np.int64))
            state_norm_list.append(aux["state_norm"].cpu().numpy().astype(np.float32))
            spatial_norm_list.append(aux["spatial_norm"].cpu().numpy().astype(np.float32))
    infer_seconds = time.time() - start
    arrays = {
        "y_pred_mm": np.concatenate(pred_list, axis=0),
        "y_true_mm": np.concatenate(target_list, axis=0),
        "persistence_mm": np.concatenate(pers_list, axis=0),
        "linear_seasonal_mm": np.concatenate(ls_list, axis=0),
        "scale_mm": np.concatenate(scale_list, axis=0),
        "pred_delta_mm": np.concatenate(delta_list, axis=0),
        "valid_mask": mask_list[0][0] if mask_list and mask_list[0].ndim == 3 else mask_list[0],
        "regime_map": regime_list[0][0] if regime_list and regime_list[0].ndim == 3 else regime_list[0],
        "current_frames": np.concatenate(current_frame_list, axis=0),
        "target_indices": np.concatenate(target_index_list, axis=0),
        "state_norm": np.concatenate(state_norm_list, axis=0),
        "spatial_norm": np.concatenate(spatial_norm_list, axis=0),
    }
    return arrays, infer_seconds


def peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def train_eval_v8_single_tile(bundle: TileBundle, config: V8Config, output_dir: Path, phase: str, reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(output_dir)
    config_path = output_dir / "config.json"
    result_path = output_dir / "result.json"
    val_pred_path = output_dir / "val_predictions.npz"
    test_pred_path = output_dir / "test_predictions.npz"
    ckpt_path = output_dir / "model.pth"
    if reuse_if_exists and all(path.exists() for path in [result_path, val_pred_path, test_pred_path, ckpt_path]):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        val_arrays = np.load(val_pred_path)
        test_arrays = np.load(test_pred_path)
        return {
            "result": result,
            "val_arrays": {name: val_arrays[name] for name in val_arrays.files},
            "test_arrays": {name: test_arrays[name] for name in test_arrays.files},
        }

    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_pack: dict[str, dict[str, np.ndarray]] | None = None
    if config.family == "CoMEx":
        expert_pack = build_comex_feature_pack({bundle.tile_id: bundle}, bundle.tile_id, config.seed)
    train_ds = V8Dataset(bundle, "train", config, None if expert_pack is None else expert_pack["train"])
    val_ds = V8Dataset(bundle, "val", config, None if expert_pack is None else expert_pack["val"])
    test_ds = V8Dataset(bundle, "test", config, None if expert_pack is None else expert_pack["test"])
    model = build_model(config).to(device)
    model = maybe_compile(model, config.compile_model)
    param_count = count_parameters(model)
    train_loader = DataLoader(train_ds, **loader_kwargs(config, shuffle=True))
    val_loader = DataLoader(val_ds, **loader_kwargs(config, shuffle=False))
    test_loader = DataLoader(test_ds, **loader_kwargs(config, shuffle=False))
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch_v8(model, train_loader, optimizer, scaler, config, device)
        val_stats = run_epoch_v8(model, val_loader, None, None, config, device)
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
    val_arrays, val_infer = collect_predictions_v8(model, val_loader, device)
    test_arrays, test_infer = collect_predictions_v8(model, test_loader, device)
    elapsed_seconds = time.time() - start_time
    peak_mem = peak_gpu_memory_mb(device)
    for arr in [val_arrays, test_arrays]:
        arr["peak_gpu_memory_mb"] = np.array([peak_mem], dtype=np.float32)
    val_rows = rows_from_arrays(config.model_id, bundle.tile_id, config.seed, f"{phase}_val", val_arrays, elapsed_seconds, val_infer, param_count)
    test_rows = rows_from_arrays(config.model_id, bundle.tile_id, config.seed, f"{phase}_test", test_arrays, elapsed_seconds, test_infer, param_count)
    val_rows["Peak GPU memory [MB]"] = peak_mem
    test_rows["Peak GPU memory [MB]"] = peak_mem
    result = {
        "model_id": config.model_id,
        "family": config.family,
        "tile": bundle.tile_id,
        "seed": config.seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": elapsed_seconds,
        "val_infer_seconds": val_infer,
        "test_infer_seconds": test_infer,
        "param_count": param_count,
        "peak_gpu_memory_mb": peak_mem,
        "history": history,
        "val_rows": val_rows.to_dict(orient="records"),
        "test_rows": test_rows.to_dict(orient="records"),
        "mean_skill_mse": float(test_rows["Skill_MSE"].mean()),
    }
    np.savez_compressed(val_pred_path, **val_arrays)
    np.savez_compressed(test_pred_path, **test_arrays)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"result": result, "val_arrays": val_arrays, "test_arrays": test_arrays}


def make_augmented_cache(base_cache: dict[str, np.ndarray], candidate_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not np.array_equal(base_cache["current_frames"].reshape(-1), candidate_arrays["current_frames"].reshape(-1)):
        raise ValueError("current_frames mismatch between base cache and candidate arrays")
    if not np.array_equal(base_cache["target_indices"], candidate_arrays["target_indices"]):
        raise ValueError("target_indices mismatch between base cache and candidate arrays")
    out = dict(base_cache)
    out["candidate_pred_mm"] = candidate_arrays["y_pred_mm"].astype(np.float32)
    return out


def expert_stack_from_augmented_cache(cache: dict[str, np.ndarray], include_candidate: bool) -> np.ndarray:
    items = [
        cache["persistence_mm"],
        cache["linear_seasonal_mm"],
        cache["m1_pred_mm"],
        cache["c3_pred_mm"],
    ]
    if include_candidate:
        items.append(cache["candidate_pred_mm"])
    return np.stack(items, axis=2).astype(np.float32)


def quick_gate_record(candidate_name: str, family: str, alone_summary: pd.DataFrame, add_summary: pd.DataFrame, m1_ref: pd.DataFrame, vcea_ref: pd.DataFrame) -> dict[str, Any]:
    alone_stats = compare_summary_to_reference(alone_summary, m1_ref, SEARCH_TILES)
    add_stats = compare_summary_to_reference(add_summary, vcea_ref, SEARCH_TILES)
    alone_pass = alone_stats["gain_vs_ref"] >= 0.02 and alone_stats["better_tiles_vs_ref"] >= 2 and alone_stats["stress_ratio"] <= 1.05
    add_pass = add_stats["gain_vs_ref"] >= 0.015 and add_stats["better_tiles_vs_ref"] >= 2 and add_stats["stress_ratio"] <= 1.05
    if add_pass:
        better_tiles = add_stats["better_tiles_vs_ref"]
        stress_ratio = add_stats["stress_ratio"]
    else:
        better_tiles = alone_stats["better_tiles_vs_ref"]
        stress_ratio = alone_stats["stress_ratio"]
    return {
        "Model": candidate_name,
        "Family": family,
        "Candidate Mean Skill_MSE": alone_stats["mean_skill_mse"],
        "Candidate gain vs M1": alone_stats["gain_vs_ref"],
        "VCEA+candidate Mean Skill_MSE": add_stats["mean_skill_mse"],
        "VCEA+candidate gain vs VCEA": add_stats["gain_vs_ref"],
        "Better quick tiles": int(max(alone_stats["better_tiles_vs_ref"], add_stats["better_tiles_vs_ref"])),
        "Stress ratio": float(min(alone_stats["stress_ratio"], add_stats["stress_ratio"])),
        "Candidate stress ratio": alone_stats["stress_ratio"],
        "Addition stress ratio": add_stats["stress_ratio"],
        "Candidate mean train seconds": alone_stats["mean_train_seconds"],
        "Quick gate pass": bool(alone_pass or add_pass),
        "Pass branch": "candidate" if alone_pass and not add_pass else "addition" if add_pass else "none",
        "Tile gate count": int(better_tiles),
        "Gate stress source": stress_ratio,
    }


def default_str_config() -> V8Config:
    return V8Config(
        model_id="STR_Mamba",
        family="STR-Mamba",
        dim=64,
        batch_size=8,
        epochs=10,
        patience=3,
        temporal_depth=1,
    )


def kino_quick_configs() -> list[V8Config]:
    configs: list[V8Config] = []
    for modes in [8, 12]:
        for no_bound in [False, True]:
            for operator_only in [False, True]:
                tag = "local" if not operator_only else "oponly"
                bound_tag = "bound" if not no_bound else "nobound"
                configs.append(
                    V8Config(
                        model_id=f"KINO_k{modes}_{tag}_{bound_tag}",
                        family="KINO",
                        dim=48,
                        batch_size=8,
                        epochs=10,
                        patience=3,
                        kino_modes=modes,
                        kino_operator_only=operator_only,
                        no_bounded_head=no_bound,
                    )
                )
    return configs


def comex_quick_configs() -> list[V8Config]:
    return [
        V8Config(
            model_id="CoMEx_dualhead",
            family="CoMEx",
            dim=48,
            batch_size=8,
            epochs=10,
            patience=3,
            comex_hard_lambda=0.08,
            comex_div_lambda=0.01,
        ),
        V8Config(
            model_id="CoMEx_dualhead_nodiv",
            family="CoMEx",
            dim=48,
            batch_size=8,
            epochs=10,
            patience=3,
            comex_hard_lambda=0.08,
            comex_div_lambda=0.0,
            comex_no_diversity=True,
        ),
    ]


def candidate_run_paths(phase: str, config: V8Config, tile: str, seed: int) -> Path:
    return V8_RUN_ROOT / phase / config.model_id / tile / f"seed_{seed}"


def aggregate_addition_rows(model_id: str, tile: str, seed: int, cache: dict[str, np.ndarray], pred: np.ndarray, elapsed: float = 0.0) -> pd.DataFrame:
    arrays = {
        "y_pred_mm": pred.astype(np.float32),
        "y_true_mm": cache["y_true_mm"].astype(np.float32),
        "persistence_mm": cache["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
        "valid_mask": cache["valid_mask"].astype(np.float32),
        "regime_map": cache["regime_map"].astype(np.int16),
        "current_frames": cache["current_frames"].astype(np.int64),
        "target_indices": cache["target_indices"].astype(np.int64),
    }
    return rows_from_arrays(model_id, tile, seed, "test_addition", arrays, elapsed=elapsed, infer_seconds=elapsed, param_count=0)


def run_candidate_quick(
    bundles: dict[str, TileBundle],
    config: V8Config,
    m1_ref: pd.DataFrame,
    vcea_ref: pd.DataFrame,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    test_rows: list[pd.DataFrame] = []
    add_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    static_cfg = best_static_cfg()
    for tile in SEARCH_TILES:
        run = train_eval_v8_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=candidate_run_paths("quick", config, tile, config.seed),
            phase="quick",
            reuse_if_exists=reuse_if_exists,
        )
        test_arrays = run["test_arrays"]
        val_arrays = run["val_arrays"]
        test_df = pd.DataFrame(run["result"]["test_rows"])
        test_rows.append(test_df)
        runtime_rows.append(
            {
                "Model": config.model_id,
                "Family": config.family,
                "Tile": tile,
                "Seed": config.seed,
                "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                "Infer seconds": float(run["result"]["test_infer_seconds"]),
                "Param count": int(run["result"]["param_count"]),
                "Peak GPU memory [MB]": float(run["result"]["peak_gpu_memory_mb"]),
            }
        )
        base_val = build_seed_expert_cache(bundles, tile, config.seed, "val")
        base_test = build_seed_expert_cache(bundles, tile, config.seed, "test")
        val_aug = make_augmented_cache(base_val, val_arrays)
        test_aug = make_augmented_cache(base_test, test_arrays)
        val_experts = expert_stack_from_augmented_cache(val_aug, include_candidate=True)
        test_experts = expert_stack_from_augmented_cache(test_aug, include_candidate=True)
        weights = estimate_static_weights_generic(val_aug["y_true_mm"], val_experts, val_aug["valid_mask"], static_cfg)
        pred_add = predict_static_generic(test_experts, weights)
        add_rows.append(aggregate_addition_rows(f"VCEA+{config.model_id}", tile, config.seed, test_aug, pred_add))
    raw_test = pd.concat(test_rows, ignore_index=True)
    raw_add = pd.concat(add_rows, ignore_index=True)
    summary_test = summarize_seeded_rows(raw_test)
    summary_add = summarize_seeded_rows(raw_add)
    quick_stats = quick_gate_record(config.model_id, config.family, summary_test, summary_add, m1_ref, vcea_ref)
    return raw_test, summary_test, summary_add, {"runtime": pd.DataFrame(runtime_rows), "quick_stats": quick_stats}


def choose_best_candidate_by_family(
    quick_records: list[dict[str, Any]],
    candidate_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    family: str,
) -> tuple[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    family_rows = [row for row in quick_records if row["Family"] == family]
    if not family_rows:
        raise ValueError(f"No {family} rows available")
    family_rows = sorted(
        family_rows,
        key=lambda row: (
            bool(row["Quick gate pass"]),
            float(max(row["Candidate gain vs M1"], row["VCEA+candidate gain vs VCEA"])),
            -float(min(row["Candidate stress ratio"], row["Addition stress ratio"])),
        ),
        reverse=True,
    )
    best_id = str(family_rows[0]["Model"])
    return best_id, candidate_frames[best_id]


def run_fullsix_for_candidate(
    bundles: dict[str, TileBundle],
    config: V8Config,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_rows: list[pd.DataFrame] = []
    add_rows: list[pd.DataFrame] = []
    safe_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    static_cfg = best_static_cfg()
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            cfg = replace(config, seed=seed, phase_tag="fullsix")
            run = train_eval_v8_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=candidate_run_paths("fullsix", cfg, tile, seed),
                phase="fullsix",
                reuse_if_exists=reuse_if_exists,
            )
            test_arrays = run["test_arrays"]
            val_arrays = run["val_arrays"]
            candidate_rows.append(pd.DataFrame(run["result"]["test_rows"]))
            runtime_rows.append(
                {
                    "Model": config.model_id,
                    "Family": config.family,
                    "Tile": tile,
                    "Seed": seed,
                    "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                    "Infer seconds": float(run["result"]["test_infer_seconds"]),
                    "Param count": int(run["result"]["param_count"]),
                    "Peak GPU memory [MB]": float(run["result"]["peak_gpu_memory_mb"]),
                }
            )
            base_val = build_seed_expert_cache(bundles, tile, seed, "val")
            base_test = build_seed_expert_cache(bundles, tile, seed, "test")
            val_aug = make_augmented_cache(base_val, val_arrays)
            test_aug = make_augmented_cache(base_test, test_arrays)
            val_experts = expert_stack_from_augmented_cache(val_aug, include_candidate=True)
            test_experts = expert_stack_from_augmented_cache(test_aug, include_candidate=True)
            weights = estimate_static_weights_generic(val_aug["y_true_mm"], val_experts, val_aug["valid_mask"], static_cfg)
            pred_add = predict_static_generic(test_experts, weights)
            add_rows.append(aggregate_addition_rows(f"VCEA+{config.model_id}", tile, seed, test_aug, pred_add))
            val_pred = predict_static_generic(val_experts, weights)
            val_disagree = val_experts.std(axis=2).astype(np.float32)
            test_disagree = test_experts.std(axis=2).astype(np.float32)
            safe_tau = calibrate_safe_thresholds(val_pred, val_aug["m1_pred_mm"], val_aug["y_true_mm"], val_disagree, val_aug["valid_mask"])
            pred_safe = apply_safe_wrapper(pred_add, test_aug["m1_pred_mm"], test_disagree, safe_tau)
            safe_rows.append(aggregate_addition_rows(f"VCEA-safe+{config.model_id}", tile, seed, test_aug, pred_safe))
    raw_candidate = pd.concat(candidate_rows, ignore_index=True)
    raw_add = pd.concat(add_rows, ignore_index=True)
    raw_safe = pd.concat(safe_rows, ignore_index=True)
    return raw_candidate, summarize_seeded_rows(raw_candidate), summarize_seeded_rows(raw_add), summarize_seeded_rows(raw_safe), pd.DataFrame(runtime_rows)


def load_m1_full6_summary() -> pd.DataFrame:
    return pd.read_csv(M1_FULL6_SUMMARY_PATH)


def load_vcea_full6_summary() -> pd.DataFrame:
    df = pd.read_csv(VCEA_FINAL_COMPARISON_PATH)
    value = float(df[df["Model"] == "VCEA"]["Mean Skill_MSE"].iloc[0])
    out_rows = []
    m1 = load_m1_full6_summary()
    for tile in TILE_ORDER:
        for horizon in [f"t+{h}" for h in SEARCH_HORIZONS]:
            out_rows.append(
                {
                    "Model": "VCEA",
                    "Tile": tile,
                    "Horizon": horizon,
                    "Skill_MSE mean": value,
                    "Skill_MSE std": 0.010451,
                    "RMSE [mm] mean": float("nan"),
                    "Persistence RMSE [mm] mean": safe_mean(m1[(m1["Tile"] == tile) & (m1["Horizon"] == horizon)]["Persistence RMSE [mm] mean"]),
                    "Elapsed seconds mean": 0.0,
                }
            )
    return pd.DataFrame(out_rows)


def evaluate_fullsix_outcome(candidate_summary: pd.DataFrame, add_summary: pd.DataFrame, safe_summary: pd.DataFrame, runtime_df: pd.DataFrame) -> dict[str, Any]:
    m1_summary = load_m1_full6_summary()
    vcea_mean = float(pd.read_csv(VCEA_FINAL_COMPARISON_PATH).query("Model == 'VCEA'")["Mean Skill_MSE"].iloc[0])
    vcea_stress = float(pd.read_csv(VCEA_FINAL_COMPARISON_PATH).query("Model == 'VCEA'")["Stress ratio"].iloc[0])
    cand_stats = compare_summary_to_reference(candidate_summary, m1_summary, TILE_ORDER)
    add_mean = float(add_summary["Skill_MSE mean"].mean())
    safe_mean = float(safe_summary["Skill_MSE mean"].mean())
    add_vs_vcea = add_mean - vcea_mean
    safe_vs_vcea = safe_mean - vcea_mean
    add_tile = (
        add_summary.groupby("Tile", as_index=False)["Skill_MSE mean"]
        .mean()
        .rename(columns={"Skill_MSE mean": "Add Skill_MSE mean"})
    )
    best_tile_gain_vs_vcea = float(add_tile["Add Skill_MSE mean"].mean() - vcea_mean)
    add_stress = float((add_summary[add_summary["Tile"].isin(STRESS_TILES)]["RMSE [mm] mean"] / add_summary[add_summary["Tile"].isin(STRESS_TILES)]["Persistence RMSE [mm] mean"]).max())
    safe_stress = float((safe_summary[safe_summary["Tile"].isin(STRESS_TILES)]["RMSE [mm] mean"] / safe_summary[safe_summary["Tile"].isin(STRESS_TILES)]["Persistence RMSE [mm] mean"]).max())
    runtime_mean = float(runtime_df["Elapsed seconds"].mean()) if not runtime_df.empty else float("nan")
    verdict = "supplementary_only"
    if cand_stats["mean_skill_mse"] >= vcea_mean - 0.02 and add_vs_vcea >= 0.02 and safe_stress <= 1.03:
        verdict = "main_model_plus_expert"
    elif cand_stats["mean_skill_mse"] >= 0.0783341230429559 + 0.05 and add_vs_vcea >= 0.02:
        verdict = "strong_new_expert"
    elif cand_stats["mean_skill_mse"] > 0.0783341230429559 and add_vs_vcea < 0.005:
        verdict = "supplementary_only"
    if cand_stats["stress_ratio"] > 1.05 or add_stress > 1.05:
        verdict = "no_go"
    return {
        "candidate_model": str(candidate_summary["Model"].iloc[0]),
        "candidate_mean_skill_mse": cand_stats["mean_skill_mse"],
        "candidate_gain_vs_m1": cand_stats["gain_vs_ref"],
        "candidate_better_tiles_vs_m1": cand_stats["better_tiles_vs_ref"],
        "candidate_stress_ratio": cand_stats["stress_ratio"],
        "candidate_mean_skill_std": cand_stats["mean_skill_std"],
        "vcea_mean_skill_mse": vcea_mean,
        "vcea_stress_ratio": vcea_stress,
        "vcea_plus_candidate_mean_skill_mse": add_mean,
        "vcea_plus_candidate_gain_vs_vcea": add_vs_vcea,
        "vcea_safe_plus_candidate_mean_skill_mse": safe_mean,
        "vcea_safe_plus_candidate_gain_vs_vcea": safe_vs_vcea,
        "vcea_plus_candidate_stress_ratio": add_stress,
        "vcea_safe_plus_candidate_stress_ratio": safe_stress,
        "mean_train_seconds": runtime_mean,
        "fullsix_verdict": verdict,
        "best_tile_gain_vs_vcea_proxy": best_tile_gain_vs_vcea,
    }


def plot_quick_heatmap(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    pivot = summary_df.pivot(index="Tile", columns="Horizon", values="Skill_MSE mean").reindex(SEARCH_TILES)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_decision_day8(quick_df: pd.DataFrame, selected_models: list[str], used_comex: bool) -> Path:
    lines = [
        "# Day-8 Quick Decision",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，搜索子集固定为 `E32N34 / E39N30 / E48N24`，原因是数据获取限制。",
        "",
        "## Quick Rank",
        "",
    ]
    for _, row in quick_df.iterrows():
        lines.append(
            f"- `{row['Model']}` ({row['Family']}): candidate gain vs M1={row['Candidate gain vs M1']:.4f}, "
            f"VCEA+candidate gain vs VCEA={row['VCEA+candidate gain vs VCEA']:.4f}, "
            f"better tiles={int(row['Better quick tiles'])}/3, stress ratio={row['Stress ratio']:.4f}, "
            f"quick gate={bool(row['Quick gate pass'])}, pass branch=`{row['Pass branch']}`。"
        )
    lines.extend(["", "## Decision", ""])
    if selected_models:
        lines.append(f"- 进入 full-six 的候选：{', '.join(selected_models)}。")
    elif used_comex:
        lines.append("- STR-Mamba 与 KINO 都未过 quick gate，按决策树已切到 CoMEx fallback。")
    else:
        lines.append("- STR-Mamba 与 KINO 都未过 quick gate，且本轮尚未切到 CoMEx。")
    return write_markdown(ACT_V8_ROOT / "decision_day8.md", lines)


def write_final_report(
    quick_df: pd.DataFrame,
    quick_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    fullsix_rows: list[dict[str, Any]],
    assumptions: list[str],
    final_verdict: str,
) -> Path:
    lines = [
        "# ACT-Former V8 完整实验探索报告",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 决策树执行说明",
        "",
        "- 先运行 `STR-Mamba` 与 `KINO` quick；只有二者都不过 quick gate 时才切到 `CoMEx`。",
        "- quick gate 口径采用文档原要求：`candidate alone gain vs M1 >= +0.02` 或 `VCEA+candidate gain vs VCEA >= +0.015`，并要求 `stress ratio <= 1.05` 且 `better quick tiles >= 2/3`。",
        "- full-six 只看 3 seeds mean/std，不接受 best-seed claim。",
        "",
        "## 本轮假设",
        "",
    ]
    for item in assumptions:
        lines.append(f"- {item}")
    lines.extend(["", "## Quick 结果", ""])
    for _, row in quick_df.iterrows():
        lines.append(
            f"- `{row['Model']}` ({row['Family']}): candidate mean Skill_MSE={row['Candidate Mean Skill_MSE']:.4f}, "
            f"candidate gain vs M1={row['Candidate gain vs M1']:.4f}, "
            f"`VCEA+candidate` gain vs VCEA={row['VCEA+candidate gain vs VCEA']:.4f}, "
            f"stress ratio={row['Stress ratio']:.4f}, quick gate={bool(row['Quick gate pass'])}。"
        )
    if fullsix_rows:
        lines.extend(["", "## Full-six 结果", ""])
        for item in fullsix_rows:
            lines.append(
                f"- `{item['candidate_model']}`: alone mean Skill_MSE={item['candidate_mean_skill_mse']:.4f}, "
                f"gain vs M1={item['candidate_gain_vs_m1']:.4f}, "
                f"`VCEA+candidate` mean Skill_MSE={item['vcea_plus_candidate_mean_skill_mse']:.4f}, "
                f"gain vs VCEA={item['vcea_plus_candidate_gain_vs_vcea']:.4f}, "
                f"`VCEA-safe+candidate` mean Skill_MSE={item['vcea_safe_plus_candidate_mean_skill_mse']:.4f}, "
                f"safe gain vs VCEA={item['vcea_safe_plus_candidate_gain_vs_vcea']:.4f}, "
                f"candidate stress={item['candidate_stress_ratio']:.4f}, add stress={item['vcea_plus_candidate_stress_ratio']:.4f}, "
                f"safe stress={item['vcea_safe_plus_candidate_stress_ratio']:.4f}, "
                f"3-seed verdict=`{item['fullsix_verdict']}`。"
            )
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            f"- final verdict = `{final_verdict}`。",
            "",
            "## 关键文件",
            "",
            f"- quick 排名：`{(ACT_V8_ROOT / 'quick_rank_table.csv').as_posix()}`",
            f"- day-8 决策：`{(ACT_V8_ROOT / 'decision_day8.md').as_posix()}`",
            f"- full-six 汇总：`{(ACT_V8_ROOT / 'fullsix_eval.csv').as_posix()}`",
            f"- 最终决策：`{(ACT_V8_ROOT / 'v8_final_decision.md').as_posix()}`",
        ]
    )
    return write_markdown(ACT_V8_ROOT / "act_v8_detailed_report.md", lines)


def run_act_v8_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    assumptions = [
        "由于当前环境未安装专用 Mamba kernel，本轮 `STR-Mamba` 的 spatial scan 采用 GRU-based directional scan，temporal branch 采用现有 SSM-lite temporal mixer；这属于对文档中 Mamba 模块的最小可执行近似。",
        "本轮 `VCEA-safe` 没有现成仓库定义，因此额外假定为：使用 validation split 上的 expert disagreement threshold 标定后，对高风险像元回退到 M1 的 static-safe 包装。",
        "quick gate 中的 `better quick tiles >= 2/3` 采用分支内判定：若走 `candidate alone` 分支，则相对 M1 统计；若走 `VCEA+candidate` 分支，则相对 VCEA 统计。",
    ]

    assumptions = [
        "本轮若运行在可用的 Linux/WSL2 CUDA 环境，并检测到 `mamba_ssm`，则 STR-Mamba 使用真实 Mamba selective state-space block；若环境不可用，才回退到旧 proxy。",
        "本轮 VCEA-safe 没有现成仓库定义，因此额外定义为：基于 validation split 的 expert disagreement threshold 标定后，对高风险像元回退到 M1 的 static-safe 包装。",
        "quick gate 中的 better quick tiles >= 2/3 采用分支内统计：candidate alone 分支相对 M1 计算，VCEA+candidate 分支相对 VCEA 计算。",
        "CoMEx fallback 缺少 inner-split OOF expert error，因此用 train-only expert disagreement 与 M1/C3 spread 近似构造 hard-case prior；该点在报告中保留为方法假设。",
    ]
    m1_quick_ref, vcea_quick_ref, _vcea_safe_quick_ref = reference_quick_m1_vcea(bundles, seed=42)
    quick_records: list[dict[str, Any]] = []
    quick_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}

    str_cfg = default_str_config()
    str_raw, str_summary, str_add_summary, str_diag = run_candidate_quick(bundles, str_cfg, m1_quick_ref, vcea_quick_ref, reuse_if_exists=reuse_if_exists)
    quick_records.append(dict(str_diag["quick_stats"]))
    quick_outputs[str_cfg.model_id] = (str_raw, str_summary, str_add_summary, str_diag)
    str_summary.to_csv(QUICK_ROOT / "strmamba_quick_skill.csv", index=False, encoding="utf-8-sig")
    str_add_summary.to_csv(QUICK_ROOT / "strmamba_addition.csv", index=False, encoding="utf-8-sig")
    str_diag["runtime"].to_csv(QUICK_ROOT / "strmamba_runtime_memory.csv", index=False, encoding="utf-8-sig")
    plot_quick_heatmap(str_summary, FIG_ROOT / "strmamba_quick_heatmap.png", "STR-Mamba quick Skill_MSE")

    for cfg in kino_quick_configs():
        raw, summary, add_summary, diag = run_candidate_quick(bundles, cfg, m1_quick_ref, vcea_quick_ref, reuse_if_exists=reuse_if_exists)
        quick_records.append(dict(diag["quick_stats"]))
        quick_outputs[cfg.model_id] = (raw, summary, add_summary, diag)

    quick_df = pd.DataFrame(quick_records).sort_values(
        ["Quick gate pass", "VCEA+candidate gain vs VCEA", "Candidate gain vs M1", "Stress ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    quick_df.to_csv(ACT_V8_ROOT / "quick_rank_table.csv", index=False, encoding="utf-8-sig")

    best_kino_id, best_kino_outputs = choose_best_kino(quick_records, quick_outputs)
    kino_summary = best_kino_outputs[1]
    kino_add_summary = best_kino_outputs[2]
    best_kino_outputs[3]["runtime"].to_csv(QUICK_ROOT / "kino_runtime_memory.csv", index=False, encoding="utf-8-sig")
    kino_summary.to_csv(QUICK_ROOT / "kino_quick_skill.csv", index=False, encoding="utf-8-sig")
    kino_add_summary.to_csv(QUICK_ROOT / "kino_addition.csv", index=False, encoding="utf-8-sig")
    plot_quick_heatmap(kino_summary, FIG_ROOT / "kino_quick_heatmap.png", "KINO quick Skill_MSE")

    selected_models = quick_df[quick_df["Quick gate pass"]]["Model"].tolist()
    used_comex = False
    write_decision_day8(quick_df, selected_models, used_comex)

    fullsix_rows: list[dict[str, Any]] = []
    fullsix_eval_rows: list[dict[str, Any]] = []
    for model_id in selected_models:
        if model_id == str_cfg.model_id:
            cfg = str_cfg
        elif model_id == best_kino_id:
            cfg = next(cfg for cfg in kino_quick_configs() if cfg.model_id == best_kino_id)
        else:
            continue
        raw_candidate, candidate_summary, add_summary, safe_summary, runtime_df = run_fullsix_for_candidate(bundles, cfg, reuse_if_exists=reuse_if_exists)
        candidate_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_alone_summary.csv", index=False, encoding="utf-8-sig")
        add_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_vcea_plus_summary.csv", index=False, encoding="utf-8-sig")
        safe_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_vcea_safe_plus_summary.csv", index=False, encoding="utf-8-sig")
        runtime_df.to_csv(FULL6_ROOT / f"{cfg.model_id}_runtime_memory.csv", index=False, encoding="utf-8-sig")
        outcome = evaluate_fullsix_outcome(candidate_summary, add_summary, safe_summary, runtime_df)
        fullsix_rows.append(outcome)
        fullsix_eval_rows.extend(
            [
                {"Model": cfg.model_id, "Track": "candidate_alone", **outcome},
                {"Model": f"VCEA+{cfg.model_id}", "Track": "vcea_plus", **outcome},
                {"Model": f"VCEA-safe+{cfg.model_id}", "Track": "vcea_safe_plus", **outcome},
            ]
        )

    fullsix_eval = pd.DataFrame(fullsix_eval_rows)
    fullsix_eval.to_csv(ACT_V8_ROOT / "fullsix_eval.csv", index=False, encoding="utf-8-sig")

    final_verdict = "no_go_return_to_vcea"
    if fullsix_rows:
        ranking = sorted(
            fullsix_rows,
            key=lambda item: (
                item["fullsix_verdict"] == "main_model_plus_expert",
                item["fullsix_verdict"] == "strong_new_expert",
                item["vcea_plus_candidate_gain_vs_vcea"],
                item["candidate_gain_vs_m1"],
                -item["vcea_safe_plus_candidate_stress_ratio"],
            ),
            reverse=True,
        )
        best = ranking[0]
        if best["fullsix_verdict"] == "main_model_plus_expert":
            final_verdict = f"promote_{best['candidate_model']}_as_main_model_plus_expert"
        elif best["fullsix_verdict"] == "strong_new_expert":
            final_verdict = f"keep_vcea_mainline_add_{best['candidate_model']}_as_strong_expert"
        elif best["fullsix_verdict"] == "supplementary_only":
            final_verdict = f"keep_vcea_mainline_{best['candidate_model']}_supplementary"
        else:
            final_verdict = f"no_go_{best['candidate_model']}_return_to_vcea"

    decision_lines = [
        "# V8 Final Decision",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        f"- final verdict = `{final_verdict}`。",
    ]
    if fullsix_rows:
        decision_lines.append("")
        decision_lines.append("## Full-six summary")
        decision_lines.append("")
        for item in fullsix_rows:
            decision_lines.append(
                f"- `{item['candidate_model']}`: alone={item['candidate_mean_skill_mse']:.4f}, "
                f"add gain vs VCEA={item['vcea_plus_candidate_gain_vs_vcea']:.4f}, "
                f"safe stress={item['vcea_safe_plus_candidate_stress_ratio']:.4f}, verdict=`{item['fullsix_verdict']}`。"
            )
    write_markdown(ACT_V8_ROOT / "v8_final_decision.md", decision_lines)
    report_path = write_final_report(quick_df, quick_outputs, fullsix_rows, assumptions, final_verdict)

    summary = {
        "report": str(report_path.resolve()),
        "quick_rank_table": str((ACT_V8_ROOT / "quick_rank_table.csv").resolve()),
        "decision_day8": str((ACT_V8_ROOT / "decision_day8.md").resolve()),
        "fullsix_eval": str((ACT_V8_ROOT / "fullsix_eval.csv").resolve()),
        "final_decision": str((ACT_V8_ROOT / "v8_final_decision.md").resolve()),
        "selected_quick_models": selected_models,
        "used_comex": used_comex,
        "final_verdict": final_verdict,
    }
    write_json(ACT_V8_ROOT / "run_summary.json", summary)
    return summary


def write_decision_day8(quick_df: pd.DataFrame, selected_models: list[str], used_comex: bool) -> Path:
    lines = [
        "# Day-8 Quick 决策",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，搜索子集固定为 `E32N34 / E39N30 / E48N24`，原因是数据获取限制。",
        "",
        "## Quick 排名",
        "",
    ]
    for _, row in quick_df.iterrows():
        lines.append(
            f"- `{row['Model']}` ({row['Family']}): candidate gain vs M1 = {row['Candidate gain vs M1']:.4f}, "
            f"VCEA+candidate gain vs VCEA = {row['VCEA+candidate gain vs VCEA']:.4f}, "
            f"better tiles = {int(row['Better quick tiles'])}/3, stress ratio = {row['Stress ratio']:.4f}, "
            f"quick gate = {bool(row['Quick gate pass'])}, pass branch = `{row['Pass branch']}`。"
        )
    lines.extend(["", "## 决策", ""])
    if selected_models:
        lines.append(f"- 进入 full-six 的候选：{', '.join(selected_models)}。")
    elif used_comex:
        lines.append("- STR-Mamba 与 KINO 都未通过 quick gate，已按决策树切换到 CoMEx fallback，但 CoMEx 仍未过线。")
    else:
        lines.append("- STR-Mamba 与 KINO 都未通过 quick gate，本轮尚未切到 CoMEx。")
    return write_markdown(ACT_V8_ROOT / "decision_day8.md", lines)


def write_final_report(
    quick_df: pd.DataFrame,
    quick_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    fullsix_rows: list[dict[str, Any]],
    assumptions: list[str],
    final_verdict: str,
) -> Path:
    lines = [
        "# ACT-Former V8 完整实验探索报告",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 决策树执行说明",
        "",
        "- 先运行 `STR-Mamba` 与 `KINO` quick；只有二者都未过线时才切到 `CoMEx`。",
        "- quick gate 采用文档原始条件：`candidate alone gain vs M1 >= +0.02` 或 `VCEA+candidate gain vs VCEA >= +0.015`，同时要求 `stress ratio <= 1.05` 且 `better quick tiles >= 2/3`。",
        "- full-six 只看 3 seeds 的 mean/std，不接受 best-seed claim。",
        "",
        "## 本轮假设",
        "",
    ]
    for item in assumptions:
        lines.append(f"- {item}")
    lines.extend(["", "## Quick 结果", ""])
    for _, row in quick_df.iterrows():
        lines.append(
            f"- `{row['Model']}` ({row['Family']}): candidate mean Skill_MSE = {row['Candidate Mean Skill_MSE']:.4f}, "
            f"candidate gain vs M1 = {row['Candidate gain vs M1']:.4f}, "
            f"`VCEA+candidate` gain vs VCEA = {row['VCEA+candidate gain vs VCEA']:.4f}, "
            f"stress ratio = {row['Stress ratio']:.4f}, quick gate = {bool(row['Quick gate pass'])}。"
        )
    if fullsix_rows:
        lines.extend(["", "## Full-six 结果", ""])
        for item in fullsix_rows:
            lines.append(
                f"- `{item['candidate_model']}`: alone mean Skill_MSE = {item['candidate_mean_skill_mse']:.4f}, "
                f"gain vs M1 = {item['candidate_gain_vs_m1']:.4f}, "
                f"`VCEA+candidate` mean Skill_MSE = {item['vcea_plus_candidate_mean_skill_mse']:.4f}, "
                f"gain vs VCEA = {item['vcea_plus_candidate_gain_vs_vcea']:.4f}, "
                f"`VCEA-safe+candidate` mean Skill_MSE = {item['vcea_safe_plus_candidate_mean_skill_mse']:.4f}, "
                f"safe gain vs VCEA = {item['vcea_safe_plus_candidate_gain_vs_vcea']:.4f}, "
                f"candidate stress = {item['candidate_stress_ratio']:.4f}, "
                f"add stress = {item['vcea_plus_candidate_stress_ratio']:.4f}, "
                f"safe stress = {item['vcea_safe_plus_candidate_stress_ratio']:.4f}, "
                f"3-seed verdict = `{item['fullsix_verdict']}`。"
            )
    else:
        lines.extend(["", "## Full-six 结果", "", "- 本轮没有候选通过 quick gate，因此未触发 full-six。"])
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            f"- final verdict = `{final_verdict}`。",
            "",
            "## 关键文件",
            "",
            f"- quick 排名：`{(ACT_V8_ROOT / 'quick_rank_table.csv').as_posix()}`",
            f"- day-8 决策：`{(ACT_V8_ROOT / 'decision_day8.md').as_posix()}`",
            f"- full-six 汇总：`{(ACT_V8_ROOT / 'fullsix_eval.csv').as_posix()}`",
            f"- 最终决策：`{(ACT_V8_ROOT / 'v8_final_decision.md').as_posix()}`",
        ]
    )
    return write_markdown(ACT_V8_ROOT / "act_v8_detailed_report.md", lines)


def run_act_v8_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    assumptions = [
        "由于当前环境未安装专用 Mamba kernel，本轮 STR-Mamba 的 spatial scan 采用 GRU-based directional scan，temporal branch 采用现有 SSM-lite temporal mixer，属于文档中 Mamba 模块的最小可执行近似。",
        "本轮 VCEA-safe 没有现成仓库定义，因此额外定义为：基于 validation split 的 expert disagreement threshold 标定后，对高风险像元回退到 M1 的 static-safe 包装。",
        "quick gate 中的 better quick tiles >= 2/3 采用分支内统计：candidate alone 分支相对 M1 计算，VCEA+candidate 分支相对 VCEA 计算。",
        "CoMEx fallback 缺少 inner-split OOF expert error，因此用 train-only expert disagreement 与 M1/C3 spread 近似构造 hard-case prior；该点在报告中作为方法假设保留。",
    ]
    m1_quick_ref, vcea_quick_ref, _vcea_safe_quick_ref = reference_quick_m1_vcea(bundles, seed=42)
    quick_records: list[dict[str, Any]] = []
    quick_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    config_map: dict[str, V8Config] = {}

    str_cfg = default_str_config()
    config_map[str_cfg.model_id] = str_cfg
    str_raw, str_summary, str_add_summary, str_diag = run_candidate_quick(bundles, str_cfg, m1_quick_ref, vcea_quick_ref, reuse_if_exists=reuse_if_exists)
    quick_records.append(dict(str_diag["quick_stats"]))
    quick_outputs[str_cfg.model_id] = (str_raw, str_summary, str_add_summary, str_diag)
    str_summary.to_csv(QUICK_ROOT / "strmamba_quick_skill.csv", index=False, encoding="utf-8-sig")
    str_add_summary.to_csv(QUICK_ROOT / "strmamba_addition.csv", index=False, encoding="utf-8-sig")
    str_diag["runtime"].to_csv(QUICK_ROOT / "strmamba_runtime_memory.csv", index=False, encoding="utf-8-sig")
    plot_quick_heatmap(str_summary, FIG_ROOT / "strmamba_quick_heatmap.png", "STR-Mamba quick Skill_MSE")

    for cfg in kino_quick_configs():
        config_map[cfg.model_id] = cfg
        raw, summary, add_summary, diag = run_candidate_quick(bundles, cfg, m1_quick_ref, vcea_quick_ref, reuse_if_exists=reuse_if_exists)
        quick_records.append(dict(diag["quick_stats"]))
        quick_outputs[cfg.model_id] = (raw, summary, add_summary, diag)

    best_kino_id, best_kino_outputs = choose_best_candidate_by_family(quick_records, quick_outputs, "KINO")
    kino_summary = best_kino_outputs[1]
    kino_add_summary = best_kino_outputs[2]
    best_kino_outputs[3]["runtime"].to_csv(QUICK_ROOT / "kino_runtime_memory.csv", index=False, encoding="utf-8-sig")
    kino_summary.to_csv(QUICK_ROOT / "kino_quick_skill.csv", index=False, encoding="utf-8-sig")
    kino_add_summary.to_csv(QUICK_ROOT / "kino_addition.csv", index=False, encoding="utf-8-sig")
    plot_quick_heatmap(kino_summary, FIG_ROOT / "kino_quick_heatmap.png", "KINO quick Skill_MSE")

    used_comex = False
    selected_models = [row["Model"] for row in quick_records if bool(row["Quick gate pass"])]
    if not selected_models:
        used_comex = True
        for cfg in comex_quick_configs():
            config_map[cfg.model_id] = cfg
            raw, summary, add_summary, diag = run_candidate_quick(bundles, cfg, m1_quick_ref, vcea_quick_ref, reuse_if_exists=reuse_if_exists)
            quick_records.append(dict(diag["quick_stats"]))
            quick_outputs[cfg.model_id] = (raw, summary, add_summary, diag)
        best_comex_id, best_comex_outputs = choose_best_candidate_by_family(quick_records, quick_outputs, "CoMEx")
        comex_summary = best_comex_outputs[1]
        comex_add_summary = best_comex_outputs[2]
        best_comex_outputs[3]["runtime"].to_csv(QUICK_ROOT / "comex_runtime_memory.csv", index=False, encoding="utf-8-sig")
        comex_summary.to_csv(QUICK_ROOT / "comex_quick_skill.csv", index=False, encoding="utf-8-sig")
        comex_add_summary.to_csv(QUICK_ROOT / "comex_addition.csv", index=False, encoding="utf-8-sig")
        plot_quick_heatmap(comex_summary, FIG_ROOT / "comex_quick_heatmap.png", "CoMEx quick Skill_MSE")
        selected_models = [
            row["Model"]
            for row in quick_records
            if bool(row["Quick gate pass"]) and str(row["Family"]) == "CoMEx"
        ]

    quick_df = pd.DataFrame(quick_records).sort_values(
        ["Quick gate pass", "VCEA+candidate gain vs VCEA", "Candidate gain vs M1", "Stress ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    quick_df.to_csv(ACT_V8_ROOT / "quick_rank_table.csv", index=False, encoding="utf-8-sig")
    write_decision_day8(quick_df, selected_models, used_comex)

    fullsix_rows: list[dict[str, Any]] = []
    fullsix_eval_rows: list[dict[str, Any]] = []
    for model_id in selected_models:
        cfg = config_map.get(model_id)
        if cfg is None:
            continue
        raw_candidate, candidate_summary, add_summary, safe_summary, runtime_df = run_fullsix_for_candidate(bundles, cfg, reuse_if_exists=reuse_if_exists)
        candidate_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_alone_summary.csv", index=False, encoding="utf-8-sig")
        add_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_vcea_plus_summary.csv", index=False, encoding="utf-8-sig")
        safe_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_vcea_safe_plus_summary.csv", index=False, encoding="utf-8-sig")
        runtime_df.to_csv(FULL6_ROOT / f"{cfg.model_id}_runtime_memory.csv", index=False, encoding="utf-8-sig")
        outcome = evaluate_fullsix_outcome(candidate_summary, add_summary, safe_summary, runtime_df)
        fullsix_rows.append(outcome)
        fullsix_eval_rows.extend(
            [
                {"Model": cfg.model_id, "Track": "candidate_alone", **outcome},
                {"Model": f"VCEA+{cfg.model_id}", "Track": "vcea_plus", **outcome},
                {"Model": f"VCEA-safe+{cfg.model_id}", "Track": "vcea_safe_plus", **outcome},
            ]
        )

    fullsix_eval = pd.DataFrame(fullsix_eval_rows)
    fullsix_eval.to_csv(ACT_V8_ROOT / "fullsix_eval.csv", index=False, encoding="utf-8-sig")

    final_verdict = "no_go_return_to_vcea"
    if fullsix_rows:
        ranking = sorted(
            fullsix_rows,
            key=lambda item: (
                item["fullsix_verdict"] == "main_model_plus_expert",
                item["fullsix_verdict"] == "strong_new_expert",
                item["vcea_plus_candidate_gain_vs_vcea"],
                item["candidate_gain_vs_m1"],
                -item["vcea_safe_plus_candidate_stress_ratio"],
            ),
            reverse=True,
        )
        best = ranking[0]
        if best["fullsix_verdict"] == "main_model_plus_expert":
            final_verdict = f"promote_{best['candidate_model']}_as_main_model_plus_expert"
        elif best["fullsix_verdict"] == "strong_new_expert":
            final_verdict = f"keep_vcea_mainline_add_{best['candidate_model']}_as_strong_expert"
        elif best["fullsix_verdict"] == "supplementary_only":
            final_verdict = f"keep_vcea_mainline_{best['candidate_model']}_supplementary"
        else:
            final_verdict = f"no_go_{best['candidate_model']}_return_to_vcea"

    decision_lines = [
        "# V8 Final Decision",
        "",
        "数据声明：本轮 V8 继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        f"- final verdict = `{final_verdict}`。",
    ]
    if fullsix_rows:
        decision_lines.extend(["", "## Full-six summary", ""])
        for item in fullsix_rows:
            decision_lines.append(
                f"- `{item['candidate_model']}`: alone = {item['candidate_mean_skill_mse']:.4f}, "
                f"add gain vs VCEA = {item['vcea_plus_candidate_gain_vs_vcea']:.4f}, "
                f"safe stress = {item['vcea_safe_plus_candidate_stress_ratio']:.4f}, "
                f"verdict = `{item['fullsix_verdict']}`。"
            )
    write_markdown(ACT_V8_ROOT / "v8_final_decision.md", decision_lines)
    report_path = write_final_report(quick_df, quick_outputs, fullsix_rows, assumptions, final_verdict)

    summary = {
        "report": str(report_path.resolve()),
        "quick_rank_table": str((ACT_V8_ROOT / "quick_rank_table.csv").resolve()),
        "decision_day8": str((ACT_V8_ROOT / "decision_day8.md").resolve()),
        "fullsix_eval": str((ACT_V8_ROOT / "fullsix_eval.csv").resolve()),
        "final_decision": str((ACT_V8_ROOT / "v8_final_decision.md").resolve()),
        "selected_quick_models": selected_models,
        "used_comex": used_comex,
        "final_verdict": final_verdict,
    }
    write_json(ACT_V8_ROOT / "run_summary.json", summary)
    return summary
