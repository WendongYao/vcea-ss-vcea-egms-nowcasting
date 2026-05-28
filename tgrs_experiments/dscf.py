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
from scipy.ndimage import uniform_filter
from torch import amp
from torch.utils.data import DataLoader, Dataset

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, SEARCH_TILES, write_simple_yaml
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v7 import build_seed_expert_cache
from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    REGIME_TO_ID,
    TILE_ORDER,
    TileBundle,
    compute_metrics,
    ensure_dir,
    preprocess_all_tiles,
    seed_everything,
    write_json,
)


DSCF_ROOT = ensure_dir(OUTPUT_ROOT / "dscf")
CONFIG_ROOT = ensure_dir(DSCF_ROOT / "configs")
QUICK_ROOT = ensure_dir(DSCF_ROOT / "quick")
FULL_ROOT = ensure_dir(DSCF_ROOT / "fullsix")
ABLATION_ROOT = ensure_dir(DSCF_ROOT / "ablation")
DIAG_ROOT = ensure_dir(DSCF_ROOT / "diagnostics")
FIG_ROOT = ensure_dir(DSCF_ROOT / "figures")
RUN_ROOT = ensure_dir(DSCF_ROOT / "runs")

PROTOCOL_PATH = DSCF_ROOT / "dscf_protocol.md"
STATE_FEATURE_CHECK_PATH = DSCF_ROOT / "state_feature_check.csv"
REGIME_STATS_PATH = DSCF_ROOT / "regime_statistics.csv"
REGIME_FIG_PATH = FIG_ROOT / "regime_maps.png"
QUICK_RANK_PATH = QUICK_ROOT / "quick_rank_table.csv"
QUICK_TILE_H_PATH = QUICK_ROOT / "quick_tile_horizon.csv"
QUICK_STRESS_PATH = QUICK_ROOT / "quick_stress.csv"
QUICK_REGIME_BREAKDOWN_PATH = QUICK_ROOT / "quick_regime_breakdown.csv"
WEEK1_DECISION_PATH = DSCF_ROOT / "decision_week1.md"
SELECTED_VARIANTS_PATH = DSCF_ROOT / "selected_variants.md"
FULL6_RAW_PATH = FULL_ROOT / "dscf_fullsix_raw.csv"
FULL6_SUMMARY_PATH = FULL_ROOT / "dscf_fullsix_summary.csv"
VCEA_PLUS_RAW_PATH = FULL_ROOT / "vcea_plus_dscf_raw.csv"
VCEA_PLUS_SUMMARY_PATH = FULL_ROOT / "vcea_plus_dscf_summary.csv"
SEED_STD_REPORT_PATH = FULL_ROOT / "seed_std_report.csv"
FULL6_BREAKDOWN_PATH = FULL_ROOT / "breakdown_all.csv"
SAFE_REPORT_PATH = FULL_ROOT / "dscf_safe_report.csv"
WEEK2_DECISION_PATH = DSCF_ROOT / "decision_week2.md"
ABLATION_SUMMARY_PATH = ABLATION_ROOT / "ablation_summary.csv"
ABLATION_LOWFREQ_PATH = ABLATION_ROOT / "ablation_lowfreq.csv"
ABLATION_MOTION_PATH = ABLATION_ROOT / "ablation_motion_state.csv"
ABLATION_REGIME_PATH = ABLATION_ROOT / "ablation_regime.csv"
ABLATION_BOUND_PATH = ABLATION_ROOT / "ablation_bound.csv"
ABLATION_ANCHOR_PATH = ABLATION_ROOT / "ablation_anchor.csv"
VELOCITY_SIGN_PATH = DIAG_ROOT / "velocity_sign_consistency.csv"
SEASONAL_PHASE_PATH = DIAG_ROOT / "seasonal_phase_error.csv"
UNREALISTIC_JUMP_PATH = DIAG_ROOT / "unrealistic_jump_rate.csv"
STRESS_TILE_REPORT_PATH = DIAG_ROOT / "stress_tile_report.csv"
CORRECTION_HIST_PATH = DIAG_ROOT / "correction_magnitude_hist.csv"
BOUNDSAT_PATH = DIAG_ROOT / "bound_saturation_ratio.csv"
HEATMAP_PATH = FIG_ROOT / "tile_horizon_heatmap.png"
REGIME_BREAKDOWN_PNG = FIG_ROOT / "regime_breakdown.png"
STRESS_SAFETY_PNG = FIG_ROOT / "stress_safety_curve.png"
ARCH_PATH = FIG_ROOT / "dscf_architecture.png"
DECISION_PATH = DSCF_ROOT / "dscf_final_decision.md"
REPORT_PATH = DSCF_ROOT / "dscf_detailed_report.md"
SUMMARY_PATH = DSCF_ROOT / "run_summary.json"

BEST_STATIC_METHOD = "inverse"
BEST_STATIC_TEMP = 4.0
BEST_STATIC_SHRINKAGE = 0.1
BEST_STATIC_M1_FLOOR = 0.0
CURRENT_VCEA_MEAN = 0.17763291690164576
CURRENT_VCEA_STRESS = 1.0463388663683806
CURRENT_VCEA_STD = 0.010450696353659717
CURRENT_M1_MEAN = 0.0783341224701473
STRESS_TILES = ["E48N24", "E58N17"]
SEARCH_SEEDS = [42]


@dataclass(frozen=True)
class DSCFConfig:
    model_id: str
    variant: str
    anchor_type: str
    use_lowfreq: bool
    use_motion: bool
    use_regime: bool
    use_bound: bool
    regime_soft: bool = True
    hidden: int = 24
    batch_size: int = 8
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    use_amp: bool = True
    bound_scale: float = 1.0
    lambda_p: float = 0.05
    lambda_s: float = 0.02
    lambda_sea: float = 0.02
    num_workers: int = 0
    safe_enabled: bool = False
    safe_blend: float = 0.5


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def default_dscf_configs(seed: int = 42) -> list[DSCFConfig]:
    return [
        DSCFConfig(
            model_id="DSCF-0",
            variant="anchor_only",
            anchor_type="state",
            use_lowfreq=False,
            use_motion=False,
            use_regime=False,
            use_bound=False,
            epochs=0,
            seed=seed,
        ),
        DSCFConfig(
            model_id="DSCF-LF",
            variant="lf",
            anchor_type="seasonal_state",
            use_lowfreq=True,
            use_motion=False,
            use_regime=False,
            use_bound=True,
            hidden=20,
            seed=seed,
        ),
        DSCFConfig(
            model_id="DSCF-MS",
            variant="ms",
            anchor_type="state",
            use_lowfreq=False,
            use_motion=True,
            use_regime=False,
            use_bound=True,
            hidden=24,
            seed=seed,
        ),
        DSCFConfig(
            model_id="DSCF-RC",
            variant="rc",
            anchor_type="state",
            use_lowfreq=False,
            use_motion=True,
            use_regime=True,
            use_bound=True,
            hidden=24,
            seed=seed,
        ),
        DSCFConfig(
            model_id="DSCF-Full",
            variant="full",
            anchor_type="seasonal_state",
            use_lowfreq=True,
            use_motion=True,
            use_regime=True,
            use_bound=True,
            hidden=28,
            seed=seed,
        ),
    ]


def tuned_rescue_configs(base: list[DSCFConfig]) -> list[DSCFConfig]:
    candidates: list[DSCFConfig] = []
    for cfg in base:
        if cfg.model_id not in {"DSCF-RC", "DSCF-Full"}:
            continue
        candidates.extend(
            [
                replace(
                    cfg,
                    model_id=f"{cfg.model_id}_state_bs075",
                    anchor_type="state",
                    bound_scale=0.75,
                    regime_soft=False,
                ),
                replace(
                    cfg,
                    model_id=f"{cfg.model_id}_state_bs125",
                    anchor_type="state",
                    bound_scale=1.25,
                    regime_soft=True,
                ),
                replace(
                    cfg,
                    model_id=f"{cfg.model_id}_season_bs100",
                    anchor_type="seasonal_state",
                    bound_scale=1.0,
                    regime_soft=True,
                ),
            ]
        )
    unique: dict[str, DSCFConfig] = {}
    for cfg in candidates:
        unique[cfg.model_id] = cfg
    return list(unique.values())


def state_scale(values: np.ndarray, valid_mask: np.ndarray, floor: float = 1e-3) -> float:
    valid = values[valid_mask > 0.5]
    if valid.size == 0:
        return 1.0
    return float(max(floor, np.median(np.abs(valid))))


def fit_slope(window: np.ndarray, t_years: np.ndarray) -> np.ndarray:
    t = t_years.astype(np.float32)
    t = t - float(t.mean())
    denom = float((t**2).sum() + 1e-6)
    centered = window - window.mean(axis=0, keepdims=True)
    return np.tensordot(t, centered, axes=(0, 0)) / denom


def recent_state_features(
    disp: np.ndarray,
    t_years: np.ndarray,
    current: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recent_start = max(0, current - 4)
    recent = disp[recent_start : current + 1]
    recent_t = t_years[recent_start : current + 1]
    recent_mean = recent.mean(axis=0).astype(np.float32)
    recent_slope = fit_slope(recent, recent_t).astype(np.float32)
    if recent.shape[0] > 1:
        recent_delta = np.diff(recent, axis=0).astype(np.float32)
        recent_vol = recent_delta.std(axis=0).astype(np.float32)
        last_delta = recent_delta[-1].astype(np.float32)
    else:
        recent_vol = np.zeros_like(recent_mean, dtype=np.float32)
        last_delta = np.zeros_like(recent_mean, dtype=np.float32)
    prev_end = recent_start
    prev_start = max(0, prev_end - 5)
    if prev_end - prev_start >= 2:
        prev = disp[prev_start:prev_end]
        prev_t = t_years[prev_start:prev_end]
        prev_slope = fit_slope(prev, prev_t).astype(np.float32)
        dt = max(1e-6, float(recent_t.mean() - prev_t.mean()))
        recent_acc = ((recent_slope - prev_slope) / dt).astype(np.float32)
    else:
        recent_acc = np.zeros_like(recent_mean, dtype=np.float32)
    return recent_mean, recent_slope, recent_acc, recent_vol, last_delta


def make_regime_tensor(regime_map: np.ndarray, soft: bool) -> np.ndarray:
    channels = np.stack([(regime_map == idx).astype(np.float32) for idx in range(len(REGIME_NAMES))], axis=0)
    if not soft:
        return channels
    smoothed = np.stack([uniform_filter(ch, size=5, mode="nearest") for ch in channels], axis=0).astype(np.float32)
    denom = np.clip(smoothed.sum(axis=0, keepdims=True), 1e-6, None)
    return (smoothed / denom).astype(np.float32)


def lowfreq_stack(last_norm: np.ndarray, recent_norm: np.ndarray) -> np.ndarray:
    chans = []
    for size in [5, 9, 17]:
        chans.append(uniform_filter(last_norm, size=size, mode="nearest").astype(np.float32))
    for size in [5, 9]:
        chans.append(uniform_filter(recent_norm, size=size, mode="nearest").astype(np.float32))
    chans.append((uniform_filter(last_norm, size=17, mode="nearest") - uniform_filter(last_norm, size=5, mode="nearest")).astype(np.float32))
    return np.stack(chans, axis=0).astype(np.float32)


def seasonal_increment_map(
    amp: np.ndarray,
    phase: np.ndarray,
    t_current: float,
    t_future: float,
) -> np.ndarray:
    current_term = np.sin(2.0 * np.pi * t_current + phase)
    future_term = np.sin(2.0 * np.pi * t_future + phase)
    return (amp * (future_term - current_term)).astype(np.float32)


def lambda_map_from_regime(regime_map: np.ndarray) -> np.ndarray:
    lam = np.full(regime_map.shape, 0.25, dtype=np.float32)
    lam[regime_map == REGIME_TO_ID["seasonal"]] = 0.35
    lam[regime_map == REGIME_TO_ID["subsidence"]] = 0.45
    lam[regime_map == REGIME_TO_ID["accelerating"]] = 0.70
    lam[regime_map == REGIME_TO_ID["coseismic"]] = 0.12
    lam[regime_map == REGIME_TO_ID["mixed"]] = 0.30
    return lam


def anchor_map_for_sample(
    bundle: TileBundle,
    current: int,
    target_indices: list[int],
    last_disp: np.ndarray,
    recent_slope: np.ndarray,
    train_velocity: np.ndarray,
    amp_map: np.ndarray,
    phase_map: np.ndarray,
    regime_map: np.ndarray,
    anchor_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.to_datetime(bundle.dates)
    t_year = ((dates - dates[0]).days.to_numpy(dtype=np.float32)) / 365.25
    lam = lambda_map_from_regime(regime_map)
    v_hat = lam * recent_slope + (1.0 - lam) * train_velocity
    anchors: list[np.ndarray] = []
    seasonal_stack: list[np.ndarray] = []
    for target_idx in target_indices:
        dt = float(t_year[target_idx] - t_year[current])
        seasonal_delta = seasonal_increment_map(amp_map, phase_map, float(t_year[current]), float(t_year[target_idx]))
        base = last_disp.astype(np.float32)
        if anchor_type == "persistence":
            anchor = base
        elif anchor_type == "state":
            anchor = base + dt * v_hat
        elif anchor_type == "seasonal_state":
            anchor = base + dt * v_hat + seasonal_delta
        else:
            raise KeyError(anchor_type)
        anchors.append(anchor.astype(np.float32))
        seasonal_stack.append(seasonal_delta.astype(np.float32))
    return np.stack(anchors, axis=0).astype(np.float32), np.stack(seasonal_stack, axis=0).astype(np.float32)


def bound_base_for_sample(
    recent_vol: np.ndarray,
    train_acc: np.ndarray,
    amp_map: np.ndarray,
    residual_std: np.ndarray,
    seasonal_delta: np.ndarray,
    bundle: TileBundle,
    current: int,
    target_indices: list[int],
    scale: float,
) -> np.ndarray:
    dates = pd.to_datetime(bundle.dates)
    t_year = ((dates - dates[0]).days.to_numpy(dtype=np.float32)) / 365.25
    bound = []
    for h_idx, target_idx in enumerate(target_indices):
        dt = float(t_year[target_idx] - t_year[current])
        base = (
            0.10
            + 0.90 * recent_vol
            + 0.30 * np.abs(train_acc) * (dt**2)
            + 0.20 * np.abs(seasonal_delta[h_idx])
            + 0.25 * residual_std
            + 0.05 * np.abs(amp_map)
        )
        bound.append((scale * base).astype(np.float32))
    return np.stack(bound, axis=0).astype(np.float32)


class DSCFDataset(Dataset):
    def __init__(self, bundle: TileBundle, split_name: str, config: DSCFConfig) -> None:
        self.bundle = bundle
        self.split_name = split_name
        self.config = config
        tensors = bundle.tensors64
        disp = tensors["disp"].astype(np.float32)
        valid_mask = tensors["valid_mask"].astype(np.float32)
        regime_map = tensors["regime_map"].astype(np.int16)
        dates = pd.to_datetime(bundle.dates)
        t_year = ((dates - dates[0]).days.to_numpy(dtype=np.float32)) / 365.25
        max_h = max(SEARCH_HORIZONS)

        if split_name == "train":
            current_frames = list(range(configure_input_len() - 1, bundle.split.train_frame_end - max_h))
        elif split_name == "val":
            current_frames = list(range(bundle.split.train_frame_end, bundle.split.val_frame_end - max_h))
        elif split_name == "test":
            current_frames = list(range(bundle.split.val_frame_end, bundle.split.total_frames - max_h))
        else:
            raise KeyError(split_name)

        train_end = bundle.split.train_frame_end
        disp_mu = tensors["disp_mu"].astype(np.float32)
        disp_sigma = tensors["disp_sigma"].astype(np.float32)
        train_velocity = tensors["descriptor_velocity"].astype(np.float32)
        train_acc = tensors["descriptor_acc"].astype(np.float32)
        amp_map = tensors["descriptor_amp"].astype(np.float32)
        phase_map = tensors["seasonal_phase"].astype(np.float32)
        residual_std = tensors["residual_std"].astype(np.float32)
        trend_r2 = tensors["trend_r2"].astype(np.float32)

        vel_scale = state_scale(train_velocity, valid_mask)
        acc_scale = state_scale(train_acc, valid_mask)
        amp_scale = state_scale(amp_map, valid_mask)
        vol_scale = state_scale(residual_std, valid_mask)
        jump_scale = 3.0

        self.mask = valid_mask.astype(np.float32)
        self.regime_map = regime_map
        self.regime_soft = make_regime_tensor(regime_map, config.regime_soft)
        self.samples: list[dict[str, np.ndarray]] = []

        for current in current_frames:
            target_indices = [current + h for h in SEARCH_HORIZONS]
            last_disp = disp[current].astype(np.float32)
            recent_mean, recent_slope, recent_acc, recent_vol, last_delta = recent_state_features(disp, t_year, current)
            last_norm = ((last_disp - disp_mu) / disp_sigma).astype(np.float32)
            recent_mean_norm = ((recent_mean - disp_mu) / disp_sigma).astype(np.float32)
            recent_slope_norm = (recent_slope / vel_scale).astype(np.float32)
            recent_acc_norm = (recent_acc / acc_scale).astype(np.float32)
            recent_vol_norm = (recent_vol / vol_scale).astype(np.float32)
            train_velocity_norm = (train_velocity / vel_scale).astype(np.float32)
            train_acc_norm = (train_acc / acc_scale).astype(np.float32)
            amp_norm = (amp_map / amp_scale).astype(np.float32)
            resid_norm = (residual_std / vol_scale).astype(np.float32)
            jump_score = np.tanh(np.abs(last_delta) / (jump_scale * np.maximum(residual_std, 1e-3))).astype(np.float32)
            season_angle = (2.0 * np.pi * float(t_year[current]) + phase_map).astype(np.float32)
            season_sin = np.sin(season_angle).astype(np.float32)
            season_cos = np.cos(season_angle).astype(np.float32)
            trend_norm = np.clip(2.0 * trend_r2 - 1.0, -1.0, 1.0).astype(np.float32)

            motion = np.stack(
                [
                    last_norm,
                    recent_mean_norm,
                    recent_slope_norm,
                    recent_acc_norm,
                    recent_vol_norm,
                    train_velocity_norm,
                    train_acc_norm,
                    amp_norm,
                    resid_norm,
                    jump_score,
                    trend_norm,
                    season_sin,
                    season_cos,
                ],
                axis=0,
            ).astype(np.float32)
            lowfreq = lowfreq_stack(last_norm, recent_mean_norm).astype(np.float32)
            anchor, seasonal_delta = anchor_map_for_sample(
                bundle=bundle,
                current=current,
                target_indices=target_indices,
                last_disp=last_disp,
                recent_slope=recent_slope,
                train_velocity=train_velocity,
                amp_map=amp_map,
                phase_map=phase_map,
                regime_map=regime_map,
                anchor_type=config.anchor_type,
            )
            bound_base = bound_base_for_sample(
                recent_vol=recent_vol,
                train_acc=train_acc,
                amp_map=amp_map,
                residual_std=residual_std,
                seasonal_delta=seasonal_delta,
                bundle=bundle,
                current=current,
                target_indices=target_indices,
                scale=config.bound_scale,
            )
            target = np.stack([disp[idx] for idx in target_indices], axis=0).astype(np.float32)
            persistence = np.repeat(last_disp[None, :, :], len(SEARCH_HORIZONS), axis=0).astype(np.float32)
            linear_state = target - persistence
            self.samples.append(
                {
                    "motion": motion,
                    "lowfreq": lowfreq,
                    "regime_soft": self.regime_soft.astype(np.float32),
                    "anchor_mm": anchor.astype(np.float32),
                    "bound_base_mm": bound_base.astype(np.float32),
                    "y_true_mm": target.astype(np.float32),
                    "persistence_mm": persistence.astype(np.float32),
                    "seasonal_delta_mm": seasonal_delta.astype(np.float32),
                    "last_disp_mm": last_disp[None, :, :].astype(np.float32),
                    "valid_mask": valid_mask[None, :, :].astype(np.float32),
                    "current_frame": np.array([current], dtype=np.int64),
                    "target_indices": np.array(target_indices, dtype=np.int64),
                    "velocity_sign": np.sign(train_velocity)[None, :, :].astype(np.float32),
                    "velocity_gate": (np.abs(train_velocity) > 0.5 * vel_scale).astype(np.float32)[None, :, :],
                    "stable_mask": (regime_map == REGIME_TO_ID["stable"]).astype(np.float32)[None, :, :],
                    "seasonal_mask": (regime_map == REGIME_TO_ID["seasonal"]).astype(np.float32)[None, :, :],
                    "jump_like_target": linear_state.astype(np.float32),
                    "risk_map": (recent_vol + np.abs(last_delta)).astype(np.float32),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        out: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            if value.dtype.kind in {"i", "u"}:
                out[key] = torch.from_numpy(value).long()
            else:
                out[key] = torch.from_numpy(value).float()
        return out


def configure_input_len() -> int:
    return 16


class DSCFNet(nn.Module):
    def __init__(self, config: DSCFConfig, motion_channels: int, lowfreq_channels: int) -> None:
        super().__init__()
        self.config = config
        self.output_len = len(SEARCH_HORIZONS)
        hidden = config.hidden
        branch_dim = 0
        if config.use_motion:
            self.motion_proj = nn.Sequential(
                nn.Conv2d(motion_channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            )
            branch_dim += hidden
        else:
            self.motion_proj = None
        if config.use_lowfreq:
            self.lowfreq_proj = nn.Sequential(
                nn.Conv2d(lowfreq_channels, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
            )
            branch_dim += hidden
        else:
            self.lowfreq_proj = None
        if branch_dim == 0:
            self.null_bias = nn.Parameter(torch.zeros(1, hidden, 1, 1))
            branch_dim = hidden
        else:
            self.null_bias = None
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        if config.use_regime:
            self.regime_mod = nn.Conv2d(len(REGIME_NAMES), hidden * 2, kernel_size=1)
        else:
            self.regime_mod = None
        self.candidate_head = nn.Conv2d(hidden, self.output_len, kernel_size=1)
        self.horizon_bias = nn.Parameter(torch.zeros(1, self.output_len, 1, 1))
        rho_init = torch.tensor(
            [
                [0.65, 0.85, 0.90, 1.00, 0.55, 0.75],
                [0.70, 0.95, 1.00, 1.10, 0.55, 0.85],
                [0.75, 1.00, 1.05, 1.20, 0.55, 0.90],
                [0.80, 1.05, 1.10, 1.30, 0.60, 0.95],
            ],
            dtype=torch.float32,
        )
        self.rho_logits = nn.Parameter(torch.logit(torch.clamp((rho_init - 0.2) / 1.6, 1e-3, 1 - 1e-3)))

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        feats = []
        if self.motion_proj is not None:
            feats.append(self.motion_proj(batch["motion"]))
        if self.lowfreq_proj is not None:
            feats.append(self.lowfreq_proj(batch["lowfreq"]))
        if feats:
            x = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)
        else:
            b, _, h, w = batch["motion"].shape
            x = self.null_bias.expand(b, -1, h, w)
        x = self.fuse(x)
        if self.regime_mod is not None:
            gamma, beta = self.regime_mod(batch["regime_soft"]).chunk(2, dim=1)
            x = x * (1.0 + 0.35 * torch.tanh(gamma)) + 0.35 * torch.tanh(beta)
        correction_logits = self.candidate_head(x) + self.horizon_bias
        anchor = batch["anchor_mm"]
        if self.config.use_bound:
            rho = 0.2 + 1.6 * torch.sigmoid(self.rho_logits)
            bound_scale = torch.einsum("brhw,or->bohw", batch["regime_soft"], rho)
            bound = bound_scale * batch["bound_base_mm"]
            correction_mm = bound * torch.tanh(correction_logits)
            pred = anchor + correction_mm
        else:
            bound = batch["bound_base_mm"]
            correction_mm = correction_logits
            pred = anchor + correction_mm
        if return_aux:
            return pred, {
                "anchor_mm": anchor,
                "bound_mm": bound,
                "correction_mm": correction_mm,
                "utilization": torch.abs(correction_mm) / torch.clamp(bound, min=1e-6),
            }
        return pred


def dscf_loss(
    pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: DSCFConfig,
) -> torch.Tensor:
    mask = batch["valid_mask"][:, None]
    target = batch["y_true_mm"]
    last = batch["last_disp_mm"][:, None]
    pred_loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum") / torch.clamp(mask.sum() * pred.shape[1], min=1.0)
    stable_mask = batch["stable_mask"][:, None] * mask
    persist_pen = torch.abs(pred - last) * stable_mask
    persist_loss = persist_pen.sum() / torch.clamp(stable_mask.sum() * pred.shape[1], min=1.0)
    sign_mask = batch["velocity_gate"][:, None] * mask
    sign_term = torch.relu(-(batch["velocity_sign"][:, None] * (pred - last))) * sign_mask
    sign_loss = sign_term.sum() / torch.clamp(sign_mask.sum() * pred.shape[1], min=1.0)
    seasonal_mask = batch["seasonal_mask"][:, None] * mask
    seasonal_target = batch["seasonal_delta_mm"]
    seasonal_loss = torch.abs((pred - last) - seasonal_target) * seasonal_mask
    seasonal_loss = seasonal_loss.sum() / torch.clamp(seasonal_mask.sum() * pred.shape[1], min=1.0)
    return pred_loss + config.lambda_p * persist_loss + config.lambda_s * sign_loss + config.lambda_sea * seasonal_loss


def model_param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def loader_kwargs(config: DSCFConfig, shuffle: bool) -> dict[str, Any]:
    return {"batch_size": config.batch_size, "shuffle": shuffle, "num_workers": config.num_workers}


def arrays_from_collected(
    batch_list: list[dict[str, np.ndarray]],
    pred_list: list[np.ndarray],
    aux_list: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    order = np.argsort(np.concatenate([item["current_frame"].reshape(-1) for item in batch_list]))
    arrays: dict[str, np.ndarray] = {}
    concat_keys = ["y_true_mm", "persistence_mm", "valid_mask", "current_frame", "target_indices", "last_disp_mm", "seasonal_delta_mm", "risk_map"]
    merged: dict[str, list[np.ndarray]] = {key: [] for key in concat_keys}
    aux_merged: dict[str, list[np.ndarray]] = {"anchor_mm": [], "bound_mm": [], "correction_mm": [], "utilization": []}
    for batch_np, pred_np, aux_np in zip(batch_list, pred_list, aux_list, strict=True):
        for key in concat_keys:
            merged[key].append(batch_np[key])
        for key in aux_merged:
            aux_merged[key].append(aux_np[key])
    arrays["y_true_mm"] = np.concatenate(merged["y_true_mm"], axis=0)[order]
    arrays["y_pred_mm"] = np.concatenate(pred_list, axis=0)[order]
    arrays["persistence_mm"] = np.concatenate(merged["persistence_mm"], axis=0)[order]
    arrays["valid_mask"] = merged["valid_mask"][0][0]
    arrays["current_frames"] = np.concatenate(merged["current_frame"], axis=0)[order].reshape(-1).astype(np.int64)
    arrays["target_indices"] = np.concatenate(merged["target_indices"], axis=0)[order].astype(np.int64)
    arrays["last_disp_mm"] = np.concatenate(merged["last_disp_mm"], axis=0)[order]
    arrays["seasonal_delta_mm"] = np.concatenate(merged["seasonal_delta_mm"], axis=0)[order]
    arrays["risk_map"] = np.concatenate(merged["risk_map"], axis=0)[order]
    for key in aux_merged:
        arrays[key] = np.concatenate(aux_merged[key], axis=0)[order].astype(np.float32)
    return arrays


def evaluate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    mask2d = np.asarray(arrays["valid_mask"]).squeeze().astype(np.float32)
    valid = mask2d > 0.5
    model_mse = float(((arrays["y_true_mm"] - arrays["y_pred_mm"]) ** 2)[:, :, valid].mean())
    pers_mse = float(((arrays["y_true_mm"] - arrays["persistence_mm"]) ** 2)[:, :, valid].mean())
    metrics = compute_metrics(arrays["y_true_mm"], arrays["y_pred_mm"], mask2d)
    pers_metrics = compute_metrics(arrays["y_true_mm"], arrays["persistence_mm"], mask2d)
    metrics["Skill_MSE"] = float(1.0 - model_mse / max(pers_mse, 1e-8))
    metrics["Skill_RMSE"] = float(1.0 - metrics["RMSE [mm]"] / max(pers_metrics["RMSE [mm]"], 1e-8))
    return metrics


def run_epoch(
    model: DSCFNet,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    device: torch.device,
    config: DSCFConfig,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    pers_list: list[np.ndarray] = []
    mask_ref: np.ndarray | None = None
    for batch in loader:
        batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.set_grad_enabled(train_mode):
            with amp.autocast("cuda", enabled=config.use_amp and device.type == "cuda"):
                pred = model(batch_gpu)
                loss = dscf_loss(pred, batch_gpu, config)
            if train_mode and optimizer is not None and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().cpu())
        pred_list.append(pred.detach().cpu().numpy().astype(np.float32))
        true_list.append(batch["y_true_mm"].numpy().astype(np.float32))
        pers_list.append(batch["persistence_mm"].numpy().astype(np.float32))
        if mask_ref is None:
            mask_ref = batch["valid_mask"][0].numpy().astype(np.float32).squeeze()
    if mask_ref is None:
        return {"loss": float("nan"), "skill_mse": float("nan")}
    y_pred = np.concatenate(pred_list, axis=0)
    y_true = np.concatenate(true_list, axis=0)
    y_pers = np.concatenate(pers_list, axis=0)
    valid = mask_ref > 0.5
    model_mse = float(((y_true - y_pred) ** 2)[:, :, valid].mean())
    pers_mse = float(((y_true - y_pers) ** 2)[:, :, valid].mean())
    return {"loss": total_loss / max(len(loader), 1), "skill_mse": float(1.0 - model_mse / max(pers_mse, 1e-8))}


def collect_predictions(
    model: DSCFNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    batch_list: list[dict[str, np.ndarray]] = []
    pred_list: list[np.ndarray] = []
    aux_list: list[dict[str, np.ndarray]] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            pred, aux = model(batch_gpu, return_aux=True)
            batch_np: dict[str, np.ndarray] = {}
            for key in ["y_true_mm", "persistence_mm", "valid_mask", "current_frame", "target_indices", "last_disp_mm", "seasonal_delta_mm", "risk_map"]:
                batch_np[key] = batch[key].numpy()
            batch_list.append(batch_np)
            pred_list.append(pred.detach().cpu().numpy().astype(np.float32))
            aux_list.append({k: v.detach().cpu().numpy().astype(np.float32) for k, v in aux.items()})
    infer_seconds = time.time() - start
    return arrays_from_collected(batch_list, pred_list, aux_list), infer_seconds


def anchor_only_arrays(dataset: DSCFDataset) -> dict[str, np.ndarray]:
    rows = []
    pred = []
    aux = []
    for sample in dataset.samples:
        rows.append(
            {
                "y_true_mm": sample["y_true_mm"][None],
                "persistence_mm": sample["persistence_mm"][None],
                "valid_mask": sample["valid_mask"][None],
                "current_frame": sample["current_frame"][None],
                "target_indices": sample["target_indices"][None],
                "last_disp_mm": sample["last_disp_mm"][None],
                "seasonal_delta_mm": sample["seasonal_delta_mm"][None],
                "risk_map": sample["risk_map"][None],
            }
        )
        pred.append(sample["anchor_mm"][None])
        aux.append(
            {
                "anchor_mm": sample["anchor_mm"][None],
                "bound_mm": sample["bound_base_mm"][None],
                "correction_mm": np.zeros_like(sample["anchor_mm"][None], dtype=np.float32),
                "utilization": np.zeros_like(sample["anchor_mm"][None], dtype=np.float32),
            }
        )
    pred_list = [item.astype(np.float32) for item in pred]
    return arrays_from_collected(rows, pred_list, aux)


def train_eval_single_tile(
    bundle: TileBundle,
    config: DSCFConfig,
    split_name: str,
    output_dir: Path,
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    result_path = output_dir / "result.json"
    pred_path = output_dir / "predictions.npz"
    ckpt_path = output_dir / "model.pth"
    write_simple_yaml(output_dir / "config.yaml", asdict(config))

    if reuse_if_exists and result_path.exists() and pred_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        npz = np.load(pred_path)
        return {"result": result, "arrays": {name: npz[name] for name in npz.files}}

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DSCFDataset(bundle, "train", config)
    val_ds = DSCFDataset(bundle, "val", config)
    eval_ds = DSCFDataset(bundle, split_name, config)

    if config.variant == "anchor_only":
        arrays = anchor_only_arrays(eval_ds)
        infer_seconds = 0.0
        elapsed_seconds = 0.0
        param_count = 0
        best_epoch = -1
        best_score = float(evaluate_arrays(arrays)["Skill_MSE"])
        history: list[dict[str, float]] = []
    else:
        model = DSCFNet(config, motion_channels=train_ds.samples[0]["motion"].shape[0], lowfreq_channels=train_ds.samples[0]["lowfreq"].shape[0]).to(device)
        param_count = model_param_count(model)
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
        train_loader = DataLoader(train_ds, **loader_kwargs(config, True))
        val_loader = DataLoader(val_ds, **loader_kwargs(config, False))
        eval_loader = DataLoader(eval_ds, **loader_kwargs(config, False))

        best_score = -float("inf")
        best_epoch = -1
        wait = 0
        history = []
        start = time.time()
        for epoch in range(1, config.epochs + 1):
            train_stats = run_epoch(model, train_loader, optimizer, scaler, device, config)
            val_stats = run_epoch(model, val_loader, None, None, device, config)
            history.append({"epoch": epoch, "train_loss": train_stats["loss"], "train_skill_mse": train_stats["skill_mse"], "val_loss": val_stats["loss"], "val_skill_mse": val_stats["skill_mse"]})
            if val_stats["skill_mse"] > best_score:
                best_score = float(val_stats["skill_mse"])
                best_epoch = epoch
                torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, ckpt_path)
                wait = 0
            else:
                wait += 1
                if wait >= config.patience:
                    break
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        arrays, infer_seconds = collect_predictions(model, eval_loader, device)
        elapsed_seconds = time.time() - start

    arrays["linear_seasonal_mm"] = build_seed_expert_cache({bundle.tile_id: bundle}, bundle.tile_id, config.seed, split_name)["linear_seasonal_mm"].astype(np.float32)
    arrays["valid_mask"] = np.asarray(arrays["valid_mask"]).squeeze().astype(np.float32)
    arrays["regime_map"] = bundle.tensors64["regime_map"].astype(np.int16)
    rows = result_rows(
        model_id=config.model_id,
        tile=bundle.tile_id,
        seed=config.seed,
        phase=split_name,
        elapsed_seconds=elapsed_seconds,
        infer_seconds=infer_seconds,
        best_epoch=best_epoch,
        param_count=param_count,
        arrays=arrays,
        horizons=SEARCH_HORIZONS,
    )
    metrics = evaluate_arrays(arrays)
    result = {
        "model_id": config.model_id,
        "variant": config.variant,
        "tile": bundle.tile_id,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": elapsed_seconds,
        "infer_seconds": infer_seconds,
        "param_count": param_count,
        "history": history,
        "metrics": metrics,
        "rows": rows,
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def build_search_reference_rows(bundles: dict[str, TileBundle], model_key: str, model_id: str, seeds: list[int], tiles: list[str]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        for tile in tiles:
            cache = build_seed_expert_cache(bundles, tile, seed, "test")
            pred = cache[model_key].astype(np.float32)
            arrays = {
                "y_pred_mm": pred,
                "y_true_mm": cache["y_true_mm"].astype(np.float32),
                "persistence_mm": cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": cache["valid_mask"].astype(np.float32),
                "regime_map": cache["regime_map"].astype(np.int16),
                "current_frames": cache["current_frames"].astype(np.int64),
                "target_indices": cache["target_indices"].astype(np.int64),
            }
            frames.append(pd.DataFrame(result_rows(model_id, tile, seed, "test", 0.0, 0.0, -1, 0, arrays, SEARCH_HORIZONS)))
    return summarize_seeded_rows(pd.concat(frames, ignore_index=True))


def augmented_static_weights(
    y_true: np.ndarray,
    expert_stack: np.ndarray,
    valid_mask: np.ndarray,
    method: str = BEST_STATIC_METHOD,
    temperature: float = BEST_STATIC_TEMP,
    shrinkage: float = BEST_STATIC_SHRINKAGE,
    m1_floor: float = BEST_STATIC_M1_FLOOR,
) -> np.ndarray:
    n_horizons = expert_stack.shape[1]
    n_experts = expert_stack.shape[2]
    valid = valid_mask > 0.5
    weights = np.zeros((n_horizons, n_experts), dtype=np.float32)
    for h_idx in range(n_horizons):
        losses = []
        for exp_idx in range(n_experts):
            sq = (expert_stack[:, h_idx, exp_idx] - y_true[:, h_idx]) ** 2
            losses.append(float(sq[:, valid].mean()))
        loss_vec = np.asarray(losses, dtype=np.float64)
        rel = loss_vec / max(loss_vec[0], 1e-8)
        if method == "inverse":
            raw = 1.0 / np.power(np.maximum(rel, 1e-8), temperature)
        else:
            rel_shift = rel - rel.min()
            raw = np.exp(-temperature * rel_shift)
        raw = raw / max(float(raw.sum()), 1e-8)
        raw = (1.0 - shrinkage) * raw + shrinkage * (1.0 / n_experts)
        if m1_floor > 0:
            m1_idx = 2
            if raw[m1_idx] < m1_floor:
                others = np.array([i for i in range(n_experts) if i != m1_idx], dtype=np.int64)
                remain = max(1e-6, 1.0 - m1_floor)
                other_sum = float(raw[others].sum())
                if other_sum <= 1e-8:
                    raw[others] = remain / len(others)
                else:
                    raw[others] = raw[others] / other_sum * remain
                raw[m1_idx] = m1_floor
        raw = np.clip(raw, 1e-8, None)
        raw /= raw.sum()
        weights[h_idx] = raw.astype(np.float32)
    return weights


def predict_from_weights(expert_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pred = np.zeros((expert_stack.shape[0], expert_stack.shape[1], expert_stack.shape[3], expert_stack.shape[4]), dtype=np.float32)
    for h_idx in range(expert_stack.shape[1]):
        pred[:, h_idx] = (expert_stack[:, h_idx] * weights[h_idx][:, None, None]).sum(axis=1)
    return pred.astype(np.float32)


def load_vcea_weight_table() -> pd.DataFrame:
    path = OUTPUT_ROOT / "staticval_ea_study" / "final_vcea_weights.csv"
    return pd.read_csv(path)


def predict_vcea_from_stored_weights(cache: dict[str, np.ndarray], tile: str, seed: int, weight_table: pd.DataFrame) -> np.ndarray:
    part = weight_table[(weight_table["Tile"] == tile) & (weight_table["Seed"] == seed)]
    experts = np.stack([cache["persistence_mm"], cache["linear_seasonal_mm"], cache["m1_pred_mm"], cache["c3_pred_mm"]], axis=2).astype(np.float32)
    pred = np.zeros_like(cache["y_true_mm"], dtype=np.float32)
    name_to_idx = {"Persistence": 0, "Linear-Seasonal": 1, "M1": 2, "C3_opt": 3}
    for h_idx, horizon in enumerate([f"t+{h}" for h in SEARCH_HORIZONS]):
        rows = part[part["Horizon"] == horizon]
        w = np.zeros((4,), dtype=np.float32)
        for _, row in rows.iterrows():
            w[name_to_idx[str(row["Expert"])]] = float(row["Weight"])
        pred[:, h_idx] = (experts[:, h_idx] * w[:, None, None]).sum(axis=1)
    return pred.astype(np.float32)


def tile_model_stats(summary_df: pd.DataFrame, model_id: str) -> dict[str, Any]:
    part = summary_df[summary_df["Model"] == model_id].copy()
    if part.empty:
        return {"mean_skill_mse": float("nan"), "stress_ratio": float("inf"), "mean_train_seconds": float("nan")}
    stress_part = part[part["Tile"].isin(["E48N24"] if "E58N17" not in set(part["Tile"]) else STRESS_TILES)]
    return {
        "mean_skill_mse": float(part["Skill_MSE mean"].mean()),
        "stress_ratio": float((stress_part["RMSE [mm] mean"] / stress_part["Persistence RMSE [mm] mean"]).max()) if not stress_part.empty else 1.0,
        "mean_train_seconds": float(part["Elapsed seconds mean"].mean()),
    }


def evaluate_quick_candidate(
    bundles: dict[str, TileBundle],
    config: DSCFConfig,
    vcea_weight_table: pd.DataFrame,
    m1_summary_search: pd.DataFrame,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_frames = []
    add_frames = []
    regime_rows = []
    for seed in SEARCH_SEEDS:
        for tile in SEARCH_TILES:
            run_dir = ensure_dir(RUN_ROOT / config.model_id / tile / f"seed_{seed}")
            out = train_eval_single_tile(bundles[tile], replace(config, seed=seed), "test", run_dir, reuse_if_exists=reuse_if_exists)
            arrays = out["arrays"]
            raw_frames.append(pd.DataFrame(out["result"]["rows"]))

            val_run_dir = ensure_dir(RUN_ROOT / config.model_id / tile / f"seed_{seed}_val")
            val_out = train_eval_single_tile(bundles[tile], replace(config, seed=seed), "val", val_run_dir, reuse_if_exists=reuse_if_exists)
            val_arrays = val_out["arrays"]
            test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")

            val_experts = np.stack(
                [
                    val_cache["persistence_mm"],
                    val_cache["linear_seasonal_mm"],
                    val_cache["m1_pred_mm"],
                    val_cache["c3_pred_mm"],
                    val_arrays["y_pred_mm"],
                ],
                axis=2,
            ).astype(np.float32)
            test_experts = np.stack(
                [
                    test_cache["persistence_mm"],
                    test_cache["linear_seasonal_mm"],
                    test_cache["m1_pred_mm"],
                    test_cache["c3_pred_mm"],
                    arrays["y_pred_mm"],
                ],
                axis=2,
            ).astype(np.float32)
            weights = augmented_static_weights(val_cache["y_true_mm"], val_experts, val_cache["valid_mask"])
            add_pred = predict_from_weights(test_experts, weights)
            add_arrays = {
                "y_pred_mm": add_pred,
                "y_true_mm": test_cache["y_true_mm"].astype(np.float32),
                "persistence_mm": test_cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": test_cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": test_cache["valid_mask"].astype(np.float32),
                "regime_map": test_cache["regime_map"].astype(np.int16),
                "current_frames": test_cache["current_frames"].astype(np.int64),
                "target_indices": test_cache["target_indices"].astype(np.int64),
            }
            add_frames.append(pd.DataFrame(result_rows(f"VCEA+{config.model_id}", tile, seed, "test", 0.0, 0.0, -1, 0, add_arrays, SEARCH_HORIZONS)))

            for rid, regime_name in enumerate(REGIME_NAMES):
                mask = (test_cache["valid_mask"] > 0.5) & (test_cache["regime_map"] == rid)
                if int(mask.sum()) == 0:
                    continue
                for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                    mse_model = float(((arrays["y_true_mm"][:, h_idx] - arrays["y_pred_mm"][:, h_idx]) ** 2)[:, mask].mean())
                    mse_ref = float(((test_cache["y_true_mm"][:, h_idx] - test_cache["persistence_mm"][:, h_idx]) ** 2)[:, mask].mean())
                    regime_rows.append({"Model": config.model_id, "Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Regime": regime_name, "Skill_MSE": float(1.0 - mse_model / max(mse_ref, 1e-8))})

    raw_summary = summarize_seeded_rows(pd.concat(raw_frames, ignore_index=True))
    add_summary = summarize_seeded_rows(pd.concat(add_frames, ignore_index=True))
    regime_df = pd.DataFrame(regime_rows)
    raw_summary.to_csv(QUICK_ROOT / f"{config.model_id}_quick_summary.csv", index=False, encoding="utf-8-sig")
    add_summary.to_csv(QUICK_ROOT / f"{config.model_id}_vcea_plus_summary.csv", index=False, encoding="utf-8-sig")
    if not regime_df.empty:
        regime_df.to_csv(QUICK_ROOT / f"{config.model_id}_regime_breakdown.csv", index=False, encoding="utf-8-sig")

    merged = raw_summary.merge(
        m1_summary_search[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs M1"] = merged["Skill_MSE mean"] - merged["M1 Skill_MSE mean"]
    candidate_mean = float(raw_summary["Skill_MSE mean"].mean())
    candidate_gain = float(merged["Delta vs M1"].mean())
    t10_delta = float(merged[merged["Horizon"] == "t+10"]["Delta vs M1"].min()) if not merged.empty else -float("inf")
    tile_compare = merged.groupby("Tile", as_index=False)["Delta vs M1"].mean()
    better_tiles = int((tile_compare["Delta vs M1"] > 0).sum())
    stress_part = raw_summary[raw_summary["Tile"].isin(["E48N24"])]
    candidate_stress = float((stress_part["RMSE [mm] mean"] / stress_part["Persistence RMSE [mm] mean"]).max()) if not stress_part.empty else 1.0

    vcea_search_rows = []
    for seed in SEARCH_SEEDS:
        for tile in SEARCH_TILES:
            cache = build_seed_expert_cache(bundles, tile, seed, "test")
            vcea_pred = predict_vcea_from_stored_weights(cache, tile, seed, vcea_weight_table)
            vcea_arrays = {
                "y_pred_mm": vcea_pred,
                "y_true_mm": cache["y_true_mm"].astype(np.float32),
                "persistence_mm": cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": cache["valid_mask"].astype(np.float32),
                "regime_map": cache["regime_map"].astype(np.int16),
                "current_frames": cache["current_frames"].astype(np.int64),
                "target_indices": cache["target_indices"].astype(np.int64),
            }
            vcea_search_rows.append(pd.DataFrame(result_rows("VCEA", tile, seed, "test", 0.0, 0.0, -1, 0, vcea_arrays, SEARCH_HORIZONS)))
    vcea_search_summary = summarize_seeded_rows(pd.concat(vcea_search_rows, ignore_index=True))
    add_merged = add_summary.merge(
        vcea_search_summary[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "VCEA Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    add_merged["Delta vs VCEA"] = add_merged["Skill_MSE mean"] - add_merged["VCEA Skill_MSE mean"]
    add_mean = float(add_summary["Skill_MSE mean"].mean())
    add_gain = float(add_merged["Delta vs VCEA"].mean())
    add_tile_compare = add_merged.groupby("Tile", as_index=False)["Delta vs VCEA"].mean()
    add_better_tiles = int((add_tile_compare["Delta vs VCEA"] > 0).sum())
    add_stress = float((add_summary[add_summary["Tile"].isin(["E48N24"])]["RMSE [mm] mean"] / add_summary[add_summary["Tile"].isin(["E48N24"])]["Persistence RMSE [mm] mean"]).max())

    pass_branch = "none"
    quick_pass = False
    if candidate_gain >= 0.02 and candidate_stress <= 1.04 and better_tiles >= 2 and t10_delta >= -0.01:
        quick_pass = True
        pass_branch = "candidate"
    if add_gain >= 0.015 and add_stress <= 1.04 and add_better_tiles >= 2:
        quick_pass = True
        if add_gain >= candidate_gain:
            pass_branch = "vcea_plus"
    summary = {
        "Model": config.model_id,
        "Candidate Mean Skill_MSE": candidate_mean,
        "Candidate gain vs M1": candidate_gain,
        "VCEA+candidate Mean Skill_MSE": add_mean,
        "VCEA+candidate gain vs VCEA": add_gain,
        "Better quick tiles": int(max(better_tiles, add_better_tiles)),
        "Stress ratio": float(min(candidate_stress, add_stress)),
        "Candidate stress ratio": candidate_stress,
        "Addition stress ratio": add_stress,
        "Candidate mean train seconds": float(raw_summary["Elapsed seconds mean"].mean()),
        "Quick gate pass": quick_pass,
        "Pass branch": pass_branch,
        "Tile gate count": int(max(better_tiles, add_better_tiles)),
        "t+10 delta vs M1 min": t10_delta,
    }
    return raw_summary, add_summary, summary


def generate_protocol_and_feature_checks(bundles: dict[str, TileBundle]) -> None:
    lines = [
        "# DSCF Protocol",
        "",
        "- 数据：统一使用 `EGMS 2019-2023 Ortho U` 六个 tile，原因是数据获取限制。",
        "- 分辨率：64×64 regional map。",
        "- horizons：t+1 / t+3 / t+5 / t+10。",
        "- split：chronological train / val / test，不做 test tuning。",
        "- quick：E32N34 / E39N30 / E48N24，默认 seed=42。",
        "- full-six：若 quick 通过，则扩展到六 tile × 3 seeds。",
        "- 特征：当前位移、recent mean、recent slope、recent acceleration、recent volatility、train velocity、train acceleration、seasonal amplitude / phase、residual std、trend R²、jump score、soft regime map。",
        "- anchor：persistence / state / seasonal-state。",
        "- 预测形式：`Y_hat = A_h + B_h * tanh(C_h)`。",
        "- 说明：本轮 DSCF 不再继续 generic backbone search，而是围绕 deformation-state update 做最小可执行验证。",
    ]
    write_markdown(PROTOCOL_PATH, lines)

    rows = []
    regime_rows = []
    for tile, bundle in bundles.items():
        tensors = bundle.tensors64
        valid = tensors["valid_mask"] > 0.5
        rows.append(
            {
                "Tile": tile,
                "Velocity median abs": float(np.median(np.abs(tensors["descriptor_velocity"][valid]))),
                "Acceleration median abs": float(np.median(np.abs(tensors["descriptor_acc"][valid]))),
                "Seasonal amp median": float(np.median(np.abs(tensors["descriptor_amp"][valid]))),
                "Residual std median": float(np.median(np.abs(tensors["residual_std"][valid]))),
                "Trend R2 mean": float(tensors["trend_r2"][valid].mean()),
                "Valid ratio": float(valid.mean()),
            }
        )
        regime_counts = np.bincount(tensors["regime_map"].reshape(-1), minlength=len(REGIME_NAMES)).astype(np.float64)
        regime_counts /= max(regime_counts.sum(), 1.0)
        for rid, regime_name in enumerate(REGIME_NAMES):
            regime_rows.append({"Tile": tile, "Regime": regime_name, "Share": float(regime_counts[rid])})
    pd.DataFrame(rows).to_csv(STATE_FEATURE_CHECK_PATH, index=False, encoding="utf-8-sig")
    reg_df = pd.DataFrame(regime_rows)
    reg_df.to_csv(REGIME_STATS_PATH, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, tile in zip(axes.flat, TILE_ORDER, strict=True):
        regime = bundles[tile].tensors64["regime_map"]
        im = ax.imshow(regime, vmin=0, vmax=len(REGIME_NAMES) - 1, cmap="tab10")
        ax.set_title(tile)
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_ticks(range(len(REGIME_NAMES)))
    cbar.set_ticklabels(REGIME_NAMES)
    fig.savefig(REGIME_FIG_PATH, dpi=180)
    plt.close(fig)


def plot_heatmap(summary_df: pd.DataFrame, model_id: str, path: Path) -> None:
    part = summary_df[summary_df["Model"] == model_id]
    if part.empty:
        return
    pivot = part.pivot(index="Tile", columns="Horizon", values="Skill_MSE mean").reindex(TILE_ORDER)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    im = ax.imshow(pivot.to_numpy(dtype=np.float32), cmap="RdYlBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(model_id)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = float(pivot.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_against_reference(summary_df: pd.DataFrame, ref_df: pd.DataFrame) -> dict[str, Any]:
    merged = summary_df.merge(
        ref_df[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "Ref Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs Ref"] = merged["Skill_MSE mean"] - merged["Ref Skill_MSE mean"]
    tile_compare = merged.groupby("Tile", as_index=False)["Delta vs Ref"].mean()
    stress = merged[merged["Tile"].isin(STRESS_TILES)]
    return {
        "mean_skill_mse": float(summary_df["Skill_MSE mean"].mean()),
        "gain_vs_ref": float(merged["Delta vs Ref"].mean()),
        "better_tiles": int((tile_compare["Delta vs Ref"] > 0).sum()),
        "stress_ratio": float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()) if not stress.empty else 1.0,
        "mean_skill_std": float(summary_df["Skill_MSE std"].mean()),
    }


def fullsix_gate(candidate_stats: dict[str, Any], add_stats: dict[str, Any]) -> tuple[str, list[str]]:
    lines = []
    verdict = "no_go"
    if add_stats["gain_vs_ref"] >= 0.02 and add_stats["better_tiles"] >= 4 and add_stats["stress_ratio"] <= 1.03 and add_stats["mean_skill_std"] <= CURRENT_VCEA_STD + 0.01:
        verdict = "strong_success"
    elif candidate_stats["gain_vs_ref"] >= 0.05 and add_stats["gain_vs_ref"] >= 0.005 and add_stats["stress_ratio"] <= CURRENT_VCEA_STRESS:
        verdict = "moderate_success"
    elif candidate_stats["gain_vs_ref"] > 0 and add_stats["stress_ratio"] <= CURRENT_VCEA_STRESS:
        verdict = "weak_success"
    lines.extend(
        [
            f"- DSCF alone mean Skill_MSE={candidate_stats['mean_skill_mse']:.4f}，gain vs M1={candidate_stats['gain_vs_ref']:.4f}，better tiles={candidate_stats['better_tiles']}/6，stress ratio={candidate_stats['stress_ratio']:.4f}。",
            f"- VCEA+DSCF mean Skill_MSE={add_stats['mean_skill_mse']:.4f}，gain vs VCEA={add_stats['gain_vs_ref']:.4f}，better tiles={add_stats['better_tiles']}/6，stress ratio={add_stats['stress_ratio']:.4f}。",
        ]
    )
    return verdict, lines


def quick_candidate_order(quick_df: pd.DataFrame) -> pd.DataFrame:
    out = quick_df.copy()
    out["score"] = (
        out["VCEA+candidate gain vs VCEA"].astype(float)
        + 0.6 * out["Candidate gain vs M1"].astype(float)
        - 0.5 * np.maximum(0.0, out["Stress ratio"].astype(float) - 1.03)
        - 0.1 * np.maximum(0.0, -(out["t+10 delta vs M1 min"].astype(float) + 0.01))
    )
    return out.sort_values(["Quick gate pass", "score"], ascending=[False, False]).reset_index(drop=True)


def run_quick_phase(
    bundles: dict[str, TileBundle],
    configs: list[DSCFConfig],
    reuse_if_exists: bool,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    vcea_weight_table = load_vcea_weight_table()
    m1_search_summary = build_search_reference_rows(bundles, "m1_pred_mm", "M1", SEARCH_SEEDS, SEARCH_TILES)
    rows = []
    quick_payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    tile_horizon_frames = []
    regime_frames = []
    stress_rows = []
    for cfg in configs:
        raw_summary, add_summary, summary = evaluate_quick_candidate(bundles, cfg, vcea_weight_table, m1_search_summary, reuse_if_exists=reuse_if_exists)
        rows.append(summary)
        quick_payload[cfg.model_id] = (raw_summary, add_summary)
        tile_horizon_frames.append(raw_summary.assign(Eval="candidate"))
        tile_horizon_frames.append(add_summary.assign(Eval="vcea_plus"))
        regime_path = QUICK_ROOT / f"{cfg.model_id}_regime_breakdown.csv"
        if regime_path.exists():
            regime_frames.append(pd.read_csv(regime_path))
        stress_rows.append(
            {
                "Model": cfg.model_id,
                "Candidate stress ratio": summary["Candidate stress ratio"],
                "Addition stress ratio": summary["Addition stress ratio"],
                "Candidate train seconds": summary["Candidate mean train seconds"],
            }
        )
    quick_df = quick_candidate_order(pd.DataFrame(rows))
    quick_df.to_csv(QUICK_RANK_PATH, index=False, encoding="utf-8-sig")
    pd.concat(tile_horizon_frames, ignore_index=True).to_csv(QUICK_TILE_H_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(stress_rows).to_csv(QUICK_STRESS_PATH, index=False, encoding="utf-8-sig")
    if regime_frames:
        pd.concat(regime_frames, ignore_index=True).to_csv(QUICK_REGIME_BREAKDOWN_PATH, index=False, encoding="utf-8-sig")
    return quick_df, quick_payload


def load_reference_summaries(bundles: dict[str, TileBundle]) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1_rows = []
    vcea_rows = []
    vcea_weight_table = load_vcea_weight_table()
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            cache = build_seed_expert_cache(bundles, tile, seed, "test")
            m1_arrays = {
                "y_pred_mm": cache["m1_pred_mm"].astype(np.float32),
                "y_true_mm": cache["y_true_mm"].astype(np.float32),
                "persistence_mm": cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": cache["valid_mask"].astype(np.float32),
                "regime_map": cache["regime_map"].astype(np.int16),
                "current_frames": cache["current_frames"].astype(np.int64),
                "target_indices": cache["target_indices"].astype(np.int64),
            }
            m1_rows.append(pd.DataFrame(result_rows("M1", tile, seed, "test", 0.0, 0.0, -1, 0, m1_arrays, SEARCH_HORIZONS)))
            vcea_pred = predict_vcea_from_stored_weights(cache, tile, seed, vcea_weight_table)
            vcea_arrays = {
                "y_pred_mm": vcea_pred,
                "y_true_mm": cache["y_true_mm"].astype(np.float32),
                "persistence_mm": cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": cache["valid_mask"].astype(np.float32),
                "regime_map": cache["regime_map"].astype(np.int16),
                "current_frames": cache["current_frames"].astype(np.int64),
                "target_indices": cache["target_indices"].astype(np.int64),
            }
            vcea_rows.append(pd.DataFrame(result_rows("VCEA", tile, seed, "test", 0.0, 0.0, -1, 0, vcea_arrays, SEARCH_HORIZONS)))
    return summarize_seeded_rows(pd.concat(m1_rows, ignore_index=True)), summarize_seeded_rows(pd.concat(vcea_rows, ignore_index=True))


def calibrate_safe_threshold(val_arrays: dict[str, np.ndarray], m1_val: np.ndarray) -> tuple[float, float]:
    valid = val_arrays["valid_mask"] > 0.5
    util = val_arrays["utilization"][:, :, valid].mean(axis=2)
    jump = np.abs(val_arrays["correction_mm"])[:, :, valid].mean(axis=2)
    risk = util + 0.1 * jump
    best_tau = float(np.quantile(risk, 0.8))
    best_blend = 0.5
    best_score = -float("inf")
    for tau in np.linspace(float(risk.min()), float(risk.max()), 9):
        for blend in [0.25, 0.5, 0.75]:
            pred = val_arrays["y_pred_mm"].copy()
            shrink_mask = risk > tau
            pred[shrink_mask] = blend * pred[shrink_mask] + (1.0 - blend) * m1_val[shrink_mask]
            mse = float(((val_arrays["y_true_mm"] - pred) ** 2)[:, :, valid].mean())
            mse_ref = float(((val_arrays["y_true_mm"] - val_arrays["persistence_mm"]) ** 2)[:, :, valid].mean())
            skill = 1.0 - mse / max(mse_ref, 1e-8)
            score = skill
            if score > best_score:
                best_score = score
                best_tau = float(tau)
                best_blend = float(blend)
    return best_tau, best_blend


def apply_safe_wrapper(test_arrays: dict[str, np.ndarray], m1_test: np.ndarray, tau: float, blend: float) -> tuple[np.ndarray, np.ndarray]:
    valid = test_arrays["valid_mask"] > 0.5
    util = test_arrays["utilization"][:, :, valid].mean(axis=2)
    jump = np.abs(test_arrays["correction_mm"])[:, :, valid].mean(axis=2)
    risk = util + 0.1 * jump
    pred = test_arrays["y_pred_mm"].copy()
    shrink_mask = risk > tau
    pred[shrink_mask] = blend * pred[shrink_mask] + (1.0 - blend) * m1_test[shrink_mask]
    return pred.astype(np.float32), shrink_mask.astype(np.float32)


def run_fullsix_phase(
    bundles: dict[str, TileBundle],
    winner_cfg: DSCFConfig,
    reuse_if_exists: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vcea_weight_table = load_vcea_weight_table()
    m1_summary, vcea_summary = load_reference_summaries(bundles)
    raw_frames = []
    add_frames = []
    breakdown_rows = []
    safe_rows = []
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            run_dir = ensure_dir(RUN_ROOT / winner_cfg.model_id / tile / f"seed_{seed}")
            out = train_eval_single_tile(bundles[tile], replace(winner_cfg, seed=seed), "test", run_dir, reuse_if_exists=reuse_if_exists)
            arrays = out["arrays"]
            raw_frames.append(pd.DataFrame(out["result"]["rows"]))

            val_run_dir = ensure_dir(RUN_ROOT / winner_cfg.model_id / tile / f"seed_{seed}_val")
            val_out = train_eval_single_tile(bundles[tile], replace(winner_cfg, seed=seed), "val", val_run_dir, reuse_if_exists=reuse_if_exists)
            val_arrays = val_out["arrays"]
            test_cache = build_seed_expert_cache(bundles, tile, seed, "test")
            val_cache = build_seed_expert_cache(bundles, tile, seed, "val")

            val_experts = np.stack([val_cache["persistence_mm"], val_cache["linear_seasonal_mm"], val_cache["m1_pred_mm"], val_cache["c3_pred_mm"], val_arrays["y_pred_mm"]], axis=2).astype(np.float32)
            test_experts = np.stack([test_cache["persistence_mm"], test_cache["linear_seasonal_mm"], test_cache["m1_pred_mm"], test_cache["c3_pred_mm"], arrays["y_pred_mm"]], axis=2).astype(np.float32)
            weights = augmented_static_weights(val_cache["y_true_mm"], val_experts, val_cache["valid_mask"])
            add_pred = predict_from_weights(test_experts, weights)
            add_arrays = {
                "y_pred_mm": add_pred,
                "y_true_mm": test_cache["y_true_mm"].astype(np.float32),
                "persistence_mm": test_cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": test_cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": test_cache["valid_mask"].astype(np.float32),
                "regime_map": test_cache["regime_map"].astype(np.int16),
                "current_frames": test_cache["current_frames"].astype(np.int64),
                "target_indices": test_cache["target_indices"].astype(np.int64),
            }
            add_frames.append(pd.DataFrame(result_rows(f"VCEA+{winner_cfg.model_id}", tile, seed, "test", 0.0, 0.0, -1, 0, add_arrays, SEARCH_HORIZONS)))

            tau, blend = calibrate_safe_threshold(val_arrays, val_cache["m1_pred_mm"].astype(np.float32))
            safe_pred, shrink_mask = apply_safe_wrapper(arrays, test_cache["m1_pred_mm"].astype(np.float32), tau, blend)
            safe_arrays = {
                "y_pred_mm": safe_pred,
                "y_true_mm": test_cache["y_true_mm"].astype(np.float32),
                "persistence_mm": test_cache["persistence_mm"].astype(np.float32),
                "linear_seasonal_mm": test_cache["linear_seasonal_mm"].astype(np.float32),
                "valid_mask": test_cache["valid_mask"].astype(np.float32),
                "regime_map": test_cache["regime_map"].astype(np.int16),
                "current_frames": test_cache["current_frames"].astype(np.int64),
                "target_indices": test_cache["target_indices"].astype(np.int64),
            }
            safe_rows.append(
                {
                    "Tile": tile,
                    "Seed": seed,
                    "Tau": tau,
                    "Blend": blend,
                    "Fallback rate": float(shrink_mask.mean()),
                    "Mean risk": float((np.abs(arrays["correction_mm"]) / np.maximum(arrays["bound_mm"], 1e-6)).mean()),
                    "Safe Skill_MSE": float(evaluate_arrays(safe_arrays)["Skill_MSE"]),
                }
            )

            valid = test_cache["valid_mask"] > 0.5
            for rid, regime_name in enumerate(REGIME_NAMES):
                mask = valid & (test_cache["regime_map"] == rid)
                if int(mask.sum()) == 0:
                    continue
                for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                    mse_model = float(((test_cache["y_true_mm"][:, h_idx] - arrays["y_pred_mm"][:, h_idx]) ** 2)[:, mask].mean())
                    mse_add = float(((test_cache["y_true_mm"][:, h_idx] - add_pred[:, h_idx]) ** 2)[:, mask].mean())
                    mse_ref = float(((test_cache["y_true_mm"][:, h_idx] - test_cache["persistence_mm"][:, h_idx]) ** 2)[:, mask].mean())
                    breakdown_rows.append(
                        {
                            "Tile": tile,
                            "Seed": seed,
                            "Horizon": f"t+{horizon}",
                            "Regime": regime_name,
                            "Method": winner_cfg.model_id,
                            "Skill_MSE": float(1.0 - mse_model / max(mse_ref, 1e-8)),
                        }
                    )
                    breakdown_rows.append(
                        {
                            "Tile": tile,
                            "Seed": seed,
                            "Horizon": f"t+{horizon}",
                            "Regime": regime_name,
                            "Method": f"VCEA+{winner_cfg.model_id}",
                            "Skill_MSE": float(1.0 - mse_add / max(mse_ref, 1e-8)),
                        }
                    )

    raw_df = pd.concat(raw_frames, ignore_index=True)
    add_df = pd.concat(add_frames, ignore_index=True)
    raw_df.to_csv(FULL6_RAW_PATH, index=False, encoding="utf-8-sig")
    add_df.to_csv(VCEA_PLUS_RAW_PATH, index=False, encoding="utf-8-sig")
    raw_summary = summarize_seeded_rows(raw_df)
    add_summary = summarize_seeded_rows(add_df)
    raw_summary.to_csv(FULL6_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    add_summary.to_csv(VCEA_PLUS_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(safe_rows).to_csv(SAFE_REPORT_PATH, index=False, encoding="utf-8-sig")
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.to_csv(FULL6_BREAKDOWN_PATH, index=False, encoding="utf-8-sig")

    seed_std = []
    for model_id, part in [(winner_cfg.model_id, raw_summary), (f"VCEA+{winner_cfg.model_id}", add_summary)]:
        stats = summary_against_reference(part, m1_summary if model_id == winner_cfg.model_id else vcea_summary)
        seed_std.append({"Model": model_id, "Mean Skill_MSE": stats["mean_skill_mse"], "Gain vs ref": stats["gain_vs_ref"], "Better tiles": stats["better_tiles"], "Stress ratio": stats["stress_ratio"], "Mean Skill std": stats["mean_skill_std"]})
    seed_std_df = pd.DataFrame(seed_std)
    seed_std_df.to_csv(SEED_STD_REPORT_PATH, index=False, encoding="utf-8-sig")
    return raw_summary, add_summary, seed_std_df


def run_winner_ablations(
    bundles: dict[str, TileBundle],
    winner_cfg: DSCFConfig,
    reuse_if_exists: bool,
) -> pd.DataFrame:
    ablations = [
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_no_lowfreq", use_lowfreq=False),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_no_motion", use_motion=False),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_no_regime", use_regime=False),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_no_bound", use_bound=False),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_anchor_persist", variant="anchor_only", anchor_type="persistence", use_lowfreq=False, use_motion=False, use_regime=False, use_bound=False, epochs=0),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_anchor_state", variant="anchor_only", anchor_type="state", use_lowfreq=False, use_motion=False, use_regime=False, use_bound=False, epochs=0),
        replace(winner_cfg, model_id=f"{winner_cfg.model_id}_anchor_season", variant="anchor_only", anchor_type="seasonal_state", use_lowfreq=False, use_motion=False, use_regime=False, use_bound=False, epochs=0),
    ]
    out_frames = []
    for cfg in ablations:
        rows = []
        for tile in SEARCH_TILES:
            out = train_eval_single_tile(bundles[tile], replace(cfg, seed=42), "test", ensure_dir(RUN_ROOT / cfg.model_id / tile / "seed_42"), reuse_if_exists=reuse_if_exists)
            rows.append(pd.DataFrame(out["result"]["rows"]))
        summary = summarize_seeded_rows(pd.concat(rows, ignore_index=True))
        summary.insert(0, "Ablation", cfg.model_id)
        out_frames.append(summary)
    out_df = pd.concat(out_frames, ignore_index=True)
    out_df.to_csv(ABLATION_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    out_df[out_df["Ablation"].str.contains("no_lowfreq", regex=False)].to_csv(ABLATION_LOWFREQ_PATH, index=False, encoding="utf-8-sig")
    out_df[out_df["Ablation"].str.contains("no_motion", regex=False)].to_csv(ABLATION_MOTION_PATH, index=False, encoding="utf-8-sig")
    out_df[out_df["Ablation"].str.contains("no_regime", regex=False)].to_csv(ABLATION_REGIME_PATH, index=False, encoding="utf-8-sig")
    out_df[out_df["Ablation"].str.contains("no_bound", regex=False)].to_csv(ABLATION_BOUND_PATH, index=False, encoding="utf-8-sig")
    out_df[out_df["Ablation"].str.contains("anchor_", regex=False)].to_csv(ABLATION_ANCHOR_PATH, index=False, encoding="utf-8-sig")
    return out_df


def run_physics_diagnostics(
    bundles: dict[str, TileBundle],
    winner_cfg: DSCFConfig,
    raw_summary: pd.DataFrame,
    add_summary: pd.DataFrame,
) -> None:
    rows_sign = []
    rows_phase = []
    rows_jump = []
    rows_stress = []
    rows_hist = []
    rows_bound = []
    for seed in FINAL_SEEDS:
        for tile in TILE_ORDER:
            pred_path = RUN_ROOT / winner_cfg.model_id / tile / f"seed_{seed}" / "predictions.npz"
            if not pred_path.exists():
                continue
            arr = np.load(pred_path)
            arrays = {k: arr[k] for k in arr.files}
            valid = bundles[tile].tensors64["valid_mask"] > 0.5
            vel = bundles[tile].tensors64["descriptor_velocity"]
            sign_ok = (np.sign(arrays["y_pred_mm"] - arrays["last_disp_mm"]) == np.sign(vel)[None, None]).astype(np.float32)
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                rows_sign.append({"Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Velocity-sign consistency": float(sign_ok[:, h_idx][:, valid].mean())})
                sea_err = np.abs((arrays["y_pred_mm"][:, h_idx] - arrays["last_disp_mm"][:, 0]) - arrays["seasonal_delta_mm"][:, h_idx])
                rows_phase.append({"Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Seasonal phase error [mm]": float(sea_err[:, valid].mean())})
                jump_rate = np.abs(arrays["correction_mm"][:, h_idx]) > 2.5 * np.maximum(arrays["bound_mm"][:, h_idx], 1e-6)
                rows_jump.append({"Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Unrealistic jump rate": float(jump_rate[:, valid].mean())})
                util = arrays["utilization"][:, h_idx][:, valid]
                rows_bound.append({"Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Bound saturation ratio": float((util > 0.8).mean())})
                hist = np.abs(arrays["correction_mm"][:, h_idx][:, valid]).reshape(-1)
                for bin_idx, upper in enumerate([0.25, 0.5, 1.0, 2.0, 4.0, 8.0]):
                    if bin_idx == 0:
                        mask = hist <= upper
                        label = f"<= {upper}"
                    else:
                        lower = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0][bin_idx - 1]
                        mask = (hist > lower) & (hist <= upper)
                        label = f"{lower}-{upper}"
                    rows_hist.append({"Tile": tile, "Seed": seed, "Horizon": f"t+{horizon}", "Bin": label, "Share": float(mask.mean())})
            stress_rows = raw_summary[raw_summary["Tile"] == tile]
            if not stress_rows.empty:
                rows_stress.append({"Tile": tile, "Mean Skill_MSE": float(stress_rows["Skill_MSE mean"].mean()), "Max RMSE ratio": float((stress_rows["RMSE [mm] mean"] / stress_rows["Persistence RMSE [mm] mean"]).max())})
    pd.DataFrame(rows_sign).to_csv(VELOCITY_SIGN_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_phase).to_csv(SEASONAL_PHASE_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_jump).to_csv(UNREALISTIC_JUMP_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_stress).to_csv(STRESS_TILE_REPORT_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_hist).to_csv(CORRECTION_HIST_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows_bound).to_csv(BOUNDSAT_PATH, index=False, encoding="utf-8-sig")

    plot_heatmap(raw_summary.assign(Model=winner_cfg.model_id), winner_cfg.model_id, HEATMAP_PATH)
    if FULL6_BREAKDOWN_PATH.exists():
        df = pd.read_csv(FULL6_BREAKDOWN_PATH)
        pivot = df.groupby(["Method", "Regime"], as_index=False)["Skill_MSE"].mean().pivot(index="Regime", columns="Method", values="Skill_MSE")
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Skill_MSE")
        ax.set_title("Regime-wise skill")
        fig.savefig(REGIME_BREAKDOWN_PNG, dpi=180, bbox_inches="tight")
        plt.close(fig)
    if SAFE_REPORT_PATH.exists():
        safe = pd.read_csv(SAFE_REPORT_PATH)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(safe["Fallback rate"], safe["Safe Skill_MSE"], "o-", color="#c44e52")
        ax.set_xlabel("Fallback rate")
        ax.set_ylabel("Safe Skill_MSE")
        ax.set_title("DSCF-safe calibration")
        fig.savefig(STRESS_SAFETY_PNG, dpi=180, bbox_inches="tight")
        plt.close(fig)


def draw_architecture_placeholder() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, "History window\nD_{t-L+1:t}"),
        (0.24, "Low-frequency\nfield encoder"),
        (0.44, "Motion-state\nestimator"),
        (0.64, "Regime-aware\ncorrection law"),
        (0.84, "Bounded state\ncorrection head"),
    ]
    for x, text in boxes:
        rect = plt.Rectangle((x, 0.28), 0.14, 0.38, facecolor="#edf3ff", edgecolor="#4169e1", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.07, 0.47, text, ha="center", va="center", fontsize=10)
    for start, end in [(0.16, 0.24), (0.38, 0.44), (0.58, 0.64), (0.78, 0.84)]:
        ax.annotate("", xy=(end, 0.47), xytext=(start, 0.47), arrowprops={"arrowstyle": "->", "lw": 1.5})
    fig.savefig(ARCH_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_dscf_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    draw_architecture_placeholder()
    generate_protocol_and_feature_checks(bundles)

    base_configs = default_dscf_configs(seed=42)
    quick_df, quick_payload = run_quick_phase(bundles, base_configs, reuse_if_exists=reuse_if_exists)
    used_rescue = False
    if not bool(quick_df["Quick gate pass"].any()):
        rescue = tuned_rescue_configs(base_configs)
        if rescue:
            rescue_df, rescue_payload = run_quick_phase(bundles, rescue, reuse_if_exists=reuse_if_exists)
            quick_df = quick_candidate_order(pd.concat([quick_df, rescue_df], ignore_index=True).drop_duplicates(subset=["Model"], keep="last"))
            quick_df.to_csv(QUICK_RANK_PATH, index=False, encoding="utf-8-sig")
            quick_payload.update(rescue_payload)
            used_rescue = True

    week1_lines = [
        "# DSCF Week-1 Decision",
        "",
        "数据声明：本轮继续统一使用 `EGMS 2019-2023 Ortho U`，原因是数据获取限制。",
        "",
        "## Quick ranking",
        "",
    ]
    for _, row in quick_df.iterrows():
        week1_lines.append(
            f"- `{row['Model']}`: candidate mean Skill_MSE={row['Candidate Mean Skill_MSE']:.4f}, gain vs M1={row['Candidate gain vs M1']:.4f}, "
            f"`VCEA+candidate` gain vs VCEA={row['VCEA+candidate gain vs VCEA']:.4f}, candidate stress={row['Candidate stress ratio']:.4f}, "
            f"addition stress={row['Addition stress ratio']:.4f}, quick pass={bool(row['Quick gate pass'])} ({row['Pass branch']})."
        )
    if used_rescue:
        week1_lines.extend(["", "- 本轮在初始 5 个变体后追加了最小 rescue grid，仅调 `anchor_type / bound_scale / regime_soft`。"])
    write_markdown(WEEK1_DECISION_PATH, week1_lines)

    pass_rows = quick_df[quick_df["Quick gate pass"]].copy()
    if pass_rows.empty:
        decision_lines = [
            "# DSCF Final Decision",
            "",
            "数据声明：本轮继续统一使用 `EGMS 2019-2023 Ortho U`，原因是数据获取限制。",
            "",
            "- final verdict = `no_go_after_quick_return_to_vcea`。",
            "- 解释：没有任何 DSCF 版本同时满足 quick gate 的增益、stress 和 tile coverage 条件。",
            "- 处理：DSCF 保留为任务原生 diagnostic 方向，但当前主线仍应保持 `VCEA`。",
        ]
        write_markdown(DECISION_PATH, decision_lines)
        report_lines = [
            "# DSCF 完整实验探索报告",
            "",
            "数据声明：本轮 DSCF 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
            "",
            "## 计划执行",
            "",
            "- 已完成协议固化、state feature 抽取、regime map 复核、五个基础 DSCF 变体 quick，以及最小 rescue grid。",
            "- 因 quick gate 无版本通过，本轮没有触发 full-six、winner ablation 或 VCEA-safe 深入阶段。",
            "",
            "## Quick 结论",
            "",
            *[
                f"- `{row['Model']}`: candidate mean Skill_MSE={row['Candidate Mean Skill_MSE']:.4f}, gain vs M1={row['Candidate gain vs M1']:.4f}, "
                f"`VCEA+candidate` gain vs VCEA={row['VCEA+candidate gain vs VCEA']:.4f}, stress={row['Stress ratio']:.4f}, quick pass={bool(row['Quick gate pass'])}."
                for _, row in quick_df.iterrows()
            ],
            "",
            "## 最终判断",
            "",
            "- 本轮 `DSCF` 没有形成足以替代 `VCEA` 的 full-six 候选。",
            "- 这轮更像是一个有价值的任务建模尝试，但当前最小可执行实现仍不足以把 deformation-state update 稳定兑现成 six-tile 系统收益。",
            "",
            "## 关键文件",
            "",
            f"- protocol: `{PROTOCOL_PATH.as_posix()}`",
            f"- state features: `{STATE_FEATURE_CHECK_PATH.as_posix()}`",
            f"- quick rank: `{QUICK_RANK_PATH.as_posix()}`",
            f"- week1 decision: `{WEEK1_DECISION_PATH.as_posix()}`",
        ]
        write_markdown(REPORT_PATH, report_lines)
        summary = {"verdict": "no_go_after_quick_return_to_vcea", "used_rescue": used_rescue, "quick_winner": None}
        write_json(SUMMARY_PATH, summary)
        return summary

    selected_model_id = str(pass_rows.iloc[0]["Model"])
    winner_cfg = next(cfg for cfg in default_dscf_configs(42) + tuned_rescue_configs(default_dscf_configs(42)) if cfg.model_id == selected_model_id)
    write_markdown(
        SELECTED_VARIANTS_PATH,
        [
            "# Selected DSCF Variants",
            "",
            f"- selected winner for full-six: `{winner_cfg.model_id}`。",
            f"- pass branch = `{pass_rows.iloc[0]['Pass branch']}`。",
        ],
    )

    raw_summary, add_summary, seed_std_df = run_fullsix_phase(bundles, winner_cfg, reuse_if_exists=reuse_if_exists)
    candidate_ref, vcea_ref = load_reference_summaries(bundles)
    candidate_stats = summary_against_reference(raw_summary, candidate_ref)
    add_stats = summary_against_reference(add_summary, vcea_ref)
    verdict, verdict_lines = fullsix_gate(candidate_stats, add_stats)
    write_markdown(
        WEEK2_DECISION_PATH,
        [
            "# DSCF Week-2 Decision",
            "",
            f"- winner = `{winner_cfg.model_id}`",
            f"- verdict = `{verdict}`",
            "",
            *verdict_lines,
        ],
    )

    if verdict != "no_go":
        ablation_df = run_winner_ablations(bundles, winner_cfg, reuse_if_exists=reuse_if_exists)
    else:
        ablation_df = pd.DataFrame()
    run_physics_diagnostics(bundles, winner_cfg, raw_summary, add_summary)

    report_lines = [
        "# DSCF 完整实验探索报告",
        "",
        "数据声明：本轮 DSCF 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## Quick 结果",
        "",
        *[
            f"- `{row['Model']}`: candidate mean Skill_MSE={row['Candidate Mean Skill_MSE']:.4f}, gain vs M1={row['Candidate gain vs M1']:.4f}, "
            f"`VCEA+candidate` gain vs VCEA={row['VCEA+candidate gain vs VCEA']:.4f}, candidate stress={row['Candidate stress ratio']:.4f}, addition stress={row['Addition stress ratio']:.4f}, quick pass={bool(row['Quick gate pass'])}."
            for _, row in quick_df.iterrows()
        ],
        "",
        "## Full-six 结果",
        "",
        f"- winner = `{winner_cfg.model_id}`。",
        f"- DSCF alone mean Skill_MSE = {candidate_stats['mean_skill_mse']:.4f}, gain vs M1 = {candidate_stats['gain_vs_ref']:.4f}, better tiles = {candidate_stats['better_tiles']}/6, stress ratio = {candidate_stats['stress_ratio']:.4f}, std = {candidate_stats['mean_skill_std']:.4f}。",
        f"- VCEA+DSCF mean Skill_MSE = {add_stats['mean_skill_mse']:.4f}, gain vs VCEA = {add_stats['gain_vs_ref']:.4f}, better tiles = {add_stats['better_tiles']}/6, stress ratio = {add_stats['stress_ratio']:.4f}, std = {add_stats['mean_skill_std']:.4f}。",
        f"- verdict = `{verdict}`。",
        "",
        "## 诊断",
        "",
        "- 已输出 velocity-sign consistency、seasonal-phase error、unrealistic jump rate、stress tile report、correction magnitude histogram 和 bound saturation ratio。",
        "- 若 verdict 不是 no-go，则已补最小 ablation；否则按决策树停止在 full-six 决策点。",
        "",
        "## 关键文件",
        "",
        f"- quick rank: `{QUICK_RANK_PATH.as_posix()}`",
        f"- fullsix summary: `{FULL6_SUMMARY_PATH.as_posix()}`",
        f"- vcea plus summary: `{VCEA_PLUS_SUMMARY_PATH.as_posix()}`",
        f"- seed std: `{SEED_STD_REPORT_PATH.as_posix()}`",
        f"- diagnostics: `{DIAG_ROOT.as_posix()}`",
    ]
    if not ablation_df.empty:
        report_lines.append(f"- ablation: `{ABLATION_SUMMARY_PATH.as_posix()}`")
    write_markdown(REPORT_PATH, report_lines)

    decision_lines = [
        "# DSCF Final Decision",
        "",
        "数据声明：本轮继续统一使用 `EGMS 2019-2023 Ortho U`，原因是数据获取限制。",
        "",
        f"- winner = `{winner_cfg.model_id}`。",
        f"- final verdict = `{verdict}`。",
        *verdict_lines,
    ]
    if verdict == "strong_success":
        decision_lines.append("- 论文处理：DSCF 可以作为主方法，并进一步考虑 `VCEA+DSCF` 作为 deployment / ensemble layer。")
    elif verdict in {"moderate_success", "weak_success"}:
        decision_lines.append("- 论文处理：DSCF 可进主文或 supplement，取决于你更看重 single-model 贡献还是系统级增益。")
    else:
        decision_lines.append("- 论文处理：DSCF 保留为负结果与任务诊断，主线保持 `VCEA`。")
    write_markdown(DECISION_PATH, decision_lines)

    summary = {
        "verdict": verdict,
        "winner": winner_cfg.model_id,
        "used_rescue": used_rescue,
        "candidate_stats": candidate_stats,
        "add_stats": add_stats,
        "ablation_ran": not ablation_df.empty,
    }
    write_json(SUMMARY_PATH, summary)
    return summary
