from __future__ import annotations

import json
import math
import subprocess
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
from .act_v3 import ActV3Config, ActV3Dataset, count_parameters, result_rows, summarize_seeded_rows
from .act_v8 import (
    apply_safe_wrapper,
    best_static_cfg,
    calibrate_safe_thresholds,
    compare_summary_to_reference,
    estimate_static_weights_generic,
    predict_static_generic,
    reference_quick_m1_vcea,
)
from .act_v7 import build_seed_expert_cache
from .pipeline import OUTPUT_ROOT, REGIME_NAMES, TILE_ORDER, TileBundle, ensure_dir, preprocess_all_tiles, seed_everything


ACT_V8_NARROW_ROOT = ensure_dir(OUTPUT_ROOT / "act_v8_narrow")
RUN_ROOT = ensure_dir(ACT_V8_NARROW_ROOT / "runs")
QUICK_ROOT = ensure_dir(ACT_V8_NARROW_ROOT / "quick")
FULL6_ROOT = ensure_dir(ACT_V8_NARROW_ROOT / "fullsix")
FIG_ROOT = ensure_dir(ACT_V8_NARROW_ROOT / "figures")

SEARCH_TILES = ["E32N34", "E39N30", "E48N24"]
STRESS_TILES = ["E48N24", "E58N17"]
QUICK_SEEDS = [42, 43]

COMMON_HORIZONS = [1, 3, 5, 10]


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def to_v3_cfg(config: "NarrowConfig") -> ActV3Config:
    return ActV3Config(
        model_id=config.model_id,
        variant="v8_narrow",
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


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class NarrowConfig:
    model_id: str
    family: str
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: COMMON_HORIZONS.copy())
    patch_size: int = 8
    dim: int = 48
    batch_size: int = 8
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 3
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    # bounded residual
    bound_eps_mm: float = 0.05
    bound_scale_lambda: float = 0.80
    bound_acc_lambda: float = 0.20
    bound_amp_lambda: float = 0.15
    no_bounded_head: bool = False
    # RCDN
    latent_dim: int = 32
    latent_size: int = 16
    rcdn_noise_levels: tuple[float, ...] = (0.60, 0.20)
    rcdn_mc_samples: int = 2
    rcdn_latent_lambda: float = 0.10
    rcdn_uncert_lambda: float = 0.03
    # CoMEx-B
    comex_hard_lambda: float = 0.08
    comex_ortho_lambda: float = 0.02
    comex_gate_bias: float = 0.0

    @property
    def output_len(self) -> int:
        return len(self.horizons)


class NarrowDataset(Dataset):
    def __init__(
        self,
        bundle: TileBundle,
        split_name: str,
        config: NarrowConfig,
        expert_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config
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
                raise ValueError(f"Expert feature frame mismatch tile={bundle.tile_id} split={split_name}")
            if not np.array_equal(sample_targets, self.expert_features["target_indices"]):
                raise ValueError(f"Expert feature target mismatch tile={bundle.tile_id} split={split_name}")

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
        out[h_idx] = np.clip(arr[h_idx] / max(scale, 1e-6), 0.0, 3.0).astype(np.float32)
    return out


def build_comexb_feature_pack(bundles: dict[str, TileBundle], tile: str, seed: int) -> dict[str, dict[str, np.ndarray]]:
    train_cache = build_seed_expert_cache(bundles, tile, seed, "train")
    train_experts = stack_base_experts(train_cache)
    train_sq = (train_experts - train_cache["y_true_mm"][:, :, None]) ** 2
    best_err = train_sq.min(axis=2).mean(axis=0)
    disagree = train_experts.std(axis=2).mean(axis=0)
    spread = np.abs(train_cache["c3_pred_mm"] - train_cache["m1_pred_mm"]).mean(axis=0)
    weak_regime = np.isin(train_cache["regime_map"], [1, 3, 4, 5]).astype(np.float32)
    hard_prior = normalize_horizon_maps(best_err + 0.5 * disagree + 0.5 * spread)
    hard_prior = (hard_prior * (1.0 + 0.25 * weak_regime[None])).astype(np.float32)

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
        cond = context[:, None] + future_embed[:, :, :, None, None] + self.h_embed[None, :, :, None, None]
        cond_flat = cond.reshape(b * self.output_len, c, h, w)
        gamma = self.gamma_head(cond_flat).reshape(b, self.output_len, c, h, w)
        beta = self.beta_head(cond_flat).reshape(b, self.output_len, c, h, w)
        feat = state[:, None] * (1.0 + 0.10 * torch.tanh(gamma)) + 0.10 * beta + future_embed[:, :, :, None, None]
        raw = self.out(feat.reshape(b * self.output_len, c, h, w))
        return raw.reshape(b, self.output_len, 1, h, w)


class ResidualDecoder(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(latent_dim + cond_dim, cond_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cond_dim, cond_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.mean_head = nn.Conv2d(cond_dim, 1, kernel_size=1)
        self.uncert_head = nn.Sequential(
            nn.Conv2d(cond_dim, cond_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cond_dim // 2, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, cond_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        up = self.up(latent)
        feat = self.fuse(torch.cat([up, cond_high], dim=1))
        mean = self.mean_head(feat)
        uncert = F.softplus(self.uncert_head(feat)) + 1e-3
        return mean, uncert, feat


class RCDNModel(nn.Module):
    def __init__(self, config: NarrowConfig) -> None:
        super().__init__()
        self.config = config
        in_channels = 1 + 1 + 1 + 4 + len(REGIME_NAMES)
        self.cond_high = nn.Sequential(
            nn.Conv2d(in_channels, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cond_low = nn.Sequential(
            nn.Conv2d(config.dim, config.dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.future_proj = nn.Sequential(nn.Linear(2, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.sigma_proj = nn.Sequential(nn.Linear(1, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim))
        self.denoiser = nn.Sequential(
            nn.Conv2d(1 + config.dim, config.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.latent_dim, config.latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.latent_head = nn.Conv2d(config.latent_dim, 1, kernel_size=1)
        self.decoder = ResidualDecoder(1, config.dim)
        self.bound_rho = nn.Parameter(torch.full((config.output_len,), 0.8))

    def _regime_onehot(self, regime: torch.Tensor) -> torch.Tensor:
        return F.one_hot(regime.long(), num_classes=len(REGIME_NAMES)).permute(0, 3, 1, 2).float()

    def _bounded(self, batch: dict[str, torch.Tensor], raw: torch.Tensor, anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.no_bounded_head:
            pred = anchor + raw
            scale = torch.ones_like(raw)
            return pred, scale
        rho = F.softplus(self.bound_rho)[None, :, None, None, None]
        acc = batch["acc_map"][:, None].abs()
        amp_map = batch["amp_map"][:, None].abs()
        scale = rho * (
            self.config.bound_eps_mm
            + self.config.bound_scale_lambda * batch["scale_mm"]
            + self.config.bound_acc_lambda * acc
            + self.config.bound_amp_lambda * amp_map
        )
        pred_delta = scale * torch.tanh(raw)
        pred = anchor + pred_delta
        return pred, scale

    def _encode_condition(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        regime = self._regime_onehot(batch["regime"])
        x = torch.cat(
            [
                batch["last_disp_mm"],
                batch["recent_mean_mm"] / 10.0,
                batch["recent_velocity_mm"] / 2.0,
                batch["desc_stack"],
                regime,
            ],
            dim=1,
        )
        cond_high = self.cond_high(x)
        cond_low = self.cond_low(cond_high)
        return cond_high, cond_low

    def _prepare_context(self, cond_low: torch.Tensor, future_time: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        b, c, h, w = cond_low.shape
        h_out = future_time.shape[1]
        future_emb = self.future_proj(future_time.reshape(b * h_out, 2)).reshape(b, h_out, c, 1, 1)
        sigma_emb = self.sigma_proj(sigma.reshape(b * h_out, 1)).reshape(b, h_out, c, 1, 1)
        context = cond_low[:, None] + future_emb + sigma_emb
        return context

    def _decode_latent(
        self,
        latent_pred: torch.Tensor,
        cond_high: torch.Tensor,
        anchor: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h_out = latent_pred.shape[:2]
        cond_high_rep = cond_high[:, None].expand(-1, h_out, -1, -1, -1).reshape(b * h_out, cond_high.shape[1], 64, 64)
        mean_raw, uncert_raw, feat = self.decoder(latent_pred.reshape(b * h_out, 1, self.config.latent_size, self.config.latent_size), cond_high_rep)
        mean_raw = mean_raw.reshape(b, h_out, 1, 64, 64)
        uncert_raw = uncert_raw.reshape(b, h_out, 1, 64, 64)
        pred, scale = self._bounded(batch, mean_raw, anchor)
        return pred, uncert_raw, scale

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond_high, cond_low = self._encode_condition(batch)
        anchor = batch["persistence_mm"]
        target_resid = batch["target_mm"] - anchor
        latent_target = F.avg_pool2d(target_resid.reshape(target_resid.shape[0] * target_resid.shape[1], 1, 64, 64), kernel_size=4).reshape(
            target_resid.shape[0], target_resid.shape[1], 1, self.config.latent_size, self.config.latent_size
        )
        noise_levels = torch.tensor(self.config.rcdn_noise_levels, device=target_resid.device, dtype=target_resid.dtype)
        sigma = noise_levels[torch.randint(0, len(noise_levels), (target_resid.shape[0], target_resid.shape[1]), device=target_resid.device)]
        zt = latent_target + sigma[:, :, None, None, None] * torch.randn_like(latent_target)
        context = self._prepare_context(cond_low, batch["future_time"], sigma)
        denoise_in = torch.cat([zt.reshape(-1, 1, self.config.latent_size, self.config.latent_size), context.reshape(-1, self.config.dim, self.config.latent_size, self.config.latent_size)], dim=1)
        hidden = self.denoiser(denoise_in)
        latent_pred = self.latent_head(hidden).reshape(target_resid.shape[0], target_resid.shape[1], 1, self.config.latent_size, self.config.latent_size)
        pred, uncert, scale = self._decode_latent(latent_pred, cond_high, anchor, batch)
        aux = {
            "latent_target": latent_target,
            "latent_pred": latent_pred,
            "uncertainty": uncert,
            "scale_mm": scale,
            "anchor_mm": anchor,
        }
        if return_aux:
            return pred, aux
        return pred

    def sample_predictions(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cond_high, cond_low = self._encode_condition(batch)
        anchor = batch["persistence_mm"]
        preds: list[torch.Tensor] = []
        uncerts: list[torch.Tensor] = []
        b, h_out = anchor.shape[:2]
        for _ in range(self.config.rcdn_mc_samples):
            z = torch.randn((b, h_out, 1, self.config.latent_size, self.config.latent_size), device=anchor.device, dtype=anchor.dtype)
            last_uncert = None
            for sigma_val in self.config.rcdn_noise_levels:
                sigma = torch.full((b, h_out), float(sigma_val), device=anchor.device, dtype=anchor.dtype)
                context = self._prepare_context(cond_low, batch["future_time"], sigma)
                denoise_in = torch.cat([z.reshape(-1, 1, self.config.latent_size, self.config.latent_size), context.reshape(-1, self.config.dim, self.config.latent_size, self.config.latent_size)], dim=1)
                hidden = self.denoiser(denoise_in)
                z = self.latent_head(hidden).reshape(b, h_out, 1, self.config.latent_size, self.config.latent_size)
                pred_step, uncert_step, _scale = self._decode_latent(z, cond_high, anchor, batch)
                last_uncert = uncert_step
            preds.append(pred_step)
            uncerts.append(last_uncert if last_uncert is not None else torch.ones_like(pred_step))
        pred_stack = torch.stack(preds, dim=0)
        uncert_stack = torch.stack(uncerts, dim=0)
        mean_pred = pred_stack.mean(dim=0)
        total_uncert = torch.sqrt(pred_stack.var(dim=0, unbiased=False) + uncert_stack.mean(dim=0) ** 2 + 1e-6)
        return mean_pred, total_uncert


class CoMExBModel(nn.Module):
    def __init__(self, config: NarrowConfig) -> None:
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
        self.hard_encoder = nn.Sequential(
            nn.Conv2d(7, config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hard_head = nn.Conv2d(config.dim, 1, kernel_size=1)
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
            pred = anchor + raw
            scale = torch.ones_like(raw)
            return pred, scale
        rho = F.softplus(self.bound_rho)[None, :, None, None, None]
        acc = batch["acc_map"][:, None].abs()
        amp_map = batch["amp_map"][:, None].abs()
        hard = batch["hard_case_prior"].abs()
        scale = rho * (
            self.config.bound_eps_mm
            + self.config.bound_scale_lambda * (batch["scale_mm"] + 0.20 * hard)
            + self.config.bound_acc_lambda * acc
            + self.config.bound_amp_lambda * amp_map
        )
        pred_delta = scale * torch.tanh(raw)
        pred = anchor + pred_delta
        return pred, scale

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        regime_onehot = self._regime_onehot(batch["regime"])
        raw_inputs = torch.cat(
            [
                batch["last_disp_mm"],
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

        b, h_out = batch["expert_mean_mm"].shape[:2]
        hard_inputs = torch.cat(
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
        hard_state = self.hard_encoder(hard_inputs.reshape(b * h_out, hard_inputs.shape[2], 64, 64))
        hard_raw = self.hard_head(hard_state).reshape(b, h_out, 1, 64, 64)

        gate_inputs = torch.cat(
            [
                batch["expert_std_mm"],
                batch["expert_spread_mm"],
                batch["hard_case_prior"],
            ],
            dim=2,
        )
        q = torch.sigmoid(self.gate_head(gate_inputs.reshape(b * h_out, gate_inputs.shape[2], 64, 64))).reshape(b, h_out, 1, 64, 64)
        q = torch.clamp(q + self.config.comex_gate_bias, 0.0, 1.0)
        raw = (1.0 - q) * cons_raw + q * hard_raw
        anchor = batch["m1_expert_mm"]
        pred, scale = self._bounded(batch, raw, anchor)
        aux = {
            "scale_mm": scale,
            "anchor_mm": anchor,
            "hard_prior": batch["hard_case_prior"],
            "selector_q": q,
        }
        if return_aux:
            return pred, aux
        return pred


def build_model(config: NarrowConfig) -> nn.Module:
    if config.family == "RCDN":
        return RCDNModel(config)
    if config.family == "CoMEx-B":
        return CoMExBModel(config)
    raise KeyError(config.family)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None]
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None]
    err = (pred - target) ** 2
    return (err * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def compute_loss(pred: torch.Tensor, batch: dict[str, torch.Tensor], aux: dict[str, torch.Tensor], config: NarrowConfig) -> tuple[torch.Tensor, float]:
    target = batch["target_mm"]
    mask = batch["mask"]
    loss = masked_smooth_l1(pred, target, mask)
    if config.family == "RCDN":
        latent_loss = F.mse_loss(aux["latent_pred"], aux["latent_target"])
        err = (pred - target) ** 2
        uncert = torch.clamp(aux["uncertainty"], min=1e-3)
        nll = 0.5 * (err / (uncert**2) + 2.0 * torch.log(uncert))
        weights = mask[:, None, None]
        nll = (nll * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)
        loss = loss + config.rcdn_latent_lambda * latent_loss + config.rcdn_uncert_lambda * nll
    elif config.family == "CoMEx-B":
        weights = mask[:, None, None] * (1.0 + aux["hard_prior"])
        hard_term = F.smooth_l1_loss(pred, target, reduction="none")
        loss = loss + config.comex_hard_lambda * (hard_term * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)
        pred_delta = (pred - aux["anchor_mm"]).reshape(pred.shape[0], pred.shape[1], -1)
        m1_delta = (batch["target_mm"] - aux["anchor_mm"]).reshape(pred.shape[0], pred.shape[1], -1)
        hard_flat = aux["hard_prior"].reshape(pred.shape[0], pred.shape[1], -1)
        pred_hard = pred_delta * hard_flat
        m1_hard = m1_delta * hard_flat
        cos = F.cosine_similarity(pred_hard, m1_hard, dim=-1)
        loss = loss + config.comex_ortho_lambda * F.relu(cos).mean()
    model_mse = masked_mse(pred, target, mask)
    pers_mse = masked_mse(batch["persistence_mm"], target, mask)
    skill_mse = 1.0 - float(model_mse.detach().item() / max(pers_mse.detach().item(), 1e-6))
    return loss, skill_mse


def loader_kwargs(config: NarrowConfig, shuffle: bool) -> dict[str, Any]:
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    config: NarrowConfig,
    device: torch.device,
) -> dict[str, float]:
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
            loss, skill_mse = compute_loss(pred, batch, aux, config)
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
    return {"loss": float(np.mean(losses)) if losses else float("nan"), "skill_mse": float(np.mean(skills)) if skills else float("nan")}


def coverage_from_arrays(pred: np.ndarray, truth: np.ndarray, uncert: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    valid = mask > 0.5
    abs_err = np.abs(pred - truth)
    sigma = np.maximum(uncert, 1e-3)
    flat_valid = valid.reshape(-1)
    abs_flat = abs_err.reshape(abs_err.shape[0], abs_err.shape[1], -1)
    sigma_flat = sigma.reshape(sigma.shape[0], sigma.shape[1], -1)
    abs_sel = abs_flat[:, :, flat_valid]
    sigma_sel = sigma_flat[:, :, flat_valid]
    cover68 = float((abs_sel <= sigma_sel).mean())
    cover90 = float((abs_sel <= 1.64 * sigma_sel).mean())
    corr = float(np.corrcoef(abs_sel.reshape(-1), sigma_sel.reshape(-1))[0, 1]) if flat_valid.sum() > 4 else float("nan")
    return cover68, cover90, corr


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, config: NarrowConfig) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    pers_list: list[np.ndarray] = []
    ls_list: list[np.ndarray] = []
    scale_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    regime_list: list[np.ndarray] = []
    current_frame_list: list[np.ndarray] = []
    target_index_list: list[np.ndarray] = []
    uncertainty_list: list[np.ndarray] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            if config.family == "RCDN":
                pred, uncert = model.sample_predictions(batch)
                scale = uncert
            else:
                pred, aux = model(batch, return_aux=True)
                uncert = aux["selector_q"] if "selector_q" in aux else aux["scale_mm"]
                scale = aux["scale_mm"]
            pred_list.append(pred.cpu().numpy()[:, :, 0].astype(np.float32))
            target_list.append(batch["target_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            pers_list.append(batch["persistence_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            ls_list.append(batch["linear_seasonal_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            scale_list.append(scale.cpu().numpy()[:, :, 0].astype(np.float32))
            uncertainty_list.append(uncert.cpu().numpy()[:, :, 0].astype(np.float32))
            mask_list.append(batch["mask"].cpu().numpy().astype(np.float32))
            regime_list.append(batch["regime"].cpu().numpy().astype(np.int16))
            current_frame_list.append(batch["current_frame"].cpu().numpy().astype(np.int64))
            target_index_list.append(batch["target_indices"].cpu().numpy().astype(np.int64))
    infer_seconds = time.time() - start
    arrays = {
        "y_pred_mm": np.concatenate(pred_list, axis=0),
        "y_true_mm": np.concatenate(target_list, axis=0),
        "persistence_mm": np.concatenate(pers_list, axis=0),
        "linear_seasonal_mm": np.concatenate(ls_list, axis=0),
        "scale_mm": np.concatenate(scale_list, axis=0),
        "uncertainty_mm": np.concatenate(uncertainty_list, axis=0),
        "valid_mask": mask_list[0][0] if mask_list and mask_list[0].ndim == 3 else mask_list[0],
        "regime_map": regime_list[0][0] if regime_list and regime_list[0].ndim == 3 else regime_list[0],
        "current_frames": np.concatenate(current_frame_list, axis=0),
        "target_indices": np.concatenate(target_index_list, axis=0),
    }
    return arrays, infer_seconds


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
            arrays={
                "y_pred_mm": arrays["y_pred_mm"],
                "y_true_mm": arrays["y_true_mm"],
                "persistence_mm": arrays["persistence_mm"],
                "linear_seasonal_mm": arrays["linear_seasonal_mm"],
                "valid_mask": arrays["valid_mask"],
                "regime_map": arrays["regime_map"],
                "current_frames": arrays["current_frames"],
                "target_indices": arrays["target_indices"],
            },
            horizons=SEARCH_HORIZONS,
        )
    )


def train_eval_single_tile(
    bundle: TileBundle,
    config: NarrowConfig,
    output_dir: Path,
    phase: str,
    bundles: dict[str, TileBundle],
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    result_path = output_dir / "result.json"
    val_pred_path = output_dir / "val_predictions.npz"
    test_pred_path = output_dir / "test_predictions.npz"
    ckpt_path = output_dir / "model.pth"
    if reuse_if_exists and all(path.exists() for path in [result_path, val_pred_path, test_pred_path, ckpt_path]):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        val_arrays = np.load(val_pred_path)
        test_arrays = np.load(test_pred_path)
        return {"result": result, "val_arrays": {name: val_arrays[name] for name in val_arrays.files}, "test_arrays": {name: test_arrays[name] for name in test_arrays.files}}

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_pack = build_comexb_feature_pack(bundles, bundle.tile_id, config.seed) if config.family == "CoMEx-B" else None
    train_ds = NarrowDataset(bundle, "train", config, None if expert_pack is None else expert_pack["train"])
    val_ds = NarrowDataset(bundle, "val", config, None if expert_pack is None else expert_pack["val"])
    test_ds = NarrowDataset(bundle, "test", config, None if expert_pack is None else expert_pack["test"])
    model = build_model(config).to(device)
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, config, device)
        val_stats = run_epoch(model, val_loader, None, None, config, device)
        history.append({"epoch": epoch, "train_loss": train_stats["loss"], "train_skill_mse": train_stats["skill_mse"], "val_loss": val_stats["loss"], "val_skill_mse": val_stats["skill_mse"]})
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
    val_arrays, val_infer = collect_predictions(model, val_loader, device, config)
    test_arrays, test_infer = collect_predictions(model, test_loader, device, config)
    elapsed = time.time() - start
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
    val_rows = rows_from_arrays(config.model_id, bundle.tile_id, config.seed, f"{phase}_val", val_arrays, elapsed, val_infer, param_count)
    test_rows = rows_from_arrays(config.model_id, bundle.tile_id, config.seed, f"{phase}_test", test_arrays, elapsed, test_infer, param_count)
    val_rows["Peak GPU memory [MB]"] = peak_mb
    test_rows["Peak GPU memory [MB]"] = peak_mb
    if config.family == "RCDN":
        c68, c90, corr = coverage_from_arrays(test_arrays["y_pred_mm"], test_arrays["y_true_mm"], test_arrays["uncertainty_mm"], test_arrays["valid_mask"])
    else:
        c68, c90, corr = float("nan"), float("nan"), float("nan")
    result = {
        "model_id": config.model_id,
        "family": config.family,
        "tile": bundle.tile_id,
        "seed": config.seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": elapsed,
        "val_infer_seconds": val_infer,
        "test_infer_seconds": test_infer,
        "param_count": param_count,
        "peak_gpu_memory_mb": peak_mb,
        "coverage68": c68,
        "coverage90": c90,
        "uncertainty_error_corr": corr,
        "history": history,
        "val_rows": val_rows.to_dict(orient="records"),
        "test_rows": test_rows.to_dict(orient="records"),
    }
    np.savez_compressed(val_pred_path, **val_arrays)
    np.savez_compressed(test_pred_path, **test_arrays)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"result": result, "val_arrays": val_arrays, "test_arrays": test_arrays}


def candidate_run_path(stage: str, config: NarrowConfig, tile: str, seed: int) -> Path:
    return RUN_ROOT / stage / config.model_id / tile / f"seed_{seed}"


def make_augmented_cache(base_cache: dict[str, np.ndarray], candidate_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not np.array_equal(base_cache["current_frames"].reshape(-1), candidate_arrays["current_frames"].reshape(-1)):
        raise ValueError("current_frames mismatch")
    if not np.array_equal(base_cache["target_indices"], candidate_arrays["target_indices"]):
        raise ValueError("target_indices mismatch")
    out = dict(base_cache)
    out["candidate_pred_mm"] = candidate_arrays["y_pred_mm"].astype(np.float32)
    return out


def expert_stack_from_augmented_cache(cache: dict[str, np.ndarray], include_candidate: bool = True) -> np.ndarray:
    items = [
        cache["persistence_mm"],
        cache["linear_seasonal_mm"],
        cache["m1_pred_mm"],
        cache["c3_pred_mm"],
    ]
    if include_candidate:
        items.append(cache["candidate_pred_mm"])
    return np.stack(items, axis=2).astype(np.float32)


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


def plot_heatmap(summary_df: pd.DataFrame, out_path: Path, title: str, tiles: list[str]) -> None:
    pivot = summary_df.pivot(index="Tile", columns="Horizon", values="Skill_MSE mean").reindex(tiles)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def load_m1_full6_summary() -> pd.DataFrame:
    path = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"
    return pd.read_csv(path)


def load_vcea_full6_metrics() -> dict[str, float]:
    path = OUTPUT_ROOT / "staticval_ea_study" / "final_aggregation_comparison.csv"
    df = pd.read_csv(path)
    row = df[df["Model"].astype(str).str.contains("StaticVal", regex=False)].iloc[0]
    return {"mean_skill_mse": float(row["Mean Skill_MSE"]), "stress_ratio": float(row["Stress ratio"])}


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def summary_to_tile_gain(summary_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    merged = summary_df.merge(
        ref_df[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "Ref Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs Ref"] = merged["Skill_MSE mean"] - merged["Ref Skill_MSE mean"]
    return merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "Ref Skill_MSE mean", "Delta vs Ref"]].mean()


def evaluate_quick_record(
    config: NarrowConfig,
    candidate_summary: pd.DataFrame,
    add_summary: pd.DataFrame,
    m1_ref: pd.DataFrame,
    vcea_ref: pd.DataFrame,
    runtime_df: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_stats = compare_summary_to_reference(candidate_summary, m1_ref, SEARCH_TILES)
    add_stats = compare_summary_to_reference(add_summary, vcea_ref, SEARCH_TILES)
    alone_pass = candidate_stats["gain_vs_ref"] >= 0.02 and candidate_stats["better_tiles_vs_ref"] >= 2 and candidate_stats["stress_ratio"] <= 1.05
    add_pass = add_stats["gain_vs_ref"] >= 0.015 and add_stats["better_tiles_vs_ref"] >= 2 and add_stats["stress_ratio"] <= 1.05
    record = {
        "Model": config.model_id,
        "Family": config.family,
        "Candidate Mean Skill_MSE": candidate_stats["mean_skill_mse"],
        "Candidate gain vs M1": candidate_stats["gain_vs_ref"],
        "VCEA+candidate Mean Skill_MSE": add_stats["mean_skill_mse"],
        "VCEA+candidate gain vs VCEA": add_stats["gain_vs_ref"],
        "Better quick tiles": int(max(candidate_stats["better_tiles_vs_ref"], add_stats["better_tiles_vs_ref"])),
        "Candidate stress ratio": candidate_stats["stress_ratio"],
        "Addition stress ratio": add_stats["stress_ratio"],
        "Stress ratio": float(min(candidate_stats["stress_ratio"], add_stats["stress_ratio"])),
        "Candidate mean train seconds": float(runtime_df["Elapsed seconds"].mean()) if not runtime_df.empty else float("nan"),
        "Quick gate pass": bool(alone_pass or add_pass),
        "Pass branch": "candidate" if alone_pass and not add_pass else "addition" if add_pass else "none",
    }
    if extra:
        record.update(extra)
    return record


def evaluate_fullsix_outcome(
    candidate_summary: pd.DataFrame,
    add_summary: pd.DataFrame,
    safe_summary: pd.DataFrame,
    runtime_df: pd.DataFrame,
) -> dict[str, Any]:
    m1_ref = load_m1_full6_summary()
    vcea = load_vcea_full6_metrics()
    candidate_stats = compare_summary_to_reference(candidate_summary, m1_ref, TILE_ORDER)
    add_gain = float(add_summary["Skill_MSE mean"].mean() - vcea["mean_skill_mse"])
    safe_gain = float(safe_summary["Skill_MSE mean"].mean() - vcea["mean_skill_mse"])
    better_tiles_vs_vcea_proxy = int((summary_to_tile_gain(add_summary, pd.DataFrame({"Tile": TILE_ORDER * len(COMMON_HORIZONS), "Horizon": np.tile([f"t+{h}" for h in COMMON_HORIZONS], len(TILE_ORDER)), "Skill_MSE mean": vcea["mean_skill_mse"]})).get("Delta vs Ref", pd.Series(dtype=float)) > 0).sum())
    candidate_mean = float(candidate_summary["Skill_MSE mean"].mean())
    add_mean = float(add_summary["Skill_MSE mean"].mean())
    safe_mean_skill = float(safe_summary["Skill_MSE mean"].mean())
    safe_stress = float((safe_summary[safe_summary["Tile"].isin(STRESS_TILES)]["RMSE [mm] mean"] / safe_summary[safe_summary["Tile"].isin(STRESS_TILES)]["Persistence RMSE [mm] mean"]).max())
    verdict = "no_go"
    if candidate_mean >= vcea["mean_skill_mse"] - 0.02 and add_gain >= 0.02 and safe_stress <= max(1.03, vcea["stress_ratio"]):
        verdict = "main_model_plus_expert"
    elif candidate_stats["gain_vs_ref"] >= 0.05 and add_gain >= 0.02:
        verdict = "strong_new_expert"
    elif add_gain >= 0.005:
        verdict = "supplementary_only"
    return {
        "candidate_model": str(candidate_summary["Model"].iloc[0]),
        "candidate_mean_skill_mse": candidate_mean,
        "candidate_gain_vs_m1": candidate_stats["gain_vs_ref"],
        "vcea_plus_candidate_mean_skill_mse": add_mean,
        "vcea_plus_candidate_gain_vs_vcea": add_gain,
        "vcea_safe_plus_candidate_mean_skill_mse": safe_mean_skill,
        "vcea_safe_plus_candidate_gain_vs_vcea": safe_gain,
        "candidate_stress_ratio": candidate_stats["stress_ratio"],
        "vcea_plus_candidate_stress_ratio": float((add_summary[add_summary["Tile"].isin(STRESS_TILES)]["RMSE [mm] mean"] / add_summary[add_summary["Tile"].isin(STRESS_TILES)]["Persistence RMSE [mm] mean"]).max()),
        "vcea_safe_plus_candidate_stress_ratio": safe_stress,
        "mean_train_seconds": float(runtime_df["Elapsed seconds"].mean()) if not runtime_df.empty else float("nan"),
        "mean_infer_seconds": float(runtime_df["Infer seconds"].mean()) if not runtime_df.empty else float("nan"),
        "mean_skill_std": float(candidate_summary["Skill_MSE std"].mean()),
        "fullsix_verdict": verdict,
        "better_tiles_vs_vcea_proxy": better_tiles_vs_vcea_proxy,
    }


def rcdn_safe_summary_from_uncertainty(
    tile: str,
    seed: int,
    base_val: dict[str, np.ndarray],
    base_test: dict[str, np.ndarray],
    val_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    add_pred_val: np.ndarray,
    add_pred_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, float], dict[str, float]]:
    percentiles = [50, 60, 70, 80, 90]
    valid = base_val["valid_mask"] > 0.5
    thresholds: dict[int, float] = {}
    for h_idx, horizon in enumerate(COMMON_HORIZONS):
        best_tau = float(np.percentile(val_arrays["uncertainty_mm"][:, h_idx][:, valid], 70))
        best_skill = -float("inf")
        for pct in percentiles:
            tau = float(np.percentile(val_arrays["uncertainty_mm"][:, h_idx][:, valid], pct))
            safe_pred = np.where(val_arrays["uncertainty_mm"][:, h_idx] > tau, base_val["m1_pred_mm"][:, h_idx], add_pred_val[:, h_idx]).astype(np.float32)
            model_mse = float(np.mean((safe_pred[:, valid] - base_val["y_true_mm"][:, h_idx][:, valid]) ** 2))
            pers_mse = float(np.mean((base_val["persistence_mm"][:, h_idx][:, valid] - base_val["y_true_mm"][:, h_idx][:, valid]) ** 2))
            skill = 1.0 - model_mse / max(pers_mse, 1e-6)
            if skill > best_skill:
                best_skill = skill
                best_tau = tau
        thresholds[horizon] = best_tau
    safe_test = add_pred_test.copy().astype(np.float32)
    for h_idx, horizon in enumerate(COMMON_HORIZONS):
        risky = test_arrays["uncertainty_mm"][:, h_idx] > thresholds[horizon]
        safe_test[:, h_idx][risky] = base_test["m1_pred_mm"][:, h_idx][risky]
    safe_rows = aggregate_addition_rows("VCEA-safe+RCDN", tile, seed, base_test, safe_test)
    stats = {
        "safe_cover68": coverage_from_arrays(test_arrays["y_pred_mm"], test_arrays["y_true_mm"], test_arrays["uncertainty_mm"], test_arrays["valid_mask"])[0],
        "safe_cover90": coverage_from_arrays(test_arrays["y_pred_mm"], test_arrays["y_true_mm"], test_arrays["uncertainty_mm"], test_arrays["valid_mask"])[1],
        "safe_corr": coverage_from_arrays(test_arrays["y_pred_mm"], test_arrays["y_true_mm"], test_arrays["uncertainty_mm"], test_arrays["valid_mask"])[2],
    }
    return safe_rows, thresholds, stats


def str_mamba_env_check() -> dict[str, Any]:
    attempts = [
        "conda run -n py311 pip install --no-build-isolation mamba-ssm causal-conv1d",
        "conda run -n py311 pip install --no-build-isolation mamba-ssm==2.2.2 causal-conv1d==1.5.3.post1",
    ]
    results: list[dict[str, Any]] = []
    for command in attempts:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=str(Path.cwd()))
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-20:])
        results.append({"command": command, "returncode": int(proc.returncode), "tail": tail})
        if proc.returncode == 0:
            break
    installed = False
    try:
        import importlib.util

        installed = importlib.util.find_spec("mamba_ssm") is not None
    except Exception:
        installed = False
    verdict = "available" if installed else "blocked"
    return {"installed": installed, "verdict": verdict, "attempts": results}


def run_quick_candidate(
    bundles: dict[str, TileBundle],
    config: NarrowConfig,
    m1_ref: pd.DataFrame,
    vcea_ref: pd.DataFrame,
    vcea_safe_ref: pd.DataFrame,
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    candidate_rows: list[pd.DataFrame] = []
    add_rows: list[pd.DataFrame] = []
    safe_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    uncert_rows: list[dict[str, Any]] = []
    for seed in QUICK_SEEDS:
        cfg = replace(config, seed=seed)
        for tile in SEARCH_TILES:
            run = train_eval_single_tile(bundles[tile], cfg, candidate_run_path("quick", cfg, tile, seed), "quick", bundles, reuse_if_exists=reuse_if_exists)
            candidate_rows.append(pd.DataFrame(run["result"]["test_rows"]))
            runtime_rows.append(
                {
                    "Model": cfg.model_id,
                    "Family": cfg.family,
                    "Tile": tile,
                    "Seed": seed,
                    "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                    "Infer seconds": float(run["result"]["test_infer_seconds"]),
                    "Param count": int(run["result"]["param_count"]),
                    "Peak GPU memory [MB]": float(run["result"]["peak_gpu_memory_mb"]),
                }
            )
            if cfg.family == "RCDN":
                uncert_rows.append(
                    {
                        "Model": cfg.model_id,
                        "Tile": tile,
                        "Seed": seed,
                        "coverage_68": float(run["result"]["coverage68"]),
                        "coverage_90": float(run["result"]["coverage90"]),
                        "uncertainty_error_corr": float(run["result"]["uncertainty_error_corr"]),
                    }
                )
            base_val = build_seed_expert_cache(bundles, tile, seed, "val")
            base_test = build_seed_expert_cache(bundles, tile, seed, "test")
            val_aug = make_augmented_cache(base_val, run["val_arrays"])
            test_aug = make_augmented_cache(base_test, run["test_arrays"])
            val_experts = expert_stack_from_augmented_cache(val_aug)
            test_experts = expert_stack_from_augmented_cache(test_aug)
            weights = estimate_static_weights_generic(val_aug["y_true_mm"], val_experts, val_aug["valid_mask"], best_static_cfg())
            pred_add_val = predict_static_generic(val_experts, weights)
            pred_add_test = predict_static_generic(test_experts, weights)
            add_rows.append(aggregate_addition_rows(f"VCEA+{cfg.model_id}", tile, seed, test_aug, pred_add_test))
            if cfg.family == "RCDN":
                safe_rows_df, thresholds, stats = rcdn_safe_summary_from_uncertainty(tile, seed, base_val, base_test, run["val_arrays"], run["test_arrays"], pred_add_val, pred_add_test)
                safe_rows_df["Model"] = f"VCEA-safe+{cfg.model_id}"
                safe_rows.append(safe_rows_df)
                uncert_rows.append(
                    {
                        "Model": cfg.model_id,
                        "Tile": tile,
                        "Seed": seed,
                        "threshold_t1": thresholds[1],
                        "threshold_t3": thresholds[3],
                        "threshold_t5": thresholds[5],
                        "threshold_t10": thresholds[10],
                        "safe_cover68": stats["safe_cover68"],
                        "safe_cover90": stats["safe_cover90"],
                        "safe_corr": stats["safe_corr"],
                    }
                )
    candidate_raw = pd.concat(candidate_rows, ignore_index=True)
    add_raw = pd.concat(add_rows, ignore_index=True)
    candidate_summary = summarize_seeded_rows(candidate_raw)
    add_summary = summarize_seeded_rows(add_raw)
    runtime_df = pd.DataFrame(runtime_rows)
    extra: dict[str, Any] = {}
    safe_summary_df = pd.DataFrame()
    if safe_rows:
        safe_summary_df = pd.concat(safe_rows, ignore_index=True)
        safe_summary = summarize_seeded_rows(safe_summary_df)
        safe_gain = float(safe_summary["Skill_MSE mean"].mean() - vcea_safe_ref["Skill_MSE mean"].mean())
        safe_stress = float((safe_summary[safe_summary["Tile"].isin(["E48N24"])]["RMSE [mm] mean"] / safe_summary[safe_summary["Tile"].isin(["E48N24"])]["Persistence RMSE [mm] mean"]).max())
        extra.update(
            {
                "VCEA-safe+candidate Mean Skill_MSE": float(safe_summary["Skill_MSE mean"].mean()),
                "VCEA-safe+candidate gain vs VCEA-safe": safe_gain,
                "VCEA-safe stress ratio": safe_stress,
                "RCDN safe quick signal": bool(safe_gain >= 0.005 and safe_stress <= 1.03),
            }
        )
    record = evaluate_quick_record(config, candidate_summary, add_summary, m1_ref, vcea_ref, runtime_df, extra=extra)
    return {
        "candidate_raw": candidate_raw,
        "candidate_summary": candidate_summary,
        "add_raw": add_raw,
        "add_summary": add_summary,
        "safe_summary": safe_summary_df,
        "runtime": runtime_df,
        "uncertainty": pd.DataFrame(uncert_rows),
        "quick_record": record,
    }


def default_rcdn_config() -> NarrowConfig:
    return NarrowConfig(
        model_id="RCDN_v1",
        family="RCDN",
        dim=48,
        latent_dim=32,
        batch_size=8,
        epochs=10,
        patience=3,
        rcdn_mc_samples=2,
    )


def default_comexb_config() -> NarrowConfig:
    return NarrowConfig(
        model_id="CoMEx_B",
        family="CoMEx-B",
        dim=48,
        batch_size=8,
        epochs=10,
        patience=3,
        comex_hard_lambda=0.08,
        comex_ortho_lambda=0.02,
    )


def write_protocol_files() -> list[Path]:
    protocol_path = write_markdown(
        ACT_V8_NARROW_ROOT / "v8_protocol.md",
        [
            "# V8 Narrow Protocol",
            "",
            "- 数据统一使用 EGMS 2019-2023 Ortho U，原因是数据获取限制。",
            "- quick 固定为 E32N34 / E39N30 / E48N24，2 seeds。",
            "- full-six 只对通过 quick gate 的候选运行，固定 3 seeds。",
            "- 统一 baselines: persistence, linear-seasonal, M1, C3_opt, VCEA, VCEA-safe, candidate-alone, VCEA+candidate。",
            "- 统一 bounded residual head；所有 validation-only calibration 不允许使用 test 调参。",
        ],
    )
    registry_path = ACT_V8_NARROW_ROOT / "common_input_registry.yaml"
    registry_path.write_text(
        "\n".join(
            [
                "inputs:",
                "  x_main: displacement history + increments + seasonal channels",
                "  desc_stack: train-window velocity/acc/amplitude/trend_r2 z-score maps",
                "  regime: six-class regime map derived from train-safe descriptors",
                "  persistence_mm: causal persistence anchor",
                "  linear_seasonal_mm: causal linear+seasonal baseline",
                "  future_time: horizon-conditioned sin/cos season encoding",
                "  expert_features: optional expert disagreement / hard-case pack for CoMEx-B",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    contract_path = write_markdown(
        ACT_V8_NARROW_ROOT / "eval_contract.md",
        [
            "# Eval Contract",
            "",
            "- Quick gate: candidate-alone gain vs M1 >= +0.02 or VCEA+candidate gain vs VCEA >= +0.015, stress ratio <= 1.05, better quick tiles >= 2/3。",
            "- Full-six GO: candidate-alone 接近 VCEA 或 VCEA+candidate >= +0.02，且 stress 不高于 1.03 或显著优于 VCEA。",
            "- 强制 No-Go: gain vs VCEA < +0.005, stress ratio > 1.05, better tiles <= 2/6, 或需要 test tuning。",
        ],
    )
    return [protocol_path, registry_path, contract_path]


def write_env_block_report(env_status: dict[str, Any]) -> Path:
    lines = [
        "# STR-Mamba-v2 环境审计",
        "",
        "数据声明：本轮继续统一使用 EGMS 2019-2023 Ortho U，原因是数据获取限制。",
        "",
        f"- 最终环境判定：`{env_status['verdict']}`。",
        f"- 是否成功获得 `mamba_ssm`：`{env_status['installed']}`。",
        "",
        "## 安装尝试",
        "",
    ]
    for item in env_status["attempts"]:
        lines.append(f"- 命令：`{item['command']}`")
        lines.append(f"  返回码：`{item['returncode']}`")
        lines.append("  末尾日志：")
        lines.append("```text")
        lines.append(item["tail"])
        lines.append("```")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 按研究报告要求，STR-Mamba-v2 不接受 GRU proxy 继续冒充真正 selective scan。",
            "- 在当前 Windows + py311 + CUDA 环境下，`mamba-ssm` 官方安装未落地，因此本轮将 STR-Mamba-v2 记为外源环境阻塞，而不是继续沿用旧 proxy quick 结果。",
        ]
    )
    return write_markdown(ACT_V8_NARROW_ROOT / "strmamba_env_block.md", lines)


def write_final_report(
    env_status: dict[str, Any],
    quick_df: pd.DataFrame,
    fullsix_rows: list[dict[str, Any]],
    final_verdict: str,
    notes: list[str],
) -> Path:
    lines = [
        "# EGMS 新架构深度研究报告实验记录",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        "## 执行摘要",
        "",
        f"- STR-Mamba-v2 环境判定：`{env_status['verdict']}`。",
        f"- 最终结论：`{final_verdict}`。",
        "",
        "## 关键说明",
        "",
    ]
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Quick 结果", ""])
    for _, row in quick_df.iterrows():
        parts = [
            f"`{row['Model']}` ({row['Family']})",
            f"candidate gain vs M1 = {row['Candidate gain vs M1']:.4f}",
            f"VCEA+candidate gain vs VCEA = {row['VCEA+candidate gain vs VCEA']:.4f}",
            f"stress ratio = {row['Stress ratio']:.4f}",
            f"better tiles = {int(row['Better quick tiles'])}/3",
            f"quick gate = {bool(row['Quick gate pass'])}",
        ]
        if "VCEA-safe+candidate gain vs VCEA-safe" in row and not pd.isna(row["VCEA-safe+candidate gain vs VCEA-safe"]):
            parts.append(f"VCEA-safe gain = {row['VCEA-safe+candidate gain vs VCEA-safe']:.4f}")
        lines.append("- " + "，".join(parts) + "。")
    if fullsix_rows:
        lines.extend(["", "## Full-six 结果", ""])
        for item in fullsix_rows:
            lines.append(
                f"- `{item['candidate_model']}`: alone = {item['candidate_mean_skill_mse']:.4f}, "
                f"add gain vs VCEA = {item['vcea_plus_candidate_gain_vs_vcea']:.4f}, "
                f"safe stress = {item['vcea_safe_plus_candidate_stress_ratio']:.4f}, verdict = `{item['fullsix_verdict']}`。"
            )
    else:
        lines.extend(["", "## Full-six 结果", "", "- 本轮没有候选通过 quick gate，因此未触发 full-six。"])
    lines.extend(
        [
            "",
            "## 关键文件",
            "",
            f"- 协议：`{(ACT_V8_NARROW_ROOT / 'v8_protocol.md').as_posix()}`",
            f"- 环境审计：`{(ACT_V8_NARROW_ROOT / 'strmamba_env_block.md').as_posix()}`",
            f"- quick 排名：`{(ACT_V8_NARROW_ROOT / 'quick_rank_table.csv').as_posix()}`",
            f"- 决策：`{(ACT_V8_NARROW_ROOT / 'final_decision.md').as_posix()}`",
        ]
    )
    return write_markdown(ACT_V8_NARROW_ROOT / "act_v8_narrow_detailed_report.md", lines)


def run_act_v8_narrow(reuse_if_exists: bool = True) -> dict[str, Any]:
    protocol_files = write_protocol_files()
    bundles = preprocess_all_tiles()
    env_status = str_mamba_env_check()
    env_report = write_env_block_report(env_status)
    m1_quick_ref, vcea_quick_ref, vcea_safe_quick_ref = reference_quick_m1_vcea(bundles, seed=42)
    quick_records: list[dict[str, Any]] = []
    fullsix_rows: list[dict[str, Any]] = []
    notes: list[str] = [
        "按新文档要求，KINO-v2 不再继续主线探索；本轮仅做 STR-Mamba-v2 环境审计、RCDN 主线与条件触发的 CoMEx-B backup。",
        "由于 mamba-ssm 在当前 Windows 环境下未成功安装，STR-Mamba-v2 没有继续沿用 GRU proxy，而是直接记为环境阻塞。",
    ]

    rcdn_cfg = default_rcdn_config()
    rcdn = run_quick_candidate(bundles, rcdn_cfg, m1_quick_ref, vcea_quick_ref, vcea_safe_quick_ref, reuse_if_exists=reuse_if_exists)
    rcdn["candidate_summary"].to_csv(QUICK_ROOT / "rcdn_quick_summary.csv", index=False, encoding="utf-8-sig")
    rcdn["candidate_summary"].to_csv(QUICK_ROOT / "rcdn_tile_horizon.csv", index=False, encoding="utf-8-sig")
    rcdn["add_summary"].to_csv(QUICK_ROOT / "vcea_plus_rcdn.csv", index=False, encoding="utf-8-sig")
    rcdn["runtime"].to_csv(QUICK_ROOT / "rcdn_runtime_memory.csv", index=False, encoding="utf-8-sig")
    rcdn["uncertainty"].to_csv(QUICK_ROOT / "rcdn_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    plot_heatmap(rcdn["candidate_summary"], FIG_ROOT / "rcdn_heatmap.png", "RCDN quick Skill_MSE", SEARCH_TILES)
    quick_records.append(rcdn["quick_record"])

    selected_models = [rcdn_cfg.model_id] if bool(rcdn["quick_record"]["Quick gate pass"]) else []
    run_comex = (not selected_models) or (not env_status["installed"])
    comex = None
    if run_comex:
        comex_cfg = default_comexb_config()
        comex = run_quick_candidate(bundles, comex_cfg, m1_quick_ref, vcea_quick_ref, vcea_safe_quick_ref, reuse_if_exists=reuse_if_exists)
        comex["candidate_summary"].to_csv(QUICK_ROOT / "comex_b_quick_summary.csv", index=False, encoding="utf-8-sig")
        comex["add_summary"].to_csv(QUICK_ROOT / "vcea_plus_comex_b.csv", index=False, encoding="utf-8-sig")
        comex["runtime"].to_csv(QUICK_ROOT / "comex_b_runtime_memory.csv", index=False, encoding="utf-8-sig")
        plot_heatmap(comex["candidate_summary"], FIG_ROOT / "comex_b_heatmap.png", "CoMEx-B quick Skill_MSE", SEARCH_TILES)
        quick_records.append(comex["quick_record"])
        if bool(comex["quick_record"]["Quick gate pass"]):
            selected_models.append(comex_cfg.model_id)

    quick_df = pd.DataFrame(quick_records).sort_values(
        ["Quick gate pass", "VCEA+candidate gain vs VCEA", "Candidate gain vs M1", "Stress ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    quick_df.to_csv(ACT_V8_NARROW_ROOT / "quick_rank_table.csv", index=False, encoding="utf-8-sig")

    final_verdict = "stop_after_quick_return_to_vcea"
    if selected_models:
        config_map = {rcdn_cfg.model_id: rcdn_cfg}
        if run_comex:
            config_map[default_comexb_config().model_id] = default_comexb_config()
        for model_id in selected_models:
            cfg = config_map[model_id]
            candidate_rows: list[pd.DataFrame] = []
            add_rows: list[pd.DataFrame] = []
            safe_rows: list[pd.DataFrame] = []
            runtime_rows: list[dict[str, Any]] = []
            for seed in FINAL_SEEDS:
                cfg_seed = replace(cfg, seed=seed)
                for tile in TILE_ORDER:
                    run = train_eval_single_tile(bundles[tile], cfg_seed, candidate_run_path("fullsix", cfg_seed, tile, seed), "fullsix", bundles, reuse_if_exists=reuse_if_exists)
                    candidate_rows.append(pd.DataFrame(run["result"]["test_rows"]))
                    runtime_rows.append(
                        {
                            "Model": cfg_seed.model_id,
                            "Tile": tile,
                            "Seed": seed,
                            "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                            "Infer seconds": float(run["result"]["test_infer_seconds"]),
                        }
                    )
                    base_val = build_seed_expert_cache(bundles, tile, seed, "val")
                    base_test = build_seed_expert_cache(bundles, tile, seed, "test")
                    val_aug = make_augmented_cache(base_val, run["val_arrays"])
                    test_aug = make_augmented_cache(base_test, run["test_arrays"])
                    val_experts = expert_stack_from_augmented_cache(val_aug)
                    test_experts = expert_stack_from_augmented_cache(test_aug)
                    weights = estimate_static_weights_generic(val_aug["y_true_mm"], val_experts, val_aug["valid_mask"], best_static_cfg())
                    pred_add_val = predict_static_generic(val_experts, weights)
                    pred_add_test = predict_static_generic(test_experts, weights)
                    add_rows.append(aggregate_addition_rows(f"VCEA+{cfg_seed.model_id}", tile, seed, test_aug, pred_add_test))
                    if cfg_seed.family == "RCDN":
                        safe_rows_df, _thresholds, _stats = rcdn_safe_summary_from_uncertainty(tile, seed, base_val, base_test, run["val_arrays"], run["test_arrays"], pred_add_val, pred_add_test)
                        safe_rows_df["Model"] = f"VCEA-safe+{cfg_seed.model_id}"
                        safe_rows.append(safe_rows_df)
                    else:
                        disagreement = test_experts.std(axis=2).astype(np.float32)
                        safe_tau = calibrate_safe_thresholds(pred_add_val, base_val["m1_pred_mm"], base_val["y_true_mm"], val_experts.std(axis=2).astype(np.float32), base_val["valid_mask"])
                        pred_safe = apply_safe_wrapper(pred_add_test, base_test["m1_pred_mm"], disagreement, safe_tau)
                        safe_rows.append(aggregate_addition_rows(f"VCEA-safe+{cfg_seed.model_id}", tile, seed, test_aug, pred_safe))
            candidate_summary = summarize_seeded_rows(pd.concat(candidate_rows, ignore_index=True))
            add_summary = summarize_seeded_rows(pd.concat(add_rows, ignore_index=True))
            safe_summary = summarize_seeded_rows(pd.concat(safe_rows, ignore_index=True))
            runtime_df = pd.DataFrame(runtime_rows)
            candidate_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_summary_main.csv", index=False, encoding="utf-8-sig")
            add_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_fullsix_vcea_integration.csv", index=False, encoding="utf-8-sig")
            safe_summary.to_csv(FULL6_ROOT / f"{cfg.model_id}_fullsix_safe_wrapper.csv", index=False, encoding="utf-8-sig")
            runtime_df.to_csv(FULL6_ROOT / f"{cfg.model_id}_runtime_memory.csv", index=False, encoding="utf-8-sig")
            fullsix_rows.append(evaluate_fullsix_outcome(candidate_summary, add_summary, safe_summary, runtime_df))
        if fullsix_rows:
            best = sorted(
                fullsix_rows,
                key=lambda item: (
                    item["fullsix_verdict"] == "main_model_plus_expert",
                    item["fullsix_verdict"] == "strong_new_expert",
                    item["vcea_plus_candidate_gain_vs_vcea"],
                    item["candidate_gain_vs_m1"],
                ),
                reverse=True,
            )[0]
            if best["fullsix_verdict"] == "main_model_plus_expert":
                final_verdict = f"promote_{best['candidate_model']}_as_main_model_plus_expert"
            elif best["fullsix_verdict"] == "strong_new_expert":
                final_verdict = f"keep_vcea_add_{best['candidate_model']}"
            elif best["fullsix_verdict"] == "supplementary_only":
                final_verdict = f"supplementary_only_{best['candidate_model']}"
            else:
                final_verdict = f"no_go_{best['candidate_model']}_return_to_vcea"
    decision_lines = [
        "# V8 Narrow Final Decision",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        f"- STR-Mamba-v2 环境判定：`{env_status['verdict']}`。",
        f"- final verdict：`{final_verdict}`。",
    ]
    if not selected_models:
        decision_lines.append("- 没有候选通过 quick gate，因此按文档 stop 条件返回 VCEA 主线。")
    else:
        decision_lines.append(f"- 进入 full-six 的候选：{', '.join(selected_models)}。")
    final_decision = write_markdown(ACT_V8_NARROW_ROOT / "final_decision.md", decision_lines)
    report = write_final_report(env_status, quick_df, fullsix_rows, final_verdict, notes)
    summary = {
        "protocol_files": [str(path.resolve()) for path in protocol_files],
        "env_report": str(env_report.resolve()),
        "report": str(report.resolve()),
        "quick_rank_table": str((ACT_V8_NARROW_ROOT / "quick_rank_table.csv").resolve()),
        "final_decision": str(final_decision.resolve()),
        "selected_models": selected_models,
        "final_verdict": final_verdict,
    }
    write_json(ACT_V8_NARROW_ROOT / "run_summary.json", summary)
    return summary
