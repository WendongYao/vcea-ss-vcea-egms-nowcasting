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
    FINAL_SEEDS,
    SEARCH_HORIZONS,
    SEARCH_TILES,
    fit_linear_seasonal_train_window,
    predict_linear_seasonal,
    write_simple_yaml,
)
from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    TILE_ORDER,
    TileBundle,
    TransformerBlock,
    compute_metrics,
    ensure_dir,
    get_tensors,
    preprocess_all_tiles,
    seed_everything,
    write_json,
)


ACT_V3_ROOT = OUTPUT_ROOT / "act_v3"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "act_v3"
V3_ERROR_MAP_ROOT = ACT_V3_ROOT / "v3_error_maps"
M1_REFERENCE_PATH = OUTPUT_ROOT / "act_v2" / "final_model_selection.csv"
M1_COMPLEXITY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step2_three_seed_summary.csv"
M0_COMPLEXITY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step2_three_seed_summary.csv"
BRANCH_NAMES = ["level", "increment", "trend", "periodic", "irregular"]


@dataclass
class ActV3Config:
    model_id: str
    variant: str
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    dim: int = 64
    depth: int = 1
    heads: int = 4
    mlp_dim: int = 128
    patch_size: int = 8
    batch_size: int = 8
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    select_by_skill: bool = True
    min_scale_mm: float = 0.15
    horizon_scale_alpha: float = 0.75
    temporal_depth: int = 1
    conv_width: int = 48
    num_workers: int = 0

    @property
    def output_len(self) -> int:
        return len(self.horizons)


def default_v3_configs() -> list[ActV3Config]:
    return [
        ActV3Config(model_id="V3-A", variant="v3a", input_len=16, dim=64, depth=1, batch_size=8, epochs=16, patience=4),
        ActV3Config(model_id="V3-B", variant="v3b", input_len=16, dim=64, depth=1, batch_size=6, epochs=18, patience=5),
        ActV3Config(model_id="V3-C", variant="v3c", input_len=16, dim=48, depth=1, batch_size=6, epochs=16, patience=4, conv_width=48),
    ]


def load_m1_reference() -> dict[str, float]:
    if not M1_REFERENCE_PATH.exists():
        return {
            "mean_skill_mse": 0.044640365517996224,
            "mean_skill_std": 0.005883697389798982,
            "train_seconds": float("nan"),
        }
    df = pd.read_csv(M1_REFERENCE_PATH)
    train_seconds = float(df["Elapsed seconds mean"].mean()) if "Elapsed seconds mean" in df.columns else float("nan")
    return {
        "mean_skill_mse": float(df["Skill_MSE mean"].mean()),
        "mean_skill_std": float(df["Skill_MSE std"].mean()),
        "train_seconds": train_seconds,
    }


M1_REFERENCE = load_m1_reference()
QUICK_SKILL_THRESHOLD = M1_REFERENCE["mean_skill_mse"] + 0.03


def make_lag_delta(disp: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(disp, dtype=np.float32)
    out[lag:] = disp[lag:] - disp[:-lag]
    return out.astype(np.float32)


def rolling_mad(delta: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros_like(delta, dtype=np.float32)
    for idx in range(delta.shape[0]):
        start = max(0, idx - window + 1)
        chunk = delta[start : idx + 1]
        median = np.median(chunk, axis=0)
        out[idx] = np.median(np.abs(chunk - median[None, :, :]), axis=0)
    return out.astype(np.float32)


def window_linear_fit(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(window.shape[0], dtype=np.float32)
    t = t - t.mean()
    denom = float(np.sum(t**2) + 1e-6)
    mean_y = window.mean(axis=0)
    slope = np.tensordot(t, window - mean_y[None, :, :], axes=(0, 0)) / denom
    fit = mean_y[None, :, :] + t[:, None, None] * slope[None, :, :]
    return slope.astype(np.float32), fit.astype(np.float32), (window - fit).astype(np.float32)


class ActV3Dataset(Dataset):
    def __init__(self, bundle: TileBundle, resolution: int, split_name: str, config: ActV3Config) -> None:
        self.bundle = bundle
        self.config = config
        tensors = get_tensors(bundle, resolution)
        disp = tensors["disp"].astype(np.float32)
        height, width = disp.shape[1], disp.shape[2]

        train_end = bundle.split.train_frame_end
        disp_mu = disp[:train_end].mean(axis=0).astype(np.float32)
        disp_sigma = disp[:train_end].std(axis=0).astype(np.float32) + 1e-6

        delta1 = make_lag_delta(disp, 1)
        delta3 = make_lag_delta(disp, 3)
        delta5 = make_lag_delta(disp, 5)
        volatility = rolling_mad(delta1, window=5)

        def normalize_map_series(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mu = series[:train_end].mean(axis=0).astype(np.float32)
            sigma = series[:train_end].std(axis=0).astype(np.float32) + 1e-6
            norm = ((series - mu[None, :, :]) / sigma[None, :, :]).astype(np.float32)
            return norm, mu.astype(np.float32), sigma.astype(np.float32)

        disp_norm = ((disp - disp_mu[None, :, :]) / disp_sigma[None, :, :]).astype(np.float32)
        delta1_norm, _, _ = normalize_map_series(delta1)
        delta3_norm, _, _ = normalize_map_series(delta3)
        delta5_norm, _, _ = normalize_map_series(delta5)
        vol_norm, _, _ = normalize_map_series(volatility)

        dates = pd.to_datetime(bundle.dates)
        time_feat = np.stack(
            [
                np.sin(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
                np.cos(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
            ],
            axis=1,
        ).astype(np.float32)
        time_expand = np.tile(time_feat[:, :, None, None], (1, 1, height, width)).astype(np.float32)

        coeffs, t_all = fit_linear_seasonal_train_window(bundle, resolution)

        if split_name == "train":
            current_frames = list(range(config.input_len - 1, bundle.split.train_frame_end - max(config.horizons)))
        elif split_name == "val":
            current_frames = list(range(bundle.split.train_frame_end, bundle.split.val_frame_end - max(config.horizons)))
        else:
            current_frames = list(range(bundle.split.val_frame_end, bundle.split.total_frames - max(config.horizons)))

        self.mask = tensors["valid_mask"].astype(np.float32)
        self.regime_map = tensors["regime_map"].astype(np.int16)
        self.samples: list[dict[str, np.ndarray]] = []

        for current in current_frames:
            start = current - config.input_len + 1
            window_disp = disp[start : current + 1]
            slope_map, fit_map, irregular_window = window_linear_fit(window_disp)
            target_indices = [current + h for h in config.horizons]

            x = np.concatenate(
                [
                    disp_norm[start : current + 1, None, :, :],
                    delta1_norm[start : current + 1, None, :, :],
                    delta3_norm[start : current + 1, None, :, :],
                    delta5_norm[start : current + 1, None, :, :],
                    vol_norm[start : current + 1, None, :, :],
                    time_expand[start : current + 1],
                ],
                axis=1,
            ).astype(np.float32)

            target_mm = np.stack([disp[idx] for idx in target_indices], axis=0)[:, None, :, :].astype(np.float32)
            last_disp = disp[current][None, :, :].astype(np.float32)
            persistence_mm = np.repeat(last_disp[None, :, :, :], config.output_len, axis=0).astype(np.float32)
            future_delta_mm = (target_mm[:, 0] - last_disp[0][None, :, :])[:, None, :, :].astype(np.float32)
            ls_mm = predict_linear_seasonal(coeffs, t_all, target_indices, height, width)[:, None, :, :].astype(np.float32)
            future_t = time_feat[target_indices].astype(np.float32)

            current_delta1 = np.abs(delta1[current])
            current_delta3 = np.abs(delta3[current]) / 3.0
            current_delta5 = np.abs(delta5[current]) / 5.0
            step_scale = np.maximum.reduce(
                [
                    volatility[current],
                    current_delta1,
                    current_delta3,
                    current_delta5,
                    np.full_like(volatility[current], config.min_scale_mm, dtype=np.float32),
                ]
            ).astype(np.float32)
            scale_mm = []
            for horizon in config.horizons:
                horizon_factor = max(1.0, config.horizon_scale_alpha * float(horizon))
                scale_mm.append(step_scale * horizon_factor + 0.05)
            scale_mm = np.stack(scale_mm, axis=0)[:, None, :, :].astype(np.float32)

            sample = {
                "x": x,
                "target_mm": target_mm,
                "target_delta_mm": future_delta_mm,
                "persistence_mm": persistence_mm,
                "linear_seasonal_mm": ls_mm,
                "future_time": future_t,
                "mask": self.mask,
                "regime": self.regime_map,
                "last_disp_mm": last_disp,
                "last_disp_norm": disp_norm[current][None, :, :].astype(np.float32),
                "level_map": disp_norm[current][None, :, :].astype(np.float32),
                "inc_hist": np.concatenate(
                    [
                        delta1_norm[start : current + 1, None, :, :],
                        delta3_norm[start : current + 1, None, :, :],
                        delta5_norm[start : current + 1, None, :, :],
                    ],
                    axis=1,
                ).astype(np.float32),
                "trend_map": (slope_map / disp_sigma).astype(np.float32)[None, :, :],
                "irregular_hist": (irregular_window / disp_sigma[None, :, :]).astype(np.float32)[:, None, :, :],
                "time_history": time_feat[start : current + 1].astype(np.float32),
                "scale_mm": scale_mm,
                "current_frame": np.array([current], dtype=np.int64),
                "target_indices": np.array(target_indices, dtype=np.int64),
            }
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {key: torch.from_numpy(value).float() if value.dtype != np.int64 else torch.from_numpy(value).long() for key, value in sample.items()}


def patchify_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    p = patch_size
    x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * c)
    return x


def unpatchify_2d(x: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    b, num_tokens, patch_area = x.shape
    side = image_size // patch_size
    p = patch_size
    x = x.view(b, side, side, p, p).permute(0, 1, 3, 2, 4).reshape(b, image_size, image_size)
    return x[:, None, :, :]


class StaticPatchEncoder(nn.Module):
    def __init__(self, image_size: int, channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * channels
        self.num_patches = (image_size // patch_size) ** 2
        self.net = nn.Sequential(
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = patchify_2d(x, self.patch_size)
        return self.net(patches)


class SequencePatchEncoder(nn.Module):
    def __init__(self, image_size: int, channels: int, patch_size: int, dim: int, depth: int, heads: int, mlp_dim: int, max_len: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * channels
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.time_pos = nn.Parameter(torch.randn(1, max_len, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        patches = patchify_2d(x.reshape(b * t, c, h, w), self.patch_size).reshape(b, t, self.num_patches, -1)
        tokens = self.patch_embed(patches) + self.time_pos[:, :t]
        tokens = tokens.permute(0, 2, 1, 3).reshape(b * self.num_patches, t, -1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.out_norm(tokens.mean(dim=1)).reshape(b, self.num_patches, -1)
        return tokens


class HorizonQueryDecoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, output_len: int, dim: int, depth: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_len = output_len
        self.num_patches = (image_size // patch_size) ** 2
        self.query_tokens = nn.Parameter(torch.randn(1, self.output_len * self.num_patches, dim) * 0.02)
        self.future_proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        total_tokens = self.num_patches + self.output_len * self.num_patches
        self.pos = nn.Parameter(torch.randn(1, total_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])
        patch_area = patch_size * patch_size
        self.out_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_area))

    def forward(self, fused_tokens: torch.Tensor, future_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, num_patches, dim = fused_tokens.shape
        assert num_patches == self.num_patches
        query = self.query_tokens.expand(b, -1, -1)
        future_embed = self.future_proj(future_time)[:, :, None, :].expand(-1, -1, self.num_patches, -1).reshape(b, self.output_len * self.num_patches, dim)
        query = query + future_embed
        tokens = torch.cat([fused_tokens, query], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        total = tokens.shape[1]
        attn_mask = torch.zeros(total, total, device=tokens.device)
        attn_mask[: self.num_patches, self.num_patches :] = float("-inf")
        for block in self.blocks:
            tokens = block(tokens, attn_mask)
        out_tokens = tokens[:, -self.output_len * self.num_patches :]
        patch_out = self.out_head(out_tokens).reshape(b * self.output_len, self.num_patches, -1)
        maps = unpatchify_2d(patch_out, self.image_size, self.patch_size).reshape(b, self.output_len, 1, self.image_size, self.image_size)
        return maps, out_tokens


class V3ADeltaTokenACT(nn.Module):
    def __init__(self, image_size: int, config: ActV3Config, num_channels: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = SequencePatchEncoder(
            image_size=image_size,
            channels=num_channels,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.temporal_depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            max_len=config.input_len,
        )
        self.decoder = HorizonQueryDecoder(
            image_size=image_size,
            patch_size=config.patch_size,
            output_len=config.output_len,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        fused_tokens = self.encoder(batch["x"])
        raw_delta, _ = self.decoder(fused_tokens, batch["future_time"])
        pred_delta = batch["scale_mm"] * torch.tanh(raw_delta)
        pred = batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred_delta,
        }
        if return_aux:
            return pred, aux
        return pred


class V3BKinematicACT(nn.Module):
    def __init__(self, image_size: int, config: ActV3Config) -> None:
        super().__init__()
        self.config = config
        self.num_patches = (image_size // config.patch_size) ** 2
        self.level_encoder = StaticPatchEncoder(image_size, 1, config.patch_size, config.dim)
        self.inc_encoder = SequencePatchEncoder(
            image_size=image_size,
            channels=3,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.temporal_depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            max_len=config.input_len,
        )
        self.trend_encoder = StaticPatchEncoder(image_size, 1, config.patch_size, config.dim)
        self.irregular_encoder = SequencePatchEncoder(
            image_size=image_size,
            channels=1,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.temporal_depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            max_len=config.input_len,
        )
        self.periodic_encoder = nn.Sequential(
            nn.Linear(4, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(config.dim * len(BRANCH_NAMES)),
            nn.Linear(config.dim * len(BRANCH_NAMES), config.dim),
            nn.GELU(),
            nn.Linear(config.dim, len(BRANCH_NAMES)),
        )
        self.decoder = HorizonQueryDecoder(
            image_size=image_size,
            patch_size=config.patch_size,
            output_len=config.output_len,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        level_tokens = self.level_encoder(batch["level_map"])
        inc_tokens = self.inc_encoder(batch["inc_hist"])
        trend_tokens = self.trend_encoder(batch["trend_map"])
        irregular_tokens = self.irregular_encoder(batch["irregular_hist"])
        periodic_feat = torch.cat([batch["time_history"].mean(dim=1), batch["time_history"][:, -1]], dim=-1)
        periodic_token = self.periodic_encoder(periodic_feat)[:, None, :].expand(-1, self.num_patches, -1)

        branch_stack = torch.stack([level_tokens, inc_tokens, trend_tokens, periodic_token, irregular_tokens], dim=2)
        gate_logits = self.fusion_gate(branch_stack.reshape(branch_stack.shape[0], self.num_patches, -1))
        branch_weights = torch.softmax(gate_logits, dim=-1)
        fused_tokens = torch.sum(branch_weights[:, :, :, None] * branch_stack, dim=2)

        raw_delta, _ = self.decoder(fused_tokens, batch["future_time"])
        pred_delta = batch["scale_mm"] * torch.tanh(raw_delta)
        pred = batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred_delta,
            "branch_weights_patch": branch_weights,
        }
        if return_aux:
            return pred, aux
        return pred


class V3CSimVPACT(nn.Module):
    def __init__(self, image_size: int, config: ActV3Config, num_channels: int) -> None:
        super().__init__()
        self.config = config
        width = config.conv_width
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(num_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_mixer = nn.Sequential(
            nn.Conv3d(config.dim, config.dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
            nn.Conv3d(config.dim, config.dim, kernel_size=(3, 3, 3), padding=1),
            nn.GELU(),
        )
        self.future_proj = nn.Sequential(
            nn.Linear(2, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(config.dim * 2, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.dim, 1, kernel_size=1),
        )

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        x = batch["x"]
        b, t, c, h, w = x.shape
        feats = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, self.config.dim, h, w)
        feats = feats.permute(0, 2, 1, 3, 4)
        mixed = self.temporal_mixer(feats).mean(dim=2)

        preds = []
        for h_idx in range(self.config.output_len):
            future_embed = self.future_proj(batch["future_time"][:, h_idx])[:, :, None, None].expand(-1, -1, h, w)
            raw_delta = self.decoder(torch.cat([mixed, future_embed], dim=1))
            pred_delta = batch["scale_mm"][:, h_idx] * torch.tanh(raw_delta)
            pred = batch["last_disp_mm"] + pred_delta
            preds.append(pred)
        pred = torch.stack(preds, dim=1)
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
        }
        if return_aux:
            return pred, aux
        return pred


def build_v3_model(config: ActV3Config, image_size: int, num_channels: int) -> nn.Module:
    if config.variant == "v3a":
        return V3ADeltaTokenACT(image_size=image_size, config=config, num_channels=num_channels)
    if config.variant == "v3b":
        return V3BKinematicACT(image_size=image_size, config=config)
    if config.variant == "v3c":
        return V3CSimVPACT(image_size=image_size, config=config, num_channels=num_channels)
    raise ValueError(f"Unknown variant: {config.variant}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None, :, :]
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, None, :, :]
    err = (pred - target) ** 2
    return (err * weights).sum() / (weights.sum() * pred.shape[1] + 1e-6)


def compute_loss(pred: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float]:
    target = batch["target_mm"]
    mask = batch["mask"]
    loss = masked_smooth_l1(pred, target, mask)
    model_mse = masked_mse(pred, target, mask)
    persistence_mse = masked_mse(batch["persistence_mm"], target, mask)
    skill_mse = 1.0 - float(model_mse.detach().item() / max(persistence_mse.detach().item(), 1e-6))
    return loss, skill_mse


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    config: ActV3Config,
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
            pred = model(batch)
            loss, skill_mse = compute_loss(pred, batch)
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
        skills.append(float(skill_mse))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "skill_mse": float(np.mean(skills)) if skills else float("nan"),
    }


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
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
    branch_weight_list: list[np.ndarray] = []
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
            if "branch_weights_patch" in aux:
                branch_weight_list.append(aux["branch_weights_patch"].cpu().numpy().astype(np.float32))
    infer_seconds = time.time() - start
    arrays: dict[str, np.ndarray] = {
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
    }
    if branch_weight_list:
        arrays["branch_weights_patch"] = np.concatenate(branch_weight_list, axis=0)
    return arrays, infer_seconds


def skill_scores(y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask > 0.5
    model_rmse = float(np.sqrt(np.mean((y_pred[..., valid] - y_true[..., valid]) ** 2)))
    pers_rmse = float(np.sqrt(np.mean((y_pers[..., valid] - y_true[..., valid]) ** 2)))
    model_mse = float(np.mean((y_pred[..., valid] - y_true[..., valid]) ** 2))
    pers_mse = float(np.mean((y_pers[..., valid] - y_true[..., valid]) ** 2))
    return {
        "Skill_MSE": 1.0 - model_mse / max(pers_mse, 1e-6),
        "Skill_RMSE": 1.0 - model_rmse / max(pers_rmse, 1e-6),
        "Persistence RMSE [mm]": pers_rmse,
        "Persistence MSE [mm2]": pers_mse,
        "Model MSE [mm2]": model_mse,
    }


def result_rows(
    model_id: str,
    tile: str,
    seed: int,
    phase: str,
    elapsed_seconds: float,
    infer_seconds: float,
    best_epoch: int,
    param_count: int,
    arrays: dict[str, np.ndarray],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_mask = arrays["valid_mask"].astype(np.float32)
    for idx, horizon in enumerate(horizons):
        y_true = arrays["y_true_mm"][:, idx : idx + 1]
        y_pred = arrays["y_pred_mm"][:, idx : idx + 1]
        y_pers = arrays["persistence_mm"][:, idx : idx + 1]
        y_ls = arrays["linear_seasonal_mm"][:, idx : idx + 1]
        model_metrics = compute_metrics(y_true, y_pred, valid_mask)
        ls_metrics = compute_metrics(y_true, y_ls, valid_mask)
        skill = skill_scores(y_true, y_pred, y_pers, valid_mask)
        rows.append(
            {
                "Model": model_id,
                "Tile": tile,
                "Seed": seed,
                "Phase": phase,
                "Horizon": f"t+{horizon}",
                **model_metrics,
                **skill,
                "LinearSeasonal RMSE [mm]": ls_metrics["RMSE [mm]"],
                "LinearSeasonal MAE [mm]": ls_metrics["MAE [mm]"],
                "Elapsed seconds": elapsed_seconds,
                "Infer seconds": infer_seconds,
                "Best epoch": best_epoch,
                "Param count": param_count,
            }
        )
    return rows


def branch_diagnostic_rows(
    model_id: str,
    tile: str,
    seed: int,
    phase: str,
    arrays: dict[str, np.ndarray],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_mask = arrays["valid_mask"] > 0.5
    regime_map = arrays["regime_map"]
    branch_weights_patch = arrays.get("branch_weights_patch")
    branch_means_by_regime: dict[str, list[float]] = {}
    if branch_weights_patch is not None:
        patch_regimes = []
        side = int(math.sqrt(branch_weights_patch.shape[1]))
        patch_h = regime_map.shape[0] // side
        patch_w = regime_map.shape[1] // side
        for py in range(side):
            for px in range(side):
                y0 = py * patch_h
                x0 = px * patch_w
                y1 = y0 + patch_h
                x1 = x0 + patch_w
                patch = regime_map[y0:y1, x0:x1].reshape(-1)
                if patch.size == 0:
                    patch_regimes.append(0)
                else:
                    patch_regimes.append(int(np.bincount(patch).argmax()))
        patch_regimes = np.array(patch_regimes, dtype=np.int64)
        branch_means_by_regime["all"] = branch_weights_patch.mean(axis=(0, 1)).tolist()
        for regime_idx, regime_name in enumerate(REGIME_NAMES):
            mask_patch = patch_regimes == regime_idx
            if int(mask_patch.sum()) == 0:
                continue
            branch_means_by_regime[regime_name] = branch_weights_patch[:, mask_patch].mean(axis=(0, 1)).tolist()

    for h_idx, horizon in enumerate(horizons):
        pred_delta = arrays["pred_delta_mm"][:, h_idx]
        scale = arrays["scale_mm"][:, h_idx]
        utilization = np.abs(pred_delta) / np.maximum(scale, 1e-6)
        rows.append(
            {
                "Model": model_id,
                "Tile": tile,
                "Seed": seed,
                "Phase": phase,
                "Horizon": f"t+{horizon}",
                "Regime": "all",
                "Branch": "n/a",
                "Mean scale [mm]": float(scale[:, valid_mask].mean()),
                "Mean abs delta [mm]": float(np.abs(pred_delta[:, valid_mask]).mean()),
                "Mean bound utilization": float(utilization[:, valid_mask].mean()),
                "Mean branch weight": float("nan"),
            }
        )
        if branch_means_by_regime:
            for regime_name, branch_values in branch_means_by_regime.items():
                for branch_name, value in zip(BRANCH_NAMES, branch_values):
                    rows.append(
                        {
                            "Model": model_id,
                            "Tile": tile,
                            "Seed": seed,
                            "Phase": phase,
                            "Horizon": f"t+{horizon}",
                            "Regime": regime_name,
                            "Branch": branch_name,
                            "Mean scale [mm]": float(scale[:, valid_mask].mean()),
                            "Mean abs delta [mm]": float(np.abs(pred_delta[:, valid_mask]).mean()),
                            "Mean bound utilization": float(utilization[:, valid_mask].mean()),
                            "Mean branch weight": float(value),
                        }
                    )
    return rows


def train_eval_single_tile(
    bundle: TileBundle,
    config: ActV3Config,
    output_dir: Path,
    phase: str,
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    config_path = output_dir / "config.yaml"
    result_path = output_dir / "result.json"
    ckpt_path = output_dir / "model.pth"
    pred_path = output_dir / "predictions.npz"

    if reuse_if_exists and result_path.exists() and ckpt_path.exists() and pred_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        arrays = np.load(pred_path)
        return {"result": result, "arrays": {name: arrays[name] for name in arrays.files}}

    write_simple_yaml(config_path, asdict(config))
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ActV3Dataset(bundle, resolution=64, split_name="train", config=config)
    val_ds = ActV3Dataset(bundle, resolution=64, split_name="val", config=config)
    test_ds = ActV3Dataset(bundle, resolution=64, split_name="test", config=config)

    num_channels = int(train_ds.samples[0]["x"].shape[1])
    model = build_v3_model(config, image_size=64, num_channels=num_channels).to(device)
    param_count = count_parameters(model)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, scaler, config, device)
        val_stats = run_epoch(model, val_loader, None, None, config, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_skill_mse": train_stats["skill_mse"],
                "val_loss": val_stats["loss"],
                "val_skill_mse": val_stats["skill_mse"],
            }
        )
        current_score = val_stats["skill_mse"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    arrays, infer_seconds = collect_predictions(model, test_loader, device)
    elapsed_seconds = time.time() - start_time

    rows = result_rows(
        model_id=config.model_id,
        tile=bundle.tile_id,
        seed=config.seed,
        phase=phase,
        elapsed_seconds=elapsed_seconds,
        infer_seconds=infer_seconds,
        best_epoch=best_epoch,
        param_count=param_count,
        arrays=arrays,
        horizons=config.horizons,
    )
    diag_rows = branch_diagnostic_rows(config.model_id, bundle.tile_id, config.seed, phase, arrays, config.horizons)
    summary_df = pd.DataFrame(rows)
    result = {
        "model_id": config.model_id,
        "variant": config.variant,
        "tile": bundle.tile_id,
        "seed": config.seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "elapsed_seconds": elapsed_seconds,
        "infer_seconds": infer_seconds,
        "param_count": param_count,
        "history": history,
        "rows": rows,
        "branch_diag_rows": diag_rows,
        "mean_skill_mse": float(summary_df["Skill_MSE"].mean()),
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def summarize_seeded_rows(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    ]
    rows: list[dict[str, Any]] = []
    for (model, tile, horizon), group in df.groupby(["Model", "Tile", "Horizon"]):
        row: dict[str, Any] = {"Model": model, "Tile": tile, "Horizon": horizon}
        for metric in metrics:
            row[f"{metric} mean"] = float(group[metric].mean())
            row[f"{metric} std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Model", "Tile", "Horizon"]).reset_index(drop=True)


def long_horizon_positive_count(df: pd.DataFrame, tile: str, skill_col: str) -> int:
    sub = df[(df["Tile"] == tile) & (df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    return int((sub[skill_col] > 0).sum())


def quick_gate(df: pd.DataFrame, m1_train_seconds: float | None = None) -> tuple[bool, dict[str, Any]]:
    mean_skill = float(df["Skill_MSE"].mean())
    e48 = df[df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm]"] / e48["Persistence RMSE [mm]"]).max()) if not e48.empty else float("inf")
    e32_pos = long_horizon_positive_count(df, "E32N34", "Skill_MSE")
    e39_pos = long_horizon_positive_count(df, "E39N30", "Skill_MSE")
    mean_train = float(df["Elapsed seconds"].mean()) if "Elapsed seconds" in df.columns else float("nan")
    complexity_ok = True
    if m1_train_seconds is not None and not math.isnan(m1_train_seconds):
        complexity_ok = mean_train <= 4.0 * m1_train_seconds
    passed = mean_skill > QUICK_SKILL_THRESHOLD and e48_ratio <= 1.03 and e32_pos >= 2 and e39_pos >= 2 and complexity_ok
    return passed, {
        "mean_skill_mse": mean_skill,
        "quick_threshold": QUICK_SKILL_THRESHOLD,
        "e48_max_rmse_ratio": e48_ratio,
        "e32_positive_long_horizons": e32_pos,
        "e39_positive_long_horizons": e39_pos,
        "mean_train_seconds": mean_train,
        "complexity_ok": complexity_ok,
    }


def three_seed_gate(summary_df: pd.DataFrame, m1_train_seconds: float | None = None) -> tuple[bool, dict[str, Any]]:
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    mean_std = float(summary_df["Skill_MSE std"].mean())
    e48 = summary_df[summary_df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm] mean"] / e48["Persistence RMSE [mm] mean"]).max()) if not e48.empty else float("inf")
    e32 = summary_df[(summary_df["Tile"] == "E32N34") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e39 = summary_df[(summary_df["Tile"] == "E39N30") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e32_stable = int((e32["Skill_MSE mean"] > e32["Skill_MSE std"]).sum())
    e39_stable = int((e39["Skill_MSE mean"] > e39["Skill_MSE std"]).sum())
    mean_train = float(summary_df["Elapsed seconds mean"].mean()) if "Elapsed seconds mean" in summary_df.columns else float("nan")
    complexity_ok = True
    if m1_train_seconds is not None and not math.isnan(m1_train_seconds):
        complexity_ok = mean_train <= 4.0 * m1_train_seconds
    passed = mean_skill > QUICK_SKILL_THRESHOLD and mean_skill > mean_std and e48_ratio <= 1.03 and e32_stable >= 2 and e39_stable >= 2 and complexity_ok
    return passed, {
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "quick_threshold": QUICK_SKILL_THRESHOLD,
        "e48_max_rmse_ratio": e48_ratio,
        "e32_positive_long_horizons_stable": e32_stable,
        "e39_positive_long_horizons_stable": e39_stable,
        "mean_train_seconds": mean_train,
        "complexity_ok": complexity_ok,
    }


def baseline_rows_for_search(bundles: dict[str, TileBundle]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tile in SEARCH_TILES:
        test_ds = ActV3Dataset(bundles[tile], resolution=64, split_name="test", config=ActV3Config(model_id="tmp", variant="v3a"))
        valid_mask = test_ds.mask
        y_true = np.stack([sample["target_mm"][:, 0] for sample in test_ds.samples], axis=0)
        y_pers = np.stack([sample["persistence_mm"][:, 0] for sample in test_ds.samples], axis=0)
        y_ls = np.stack([sample["linear_seasonal_mm"][:, 0] for sample in test_ds.samples], axis=0)
        for idx, horizon in enumerate(SEARCH_HORIZONS):
            y_true_h = y_true[:, idx : idx + 1]
            y_pers_h = y_pers[:, idx : idx + 1]
            y_ls_h = y_ls[:, idx : idx + 1]
            pers_metrics = compute_metrics(y_true_h, y_pers_h, valid_mask)
            ls_metrics = compute_metrics(y_true_h, y_ls_h, valid_mask)
            rows.append(
                {
                    "Model": "persistence",
                    "Tile": tile,
                    "Seed": -1,
                    "Phase": "baseline",
                    "Horizon": f"t+{horizon}",
                    **pers_metrics,
                    "Skill_MSE": 0.0,
                    "Skill_RMSE": 0.0,
                    "Persistence RMSE [mm]": pers_metrics["RMSE [mm]"],
                    "LinearSeasonal RMSE [mm]": ls_metrics["RMSE [mm]"],
                    "LinearSeasonal MAE [mm]": ls_metrics["MAE [mm]"],
                    "Elapsed seconds": 0.0,
                    "Infer seconds": 0.0,
                    "Param count": 0,
                }
            )
            ls_skill_mse = 1.0 - (ls_metrics["RMSE [mm]"] ** 2) / max(pers_metrics["RMSE [mm]"] ** 2, 1e-6)
            ls_skill_rmse = 1.0 - ls_metrics["RMSE [mm]"] / max(pers_metrics["RMSE [mm]"], 1e-6)
            rows.append(
                {
                    "Model": "linear_seasonal",
                    "Tile": tile,
                    "Seed": -1,
                    "Phase": "baseline",
                    "Horizon": f"t+{horizon}",
                    **ls_metrics,
                    "Skill_MSE": ls_skill_mse,
                    "Skill_RMSE": ls_skill_rmse,
                    "Persistence RMSE [mm]": pers_metrics["RMSE [mm]"],
                    "LinearSeasonal RMSE [mm]": ls_metrics["RMSE [mm]"],
                    "LinearSeasonal MAE [mm]": ls_metrics["MAE [mm]"],
                    "Elapsed seconds": 0.0,
                    "Infer seconds": 0.0,
                    "Param count": 0,
                }
            )
    return pd.DataFrame(rows)


def load_existing_baseline_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if M1_COMPLEXITY_PATH.exists():
        df = pd.read_csv(M1_COMPLEXITY_PATH)
        for source, target in [("C1_M0_no_gate", "M0_B7"), ("C3_M1_patch", "M1")]:
            sub = df[df["Model"] == source].copy()
            if sub.empty:
                continue
            sub["Model"] = target
            frames.append(sub)
    if M1_REFERENCE_PATH.exists():
        m1 = pd.read_csv(M1_REFERENCE_PATH)
        if "Elapsed seconds mean" not in m1.columns:
            m1["Elapsed seconds mean"] = float("nan")
        m1["Infer seconds mean"] = float("nan")
        m1["Param count mean"] = float("nan")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def quick_search_model(
    bundles: dict[str, TileBundle],
    config: ActV3Config,
    csv_name: str,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for tile in SEARCH_TILES:
        print(f"[quick] start model={config.model_id} tile={tile} seed={config.seed}", flush=True)
        run = train_eval_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=ACT_V3_ROOT / "runs" / "quick" / config.model_id / tile / f"seed_{config.seed}",
            phase="quick",
            reuse_if_exists=reuse_if_exists,
        )
        print(f"[quick] done model={config.model_id} tile={tile} seed={config.seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
        rows.extend(run["result"]["rows"])
        diag_rows.extend(run["result"]["branch_diag_rows"])
    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    df.to_csv(ACT_V3_ROOT / csv_name, index=False, encoding="utf-8-sig")
    return df, diag_df


def run_three_seed(
    bundles: dict[str, TileBundle],
    base_config: ActV3Config,
    phase: str,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(base_config, seed=seed)
        for tile in SEARCH_TILES:
            print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
            run = train_eval_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=ACT_V3_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}",
                phase=phase,
                reuse_if_exists=reuse_if_exists,
            )
            print(f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
            rows.extend(run["result"]["rows"])
            diag_rows.extend(run["result"]["branch_diag_rows"])
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def complexity_row_from_summary(model: str, summary_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "Model": model,
        "Param count": float(summary_df["Param count mean"].mean()) if "Param count mean" in summary_df.columns else float("nan"),
        "Train seconds": float(summary_df["Elapsed seconds mean"].mean()) if "Elapsed seconds mean" in summary_df.columns else float("nan"),
        "Infer seconds": float(summary_df["Infer seconds mean"].mean()) if "Infer seconds mean" in summary_df.columns else float("nan"),
    }


def build_complexity_table(best_model: str, best_summary: pd.DataFrame, quick_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_df = load_existing_baseline_summary()
    if not base_df.empty:
        for model_name in sorted(base_df["Model"].unique()):
            rows.append(complexity_row_from_summary(model_name, base_df[base_df["Model"] == model_name]))
    for model_name, summary in quick_summaries.items():
        rows.append(complexity_row_from_summary(model_name, summary))
    if best_model not in quick_summaries:
        rows.append(complexity_row_from_summary(best_model, best_summary))
    df = pd.DataFrame(rows).drop_duplicates(subset=["Model"], keep="last").sort_values("Model")
    df.to_csv(ACT_V3_ROOT / "v3_complexity_table.csv", index=False, encoding="utf-8-sig")
    return df


def make_error_maps(model_id: str, phase: str, seed: int = 42) -> None:
    ensure_dir(V3_ERROR_MAP_ROOT)
    target_tiles = ["E32N34", "E48N24"]
    for tile in target_tiles:
        pred_path = ACT_V3_ROOT / "runs" / phase / model_id / tile / f"seed_{seed}" / "predictions.npz"
        if not pred_path.exists():
            continue
        arrays = np.load(pred_path)
        valid_mask = arrays["valid_mask"] > 0.5
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for row_idx, (h_idx, horizon) in enumerate([(0, "t+1"), (3, "t+10")]):
            y_true = arrays["y_true_mm"][:, h_idx]
            y_pred = arrays["y_pred_mm"][:, h_idx]
            y_pers = arrays["persistence_mm"][:, h_idx]
            true_mean = y_true.mean(axis=0)
            pred_mean = y_pred.mean(axis=0)
            abs_err = np.abs(y_pred - y_true).mean(axis=0)
            pers_gain = ((y_pers - y_true) ** 2 - (y_pred - y_true) ** 2).mean(axis=0)
            panels = [
                (true_mean, f"{tile} {horizon} GT"),
                (pred_mean, f"{tile} {horizon} Pred"),
                (abs_err, f"{tile} {horizon} |Err|"),
                (pers_gain, f"{tile} {horizon} Gain"),
            ]
            for col_idx, (img, title) in enumerate(panels):
                ax = axes[row_idx, col_idx]
                show = np.where(valid_mask, img, np.nan)
                im = ax.imshow(show, cmap="viridis")
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(V3_ERROR_MAP_ROOT / f"{model_id}_{tile}_error_maps.png", dpi=220)
        plt.close(fig)


def write_final_report(
    quick_records: list[dict[str, Any]],
    best_model: str | None,
    best_three_seed_summary: pd.DataFrame | None,
    best_three_seed_gate: dict[str, Any] | None,
    final_model: str,
    quick_summaries: dict[str, pd.DataFrame],
    complexity_df: pd.DataFrame,
) -> Path:
    report_path = ACT_V3_ROOT / "final_model_decision.md"
    skill_table_path = ACT_V3_ROOT / "v3_skill_table.csv"
    search_csv_path = ACT_V3_ROOT / "v3_model_search.csv"
    branch_diag_path = ACT_V3_ROOT / "v3_branch_diagnostics.csv"

    lines = [
        "# ACT-Former v3 Go/No-Go 详细决策报告",
        "",
        "数据声明：本轮 v3 结构改进实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile 中已固定的三 tile 搜索子集 `E32N34 / E39N30 / E48N24`。这样做是因为数据获取限制下，当前 TGRS 主线仍以该统一数据版本为唯一合法比较基础。",
        "",
        "## 方案与停止规则",
        "",
        "- 决策树按文档执行：`V3-A quick -> 若不过则 V3-B quick -> 若仍不过则一次 V3-C quick -> 入围者再跑 3 seeds`。",
        f"- M1 参考门槛固定为三 tile 平均 `Skill_MSE = {M1_REFERENCE['mean_skill_mse']:.4f}`，因此 v3 的 quick / final 目标阈值为 `> {QUICK_SKILL_THRESHOLD:.4f}`。",
        "- 额外安全条件：`E48N24` 最差 `RMSE / persistence <= 1.03`，并要求 `E32N34/E39N30` 的 `t+3/t+5/t+10` 多数为正 skill。",
        "",
        "## 实现假设",
        "",
        "- 为保证与既有 M1 split 一致，V3-A/B/C 的 quick 与 3-seed 全部固定使用 `input_len=16`，没有再单独打开 `L=32` 分支。",
        "- `bounded residual scale` 采用最近增量的 `MAD + 多尺度增量幅值上界` 构造，并随 horizon 按线性系数放大；没有额外训练可学习的 scale head。",
        "- V3-B 的 periodic branch 采用 `输入窗口 DOY 均值 + 最后时刻 DOY` 的轻量编码，而不是额外谐波投影。",
        "- V3-C 实现为轻量 SimVP-ACT proxy，用于一次性 sanity check，不等同于完整 U-Former 复现。",
        "",
        "## Quick Search 结果",
        "",
    ]
    for record in quick_records:
        gate = record["gate"]
        lines.extend(
            [
                f"### {record['model_id']}",
                "",
                f"- 配置文件：`{record['config_path']}`",
                f"- Quick 结果：`{record['quick_csv']}`",
                f"- 平均 Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- Quick 门槛 = {gate['quick_threshold']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 正 skill 长 horizon 数 = {gate['e32_positive_long_horizons']}",
                f"- E39N30 正 skill 长 horizon 数 = {gate['e39_positive_long_horizons']}",
                f"- 复杂度是否通过 = `{'是' if gate['complexity_ok'] else '否'}`",
                f"- 是否通过 quick gate = `{'是' if record['quick_passed'] else '否'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 3-Seed 复验",
            "",
        ]
    )
    if best_model is None or best_three_seed_summary is None or best_three_seed_gate is None:
        lines.append("- 没有任何 V3 候选通过 quick gate，因此没有进入 3-seed。")
        lines.append("")
    else:
        lines.extend(
            [
                f"- 入围模型：`{best_model}`",
                f"- 3-seed 汇总：`{(ACT_V3_ROOT / 'v3_best_3seed.csv').as_posix()}`",
                f"- 3-seed 平均 Skill_MSE = {best_three_seed_gate['mean_skill_mse']:.4f}",
                f"- 平均 Skill_MSE std = {best_three_seed_gate['mean_skill_std']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {best_three_seed_gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 稳定正 skill 长 horizon 数 = {best_three_seed_gate['e32_positive_long_horizons_stable']}",
                f"- E39N30 稳定正 skill 长 horizon 数 = {best_three_seed_gate['e39_positive_long_horizons_stable']}",
                f"- 3-seed 是否最终通过 = `{'是' if final_model == best_model else '否'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 复杂度与诊断文件",
            "",
            f"- 主搜索表：`{search_csv_path.as_posix()}`",
            f"- Skill 表：`{skill_table_path.as_posix()}`",
            f"- Complexity 表：`{(ACT_V3_ROOT / 'v3_complexity_table.csv').as_posix()}`",
            f"- Branch diagnostics：`{branch_diag_path.as_posix()}`",
            f"- Error maps 目录：`{V3_ERROR_MAP_ROOT.as_posix()}`",
            "",
            "### Complexity 摘要",
            "",
        ]
    )
    for _, row in complexity_df.iterrows():
        lines.append(
            f"- {row['Model']}: params={int(row['Param count']) if not pd.isna(row['Param count']) else 'nan'}, "
            f"train={row['Train seconds']:.2f}s, infer={row['Infer seconds']:.4f}s."
        )
    lines.extend(
        [
            "",
            "## 最终决策",
            "",
        ]
    )
    if final_model == "M1":
        lines.extend(
            [
                "- `V3-A / V3-B / V3-C` 均未满足文档规定的 Go 门槛，或 3-seed 后稳定性不足。",
                "- 最终模型回退到 `M1`，并将 V3 搜索保留为补充材料中的结构搜索证据。",
                "- 论文主线应保持：在 EGMS short-horizon regional nowcasting 中，最稳健的结构仍是强 persistence fallback 下的保守 residual correction。",
            ]
        )
    else:
        lines.extend(
            [
                f"- 最终通过的 v3 模型是 `{final_model}`。",
                f"- 其 3-seed 平均 Skill_MSE = {best_three_seed_gate['mean_skill_mse']:.4f}，已经超过门槛 {QUICK_SKILL_THRESHOLD:.4f}。",
                "- 后续若继续跑 E1-E8 或正文主实验，应以该 v3 模型替换旧的 M1 主模型。",
            ]
        )
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_next_round_plan(final_model: str, best_model: str | None) -> Path:
    path = ACT_V3_ROOT / "next_round_plan.md"
    if final_model == "M1":
        text = "\n".join(
            [
                "# 下一轮计划",
                "",
                "结论：V3 未通过，本轮主模型回退到 M1。",
                "",
                "后续动作：",
                "- 保留 `act_v3` 全套结果作为 supplementary architecture search evidence。",
                "- 正文与后续 E1-E8 重跑仍使用 M1。",
                "- 方法段将 v3 讨论压缩到“尝试过 latent decomposition / SimVP proxy，但在当前 EGMS 2019-2023 设定下未稳定超过 M1”。",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "# 下一轮计划",
                "",
                f"结论：V3 成功，当前最佳模型是 {best_model}.",
                "",
                "后续动作：",
                f"- 以 `{best_model}` 作为新的主模型继续后续六 tile 主实验或 E1-E8 重跑。",
                "- 更新正文贡献点，把重点改为 latent kinematic decomposition + bounded future increment decoding。",
                "- 在 supplementary 中保留 M1 与其他 V3 负结果作为结构搜索证据。",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")
    return path


def run_act_v3_experiments() -> dict[str, Any]:
    ensure_dir(ACT_V3_ROOT)
    ensure_dir(CONFIG_ROOT)
    ensure_dir(V3_ERROR_MAP_ROOT)
    bundles = preprocess_all_tiles()

    baseline_rows = baseline_rows_for_search(bundles)
    baseline_summary = summarize_seeded_rows(baseline_rows)

    quick_records: list[dict[str, Any]] = []
    quick_summary_tables: dict[str, pd.DataFrame] = {}
    all_quick_rows: list[pd.DataFrame] = []
    all_diag_rows: list[pd.DataFrame] = []
    best_model: str | None = None
    best_config: ActV3Config | None = None
    best_three_seed_summary: pd.DataFrame | None = None
    best_three_seed_gate: dict[str, Any] | None = None
    final_model = "M1"

    print("[v3] quick search start", flush=True)
    for idx, config in enumerate(default_v3_configs()):
        write_simple_yaml(CONFIG_ROOT / f"{config.model_id}.yaml", asdict(config))
        quick_df, diag_df = quick_search_model(bundles, config, csv_name=f"{config.model_id.lower().replace('-', '_')}_quick.csv")
        quick_summary = summarize_seeded_rows(quick_df)
        quick_summary_tables[config.model_id] = quick_summary
        all_quick_rows.append(quick_df)
        if not diag_df.empty:
            all_diag_rows.append(diag_df)
        passed, gate = quick_gate(quick_df, m1_train_seconds=M1_REFERENCE["train_seconds"])
        quick_records.append(
            {
                "model_id": config.model_id,
                "variant": config.variant,
                "config_path": str(CONFIG_ROOT / f"{config.model_id}.yaml"),
                "quick_csv": str(ACT_V3_ROOT / f"{config.model_id.lower().replace('-', '_')}_quick.csv"),
                "quick_passed": passed,
                "gate": gate,
            }
        )
        if passed:
            best_model = config.model_id
            best_config = config
            print(f"[v3] quick gate passed by {config.model_id}", flush=True)
            break
        if idx == 1:
            print("[v3] V3-A and V3-B did not pass; running one V3-C sanity check", flush=True)

    search_summary_rows = []
    for record in quick_records:
        search_summary_rows.append(
            {
                "Model": record["model_id"],
                "Variant": record["variant"],
                "Mean Skill_MSE": record["gate"]["mean_skill_mse"],
                "Quick threshold": record["gate"]["quick_threshold"],
                "E48 max RMSE ratio": record["gate"]["e48_max_rmse_ratio"],
                "E32 positive long horizons": record["gate"]["e32_positive_long_horizons"],
                "E39 positive long horizons": record["gate"]["e39_positive_long_horizons"],
                "Mean train seconds": record["gate"]["mean_train_seconds"],
                "Complexity OK": record["gate"]["complexity_ok"],
                "Quick passed": record["quick_passed"],
            }
        )
    pd.DataFrame(search_summary_rows).to_csv(ACT_V3_ROOT / "v3_model_search.csv", index=False, encoding="utf-8-sig")

    skill_frames = [baseline_summary]
    for df in quick_summary_tables.values():
        skill_frames.append(df)

    if best_config is not None:
        print(f"[v3] three-seed validation start for {best_config.model_id}", flush=True)
        three_seed_df, three_seed_diag = run_three_seed(bundles, best_config, phase="final_3seed", reuse_if_exists=True)
        best_three_seed_summary = summarize_seeded_rows(three_seed_df)
        best_three_seed_summary.to_csv(ACT_V3_ROOT / "v3_best_3seed.csv", index=False, encoding="utf-8-sig")
        all_diag_rows.append(three_seed_diag)
        passed3, gate3 = three_seed_gate(best_three_seed_summary, m1_train_seconds=M1_REFERENCE["train_seconds"])
        best_three_seed_gate = gate3
        if passed3:
            final_model = best_config.model_id
            best_model = best_config.model_id
            print(f"[v3] final pass by {best_config.model_id}", flush=True)
        else:
            print(f"[v3] 3-seed did not pass for {best_config.model_id}; fallback to M1", flush=True)
        skill_frames.append(best_three_seed_summary.assign(Model=best_config.model_id))

    skill_table = pd.concat(skill_frames, ignore_index=True)
    skill_table.to_csv(ACT_V3_ROOT / "v3_skill_table.csv", index=False, encoding="utf-8-sig")

    branch_diag_df = pd.concat(all_diag_rows, ignore_index=True) if all_diag_rows else pd.DataFrame()
    branch_diag_df.to_csv(ACT_V3_ROOT / "v3_branch_diagnostics.csv", index=False, encoding="utf-8-sig")

    complexity_df = build_complexity_table(best_model or "M1", best_three_seed_summary if best_three_seed_summary is not None else pd.DataFrame(), quick_summary_tables)
    if best_model is not None:
        make_error_maps(best_model, phase="final_3seed" if final_model == best_model else "quick")

    report_path = write_final_report(
        quick_records=quick_records,
        best_model=best_model,
        best_three_seed_summary=best_three_seed_summary,
        best_three_seed_gate=best_three_seed_gate,
        final_model=final_model,
        quick_summaries=quick_summary_tables,
        complexity_df=complexity_df,
    )
    next_round_path = write_next_round_plan(final_model=final_model, best_model=best_model)

    summary = {
        "report": str(report_path),
        "next_round_plan": str(next_round_path),
        "final_model": final_model,
        "best_v3_candidate": best_model,
        "quick_threshold": QUICK_SKILL_THRESHOLD,
        "m1_reference_skill": M1_REFERENCE["mean_skill_mse"],
        "three_seed_gate": best_three_seed_gate,
        "search_csv": str(ACT_V3_ROOT / "v3_model_search.csv"),
        "skill_table": str(ACT_V3_ROOT / "v3_skill_table.csv"),
        "complexity_table": str(ACT_V3_ROOT / "v3_complexity_table.csv"),
        "branch_diagnostics": str(ACT_V3_ROOT / "v3_branch_diagnostics.csv"),
    }
    write_json(ACT_V3_ROOT / "run_summary.json", summary)
    return summary
