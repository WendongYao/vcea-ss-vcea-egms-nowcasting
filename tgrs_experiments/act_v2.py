from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset

from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    TileBundle,
    TransformerBlock,
    compute_metrics,
    ensure_dir,
    get_tensors,
    preprocess_all_tiles,
    seed_everything,
    write_json,
)


ACT_V2_ROOT = OUTPUT_ROOT / "act_v2"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "act_v2"
SEARCH_TILES = ["E32N34", "E39N30", "E48N24"]
SEARCH_HORIZONS = [1, 3, 5, 10]
FINAL_SEEDS = [42, 43, 44]
EXPERT_NAMES = ["persistence", "linear_seasonal", "neural"]


@dataclass
class ActV2Config:
    model_id: str
    gate_mode: str
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    dim: int = 64
    depth: int = 1
    heads: int = 4
    mlp_dim: int = 128
    patch_size: int = 8
    temporal_patch: int = 1
    batch_size: int = 8
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    gate_bias: float = -2.0
    gate_use_descriptors: bool = False
    skill_lambda: float = 0.0
    skill_margin: float = 0.0
    select_by_skill: bool = False
    query_enabled: bool = True
    residual_enabled: bool = True
    gate_granularity: str = "patch"
    fixed_gate_value: float | None = None
    num_workers: int = 0

    @property
    def output_len(self) -> int:
        return len(self.horizons)


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
            continue
        if isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
            continue
        if value is None:
            lines.append(f"{key}: null")
            continue
        lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def fit_linear_seasonal_train_window(bundle: TileBundle, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    tensors = get_tensors(bundle, resolution)
    disp = tensors["disp"].astype(np.float32)
    dates = pd.to_datetime(bundle.dates)
    t_all = ((dates - dates[0]).days.to_numpy(dtype=np.float32)) / 365.25
    train_end = bundle.split.train_frame_end
    t_train = t_all[:train_end]
    y_train = disp[:train_end].reshape(train_end, -1)
    design = np.stack(
        [
            np.ones_like(t_train, dtype=np.float32),
            t_train,
            np.sin(2.0 * np.pi * t_train),
            np.cos(2.0 * np.pi * t_train),
        ],
        axis=1,
    )
    inv = np.linalg.inv(design.T @ design + 1e-6 * np.eye(4, dtype=np.float32))
    coeffs = inv @ design.T @ y_train
    return coeffs.astype(np.float32), t_all.astype(np.float32)


def predict_linear_seasonal(coeffs: np.ndarray, t_all: np.ndarray, target_indices: list[int], height: int, width: int) -> np.ndarray:
    coeffs = coeffs.reshape(4, -1)
    target_t = np.array([t_all[idx] for idx in target_indices], dtype=np.float32)
    design_target = np.stack(
        [
            np.ones_like(target_t, dtype=np.float32),
            target_t,
            np.sin(2.0 * np.pi * target_t),
            np.cos(2.0 * np.pi * target_t),
        ],
        axis=1,
    )
    pred_flat = design_target @ coeffs
    return pred_flat.reshape(len(target_indices), height, width).astype(np.float32)


class ActV2Dataset(Dataset):
    def __init__(self, bundle: TileBundle, resolution: int, split_name: str, config: ActV2Config) -> None:
        self.bundle = bundle
        self.config = config
        tensors = get_tensors(bundle, resolution)
        disp = tensors["disp"].astype(np.float32)
        height, width = disp.shape[1], disp.shape[2]

        train_end = bundle.split.train_frame_end
        disp_mu = disp[:train_end].mean(axis=0).astype(np.float32)
        disp_sigma = disp[:train_end].std(axis=0).astype(np.float32) + 1e-6

        increments = np.diff(disp, axis=0, prepend=disp[0:1]).astype(np.float32)
        inc_train = increments[1:train_end] if train_end > 1 else increments[:1]
        inc_mu = inc_train.mean(axis=0).astype(np.float32)
        inc_sigma = inc_train.std(axis=0).astype(np.float32) + 1e-6

        disp_norm = ((disp - disp_mu[None, :, :]) / disp_sigma[None, :, :]).astype(np.float32)
        inc_norm = ((increments - inc_mu[None, :, :]) / inc_sigma[None, :, :]).astype(np.float32)

        desc_stack = np.stack(
            [
                tensors["descriptor_velocity"],
                tensors["descriptor_amp"],
                tensors["trend_r2"],
                tensors["descriptor_acc"],
            ],
            axis=0,
        ).astype(np.float32)
        desc_mu = desc_stack.mean(axis=(1, 2), keepdims=True)
        desc_sigma = desc_stack.std(axis=(1, 2), keepdims=True) + 1e-6
        gate_desc = ((desc_stack - desc_mu) / desc_sigma).astype(np.float32)

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

        self.xs: list[np.ndarray] = []
        self.targets_mm: list[np.ndarray] = []
        self.persistence_mm: list[np.ndarray] = []
        self.linear_seasonal_mm: list[np.ndarray] = []
        self.future_time: list[np.ndarray] = []
        self.mask = tensors["valid_mask"].astype(np.float32)
        self.regime_map = tensors["regime_map"].astype(np.int16)
        self.gate_desc = gate_desc
        self.disp_mu = disp_mu.astype(np.float32)
        self.disp_sigma = disp_sigma.astype(np.float32)
        self.current_frames: list[int] = []
        self.target_indices_list: list[np.ndarray] = []

        for current in current_frames:
            start = current - config.input_len + 1
            input_blocks = [
                disp_norm[start : current + 1, None, :, :],
                inc_norm[start : current + 1, None, :, :],
                time_expand[start : current + 1],
            ]
            x = np.concatenate(input_blocks, axis=1).astype(np.float32)
            target_indices = [current + h for h in config.horizons]
            target_mm = np.stack([disp[idx] for idx in target_indices], axis=0)[:, None, :, :].astype(np.float32)
            persistence_mm = np.repeat(disp[current][None, None, :, :], len(config.horizons), axis=0).astype(np.float32)
            ls_mm = predict_linear_seasonal(coeffs, t_all, target_indices, height, width)[:, None, :, :].astype(np.float32)
            future_t = time_feat[target_indices].astype(np.float32)

            self.xs.append(x)
            self.targets_mm.append(target_mm)
            self.persistence_mm.append(persistence_mm)
            self.linear_seasonal_mm.append(ls_mm)
            self.future_time.append(future_t)
            self.current_frames.append(int(current))
            self.target_indices_list.append(np.array(target_indices, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.xs[index]).float(),
            "target_mm": torch.from_numpy(self.targets_mm[index]).float(),
            "persistence_mm": torch.from_numpy(self.persistence_mm[index]).float(),
            "linear_seasonal_mm": torch.from_numpy(self.linear_seasonal_mm[index]).float(),
            "future_time": torch.from_numpy(self.future_time[index]).float(),
            "mask": torch.from_numpy(self.mask).float(),
            "regime": torch.from_numpy(self.regime_map).long(),
            "gate_desc": torch.from_numpy(self.gate_desc).float(),
            "disp_mu": torch.from_numpy(self.disp_mu).float(),
            "disp_sigma": torch.from_numpy(self.disp_sigma).float(),
            "current_frame": torch.tensor(self.current_frames[index], dtype=torch.long),
            "target_indices": torch.from_numpy(self.target_indices_list[index]).long(),
        }


class TemporalPatchBackbone(nn.Module):
    def __init__(self, image_size: int, num_channels: int, config: ActV2Config) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = config.patch_size
        self.temporal_patch = config.temporal_patch
        self.input_len = config.input_len
        self.output_len = config.output_len
        self.num_channels = num_channels
        self.grouped_len = config.input_len // config.temporal_patch
        if config.input_len % config.temporal_patch != 0:
            raise ValueError(f"input_len={config.input_len} must be divisible by temporal_patch={config.temporal_patch}")

        patch_area = self.patch_size * self.patch_size
        self.patch_area = patch_area
        self.num_patches = (image_size // self.patch_size) ** 2
        patch_dim = patch_area * num_channels * config.temporal_patch
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )
        total_tokens = (self.grouped_len + self.output_len) * self.num_patches
        self.pos = nn.Parameter(torch.randn(1, total_tokens, config.dim) * 0.02)
        self.use_query_tokens = config.query_enabled
        self.query_tokens = (
            nn.Parameter(torch.randn(1, self.output_len * self.num_patches, config.dim) * 0.02)
            if self.use_query_tokens
            else None
        )
        self.blocks = nn.ModuleList([TransformerBlock(config.dim, config.heads, config.mlp_dim) for _ in range(config.depth)])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, patch_area),
        )

    def _group_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_patch == 1:
            return x
        b, t, c, h, w = x.shape
        x = x.view(b, self.grouped_len, self.temporal_patch, c, h, w)
        return x.reshape(b, self.grouped_len, self.temporal_patch * c, h, w)

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, t, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 4, 6, 2).reshape(b, t, -1, p * p * c)
        return self.patch_embed(x)

    def _from_patches(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = self.patch_size
        side = self.image_size // p
        x = x.view(b, self.output_len, side, side, p, p)
        x = x.permute(0, 1, 2, 4, 3, 5).reshape(b, self.output_len, 1, self.image_size, self.image_size)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._group_time(x)
        b, t, _, _, _ = x.shape
        tokens = self._to_patches(x).reshape(b, t * self.num_patches, -1)
        out_tokens = self.output_len * self.num_patches
        if self.use_query_tokens:
            assert self.query_tokens is not None
            query = self.query_tokens.expand(b, -1, -1)
            tokens = torch.cat([tokens, query], dim=1)
        else:
            # No learned query tokens: recycle the last observed patch tokens as decoder inputs.
            tokens = torch.cat([tokens, tokens[:, -self.num_patches :].repeat(1, self.output_len, 1)], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]

        in_tokens = t * self.num_patches
        attn_mask = torch.zeros(in_tokens + out_tokens, in_tokens + out_tokens, device=x.device)
        attn_mask[:in_tokens, in_tokens:] = float("-inf")

        for block in self.blocks:
            tokens = block(tokens, attn_mask)

        query_tokens = tokens[:, -out_tokens:]
        delta_norm = self.out_proj(query_tokens)
        delta_norm = self._from_patches(delta_norm)
        return delta_norm, query_tokens


def pool_to_patch_tokens_2d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    pooled = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(2).transpose(1, 2)


def pool_to_patch_tokens_3d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, h_out, c, h, w = x.shape
    pooled = F.avg_pool2d(x.reshape(b * h_out, c, h, w), kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(2).transpose(1, 2).reshape(b, h_out, -1, c)


def patch_weights_to_maps(weights: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    b, h_out, num_patches, num_experts = weights.shape
    side = image_size // patch_size
    maps = weights.permute(0, 1, 3, 2).reshape(b, h_out, num_experts, side, side)
    maps = maps.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
    return maps


def patch_tokens_to_feature_maps(tokens: torch.Tensor, image_size: int, patch_size: int) -> torch.Tensor:
    b, h_out, num_patches, dim = tokens.shape
    side = image_size // patch_size
    maps = tokens.reshape(b, h_out, side, side, dim).permute(0, 1, 4, 2, 3)
    maps = F.interpolate(
        maps.reshape(b * h_out, dim, side, side),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return maps.reshape(b, h_out, dim, image_size, image_size)


class BaselineGatedResidualQueryACT(nn.Module):
    def __init__(self, image_size: int, num_channels: int, config: ActV2Config) -> None:
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.patch_size = config.patch_size
        self.output_len = config.output_len
        self.gate_granularity = config.gate_granularity
        self.backbone = TemporalPatchBackbone(image_size=image_size, num_channels=num_channels, config=config)

        gate_feature_dim = 2
        if config.gate_mode == "softmax3":
            gate_feature_dim += 1
        if config.gate_use_descriptors:
            gate_feature_dim += 4

        if config.gate_mode == "sigmoid":
            if self.gate_granularity == "pixel":
                self.gate_head = nn.Sequential(
                    nn.Conv2d(config.dim + gate_feature_dim, config.dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(config.dim, 1, kernel_size=1),
                )
                nn.init.constant_(self.gate_head[-1].bias, config.gate_bias)
            else:
                self.gate_head = nn.Sequential(
                    nn.LayerNorm(config.dim + gate_feature_dim),
                    nn.Linear(config.dim + gate_feature_dim, config.dim),
                    nn.GELU(),
                    nn.Linear(config.dim, 1),
                )
                nn.init.constant_(self.gate_head[-1].bias, config.gate_bias)
        elif config.gate_mode == "softmax3":
            self.gate_head = nn.Sequential(
                nn.LayerNorm(config.dim + gate_feature_dim),
                nn.Linear(config.dim + gate_feature_dim, config.dim),
                nn.GELU(),
                nn.Linear(config.dim, 3),
            )
            nn.init.constant_(self.gate_head[-1].bias, 0.0)
            with torch.no_grad():
                self.gate_head[-1].bias[0] = 1.0
                self.gate_head[-1].bias[1] = 0.0
                self.gate_head[-1].bias[2] = -1.0
        else:
            self.gate_head = None

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        x = batch["x"]
        delta_norm, query_tokens = self.backbone(x)
        b = x.shape[0]
        disp_mu = batch["disp_mu"][:, None, None, :, :]
        sigma = batch["disp_sigma"][:, None, None, :, :]
        persistence = batch["persistence_mm"]
        ls_mm = batch["linear_seasonal_mm"]
        if self.config.residual_enabled:
            nn_map = persistence + delta_norm * sigma
        else:
            nn_map = disp_mu + delta_norm * sigma

        if self.config.gate_mode == "none":
            weights = torch.zeros(
                (b, self.output_len, 3, self.image_size, self.image_size),
                device=x.device,
                dtype=x.dtype,
            )
            weights[:, :, 2] = 1.0
            pred = nn_map
        else:
            token_feat = query_tokens.view(b, self.output_len, self.backbone.num_patches, -1)
            future_time_patch = batch["future_time"][:, :, None, :].expand(-1, -1, self.backbone.num_patches, -1)

            if self.config.gate_mode == "sigmoid":
                if self.config.fixed_gate_value is not None:
                    gate_maps = torch.full(
                        (b, self.output_len, self.image_size, self.image_size),
                        float(self.config.fixed_gate_value),
                        device=x.device,
                        dtype=x.dtype,
                    )
                elif self.gate_granularity == "scalar":
                    token_global = token_feat.mean(dim=2)
                    gate_inputs = [token_global, batch["future_time"]]
                    if self.config.gate_use_descriptors:
                        desc_global = batch["gate_desc"].flatten(2).mean(dim=2)
                        gate_inputs.append(desc_global[:, None, :].expand(-1, self.output_len, -1))
                    gate_feat = torch.cat(gate_inputs, dim=-1)
                    gate = torch.sigmoid(self.gate_head(gate_feat))
                    gate_maps = gate[:, :, 0, None, None].expand(-1, -1, self.image_size, self.image_size)
                elif self.gate_granularity == "pixel":
                    token_maps = patch_tokens_to_feature_maps(token_feat, self.image_size, self.patch_size)
                    future_time_maps = batch["future_time"][:, :, :, None, None].expand(-1, -1, -1, self.image_size, self.image_size)
                    pixel_inputs = [token_maps, future_time_maps]
                    if self.config.gate_use_descriptors:
                        desc_maps = batch["gate_desc"][:, None, :, :, :].expand(-1, self.output_len, -1, -1, -1)
                        pixel_inputs.append(desc_maps)
                    gate_feat = torch.cat(pixel_inputs, dim=2)
                    gate = torch.sigmoid(
                        self.gate_head(
                            gate_feat.reshape(b * self.output_len, gate_feat.shape[2], self.image_size, self.image_size)
                        )
                    )
                    gate_maps = gate.reshape(b, self.output_len, self.image_size, self.image_size)
                else:
                    gate_inputs = [token_feat, future_time_patch]
                    if self.config.gate_use_descriptors:
                        desc_patch = pool_to_patch_tokens_2d(batch["gate_desc"], self.patch_size)
                        desc_patch = desc_patch[:, None, :, :].expand(-1, self.output_len, -1, -1)
                        gate_inputs.append(desc_patch)
                    gate_feat = torch.cat(gate_inputs, dim=-1)
                    gate = torch.sigmoid(self.gate_head(gate_feat))
                    gate_maps = patch_weights_to_maps(gate, self.image_size, self.patch_size).squeeze(2)
                pred = persistence + gate_maps[:, :, None, :, :] * (nn_map - persistence)
                weights = torch.zeros((b, self.output_len, 3, self.image_size, self.image_size), device=x.device, dtype=x.dtype)
                weights[:, :, 0] = 1.0 - gate_maps
                weights[:, :, 2] = gate_maps
            else:
                gate_inputs = [token_feat, future_time_patch]
                ls_gap = (ls_mm - persistence) / sigma
                gate_inputs.append(pool_to_patch_tokens_3d(ls_gap, self.patch_size))
                if self.config.gate_use_descriptors:
                    desc_patch = pool_to_patch_tokens_2d(batch["gate_desc"], self.patch_size)
                    desc_patch = desc_patch[:, None, :, :].expand(-1, self.output_len, -1, -1)
                    gate_inputs.append(desc_patch)
                gate_feat = torch.cat(gate_inputs, dim=-1)
                gate = torch.softmax(self.gate_head(gate_feat), dim=-1)
                weight_maps = patch_weights_to_maps(gate, self.image_size, self.patch_size)
                pred = (
                    weight_maps[:, :, 0:1] * persistence
                    + weight_maps[:, :, 1:2] * ls_mm
                    + weight_maps[:, :, 2:3] * nn_map
                )
                weights = weight_maps

        aux = {
            "expert_weights": weights,
            "neural_map_mm": nn_map,
        }
        if return_aux:
            return pred, aux
        return pred


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


def compute_loss(pred: torch.Tensor, batch: dict[str, torch.Tensor], config: ActV2Config) -> tuple[torch.Tensor, float]:
    target = batch["target_mm"]
    mask = batch["mask"]
    base = masked_smooth_l1(pred, target, mask)
    model_mse = masked_mse(pred, target, mask)
    persistence_mse = masked_mse(batch["persistence_mm"], target, mask)
    skill_mse = 1.0 - float(model_mse.detach().item() / max(persistence_mse.detach().item(), 1e-6))
    if config.skill_lambda > 0:
        penalty = F.relu(model_mse - persistence_mse + config.skill_margin)
        base = base + config.skill_lambda * penalty
    return base, skill_mse


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    config: ActV2Config,
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
            loss, skill_mse = compute_loss(pred, batch, config)
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


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    pers_list: list[np.ndarray] = []
    ls_list: list[np.ndarray] = []
    weight_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    regime_list: list[np.ndarray] = []
    current_frame_list: list[np.ndarray] = []
    target_index_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred, aux = model(batch, return_aux=True)
            pred_list.append(pred.cpu().numpy()[:, :, 0].astype(np.float32))
            target_list.append(batch["target_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            pers_list.append(batch["persistence_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            ls_list.append(batch["linear_seasonal_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            weight_list.append(aux["expert_weights"].cpu().numpy().astype(np.float32))
            mask_list.append(batch["mask"].cpu().numpy().astype(np.float32))
            regime_list.append(batch["regime"].cpu().numpy().astype(np.int16))
            current_frame_list.append(batch["current_frame"].cpu().numpy().astype(np.int64))
            target_index_list.append(batch["target_indices"].cpu().numpy().astype(np.int64))
    return {
        "y_pred_mm": np.concatenate(pred_list, axis=0),
        "y_true_mm": np.concatenate(target_list, axis=0),
        "persistence_mm": np.concatenate(pers_list, axis=0),
        "linear_seasonal_mm": np.concatenate(ls_list, axis=0),
        "gate_weights": np.concatenate(weight_list, axis=0),
        "valid_mask": mask_list[0][0] if mask_list and mask_list[0].ndim == 3 else mask_list[0],
        "regime_map": regime_list[0][0] if regime_list and regime_list[0].ndim == 3 else regime_list[0],
        "current_frames": np.concatenate(current_frame_list, axis=0),
        "target_indices": np.concatenate(target_index_list, axis=0),
    }


def skill_scores(y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    model_rmse = float(np.sqrt(np.mean((y_pred[..., valid_mask > 0.5] - y_true[..., valid_mask > 0.5]) ** 2)))
    pers_rmse = float(np.sqrt(np.mean((y_pers[..., valid_mask > 0.5] - y_true[..., valid_mask > 0.5]) ** 2)))
    model_mse = float(np.mean((y_pred[..., valid_mask > 0.5] - y_true[..., valid_mask > 0.5]) ** 2))
    pers_mse = float(np.mean((y_pers[..., valid_mask > 0.5] - y_true[..., valid_mask > 0.5]) ** 2))
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
    best_epoch: int,
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
                "Best epoch": best_epoch,
            }
        )
    return rows


def gate_weight_rows(
    model_id: str,
    tile: str,
    seed: int,
    phase: str,
    arrays: dict[str, np.ndarray],
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weights = arrays["gate_weights"]
    valid_mask = arrays["valid_mask"].astype(np.float32)
    regime_map = arrays["regime_map"].astype(np.int16)
    for h_idx, horizon in enumerate(horizons):
        for regime_name, regime_mask in [("all", valid_mask > 0.5)] + [
            (name, (valid_mask > 0.5) & (regime_map == idx)) for idx, name in enumerate(REGIME_NAMES)
        ]:
            if regime_mask.sum() < 10:
                continue
            for expert_idx, expert in enumerate(EXPERT_NAMES):
                rows.append(
                    {
                        "Model": model_id,
                        "Tile": tile,
                        "Seed": seed,
                        "Phase": phase,
                        "Horizon": f"t+{horizon}",
                        "Regime": regime_name,
                        "Expert": expert,
                        "Mean weight": float(weights[:, h_idx, expert_idx][:, regime_mask].mean()),
                    }
                )
    return rows


def build_model(config: ActV2Config, image_size: int, num_channels: int) -> nn.Module:
    return BaselineGatedResidualQueryACT(image_size=image_size, num_channels=num_channels, config=config)


def train_eval_single_tile(
    bundle: TileBundle,
    config: ActV2Config,
    output_dir: Path,
    phase: str,
    resolution: int = 64,
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
        return {
            "result": result,
            "arrays": {name: arrays[name] for name in arrays.files},
        }

    write_simple_yaml(config_path, asdict(config))
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ActV2Dataset(bundle, resolution=resolution, split_name="train", config=config)
    val_ds = ActV2Dataset(bundle, resolution=resolution, split_name="val", config=config)
    test_ds = ActV2Dataset(bundle, resolution=resolution, split_name="test", config=config)

    num_channels = train_ds.xs[0].shape[1]
    model = build_model(config, image_size=resolution, num_channels=num_channels).to(device)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    best_score = -float("inf") if config.select_by_skill else float("inf")
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
        current_score = val_stats["skill_mse"] if config.select_by_skill else val_stats["loss"]
        improved = current_score > best_score if config.select_by_skill else current_score < best_score
        if improved:
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
    arrays = collect_predictions(model, test_loader, device)
    elapsed_seconds = time.time() - start_time

    rows = result_rows(config.model_id, bundle.tile_id, config.seed, phase, elapsed_seconds, best_epoch, arrays, config.horizons)
    gate_rows = gate_weight_rows(config.model_id, bundle.tile_id, config.seed, phase, arrays, config.horizons)
    summary_df = pd.DataFrame(rows)
    result = {
        "model_id": config.model_id,
        "tile": bundle.tile_id,
        "seed": config.seed,
        "phase": phase,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "select_by_skill": config.select_by_skill,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
        "rows": rows,
        "gate_rows": gate_rows,
        "mean_skill_mse": float(summary_df["Skill_MSE"].mean()),
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def quick_search_model(
    bundles: dict[str, TileBundle],
    config: ActV2Config,
    phase: str,
    csv_name: str,
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    result_payload: dict[str, Any] = {"model_id": config.model_id, "phase": phase, "tiles": {}}
    for tile in SEARCH_TILES:
        run = train_eval_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=ACT_V2_ROOT / "runs" / phase / config.model_id / tile / f"seed_{config.seed}",
            phase=phase,
            reuse_if_exists=reuse_if_exists,
        )
        rows.extend(run["result"]["rows"])
        gate_rows.extend(run["result"]["gate_rows"])
        result_payload["tiles"][tile] = run["result"]
    df = pd.DataFrame(rows)
    df.to_csv(ACT_V2_ROOT / csv_name, index=False, encoding="utf-8-sig")
    gate_df = pd.DataFrame(gate_rows)
    if not gate_df.empty:
        gate_df.to_csv(ACT_V2_ROOT / "gate_weight_summary.csv", index=False, encoding="utf-8-sig")
    result_payload["table"] = str(ACT_V2_ROOT / csv_name)
    result_payload["mean_skill_mse"] = float(df["Skill_MSE"].mean()) if not df.empty else float("nan")
    return {"rows": df, "gate_rows": gate_df, "payload": result_payload}


def long_horizon_positive_count(df: pd.DataFrame, tile: str) -> int:
    sub = df[(df["Tile"] == tile) & (df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    return int((sub["Skill_MSE"] > 0).sum())


def one_seed_go(df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    mean_skill = float(df["Skill_MSE"].mean())
    e48 = df[df["Tile"] == "E48N24"]
    e48_ratio = (e48["RMSE [mm]"] / e48["Persistence RMSE [mm]"]).max() if not e48.empty else float("inf")
    e32_pos = long_horizon_positive_count(df, "E32N34")
    e39_pos = long_horizon_positive_count(df, "E39N30")
    passed = mean_skill > 0 and e48_ratio <= 1.05 and e32_pos >= 2 and e39_pos >= 2
    return passed, {
        "mean_skill_mse": mean_skill,
        "e48_max_rmse_ratio": float(e48_ratio),
        "e32_positive_long_horizons": e32_pos,
        "e39_positive_long_horizons": e39_pos,
    }


def run_three_seed_validation(
    bundles: dict[str, TileBundle],
    base_config: ActV2Config,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(base_config, seed=seed)
        for tile in SEARCH_TILES:
            run = train_eval_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=ACT_V2_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}",
                phase=phase,
                reuse_if_exists=True,
            )
            rows.extend(run["result"]["rows"])
            gate_rows.extend(run["result"]["gate_rows"])
    df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    summary_rows: list[dict[str, Any]] = []
    for (tile, horizon), group in df.groupby(["Tile", "Horizon"]):
        summary_rows.append(
            {
                "Model": base_config.model_id,
                "Tile": tile,
                "Horizon": horizon,
                "RMSE mean [mm]": group["RMSE [mm]"].mean(),
                "RMSE std [mm]": group["RMSE [mm]"].std(ddof=1),
                "MAE mean [mm]": group["MAE [mm]"].mean(),
                "MAE std [mm]": group["MAE [mm]"].std(ddof=1),
                "Skill_MSE mean": group["Skill_MSE"].mean(),
                "Skill_MSE std": group["Skill_MSE"].std(ddof=1),
                "Skill_RMSE mean": group["Skill_RMSE"].mean(),
                "Skill_RMSE std": group["Skill_RMSE"].std(ddof=1),
                "Persistence RMSE [mm]": group["Persistence RMSE [mm]"].mean(),
                "LinearSeasonal RMSE [mm]": group["LinearSeasonal RMSE [mm]"].mean(),
                "Elapsed seconds mean": group["Elapsed seconds"].mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ACT_V2_ROOT / "final_model_selection.csv", index=False, encoding="utf-8-sig")
    if not gate_df.empty:
        gate_df.to_csv(ACT_V2_ROOT / "gate_weight_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df, gate_df


def three_seed_go(summary_df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    skill_std = float(summary_df["Skill_MSE std"].mean())
    e48 = summary_df[summary_df["Tile"] == "E48N24"]
    e48_ratio = (e48["RMSE mean [mm]"] / e48["Persistence RMSE [mm]"]).max() if not e48.empty else float("inf")
    e32 = summary_df[(summary_df["Tile"] == "E32N34") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e39 = summary_df[(summary_df["Tile"] == "E39N30") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e32_pos = int((e32["Skill_MSE mean"] > e32["Skill_MSE std"]).sum())
    e39_pos = int((e39["Skill_MSE mean"] > e39["Skill_MSE std"]).sum())
    passed = mean_skill > 0 and mean_skill > skill_std and e48_ratio <= 1.05 and e32_pos >= 2 and e39_pos >= 2
    return passed, {
        "mean_skill_mse": mean_skill,
        "mean_skill_std": skill_std,
        "e48_max_rmse_ratio": float(e48_ratio),
        "e32_positive_long_horizons_stable": e32_pos,
        "e39_positive_long_horizons_stable": e39_pos,
    }


def baseline_master_table(bundles: dict[str, TileBundle], m0_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tile in SEARCH_TILES:
        test_ds = ActV2Dataset(bundles[tile], resolution=64, split_name="test", config=ActV2Config(model_id="tmp", gate_mode="none"))
        valid_mask = test_ds.mask
        y_true = np.stack([arr[:, 0] for arr in test_ds.targets_mm], axis=0)
        y_pers = np.stack([arr[:, 0] for arr in test_ds.persistence_mm], axis=0)
        y_ls = np.stack([arr[:, 0] for arr in test_ds.linear_seasonal_mm], axis=0)
        for idx, horizon in enumerate(SEARCH_HORIZONS):
            pers_metrics = compute_metrics(y_true[:, idx : idx + 1], y_pers[:, idx : idx + 1], valid_mask)
            ls_metrics = compute_metrics(y_true[:, idx : idx + 1], y_ls[:, idx : idx + 1], valid_mask)
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
                    "Model MSE [mm2]": pers_metrics["RMSE [mm]"] ** 2,
                    "Persistence MSE [mm2]": pers_metrics["RMSE [mm]"] ** 2,
                    "Elapsed seconds": 0.0,
                    "Best epoch": 0,
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
                    "Model MSE [mm2]": ls_metrics["RMSE [mm]"] ** 2,
                    "Persistence MSE [mm2]": pers_metrics["RMSE [mm]"] ** 2,
                    "Elapsed seconds": 0.0,
                    "Best epoch": 0,
                }
            )
    full_df = pd.concat([pd.DataFrame(rows), m0_df], ignore_index=True)
    full_df.to_csv(ACT_V2_ROOT / "master_skill_table.csv", index=False, encoding="utf-8-sig")
    return full_df


def write_go_report(
    master_df: pd.DataFrame,
    quick_results: list[dict[str, Any]],
    selected_config: ActV2Config | None,
    final_summary: pd.DataFrame | None,
    stop_reason: str,
) -> Path:
    path = ACT_V2_ROOT / "go_no_go_act_v2.md"
    existing_outputs = [
        p
        for p in [
            ACT_V2_ROOT / "master_skill_table.csv",
            ACT_V2_ROOT / "model_search_m0.csv",
            ACT_V2_ROOT / "model_search_m1.csv",
            ACT_V2_ROOT / "model_search_m2.csv",
            ACT_V2_ROOT / "model_search_m2_skill.csv",
            ACT_V2_ROOT / "model_search_m3.csv",
            ACT_V2_ROOT / "model_search_m4.csv",
            ACT_V2_ROOT / "gate_weight_summary.csv",
            ACT_V2_ROOT / "final_model_selection.csv",
            ACT_V2_ROOT / "run_summary.json",
        ]
        if p.exists()
    ]
    lines = [
        "# ACT-Former v2 Go/No-Go 报告",
        "",
        "数据声明：本轮 ACT-Former v2 改进实验继续统一使用 **EGMS 2019-2023 Ortho U**。这是因为当前数据获取限制下，仅该时间段相关 tile 已完成本地载入与处理。",
        "",
        "## GO 门槛",
        "",
        "- 三个搜索 tile：`E32N34 / E39N30 / E48N24`。",
        "- 四个 horizon：`t+1/t+3/t+5/t+10`。",
        "- 核心指标：`Skill_MSE = 1 - MSE_model / MSE_persistence`。",
        "- 一轮 quick search 的最低目标：平均 `Skill_MSE > 0`，并且 `E32N34/E39N30` 多数长 horizon 为正 skill，`E48N24` 不比 persistence 差超过约 5%。",
        "- 最终 GO 需要 3 seeds 稳定复现，且正 skill 不能完全被 std 覆盖。",
        "",
        "## Master Skill Table",
        "",
        f"- 文件：`{(ACT_V2_ROOT / 'master_skill_table.csv').as_posix()}`",
        f"- 当前包含行数：{len(master_df)}。",
        "",
        "## Quick Search 结果",
        "",
    ]
    for item in quick_results:
        gate = item["gate"]
        lines.extend(
            [
                f"### {item['model_id']}",
                "",
                f"- 平均 Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 正 skill 长 horizon 数 = {gate['e32_positive_long_horizons']}",
                f"- E39N30 正 skill 长 horizon 数 = {gate['e39_positive_long_horizons']}",
                f"- 是否通过单 seed GO：`{'是' if item['passed'] else '否'}`",
                f"- 结果文件：`{(ACT_V2_ROOT / item['csv_name']).as_posix()}`",
                "",
            ]
        )
    if selected_config is not None and final_summary is not None:
        passed_3seed, gate3 = three_seed_go(final_summary)
        lines.extend(
            [
                "## 3-Seed 复验",
                "",
                f"- 选中模型：`{selected_config.model_id}`",
                f"- 平均 Skill_MSE = {gate3['mean_skill_mse']:.4f}",
                f"- 平均 Skill_MSE std = {gate3['mean_skill_std']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {gate3['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 稳定正 skill 长 horizon 数 = {gate3['e32_positive_long_horizons_stable']}",
                f"- E39N30 稳定正 skill 长 horizon 数 = {gate3['e39_positive_long_horizons_stable']}",
                f"- 最终是否达到 GO：`{'是' if passed_3seed else '否'}`",
                f"- 汇总文件：`{(ACT_V2_ROOT / 'final_model_selection.csv').as_posix()}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 停止原因",
            "",
            f"- {stop_reason}",
            "",
            "## 假设与说明",
            "",
            "- 为避免未来信息泄漏，gate descriptor 仅使用训练窗口可得的 `velocity / amplitude / trend_R² / acceleration`，没有把 full-series `change_score` 直接输入 gate。",
            "- `M3` 的 temporal patch 采用可执行近似实现：把更长历史切成 temporal groups 后再做空间 patch 编码，以满足长历史/局部时间片建模目标。",
            "- 若某个候选模型在单 seed 上已明显不满足 GO 约束，则继续推进到下一个计划模型，而不做无意义的 3-seed 扩展。",
        ]
    )
    output_insert_at = lines.index("## 假设与说明")
    output_lines = ["## 输出文件", ""] + [f"- `{p.as_posix()}`" for p in existing_outputs] + [""]
    lines[output_insert_at:output_insert_at] = output_lines
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def run_act_v2_model_improvement() -> dict[str, Any]:
    ensure_dir(ACT_V2_ROOT)
    ensure_dir(CONFIG_ROOT)
    bundles = preprocess_all_tiles()

    quick_results: list[dict[str, Any]] = []
    selected_config: ActV2Config | None = None
    final_summary: pd.DataFrame | None = None
    stop_reason = ""
    three_seed_failures: list[str] = []

    m0 = ActV2Config(model_id="M0", gate_mode="none")
    write_simple_yaml(CONFIG_ROOT / "M0.yaml", asdict(m0))
    m0_run = quick_search_model(bundles, m0, phase="quick", csv_name="model_search_m0.csv")
    master_df = baseline_master_table(bundles, m0_run["rows"])

    def evaluate_candidate(config: ActV2Config, csv_name: str) -> bool:
        nonlocal master_df, selected_config, final_summary, stop_reason
        run = quick_search_model(bundles, config, phase="quick", csv_name=csv_name)
        passed, gate = one_seed_go(run["rows"])
        record = {"model_id": config.model_id, "passed": passed, "gate": gate, "csv_name": csv_name}
        quick_results.append(record)
        master_df = pd.concat([master_df, run["rows"]], ignore_index=True)
        master_df.to_csv(ACT_V2_ROOT / "master_skill_table.csv", index=False, encoding="utf-8-sig")
        if not passed:
            return False

        summary_df, _ = run_three_seed_validation(bundles, config, phase=f"final_3seed_{config.model_id}")
        final_passed, final_gate = three_seed_go(summary_df)
        record["three_seed_passed"] = final_passed
        record["three_seed_gate"] = final_gate
        if final_passed:
            selected_config = config
            final_summary = summary_df
            stop_reason = f"`{config.model_id}` 在 3 seeds 复验后达到 GO 门槛：平均 Skill_MSE={final_gate['mean_skill_mse']:.4f}。"
            return True

        three_seed_failures.append(
            f"{config.model_id}: 单 seed 通过，但 3 seeds 未稳定过 GO（mean Skill_MSE={final_gate['mean_skill_mse']:.4f}, std={final_gate['mean_skill_std']:.4f}）。"
        )
        return False

    m1 = ActV2Config(model_id="M1", gate_mode="sigmoid")
    write_simple_yaml(CONFIG_ROOT / "M1.yaml", asdict(m1))
    if evaluate_candidate(m1, "model_search_m1.csv"):
        report_path = write_go_report(master_df, quick_results, selected_config, final_summary, stop_reason)
        summary = {
            "report": str(report_path),
            "master_skill_table": str(ACT_V2_ROOT / "master_skill_table.csv"),
            "selected_model": selected_config.model_id if selected_config is not None else None,
            "stop_reason": stop_reason,
        }
        write_json(ACT_V2_ROOT / "run_summary.json", summary)
        return summary

    m2 = ActV2Config(model_id="M2", gate_mode="softmax3")
    write_simple_yaml(CONFIG_ROOT / "M2.yaml", asdict(m2))
    if evaluate_candidate(m2, "model_search_m2.csv"):
        report_path = write_go_report(master_df, quick_results, selected_config, final_summary, stop_reason)
        summary = {
            "report": str(report_path),
            "master_skill_table": str(ACT_V2_ROOT / "master_skill_table.csv"),
            "selected_model": selected_config.model_id if selected_config is not None else None,
            "stop_reason": stop_reason,
        }
        write_json(ACT_V2_ROOT / "run_summary.json", summary)
        return summary

    best_m2_skill_cfg: ActV2Config | None = None
    best_m2_skill_mean = -float("inf")
    skill_rows_all: list[pd.DataFrame] = []
    for lam in [0.1, 0.3, 1.0]:
        cfg = ActV2Config(model_id=f"M2Skill_lam{str(lam).replace('.', 'p')}", gate_mode="softmax3", skill_lambda=lam, select_by_skill=True)
        write_simple_yaml(CONFIG_ROOT / f"{cfg.model_id}.yaml", asdict(cfg))
        run = quick_search_model(bundles, cfg, phase="quick", csv_name=f"{cfg.model_id}.csv")
        passed, gate = one_seed_go(run["rows"])
        quick_results.append({"model_id": cfg.model_id, "passed": passed, "gate": gate, "csv_name": f"{cfg.model_id}.csv"})
        skill_rows_all.append(run["rows"])
        master_df = pd.concat([master_df, run["rows"]], ignore_index=True)
        master_df.to_csv(ACT_V2_ROOT / "master_skill_table.csv", index=False, encoding="utf-8-sig")
        if gate["mean_skill_mse"] > best_m2_skill_mean:
            best_m2_skill_mean = gate["mean_skill_mse"]
            best_m2_skill_cfg = cfg
        if passed:
            summary_df, _ = run_three_seed_validation(bundles, cfg, phase=f"final_3seed_{cfg.model_id}")
            final_passed, final_gate = three_seed_go(summary_df)
            quick_results[-1]["three_seed_passed"] = final_passed
            quick_results[-1]["three_seed_gate"] = final_gate
            if final_passed:
                selected_config = cfg
                final_summary = summary_df
                stop_reason = f"`{cfg.model_id}` 在 3 seeds 复验后达到 GO 门槛：平均 Skill_MSE={final_gate['mean_skill_mse']:.4f}。"
                break
            three_seed_failures.append(
                f"{cfg.model_id}: 单 seed 通过，但 3 seeds 未稳定过 GO（mean Skill_MSE={final_gate['mean_skill_mse']:.4f}, std={final_gate['mean_skill_std']:.4f}）。"
            )
    if skill_rows_all:
        pd.concat(skill_rows_all, ignore_index=True).to_csv(ACT_V2_ROOT / "model_search_m2_skill.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(ACT_V2_ROOT / "model_search_m2_skill.csv", index=False, encoding="utf-8-sig")
    if selected_config is not None:
        report_path = write_go_report(master_df, quick_results, selected_config, final_summary, stop_reason)
        summary = {
            "report": str(report_path),
            "master_skill_table": str(ACT_V2_ROOT / "master_skill_table.csv"),
            "selected_model": selected_config.model_id if selected_config is not None else None,
            "stop_reason": stop_reason,
        }
        write_json(ACT_V2_ROOT / "run_summary.json", summary)
        return summary

    m3_lambda = best_m2_skill_cfg.skill_lambda if best_m2_skill_cfg is not None else 0.3
    m3 = ActV2Config(
        model_id="M3",
        gate_mode="softmax3",
        input_len=32,
        temporal_patch=4,
        batch_size=4,
        epochs=18,
        patience=5,
        skill_lambda=m3_lambda,
        select_by_skill=True,
    )
    write_simple_yaml(CONFIG_ROOT / "M3.yaml", asdict(m3))
    if evaluate_candidate(m3, "model_search_m3.csv"):
        report_path = write_go_report(master_df, quick_results, selected_config, final_summary, stop_reason)
        summary = {
            "report": str(report_path),
            "master_skill_table": str(ACT_V2_ROOT / "master_skill_table.csv"),
            "selected_model": selected_config.model_id if selected_config is not None else None,
            "stop_reason": stop_reason,
        }
        write_json(ACT_V2_ROOT / "run_summary.json", summary)
        return summary

    m4_lambda = best_m2_skill_cfg.skill_lambda if best_m2_skill_cfg is not None else 0.3
    m4 = ActV2Config(
        model_id="M4",
        gate_mode="softmax3",
        gate_use_descriptors=True,
        skill_lambda=m4_lambda,
        select_by_skill=True,
    )
    write_simple_yaml(CONFIG_ROOT / "M4.yaml", asdict(m4))
    evaluate_candidate(m4, "model_search_m4.csv")

    if selected_config is None:
        if three_seed_failures:
            stop_reason = "已完成 M1、M2、M2+Skill、M3、M4。部分模型曾通过单 seed，但 3 seeds 不稳定；最终仍未达到 GO 门槛。"
        else:
            stop_reason = "M1、M2、M2+Skill、M3、M4 已全部执行完，仍未稳定超过 GO 门槛，因此按计划以 no-go 结束。"

    report_path = write_go_report(master_df, quick_results, selected_config, final_summary, stop_reason)
    summary = {
        "report": str(report_path),
        "master_skill_table": str(ACT_V2_ROOT / "master_skill_table.csv"),
        "selected_model": selected_config.model_id if selected_config is not None else None,
        "stop_reason": stop_reason,
    }
    write_json(ACT_V2_ROOT / "run_summary.json", summary)
    return summary
