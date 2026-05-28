from __future__ import annotations

import json
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

from .act_v2 import ACT_V2_ROOT, ActV2Config, ActV2Dataset, FINAL_SEEDS, SEARCH_HORIZONS, SEARCH_TILES, build_model as build_m1_model, write_simple_yaml
from .act_v3 import ActV3Dataset, compute_loss, count_parameters, result_rows, summarize_seeded_rows
from .act_v4 import ACT_V4_ROOT, CONFIG_ROOT, ConvNeXtLiteEncoder, M1_REFERENCE, TemporalMixer2d, select_v4_channels, v4_diag_rows
from .pipeline import OUTPUT_ROOT, TileBundle, ensure_dir, preprocess_all_tiles, seed_everything, write_json


V4C_RESCUE_ROOT = ACT_V4_ROOT / "v4c_efficiency_rescue"
PROFILE_TILE = "E32N34"
TARGET_TRAIN_SECONDS = 5.0 * M1_REFERENCE["train_seconds"]
STRONG_SKILL_THRESHOLD = 0.085
PRACTICAL_SKILL_THRESHOLD = 0.075
PRACTICAL_TRAIN_SECONDS = 30.0


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class V4CRescueConfig:
    model_id: str
    variant: str
    resolution: int = 64
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    dim: int = 48
    scale_factors: list[int] = field(default_factory=lambda: [1, 2, 4])
    scale_dims: list[int] = field(default_factory=lambda: [20, 20, 20])
    scale_depths: list[int] = field(default_factory=lambda: [2, 1, 1])
    batch_size: int = 8
    epochs: int = 16
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    horizon_scale_alpha: float = 0.75
    residual_beta_mm: float = 0.05
    temporal_mode: str = "ssm"  # ssm, depthwise, none
    shared_temporal: bool = False
    no_temporal_mixer: bool = False
    no_bounded_residual: bool = False
    no_volatility_scale: bool = False
    direct_displacement_head: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    channels_last: bool = True
    cache_pyramid: bool = True
    compile_model: bool = False

    @property
    def output_len(self) -> int:
        return len(self.horizons)


def default_rescue_configs() -> list[V4CRescueConfig]:
    return [
        V4CRescueConfig(
            model_id="C0_V4C_optimized",
            variant="c0",
            dim=60,
            scale_factors=[1, 2, 4],
            scale_dims=[20, 20, 20],
            scale_depths=[2, 1, 1],
            temporal_mode="ssm",
            shared_temporal=False,
            batch_size=8,
            epochs=16,
            patience=4,
        ),
        V4CRescueConfig(
            model_id="C1_V4C_two_scale",
            variant="c1",
            dim=48,
            scale_factors=[1, 2],
            scale_dims=[20, 20],
            scale_depths=[2, 1],
            temporal_mode="ssm",
            shared_temporal=False,
            batch_size=8,
            epochs=16,
            patience=4,
        ),
        V4CRescueConfig(
            model_id="C2_V4C_shared_ssm",
            variant="c2",
            dim=48,
            scale_factors=[1, 2, 4],
            scale_dims=[16, 16, 16],
            scale_depths=[1, 1, 1],
            temporal_mode="ssm",
            shared_temporal=True,
            batch_size=8,
            epochs=16,
            patience=4,
        ),
        V4CRescueConfig(
            model_id="C3_V4C_depthwise_temporal",
            variant="c3",
            dim=48,
            scale_factors=[1, 2, 4],
            scale_dims=[16, 16, 16],
            scale_depths=[1, 1, 1],
            temporal_mode="depthwise",
            shared_temporal=False,
            batch_size=8,
            epochs=16,
            patience=4,
        ),
    ]


class PyramidCachedDataset(Dataset):
    def __init__(self, base_ds: ActV3Dataset, scale_factors: list[int], cache_pyramid: bool) -> None:
        self.scale_factors = scale_factors
        self.samples: list[dict[str, torch.Tensor]] = []
        keep = [0, 1, 2, 4, 5, 6]
        for sample in base_ds.samples:
            prepared: dict[str, torch.Tensor] = {}
            for key, value in sample.items():
                tensor = torch.from_numpy(value)
                prepared[key] = tensor.float() if value.dtype != np.int64 else tensor.long()

            x_v4 = prepared["x"][:, keep, :, :].contiguous()
            prepared["x_v4"] = x_v4
            if cache_pyramid:
                for factor in scale_factors:
                    if factor == 1:
                        continue
                    prepared[f"x_v4_{factor}"] = F.avg_pool2d(x_v4, kernel_size=factor, stride=factor)
            self.samples.append(prepared)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


class VectorizedHorizonDecoder2d(nn.Module):
    def __init__(self, dim: int, output_len: int) -> None:
        super().__init__()
        self.output_len = output_len
        self.horizon_embed = nn.Parameter(torch.randn(output_len, dim) * 0.02)
        self.future_proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1),
        )

    def forward(self, state: torch.Tensor, future_time: torch.Tensor) -> torch.Tensor:
        b, c, h, w = state.shape
        future_embed = self.future_proj(future_time.reshape(b * self.output_len, 2)).reshape(b, self.output_len, c)
        horizon_embed = self.horizon_embed[None, :, :, None, None]
        feat = state[:, None, :, :, :] + future_embed[:, :, :, None, None] + horizon_embed
        raw = self.decoder(feat.reshape(b * self.output_len, c, h, w))
        return raw.reshape(b, self.output_len, 1, h, w)


class DepthwiseTemporalMixer2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim)
        self.pw = nn.Conv3d(dim, dim, kernel_size=1)
        self.gate = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = seq.permute(0, 2, 1, 3, 4).contiguous()
        hidden = F.gelu(self.dw(x))
        cand = self.pw(hidden)
        gate = torch.sigmoid(self.gate(x))
        out = x + gate * cand
        final = out[:, :, -1].contiguous()
        norm = final.pow(2).mean(dim=(1, 2, 3)).sqrt()
        return final, norm


def build_temporal_mixer(dim: int, depth: int, mode: str) -> nn.Module:
    if mode == "depthwise":
        return DepthwiseTemporalMixer2d(dim)
    if mode == "none":
        return TemporalMixer2d(dim=dim, depth=depth, mode="none")
    return TemporalMixer2d(dim=dim, depth=depth, mode="ssm")


class RescueV4CModel(nn.Module):
    def __init__(self, config: V4CRescueConfig, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.scale_factors = list(config.scale_factors)
        self.channels_last = config.channels_last
        self.encoders = nn.ModuleList(
            [
                ConvNeXtLiteEncoder(in_channels=in_channels, dim=dim, blocks=2 if idx == 0 else 1)
                for idx, dim in enumerate(config.scale_dims)
            ]
        )
        if config.shared_temporal:
            if len(set(config.scale_dims)) != 1:
                raise ValueError("shared_temporal requires equal scale_dims")
            shared_depth = max(config.scale_depths)
            self.shared_temporal = build_temporal_mixer(config.scale_dims[0], shared_depth, "none" if config.no_temporal_mixer else config.temporal_mode)
            self.mixers = None
        else:
            self.shared_temporal = None
            self.mixers = nn.ModuleList(
                [
                    build_temporal_mixer(dim, depth, "none" if config.no_temporal_mixer else config.temporal_mode)
                    for dim, depth in zip(config.scale_dims, config.scale_depths)
                ]
            )
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(config.scale_dims), config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = VectorizedHorizonDecoder2d(dim=config.dim, output_len=config.output_len)
        self.alpha = nn.Parameter(torch.linspace(1.0, 1.0 + 0.25 * (config.output_len - 1), steps=config.output_len))
        self.beta = nn.Parameter(torch.full((config.output_len,), config.residual_beta_mm))

    def _run_encoder(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w).contiguous()
        if self.channels_last:
            flat = flat.contiguous(memory_format=torch.channels_last)
        encoded = encoder(flat)
        return encoded.reshape(b, t, -1, h, w)

    def _run_mixer(self, idx: int, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_temporal is not None:
            return self.shared_temporal(encoded)
        assert self.mixers is not None
        return self.mixers[idx](encoded)

    def _base_size(self, batch: dict[str, torch.Tensor]) -> tuple[int, int]:
        x = batch["x_v4"]
        return int(x.shape[-2]), int(x.shape[-1])

    def _scale_input(self, batch: dict[str, torch.Tensor], factor: int) -> torch.Tensor:
        if factor == 1:
            return batch["x_v4"]
        key = f"x_v4_{factor}"
        if key in batch:
            return batch[key]
        x = batch["x_v4"]
        b, t, c, h, w = x.shape
        pooled = F.avg_pool2d(x.reshape(b * t, c, h, w), kernel_size=factor, stride=factor)
        return pooled.reshape(b, t, c, pooled.shape[-2], pooled.shape[-1])

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
        profile: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]] | torch.Tensor:
        device = batch["future_time"].device
        base_h, base_w = self._base_size(batch)
        section_times: dict[str, float] = {}
        states: list[torch.Tensor] = []
        norms: list[torch.Tensor] = []

        for idx, factor in enumerate(self.scale_factors):
            section_prefix = f"scale_{factor}"
            x_scale = self._scale_input(batch, factor)
            if profile:
                sync_cuda(device)
                t0 = time.perf_counter()
            encoded = self._run_encoder(self.encoders[idx], x_scale)
            if profile:
                sync_cuda(device)
                section_times[f"{section_prefix}_encoder_ms"] = section_times.get(f"{section_prefix}_encoder_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
            state, norm = self._run_mixer(idx, encoded)
            if profile:
                sync_cuda(device)
                section_times[f"{section_prefix}_temporal_ms"] = section_times.get(f"{section_prefix}_temporal_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            if factor != 1:
                if profile:
                    sync_cuda(device)
                    t0 = time.perf_counter()
                state = F.interpolate(state, size=(base_h, base_w), mode="bilinear", align_corners=False)
                if profile:
                    sync_cuda(device)
                    section_times[f"{section_prefix}_upsample_ms"] = section_times.get(f"{section_prefix}_upsample_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            states.append(state)
            norms.append(norm)

        if profile:
            sync_cuda(device)
            t0 = time.perf_counter()
        fused = self.fuse(torch.cat(states, dim=1))
        if profile:
            sync_cuda(device)
            section_times["fuse_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
        raw = self.decoder(fused, batch["future_time"])
        if profile:
            sync_cuda(device)
            section_times["decoder_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()

        if self.config.no_bounded_residual:
            pred_delta = raw
        else:
            if self.config.no_volatility_scale:
                scale = torch.ones_like(batch["scale_mm"]) * self.beta[None, :, None, None, None]
            else:
                scale = self.alpha[None, :, None, None, None] * batch["scale_mm"] + self.beta[None, :, None, None, None]
            pred_delta = scale * torch.tanh(raw)
        pred = raw if self.config.direct_displacement_head else batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        if profile:
            sync_cuda(device)
            section_times["residual_head_ms"] = (time.perf_counter() - t0) * 1000.0

        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
            "state_norm": torch.stack(norms, dim=0).mean(dim=0),
            "spectral_norm": torch.zeros_like(norms[0]),
        }
        if profile:
            return pred, aux, section_times
        if return_aux:
            return pred, aux
        return pred


class C4HybridModel(nn.Module):
    def __init__(self, config: V4CRescueConfig, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.scale_factors = list(config.scale_factors)
        self.channels_last = config.channels_last
        if len(self.scale_factors) != 3 or len(config.scale_dims) != 3:
            raise ValueError("C4HybridModel expects exactly three scales")
        if config.scale_dims[1] != config.scale_dims[2]:
            raise ValueError("C4HybridModel expects matching low-resolution dimensions")
        self.encoders = nn.ModuleList(
            [
                ConvNeXtLiteEncoder(in_channels=in_channels, dim=dim, blocks=2 if idx == 0 else 1)
                for idx, dim in enumerate(config.scale_dims)
            ]
        )
        self.high_res_mixer = build_temporal_mixer(config.scale_dims[0], config.scale_depths[0], "depthwise")
        low_mode = "none" if config.no_temporal_mixer else "ssm"
        self.low_res_shared_mixer = build_temporal_mixer(config.scale_dims[1], max(config.scale_depths[1:]), low_mode)
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(config.scale_dims), config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = VectorizedHorizonDecoder2d(dim=config.dim, output_len=config.output_len)
        self.alpha = nn.Parameter(torch.linspace(1.0, 1.0 + 0.25 * (config.output_len - 1), steps=config.output_len))
        self.beta = nn.Parameter(torch.full((config.output_len,), config.residual_beta_mm))

    def _run_encoder(self, encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w).contiguous()
        if self.channels_last:
            flat = flat.contiguous(memory_format=torch.channels_last)
        encoded = encoder(flat)
        return encoded.reshape(b, t, -1, h, w)

    def _scale_input(self, batch: dict[str, torch.Tensor], factor: int) -> torch.Tensor:
        if factor == 1:
            return batch["x_v4"]
        key = f"x_v4_{factor}"
        if key in batch:
            return batch[key]
        x = batch["x_v4"]
        b, t, c, h, w = x.shape
        pooled = F.avg_pool2d(x.reshape(b * t, c, h, w), kernel_size=factor, stride=factor)
        return pooled.reshape(b, t, c, pooled.shape[-2], pooled.shape[-1])

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_aux: bool = False,
        profile: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]] | torch.Tensor:
        device = batch["future_time"].device
        base_h, base_w = int(batch["x_v4"].shape[-2]), int(batch["x_v4"].shape[-1])
        section_times: dict[str, float] = {}
        states: list[torch.Tensor] = []
        norms: list[torch.Tensor] = []

        for idx, factor in enumerate(self.scale_factors):
            section_prefix = f"scale_{factor}"
            x_scale = self._scale_input(batch, factor)
            if profile:
                sync_cuda(device)
                t0 = time.perf_counter()
            encoded = self._run_encoder(self.encoders[idx], x_scale)
            if profile:
                sync_cuda(device)
                section_times[f"{section_prefix}_encoder_ms"] = section_times.get(f"{section_prefix}_encoder_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
            if idx == 0:
                state, norm = self.high_res_mixer(encoded)
            else:
                state, norm = self.low_res_shared_mixer(encoded)
            if profile:
                sync_cuda(device)
                section_times[f"{section_prefix}_temporal_ms"] = section_times.get(f"{section_prefix}_temporal_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            if factor != 1:
                if profile:
                    sync_cuda(device)
                    t0 = time.perf_counter()
                state = F.interpolate(state, size=(base_h, base_w), mode="bilinear", align_corners=False)
                if profile:
                    sync_cuda(device)
                    section_times[f"{section_prefix}_upsample_ms"] = section_times.get(f"{section_prefix}_upsample_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            states.append(state)
            norms.append(norm)

        if profile:
            sync_cuda(device)
            t0 = time.perf_counter()
        fused = self.fuse(torch.cat(states, dim=1))
        if profile:
            sync_cuda(device)
            section_times["fuse_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
        raw = self.decoder(fused, batch["future_time"])
        if profile:
            sync_cuda(device)
            section_times["decoder_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()

        if self.config.no_bounded_residual:
            pred_delta = raw
        else:
            if self.config.no_volatility_scale:
                scale = torch.ones_like(batch["scale_mm"]) * self.beta[None, :, None, None, None]
            else:
                scale = self.alpha[None, :, None, None, None] * batch["scale_mm"] + self.beta[None, :, None, None, None]
            pred_delta = scale * torch.tanh(raw)
        pred = raw if self.config.direct_displacement_head else batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        if profile:
            sync_cuda(device)
            section_times["residual_head_ms"] = (time.perf_counter() - t0) * 1000.0

        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
            "state_norm": torch.stack(norms, dim=0).mean(dim=0),
            "spectral_norm": torch.zeros_like(norms[0]),
        }
        if profile:
            return pred, aux, section_times
        if return_aux:
            return pred, aux
        return pred


def build_rescue_model(config: V4CRescueConfig, in_channels: int) -> nn.Module:
    if config.variant == "c4":
        return C4HybridModel(config=config, in_channels=in_channels)
    return RescueV4CModel(config=config, in_channels=in_channels)


def loader_kwargs(config: V4CRescueConfig, shuffle: bool) -> dict[str, Any]:
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


def replace_default_v3(config: V4CRescueConfig):
    from .act_v3 import ActV3Config

    return ActV3Config(
        model_id=config.model_id,
        variant="v3a",
        input_len=config.input_len,
        horizons=config.horizons,
        dim=config.dim,
        depth=1,
        heads=4,
        mlp_dim=128,
        patch_size=8,
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


def batch_to_device_nb(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def run_epoch_rescue(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
    config: V4CRescueConfig,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    skills: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = batch_to_device_nb(batch, device)
        if training:
            assert optimizer is not None
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


def collect_predictions_rescue(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
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
    spectral_norm_list: list[np.ndarray] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device_nb(batch, device)
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
            spectral_norm_list.append(aux["spectral_norm"].cpu().numpy().astype(np.float32))
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
        "spectral_norm": np.concatenate(spectral_norm_list, axis=0),
    }
    return arrays, infer_seconds


def prepare_rescue_datasets(bundle: TileBundle, config: V4CRescueConfig) -> tuple[PyramidCachedDataset, PyramidCachedDataset, PyramidCachedDataset]:
    train_ds = PyramidCachedDataset(ActV3Dataset(bundle, resolution=config.resolution, split_name="train", config=replace_default_v3(config)), config.scale_factors, config.cache_pyramid)
    val_ds = PyramidCachedDataset(ActV3Dataset(bundle, resolution=config.resolution, split_name="val", config=replace_default_v3(config)), config.scale_factors, config.cache_pyramid)
    test_ds = PyramidCachedDataset(ActV3Dataset(bundle, resolution=config.resolution, split_name="test", config=replace_default_v3(config)), config.scale_factors, config.cache_pyramid)
    return train_ds, val_ds, test_ds


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model


def train_eval_rescue_single_tile(
    bundle: TileBundle,
    config: V4CRescueConfig,
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
    train_ds, val_ds, test_ds = prepare_rescue_datasets(bundle, config)

    model = build_rescue_model(config, in_channels=6).to(device)
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

    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch_rescue(model, train_loader, optimizer, scaler, config, device)
        val_stats = run_epoch_rescue(model, val_loader, None, None, config, device)
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
    arrays, infer_seconds = collect_predictions_rescue(model, test_loader, device)
    elapsed_seconds = time.time() - start_time
    rows = result_rows(config.model_id, bundle.tile_id, config.seed, phase, elapsed_seconds, infer_seconds, best_epoch, param_count, arrays, config.horizons)
    diag_rows = v4_diag_rows(config.model_id, bundle.tile_id, config.seed, phase, arrays, config.horizons)
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
        "diag_rows": diag_rows,
        "mean_skill_mse": float(summary_df["Skill_MSE"].mean()),
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def quick_gate_rescue(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    mean_skill = float(df["Skill_MSE"].mean())
    e48 = df[df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm]"] / e48["Persistence RMSE [mm]"]).max()) if not e48.empty else float("inf")
    mean_train = float(df["Elapsed seconds"].mean())
    if mean_skill >= STRONG_SKILL_THRESHOLD and mean_train <= TARGET_TRAIN_SECONDS and e48_ratio <= 1.03:
        status = "strong_candidate"
    elif mean_skill >= PRACTICAL_SKILL_THRESHOLD and mean_train <= PRACTICAL_TRAIN_SECONDS and e48_ratio <= 1.03:
        status = "practical_candidate"
    else:
        status = "fail"
    return status, {
        "mean_skill_mse": mean_skill,
        "e48_max_rmse_ratio": e48_ratio,
        "mean_train_seconds": mean_train,
        "strong_skill_threshold": STRONG_SKILL_THRESHOLD,
        "practical_skill_threshold": PRACTICAL_SKILL_THRESHOLD,
        "train_seconds_target": TARGET_TRAIN_SECONDS,
        "practical_train_seconds_target": PRACTICAL_TRAIN_SECONDS,
        "status": status,
    }


def three_seed_gate_rescue(summary_df: pd.DataFrame, m1_summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    mean_std = float(summary_df["Skill_MSE std"].mean())
    e48 = summary_df[summary_df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm] mean"] / e48["Persistence RMSE [mm] mean"]).max()) if not e48.empty else float("inf")
    mean_train = float(summary_df["Elapsed seconds mean"].mean())
    long_only = summary_df[summary_df["Horizon"].isin(["t+3", "t+5", "t+10"])]
    stable_count = int((long_only["Skill_MSE mean"] > long_only["Skill_MSE std"]).sum())
    m1_long = m1_summary[m1_summary["Horizon"].isin(["t+3", "t+5", "t+10"])]
    merged = summary_df.merge(
        m1_long[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    better_tiles = 0
    for tile in SEARCH_TILES:
        tile_sub = merged[(merged["Tile"] == tile) & (merged["Horizon"].isin(["t+3", "t+5", "t+10"]))]
        if tile_sub.empty:
            continue
        if float(tile_sub["Skill_MSE mean"].mean()) > float(tile_sub["M1 Skill_MSE mean"].mean()):
            better_tiles += 1
    if mean_skill >= PRACTICAL_SKILL_THRESHOLD and mean_train <= TARGET_TRAIN_SECONDS and e48_ratio <= 1.03 and mean_skill > mean_std and better_tiles >= 2 and stable_count >= 6:
        status = "pass"
    else:
        status = "fail"
    return status, {
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "e48_max_rmse_ratio": e48_ratio,
        "mean_train_seconds": mean_train,
        "better_tiles_vs_m1_long_horizon": better_tiles,
        "stable_positive_long_horizons": stable_count,
        "train_seconds_target": TARGET_TRAIN_SECONDS,
        "status": status,
    }


def quick_run_rescue(bundles: dict[str, TileBundle], config: V4CRescueConfig, csv_name: str, reuse_if_exists: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for tile in SEARCH_TILES:
        print(f"[quick] start model={config.model_id} tile={tile} seed={config.seed}", flush=True)
        run = train_eval_rescue_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=V4C_RESCUE_ROOT / "runs" / "quick" / config.model_id / tile / f"seed_{config.seed}",
            phase="quick",
            reuse_if_exists=reuse_if_exists,
        )
        print(f"[quick] done model={config.model_id} tile={tile} seed={config.seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
        rows.extend(run["result"]["rows"])
        diag_rows.extend(run["result"]["diag_rows"])
    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    df.to_csv(V4C_RESCUE_ROOT / csv_name, index=False, encoding="utf-8-sig")
    return df, diag_df


def three_seed_run_rescue(
    bundles: dict[str, TileBundle],
    config: V4CRescueConfig,
    phase: str,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(config, seed=seed)
        for tile in SEARCH_TILES:
            print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
            run = train_eval_rescue_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=V4C_RESCUE_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}",
                phase=phase,
                reuse_if_exists=reuse_if_exists,
            )
            print(f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
            rows.extend(run["result"]["rows"])
            diag_rows.extend(run["result"]["diag_rows"])
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def ablation_variants(config: V4CRescueConfig) -> list[V4CRescueConfig]:
    single_scale_dim = config.scale_dims[0]
    return [
        replace(config, model_id=f"{config.model_id}_single_scale", scale_factors=[1], scale_dims=[single_scale_dim], scale_depths=[config.scale_depths[0]]),
        replace(config, model_id=f"{config.model_id}_no_temporal", no_temporal_mixer=True),
        replace(config, model_id=f"{config.model_id}_no_bound", no_bounded_residual=True),
        replace(config, model_id=f"{config.model_id}_no_vol", no_volatility_scale=True),
        replace(config, model_id=f"{config.model_id}_direct_disp", direct_displacement_head=True),
    ]


class ForwardHookTimer:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.device = device
        self.starts: dict[str, float] = {}
        self.records: dict[str, list[float]] = {}
        self.handles = []
        for name, module in model.named_children():
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name: str):
        def hook(_module: nn.Module, _inputs: Any) -> None:
            sync_cuda(self.device)
            self.starts[name] = time.perf_counter()

        return hook

    def _post(self, name: str):
        def hook(_module: nn.Module, _inputs: Any, _output: Any) -> None:
            sync_cuda(self.device)
            elapsed = (time.perf_counter() - self.starts.pop(name, time.perf_counter())) * 1000.0
            self.records.setdefault(name, []).append(elapsed)

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def profile_training_model(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    model.train(True)
    timer = ForwardHookTimer(model, device)
    stage_totals = {"data_ms": [], "forward_ms": [], "loss_ms": [], "backward_ms": [], "step_ms": [], "transfer_ms": []}
    module_section_totals: dict[str, list[float]] = {}
    autocast_enabled = use_amp and device.type == "cuda"
    iterator = iter(loader)
    prev_end = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = min(6, len(loader))
    measured_steps = max(0, total_steps - 1)
    for step_idx in range(total_steps):
        sync_cuda(device)
        data_start = prev_end
        batch = next(iterator)
        sync_cuda(device)
        data_ms = (time.perf_counter() - data_start) * 1000.0

        transfer_start = time.perf_counter()
        batch = batch_to_device_nb(batch, device)
        sync_cuda(device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            if hasattr(model, "forward"):
                if model_name.startswith("V4-C"):
                    pred, _aux, section_times = model(batch, return_aux=False, profile=True)  # type: ignore[misc]
                    for key, value in section_times.items():
                        module_section_totals.setdefault(key, []).append(value)
                else:
                    pred = model(batch)
            sync_cuda(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            loss_start = time.perf_counter()
            loss, _skill = compute_loss(pred, batch)
            sync_cuda(device)
            loss_ms = (time.perf_counter() - loss_start) * 1000.0

        backward_start = time.perf_counter()
        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sync_cuda(device)
        backward_ms = (time.perf_counter() - backward_start) * 1000.0

        step_start = time.perf_counter()
        if scaler is not None and autocast_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        sync_cuda(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        prev_end = time.perf_counter()

        if step_idx == 0:
            continue
        stage_totals["data_ms"].append(data_ms)
        stage_totals["transfer_ms"].append(transfer_ms)
        stage_totals["forward_ms"].append(forward_ms)
        stage_totals["loss_ms"].append(loss_ms)
        stage_totals["backward_ms"].append(backward_ms)
        stage_totals["step_ms"].append(step_ms)

    timer.close()
    module_rows = []
    for name, values in timer.records.items():
        module_rows.append({"Model": model_name, "Module": name, "Mean ms": float(np.mean(values)), "Std ms": float(np.std(values)), "Source": "hook"})
    for name, values in module_section_totals.items():
        module_rows.append({"Model": model_name, "Module": name, "Mean ms": float(np.mean(values)), "Std ms": float(np.std(values)), "Source": "section"})

    profile_row = {
        "Model": model_name,
        "Tile": PROFILE_TILE,
        "Batch size": loader.batch_size,
        "Num workers": loader.num_workers,
        "Measured steps": measured_steps,
        "Data ms": float(np.mean(stage_totals["data_ms"])) if stage_totals["data_ms"] else float("nan"),
        "Transfer ms": float(np.mean(stage_totals["transfer_ms"])) if stage_totals["transfer_ms"] else float("nan"),
        "Forward ms": float(np.mean(stage_totals["forward_ms"])) if stage_totals["forward_ms"] else float("nan"),
        "Loss ms": float(np.mean(stage_totals["loss_ms"])) if stage_totals["loss_ms"] else float("nan"),
        "Backward ms": float(np.mean(stage_totals["backward_ms"])) if stage_totals["backward_ms"] else float("nan"),
        "Step ms": float(np.mean(stage_totals["step_ms"])) if stage_totals["step_ms"] else float("nan"),
        "Total iter ms": float(
            np.mean(stage_totals["data_ms"])
            + np.mean(stage_totals["transfer_ms"])
            + np.mean(stage_totals["forward_ms"])
            + np.mean(stage_totals["loss_ms"])
            + np.mean(stage_totals["backward_ms"])
            + np.mean(stage_totals["step_ms"])
        )
        if measured_steps > 0
        else float("nan"),
    }
    memory_row = {
        "Model": model_name,
        "Tile": PROFILE_TILE,
        "Batch size": loader.batch_size,
        "Param count": count_parameters(model),
        "Peak allocated MB": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else float("nan"),
        "Peak reserved MB": float(torch.cuda.max_memory_reserved(device) / (1024**2)) if device.type == "cuda" else float("nan"),
        "Parameter MB": float(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)),
    }
    return profile_row, module_rows, memory_row


def run_profiling(bundles: dict[str, TileBundle]) -> dict[str, Path]:
    ensure_dir(V4C_RESCUE_ROOT)
    bundle = bundles[PROFILE_TILE]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 8
    num_workers = 2

    m1_cfg = ActV2Config(model_id="M1_profile", gate_mode="sigmoid", batch_size=batch_size, num_workers=num_workers, use_amp=True)
    m1_train_ds = ActV2Dataset(bundle, resolution=64, split_name="train", config=m1_cfg)
    m1_loader = DataLoader(
        m1_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    m1_model = build_m1_model(m1_cfg, image_size=64, num_channels=m1_train_ds.xs[0].shape[1]).to(device)
    m1_optimizer = optim.AdamW(m1_model.parameters(), lr=m1_cfg.lr, weight_decay=m1_cfg.weight_decay)
    m1_scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    v4_cfg = V4CRescueConfig(
        model_id="V4-C_profile",
        variant="v4c_profile",
        dim=60,
        scale_factors=[1, 2, 4],
        scale_dims=[20, 20, 20],
        scale_depths=[2, 1, 1],
        temporal_mode="ssm",
        shared_temporal=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    v4_train_ds, _, _ = prepare_rescue_datasets(bundle, v4_cfg)
    v4_loader = DataLoader(v4_train_ds, **loader_kwargs(v4_cfg, shuffle=True))
    v4_model = build_rescue_model(v4_cfg, in_channels=6).to(device)
    v4_optimizer = optim.AdamW(v4_model.parameters(), lr=v4_cfg.lr, weight_decay=v4_cfg.weight_decay)
    v4_scaler = amp.GradScaler("cuda", enabled=device.type == "cuda")

    profile_rows = []
    module_rows = []
    memory_rows = []
    for model_name, model, loader, optimizer, scaler, use_amp in [
        ("M1", m1_model, m1_loader, m1_optimizer, m1_scaler, m1_cfg.use_amp),
        ("V4-C", v4_model, v4_loader, v4_optimizer, v4_scaler, v4_cfg.use_amp),
    ]:
        row, modules, memory = profile_training_model(model_name, model, loader, optimizer, scaler, device, use_amp)
        profile_rows.append(row)
        module_rows.extend(modules)
        memory_rows.append(memory)

    profile_path = V4C_RESCUE_ROOT / "v4c_profile.csv"
    module_path = V4C_RESCUE_ROOT / "v4c_module_time_breakdown.csv"
    memory_path = V4C_RESCUE_ROOT / "v4c_memory_breakdown.csv"
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(module_rows).to_csv(module_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(memory_rows).to_csv(memory_path, index=False, encoding="utf-8-sig")
    return {"profile": profile_path, "module": module_path, "memory": memory_path}


def write_decision_report(
    profiling_paths: dict[str, Path],
    quick_records: list[dict[str, Any]],
    final_candidate_name: str | None,
    final_candidate_status: str | None,
    three_seed_stats: dict[str, Any] | None,
    ran_ablation: bool,
) -> Path:
    path = V4C_RESCUE_ROOT / "v4c_efficiency_rescue_decision.md"
    lines = [
        "# V4-C Efficiency Rescue Decision",
        "",
        "数据声明：本轮 rescue 继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        "## Profiling",
        "",
        f"- profile 文件：`{profiling_paths['profile'].as_posix()}`",
        f"- module breakdown：`{profiling_paths['module'].as_posix()}`",
        f"- memory breakdown：`{profiling_paths['memory'].as_posix()}`",
        "",
        "## Quick Search",
        "",
    ]
    for record in quick_records:
        gate = record["gate"]
        lines.extend(
            [
                f"### {record['model_id']}",
                "",
                f"- quick 文件：`{record['csv_path']}`",
                f"- Mean Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- E48N24 worst ratio = {gate['e48_max_rmse_ratio']:.4f}",
                f"- Mean train seconds = {gate['mean_train_seconds']:.2f}",
                f"- Quick status = `{gate['status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Decision",
            "",
            f"- 最优候选：`{final_candidate_name if final_candidate_name is not None else '无'}`",
            f"- 候选 quick 状态：`{final_candidate_status if final_candidate_status is not None else 'fail'}`",
        ]
    )
    if three_seed_stats is None:
        lines.append("- 没有候选进入 3 seeds，因此 rescue 到此停止。")
    else:
        lines.extend(
            [
                f"- 3-seed Mean Skill_MSE = {three_seed_stats['mean_skill_mse']:.4f}",
                f"- 3-seed Mean Skill_MSE std = {three_seed_stats['mean_skill_std']:.4f}",
                f"- 3-seed E48N24 worst ratio = {three_seed_stats['e48_max_rmse_ratio']:.4f}",
                f"- 3-seed Mean train seconds = {three_seed_stats['mean_train_seconds']:.2f}",
                f"- 3-seed status = `{three_seed_stats['status']}`",
                f"- 是否执行 ablation = `{'是' if ran_ablation else '否'}`",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_v4c_efficiency_rescue() -> dict[str, Any]:
    ensure_dir(V4C_RESCUE_ROOT)
    bundles = preprocess_all_tiles()
    profiling_paths = run_profiling(bundles)

    quick_records: list[dict[str, Any]] = []
    quick_summary_tables: dict[str, pd.DataFrame] = {}
    quick_diag_frames: list[pd.DataFrame] = []
    best_record: dict[str, Any] | None = None

    for config in default_rescue_configs():
        write_simple_yaml(CONFIG_ROOT / f"{config.model_id}.yaml", asdict(config))
        quick_df, diag_df = quick_run_rescue(bundles, config, csv_name=f"{config.model_id.lower()}_quick.csv", reuse_if_exists=True)
        quick_summary = summarize_seeded_rows(quick_df)
        quick_summary_tables[config.model_id] = quick_summary
        if not diag_df.empty:
            quick_diag_frames.append(diag_df)
        status, gate = quick_gate_rescue(quick_df)
        record = {
            "model_id": config.model_id,
            "config": config,
            "gate": gate,
            "quick_status": status,
            "csv_path": str((V4C_RESCUE_ROOT / f"{config.model_id.lower()}_quick.csv").resolve()),
        }
        quick_records.append(record)
        if best_record is None:
            best_record = record
        else:
            cur_rank = (2 if status == "strong_candidate" else 1 if status == "practical_candidate" else 0, gate["mean_skill_mse"])
            best_rank = (
                2 if best_record["quick_status"] == "strong_candidate" else 1 if best_record["quick_status"] == "practical_candidate" else 0,
                best_record["gate"]["mean_skill_mse"],
            )
            if cur_rank > best_rank:
                best_record = record

    quick_search_rows = []
    for record in quick_records:
        gate = record["gate"]
        quick_search_rows.append(
            {
                "Model": record["model_id"],
                "Mean Skill_MSE": gate["mean_skill_mse"],
                "E48 max RMSE ratio": gate["e48_max_rmse_ratio"],
                "Mean train seconds": gate["mean_train_seconds"],
                "Strong skill threshold": gate["strong_skill_threshold"],
                "Practical skill threshold": gate["practical_skill_threshold"],
                "Train seconds target": gate["train_seconds_target"],
                "Quick status": gate["status"],
            }
        )
    quick_search_path = V4C_RESCUE_ROOT / "v4c_rescue_quick_search.csv"
    pd.DataFrame(quick_search_rows).to_csv(quick_search_path, index=False, encoding="utf-8-sig")

    three_seed_summary: pd.DataFrame | None = None
    three_seed_stats: dict[str, Any] | None = None
    ran_ablation = False
    ablation_path: Path | None = None

    if best_record is not None and best_record["quick_status"] in {"strong_candidate", "practical_candidate"}:
        config = best_record["config"]
        m1_summary = pd.read_csv(Path(OUTPUT_ROOT) / "act_v2" / "final_model_selection.csv")
        three_df, three_diag = three_seed_run_rescue(bundles, config, phase="best_3seed", reuse_if_exists=True)
        three_seed_summary = summarize_seeded_rows(three_df)
        if not three_diag.empty:
            quick_diag_frames.append(three_diag)
        three_seed_path = V4C_RESCUE_ROOT / "v4c_rescue_best_3seed.csv"
        three_seed_summary.to_csv(three_seed_path, index=False, encoding="utf-8-sig")
        status, stats = three_seed_gate_rescue(three_seed_summary, m1_summary)
        three_seed_stats = stats
        if status == "pass":
            ran_ablation = True
            frames = []
            for ab_cfg in ablation_variants(config):
                ab_df, ab_diag = quick_run_rescue(bundles, replace(ab_cfg, seed=42), csv_name=f"{ab_cfg.model_id.lower()}_ablation.csv", reuse_if_exists=True)
                ab_df["Ablation"] = ab_cfg.model_id
                frames.append(ab_df)
                if not ab_diag.empty:
                    quick_diag_frames.append(ab_diag)
            if frames:
                ablation_path = V4C_RESCUE_ROOT / "v4c_rescue_ablation.csv"
                pd.concat(frames, ignore_index=True).to_csv(ablation_path, index=False, encoding="utf-8-sig")

    skill_frames = [pd.read_csv(Path(OUTPUT_ROOT) / "act_v4" / "forced_v4bc" / "v4c_accuracy_override_3seed.csv")] if (Path(OUTPUT_ROOT) / "act_v4" / "forced_v4bc" / "v4c_accuracy_override_3seed.csv").exists() else []
    skill_frames.extend(quick_summary_tables.values())
    if three_seed_summary is not None:
        skill_frames.append(three_seed_summary)
    if skill_frames:
        pd.concat(skill_frames, ignore_index=True).to_csv(V4C_RESCUE_ROOT / "v4c_rescue_skill_table.csv", index=False, encoding="utf-8-sig")

    if quick_diag_frames:
        pd.concat(quick_diag_frames, ignore_index=True).to_csv(V4C_RESCUE_ROOT / "v4c_rescue_diag.csv", index=False, encoding="utf-8-sig")

    report_path = write_decision_report(
        profiling_paths=profiling_paths,
        quick_records=quick_records,
        final_candidate_name=best_record["model_id"] if best_record is not None else None,
        final_candidate_status=best_record["quick_status"] if best_record is not None else None,
        three_seed_stats=three_seed_stats,
        ran_ablation=ran_ablation,
    )
    summary = {
        "report": str(report_path),
        "profile": str(profiling_paths["profile"]),
        "module_breakdown": str(profiling_paths["module"]),
        "memory_breakdown": str(profiling_paths["memory"]),
        "quick_search": str(quick_search_path),
        "best_candidate": best_record["model_id"] if best_record is not None else None,
        "best_candidate_status": best_record["quick_status"] if best_record is not None else None,
        "three_seed_stats": three_seed_stats,
        "ablation": str(ablation_path) if ablation_path is not None else None,
    }
    write_json(V4C_RESCUE_ROOT / "run_summary.json", summary)
    return summary
