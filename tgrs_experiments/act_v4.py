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
from torch.utils.data import DataLoader

from .act_v3 import (
    ActV3Dataset,
    M1_REFERENCE,
    baseline_rows_for_search,
    batch_to_device,
    complexity_row_from_summary,
    compute_loss,
    count_parameters,
    load_existing_baseline_summary,
    result_rows,
    run_epoch,
    summarize_seeded_rows,
)
from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, SEARCH_TILES, write_simple_yaml
from .pipeline import OUTPUT_ROOT, TILE_ORDER, TileBundle, ensure_dir, preprocess_all_tiles, seed_everything, write_json


ACT_V4_ROOT = OUTPUT_ROOT / "act_v4"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "act_v4"
V4_ERROR_MAP_ROOT = ACT_V4_ROOT / "v4_error_maps"

STRONG_GO_THRESHOLD = M1_REFERENCE["mean_skill_mse"] + 0.03
WEAK_GO_THRESHOLD = M1_REFERENCE["mean_skill_mse"]
M1_TRAIN_SECONDS = M1_REFERENCE["train_seconds"]


@dataclass
class ActV4Config:
    model_id: str
    variant: str
    input_len: int = 16
    horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    dim: int = 64
    ssm_depth: int = 2
    conv_blocks: int = 2
    fno_modes: int = 8
    batch_size: int = 8
    epochs: int = 16
    lr: float = 3e-4
    weight_decay: float = 5e-5
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    horizon_scale_alpha: float = 0.75
    residual_beta_mm: float = 0.05
    no_ssm: bool = False
    temporal_mode: str = "ssm"  # ssm, gru, none
    no_bounded_residual: bool = False
    no_volatility_scale: bool = False
    no_fno: bool = False
    direct_displacement_head: bool = False
    num_workers: int = 0

    @property
    def output_len(self) -> int:
        return len(self.horizons)


def default_v4_configs() -> list[ActV4Config]:
    return [
        ActV4Config(model_id="V4-A", variant="v4a", input_len=16, dim=64, ssm_depth=2, conv_blocks=2, batch_size=8, epochs=16, patience=4),
        ActV4Config(model_id="V4-B", variant="v4b", input_len=16, dim=64, ssm_depth=2, conv_blocks=2, fno_modes=8, batch_size=8, epochs=18, patience=5),
        ActV4Config(model_id="V4-C", variant="v4c", input_len=16, dim=60, ssm_depth=1, conv_blocks=2, batch_size=6, epochs=16, patience=4),
    ]


def select_v4_channels(x: torch.Tensor) -> torch.Tensor:
    # Keep D, delta1, delta3, local_volatility, sin(DOY), cos(DOY); drop delta5.
    keep = [0, 1, 2, 4, 5, 6]
    return x[:, :, keep, :, :]


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + x


class ConvNeXtLiteEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvNeXtLiteBlock(dim) for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class SelectiveSSMCell2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.to_gates = nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, z: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(z)
        a, b, c = torch.chunk(self.to_gates(z), 3, dim=1)
        a = torch.sigmoid(a)
        b = torch.tanh(b)
        c = torch.sigmoid(c)
        state = a * state + b * z
        out = self.to_out(c * state) + z
        return out, state


class GRULiteCell2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.to_update = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.to_reset = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.to_candidate = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, z: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(z)
        cat = torch.cat([z, state], dim=1)
        update = torch.sigmoid(self.to_update(cat))
        reset = torch.sigmoid(self.to_reset(cat))
        cand = torch.tanh(self.to_candidate(torch.cat([z, reset * state], dim=1)))
        state = (1.0 - update) * state + update * cand
        return state + z, state


class TemporalMixer2d(nn.Module):
    def __init__(self, dim: int, depth: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        if mode == "ssm":
            self.cells = nn.ModuleList([SelectiveSSMCell2d(dim) for _ in range(depth)])
        elif mode == "gru":
            self.cells = nn.ModuleList([GRULiteCell2d(dim) for _ in range(depth)])
        elif mode == "none":
            self.cells = nn.ModuleList()
        else:
            raise ValueError(f"Unknown temporal mode: {mode}")

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "none":
            final = seq[:, -1]
            norm = final.pow(2).mean(dim=(1, 2, 3)).sqrt()
            return final, norm

        cur = seq
        state_norms: list[torch.Tensor] = []
        for cell in self.cells:
            state = torch.zeros_like(cur[:, 0])
            outputs = []
            for t in range(cur.shape[1]):
                out, state = cell(cur[:, t], state)
                outputs.append(out)
                state_norms.append(state.pow(2).mean(dim=(1, 2, 3)).sqrt())
            cur = torch.stack(outputs, dim=1)
        final = cur[:, -1]
        state_norm = torch.stack(state_norms, dim=1).mean(dim=1)
        return final, state_norm


class SpectralPropagator2d(nn.Module):
    def __init__(self, dim: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.weight_top = nn.Parameter(torch.randn(dim, modes, modes, 2) * 0.02)
        self.weight_bottom = nn.Parameter(torch.randn(dim, modes, modes, 2) * 0.02)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def _to_complex(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(weight.contiguous())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        m_h = min(self.modes, h)
        m_w = min(self.modes, w // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)
        wt_top = self._to_complex(self.weight_top[:, :m_h, :m_w])
        wt_bottom = self._to_complex(self.weight_bottom[:, :m_h, :m_w])
        out_ft[:, :, :m_h, :m_w] = x_ft[:, :, :m_h, :m_w] * wt_top[None]
        out_ft[:, :, -m_h:, :m_w] = x_ft[:, :, -m_h:, :m_w] * wt_bottom[None]
        propagated = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        propagated = self.out_proj(propagated)
        residual = x + propagated
        spectral_norm = propagated.pow(2).mean(dim=(1, 2, 3)).sqrt()
        return residual, spectral_norm


class HorizonDecoder2d(nn.Module):
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
        preds = []
        for h_idx in range(self.output_len):
            horizon_token = self.horizon_embed[h_idx][None, :, None, None]
            future_token = self.future_proj(future_time[:, h_idx])[:, :, None, None]
            feat = state + horizon_token + future_token
            preds.append(self.decoder(feat))
        return torch.stack(preds, dim=1)


class SingleScaleConvSSMBackbone(nn.Module):
    def __init__(self, in_channels: int, dim: int, conv_blocks: int, ssm_depth: int, temporal_mode: str, fno_modes: int | None = None) -> None:
        super().__init__()
        self.encoder = ConvNeXtLiteEncoder(in_channels=in_channels, dim=dim, blocks=conv_blocks)
        self.temporal = TemporalMixer2d(dim=dim, depth=ssm_depth, mode=temporal_mode)
        self.fno = SpectralPropagator2d(dim=dim, modes=fno_modes) if fno_modes is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x.shape
        encoded = self.encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1, h, w)
        state, state_norm = self.temporal(encoded)
        if self.fno is None:
            spectral_norm = torch.zeros_like(state_norm)
            return state, state_norm, spectral_norm
        state, spectral_norm = self.fno(state)
        return state, state_norm, spectral_norm


class V4ABaseModel(nn.Module):
    def __init__(self, config: ActV4Config, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.backbone = SingleScaleConvSSMBackbone(
            in_channels=in_channels,
            dim=config.dim,
            conv_blocks=config.conv_blocks,
            ssm_depth=config.ssm_depth,
            temporal_mode="none" if config.no_ssm else config.temporal_mode,
            fno_modes=None,
        )
        self.decoder = HorizonDecoder2d(dim=config.dim, output_len=config.output_len)
        self.alpha = nn.Parameter(torch.linspace(1.0, 1.0 + 0.25 * (config.output_len - 1), steps=config.output_len))
        self.beta = nn.Parameter(torch.full((config.output_len,), config.residual_beta_mm))

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        x = select_v4_channels(batch["x"])
        state, state_norm, spectral_norm = self.backbone(x)
        raw = self.decoder(state, batch["future_time"])
        if self.config.no_bounded_residual:
            pred_delta = raw
        else:
            scale = batch["scale_mm"].clone()
            if self.config.no_volatility_scale:
                scale = torch.ones_like(scale) * self.beta[None, :, None, None, None]
            else:
                scale = self.alpha[None, :, None, None, None] * batch["scale_mm"] + self.beta[None, :, None, None, None]
            pred_delta = scale * torch.tanh(raw)
        if self.config.direct_displacement_head:
            pred = raw
        else:
            pred = batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
            "state_norm": state_norm,
            "spectral_norm": spectral_norm,
        }
        if return_aux:
            return pred, aux
        return pred


class V4BModel(nn.Module):
    def __init__(self, config: ActV4Config, in_channels: int) -> None:
        super().__init__()
        self.config = config
        self.backbone = SingleScaleConvSSMBackbone(
            in_channels=in_channels,
            dim=config.dim,
            conv_blocks=config.conv_blocks,
            ssm_depth=config.ssm_depth,
            temporal_mode="none" if config.no_ssm else config.temporal_mode,
            fno_modes=None if config.no_fno else config.fno_modes,
        )
        self.decoder = HorizonDecoder2d(dim=config.dim, output_len=config.output_len)
        self.alpha = nn.Parameter(torch.linspace(1.0, 1.0 + 0.25 * (config.output_len - 1), steps=config.output_len))
        self.beta = nn.Parameter(torch.full((config.output_len,), config.residual_beta_mm))

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        x = select_v4_channels(batch["x"])
        state, state_norm, spectral_norm = self.backbone(x)
        raw = self.decoder(state, batch["future_time"])
        if self.config.no_bounded_residual:
            pred_delta = raw
        else:
            scale = batch["scale_mm"].clone()
            if self.config.no_volatility_scale:
                scale = torch.ones_like(scale) * self.beta[None, :, None, None, None]
            else:
                scale = self.alpha[None, :, None, None, None] * batch["scale_mm"] + self.beta[None, :, None, None, None]
            pred_delta = scale * torch.tanh(raw)
        if self.config.direct_displacement_head:
            pred = raw
        else:
            pred = batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
            "state_norm": state_norm,
            "spectral_norm": spectral_norm,
        }
        if return_aux:
            return pred, aux
        return pred


class V4CModel(nn.Module):
    def __init__(self, config: ActV4Config, in_channels: int) -> None:
        super().__init__()
        self.config = config
        scale_dims = [20, 20, 20]
        self.backbone64 = SingleScaleConvSSMBackbone(in_channels=in_channels, dim=scale_dims[0], conv_blocks=config.conv_blocks, ssm_depth=config.ssm_depth, temporal_mode="none" if config.no_ssm else config.temporal_mode, fno_modes=None)
        self.backbone32 = SingleScaleConvSSMBackbone(in_channels=in_channels, dim=scale_dims[1], conv_blocks=1, ssm_depth=1, temporal_mode="none" if config.no_ssm else config.temporal_mode, fno_modes=None)
        self.backbone16 = SingleScaleConvSSMBackbone(in_channels=in_channels, dim=scale_dims[2], conv_blocks=1, ssm_depth=1, temporal_mode="none" if config.no_ssm else config.temporal_mode, fno_modes=None)
        self.fuse = nn.Sequential(
            nn.Conv2d(sum(scale_dims), config.dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(config.dim, config.dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.decoder = HorizonDecoder2d(dim=config.dim, output_len=config.output_len)
        self.alpha = nn.Parameter(torch.linspace(1.0, 1.0 + 0.25 * (config.output_len - 1), steps=config.output_len))
        self.beta = nn.Parameter(torch.full((config.output_len,), config.residual_beta_mm))

    def _downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = F.avg_pool2d(x.reshape(b * t, c, h, w), kernel_size=factor, stride=factor)
        hh, ww = x.shape[-2:]
        return x.reshape(b, t, c, hh, ww)

    def forward(self, batch: dict[str, torch.Tensor], return_aux: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        x = select_v4_channels(batch["x"])
        s64, n64, _ = self.backbone64(x)
        s32, n32, _ = self.backbone32(self._downsample(x, 2))
        s16, n16, _ = self.backbone16(self._downsample(x, 4))
        s32 = F.interpolate(s32, size=s64.shape[-2:], mode="bilinear", align_corners=False)
        s16 = F.interpolate(s16, size=s64.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([s64, s32, s16], dim=1))
        raw = self.decoder(fused, batch["future_time"])
        if self.config.no_bounded_residual:
            pred_delta = raw
        else:
            scale = batch["scale_mm"].clone()
            if self.config.no_volatility_scale:
                scale = torch.ones_like(scale) * self.beta[None, :, None, None, None]
            else:
                scale = self.alpha[None, :, None, None, None] * batch["scale_mm"] + self.beta[None, :, None, None, None]
            pred_delta = scale * torch.tanh(raw)
        if self.config.direct_displacement_head:
            pred = raw
        else:
            pred = batch["last_disp_mm"][:, None, :, :, :] + pred_delta
        aux = {
            "scale_mm": batch["scale_mm"],
            "pred_delta_mm": pred - batch["last_disp_mm"][:, None, :, :, :],
            "state_norm": (n64 + n32 + n16) / 3.0,
            "spectral_norm": torch.zeros_like(n64),
        }
        if return_aux:
            return pred, aux
        return pred


def build_v4_model(config: ActV4Config, in_channels: int) -> nn.Module:
    if config.variant == "v4a":
        return V4ABaseModel(config=config, in_channels=in_channels)
    if config.variant == "v4b":
        return V4BModel(config=config, in_channels=in_channels)
    if config.variant == "v4c":
        return V4CModel(config=config, in_channels=in_channels)
    raise ValueError(f"Unknown variant: {config.variant}")


def collect_predictions_v4(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
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


def v4_diag_rows(model_id: str, tile: str, seed: int, phase: str, arrays: dict[str, np.ndarray], horizons: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_mask = arrays["valid_mask"] > 0.5
    state_norm = arrays["state_norm"]
    spectral_norm = arrays["spectral_norm"]
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
                "Mean scale [mm]": float(scale[:, valid_mask].mean()),
                "Mean abs delta [mm]": float(np.abs(pred_delta[:, valid_mask]).mean()),
                "Mean bound utilization": float(utilization[:, valid_mask].mean()),
                "Mean state norm": float(np.mean(state_norm)),
                "Mean spectral norm": float(np.mean(spectral_norm)),
            }
        )
    return rows


def train_eval_single_tile(
    bundle: TileBundle,
    config: ActV4Config,
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

    train_ds = ActV3Dataset(bundle, resolution=64, split_name="train", config=replace_default_v3(config))
    val_ds = ActV3Dataset(bundle, resolution=64, split_name="val", config=replace_default_v3(config))
    test_ds = ActV3Dataset(bundle, resolution=64, split_name="test", config=replace_default_v3(config))

    num_channels = 6
    model = build_v4_model(config, in_channels=num_channels).to(device)
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
        train_stats = run_epoch(model, train_loader, optimizer, scaler, replace_default_v3(config), device)
        val_stats = run_epoch(model, val_loader, None, None, replace_default_v3(config), device)
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
    arrays, infer_seconds = collect_predictions_v4(model, test_loader, device)
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


def replace_default_v3(config: ActV4Config):
    # Minimal adapter so we can reuse ActV3Dataset and run_epoch.
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


def quick_gate(df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    mean_skill = float(df["Skill_MSE"].mean())
    e48 = df[df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm]"] / e48["Persistence RMSE [mm]"]).max()) if not e48.empty else float("inf")
    e32_pos = int(((df["Tile"] == "E32N34") & (df["Horizon"].isin(["t+3", "t+5", "t+10"])) & (df["Skill_MSE"] > 0)).sum())
    e39_pos = int(((df["Tile"] == "E39N30") & (df["Horizon"].isin(["t+3", "t+5", "t+10"])) & (df["Skill_MSE"] > 0)).sum())
    mean_train = float(df["Elapsed seconds"].mean())
    complexity_ok = mean_train <= 5.0 * M1_TRAIN_SECONDS
    if mean_skill >= STRONG_GO_THRESHOLD and e48_ratio <= 1.03 and e32_pos >= 2 and e39_pos >= 2 and complexity_ok:
        status = "strong"
    elif mean_skill > WEAK_GO_THRESHOLD and e48_ratio <= 1.03 and complexity_ok:
        status = "weak"
    elif mean_skill >= WEAK_GO_THRESHOLD - 0.01 and e48_ratio <= 1.03 and complexity_ok:
        status = "close"
    else:
        status = "fail"
    return status, {
        "mean_skill_mse": mean_skill,
        "strong_go_threshold": STRONG_GO_THRESHOLD,
        "weak_go_threshold": WEAK_GO_THRESHOLD,
        "e48_max_rmse_ratio": e48_ratio,
        "e32_positive_long_horizons": e32_pos,
        "e39_positive_long_horizons": e39_pos,
        "mean_train_seconds": mean_train,
        "complexity_ok": complexity_ok,
        "status": status,
    }


def three_seed_gate(summary_df: pd.DataFrame, m1_summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    mean_std = float(summary_df["Skill_MSE std"].mean())
    e48 = summary_df[summary_df["Tile"] == "E48N24"]
    e48_ratio = float((e48["RMSE [mm] mean"] / e48["Persistence RMSE [mm] mean"]).max()) if not e48.empty else float("inf")
    e32 = summary_df[(summary_df["Tile"] == "E32N34") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e39 = summary_df[(summary_df["Tile"] == "E39N30") & (summary_df["Horizon"].isin(["t+3", "t+5", "t+10"]))]
    e32_stable = int((e32["Skill_MSE mean"] > e32["Skill_MSE std"]).sum())
    e39_stable = int((e39["Skill_MSE mean"] > e39["Skill_MSE std"]).sum())
    mean_train = float(summary_df["Elapsed seconds mean"].mean())
    complexity_ok = mean_train <= 5.0 * M1_TRAIN_SECONDS

    m1_long = m1_summary[m1_summary["Horizon"].isin(["t+3", "t+5", "t+10"])]
    merged = summary_df.merge(m1_long[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}), on=["Tile", "Horizon"], how="left")
    better_tiles = 0
    for tile in SEARCH_TILES:
        tile_sub = merged[(merged["Tile"] == tile) & (merged["Horizon"].isin(["t+3", "t+5", "t+10"]))]
        if tile_sub.empty:
            continue
        if float(tile_sub["Skill_MSE mean"].mean()) > float(tile_sub["M1 Skill_MSE mean"].mean()):
            better_tiles += 1

    if mean_skill >= STRONG_GO_THRESHOLD and e48_ratio <= 1.03 and e32_stable >= 2 and e39_stable >= 2 and complexity_ok and better_tiles >= 2 and mean_skill > mean_std:
        status = "strong"
    elif mean_skill > WEAK_GO_THRESHOLD and e48_ratio <= 1.03 and complexity_ok:
        status = "weak"
    else:
        status = "fail"
    return status, {
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "strong_go_threshold": STRONG_GO_THRESHOLD,
        "weak_go_threshold": WEAK_GO_THRESHOLD,
        "e48_max_rmse_ratio": e48_ratio,
        "e32_positive_long_horizons_stable": e32_stable,
        "e39_positive_long_horizons_stable": e39_stable,
        "mean_train_seconds": mean_train,
        "complexity_ok": complexity_ok,
        "better_tiles_vs_m1_long_horizon": better_tiles,
        "status": status,
    }


def quick_run(bundles: dict[str, TileBundle], config: ActV4Config, csv_name: str, reuse_if_exists: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for tile in SEARCH_TILES:
        print(f"[quick] start model={config.model_id} tile={tile} seed={config.seed}", flush=True)
        run = train_eval_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=ACT_V4_ROOT / "runs" / "quick" / config.model_id / tile / f"seed_{config.seed}",
            phase="quick",
            reuse_if_exists=reuse_if_exists,
        )
        print(f"[quick] done model={config.model_id} tile={tile} seed={config.seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
        rows.extend(run["result"]["rows"])
        diag_rows.extend(run["result"]["diag_rows"])
    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(diag_rows)
    df.to_csv(ACT_V4_ROOT / csv_name, index=False, encoding="utf-8-sig")
    return df, diag_df


def three_seed_run(bundles: dict[str, TileBundle], config: ActV4Config, phase: str, tiles: list[str], reuse_if_exists: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(config, seed=seed)
        for tile in tiles:
            print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
            run = train_eval_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=ACT_V4_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}",
                phase=phase,
                reuse_if_exists=reuse_if_exists,
            )
            print(f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f}", flush=True)
            rows.extend(run["result"]["rows"])
            diag_rows.extend(run["result"]["diag_rows"])
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def ablation_configs(base: ActV4Config) -> list[ActV4Config]:
    cfgs = [
        replace(base, model_id=f"{base.model_id}_noSSM", no_ssm=True, temporal_mode="none"),
        replace(base, model_id=f"{base.model_id}_GRU", temporal_mode="gru"),
        replace(base, model_id=f"{base.model_id}_noBound", no_bounded_residual=True),
        replace(base, model_id=f"{base.model_id}_noVol", no_volatility_scale=True),
        replace(base, model_id=f"{base.model_id}_directDisp", direct_displacement_head=True),
    ]
    if base.variant == "v4b":
        cfgs.append(replace(base, model_id=f"{base.model_id}_noFNO", no_fno=True))
    return cfgs


def write_final_report(
    quick_records: list[dict[str, Any]],
    final_candidate: str | None,
    final_status: str,
    three_seed_summary: pd.DataFrame | None,
    three_seed_gate_stats: dict[str, Any] | None,
    complexity_df: pd.DataFrame,
    ran_v4c: bool,
) -> Path:
    path = ACT_V4_ROOT / "v4_final_decision.md"
    lines = [
        "# ACT-Former v4 Go/No-Go 详细实验报告",
        "",
        "数据声明：本轮 v4 状态空间神经算子实验继续统一使用 **EGMS 2019-2023 Ortho U**，搜索子集固定为 `E32N34 / E39N30 / E48N24`。",
        "",
        "## 规则与假设",
        "",
        f"- 按文档，M1 参考线固定为 `Skill_MSE = {WEAK_GO_THRESHOLD:.4f}`，Strong GO 阈值为 `>= {STRONG_GO_THRESHOLD:.4f}`。",
        "- 由于决策树中“close to M1”没有精确定义，本轮额外假定：`mean Skill_MSE >= M1 - 0.01` 且 `E48N24 ratio <= 1.03` 视为 close。",
        "- 如果 quick 结果只达到 weak GO 或 close，但没有达到 strong GO，仍允许对更优候选做 3-seed，以确认它是否只是偶然提升。",
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
                f"- Quick 结果文件：`{record['quick_csv']}`",
                f"- 平均 Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- Strong GO 阈值 = {gate['strong_go_threshold']:.4f}",
                f"- Weak GO 阈值 = {gate['weak_go_threshold']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 正 skill 长 horizon 数 = {gate['e32_positive_long_horizons']}",
                f"- E39N30 正 skill 长 horizon 数 = {gate['e39_positive_long_horizons']}",
                f"- 复杂度是否通过 = `{'是' if gate['complexity_ok'] else '否'}`",
                f"- Quick 判定 = `{gate['status']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 3-Seed / Ablation / Full Expansion",
            "",
        ]
    )
    if three_seed_summary is None or three_seed_gate_stats is None:
        lines.append("- 没有 quick 候选满足进入 3-seed 的条件，因此没有执行 3-seed、ablation 或 six-tile expansion。")
        lines.append("")
    else:
        lines.extend(
            [
                f"- 入围候选：`{final_candidate}`",
                f"- 3-seed 文件：`{(ACT_V4_ROOT / 'v4_best_3seed.csv').as_posix()}`",
                f"- 3-seed 平均 Skill_MSE = {three_seed_gate_stats['mean_skill_mse']:.4f}",
                f"- 3-seed 平均 Skill_MSE std = {three_seed_gate_stats['mean_skill_std']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {three_seed_gate_stats['e48_max_rmse_ratio']:.4f}",
                f"- Long horizon 上优于 M1 的 tile 数 = {three_seed_gate_stats['better_tiles_vs_m1_long_horizon']}",
                f"- 最终 3-seed 判定 = `{three_seed_gate_stats['status']}`",
                "",
            ]
        )
        if final_status == "strong":
            lines.append(f"- 由于 `{final_candidate}` 达到 Strong GO，本轮还执行了结构消融和 six-tile expansion。")
            lines.append(f"- Ablation 文件：`{(ACT_V4_ROOT / 'v4_ablation.csv').as_posix()}`")
            lines.append(f"- Six-tile 文件：`{(ACT_V4_ROOT / 'v4_full_six_tile.csv').as_posix()}`")
            lines.append("")

    lines.extend(
        [
            "## Complexity",
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
            "## 最终结论",
            "",
        ]
    )
    if final_status == "strong":
        lines.extend(
            [
                f"- `{final_candidate}` 达到 Strong GO，可替代 M1 成为主模型。",
                "- 后续论文主线应转为 state-space deformation operator。",
            ]
        )
    elif final_status == "weak":
        lines.extend(
            [
                f"- `{final_candidate}` 仅达到 weak GO 或 supplementary 水平，不能替代 M1。",
                "- 论文主线仍保留 M1，V4 仅作为 supplementary architecture candidate。",
            ]
        )
    else:
        lines.extend(
            [
                "- `V4-A / V4-B / V4-C` 均未满足替代 M1 的条件。",
                "- 按计划，V4 是最后一个模型家族；因此模型搜索到此停止，最终主模型仍为 `M1`。",
                "- 论文主线应写成：在系统探索 query Transformer、baseline-bank、kinematic decomposition、state-space/operator 后，最稳健结构仍是 persistence-gated residual correction。",
            ]
        )
    lines.append("")
    if ran_v4c:
        lines.append("- 本轮已经执行过一次 `V4-C` optional sanity check，因此不再继续扩展新的模型家族。")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_act_v4_experiments() -> dict[str, Any]:
    ensure_dir(ACT_V4_ROOT)
    ensure_dir(CONFIG_ROOT)
    ensure_dir(V4_ERROR_MAP_ROOT)
    bundles = preprocess_all_tiles()

    baseline_rows = baseline_rows_for_search(bundles)
    baseline_summary = summarize_seeded_rows(baseline_rows)
    baseline_summary.to_csv(ACT_V4_ROOT / "v4_skill_table.csv", index=False, encoding="utf-8-sig")

    m1_summary = pd.read_csv(Path(OUTPUT_ROOT) / "act_v2" / "final_model_selection.csv")
    quick_records: list[dict[str, Any]] = []
    quick_summaries: dict[str, pd.DataFrame] = {}
    quick_diag_frames: list[pd.DataFrame] = []
    final_candidate_cfg: ActV4Config | None = None
    final_candidate_name: str | None = None
    final_status = "nogo"
    three_seed_summary: pd.DataFrame | None = None
    three_seed_gate_stats: dict[str, Any] | None = None
    ran_v4c = False

    print("[v4] quick start: V4-A", flush=True)
    v4a = default_v4_configs()[0]
    write_simple_yaml(CONFIG_ROOT / "V4-A.yaml", asdict(v4a))
    v4a_df, v4a_diag = quick_run(bundles, v4a, "v4_a_quick.csv")
    v4a_summary = summarize_seeded_rows(v4a_df)
    quick_summaries["V4-A"] = v4a_summary
    quick_diag_frames.append(v4a_diag)
    v4a_status, v4a_gate = quick_gate(v4a_df)
    quick_records.append({"model_id": "V4-A", "quick_csv": str(ACT_V4_ROOT / "v4_a_quick.csv"), "gate": v4a_gate})

    should_try_v4b = v4a_status in {"strong", "weak", "close"}
    v4b_df = pd.DataFrame()
    v4b_summary = pd.DataFrame()
    v4b_status = "not_run"
    v4b_gate: dict[str, Any] | None = None
    if should_try_v4b:
        print("[v4] quick start: V4-B", flush=True)
        v4b = default_v4_configs()[1]
        write_simple_yaml(CONFIG_ROOT / "V4-B.yaml", asdict(v4b))
        v4b_df, v4b_diag = quick_run(bundles, v4b, "v4_b_quick.csv")
        v4b_summary = summarize_seeded_rows(v4b_df)
        quick_summaries["V4-B"] = v4b_summary
        quick_diag_frames.append(v4b_diag)
        v4b_status, v4b_gate = quick_gate(v4b_df)
        quick_records.append({"model_id": "V4-B", "quick_csv": str(ACT_V4_ROOT / "v4_b_quick.csv"), "gate": v4b_gate})

    candidate_name: str | None = None
    candidate_cfg: ActV4Config | None = None
    candidate_status: str | None = None
    candidate_mean_skill = -float("inf")

    def consider_candidate(name: str, cfg: ActV4Config, status: str, gate: dict[str, Any]) -> None:
        nonlocal candidate_name, candidate_cfg, candidate_status, candidate_mean_skill
        mean_skill = gate["mean_skill_mse"]
        if status in {"strong", "weak"} and mean_skill > candidate_mean_skill:
            candidate_name = name
            candidate_cfg = cfg
            candidate_status = status
            candidate_mean_skill = mean_skill

    consider_candidate("V4-A", v4a, v4a_status, v4a_gate)
    if v4b_gate is not None:
        consider_candidate("V4-B", default_v4_configs()[1], v4b_status, v4b_gate)

    if v4b_gate is None:
        if v4a_status not in {"strong", "weak", "close"}:
            print("[v4] V4-A collapsed and not close; stopping before V4-B", flush=True)
    else:
        if candidate_name is None and v4a_gate["mean_skill_mse"] <= WEAK_GO_THRESHOLD and v4b_gate["mean_skill_mse"] <= WEAK_GO_THRESHOLD:
            ran_v4c = True

    if ran_v4c:
        print("[v4] quick start: V4-C optional sanity check", flush=True)
        v4c = default_v4_configs()[2]
        write_simple_yaml(CONFIG_ROOT / "V4-C.yaml", asdict(v4c))
        v4c_df, v4c_diag = quick_run(bundles, v4c, "v4_c_quick.csv")
        v4c_summary = summarize_seeded_rows(v4c_df)
        quick_summaries["V4-C"] = v4c_summary
        quick_diag_frames.append(v4c_diag)
        v4c_status, v4c_gate = quick_gate(v4c_df)
        quick_records.append({"model_id": "V4-C", "quick_csv": str(ACT_V4_ROOT / "v4_c_quick.csv"), "gate": v4c_gate})
        consider_candidate("V4-C", v4c, v4c_status, v4c_gate)

    search_rows = []
    for record in quick_records:
        gate = record["gate"]
        search_rows.append(
            {
                "Model": record["model_id"],
                "Mean Skill_MSE": gate["mean_skill_mse"],
                "Strong GO threshold": gate["strong_go_threshold"],
                "Weak GO threshold": gate["weak_go_threshold"],
                "E48 max RMSE ratio": gate["e48_max_rmse_ratio"],
                "E32 positive long horizons": gate["e32_positive_long_horizons"],
                "E39 positive long horizons": gate["e39_positive_long_horizons"],
                "Mean train seconds": gate["mean_train_seconds"],
                "Complexity OK": gate["complexity_ok"],
                "Quick status": gate["status"],
            }
        )
    pd.DataFrame(search_rows).to_csv(ACT_V4_ROOT / "v4_model_search.csv", index=False, encoding="utf-8-sig")

    if candidate_cfg is not None:
        print(f"[v4] three-seed start for {candidate_name}", flush=True)
        final_candidate_cfg = candidate_cfg
        final_candidate_name = candidate_name
        three_df, three_diag = three_seed_run(bundles, candidate_cfg, phase="final_3seed", tiles=SEARCH_TILES, reuse_if_exists=True)
        three_seed_summary = summarize_seeded_rows(three_df)
        three_seed_summary.to_csv(ACT_V4_ROOT / "v4_best_3seed.csv", index=False, encoding="utf-8-sig")
        quick_diag_frames.append(three_diag)
        final_status, three_seed_gate_stats = three_seed_gate(three_seed_summary, m1_summary)
        print(f"[v4] three-seed status for {candidate_name}: {final_status}", flush=True)
        if final_status == "strong":
            print("[v4] strong GO reached; running ablation and six-tile expansion", flush=True)
            ablation_rows: list[pd.DataFrame] = []
            for cfg in ablation_configs(candidate_cfg):
                ab_df, ab_diag = quick_run(bundles, replace(cfg, seed=42), f"{cfg.model_id.lower()}_ablation.csv")
                ab_df["Ablation"] = cfg.model_id
                ablation_rows.append(ab_df)
                quick_diag_frames.append(ab_diag)
            if ablation_rows:
                pd.concat(ablation_rows, ignore_index=True).to_csv(ACT_V4_ROOT / "v4_ablation.csv", index=False, encoding="utf-8-sig")
            full_df, full_diag = three_seed_run(bundles, candidate_cfg, phase="full_six_tile", tiles=TILE_ORDER, reuse_if_exists=True)
            full_df.to_csv(ACT_V4_ROOT / "v4_full_six_tile.csv", index=False, encoding="utf-8-sig")
            quick_diag_frames.append(full_diag)

    skill_frames = [baseline_summary]
    for df in quick_summaries.values():
        skill_frames.append(df)
    if three_seed_summary is not None:
        skill_frames.append(three_seed_summary)
    skill_table = pd.concat(skill_frames, ignore_index=True)
    skill_table.to_csv(ACT_V4_ROOT / "v4_skill_table.csv", index=False, encoding="utf-8-sig")

    diag_df = pd.concat(quick_diag_frames, ignore_index=True) if quick_diag_frames else pd.DataFrame()
    diag_df.to_csv(ACT_V4_ROOT / "v4_branch_diagnostics.csv", index=False, encoding="utf-8-sig")

    complexity_df = build_v4_complexity_table(final_candidate_name or "M1", three_seed_summary if three_seed_summary is not None else pd.DataFrame(), quick_summaries)
    complexity_df.to_csv(ACT_V4_ROOT / "v4_complexity_table.csv", index=False, encoding="utf-8-sig")

    if (ACT_V4_ROOT / "v4_complexity_table.csv").exists():
        comp = pd.read_csv(ACT_V4_ROOT / "v4_complexity_table.csv")
        from .act_v2 import ActV2Config, build_model as build_v2_model

        m0 = build_v2_model(ActV2Config(model_id="M0", gate_mode="none"), image_size=64, num_channels=4)
        m1 = build_v2_model(ActV2Config(model_id="M1", gate_mode="sigmoid"), image_size=64, num_channels=4)
        comp.loc[comp["Model"] == "M0_B7", "Param count"] = count_parameters(m0)
        comp.loc[comp["Model"] == "M1", "Param count"] = count_parameters(m1)
        comp.to_csv(ACT_V4_ROOT / "v4_complexity_table.csv", index=False, encoding="utf-8-sig")
        complexity_df = comp

    if "V4-A" in quick_summaries:
        make_v4_error_maps("V4-A", phase="quick", seed=42)
    if "V4-B" in quick_summaries:
        make_v4_error_maps("V4-B", phase="quick", seed=42)
    if "V4-C" in quick_summaries:
        make_v4_error_maps("V4-C", phase="quick", seed=42)

    report_path = write_final_report(
        quick_records=quick_records,
        final_candidate=final_candidate_name,
        final_status=final_status,
        three_seed_summary=three_seed_summary,
        three_seed_gate_stats=three_seed_gate_stats,
        complexity_df=complexity_df,
        ran_v4c=ran_v4c,
    )
    next_round_path = write_v4_next_round_plan(
        final_model="M1" if final_status != "strong" else (final_candidate_name or "M1"),
        best_model=final_candidate_name,
        final_status=final_status,
    )

    summary = {
        "report": str(report_path),
        "next_round_plan": str(next_round_path),
        "final_status": final_status,
        "final_model": "M1" if final_status != "strong" else final_candidate_name,
        "best_v4_candidate": final_candidate_name,
        "strong_go_threshold": STRONG_GO_THRESHOLD,
        "weak_go_threshold": WEAK_GO_THRESHOLD,
        "three_seed_gate": three_seed_gate_stats,
        "search_csv": str(ACT_V4_ROOT / "v4_model_search.csv"),
        "skill_table": str(ACT_V4_ROOT / "v4_skill_table.csv"),
        "complexity_table": str(ACT_V4_ROOT / "v4_complexity_table.csv"),
        "diagnostics": str(ACT_V4_ROOT / "v4_branch_diagnostics.csv"),
    }
    write_json(ACT_V4_ROOT / "run_summary.json", summary)
    return summary


def build_v4_complexity_table(best_model: str, best_summary: pd.DataFrame, quick_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_df = load_existing_baseline_summary()
    if not base_df.empty:
        for model_name in sorted(base_df["Model"].unique()):
            rows.append(complexity_row_from_summary(model_name, base_df[base_df["Model"] == model_name]))
    for model_name, summary in quick_summaries.items():
        rows.append(complexity_row_from_summary(model_name, summary))
    if best_model not in quick_summaries and not best_summary.empty:
        rows.append(complexity_row_from_summary(best_model, best_summary))
    return pd.DataFrame(rows).drop_duplicates(subset=["Model"], keep="last").sort_values("Model")


def make_v4_error_maps(model_id: str, phase: str, seed: int = 42) -> None:
    ensure_dir(V4_ERROR_MAP_ROOT)
    target_tiles = ["E32N34", "E48N24"]
    for tile in target_tiles:
        pred_path = ACT_V4_ROOT / "runs" / phase / model_id / tile / f"seed_{seed}" / "predictions.npz"
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
        fig.savefig(V4_ERROR_MAP_ROOT / f"{model_id}_{tile}_error_maps.png", dpi=220)
        plt.close(fig)


def write_v4_next_round_plan(final_model: str, best_model: str | None, final_status: str) -> Path:
    path = ACT_V4_ROOT / "next_round_plan.md"
    if final_status == "strong" and best_model is not None:
        text = "\n".join(
            [
                "# 下一轮计划",
                "",
                f"结论：V4 成功，当前最佳模型是 `{best_model}`，并已超过既有 M1 门槛。",
                "",
                "后续动作：",
                f"- 以 `{best_model}` 作为新的主模型，继续六 tile 主实验或回填 E1-E8 主表。",
                "- 方法段主贡献改写为 state-space deformation operator，并补足结构消融与复杂度比较。",
                "- 保留 M1 与其余 V4 候选的负结果，作为结构搜索证据。",
                "",
            ]
        )
    else:
        text = "\n".join(
            [
                "# 下一轮计划",
                "",
                f"结论：V4 未能替代 `{final_model}`，当前主模型保持 `{final_model}`。",
                "",
                "后续动作：",
                "- 保留 `act_v4` 全套 quick/3-seed/ablation 结果，作为最后一轮结构搜索证据。",
                f"- 正文和后续补实验继续使用 `{final_model}`。",
                "- 方法段把 V4 压缩为失败的 state-space / neural-operator 尝试，并强调 persistence-gated residual correction 仍是最稳健方案。",
                "",
            ]
        )
    path.write_text(text, encoding="utf-8")
    return path


def write_final_report(
    quick_records: list[dict[str, Any]],
    final_candidate: str | None,
    final_status: str,
    three_seed_summary: pd.DataFrame | None,
    three_seed_gate_stats: dict[str, Any] | None,
    complexity_df: pd.DataFrame,
    ran_v4c: bool,
) -> Path:
    path = ACT_V4_ROOT / "v4_final_decision.md"
    lines = [
        "# ACT-Former v4 Go/No-Go 详细实验报告",
        "",
        "数据声明：本轮 v4 状态空间神经算子实验继续统一使用 **EGMS 2019-2023 Ortho U**，搜索子集固定为 `E32N34 / E39N30 / E48N24`。这样做是因为数据获取限制下，当前主线只能在统一数据版本上进行合法比较。",
        "",
        "## 规则与假设",
        "",
        f"- 按文档，M1 参考线固定为 `Skill_MSE = {WEAK_GO_THRESHOLD:.4f}`，Strong GO 阈值为 `>= {STRONG_GO_THRESHOLD:.4f}`。",
        "- 由于决策树中的“close to M1”没有给出精确定义，本轮额外采用如下操作化假设：`mean Skill_MSE >= M1 - 0.01` 且 `E48N24 worst RMSE ratio <= 1.03` 视为 close。",
        "- 如果 quick 结果只达到 weak GO 或 close，但没有达到 strong GO，仍允许对更优候选做 3-seed，以确认其是否只是偶然提升。",
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
                f"- Quick 结果文件：`{record['quick_csv']}`",
                f"- 平均 Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- Strong GO 阈值 = {gate['strong_go_threshold']:.4f}",
                f"- Weak GO 阈值 = {gate['weak_go_threshold']:.4f}",
                f"- E48N24 最差 `RMSE / persistence` = {gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 正 skill 的长 horizon 数 = {gate['e32_positive_long_horizons']}",
                f"- E39N30 正 skill 的长 horizon 数 = {gate['e39_positive_long_horizons']}",
                f"- 复杂度是否通过 = `{'是' if gate['complexity_ok'] else '否'}`",
                f"- Quick 判定 = `{gate['status']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## 3-Seed / Ablation / Full Expansion",
            "",
        ]
    )
    if three_seed_summary is None or three_seed_gate_stats is None:
        lines.append("- 没有 quick 候选满足进入 3-seed 的条件，因此没有执行 3-seed、ablation 或 six-tile expansion。")
        lines.append("")
    else:
        lines.extend(
            [
                f"- 入围候选：`{final_candidate}`",
                f"- 3-seed 文件：`{(ACT_V4_ROOT / 'v4_best_3seed.csv').as_posix()}`",
                f"- 3-seed 平均 Skill_MSE = {three_seed_gate_stats['mean_skill_mse']:.4f}",
                f"- 3-seed 平均 Skill_MSE std = {three_seed_gate_stats['mean_skill_std']:.4f}",
                f"- E48N24 最差 `RMSE / persistence` = {three_seed_gate_stats['e48_max_rmse_ratio']:.4f}",
                f"- Long horizon 上优于 M1 的 tile 数 = {three_seed_gate_stats['better_tiles_vs_m1_long_horizon']}",
                f"- 最终 3-seed 判定 = `{three_seed_gate_stats['status']}`",
                "",
            ]
        )
        if final_status == "strong":
            lines.append(f"- 由于 `{final_candidate}` 达到 Strong GO，本轮还执行了结构消融和 six-tile expansion。")
            lines.append(f"- Ablation 文件：`{(ACT_V4_ROOT / 'v4_ablation.csv').as_posix()}`")
            lines.append(f"- Six-tile 文件：`{(ACT_V4_ROOT / 'v4_full_six_tile.csv').as_posix()}`")
            lines.append("")

    lines.extend(
        [
            "## Complexity",
            "",
        ]
    )
    for _, row in complexity_df.iterrows():
        train_seconds = float(row["Train seconds"]) if not pd.isna(row["Train seconds"]) else float("nan")
        infer_seconds = float(row["Infer seconds"]) if not pd.isna(row["Infer seconds"]) else float("nan")
        param_count = int(row["Param count"]) if not pd.isna(row["Param count"]) else "nan"
        lines.append(f"- {row['Model']}: params={param_count}, train={train_seconds:.2f}s, infer={infer_seconds:.4f}s.")

    lines.extend(
        [
            "",
            "## 最终结论",
            "",
        ]
    )
    if final_status == "strong":
        lines.extend(
            [
                f"- `{final_candidate}` 达到 Strong GO，可替代 M1 成为主模型。",
                "- 后续论文主线应转为 state-space deformation operator。",
            ]
        )
    elif final_status == "weak":
        lines.extend(
            [
                f"- `{final_candidate}` 仅达到 weak GO 或 supplementary 水平，不能替代 M1。",
                "- 论文主线仍保持 M1，V4 仅作为 supplementary architecture candidate。",
            ]
        )
    else:
        lines.extend(
            [
                "- `V4-A / V4-B / V4-C` 均未满足替代 M1 的条件。",
                "- 按计划，V4 是最后一个模型家族；因此模型搜索到此停止，最终主模型仍为 `M1`。",
                "- 论文主线应写成：在系统探索 query Transformer、baseline-bank、kinematic decomposition、state-space / operator 之后，最稳健结构仍然是 persistence-gated residual correction。",
            ]
        )
    lines.append("")
    if ran_v4c:
        lines.append("- 本轮已经执行过一次 `V4-C` optional sanity check，因此不再继续扩展新的模型家族。")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
