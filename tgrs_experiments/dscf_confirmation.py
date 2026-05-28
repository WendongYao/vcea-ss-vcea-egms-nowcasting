from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset

from .act_v2 import ActV2Config, FINAL_SEEDS, build_model as build_m1_model, compute_loss as compute_m1_loss, fit_linear_seasonal_train_window, predict_linear_seasonal, train_eval_single_tile, write_simple_yaml
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v4c_c3_candidate import c3_runtime_optimized_config, m1_scalar_config
from .act_v4c_efficiency import train_eval_rescue_single_tile
from .act_v5_selector import infer_c3_split, infer_m1_split
from .pipeline import (
    OUTPUT_ROOT,
    RAW_DIR,
    REGIME_NAMES,
    REGIME_TO_ID,
    TILE_ORDER,
    TileBundle,
    TileMetadata,
    aggregate_to_grid,
    build_split,
    compute_metrics,
    detect_date_columns,
    ensure_dir,
    extract_raw_archives,
    fit_descriptors,
    get_tensors,
    preprocess_all_tiles,
    read_xml_metadata,
    regime_maps_from_full_series,
    seed_everything,
    write_json,
)


CONFIRM_ROOT = ensure_dir(OUTPUT_ROOT / "confirmation")
EXTERNAL_ROOT = ensure_dir(CONFIRM_ROOT / "external_tiles")
RESOLUTION_ROOT = ensure_dir(CONFIRM_ROOT / "resolution_sensitivity")
HOTSPOT_ROOT = ensure_dir(CONFIRM_ROOT / "hotspot_optional")
FIG_ROOT = ensure_dir(CONFIRM_ROOT / "figures")

EXTERNAL_CANDIDATES = ["E37N42", "E45N51", "E30N35", "E42N21"]
RESOLUTION_TILES = ["E32N34", "E39N30", "E48N24"]
RESOLUTIONS = [64, 128, 512]
RESOLUTION_SEED = 42
FINAL_STATIC_CFG = {"method": "inverse", "temperature": 4.0, "shrinkage": 0.1, "m1_floor": 0.0}
MAINLINE_LABEL = "VCEA"
PHASE_EXTERNAL = "confirmation_external_locked"
PHASE_RESOLUTION = "confirmation_resolution"

EXTERNAL_MANIFEST_PATH = EXTERNAL_ROOT / "external_tile_manifest.csv"
FREEZE_MANIFEST_PATH = EXTERNAL_ROOT / "freeze_manifest.yaml"
EXTERNAL_PREPROCESS_REPORT_PATH = EXTERNAL_ROOT / "external_preprocess_report.md"
EXTERNAL_BASELINE_PATH = EXTERNAL_ROOT / "baseline_external.csv"
EXTERNAL_SUMMARY_PATH = EXTERNAL_ROOT / "external_summary.csv"
EXTERNAL_TILE_HORIZON_PATH = EXTERNAL_ROOT / "external_tile_horizon.csv"
EXTERNAL_REGIME_BREAKDOWN_PATH = EXTERNAL_ROOT / "external_regime_breakdown.csv"
EXTERNAL_STRESS_PATH = EXTERNAL_ROOT / "external_stress_report.csv"

RESOLUTION_DATA_CHECK_PATH = RESOLUTION_ROOT / "resolution_data_check.csv"
RESOLUTION_SUMMARY_PATH = RESOLUTION_ROOT / "resolution_64_128_512_summary.csv"
HOTSPOT_METRICS_PATH = RESOLUTION_ROOT / "hotspot_metrics.csv"
BOUNDARY_METRICS_PATH = RESOLUTION_ROOT / "boundary_gradient_metrics.csv"
MEMORY_RUNTIME_PATH = RESOLUTION_ROOT / "memory_runtime.csv"
RESOLUTION_QUICK_PATH = RESOLUTION_ROOT / "resolution_quick_rows.csv"
RESOLUTION_M1_3SEED_PATH = RESOLUTION_ROOT / "resolution_m1_three_seed_summary.csv"

CONFIRM_DECISION_PATH = CONFIRM_ROOT / "confirmation_decision.md"
FINAL_REPORT_PATH = CONFIRM_ROOT / "dscf_confirmation_tile_resolution_report.md"
SUMMARY_PATH = CONFIRM_ROOT / "run_summary.json"


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def model_param_count(model: torch.nn.Module) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_static_weights(loss_vec: np.ndarray, method: str, temperature: float, shrinkage: float, m1_floor: float, m1_index: int) -> np.ndarray:
    rel = loss_vec / max(float(loss_vec[0]), 1e-8)
    if method == "inverse":
        raw = 1.0 / np.power(np.maximum(rel, 1e-8), temperature)
    else:
        rel_shift = rel - rel.min()
        raw = np.exp(-temperature * rel_shift)
    raw = raw / max(float(raw.sum()), 1e-8)
    raw = (1.0 - shrinkage) * raw + shrinkage * (1.0 / len(raw))
    if m1_floor > 0 and raw[m1_index] < m1_floor:
        remain = max(1e-6, 1.0 - m1_floor)
        others = np.asarray([i for i in range(len(raw)) if i != m1_index], dtype=np.int64)
        other_sum = max(float(raw[others].sum()), 1e-8)
        raw[others] = raw[others] / other_sum * remain
        raw[m1_index] = m1_floor
    raw = np.clip(raw, 1e-8, None)
    raw = raw / raw.sum()
    return raw.astype(np.float32)


def skill_mse_for_mask(y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = valid_mask > 0.5
    model_mse = float(np.mean((y_pred[..., mask] - y_true[..., mask]) ** 2))
    pers_mse = float(np.mean((y_pers[..., mask] - y_true[..., mask]) ** 2))
    return float(1.0 - model_mse / max(pers_mse, 1e-6))


def rows_from_prediction(model_id: str, tile: str, seed: int, phase: str, arrays: dict[str, np.ndarray], pred: np.ndarray, elapsed: float = 0.0, infer: float = 0.0, best_epoch: int = -1, param_count: int = 0) -> pd.DataFrame:
    payload = {
        "y_pred_mm": pred.astype(np.float32),
        "y_true_mm": arrays["y_true_mm"].astype(np.float32),
        "persistence_mm": arrays["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": arrays["linear_seasonal_mm"].astype(np.float32),
        "valid_mask": arrays["valid_mask"].astype(np.float32),
        "regime_map": arrays["regime_map"].astype(np.int16),
        "current_frames": arrays["current_frames"].astype(np.int64),
        "target_indices": arrays["target_indices"].astype(np.int64),
    }
    return pd.DataFrame(
        result_rows(
            model_id=model_id,
            tile=tile,
            seed=seed,
            phase=phase,
            elapsed_seconds=elapsed,
            infer_seconds=infer,
            best_epoch=best_epoch,
            param_count=param_count,
            arrays=payload,
            horizons=[1, 3, 5, 10],
        )
    )


def summarize_rows_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if col not in set(keys + ["Seed", "Phase"]) and pd.api.types.is_numeric_dtype(df[col])
    ]
    grouped = df.groupby(keys, as_index=False)
    mean_df = grouped[numeric_cols].mean()
    std_df = grouped[numeric_cols].std().fillna(0.0)
    rename_mean = {col: f"{col} mean" for col in numeric_cols}
    rename_std = {col: f"{col} std" for col in numeric_cols}
    mean_df = mean_df.rename(columns=rename_mean)
    std_df = std_df.rename(columns=rename_std)
    merged = mean_df.merge(std_df, left_on=keys, right_on=keys, how="left")
    return merged


def build_resolution_tensors(
    easting: np.ndarray,
    northing: np.ndarray,
    disp: np.ndarray,
    dates: pd.DatetimeIndex,
    split: Any,
    fitted_points: dict[str, np.ndarray],
    grid_size: int,
    include_disp_stats: bool,
) -> dict[str, np.ndarray]:
    disp_grid, valid_grid, count_grid = aggregate_to_grid(easting, northing, disp, grid_size)
    vel, _, _ = aggregate_to_grid(easting, northing, fitted_points["velocity"], grid_size)
    acc, _, _ = aggregate_to_grid(easting, northing, fitted_points["acceleration"], grid_size)
    amp_map, _, _ = aggregate_to_grid(easting, northing, fitted_points["seasonal_amp"], grid_size)
    phase_map, _, _ = aggregate_to_grid(easting, northing, fitted_points["seasonal_phase"], grid_size)
    resid_map, _, _ = aggregate_to_grid(easting, northing, fitted_points["residual_std"], grid_size)
    r2_map, _, _ = aggregate_to_grid(easting, northing, fitted_points["trend_r2"], grid_size)
    full = regime_maps_from_full_series(disp_grid, dates)
    time_sin = np.sin(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25).astype(np.float32)
    time_cos = np.cos(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25).astype(np.float32)
    tensors = {
        "disp": disp_grid.astype(np.float32),
        "valid_mask": valid_grid.astype(np.float32),
        "count_map": count_grid.astype(np.float32),
        "descriptor_velocity": vel.astype(np.float32),
        "descriptor_acc": acc.astype(np.float32),
        "descriptor_amp": amp_map.astype(np.float32),
        "seasonal_phase": phase_map.astype(np.float32),
        "residual_std": resid_map.astype(np.float32),
        "trend_r2": r2_map.astype(np.float32),
        "regime_map": full["regime_map"].astype(np.int16),
        "velocity_full": full["velocity_full"].astype(np.float32),
        "acceleration_full": full["acceleration_full"].astype(np.float32),
        "seasonal_amp_full": full["seasonal_amp_full"].astype(np.float32),
        "seasonal_phase_full": full["seasonal_phase_full"].astype(np.float32),
        "change_score_full": full["change_score_full"].astype(np.float32),
        "time_sin": time_sin,
        "time_cos": time_cos,
    }
    if include_disp_stats:
        tensors["disp_mu"] = disp_grid[: split.train_frame_end].mean(axis=0).astype(np.float32)
        tensors["disp_sigma"] = (disp_grid[: split.train_frame_end].std(axis=0).astype(np.float32) + 1e-6)
    return tensors


def build_bundle_from_tile_id(tile_id: str, raw_files: dict[str, dict[str, Path]], extra_resolutions: list[int] | None = None) -> TileBundle:
    extra_resolutions = extra_resolutions or []
    raw = raw_files[tile_id]
    frame = pd.read_csv(raw["csv"])
    dates = pd.to_datetime(detect_date_columns(frame), format="%Y%m%d")
    split = build_split(len(dates), input_len=16, horizons=[1, 3, 5, 10])
    easting = frame["easting"].to_numpy(dtype=np.float32)
    northing = frame["northing"].to_numpy(dtype=np.float32)
    disp = frame[[d.strftime("%Y%m%d") for d in dates]].to_numpy(dtype=np.float32)
    t_all = ((dates - dates[0]).days.values.astype(np.float32)) / 365.25
    fitted_points = fit_descriptors(disp[:, : split.train_frame_end], t_all[: split.train_frame_end])

    tensors64 = build_resolution_tensors(easting, northing, disp, dates, split, fitted_points, 64, include_disp_stats=True)
    tensors128 = build_resolution_tensors(easting, northing, disp, dates, split, fitted_points, 128, include_disp_stats=False)
    extra_tensors: dict[int, dict[str, np.ndarray]] = {}
    for resolution in extra_resolutions:
        if resolution in {64, 128}:
            continue
        extra_tensors[resolution] = build_resolution_tensors(
            easting, northing, disp, dates, split, fitted_points, resolution, include_disp_stats=False
        )
    product_level, production_date = read_xml_metadata(raw["xml"])
    meta = TileMetadata(
        tile_id=tile_id,
        role="external" if tile_id not in TILE_ORDER else "main",
        zip_name=raw["zip"].name,
        csv_name=raw["csv"].name,
        xml_name=raw["xml"].name,
        product_level=product_level,
        production_date=production_date,
        time_span=f"{dates[0].date()} - {dates[-1].date()}",
        num_epochs=len(dates),
        num_raw_points=len(frame),
        valid_grid_ratio_64=float(tensors64["valid_mask"].mean()),
        valid_grid_ratio_128=float(tensors128["valid_mask"].mean()),
        displacement_min=float(np.nanmin(disp)),
        displacement_max=float(np.nanmax(disp)),
        displacement_std=float(np.nanstd(disp)),
        velocity_min=float(np.nanmin(fitted_points["velocity"])),
        velocity_max=float(np.nanmax(fitted_points["velocity"])),
        velocity_std=float(np.nanstd(fitted_points["velocity"])),
        seasonal_amp_min=float(np.nanmin(fitted_points["seasonal_amp"])),
        seasonal_amp_max=float(np.nanmax(fitted_points["seasonal_amp"])),
        seasonal_amp_std=float(np.nanstd(fitted_points["seasonal_amp"])),
        missing_ratio=float(np.isnan(disp).mean()),
        assigned_regime=str(max({name: float((tensors64["regime_map"] == rid).mean()) for name, rid in REGIME_TO_ID.items()}.items(), key=lambda kv: kv[1])[0]),
    )
    return TileBundle(
        tile_id=tile_id,
        dates=[d.strftime("%Y-%m-%d") for d in dates],
        split=split,
        prod_date=production_date,
        product_level=product_level,
        raw_csv_path=str(raw["csv"]),
        metadata=meta,
        paths={"tile_dir": str(raw["dir"]), "csv": str(raw["csv"]), "xml": str(raw["xml"])},
        tensors64=tensors64,
        tensors128=tensors128,
        extra_tensors=extra_tensors,
        hotspot=None,
    )


def attach_resolution(bundle: TileBundle, resolution: int) -> None:
    if resolution in {64, 128} or resolution in bundle.extra_tensors:
        return
    raw_files = extract_raw_archives()
    refreshed = build_bundle_from_tile_id(bundle.tile_id, raw_files, extra_resolutions=[resolution])
    bundle.extra_tensors[resolution] = refreshed.extra_tensors[resolution]


def hotspot_mask_from_train(tensors: dict[str, np.ndarray]) -> np.ndarray:
    valid = tensors["valid_mask"] > 0.5
    score = (
        np.abs(tensors["descriptor_velocity"])
        + 0.5 * np.abs(tensors["descriptor_acc"])
        + tensors["descriptor_amp"]
        + np.maximum(tensors["change_score_full"], 0.0)
    )
    q = float(np.quantile(score[valid], 0.90)) if valid.any() else 0.0
    mask = valid & (score >= q)
    if mask.sum() < max(16, int(valid.sum() * 0.02)):
        q = float(np.quantile(score[valid], 0.85)) if valid.any() else 0.0
        mask = valid & (score >= q)
    return mask.astype(np.float32)


def gradient_rmse(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray) -> float:
    true_gx = np.gradient(y_true, axis=-1)
    true_gy = np.gradient(y_true, axis=-2)
    pred_gx = np.gradient(y_pred, axis=-1)
    pred_gy = np.gradient(y_pred, axis=-2)
    true_mag = np.sqrt(true_gx**2 + true_gy**2)
    pred_mag = np.sqrt(pred_gx**2 + pred_gy**2)
    mask = valid_mask > 0.5
    return float(np.sqrt(np.mean((pred_mag[..., mask] - true_mag[..., mask]) ** 2)))


def regime_breakdown_rows(model_id: str, tile: str, seed: int, arrays: dict[str, np.ndarray], pred: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = arrays["valid_mask"] > 0.5
    regime = arrays["regime_map"]
    for h_idx, horizon in enumerate([1, 3, 5, 10]):
        y_true = arrays["y_true_mm"][:, h_idx : h_idx + 1]
        y_pred = pred[:, h_idx : h_idx + 1]
        y_pers = arrays["persistence_mm"][:, h_idx : h_idx + 1]
        for rid, name in enumerate(REGIME_NAMES):
            regime_mask = valid & (regime == rid)
            if regime_mask.sum() < 12:
                continue
            metrics = compute_metrics(y_true, y_pred, regime_mask.astype(np.float32))
            rows.append(
                {
                    "Model": model_id,
                    "Tile": tile,
                    "Seed": seed,
                    "Horizon": f"t+{horizon}",
                    "Regime": name,
                    "Skill_MSE": skill_mse_for_mask(y_true, y_pred, y_pers, regime_mask.astype(np.float32)),
                    "RMSE [mm]": metrics["RMSE [mm]"],
                    "MAE [mm]": metrics["MAE [mm]"],
                    "SSIM": metrics["SSIM"],
                }
            )
    return rows


class OnTheFlyActDataset(Dataset):
    def __init__(self, bundle: TileBundle, resolution: int, split_name: str, config: ActV2Config) -> None:
        self.bundle = bundle
        self.resolution = resolution
        self.config = config
        tensors = get_tensors(bundle, resolution)
        self.disp = tensors["disp"].astype(np.float32)
        self.mask = tensors["valid_mask"].astype(np.float32)
        self.regime_map = tensors["regime_map"].astype(np.int16)
        self.gate_desc = np.stack(
            [
                tensors["descriptor_velocity"],
                tensors["descriptor_amp"],
                tensors["trend_r2"],
                tensors["descriptor_acc"],
            ],
            axis=0,
        ).astype(np.float32)
        desc_mu = self.gate_desc.mean(axis=(1, 2), keepdims=True)
        desc_sigma = self.gate_desc.std(axis=(1, 2), keepdims=True) + 1e-6
        self.gate_desc = ((self.gate_desc - desc_mu) / desc_sigma).astype(np.float32)

        train_end = bundle.split.train_frame_end
        self.disp_mu = self.disp[:train_end].mean(axis=0).astype(np.float32)
        self.disp_sigma = self.disp[:train_end].std(axis=0).astype(np.float32) + 1e-6
        increments = np.diff(self.disp, axis=0, prepend=self.disp[0:1]).astype(np.float32)
        inc_train = increments[1:train_end] if train_end > 1 else increments[:1]
        self.inc_mu = inc_train.mean(axis=0).astype(np.float32)
        self.inc_sigma = inc_train.std(axis=0).astype(np.float32) + 1e-6

        dates = pd.to_datetime(bundle.dates)
        self.time_feat = np.stack(
            [
                np.sin(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
                np.cos(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
            ],
            axis=1,
        ).astype(np.float32)
        self.coeffs, self.t_all = fit_linear_seasonal_train_window(bundle, resolution)
        if split_name == "train":
            self.current_frames = list(range(config.input_len - 1, bundle.split.train_frame_end - max(config.horizons)))
        elif split_name == "val":
            self.current_frames = list(range(bundle.split.train_frame_end, bundle.split.val_frame_end - max(config.horizons)))
        else:
            self.current_frames = list(range(bundle.split.val_frame_end, bundle.split.total_frames - max(config.horizons)))

    def __len__(self) -> int:
        return len(self.current_frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        current = self.current_frames[index]
        start = current - self.config.input_len + 1
        disp_window = self.disp[start : current + 1]
        disp_norm = ((disp_window - self.disp_mu[None, :, :]) / self.disp_sigma[None, :, :]).astype(np.float32)
        if start == 0:
            prev = self.disp[start : current + 1]
            increments = np.diff(prev, axis=0, prepend=prev[0:1]).astype(np.float32)
        else:
            prev = self.disp[start - 1 : current + 1]
            increments = np.diff(prev, axis=0).astype(np.float32)
        inc_norm = ((increments - self.inc_mu[None, :, :]) / self.inc_sigma[None, :, :]).astype(np.float32)
        h, w = self.disp.shape[1], self.disp.shape[2]
        time_expand = np.tile(self.time_feat[start : current + 1, :, None, None], (1, 1, h, w)).astype(np.float32)
        x = np.concatenate([disp_norm[:, None, :, :], inc_norm[:, None, :, :], time_expand], axis=1).astype(np.float32)

        target_indices = [current + h_step for h_step in self.config.horizons]
        target_mm = np.stack([self.disp[idx] for idx in target_indices], axis=0)[:, None, :, :].astype(np.float32)
        persistence_mm = np.repeat(self.disp[current][None, None, :, :], len(self.config.horizons), axis=0).astype(np.float32)
        ls_mm = predict_linear_seasonal(self.coeffs, self.t_all, target_indices, h, w)[:, None, :, :].astype(np.float32)
        future_t = self.time_feat[target_indices].astype(np.float32)
        return {
            "x": torch.from_numpy(x).float(),
            "target_mm": torch.from_numpy(target_mm).float(),
            "persistence_mm": torch.from_numpy(persistence_mm).float(),
            "linear_seasonal_mm": torch.from_numpy(ls_mm).float(),
            "future_time": torch.from_numpy(future_t).float(),
            "mask": torch.from_numpy(self.mask).float(),
            "regime": torch.from_numpy(self.regime_map).long(),
            "gate_desc": torch.from_numpy(self.gate_desc).float(),
            "disp_mu": torch.from_numpy(self.disp_mu).float(),
            "disp_sigma": torch.from_numpy(self.disp_sigma).float(),
            "current_frame": torch.tensor(current, dtype=torch.long),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
        }


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def resolution_config(resolution: int, seed: int) -> ActV2Config:
    patch_size = {64: 8, 128: 16, 512: 64}[resolution]
    batch_size = {64: 8, 128: 2, 512: 1}[resolution]
    epochs = {64: 12, 128: 10, 512: 6}[resolution]
    patience = {64: 4, 128: 3, 512: 2}[resolution]
    return ActV2Config(
        model_id=f"M1_res{resolution}",
        gate_mode="sigmoid",
        gate_granularity="scalar",
        patch_size=patch_size,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        seed=seed,
        num_workers=0,
        use_amp=True,
    )


def run_epoch_dynamic(model: torch.nn.Module, loader: DataLoader, optimizer: optim.Optimizer | None, scaler: amp.GradScaler | None, config: ActV2Config, device: torch.device) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    skills: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = batch_to_device(batch, device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            pred = model(batch)
            loss, skill_mse = compute_m1_loss(pred, batch, config)
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
    return {"loss": float(np.mean(losses)), "skill_mse": float(np.mean(skills))}


def collect_predictions_dynamic(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    model.eval()
    pred_list: list[np.ndarray] = []
    target_list: list[np.ndarray] = []
    pers_list: list[np.ndarray] = []
    ls_list: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    regime_list: list[np.ndarray] = []
    current_frame_list: list[np.ndarray] = []
    target_index_list: list[np.ndarray] = []
    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            pred, aux = model(batch, return_aux=True)
            pred_list.append(pred.cpu().numpy()[:, :, 0].astype(np.float32))
            target_list.append(batch["target_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            pers_list.append(batch["persistence_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            ls_list.append(batch["linear_seasonal_mm"].cpu().numpy()[:, :, 0].astype(np.float32))
            weights_list.append(aux["expert_weights"].cpu().numpy().astype(np.float32))
            mask_list.append(batch["mask"].cpu().numpy().astype(np.float32))
            regime_list.append(batch["regime"].cpu().numpy().astype(np.int16))
            current_frame_list.append(batch["current_frame"].cpu().numpy().astype(np.int64))
            target_index_list.append(batch["target_indices"].cpu().numpy().astype(np.int64))
    infer_seconds = time.time() - start
    return {
        "y_pred_mm": np.concatenate(pred_list, axis=0),
        "y_true_mm": np.concatenate(target_list, axis=0),
        "persistence_mm": np.concatenate(pers_list, axis=0),
        "linear_seasonal_mm": np.concatenate(ls_list, axis=0),
        "gate_weights": np.concatenate(weights_list, axis=0),
        "valid_mask": mask_list[0][0] if mask_list[0].ndim == 3 else mask_list[0],
        "regime_map": regime_list[0][0] if regime_list[0].ndim == 3 else regime_list[0],
        "current_frames": np.concatenate(current_frame_list, axis=0),
        "target_indices": np.concatenate(target_index_list, axis=0),
    }, infer_seconds


def train_eval_dynamic_m1(bundle: TileBundle, resolution: int, seed: int, output_dir: Path, reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(output_dir)
    result_path = output_dir / "result.json"
    pred_path = output_dir / "predictions.npz"
    ckpt_path = output_dir / "model.pth"
    config = resolution_config(resolution, seed)
    if reuse_if_exists and result_path.exists() and pred_path.exists() and ckpt_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        arr = np.load(pred_path)
        return {"result": result, "arrays": {k: arr[k] for k in arr.files}}

    write_simple_yaml(output_dir / "config.yaml", asdict(config))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = OnTheFlyActDataset(bundle, resolution, "train", config)
    val_ds = OnTheFlyActDataset(bundle, resolution, "val", config)
    test_ds = OnTheFlyActDataset(bundle, resolution, "test", config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_m1_model(config, image_size=resolution, num_channels=4).to(device)
    param_count = model_param_count(model)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    best_epoch = -1
    best_score = float("inf")
    wait = 0
    history: list[dict[str, float]] = []
    start = time.time()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch_dynamic(model, train_loader, optimizer, scaler, config, device)
        val_stats = run_epoch_dynamic(model, val_loader, None, None, config, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_skill_mse": train_stats["skill_mse"],
                "val_loss": val_stats["loss"],
                "val_skill_mse": val_stats["skill_mse"],
            }
        )
        if val_stats["loss"] < best_score:
            best_score = val_stats["loss"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(config), "resolution": resolution}, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    arrays, infer_seconds = collect_predictions_dynamic(model, test_loader, device)
    elapsed = time.time() - start
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
    rows = rows_from_prediction(f"M1_res{resolution}", bundle.tile_id, seed, PHASE_RESOLUTION, arrays, arrays["y_pred_mm"], elapsed, infer_seconds, best_epoch, param_count)
    result = {
        "model_id": f"M1_res{resolution}",
        "tile": bundle.tile_id,
        "seed": seed,
        "resolution": resolution,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "infer_seconds": infer_seconds,
        "param_count": param_count,
        "peak_gpu_memory_mb": peak_mb,
        "history": history,
        "rows": rows.to_dict(orient="records"),
    }
    np.savez_compressed(pred_path, **arrays)
    write_json(result_path, result)
    return {"result": result, "arrays": arrays}


def baseline_cache_from_m1_arrays(bundle: TileBundle, resolution: int, m1_arrays: dict[str, np.ndarray], split_name: str) -> dict[str, np.ndarray]:
    dataset = OnTheFlyActDataset(bundle, resolution, split_name, resolution_config(resolution, RESOLUTION_SEED))
    current_frames = np.asarray(dataset.current_frames, dtype=np.int64)
    target_indices = np.stack([np.asarray([c + h for h in [1, 3, 5, 10]], dtype=np.int64) for c in current_frames], axis=0)
    disp = get_tensors(bundle, resolution)["disp"].astype(np.float32)
    y_true = np.stack([disp[target_indices[:, i]] for i in range(target_indices.shape[1])], axis=1).astype(np.float32)
    persistence = np.repeat(disp[current_frames][:, None, :, :], 4, axis=1).astype(np.float32)
    coeffs, t_all = fit_linear_seasonal_train_window(bundle, resolution)
    ls = np.stack(
        [
            predict_linear_seasonal(coeffs, t_all, target_indices[row_idx].tolist(), disp.shape[1], disp.shape[2])
            for row_idx in range(target_indices.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "y_true_mm": y_true,
        "persistence_mm": persistence,
        "linear_seasonal_mm": ls,
        "m1_pred_mm": m1_arrays["y_pred_mm"].astype(np.float32),
        "valid_mask": get_tensors(bundle, resolution)["valid_mask"].astype(np.float32),
        "regime_map": get_tensors(bundle, resolution)["regime_map"].astype(np.int16),
        "current_frames": current_frames.astype(np.int64),
        "target_indices": target_indices.astype(np.int64),
    }


def static_proxy_prediction_3expert(val_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = val_cache["valid_mask"] > 0.5
    weights = np.zeros((4, 3), dtype=np.float32)
    for h_idx in range(4):
        losses = np.asarray(
            [
                float(((val_cache["persistence_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
                float(((val_cache["linear_seasonal_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
                float(((val_cache["m1_pred_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
            ],
            dtype=np.float64,
        )
        weights[h_idx] = fit_static_weights(
            losses,
            method=FINAL_STATIC_CFG["method"],
            temperature=FINAL_STATIC_CFG["temperature"],
            shrinkage=FINAL_STATIC_CFG["shrinkage"],
            m1_floor=FINAL_STATIC_CFG["m1_floor"],
            m1_index=2,
        )
    pred = np.zeros_like(test_cache["y_true_mm"], dtype=np.float32)
    for h_idx in range(4):
        pred[:, h_idx] = (
            weights[h_idx, 0] * test_cache["persistence_mm"][:, h_idx]
            + weights[h_idx, 1] * test_cache["linear_seasonal_mm"][:, h_idx]
            + weights[h_idx, 2] * test_cache["m1_pred_mm"][:, h_idx]
        )
    return pred.astype(np.float32), weights


def external_static_prediction_4expert(val_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = val_cache["valid_mask"] > 0.5
    weights = np.zeros((4, 4), dtype=np.float32)
    for h_idx in range(4):
        losses = np.asarray(
            [
                float(((val_cache["persistence_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
                float(((val_cache["linear_seasonal_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
                float(((val_cache["m1_pred_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
                float(((val_cache["c3_pred_mm"][:, h_idx] - val_cache["y_true_mm"][:, h_idx]) ** 2)[:, valid].mean()),
            ],
            dtype=np.float64,
        )
        weights[h_idx] = fit_static_weights(
            losses,
            method=FINAL_STATIC_CFG["method"],
            temperature=FINAL_STATIC_CFG["temperature"],
            shrinkage=FINAL_STATIC_CFG["shrinkage"],
            m1_floor=FINAL_STATIC_CFG["m1_floor"],
            m1_index=2,
        )
    pred = np.zeros_like(test_cache["y_true_mm"], dtype=np.float32)
    for h_idx in range(4):
        pred[:, h_idx] = (
            weights[h_idx, 0] * test_cache["persistence_mm"][:, h_idx]
            + weights[h_idx, 1] * test_cache["linear_seasonal_mm"][:, h_idx]
            + weights[h_idx, 2] * test_cache["m1_pred_mm"][:, h_idx]
            + weights[h_idx, 3] * test_cache["c3_pred_mm"][:, h_idx]
        )
    return pred.astype(np.float32), weights


def external_cache_from_runs(bundle: TileBundle, m1_run_dir: Path, c3_run_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    m1 = infer_m1_split(bundle, m1_run_dir, split_name)
    c3 = infer_c3_split(bundle, c3_run_dir, split_name)
    if not np.array_equal(m1["current_frames"].reshape(-1), c3["current_frames"].reshape(-1)):
        raise ValueError(f"Frame mismatch for tile={bundle.tile_id} split={split_name}")
    if not np.array_equal(m1["target_indices"], c3["target_indices"]):
        raise ValueError(f"Target mismatch for tile={bundle.tile_id} split={split_name}")
    tensors = bundle.tensors64
    return {
        "y_true_mm": m1["y_true_mm"].astype(np.float32),
        "persistence_mm": m1["persistence_mm"].astype(np.float32),
        "linear_seasonal_mm": m1["linear_seasonal_mm"].astype(np.float32),
        "m1_pred_mm": m1["y_pred_mm"].astype(np.float32),
        "c3_pred_mm": c3["y_pred_mm"].astype(np.float32),
        "valid_mask": m1["valid_mask"].astype(np.float32),
        "regime_map": m1["regime_map"].astype(np.int16),
        "current_frames": m1["current_frames"].reshape(-1).astype(np.int64),
        "target_indices": m1["target_indices"].astype(np.int64),
        "residual_std": tensors["residual_std"].astype(np.float32),
    }


def summarize_external_status(summary_df: pd.DataFrame) -> dict[str, Any]:
    model_rows = []
    for model, part in summary_df.groupby("Model"):
        stress = part[part["Tile"].isin(["E37N42", "E45N51", "E30N35", "E42N21"])]
        model_rows.append(
            {
                "Model": model,
                "Mean Skill_MSE": float(part["Skill_MSE mean"].mean()),
                "Better tiles vs persistence": int((part.groupby("Tile")["Skill_MSE mean"].mean() > 0).sum()),
                "Stress ratio": float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()),
                "Std": float(part["Skill_MSE std"].mean()) if "Skill_MSE std" in part else 0.0,
            }
        )
    return {"rows": model_rows}


def make_confirmation_plots(external_summary: pd.DataFrame, resolution_summary: pd.DataFrame) -> None:
    pivot = external_summary[external_summary["Model"].isin(["Persistence", "Linear-Seasonal", "M1", "C3_opt", MAINLINE_LABEL])].pivot_table(index="Tile", columns=["Model", "Horizon"], values="Skill_MSE mean")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-0.2, vmax=0.3)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{m}\n{h}" for m, h in pivot.columns], rotation=90, fontsize=8)
    ax.set_title("External tile-horizon Skill_MSE heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_external_tile_generalization.png", dpi=220)
    plt.close(fig)

    res_plot = resolution_summary[resolution_summary["Model"].isin(["Persistence", "Linear-Seasonal", "M1", "StaticVal-PLM1"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model, part in res_plot.groupby("Model"):
        grouped = part.groupby("Resolution")["Skill_MSE mean"].mean().reindex(RESOLUTIONS)
        ax.plot(grouped.index.astype(str), grouped.values, marker="o", linewidth=2, label=model)
    ax.set_ylabel("Mean Skill_MSE")
    ax.set_title("Resolution sensitivity (64 / 128 / 512)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_resolution_sensitivity.png", dpi=220)
    plt.close(fig)


def build_freeze_manifest() -> None:
    lines = [
        "data_product: EGMS 2019-2023 Ortho U",
        "mainline_after_dscf: VCEA",
        "dscf_status: no_go_after_quick_return_to_vcea",
        "external_eval_models:",
        "  - Persistence",
        "  - Linear-Seasonal",
        "  - M1",
        "  - C3_opt",
        "  - VCEA",
        "external_eval_notes:",
        "  - exact VCEA uses frozen four-expert static validation weighting on each external tile",
        "resolution_sensitivity_models:",
        "  - Persistence",
        "  - Linear-Seasonal",
        "  - M1",
        "  - StaticVal-PLM1 proxy",
        "resolution_assumption:",
        "  exact four-expert VCEA at 128/512 is not run because it would require high-resolution C3 retraining and would change sensitivity into a new search",
        "static_weighting:",
        f"  method: {FINAL_STATIC_CFG['method']}",
        f"  temperature: {FINAL_STATIC_CFG['temperature']}",
        f"  shrinkage: {FINAL_STATIC_CFG['shrinkage']}",
        f"  m1_floor: {FINAL_STATIC_CFG['m1_floor']}",
    ]
    FREEZE_MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_external_experiment() -> dict[str, Any]:
    raw_files = extract_raw_archives()
    available_tiles = [tile for tile in EXTERNAL_CANDIDATES if tile in raw_files]
    bundles = {tile: build_bundle_from_tile_id(tile, raw_files) for tile in available_tiles}
    manifest_rows = []
    for tile, bundle in bundles.items():
        valid = bundle.tensors64["valid_mask"] > 0.5
        regime = bundle.tensors64["regime_map"]
        regime_props = {name: float(((regime == rid) & valid).sum() / max(int(valid.sum()), 1)) for name, rid in REGIME_TO_ID.items()}
        manifest_rows.append(
            {
                "Tile": tile,
                "Role": "external",
                "Dominant regime": bundle.metadata.assigned_regime,
                "Num points": bundle.metadata.num_raw_points,
                "Epochs": bundle.metadata.num_epochs,
                "Valid64": bundle.metadata.valid_grid_ratio_64,
                "Valid128": bundle.metadata.valid_grid_ratio_128,
                **{f"Regime {k}": v for k, v in regime_props.items()},
            }
        )
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(EXTERNAL_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    write_markdown(
        EXTERNAL_PREPROCESS_REPORT_PATH,
        [
            "# External Preprocess Report",
            "",
            "数据声明：本轮确认实验继续统一使用 **EGMS 2019-2023 Ortho U**，external set 仅作为冻结后的 locked evaluation。",
            "",
            f"- external candidates 落地数量：{len(available_tiles)}。",
            f"- external tile 列表：{', '.join(available_tiles)}。",
            "- 说明：由于 DSCF main-six 已判定 no-go，本轮 external validation 只用于 `VCEA/M1` robustness，不用于重新搜索 DSCF。",
        ],
    )

    base_rows: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []
    for tile, bundle in bundles.items():
        for seed in FINAL_SEEDS:
            m1_cfg = replace(m1_scalar_config(), seed=seed)
            m1_dir = EXTERNAL_ROOT / "runs" / "M1" / tile / f"seed_{seed}"
            c3_cfg = replace(c3_runtime_optimized_config(), seed=seed)
            c3_dir = EXTERNAL_ROOT / "runs" / "C3_opt" / tile / f"seed_{seed}"

            m1_run = train_eval_single_tile(bundle, m1_cfg, m1_dir, PHASE_EXTERNAL, resolution=64, reuse_if_exists=True)
            c3_run = train_eval_rescue_single_tile(bundle, c3_cfg, c3_dir, PHASE_EXTERNAL, reuse_if_exists=True)
            val_cache = external_cache_from_runs(bundle, m1_dir, c3_dir, "val")
            test_cache = external_cache_from_runs(bundle, m1_dir, c3_dir, "test")
            vcea_pred, weights = external_static_prediction_4expert(val_cache, test_cache)
            _ = weights

            base_rows.append(rows_from_prediction("Persistence", tile, seed, PHASE_EXTERNAL, test_cache, test_cache["persistence_mm"]))
            base_rows.append(rows_from_prediction("Linear-Seasonal", tile, seed, PHASE_EXTERNAL, test_cache, test_cache["linear_seasonal_mm"]))
            base_rows.append(rows_from_prediction("M1", tile, seed, PHASE_EXTERNAL, test_cache, test_cache["m1_pred_mm"]))
            base_rows.append(rows_from_prediction("C3_opt", tile, seed, PHASE_EXTERNAL, test_cache, test_cache["c3_pred_mm"]))
            base_rows.append(rows_from_prediction(MAINLINE_LABEL, tile, seed, PHASE_EXTERNAL, test_cache, vcea_pred))
            regime_rows.extend(regime_breakdown_rows("M1", tile, seed, test_cache, test_cache["m1_pred_mm"]))
            regime_rows.extend(regime_breakdown_rows("C3_opt", tile, seed, test_cache, test_cache["c3_pred_mm"]))
            regime_rows.extend(regime_breakdown_rows(MAINLINE_LABEL, tile, seed, test_cache, vcea_pred))

    external_rows = pd.concat(base_rows, ignore_index=True)
    external_rows.to_csv(EXTERNAL_BASELINE_PATH, index=False, encoding="utf-8-sig")
    summary = summarize_seeded_rows(external_rows)
    summary.to_csv(EXTERNAL_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(EXTERNAL_TILE_HORIZON_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(regime_rows).to_csv(EXTERNAL_REGIME_BREAKDOWN_PATH, index=False, encoding="utf-8-sig")

    stress_rows = []
    for model, part in summary.groupby("Model"):
        for tile, tile_part in part.groupby("Tile"):
            stress_rows.append(
                {
                    "Model": model,
                    "Tile": tile,
                    "Mean Skill_MSE": float(tile_part["Skill_MSE mean"].mean()),
                    "Worst RMSE/persistence": float((tile_part["RMSE [mm] mean"] / tile_part["Persistence RMSE [mm] mean"]).max()),
                }
            )
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(EXTERNAL_STRESS_PATH, index=False, encoding="utf-8-sig")
    return {"manifest": manifest_df, "summary": summary, "stress": stress_df}


def resolution_metric_rows(model_id: str, tile: str, resolution: int, seed: int, arrays: dict[str, np.ndarray], pred: np.ndarray, elapsed: float, infer: float, best_epoch: int, param_count: int) -> pd.DataFrame:
    df = rows_from_prediction(model_id, tile, seed, PHASE_RESOLUTION, arrays, pred, elapsed, infer, best_epoch, param_count)
    df.insert(2, "Resolution", resolution)
    return df


def run_resolution_experiment(main_bundles: dict[str, TileBundle]) -> dict[str, Any]:
    data_check_rows = []
    quick_rows: list[pd.DataFrame] = []
    hotspot_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for tile in RESOLUTION_TILES:
        bundle = main_bundles[tile]
        for resolution in RESOLUTIONS:
            if resolution == 512:
                attach_resolution(bundle, 512)
            tensors = get_tensors(bundle, resolution)
            data_check_rows.append(
                {
                    "Tile": tile,
                    "Resolution": resolution,
                    "Disp shape": "x".join(str(v) for v in tensors["disp"].shape),
                    "Valid ratio": float(tensors["valid_mask"].mean()),
                    "Hotspot pixels": int(hotspot_mask_from_train(tensors).sum()),
                }
            )

            run = train_eval_dynamic_m1(bundle, resolution, RESOLUTION_SEED, RESOLUTION_ROOT / "runs" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}", reuse_if_exists=True)
            m1_arrays = run["arrays"]
            val_run = infer_m1_split(bundle, RESOLUTION_ROOT / "runs" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}", "val") if resolution == 64 else None
            if resolution != 64:
                val_dataset = OnTheFlyActDataset(bundle, resolution, "val", resolution_config(resolution, RESOLUTION_SEED))
                val_loader = DataLoader(val_dataset, batch_size=resolution_config(resolution, RESOLUTION_SEED).batch_size, shuffle=False, num_workers=0)
                payload = torch.load(RESOLUTION_ROOT / "runs" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}" / "model.pth", map_location="cpu")
                model = build_m1_model(resolution_config(resolution, RESOLUTION_SEED), image_size=resolution, num_channels=4)
                model.load_state_dict(payload["state_dict"])
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                val_run, _ = collect_predictions_dynamic(model, val_loader, device)
                del model
                sync_cuda(device)
                gc.collect()

            assert val_run is not None
            val_cache = baseline_cache_from_m1_arrays(bundle, resolution, val_run, "val")
            test_cache = baseline_cache_from_m1_arrays(bundle, resolution, m1_arrays, "test")
            proxy_pred, proxy_weights = static_proxy_prediction_3expert(val_cache, test_cache)
            _ = proxy_weights

            param_count = int(run["result"]["param_count"])
            elapsed = float(run["result"]["elapsed_seconds"])
            infer = float(run["result"]["infer_seconds"])
            peak_mb = float(run["result"]["peak_gpu_memory_mb"])
            quick_rows.append(resolution_metric_rows("Persistence", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["persistence_mm"], 0.0, 0.0, -1, 0))
            quick_rows.append(resolution_metric_rows("Linear-Seasonal", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["linear_seasonal_mm"], 0.0, 0.0, -1, 0))
            quick_rows.append(resolution_metric_rows("M1", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["m1_pred_mm"], elapsed, infer, int(run["result"]["best_epoch"]), param_count))
            quick_rows.append(resolution_metric_rows("StaticVal-PLM1", tile, resolution, RESOLUTION_SEED, test_cache, proxy_pred, 0.0, 0.0, -1, 0))
            runtime_rows.append(
                {
                    "Tile": tile,
                    "Resolution": resolution,
                    "Model": "M1",
                    "Elapsed seconds": elapsed,
                    "Infer seconds": infer,
                    "Peak GPU memory [MB]": peak_mb,
                    "Param count": param_count,
                }
            )
            hotspot_mask = hotspot_mask_from_train(tensors)
            for model_id, pred in [
                ("Persistence", test_cache["persistence_mm"]),
                ("Linear-Seasonal", test_cache["linear_seasonal_mm"]),
                ("M1", test_cache["m1_pred_mm"]),
                ("StaticVal-PLM1", proxy_pred),
            ]:
                hotspot_metrics = compute_metrics(test_cache["y_true_mm"], pred, hotspot_mask.astype(np.float32))
                hotspot_rows.append(
                    {
                        "Tile": tile,
                        "Resolution": resolution,
                        "Model": model_id,
                        "Hotspot RMSE [mm]": hotspot_metrics["RMSE [mm]"],
                        "Hotspot MAE [mm]": hotspot_metrics["MAE [mm]"],
                        "Hotspot SSIM": hotspot_metrics["SSIM"],
                        "Hotspot Skill_MSE": skill_mse_for_mask(test_cache["y_true_mm"], pred, test_cache["persistence_mm"], hotspot_mask.astype(np.float32)),
                    }
                )
                boundary_rows.append(
                    {
                        "Tile": tile,
                        "Resolution": resolution,
                        "Model": model_id,
                        "Gradient RMSE": gradient_rmse(test_cache["y_true_mm"], pred, test_cache["valid_mask"]),
                    }
                )
            if resolution == 512:
                bundle.extra_tensors.pop(512, None)
                gc.collect()

    pd.DataFrame(data_check_rows).to_csv(RESOLUTION_DATA_CHECK_PATH, index=False, encoding="utf-8-sig")
    quick_df = pd.concat(quick_rows, ignore_index=True)
    quick_df.to_csv(RESOLUTION_QUICK_PATH, index=False, encoding="utf-8-sig")
    quick_summary = summarize_rows_by_keys(quick_df, ["Model", "Tile", "Resolution", "Horizon"])
    quick_summary.to_csv(RESOLUTION_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    hotspot_df = pd.DataFrame(hotspot_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    hotspot_df.to_csv(HOTSPOT_METRICS_PATH, index=False, encoding="utf-8-sig")
    boundary_df.to_csv(BOUNDARY_METRICS_PATH, index=False, encoding="utf-8-sig")
    runtime_df.to_csv(MEMORY_RUNTIME_PATH, index=False, encoding="utf-8-sig")

    three_seed_summary = pd.DataFrame()
    m1_quick = quick_summary[quick_summary["Model"] == "M1"]
    skill_spread = float(m1_quick.groupby("Resolution")["Skill_MSE mean"].mean().max() - m1_quick.groupby("Resolution")["Skill_MSE mean"].mean().min()) if not m1_quick.empty else 0.0
    hotspot_spread = float(hotspot_df[hotspot_df["Model"] == "M1"].groupby("Resolution")["Hotspot Skill_MSE"].mean().max() - hotspot_df[hotspot_df["Model"] == "M1"].groupby("Resolution")["Hotspot Skill_MSE"].mean().min()) if not hotspot_df.empty else 0.0
    trigger_three_seed = skill_spread >= 0.01 or hotspot_spread >= 0.01
    if trigger_three_seed:
        full_rows: list[pd.DataFrame] = []
        for tile in RESOLUTION_TILES:
            bundle = main_bundles[tile]
            for resolution in RESOLUTIONS:
                if resolution == 512:
                    attach_resolution(bundle, 512)
                for seed in FINAL_SEEDS:
                    run = train_eval_dynamic_m1(bundle, resolution, seed, RESOLUTION_ROOT / "runs_three_seed" / tile / f"res_{resolution}" / f"seed_{seed}", reuse_if_exists=True)
                    arrays = run["arrays"]
                    full_rows.append(
                        resolution_metric_rows("M1", tile, resolution, seed, arrays, arrays["y_pred_mm"], float(run["result"]["elapsed_seconds"]), float(run["result"]["infer_seconds"]), int(run["result"]["best_epoch"]), int(run["result"]["param_count"]))
                    )
                if resolution == 512:
                    bundle.extra_tensors.pop(512, None)
                    gc.collect()
        three_seed_summary = summarize_rows_by_keys(pd.concat(full_rows, ignore_index=True), ["Model", "Tile", "Resolution", "Horizon"])
        three_seed_summary.to_csv(RESOLUTION_M1_3SEED_PATH, index=False, encoding="utf-8-sig")

    return {
        "quick_summary": quick_summary,
        "hotspot": hotspot_df,
        "boundary": boundary_df,
        "runtime": runtime_df,
        "three_seed_summary": three_seed_summary,
    }


def write_decision(external_outputs: dict[str, Any], resolution_outputs: dict[str, Any]) -> dict[str, Any]:
    external_summary = external_outputs["summary"]
    resolution_summary = resolution_outputs["quick_summary"]
    vcea = external_summary[external_summary["Model"] == MAINLINE_LABEL]
    m1_ext = external_summary[external_summary["Model"] == "M1"]
    vcea_gain_vs_m1 = float(vcea["Skill_MSE mean"].mean() - m1_ext["Skill_MSE mean"].mean()) if not vcea.empty and not m1_ext.empty else float("nan")
    vcea_stress = float((vcea["RMSE [mm] mean"] / vcea["Persistence RMSE [mm] mean"]).max()) if not vcea.empty else float("nan")
    m1_res = resolution_summary[resolution_summary["Model"] == "M1"].copy()
    res64 = float(m1_res[m1_res["Resolution"] == 64]["Skill_MSE mean"].mean()) if not m1_res.empty else float("nan")
    res128 = float(m1_res[m1_res["Resolution"] == 128]["Skill_MSE mean"].mean()) if not m1_res.empty else float("nan")
    res512 = float(m1_res[m1_res["Resolution"] == 512]["Skill_MSE mean"].mean()) if not m1_res.empty else float("nan")
    verdict = {
        "external_tiles": int(external_outputs["manifest"].shape[0]),
        "external_vcea_mean_skill_mse": float(vcea["Skill_MSE mean"].mean()) if not vcea.empty else float("nan"),
        "external_vcea_gain_vs_m1": vcea_gain_vs_m1,
        "external_vcea_stress_ratio": vcea_stress,
        "resolution_m1_skill_64": res64,
        "resolution_m1_skill_128": res128,
        "resolution_m1_skill_512": res512,
        "dscf_mainline_status": "no_go_after_quick_return_to_vcea",
    }
    lines = [
        "# Confirmation Decision",
        "",
        "数据声明：本轮确认实验继续统一使用 **EGMS 2019-2023 Ortho U**。新增 external tiles 同样来自 `2019-2023 Ortho U`，原因是当前数据可得性限制。",
        "",
        "## Protocol freeze",
        "",
        "- DSCF main-six 已经判定 `no_go_after_quick_return_to_vcea`，因此本轮确认实验不用于拯救 DSCF。",
        "- External validation 作为冻结后 robustness 检查执行，主线模型为 `VCEA`，并保留 `M1/C3_opt/Persistence/Linear-Seasonal` 作为对照。",
        "- Resolution sensitivity 扩展为 `64/128/512`。其中 `128/512` 运行的是 exact `M1` 与 `StaticVal-PLM1 proxy`；没有把高分辨率 `C3_opt` retrain 引入为新搜索。",
        "",
        "## External summary",
        "",
        f"- external VCEA mean Skill_MSE = {verdict['external_vcea_mean_skill_mse']:.4f}。",
        f"- external VCEA gain vs M1 = {verdict['external_vcea_gain_vs_m1']:.4f}。",
        f"- external VCEA stress ratio = {verdict['external_vcea_stress_ratio']:.4f}。",
        "",
        "## Resolution summary",
        "",
        f"- M1 mean Skill_MSE @64 = {verdict['resolution_m1_skill_64']:.4f}。",
        f"- M1 mean Skill_MSE @128 = {verdict['resolution_m1_skill_128']:.4f}。",
        f"- M1 mean Skill_MSE @512 = {verdict['resolution_m1_skill_512']:.4f}。",
        "",
        "## Decision",
        "",
        "- 若 external 上 `VCEA` 维持正 gain 且 stress 未显著恶化，则 current mainline 的跨 tile robustness 得到补强。",
        "- 若 `128/512` 只改善 hotspot 而不改善 overall，应写成 regional screening + hotspot follow-up 边界，而不是推翻 64×64 主线。",
        "- 若 `512` 没有明显带来稳定 gain，则可以更强地 defend 当前 coarse regional setting。",
    ]
    write_markdown(CONFIRM_DECISION_PATH, lines)
    write_json(SUMMARY_PATH, verdict)
    return verdict


def write_final_report(external_outputs: dict[str, Any], resolution_outputs: dict[str, Any], verdict: dict[str, Any]) -> None:
    external_summary = external_outputs["summary"]
    resolution_summary = resolution_outputs["quick_summary"]
    hotspot = resolution_outputs["hotspot"]
    runtime = resolution_outputs["runtime"]
    lines = [
        "# DSCF Confirmatory Experiment Package",
        "",
        "数据声明：本轮确认实验继续统一使用 **EGMS 2019-2023 Ortho U**。外部 tile 也统一来自 `2019-2023 Ortho U`，原因是数据获取限制。",
        "",
        "## 执行原则",
        "",
        "- 按 `DSCF_confirmatory_experiment_plan_tile_resolution` 执行，但遵守其中的 stop rule：由于 `DSCF` main-six 已 no-go，本轮确认实验不再用于救 `DSCF`，而是作为冻结主线后的 `VCEA/M1` robustness + resolution sensitivity 包。",
        "- 用户额外要求把分辨率实验从 `64/128` 扩大到 `64/128/512`，本轮已执行。",
        "- 由于 exact 四专家 `VCEA` 在 `128/512` 需要高分辨率 `C3_opt` retrain，会把 sensitivity 变成新搜索；因此高分辨率 aggregation 采用了 `StaticVal-PLM1 proxy`。这是一条显式假设，已冻结到 `freeze_manifest.yaml`。",
        "",
        "## External tile validation",
        "",
        f"- external tiles 数量 = {external_outputs['manifest'].shape[0]}。",
        f"- 主结论：external `VCEA` mean Skill_MSE = {verdict['external_vcea_mean_skill_mse']:.4f}，gain vs M1 = {verdict['external_vcea_gain_vs_m1']:.4f}，stress ratio = {verdict['external_vcea_stress_ratio']:.4f}。",
        "- 解释：这一步用于确认 current aggregation mainline 是否能从 original six tiles 外推到锁定 external set，而不是重新选择模型。",
        "",
        "## Resolution sensitivity",
        "",
        f"- M1 mean Skill_MSE @64 = {verdict['resolution_m1_skill_64']:.4f}。",
        f"- M1 mean Skill_MSE @128 = {verdict['resolution_m1_skill_128']:.4f}。",
        f"- M1 mean Skill_MSE @512 = {verdict['resolution_m1_skill_512']:.4f}。",
        "",
        "### Hotspot / boundary / runtime",
        "",
        f"- hotspot metrics 文件：`{HOTSPOT_METRICS_PATH.as_posix()}`。",
        f"- boundary gradient 文件：`{BOUNDARY_METRICS_PATH.as_posix()}`。",
        f"- runtime / memory 文件：`{MEMORY_RUNTIME_PATH.as_posix()}`。",
        "",
        "## 主要输出",
        "",
        f"- external manifest: `{EXTERNAL_MANIFEST_PATH.as_posix()}`",
        f"- freeze manifest: `{FREEZE_MANIFEST_PATH.as_posix()}`",
        f"- external summary: `{EXTERNAL_SUMMARY_PATH.as_posix()}`",
        f"- external regime breakdown: `{EXTERNAL_REGIME_BREAKDOWN_PATH.as_posix()}`",
        f"- resolution summary: `{RESOLUTION_SUMMARY_PATH.as_posix()}`",
        f"- confirmation decision: `{CONFIRM_DECISION_PATH.as_posix()}`",
        "",
        "## 备注",
        "",
        "- 若 PowerShell 里看到中文乱码，那是终端编码问题；文件本身按 UTF-8 写出。",
    ]
    if not external_summary.empty:
        top = external_summary.groupby("Model", as_index=False)["Skill_MSE mean"].mean().sort_values("Skill_MSE mean", ascending=False)
        lines.extend(["", "## External model ranking", ""])
        for _, row in top.iterrows():
            lines.append(f"- {row['Model']}: mean Skill_MSE = {row['Skill_MSE mean']:.4f}")
    if not resolution_summary.empty:
        lines.extend(["", "## Resolution quick ranking", ""])
        rank = resolution_summary.groupby(["Model", "Resolution"], as_index=False)["Skill_MSE mean"].mean().sort_values(["Model", "Resolution"])
        for _, row in rank.iterrows():
            lines.append(f"- {row['Model']} @ {int(row['Resolution'])}: mean Skill_MSE = {row['Skill_MSE mean']:.4f}")
    if not hotspot.empty:
        best_hotspot = hotspot.groupby(["Model", "Resolution"], as_index=False)["Hotspot Skill_MSE"].mean().sort_values("Hotspot Skill_MSE", ascending=False).head(6)
        lines.extend(["", "## Hotspot signal", ""])
        for _, row in best_hotspot.iterrows():
            lines.append(f"- {row['Model']} @ {int(row['Resolution'])}: hotspot Skill_MSE = {row['Hotspot Skill_MSE']:.4f}")
    if not runtime.empty:
        lines.extend(["", "## Runtime snapshot", ""])
        rt = runtime.groupby("Resolution", as_index=False)[["Elapsed seconds", "Infer seconds", "Peak GPU memory [MB]"]].mean().sort_values("Resolution")
        for _, row in rt.iterrows():
            lines.append(
                f"- M1 @ {int(row['Resolution'])}: train={row['Elapsed seconds']:.2f}s, infer={row['Infer seconds']:.2f}s, peak_mem={row['Peak GPU memory [MB]']:.1f}MB"
            )
    write_markdown(FINAL_REPORT_PATH, lines)


def run_confirmation_experiments() -> dict[str, Any]:
    build_freeze_manifest()
    main_bundles = preprocess_all_tiles()
    external_outputs = run_external_experiment()
    resolution_outputs = run_resolution_experiment(main_bundles)
    make_confirmation_plots(external_outputs["summary"], resolution_outputs["quick_summary"])
    verdict = write_decision(external_outputs, resolution_outputs)
    write_final_report(external_outputs, resolution_outputs, verdict)
    return {"external": external_outputs, "resolution": resolution_outputs, "verdict": verdict}
