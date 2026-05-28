from __future__ import annotations

import gc
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .act_v2 import ActV2Config, build_model as build_act_model, train_eval_single_tile
from .act_v4c_c3_candidate import c3_runtime_optimized_config
from .act_v3 import make_lag_delta, rolling_mad
from .act_v4c_efficiency import V4CRescueConfig, build_rescue_model, loader_kwargs, prepare_rescue_datasets, train_eval_rescue_single_tile
from .act_v5_selector import infer_c3_split
from .act_v7 import COEAConfig, replay_coea, warm_start_weights
from .dscf_confirmation import (
    RESOLUTION_SEED,
    RESOLUTION_TILES,
    RESOLUTIONS,
    OnTheFlyActDataset,
    attach_resolution,
    baseline_cache_from_m1_arrays,
    collect_predictions_dynamic,
    external_static_prediction_4expert,
    gradient_rmse,
    hotspot_mask_from_train,
    resolution_config,
    resolution_metric_rows,
    summarize_rows_by_keys,
    sync_cuda,
    train_eval_dynamic_m1,
)
from .pipeline import OUTPUT_ROOT, TileBundle, ensure_dir, get_tensors, preprocess_all_tiles, write_json


EXT_ROOT = ensure_dir(OUTPUT_ROOT / "confirmation" / "model_family_resolution_extension")
RUN_ROOT = ensure_dir(EXT_ROOT / "runs")
SUMMARY_PATH = EXT_ROOT / "family_resolution_summary.csv"
QUICK_ROWS_PATH = EXT_ROOT / "family_resolution_quick_rows.csv"
RUNTIME_PATH = EXT_ROOT / "family_resolution_runtime.csv"
HOTSPOT_PATH = EXT_ROOT / "family_resolution_hotspot.csv"
BOUNDARY_PATH = EXT_ROOT / "family_resolution_boundary.csv"
RANKING_PATH = EXT_ROOT / "family_resolution_ranking.csv"
REPORT_PATH = EXT_ROOT / "family_resolution_report.md"
DECISION_PATH = EXT_ROOT / "family_resolution_decision.md"
VERDICT_PATH = EXT_ROOT / "family_resolution_verdict.json"
WEIGHTS_PATH = EXT_ROOT / "aggregation_weight_summary.csv"

COEA_RESOLUTION_CONFIG = COEAConfig(
    model_id="COEA_exact",
    update_mode="ewa",
    eta=4.0,
    warm_start=True,
    m1_floor=0.5,
    granularity="tile_horizon",
)


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def count_state_dict_params(path: Path) -> int:
    payload = torch.load(path, map_location="cpu")
    return int(sum(int(v.numel()) for v in payload["state_dict"].values()))


def m0_resolution_config(resolution: int, seed: int) -> ActV2Config:
    patch_size = {64: 8, 128: 16, 512: 64}[resolution]
    batch_size = {64: 8, 128: 2, 512: 1}[resolution]
    epochs = {64: 12, 128: 10, 512: 6}[resolution]
    patience = {64: 4, 128: 3, 512: 2}[resolution]
    return ActV2Config(
        model_id=f"M0_res{resolution}",
        gate_mode="none",
        patch_size=patch_size,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        seed=seed,
        num_workers=0,
        use_amp=True,
        query_enabled=True,
        residual_enabled=True,
    )


def c3_resolution_config(resolution: int, seed: int):
    base = c3_runtime_optimized_config()
    batch_size = {64: 8, 128: 2, 512: 1}[resolution]
    epochs = {64: 16, 128: 10, 512: 5}[resolution]
    patience = {64: 4, 128: 3, 512: 2}[resolution]
    if resolution == 512:
        # Keep the 512 attempt executable: same family, but a compute-capped lite head.
        epochs = 2
        patience = 1
        dim = 24
        scale_dims = [8, 8, 8]
    else:
        dim = base.dim
        scale_dims = base.scale_dims
    return replace(
        base,
        model_id=f"C3_opt_res{resolution}",
        resolution=resolution,
        dim=dim,
        scale_dims=scale_dims,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        seed=seed,
        num_workers=0,
        pin_memory=False,
        compile_model=False,
    )


def infer_dynamic_m1_split(bundle: TileBundle, resolution: int, run_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    payload = torch.load(run_dir / "model.pth", map_location="cpu")
    config = ActV2Config(**payload["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OnTheFlyActDataset(bundle, resolution, split_name, config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_act_model(config, image_size=resolution, num_channels=4).to(device)
    model.load_state_dict(payload["state_dict"])
    arrays, _infer_seconds = collect_predictions_dynamic(model, loader, device)
    del model
    sync_cuda(device)
    gc.collect()
    return arrays


class OnTheFlyC3TransferDataset(torch.utils.data.Dataset):
    def __init__(self, bundle: TileBundle, resolution: int, split_name: str, config: V4CRescueConfig) -> None:
        self.bundle = bundle
        self.config = config
        tensors = get_tensors(bundle, resolution)
        self.disp = tensors["disp"].astype(np.float32)
        train_end = bundle.split.train_frame_end
        self.disp_mu = self.disp[:train_end].mean(axis=0).astype(np.float32)
        self.disp_sigma = self.disp[:train_end].std(axis=0).astype(np.float32) + 1e-6
        self.delta1 = make_lag_delta(self.disp, 1)
        self.delta3 = make_lag_delta(self.disp, 3)
        self.volatility = rolling_mad(self.delta1, window=5)
        self.delta1_mu = self.delta1[:train_end].mean(axis=0).astype(np.float32)
        self.delta1_sigma = self.delta1[:train_end].std(axis=0).astype(np.float32) + 1e-6
        self.delta3_mu = self.delta3[:train_end].mean(axis=0).astype(np.float32)
        self.delta3_sigma = self.delta3[:train_end].std(axis=0).astype(np.float32) + 1e-6
        self.vol_mu = self.volatility[:train_end].mean(axis=0).astype(np.float32)
        self.vol_sigma = self.volatility[:train_end].std(axis=0).astype(np.float32) + 1e-6

        dates = pd.to_datetime(bundle.dates)
        self.time_feat = np.stack(
            [
                np.sin(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
                np.cos(2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25),
            ],
            axis=1,
        ).astype(np.float32)

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
        disp_norm = ((self.disp[start : current + 1] - self.disp_mu[None, :, :]) / self.disp_sigma[None, :, :]).astype(np.float32)
        delta1_norm = ((self.delta1[start : current + 1] - self.delta1_mu[None, :, :]) / self.delta1_sigma[None, :, :]).astype(np.float32)
        delta3_norm = ((self.delta3[start : current + 1] - self.delta3_mu[None, :, :]) / self.delta3_sigma[None, :, :]).astype(np.float32)
        vol_norm = ((self.volatility[start : current + 1] - self.vol_mu[None, :, :]) / self.vol_sigma[None, :, :]).astype(np.float32)
        h, w = self.disp.shape[1], self.disp.shape[2]
        time_expand = np.tile(self.time_feat[start : current + 1, :, None, None], (1, 1, h, w)).astype(np.float32)
        x_v4 = np.concatenate(
            [
                disp_norm[:, None, :, :],
                delta1_norm[:, None, :, :],
                delta3_norm[:, None, :, :],
                vol_norm[:, None, :, :],
                time_expand,
            ],
            axis=1,
        ).astype(np.float32)
        target_indices = [current + h_step for h_step in self.config.horizons]
        last_disp = self.disp[current][None, :, :].astype(np.float32)
        current_delta5 = np.abs(self.disp[current] - self.disp[max(0, current - 5)]) / 5.0
        step_scale = np.maximum.reduce(
            [
                self.volatility[current],
                np.abs(self.delta1[current]),
                np.abs(self.delta3[current]) / 3.0,
                current_delta5.astype(np.float32),
                np.full_like(self.volatility[current], 0.15, dtype=np.float32),
            ]
        ).astype(np.float32)
        scale_mm = []
        for horizon in self.config.horizons:
            horizon_factor = max(1.0, self.config.horizon_scale_alpha * float(horizon))
            scale_mm.append(step_scale * horizon_factor + 0.05)
        scale_mm = np.stack(scale_mm, axis=0)[:, None, :, :].astype(np.float32)
        future_t = self.time_feat[target_indices].astype(np.float32)
        return {
            "x_v4": torch.from_numpy(x_v4).float(),
            "future_time": torch.from_numpy(future_t).float(),
            "scale_mm": torch.from_numpy(scale_mm).float(),
            "last_disp_mm": torch.from_numpy(last_disp).float(),
            "current_frame": torch.tensor(current, dtype=torch.long),
            "target_indices": torch.tensor(target_indices, dtype=torch.long),
        }


def infer_c3_split_runtime(bundle: TileBundle, run_dir: Path, split_name: str, resolution_override: int | None = None) -> tuple[dict[str, np.ndarray], float]:
    payload = torch.load(run_dir / "model.pth", map_location="cpu")
    config = V4CRescueConfig(**payload["config"])
    if resolution_override is not None:
        config = replace(config, resolution=resolution_override, batch_size=1, num_workers=0, pin_memory=False, compile_model=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resolution_override is not None and resolution_override >= 512:
        dataset = OnTheFlyC3TransferDataset(bundle, resolution_override, split_name, config)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        train_ds, val_ds, test_ds = prepare_rescue_datasets(bundle, config)
        mapping = {"train": train_ds, "val": val_ds, "test": test_ds}
        dataset = mapping[split_name]
        loader = DataLoader(dataset, **loader_kwargs(config, shuffle=False))
    model = build_rescue_model(config, in_channels=6).to(device)
    model.load_state_dict(payload["state_dict"])
    start = time.time()
    arrays = collect_predictions_incremental_c3(model, loader, device, use_fp16_storage=(config.resolution >= 512))
    elapsed = time.time() - start
    del model
    sync_cuda(device)
    gc.collect()
    return arrays, elapsed


def collect_predictions_incremental_c3(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16_storage: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    ds = loader.dataset
    sample0 = ds[0]
    n_samples = len(ds)
    if "target_mm" in sample0:
        n_horizons = int(sample0["target_mm"].shape[0])
        height = int(sample0["target_mm"].shape[-2])
        width = int(sample0["target_mm"].shape[-1])
    else:
        n_horizons = int(sample0["scale_mm"].shape[0])
        height = int(sample0["scale_mm"].shape[-2])
        width = int(sample0["scale_mm"].shape[-1])
    pred_dtype = np.float16 if use_fp16_storage else np.float32
    y_pred = np.empty((n_samples, n_horizons, height, width), dtype=pred_dtype)
    scale_mm = np.empty((n_samples, n_horizons, height, width), dtype=pred_dtype)
    current_frames = np.empty((n_samples,), dtype=np.int64)
    target_indices = np.empty((n_samples, n_horizons), dtype=np.int64)
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            pred, aux = model(batch, return_aux=True)
            bsz = int(pred.shape[0])
            pred_np = pred.detach().cpu().numpy()[:, :, 0]
            scale_np = aux["scale_mm"].detach().cpu().numpy()[:, :, 0]
            y_pred[cursor : cursor + bsz] = pred_np.astype(pred_dtype, copy=False)
            scale_mm[cursor : cursor + bsz] = scale_np.astype(pred_dtype, copy=False)
            current_frames[cursor : cursor + bsz] = batch["current_frame"].detach().cpu().numpy().reshape(-1)
            target_indices[cursor : cursor + bsz] = batch["target_indices"].detach().cpu().numpy()
            cursor += bsz
    return {
        "y_pred_mm": y_pred,
        "scale_mm": scale_mm,
        "current_frames": current_frames,
        "target_indices": target_indices,
    }


def build_exact_cache(
    bundle: TileBundle,
    resolution: int,
    m1_arrays: dict[str, np.ndarray],
    c3_arrays: dict[str, np.ndarray],
    split_name: str,
) -> dict[str, np.ndarray]:
    cache = baseline_cache_from_m1_arrays(bundle, resolution, m1_arrays, split_name)
    if not np.array_equal(cache["current_frames"], c3_arrays["current_frames"].reshape(-1)):
        raise ValueError(f"Frame mismatch for tile={bundle.tile_id} resolution={resolution} split={split_name}")
    if not np.array_equal(cache["target_indices"], c3_arrays["target_indices"]):
        raise ValueError(f"Target mismatch for tile={bundle.tile_id} resolution={resolution} split={split_name}")
    tensors = get_tensors(bundle, resolution)
    cache["c3_pred_mm"] = c3_arrays["y_pred_mm"]
    cache["scale_mm"] = c3_arrays["scale_mm"]
    cache["residual_std"] = tensors["residual_std"].astype(np.float32)
    cache["descriptor_amp"] = tensors["descriptor_amp"].astype(np.float32)
    cache["trend_r2"] = tensors["trend_r2"].astype(np.float32)
    return cache


def attach_c3_to_base_cache(
    bundle: TileBundle,
    resolution: int,
    base_cache: dict[str, np.ndarray],
    c3_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cache = {key: value for key, value in base_cache.items()}
    if not np.array_equal(cache["current_frames"], c3_arrays["current_frames"].reshape(-1)):
        raise ValueError(f"Frame mismatch for tile={bundle.tile_id} resolution={resolution}")
    if not np.array_equal(cache["target_indices"], c3_arrays["target_indices"]):
        raise ValueError(f"Target mismatch for tile={bundle.tile_id} resolution={resolution}")
    tensors = get_tensors(bundle, resolution)
    cache["c3_pred_mm"] = c3_arrays["y_pred_mm"].astype(np.float32)
    cache["scale_mm"] = c3_arrays["scale_mm"].astype(np.float32)
    cache["residual_std"] = tensors["residual_std"].astype(np.float32)
    cache["descriptor_amp"] = tensors["descriptor_amp"].astype(np.float32)
    cache["trend_r2"] = tensors["trend_r2"].astype(np.float32)
    return cache


def run_exact_vcea(val_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.time()
    pred, weights = external_static_prediction_4expert(val_cache, test_cache)
    elapsed = time.time() - start
    return pred.astype(np.float32), weights.astype(np.float32), elapsed


def run_exact_coea(val_cache: dict[str, np.ndarray], test_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.time()
    init_weights = warm_start_weights(val_cache, COEA_RESOLUTION_CONFIG)
    pred, weight_snapshots, _ = replay_coea(test_cache, COEA_RESOLUTION_CONFIG, init_weights=init_weights, return_trace=False)
    elapsed = time.time() - start
    return pred.astype(np.float32), weight_snapshots.astype(np.float32), elapsed


def summarize_weight_rows(weight_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(weight_rows)
    if df.empty:
        return df
    return (
        df.groupby(["Model", "Tile", "Resolution", "Horizon", "Expert"], as_index=False)["Weight"]
        .mean()
        .sort_values(["Model", "Resolution", "Tile", "Horizon", "Expert"])
    )


def model_level_verdict(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, part in summary_df.groupby("Model"):
        by_res = (
            part.groupby("Resolution", as_index=False)
            .agg(
                Mean_Skill_MSE=("Skill_MSE mean", "mean"),
                Worst_Stress_Ratio=("RMSE [mm] mean", lambda s: float((s / part.loc[s.index, "Persistence RMSE [mm] mean"]).max())),
            )
        )
        for _, row in by_res.iterrows():
            resolution = int(row["Resolution"])
            res_part = part[part["Resolution"] == resolution]
            rows.append(
                {
                    "Model": model,
                    "Resolution": resolution,
                    "Mean Skill_MSE": float(res_part["Skill_MSE mean"].mean()),
                    "Mean RMSE [mm]": float(res_part["RMSE [mm] mean"].mean()),
                    "Stress ratio": float((res_part["RMSE [mm] mean"] / res_part["Persistence RMSE [mm] mean"]).max()),
                    "Better tiles vs M1": int(
                        (
                            res_part.groupby("Tile")["Skill_MSE mean"].mean()
                            > summary_df[(summary_df["Model"] == "M1") & (summary_df["Resolution"] == resolution)]
                            .groupby("Tile")["Skill_MSE mean"]
                            .mean()
                        ).sum()
                    )
                    if model != "M1"
                    else int(res_part["Tile"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["Resolution", "Mean Skill_MSE"], ascending=[True, False])


def run_family_resolution_extension() -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    exact_val_cache_128: dict[str, dict[str, np.ndarray]] = {}
    quick_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for tile in RESOLUTION_TILES:
        bundle = bundles[tile]
        for resolution in RESOLUTIONS:
            if resolution == 512:
                attach_resolution(bundle, 512)

            tensors = get_tensors(bundle, resolution)
            hotspot_mask = hotspot_mask_from_train(tensors).astype(np.float32)

            m1_run_dir = RUN_ROOT / "M1" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}"
            m1_run = train_eval_dynamic_m1(bundle, resolution, RESOLUTION_SEED, m1_run_dir, reuse_if_exists=True)
            m1_test = m1_run["arrays"]
            m1_val = infer_dynamic_m1_split(bundle, resolution, m1_run_dir, "val")
            m1_elapsed = float(m1_run["result"]["elapsed_seconds"])
            m1_infer = float(m1_run["result"]["infer_seconds"])
            m1_best_epoch = int(m1_run["result"]["best_epoch"])
            m1_param_count = int(m1_run["result"]["param_count"])
            val_base = baseline_cache_from_m1_arrays(bundle, resolution, m1_val, "val")
            test_base = baseline_cache_from_m1_arrays(bundle, resolution, m1_test, "test")
            del m1_val, m1_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            m0_cfg = m0_resolution_config(resolution, RESOLUTION_SEED)
            m0_run_dir = RUN_ROOT / "M0" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}"
            m0_run = train_eval_single_tile(bundle, m0_cfg, m0_run_dir, phase="family_resolution_extension", resolution=resolution, reuse_if_exists=True)
            m0_pred = m0_run["arrays"]["y_pred_mm"].astype(np.float16 if resolution == 512 else np.float32)
            del m0_run["arrays"]
            gc.collect()

            if resolution == 512:
                c3_source_dir = RUN_ROOT / "C3_opt" / tile / "res_128" / f"seed_{RESOLUTION_SEED}"
                c3_test, c3_test_infer = infer_c3_split_runtime(bundle, c3_source_dir, "test", resolution_override=512)
                c3_elapsed = c3_test_infer
                c3_param_count = count_state_dict_params(c3_source_dir / "model.pth")
                c3_best_epoch = -1
            else:
                c3_cfg = c3_resolution_config(resolution, RESOLUTION_SEED)
                c3_run_dir = RUN_ROOT / "C3_opt" / tile / f"res_{resolution}" / f"seed_{RESOLUTION_SEED}"
                c3_run = train_eval_rescue_single_tile(bundle, c3_cfg, c3_run_dir, phase="family_resolution_extension", reuse_if_exists=True)
                c3_test = c3_run["arrays"]
                c3_val, _ = infer_c3_split_runtime(bundle, c3_run_dir, "val")
                c3_elapsed = float(c3_run["result"]["elapsed_seconds"])
                c3_test_infer = float(c3_run["result"]["infer_seconds"])
                c3_param_count = int(c3_run["result"]["param_count"])
                c3_best_epoch = int(c3_run["result"]["best_epoch"])

            if resolution == 512:
                if tile not in exact_val_cache_128:
                    raise KeyError(f"Missing 128 validation cache for tile={tile}")
                val_cache = exact_val_cache_128[tile]
            else:
                val_cache = attach_c3_to_base_cache(bundle, resolution, val_base, c3_val)
                if resolution == 128:
                    exact_val_cache_128[tile] = val_cache
            test_cache = attach_c3_to_base_cache(bundle, resolution, test_base, c3_test)
            if resolution == 512:
                del val_base, test_base, c3_test
            else:
                del val_base, test_base, c3_val, c3_test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vcea_pred, vcea_weights, vcea_elapsed = run_exact_vcea(val_cache, test_cache)
            coea_pred, coea_weights, coea_elapsed = run_exact_coea(val_cache, test_cache)

            m0_param_count = count_state_dict_params(m0_run_dir / "model.pth")
            quick_rows.append(resolution_metric_rows("Persistence", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["persistence_mm"], 0.0, 0.0, -1, 0))
            quick_rows.append(resolution_metric_rows("Linear-Seasonal", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["linear_seasonal_mm"], 0.0, 0.0, -1, 0))
            quick_rows.append(resolution_metric_rows("M1", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["m1_pred_mm"], m1_elapsed, m1_infer, m1_best_epoch, m1_param_count))
            quick_rows.append(resolution_metric_rows("M0", tile, resolution, RESOLUTION_SEED, test_cache, m0_pred.astype(np.float32, copy=False), float(m0_run["result"]["elapsed_seconds"]), 0.0, int(m0_run["result"]["best_epoch"]), m0_param_count))
            quick_rows.append(resolution_metric_rows("C3_opt", tile, resolution, RESOLUTION_SEED, test_cache, test_cache["c3_pred_mm"], c3_elapsed, c3_test_infer, c3_best_epoch, c3_param_count))
            quick_rows.append(resolution_metric_rows("VCEA_exact", tile, resolution, RESOLUTION_SEED, test_cache, vcea_pred, vcea_elapsed, vcea_elapsed, -1, 0))
            quick_rows.append(resolution_metric_rows("COEA_exact", tile, resolution, RESOLUTION_SEED, test_cache, coea_pred, coea_elapsed, coea_elapsed, -1, 0))

            runtime_rows.extend(
                [
                    {"Tile": tile, "Resolution": resolution, "Model": "M1", "Elapsed seconds": float(m1_elapsed), "Infer seconds": float(m1_infer), "Peak GPU memory [MB]": float(m1_run["result"]["peak_gpu_memory_mb"]), "Param count": int(m1_param_count)},
                    {"Tile": tile, "Resolution": resolution, "Model": "M0", "Elapsed seconds": float(m0_run["result"]["elapsed_seconds"]), "Infer seconds": 0.0, "Peak GPU memory [MB]": float("nan"), "Param count": int(m0_param_count)},
                    {"Tile": tile, "Resolution": resolution, "Model": "C3_opt", "Elapsed seconds": float(c3_elapsed), "Infer seconds": float(c3_test_infer), "Peak GPU memory [MB]": float("nan"), "Param count": int(c3_param_count)},
                    {"Tile": tile, "Resolution": resolution, "Model": "VCEA_exact", "Elapsed seconds": float(vcea_elapsed), "Infer seconds": float(vcea_elapsed), "Peak GPU memory [MB]": float("nan"), "Param count": 0},
                    {"Tile": tile, "Resolution": resolution, "Model": "COEA_exact", "Elapsed seconds": float(coea_elapsed), "Infer seconds": float(coea_elapsed), "Peak GPU memory [MB]": float("nan"), "Param count": 0},
                ]
            )

            for model_id, pred in [
                ("Persistence", test_cache["persistence_mm"]),
                ("Linear-Seasonal", test_cache["linear_seasonal_mm"]),
                ("M1", test_cache["m1_pred_mm"]),
                ("M0", m0_pred.astype(np.float32, copy=False)),
                ("C3_opt", test_cache["c3_pred_mm"]),
                ("VCEA_exact", vcea_pred),
                ("COEA_exact", coea_pred),
            ]:
                hotspot_metrics = compute_metrics = {
                    "Skill_MSE": float(
                        1.0
                        - np.mean((pred[:, :, hotspot_mask > 0.5] - test_cache["y_true_mm"][:, :, hotspot_mask > 0.5]) ** 2)
                        / max(
                            float(
                                np.mean(
                                    (test_cache["persistence_mm"][:, :, hotspot_mask > 0.5] - test_cache["y_true_mm"][:, :, hotspot_mask > 0.5]) ** 2
                                )
                            ),
                            1e-6,
                        )
                    ),
                    "RMSE": float(np.sqrt(np.mean((pred[:, :, hotspot_mask > 0.5] - test_cache["y_true_mm"][:, :, hotspot_mask > 0.5]) ** 2))),
                    "MAE": float(np.mean(np.abs(pred[:, :, hotspot_mask > 0.5] - test_cache["y_true_mm"][:, :, hotspot_mask > 0.5]))),
                }
                hotspot_rows.append(
                    {
                        "Tile": tile,
                        "Resolution": resolution,
                        "Model": model_id,
                        "Hotspot Skill_MSE": hotspot_metrics["Skill_MSE"],
                        "Hotspot RMSE [mm]": hotspot_metrics["RMSE"],
                        "Hotspot MAE [mm]": hotspot_metrics["MAE"],
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

            for h_idx, horizon in enumerate([1, 3, 5, 10]):
                for exp_idx, expert_name in enumerate(["persistence", "linear_seasonal", "M1", "C3_opt"]):
                    weight_rows.append(
                        {
                            "Model": "VCEA_exact",
                            "Tile": tile,
                            "Resolution": resolution,
                            "Horizon": f"t+{horizon}",
                            "Expert": expert_name,
                            "Weight": float(vcea_weights[h_idx, exp_idx]),
                        }
                    )
                    weight_rows.append(
                        {
                            "Model": "COEA_exact",
                            "Tile": tile,
                            "Resolution": resolution,
                            "Horizon": f"t+{horizon}",
                            "Expert": expert_name,
                            "Weight": float(coea_weights[:, h_idx, exp_idx].mean()),
                        }
                    )

            if resolution == 512:
                del m0_pred
                bundle.extra_tensors.pop(512, None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    quick_df = pd.concat(quick_rows, ignore_index=True)
    summary_df = summarize_rows_by_keys(quick_df, ["Model", "Tile", "Resolution", "Horizon"])
    runtime_df = pd.DataFrame(runtime_rows)
    hotspot_df = pd.DataFrame(hotspot_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    ranking_df = model_level_verdict(summary_df)
    weight_df = summarize_weight_rows(weight_rows)

    quick_df.to_csv(QUICK_ROWS_PATH, index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    runtime_df.to_csv(RUNTIME_PATH, index=False, encoding="utf-8-sig")
    hotspot_df.to_csv(HOTSPOT_PATH, index=False, encoding="utf-8-sig")
    boundary_df.to_csv(BOUNDARY_PATH, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")
    weight_df.to_csv(WEIGHTS_PATH, index=False, encoding="utf-8-sig")

    verdict = {
        "models_tested_exact": ["M0", "M1", "C3_opt", "VCEA_exact", "COEA_exact"],
        "tiles": RESOLUTION_TILES,
        "resolutions": RESOLUTIONS,
        "best_model_by_resolution": {
            str(res): str(ranking_df[ranking_df["Resolution"] == res].iloc[0]["Model"]) for res in RESOLUTIONS
        },
    }
    write_json(VERDICT_PATH, verdict)
    return {
        "quick": quick_df,
        "summary": summary_df,
        "runtime": runtime_df,
        "hotspot": hotspot_df,
        "boundary": boundary_df,
        "ranking": ranking_df,
        "weights": weight_df,
        "verdict": verdict,
    }


def write_outputs(outputs: dict[str, Any]) -> None:
    ranking = outputs["ranking"]
    summary = outputs["summary"]
    hotspot = outputs["hotspot"]
    runtime = outputs["runtime"]

    lines = [
        "# Family Resolution Extension Report",
        "",
        "数据声明：本轮补充实验继续统一使用 **EGMS 2019-2023 Ortho U**，原因是数据获取限制。",
        "",
        "## 这轮补了什么",
        "",
        "- 之前 exact 分辨率测试只覆盖了 `M1`；`128/512` 上的聚合部分是 proxy，不是 exact 四专家版本。",
        "- 本轮补做了 `M0 / C3_opt / exact VCEA / exact COEA` 的 `64/128/512` 分辨率测试。",
        "- 评估 tile 继续使用 `E32N34 / E39N30 / E48N24`，保持与前一轮 resolution sensitivity 一致。",
        "- 额外假设：`C3_opt@512` 的 direct training 触发了 `bad allocation`，因此这里改成 `128→512` 的 transfer inference 尝试，并在结果里单独说明。",
        "- 额外假设：`512` 上的 `VCEA_exact/COEA_exact` 校准使用同 tile 的 `128` validation cache；`512` 端只做 test-time aggregation 评估。",
        "- runtime 说明：`VCEA_exact/COEA_exact` 的时间只统计 aggregation/calibration 本身，不包含生成 `M1/C3_opt` expert 预测的训练成本。",
        "",
        "## Model-level ranking",
        "",
    ]
    for resolution in RESOLUTIONS:
        part = ranking[ranking["Resolution"] == resolution]
        lines.append(f"### Resolution {resolution}")
        lines.append("")
        for _, row in part.iterrows():
            lines.append(
                f"- {row['Model']}: mean Skill_MSE = {row['Mean Skill_MSE']:.4f}, stress ratio = {row['Stress ratio']:.4f}, better tiles vs M1 = {int(row['Better tiles vs M1'])}"
            )
        lines.append("")

    lines.extend(
        [
            "## 关键观察",
            "",
            f"- 结果总表：`{SUMMARY_PATH.as_posix()}`",
            f"- ranking：`{RANKING_PATH.as_posix()}`",
            f"- hotspot：`{HOTSPOT_PATH.as_posix()}`",
            f"- runtime：`{RUNTIME_PATH.as_posix()}`",
            f"- aggregation weights：`{WEIGHTS_PATH.as_posix()}`",
            "",
            "## 说明",
            "",
            "- 这轮重点是 family-level resolution sensitivity，不是重新做 full-six 主模型选择。",
            "- `VCEA_exact/COEA_exact` 均基于同一套 high-resolution `M1 + C3_opt + persistence + linear-seasonal` expert cache 复算。",
        ]
    )
    write_markdown(REPORT_PATH, lines)

    best_64 = ranking[ranking["Resolution"] == 64].iloc[0]
    best_128 = ranking[ranking["Resolution"] == 128].iloc[0]
    best_512 = ranking[ranking["Resolution"] == 512].iloc[0]
    decision_lines = [
        "# Family Resolution Extension Decision",
        "",
        "数据声明：本轮补充实验继续统一使用 **EGMS 2019-2023 Ortho U**。",
        "",
        f"- 64×64 最优模型：`{best_64['Model']}`，mean Skill_MSE = {best_64['Mean Skill_MSE']:.4f}。",
        f"- 128×128 最优模型：`{best_128['Model']}`，mean Skill_MSE = {best_128['Mean Skill_MSE']:.4f}。",
        f"- 512×512 最优模型：`{best_512['Model']}`，mean Skill_MSE = {best_512['Mean Skill_MSE']:.4f}。",
        "",
        "- 这一步回答的是“除了 M1 以外，其他 V1-V7 代表模型在更高分辨率上会不会更好”。",
        "- 若高分辨率最优者不是 M1，需要结合 stress ratio、hotspot 和 runtime 一起解释，而不能只看 mean Skill_MSE。",
        "- 注意：`VCEA_exact/COEA_exact` 的 runtime 是聚合自身时间，不是 end-to-end expert generation 成本。",
    ]
    write_markdown(DECISION_PATH, decision_lines)


def main() -> None:
    outputs = run_family_resolution_extension()
    write_outputs(outputs)


if __name__ == "__main__":
    main()
