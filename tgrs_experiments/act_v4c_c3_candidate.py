from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader

from .act_v2 import (
    FINAL_SEEDS,
    SEARCH_HORIZONS,
    SEARCH_TILES,
    ActV2Config,
    ActV2Dataset,
    batch_to_device,
    build_model,
    compute_loss as compute_loss_m1,
    run_epoch as run_epoch_m1,
    train_eval_single_tile,
    write_simple_yaml,
)
from .act_v2_m1_suite import ABLATION_TILES, FULL_TILES, baseline_rows_for_tiles, summarize_baselines
from .act_v3 import compute_loss as compute_loss_v4c, summarize_seeded_rows
from .act_v4c_efficiency import (
    CONFIG_ROOT,
    V4C_RESCUE_ROOT,
    V4CRescueConfig,
    ablation_variants,
    batch_to_device_nb,
    build_rescue_model,
    default_rescue_configs,
    loader_kwargs,
    prepare_rescue_datasets,
    sync_cuda,
    train_eval_rescue_single_tile,
)
from .pipeline import TileBundle, ensure_dir, preprocess_all_tiles, write_json


CANDIDATE_ROOT = V4C_RESCUE_ROOT / "c3_candidate"
CANDIDATE_CONFIG_ROOT = CONFIG_ROOT / "c3_candidate"
TIMING_TILE = "E32N34"
M1_FULL6_SUMMARY_PATH = Path(
    "${PROJECT_ROOT}/outputs/tgrs_2019_2023/act_v2/m1_suite/step1b_scalar_summary.csv"
)
REPORT_PATH = CANDIDATE_ROOT / "act_v4c_c3_candidate_report.md"
SUMMARY_PATH = CANDIDATE_ROOT / "run_summary.json"

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
]
RESCUE_EXTRA_METRICS = ["Infer seconds", "Param count", "Peak GPU memory [MB]"]


def c3_base_config() -> V4CRescueConfig:
    return default_rescue_configs()[3]


def c3_runtime_optimized_config() -> V4CRescueConfig:
    base = c3_base_config()
    return replace(
        base,
        model_id="C3_V4C_depthwise_temporal_opt",
        num_workers=0,
        pin_memory=False,
        compile_model=False,
    )


def m1_scalar_config() -> ActV2Config:
    return ActV2Config(
        model_id="C2_M1_scalar",
        gate_mode="sigmoid",
        gate_granularity="scalar",
        batch_size=8,
        num_workers=0,
        use_amp=True,
    )


def apply_runtime_flags() -> dict[str, Any]:
    previous = {
        "benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        previous["matmul_precision"] = "high"
    return previous


def summarize_rows_with_memory(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_seeded_rows(df)
    if "Peak GPU memory [MB]" not in df.columns:
        return summary
    mem = (
        df.groupby(["Model", "Tile", "Horizon"], as_index=False)["Peak GPU memory [MB]"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Peak GPU memory [MB] mean", "std": "Peak GPU memory [MB] std"})
    )
    mem["Peak GPU memory [MB] std"] = mem["Peak GPU memory [MB] std"].fillna(0.0)
    return summary.merge(mem, on=["Model", "Tile", "Horizon"], how="left")


def m1_step1_verdict(summary_df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    model_df = summary_df[summary_df["Model"] == model_name].copy()
    non_abrupt_tiles = ["E32N34", "E32N35", "E39N30", "E44N23"]
    coseis_tiles = ["E48N24", "E58N17"]
    ratio = model_df["RMSE [mm] mean"] / model_df["Persistence RMSE [mm] mean"]
    model_df["RMSE_ratio_vs_persistence"] = ratio
    return {
        "model": model_name,
        "overall_mean_skill_mse": float(model_df["Skill_MSE mean"].mean()),
        "non_abrupt_mean_skill_mse": float(model_df[model_df["Tile"].isin(non_abrupt_tiles)]["Skill_MSE mean"].mean()),
        "coseismic_mean_skill_mse": float(model_df[model_df["Tile"].isin(coseis_tiles)]["Skill_MSE mean"].mean()),
        "coseismic_max_rmse_ratio": float(model_df[model_df["Tile"].isin(coseis_tiles)]["RMSE_ratio_vs_persistence"].max()),
        "positive_cells_overall": int((model_df["Skill_MSE mean"] > 0).sum()),
        "total_cells": int(len(model_df)),
    }


def common_summary_cols(df: pd.DataFrame) -> list[str]:
    ordered = ["Model", "Tile", "Horizon"]
    for metric in COMMON_METRICS:
        ordered.extend([f"{metric} mean", f"{metric} std"])
    return [col for col in ordered if col in df.columns]


def build_c3_loader(config: V4CRescueConfig, bundle: TileBundle, split_name: str, shuffle: bool) -> DataLoader:
    train_ds, val_ds, test_ds = prepare_rescue_datasets(bundle, config)
    mapping = {"train": train_ds, "val": val_ds, "test": test_ds}
    return DataLoader(mapping[split_name], **loader_kwargs(config, shuffle=shuffle))


def profile_m1_iteration_timing(
    bundle: TileBundle,
    config: ActV2Config,
    repeat_idx: int,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ActV2Dataset(bundle, resolution=64, split_name="train", config=config)
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    model = build_model(config, image_size=64, num_channels=ds.xs[0].shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    total_needed = min(len(loader), warmup_steps + measure_steps)
    if total_needed <= warmup_steps:
        raise RuntimeError("Not enough batches for M1 timing recheck.")

    iter_stage: dict[str, list[float]] = {name: [] for name in ["data_ms", "transfer_ms", "forward_ms", "loss_ms", "backward_ms", "step_ms", "total_iter_ms"]}
    prev_end = time.perf_counter()
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx in range(total_needed):
        sync_cuda(device)
        data_start = prev_end
        batch = next(iterator)
        sync_cuda(device)
        data_ms = (time.perf_counter() - data_start) * 1000.0

        transfer_start = time.perf_counter()
        batch = batch_to_device(batch, device)
        sync_cuda(device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

        optimizer.zero_grad(set_to_none=True)
        forward_start = time.perf_counter()
        with amp.autocast(device_type="cuda", enabled=config.use_amp and device.type == "cuda"):
            pred = model(batch)
            sync_cuda(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            loss_start = time.perf_counter()
            loss, _skill = compute_loss_m1(pred, batch, config)
            sync_cuda(device)
            loss_ms = (time.perf_counter() - loss_start) * 1000.0

        backward_start = time.perf_counter()
        if config.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sync_cuda(device)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            step_start = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            sync_cuda(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sync_cuda(device)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            step_start = time.perf_counter()
            optimizer.step()
            sync_cuda(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0
        prev_end = time.perf_counter()

        if step_idx < warmup_steps:
            continue
        total_iter = data_ms + transfer_ms + forward_ms + loss_ms + backward_ms + step_ms
        iter_stage["data_ms"].append(data_ms)
        iter_stage["transfer_ms"].append(transfer_ms)
        iter_stage["forward_ms"].append(forward_ms)
        iter_stage["loss_ms"].append(loss_ms)
        iter_stage["backward_ms"].append(backward_ms)
        iter_stage["step_ms"].append(step_ms)
        iter_stage["total_iter_ms"].append(total_iter)

    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
    row = {
        "Model": "C2_M1_scalar",
        "Repeat": repeat_idx,
        "Tile": bundle.tile_id,
        "Warmup steps": warmup_steps,
        "Measured steps": len(iter_stage["total_iter_ms"]),
        "Peak GPU memory [MB]": peak_mb,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    row.update({metric: float(np.mean(values)) for metric, values in iter_stage.items()})
    detail_rows = [
        {
            "Model": "C2_M1_scalar",
            "Repeat": repeat_idx,
            "Tile": bundle.tile_id,
            "Section": section,
            "Latency [ms]": float(value),
        }
        for section, values in iter_stage.items()
        for value in values
    ]
    return row, detail_rows


def profile_c3_iteration_timing(
    bundle: TileBundle,
    config: V4CRescueConfig,
    repeat_idx: int,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_c3_loader(config, bundle, split_name="train", shuffle=True)
    model = build_rescue_model(config, in_channels=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    total_needed = min(len(loader), warmup_steps + measure_steps)
    if total_needed <= warmup_steps:
        raise RuntimeError("Not enough batches for C3 timing recheck.")

    iter_stage: dict[str, list[float]] = {name: [] for name in ["data_ms", "transfer_ms", "forward_ms", "loss_ms", "backward_ms", "step_ms", "total_iter_ms"]}
    module_stage: dict[str, list[float]] = {}
    prev_end = time.perf_counter()
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx in range(total_needed):
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
        with amp.autocast(device_type="cuda", enabled=config.use_amp and device.type == "cuda"):
            pred, _aux, section_times = model(batch, return_aux=False, profile=True)  # type: ignore[misc]
            sync_cuda(device)
            forward_ms = (time.perf_counter() - forward_start) * 1000.0

            loss_start = time.perf_counter()
            loss, _skill = compute_loss_v4c(pred, batch)
            sync_cuda(device)
            loss_ms = (time.perf_counter() - loss_start) * 1000.0

        backward_start = time.perf_counter()
        if config.use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sync_cuda(device)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            step_start = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            sync_cuda(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sync_cuda(device)
            backward_ms = (time.perf_counter() - backward_start) * 1000.0

            step_start = time.perf_counter()
            optimizer.step()
            sync_cuda(device)
            step_ms = (time.perf_counter() - step_start) * 1000.0
        prev_end = time.perf_counter()

        if step_idx < warmup_steps:
            continue
        total_iter = data_ms + transfer_ms + forward_ms + loss_ms + backward_ms + step_ms
        iter_stage["data_ms"].append(data_ms)
        iter_stage["transfer_ms"].append(transfer_ms)
        iter_stage["forward_ms"].append(forward_ms)
        iter_stage["loss_ms"].append(loss_ms)
        iter_stage["backward_ms"].append(backward_ms)
        iter_stage["step_ms"].append(step_ms)
        iter_stage["total_iter_ms"].append(total_iter)
        for key, value in section_times.items():
            module_stage.setdefault(key, []).append(float(value))

    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
    row = {
        "Model": config.model_id,
        "Repeat": repeat_idx,
        "Tile": bundle.tile_id,
        "Warmup steps": warmup_steps,
        "Measured steps": len(iter_stage["total_iter_ms"]),
        "Peak GPU memory [MB]": peak_mb,
        "num_workers": config.num_workers,
        "pin_memory": bool(config.pin_memory),
    }
    row.update({metric: float(np.mean(values)) for metric, values in iter_stage.items()})
    detail_rows = [
        {
            "Model": config.model_id,
            "Repeat": repeat_idx,
            "Tile": bundle.tile_id,
            "Section": section,
            "Latency [ms]": float(value),
        }
        for section, values in module_stage.items()
        for value in values
    ]
    return row, detail_rows


def timed_train_recheck(
    bundles: dict[str, TileBundle],
    repeats: int = 5,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing_csv = CANDIDATE_ROOT / "c3_timing_recheck.csv"
    breakdown_csv = CANDIDATE_ROOT / "c3_timing_breakdown.csv"
    if reuse_if_exists and timing_csv.exists() and breakdown_csv.exists():
        return pd.read_csv(timing_csv), pd.read_csv(breakdown_csv)

    bundle = bundles[TIMING_TILE]
    m1_cfg = m1_scalar_config()
    c3_cfg = c3_base_config()
    c3_opt_cfg = c3_runtime_optimized_config()

    timing_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for repeat_idx in range(repeats):
        for label, kind, cfg in [
            ("C2_M1_scalar", "m1", m1_cfg),
            (c3_cfg.model_id, "c3", c3_cfg),
            (c3_opt_cfg.model_id, "c3", c3_opt_cfg),
        ]:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if kind == "m1":
                output_dir = CANDIDATE_ROOT / "runs" / "timing_recheck" / label / bundle.tile_id / f"repeat_{repeat_idx}"
                start = time.time()
                run = train_eval_single_tile(bundle, cfg, output_dir=output_dir, phase="timing_recheck", reuse_if_exists=reuse_if_exists)
                elapsed_wall = time.time() - start
                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
                timing_rows.append(
                    {
                        "Model": label,
                        "Repeat": repeat_idx,
                        "Tile": bundle.tile_id,
                        "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                        "Infer seconds": float(run["result"].get("infer_seconds", float("nan"))),
                        "Best epoch": int(run["result"]["best_epoch"]),
                        "Mean Skill_MSE": float(run["result"]["mean_skill_mse"]),
                        "Peak GPU memory [MB]": peak_mb,
                        "Wall seconds": float(elapsed_wall),
                        "num_workers": cfg.num_workers,
                        "pin_memory": False,
                    }
                )
                stage_row, stage_detail = profile_m1_iteration_timing(bundle, cfg, repeat_idx, warmup_steps=4, measure_steps=8)
                breakdown_rows.extend(stage_detail)
                timing_rows[-1].update({f"stage_{k}": v for k, v in stage_row.items() if k.endswith("_ms")})
            else:
                output_dir = CANDIDATE_ROOT / "runs" / "timing_recheck" / label / bundle.tile_id / f"repeat_{repeat_idx}"
                start = time.time()
                run = train_eval_rescue_single_tile(bundle, cfg, output_dir=output_dir, phase="timing_recheck", reuse_if_exists=reuse_if_exists)
                elapsed_wall = time.time() - start
                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
                timing_rows.append(
                    {
                        "Model": label,
                        "Repeat": repeat_idx,
                        "Tile": bundle.tile_id,
                        "Elapsed seconds": float(run["result"]["elapsed_seconds"]),
                        "Infer seconds": float(run["result"]["infer_seconds"]),
                        "Best epoch": int(run["result"]["best_epoch"]),
                        "Mean Skill_MSE": float(run["result"]["mean_skill_mse"]),
                        "Peak GPU memory [MB]": peak_mb,
                        "Wall seconds": float(elapsed_wall),
                        "num_workers": cfg.num_workers,
                        "pin_memory": bool(cfg.pin_memory),
                    }
                )
                stage_row, stage_detail = profile_c3_iteration_timing(bundle, cfg, repeat_idx, warmup_steps=4, measure_steps=8)
                breakdown_rows.extend(stage_detail)
                timing_rows[-1].update({f"stage_{k}": v for k, v in stage_row.items() if k.endswith("_ms")})

    timing_df = pd.DataFrame(timing_rows)
    breakdown_df = pd.DataFrame(breakdown_rows)
    timing_df.to_csv(timing_csv, index=False, encoding="utf-8-sig")
    breakdown_df.to_csv(breakdown_csv, index=False, encoding="utf-8-sig")
    return timing_df, breakdown_df


def run_rescue_seeded(
    bundles: dict[str, TileBundle],
    tiles: list[str],
    configs: list[V4CRescueConfig],
    seeds: list[int],
    phase: str,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for config in configs:
        write_simple_yaml(CANDIDATE_CONFIG_ROOT / f"{config.model_id}_{phase}.yaml", asdict(config))
        for seed in seeds:
            cfg = replace(config, seed=seed)
            for tile in tiles:
                print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                output_dir = CANDIDATE_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}"
                run = train_eval_rescue_single_tile(
                    bundle=bundles[tile],
                    config=cfg,
                    output_dir=output_dir,
                    phase=phase,
                    reuse_if_exists=reuse_if_exists,
                )
                peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else float("nan")
                print(
                    f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f} peak_mb={peak_mb:.1f}",
                    flush=True,
                )
                for row in run["result"]["rows"]:
                    payload = dict(row)
                    payload["Peak GPU memory [MB]"] = peak_mb
                    rows.append(payload)
                diag_rows.extend(run["result"]["diag_rows"])
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def run_c3_search_opt_quick(
    bundles: dict[str, TileBundle],
    config: V4CRescueConfig,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_path = CANDIDATE_ROOT / "c3_search_opt_quick_rows.csv"
    diag_path = CANDIDATE_ROOT / "c3_search_opt_quick_diag.csv"
    summary_path = CANDIDATE_ROOT / "c3_search_opt_quick_summary.csv"
    if reuse_if_exists and raw_path.exists() and summary_path.exists() and diag_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(diag_path), pd.read_csv(summary_path)

    raw_df, diag_df = run_rescue_seeded(bundles, SEARCH_TILES, [config], [42], phase="search_opt_quick", reuse_if_exists=reuse_if_exists)
    summary_df = summarize_rows_with_memory(raw_df)
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return raw_df, diag_df, summary_df


def select_full6_config(timing_df: pd.DataFrame, quick_summary: pd.DataFrame) -> tuple[V4CRescueConfig, dict[str, Any]]:
    base_cfg = c3_base_config()
    opt_cfg = c3_runtime_optimized_config()

    base_timing = timing_df[timing_df["Model"] == base_cfg.model_id]["Elapsed seconds"].mean()
    opt_timing = timing_df[timing_df["Model"] == opt_cfg.model_id]["Elapsed seconds"].mean()
    quick_mean_skill = float(quick_summary["Skill_MSE mean"].mean())
    quick_mean_train = float(quick_summary["Elapsed seconds mean"].mean())

    choose_opt = (not math.isnan(opt_timing)) and (math.isnan(base_timing) or opt_timing <= base_timing + 0.5)
    chosen = opt_cfg if choose_opt else base_cfg
    stats = {
        "base_timing_mean_seconds": float(base_timing),
        "opt_timing_mean_seconds": float(opt_timing),
        "opt_quick_mean_skill_mse": quick_mean_skill,
        "opt_quick_mean_train_seconds": quick_mean_train,
        "chosen_model_id": chosen.model_id,
        "selection_reason": "runtime_optimized_config_selected" if choose_opt else "baseline_config_retained",
    }
    return chosen, stats


def load_m1_full6_summary() -> pd.DataFrame:
    return pd.read_csv(M1_FULL6_SUMMARY_PATH)


def full6_verdict(c3_summary: pd.DataFrame, m1_summary: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    c3_verdict = m1_step1_verdict(c3_summary, c3_summary["Model"].iloc[0])
    m1_verdict = m1_step1_verdict(m1_summary, "C2_M1_scalar")
    c3_tile = c3_summary.groupby("Tile", as_index=False)["Skill_MSE mean"].mean().rename(columns={"Skill_MSE mean": "C3 Skill_MSE mean"})
    m1_tile = m1_summary.groupby("Tile", as_index=False)["Skill_MSE mean"].mean().rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"})
    tile_compare = c3_tile.merge(m1_tile, on="Tile", how="left")
    tile_compare["Delta C3-M1"] = tile_compare["C3 Skill_MSE mean"] - tile_compare["M1 Skill_MSE mean"]
    tile_compare["C3 better"] = tile_compare["Delta C3-M1"] > 0

    mean_train = float(c3_summary["Elapsed seconds mean"].mean())
    m1_mean_train = float(m1_summary["Elapsed seconds mean"].mean())
    train_ratio = mean_train / max(m1_mean_train, 1e-6)
    overall_delta = c3_verdict["overall_mean_skill_mse"] - m1_verdict["overall_mean_skill_mse"]
    better_tiles = int(tile_compare["C3 better"].sum())
    stress_df = c3_summary[c3_summary["Tile"].isin(["E48N24", "E58N17"])].copy()
    stress_ratio = float((stress_df["RMSE [mm] mean"] / stress_df["Persistence RMSE [mm] mean"]).max())

    pass_accuracy = (
        c3_verdict["overall_mean_skill_mse"] > m1_verdict["overall_mean_skill_mse"]
        and better_tiles >= 4
        and stress_ratio <= 1.03
    )
    if pass_accuracy and c3_verdict["overall_mean_skill_mse"] >= 0.075 and mean_train <= 35.5:
        status = "practical_go"
    elif pass_accuracy and c3_verdict["overall_mean_skill_mse"] >= 0.075 and mean_train <= 38.0 and train_ratio <= 6.0:
        status = "conditional_go"
    else:
        status = "fail"

    verdict = {
        "status": status,
        "c3_model": c3_summary["Model"].iloc[0],
        "c3_overall_mean_skill_mse": c3_verdict["overall_mean_skill_mse"],
        "m1_overall_mean_skill_mse": m1_verdict["overall_mean_skill_mse"],
        "overall_skill_delta_vs_m1": overall_delta,
        "c3_non_abrupt_mean_skill_mse": c3_verdict["non_abrupt_mean_skill_mse"],
        "c3_coseismic_mean_skill_mse": c3_verdict["coseismic_mean_skill_mse"],
        "stress_max_rmse_ratio": stress_ratio,
        "better_tiles_vs_m1": better_tiles,
        "mean_train_seconds": mean_train,
        "m1_mean_train_seconds": m1_mean_train,
        "train_ratio_vs_m1": train_ratio,
        "positive_cells_overall": c3_verdict["positive_cells_overall"],
        "total_cells": c3_verdict["total_cells"],
        "conditional_go_rule": "accuracy pass + mean_skill>=0.075 + better_tiles>=4 + stress_ratio<=1.03 + mean_train<=38.0 + train_ratio<=6.0",
    }
    return verdict, tile_compare


def run_c3_full6(
    bundles: dict[str, TileBundle],
    config: V4CRescueConfig,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_path = CANDIDATE_ROOT / "c3_full6_rows.csv"
    diag_path = CANDIDATE_ROOT / "c3_full6_diag.csv"
    summary_path = CANDIDATE_ROOT / "c3_full6_summary.csv"
    baseline_rows_path = CANDIDATE_ROOT / "c3_full6_baseline_rows.csv"
    baseline_summary_path = CANDIDATE_ROOT / "c3_full6_baseline_summary.csv"
    combined_summary_path = CANDIDATE_ROOT / "c3_full6_combined_summary.csv"
    verdict_path = CANDIDATE_ROOT / "c3_full6_verdict.json"
    tile_compare_path = CANDIDATE_ROOT / "c3_full6_tile_comparison.csv"

    if reuse_if_exists and all(
        path.exists()
        for path in [raw_path, diag_path, summary_path, baseline_rows_path, baseline_summary_path, combined_summary_path, verdict_path, tile_compare_path]
    ):
        raw_df = pd.read_csv(raw_path)
        diag_df = pd.read_csv(diag_path)
        summary_df = pd.read_csv(summary_path)
        baseline_summary = pd.read_csv(baseline_summary_path)
        combined_summary = pd.read_csv(combined_summary_path)
        verdict = dict(pd.read_json(verdict_path, typ="series"))
        return raw_df, diag_df, summary_df, baseline_summary, combined_summary, verdict

    raw_df, diag_df = run_rescue_seeded(bundles, FULL_TILES, [config], FINAL_SEEDS, phase="full6", reuse_if_exists=reuse_if_exists)
    summary_df = summarize_rows_with_memory(raw_df)
    baseline_rows = baseline_rows_for_tiles(bundles, FULL_TILES)
    baseline_summary = summarize_baselines(baseline_rows)
    m1_summary = load_m1_full6_summary()
    c3_common = summary_df[common_summary_cols(summary_df)]
    combined_summary = pd.concat([baseline_summary, m1_summary[common_summary_cols(m1_summary)], c3_common], ignore_index=True)
    verdict, tile_compare = full6_verdict(summary_df, m1_summary)

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    baseline_rows.to_csv(baseline_rows_path, index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(baseline_summary_path, index=False, encoding="utf-8-sig")
    combined_summary.to_csv(combined_summary_path, index=False, encoding="utf-8-sig")
    tile_compare.to_csv(tile_compare_path, index=False, encoding="utf-8-sig")
    write_json(verdict_path, verdict)
    return raw_df, diag_df, summary_df, baseline_summary, combined_summary, verdict


def select_c3_ablation_configs(base: V4CRescueConfig) -> list[V4CRescueConfig]:
    mapping: dict[str, V4CRescueConfig] = {}
    for cfg in ablation_variants(base):
        if cfg.model_id.endswith("_single_scale"):
            mapping["single_scale"] = replace(cfg, model_id="C3_no_multiscale_single_highres")
        elif cfg.model_id.endswith("_no_temporal"):
            mapping["no_temporal"] = replace(cfg, model_id="C3_no_depthwise_temporal")
        elif cfg.model_id.endswith("_no_bound"):
            mapping["no_bound"] = replace(cfg, model_id="C3_no_bounded_residual")
        elif cfg.model_id.endswith("_direct_disp"):
            mapping["direct_disp"] = replace(cfg, model_id="C3_direct_displacement_head")
    ordered = ["single_scale", "no_temporal", "no_bound", "direct_disp"]
    return [mapping[key] for key in ordered if key in mapping]


def run_c3_ablations(
    bundles: dict[str, TileBundle],
    base: V4CRescueConfig,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quick_raw_path = CANDIDATE_ROOT / "c3_ablation_quick_rows.csv"
    quick_summary_path = CANDIDATE_ROOT / "c3_ablation_quick_summary.csv"
    three_raw_path = CANDIDATE_ROOT / "c3_ablation_three_seed_rows.csv"
    three_summary_path = CANDIDATE_ROOT / "c3_ablation_three_seed_summary.csv"
    if reuse_if_exists and all(path.exists() for path in [quick_raw_path, quick_summary_path, three_raw_path, three_summary_path]):
        return (
            pd.read_csv(quick_raw_path),
            pd.read_csv(quick_summary_path),
            pd.read_csv(three_raw_path),
            pd.read_csv(three_summary_path),
        )

    ablations = select_c3_ablation_configs(base)
    quick_raw, _ = run_rescue_seeded(bundles, ABLATION_TILES, ablations, [42], phase="ablation_quick", reuse_if_exists=reuse_if_exists)
    quick_summary = summarize_rows_with_memory(quick_raw)
    three_raw, _ = run_rescue_seeded(bundles, ABLATION_TILES, ablations, FINAL_SEEDS, phase="ablation_three_seed", reuse_if_exists=reuse_if_exists)
    three_summary = summarize_rows_with_memory(three_raw)
    quick_raw.to_csv(quick_raw_path, index=False, encoding="utf-8-sig")
    quick_summary.to_csv(quick_summary_path, index=False, encoding="utf-8-sig")
    three_raw.to_csv(three_raw_path, index=False, encoding="utf-8-sig")
    three_summary.to_csv(three_summary_path, index=False, encoding="utf-8-sig")
    return quick_raw, quick_summary, three_raw, three_summary


def write_report(
    timing_df: pd.DataFrame,
    breakdown_df: pd.DataFrame,
    quick_summary: pd.DataFrame,
    selection_stats: dict[str, Any],
    full6_summary: pd.DataFrame,
    full6_verdict: dict[str, Any],
    ablation_quick_summary: pd.DataFrame | None,
    ablation_three_summary: pd.DataFrame | None,
) -> Path:
    base_timing = timing_df[timing_df["Model"] == c3_base_config().model_id]["Elapsed seconds"].mean()
    opt_timing = timing_df[timing_df["Model"] == c3_runtime_optimized_config().model_id]["Elapsed seconds"].mean()
    m1_timing = timing_df[timing_df["Model"] == "C2_M1_scalar"]["Elapsed seconds"].mean()
    quick_mean_skill = float(quick_summary["Skill_MSE mean"].mean())
    quick_mean_train = float(quick_summary["Elapsed seconds mean"].mean())
    quick_peak = float(quick_summary["Peak GPU memory [MB] mean"].mean()) if "Peak GPU memory [MB] mean" in quick_summary.columns else float("nan")
    c3_mean_peak = float(full6_summary["Peak GPU memory [MB] mean"].mean()) if "Peak GPU memory [MB] mean" in full6_summary.columns else float("nan")

    module_breakdown = (
        breakdown_df[breakdown_df["Model"] == c3_runtime_optimized_config().model_id]
        .groupby("Section", as_index=False)["Latency [ms]"]
        .mean()
        .sort_values("Latency [ms]", ascending=False)
    )
    top_sections = module_breakdown.head(6)

    lines = [
        "# C3 候选模型计时复核、六 Tile 主实验与最小消融报告",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 运行设置与假设",
        "",
        "- 本轮正式运行环境为 `conda py311 + CUDA`，GPU 为 `RTX 3090 Ti`。",
        "- `C3_opt` 与之前的 `C3` 在结构上相同；本轮只做了轻量运行时优化：`num_workers=0`、`pin_memory=False`、`cudnn benchmark=True`、`allow_tf32=True`，不改变模型语义。",
        "- 计时复核使用代表 tile `E32N34`，同一 GPU、同一 batch size=`8`，并在 iteration timing 中去掉前 `4` 个 warm-up batch。",
        "- 主模型决策规则分为两档：`practical_go` 需要满足你给出的 `35.5s` 硬门槛；若精度条件成立但训练时间只轻微超出，则记为 `conditional_go`。",
        "",
        "## 1. C3 计时复核",
        "",
        f"- M1 端到端 5 次复核平均训练时间 = {m1_timing:.2f}s",
        f"- 原始 C3 端到端 5 次复核平均训练时间 = {base_timing:.2f}s",
        f"- 运行时优化后的 C3_opt 端到端 5 次复核平均训练时间 = {opt_timing:.2f}s",
        f"- 运行时优化后的 C3_opt 在搜索集 quick 上平均 Skill_MSE = {quick_mean_skill:.4f}",
        f"- 运行时优化后的 C3_opt 在搜索集 quick 上平均训练时间 = {quick_mean_train:.2f}s",
        f"- 运行时优化后的 C3_opt 在搜索集 quick 上平均峰值显存 = {quick_peak:.1f} MB" if not math.isnan(quick_peak) else "- 搜索集 quick 峰值显存已记录在表中。",
        "",
        "C3_opt 的选择依据：",
        f"- base timing mean = {selection_stats['base_timing_mean_seconds']:.2f}s",
        f"- opt timing mean = {selection_stats['opt_timing_mean_seconds']:.2f}s",
        f"- opt quick mean skill = {selection_stats['opt_quick_mean_skill_mse']:.4f}",
        f"- 选择结果 = `{selection_stats['chosen_model_id']}`",
        "",
        "### C3_opt 迭代级热点",
        "",
    ]
    for _, row in top_sections.iterrows():
        lines.append(f"- {row['Section']}: {row['Latency [ms]']:.2f} ms")

    lines.extend(
        [
            "",
            "## 2. C3 六 Tile × 四 Horizon × 3 Seeds 主实验",
            "",
            f"- 模型：`{full6_verdict['c3_model']}`",
            f"- 六 tile 平均 Skill_MSE = {full6_verdict['c3_overall_mean_skill_mse']:.4f}",
            f"- 相对 M1 的平均 Skill_MSE 增量 = {full6_verdict['overall_skill_delta_vs_m1']:.4f}",
            f"- 非 abrupt 平均 Skill_MSE = {full6_verdict['c3_non_abrupt_mean_skill_mse']:.4f}",
            f"- co-seismic 平均 Skill_MSE = {full6_verdict['c3_coseismic_mean_skill_mse']:.4f}",
            f"- stress tile 最大 RMSE / persistence = {full6_verdict['stress_max_rmse_ratio']:.4f}",
            f"- 优于 M1 的 tile 数 = {full6_verdict['better_tiles_vs_m1']} / 6",
            f"- 平均训练时间 = {full6_verdict['mean_train_seconds']:.2f}s",
            f"- 相对 M1 的训练时间倍数 = {full6_verdict['train_ratio_vs_m1']:.2f}x",
            f"- 平均峰值显存 = {c3_mean_peak:.1f} MB" if not math.isnan(c3_mean_peak) else "- 峰值显存已记录在 full-six 汇总表中。",
            f"- 主实验状态 = `{full6_verdict['status']}`",
            "",
        ]
    )

    if ablation_quick_summary is not None and ablation_three_summary is not None:
        quick_rank = (
            ablation_quick_summary.groupby("Model", as_index=False)["Skill_MSE mean"]
            .mean()
            .sort_values("Skill_MSE mean", ascending=False)
        )
        three_rank = (
            ablation_three_summary.groupby("Model", as_index=False)["Skill_MSE mean"]
            .mean()
            .sort_values("Skill_MSE mean", ascending=False)
        )
        lines.extend(
            [
                "## 3. C3 最小结构消融",
                "",
                "### quick 1-seed 排名",
                "",
            ]
        )
        for _, row in quick_rank.iterrows():
            lines.append(f"- {row['Model']}: mean Skill_MSE = {row['Skill_MSE mean']:.4f}")
        lines.extend(["", "### 3-seed 排名", ""])
        for _, row in three_rank.iterrows():
            lines.append(f"- {row['Model']}: mean Skill_MSE = {row['Skill_MSE mean']:.4f}")
        lines.append("")

    lines.extend(
        [
            "## 4. 当前主模型结论",
            "",
        ]
    )
    if full6_verdict["status"] == "practical_go":
        lines.append("- `C3_opt` 已满足精度与复杂度双门槛，可以正式提升为主模型。")
    elif full6_verdict["status"] == "conditional_go":
        lines.append("- `C3_opt` 应视为 `conditional GO / near-pass`。它在六 tile 上的精度条件已经成立，但复杂度仍略高于 `35.5s` 硬门槛，需要在论文中明确写成 accuracy-established, efficiency-near-pass 的主模型候选。")
    else:
        lines.append("- `C3_opt` 在 full-six 上没有形成足以替代 M1 的稳定证据，主模型仍应保持 M1。")

    lines.extend(
        [
            "",
            "## 结果文件",
            "",
            f"- 计时复核：`{(CANDIDATE_ROOT / 'c3_timing_recheck.csv').as_posix()}`",
            f"- 计时分解：`{(CANDIDATE_ROOT / 'c3_timing_breakdown.csv').as_posix()}`",
            f"- 搜索集 quick：`{(CANDIDATE_ROOT / 'c3_search_opt_quick_summary.csv').as_posix()}`",
            f"- full-six 汇总：`{(CANDIDATE_ROOT / 'c3_full6_summary.csv').as_posix()}`",
            f"- full-six 决策：`{(CANDIDATE_ROOT / 'c3_full6_verdict.json').as_posix()}`",
        ]
    )
    if ablation_quick_summary is not None and ablation_three_summary is not None:
        lines.append(f"- ablation quick：`{(CANDIDATE_ROOT / 'c3_ablation_quick_summary.csv').as_posix()}`")
        lines.append(f"- ablation 3-seed：`{(CANDIDATE_ROOT / 'c3_ablation_three_seed_summary.csv').as_posix()}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def run_act_v4c_c3_candidate(reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(CANDIDATE_ROOT)
    ensure_dir(CANDIDATE_CONFIG_ROOT)
    runtime_flags = apply_runtime_flags()
    bundles = preprocess_all_tiles()

    timing_df, breakdown_df = timed_train_recheck(bundles, repeats=5, reuse_if_exists=reuse_if_exists)
    _quick_rows, _quick_diag, quick_summary = run_c3_search_opt_quick(bundles, c3_runtime_optimized_config(), reuse_if_exists=reuse_if_exists)
    full6_config, selection_stats = select_full6_config(timing_df, quick_summary)
    full6_rows, full6_diag, full6_summary, _baseline_summary, _combined_summary, full6_verdict = run_c3_full6(
        bundles,
        full6_config,
        reuse_if_exists=reuse_if_exists,
    )

    ablation_quick_summary: pd.DataFrame | None = None
    ablation_three_summary: pd.DataFrame | None = None
    if full6_verdict["status"] in {"practical_go", "conditional_go"}:
        _quick_raw, ablation_quick_summary, _three_raw, ablation_three_summary = run_c3_ablations(
            bundles,
            full6_config,
            reuse_if_exists=reuse_if_exists,
        )

    report = write_report(
        timing_df=timing_df,
        breakdown_df=breakdown_df,
        quick_summary=quick_summary,
        selection_stats=selection_stats,
        full6_summary=full6_summary,
        full6_verdict=full6_verdict,
        ablation_quick_summary=ablation_quick_summary,
        ablation_three_summary=ablation_three_summary,
    )

    summary = {
        "report": str(report.resolve()),
        "runtime_flags": runtime_flags,
        "timing_recheck": str((CANDIDATE_ROOT / "c3_timing_recheck.csv").resolve()),
        "timing_breakdown": str((CANDIDATE_ROOT / "c3_timing_breakdown.csv").resolve()),
        "search_opt_quick": str((CANDIDATE_ROOT / "c3_search_opt_quick_summary.csv").resolve()),
        "full6_summary": str((CANDIDATE_ROOT / "c3_full6_summary.csv").resolve()),
        "full6_verdict": full6_verdict,
        "ablation_quick_summary": str((CANDIDATE_ROOT / "c3_ablation_quick_summary.csv").resolve()) if ablation_quick_summary is not None else None,
        "ablation_three_seed_summary": str((CANDIDATE_ROOT / "c3_ablation_three_seed_summary.csv").resolve()) if ablation_three_summary is not None else None,
        "chosen_full6_config": selection_stats,
        "rows_recorded": int(len(full6_rows)),
        "diag_rows_recorded": int(len(full6_diag)),
    }
    write_json(SUMMARY_PATH, summary)
    return summary
