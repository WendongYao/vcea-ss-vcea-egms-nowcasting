from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .act_v2 import (
    ACT_V2_ROOT,
    CONFIG_ROOT,
    EXPERT_NAMES,
    FINAL_SEEDS,
    SEARCH_HORIZONS,
    ActV2Config,
    ActV2Dataset,
    build_model,
    collect_predictions,
    train_eval_single_tile,
    write_simple_yaml,
)
from .pipeline import REGIME_NAMES, TILE_ORDER, TileBundle, compute_metrics, ensure_dir, preprocess_all_tiles, write_json


SUITE_ROOT = ACT_V2_ROOT / "m1_suite"
SUITE_CONFIG_ROOT = CONFIG_ROOT / "m1_suite"
FULL_TILES = TILE_ORDER
ABLATION_TILES = ["E32N34", "E39N30", "E48N24"]
REQUIRED_ABLATIONS = [
    "C1_M0_no_gate",
    "C2_M1_scalar",
    "C3_M1_patch",
    "C5_M1_no_query",
    "C6_M1_direct_disp",
    "C7_M1_gate_fixed1",
]
OPTIONAL_ABLATIONS = ["C4_M1_pixel", "C8_M1_gate_fixed0"]
METRIC_COLS = [
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


def default_step1_configs() -> dict[str, ActV2Config]:
    return {
        "M0_B7": ActV2Config(model_id="M0_B7", gate_mode="none"),
        "M1_patch": ActV2Config(model_id="M1_patch", gate_mode="sigmoid", gate_granularity="patch"),
    }


def default_ablation_configs() -> dict[str, ActV2Config]:
    return {
        "C1_M0_no_gate": ActV2Config(model_id="C1_M0_no_gate", gate_mode="none"),
        "C2_M1_scalar": ActV2Config(model_id="C2_M1_scalar", gate_mode="sigmoid", gate_granularity="scalar"),
        "C3_M1_patch": ActV2Config(model_id="C3_M1_patch", gate_mode="sigmoid", gate_granularity="patch"),
        "C4_M1_pixel": ActV2Config(model_id="C4_M1_pixel", gate_mode="sigmoid", gate_granularity="pixel", batch_size=4),
        "C5_M1_no_query": ActV2Config(model_id="C5_M1_no_query", gate_mode="sigmoid", query_enabled=False),
        "C6_M1_direct_disp": ActV2Config(model_id="C6_M1_direct_disp", gate_mode="sigmoid", residual_enabled=False),
        "C7_M1_gate_fixed1": ActV2Config(model_id="C7_M1_gate_fixed1", gate_mode="sigmoid", fixed_gate_value=1.0),
        "C8_M1_gate_fixed0": ActV2Config(model_id="C8_M1_gate_fixed0", gate_mode="sigmoid", fixed_gate_value=0.0),
    }


def summarize_seeded_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, tile, horizon), group in df.groupby(["Model", "Tile", "Horizon"], as_index=False):
        row: dict[str, Any] = {"Model": model, "Tile": tile, "Horizon": horizon}
        for metric in METRIC_COLS:
            row[f"{metric} mean"] = float(group[metric].mean())
            row[f"{metric} std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Model", "Tile", "Horizon"]).reset_index(drop=True)


def baseline_rows_for_tiles(bundles: dict[str, TileBundle], tiles: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tile in tiles:
        test_ds = ActV2Dataset(bundles[tile], resolution=64, split_name="test", config=ActV2Config(model_id="tmp", gate_mode="none"))
        valid_mask = test_ds.mask
        y_true = np.stack([arr[:, 0] for arr in test_ds.targets_mm], axis=0)
        y_pers = np.stack([arr[:, 0] for arr in test_ds.persistence_mm], axis=0)
        y_ls = np.stack([arr[:, 0] for arr in test_ds.linear_seasonal_mm], axis=0)
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
                    "Elapsed seconds": 0.0,
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
                    "Elapsed seconds": 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_baselines(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, tile, horizon), group in df.groupby(["Model", "Tile", "Horizon"], as_index=False):
        row: dict[str, Any] = {"Model": model, "Tile": tile, "Horizon": horizon}
        for metric in METRIC_COLS:
            row[f"{metric} mean"] = float(group[metric].mean())
            row[f"{metric} std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Model", "Tile", "Horizon"]).reset_index(drop=True)


def run_seeded_configs(
    bundles: dict[str, TileBundle],
    tiles: list[str],
    configs: list[ActV2Config],
    phase: str,
    seeds: list[int],
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(SUITE_ROOT)
    ensure_dir(SUITE_CONFIG_ROOT)
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for config in configs:
        write_simple_yaml(SUITE_CONFIG_ROOT / f"{config.model_id}.yaml", asdict(config))
        for seed in seeds:
            cfg = replace(config, seed=seed)
            for tile in tiles:
                print(f"[{phase}] start model={cfg.model_id} tile={tile} seed={seed}", flush=True)
                output_dir = SUITE_ROOT / "runs" / phase / cfg.model_id / tile / f"seed_{seed}"
                run = train_eval_single_tile(
                    bundle=bundles[tile],
                    config=cfg,
                    output_dir=output_dir,
                    phase=phase,
                    reuse_if_exists=reuse_if_exists,
                )
                print(
                    f"[{phase}] done model={cfg.model_id} tile={tile} seed={seed} mean_skill={run['result']['mean_skill_mse']:.4f}",
                    flush=True,
                )
                for row in run["result"]["rows"]:
                    row = dict(row)
                    row["RunDir"] = output_dir.as_posix()
                    rows.append(row)
                for row in run["result"]["gate_rows"]:
                    row = dict(row)
                    row["RunDir"] = output_dir.as_posix()
                    gate_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(gate_rows)


def gate_distribution_from_run(run_dir: Path, model_label: str, tile: str, seed: int) -> list[dict[str, Any]]:
    arrays = np.load(run_dir / "predictions.npz")
    weights = arrays["gate_weights"]
    valid_mask = arrays["valid_mask"] > 0.5
    rows: list[dict[str, Any]] = []
    for h_idx, horizon in enumerate(SEARCH_HORIZONS):
        gate_vals = weights[:, h_idx, 2][:, valid_mask].reshape(-1)
        rows.append(
            {
                "Model": model_label,
                "Tile": tile,
                "Seed": seed,
                "Horizon": f"t+{horizon}",
                "Gate mean": float(np.mean(gate_vals)),
                "Gate std": float(np.std(gate_vals)),
                "Gate p10": float(np.quantile(gate_vals, 0.10)),
                "Gate p50": float(np.quantile(gate_vals, 0.50)),
                "Gate p90": float(np.quantile(gate_vals, 0.90)),
                "Gate>0.25 share": float(np.mean(gate_vals > 0.25)),
                "Gate>0.50 share": float(np.mean(gate_vals > 0.50)),
                "Gate>0.75 share": float(np.mean(gate_vals > 0.75)),
            }
        )
    return rows


def collect_gate_distribution(phase: str, configs: list[ActV2Config], tiles: list[str], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in configs:
        for seed in seeds:
            for tile in tiles:
                run_dir = SUITE_ROOT / "runs" / phase / config.model_id / tile / f"seed_{seed}"
                if not (run_dir / "predictions.npz").exists():
                    continue
                rows.extend(gate_distribution_from_run(run_dir, config.model_id, tile, seed))
    return pd.DataFrame(rows)


def evaluate_step1(summary_df: pd.DataFrame, model_name: str) -> dict[str, Any]:
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


def pick_final_gate_variant(ablation_summary: pd.DataFrame) -> dict[str, Any]:
    c2 = ablation_summary[ablation_summary["Model"] == "C2_M1_scalar"]["Skill_MSE mean"].mean()
    c3 = ablation_summary[ablation_summary["Model"] == "C3_M1_patch"]["Skill_MSE mean"].mean()
    delta = float(c3 - c2)
    selected = "C3_M1_patch" if delta > 0.01 else "C2_M1_scalar"
    return {
        "selected_model": selected,
        "c2_mean_skill_mse": float(c2),
        "c3_mean_skill_mse": float(c3),
        "delta_c3_minus_c2": delta,
        "selection_rule": "假定 |C3-C2|<=0.01 视为“差不多”，此时优先选择更简单的 C2；只有当 C3 的平均 Skill_MSE 明显高出 0.01 以上时才保留 C3。",
    }


def load_prediction_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    arr = np.load(run_dir / "predictions.npz")
    return {name: arr[name] for name in arr.files}


def make_gate_vs_horizon(selected_model: str, phase: str, tiles: list[str], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for tile in tiles:
            run_dir = SUITE_ROOT / "runs" / phase / selected_model / tile / f"seed_{seed}"
            arrays = load_prediction_arrays(run_dir)
            valid_mask = arrays["valid_mask"] > 0.5
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                gate_vals = arrays["gate_weights"][:, h_idx, 2][:, valid_mask].reshape(-1)
                rows.append(
                    {
                        "Model": selected_model,
                        "Tile": tile,
                        "Seed": seed,
                        "Horizon": f"t+{horizon}",
                        "Gate mean": float(np.mean(gate_vals)),
                        "Gate p10": float(np.quantile(gate_vals, 0.10)),
                        "Gate p50": float(np.quantile(gate_vals, 0.50)),
                        "Gate p90": float(np.quantile(gate_vals, 0.90)),
                    }
                )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = df.groupby("Horizon", as_index=False)["Gate mean"].mean()
    ax.plot(plot_df["Horizon"], plot_df["Gate mean"], marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_title("Gate vs Horizon")
    ax.set_ylabel("Mean neural gate")
    ax.set_xlabel("Horizon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(SUITE_ROOT / "step3_gate_vs_horizon.png", dpi=200)
    plt.close(fig)
    return df


def make_gate_vs_regime(selected_model: str, phase: str, tiles: list[str], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for tile in tiles:
            run_dir = SUITE_ROOT / "runs" / phase / selected_model / tile / f"seed_{seed}"
            arrays = load_prediction_arrays(run_dir)
            valid_mask = arrays["valid_mask"] > 0.5
            regime_map = arrays["regime_map"]
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                gate_map = arrays["gate_weights"][:, h_idx, 2]
                for regime_idx, regime_name in enumerate(REGIME_NAMES):
                    mask = valid_mask & (regime_map == regime_idx)
                    if mask.sum() < 10:
                        continue
                    gate_vals = gate_map[:, mask].reshape(-1)
                    rows.append(
                        {
                            "Model": selected_model,
                            "Tile": tile,
                            "Seed": seed,
                            "Horizon": f"t+{horizon}",
                            "Regime": regime_name,
                            "Gate mean": float(np.mean(gate_vals)),
                        }
                    )
    df = pd.DataFrame(rows)
    plot_df = df.groupby("Regime", as_index=False)["Gate mean"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(plot_df["Regime"], plot_df["Gate mean"], color="#2ca02c")
    ax.set_title("Gate vs Regime")
    ax.set_ylabel("Mean neural gate")
    ax.set_xlabel("Regime")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(SUITE_ROOT / "step3_gate_vs_regime.png", dpi=200)
    plt.close(fig)
    return df


def make_gate_vs_improvement(selected_model: str, phase: str, tiles: list[str], seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
    labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
    for seed in seeds:
        for tile in tiles:
            run_dir = SUITE_ROOT / "runs" / phase / selected_model / tile / f"seed_{seed}"
            arrays = load_prediction_arrays(run_dir)
            valid_mask = arrays["valid_mask"] > 0.5
            gate_mean = arrays["gate_weights"][:, :, 2].mean(axis=0)
            model_mse = ((arrays["y_pred_mm"] - arrays["y_true_mm"]) ** 2).mean(axis=0)
            pers_mse = ((arrays["persistence_mm"] - arrays["y_true_mm"]) ** 2).mean(axis=0)
            gain = pers_mse - model_mse
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                gate_vals = gate_mean[h_idx][valid_mask]
                gain_vals = gain[h_idx][valid_mask]
                bucket = pd.cut(gate_vals, bins=bins, labels=labels, include_lowest=True, right=False)
                for label in labels:
                    mask = bucket == label
                    if int(np.sum(mask)) == 0:
                        continue
                    rows.append(
                        {
                            "Model": selected_model,
                            "Tile": tile,
                            "Seed": seed,
                            "Horizon": f"t+{horizon}",
                            "Gate bin": label,
                            "Mean error reduction [mm2]": float(np.mean(gain_vals[mask])),
                            "Positive-improvement share": float(np.mean(gain_vals[mask] > 0)),
                        }
                    )
    df = pd.DataFrame(rows)
    plot_df = df.groupby("Gate bin", as_index=False)["Mean error reduction [mm2]"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_df["Gate bin"], plot_df["Mean error reduction [mm2]"], color="#ff7f0e")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Gate vs Error Improvement")
    ax.set_ylabel("Persistence MSE - Model MSE")
    ax.set_xlabel("Mean gate bin")
    fig.tight_layout()
    fig.savefig(SUITE_ROOT / "step3_gate_vs_error_improvement.png", dpi=200)
    plt.close(fig)
    return df


def make_gate_maps(selected_model: str, phase: str, seed: int, tiles: list[str]) -> None:
    show_tiles = [tile for tile in tiles if tile in {"E32N34", "E39N30", "E48N24", "E58N17"}]
    fig, axes = plt.subplots(len(show_tiles), 2, figsize=(8.5, 3.0 * len(show_tiles)))
    if len(show_tiles) == 1:
        axes = np.array([axes])
    for row_idx, tile in enumerate(show_tiles):
        arrays = load_prediction_arrays(SUITE_ROOT / "runs" / phase / selected_model / tile / f"seed_{seed}")
        for col_idx, (h_idx, horizon) in enumerate([(0, "t+1"), (3, "t+10")]):
            gate_map = arrays["gate_weights"][:, h_idx, 2].mean(axis=0)
            ax = axes[row_idx, col_idx]
            im = ax.imshow(gate_map, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_title(f"{tile} {horizon}")
            ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(SUITE_ROOT / "step3_gate_maps.png", dpi=220)
    plt.close(fig)


def detect_event_index(bundle: TileBundle) -> int:
    disp = bundle.tensors64["disp"].astype(np.float32)
    mask = bundle.tensors64["valid_mask"] > 0.5
    frame_delta = np.abs(np.diff(disp[:, mask], axis=0)).mean(axis=1)
    return int(frame_delta.argmax() + 1)


def infer_model_for_split(bundle: TileBundle, config: ActV2Config, ckpt_path: Path, split_name: str) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ActV2Dataset(bundle, resolution=64, split_name=split_name, config=config)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    num_channels = ds.xs[0].shape[1]
    model = build_model(config, image_size=64, num_channels=num_channels).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return collect_predictions(model, loader, device)


def make_gate_time_curve(
    bundles: dict[str, TileBundle],
    analysis_model: str,
    config: ActV2Config,
    phase: str,
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tile in ["E48N24", "E58N17"]:
        bundle = bundles[tile]
        event_idx = detect_event_index(bundle)
        event_date = pd.to_datetime(bundle.dates[event_idx])
        for seed in seeds:
            run_dir = SUITE_ROOT / "runs" / phase / analysis_model / tile / f"seed_{seed}"
            ckpt_path = run_dir / "model.pth"
            split_arrays = [infer_model_for_split(bundle, replace(config, seed=seed), ckpt_path, split) for split in ["train", "val", "test"]]
            all_current = np.concatenate([arr["current_frames"] for arr in split_arrays], axis=0)
            all_weights = np.concatenate([arr["gate_weights"] for arr in split_arrays], axis=0)
            valid_mask = split_arrays[0]["valid_mask"] > 0.5
            order = np.argsort(all_current)
            all_current = all_current[order]
            all_weights = all_weights[order]
            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                gate_means = np.array([all_weights[i, h_idx, 2][valid_mask].mean() for i in range(all_weights.shape[0])], dtype=np.float32)
                for frame_idx, gate_mean in zip(all_current.tolist(), gate_means.tolist()):
                    rows.append(
                        {
                            "Model": analysis_model,
                            "Tile": tile,
                            "Seed": seed,
                            "Horizon": f"t+{horizon}",
                            "Current frame": int(frame_idx),
                            "Date": pd.to_datetime(bundle.dates[frame_idx]).strftime("%Y-%m-%d"),
                            "Gate mean": float(gate_mean),
                            "Auto event date": event_date.strftime("%Y-%m-%d"),
                        }
                    )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for ax, tile in zip(axes, ["E48N24", "E58N17"]):
        sub = df[(df["Tile"] == tile) & (df["Horizon"].isin(["t+1", "t+10"]))].copy()
        sub["Date"] = pd.to_datetime(sub["Date"])
        for horizon, color in [("t+1", "#1f77b4"), ("t+10", "#d62728")]:
            h_sub = sub[sub["Horizon"] == horizon].groupby("Date", as_index=False)["Gate mean"].mean().sort_values("Date")
            ax.plot(h_sub["Date"], h_sub["Gate mean"], label=horizon, color=color, linewidth=1.8)
        event_date = pd.to_datetime(sub["Auto event date"].iloc[0])
        ax.axvline(event_date, color="black", linestyle="--", linewidth=1.2, label="auto event")
        ax.set_title(tile)
        ax.set_ylabel("Mean gate")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(SUITE_ROOT / "step3_gate_time_curve.png", dpi=220)
    plt.close(fig)
    return df


def step1_main_experiment(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    print("[step1] preparing baselines", flush=True)
    baseline_df = baseline_rows_for_tiles(bundles, FULL_TILES)
    baseline_summary = summarize_baselines(baseline_df)
    baseline_df.to_csv(SUITE_ROOT / "step1_baseline_rows.csv", index=False, encoding="utf-8-sig")
    baseline_summary.to_csv(SUITE_ROOT / "step1_baseline_summary.csv", index=False, encoding="utf-8-sig")

    configs = list(default_step1_configs().values())
    model_df, gate_df = run_seeded_configs(bundles, FULL_TILES, configs, phase="step1_main", seeds=FINAL_SEEDS, reuse_if_exists=True)
    summary_df = summarize_seeded_rows(model_df)
    gate_dist_df = collect_gate_distribution("step1_main", configs, FULL_TILES, FINAL_SEEDS)

    model_df.to_csv(SUITE_ROOT / "step1_model_rows.csv", index=False, encoding="utf-8-sig")
    gate_df.to_csv(SUITE_ROOT / "step1_gate_rows.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUITE_ROOT / "step1_model_summary.csv", index=False, encoding="utf-8-sig")
    gate_dist_df.to_csv(SUITE_ROOT / "step1_gate_distribution.csv", index=False, encoding="utf-8-sig")

    combined_summary = pd.concat([baseline_summary, summary_df], ignore_index=True)
    combined_summary.to_csv(SUITE_ROOT / "step1_combined_summary.csv", index=False, encoding="utf-8-sig")
    verdict = evaluate_step1(summary_df, "M1_patch")
    write_json(SUITE_ROOT / "step1_verdict.json", verdict)
    print(f"[step1] verdict overall_mean_skill={verdict['overall_mean_skill_mse']:.4f}", flush=True)
    return {
        "baseline_rows": baseline_df,
        "baseline_summary": baseline_summary,
        "model_rows": model_df,
        "model_summary": summary_df,
        "combined_summary": combined_summary,
        "gate_rows": gate_df,
        "gate_distribution": gate_dist_df,
        "verdict": verdict,
    }


def step2_ablation(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    print("[step2] quick ablations", flush=True)
    configs = default_ablation_configs()
    quick_configs = list(configs.values())
    quick_df, quick_gate_df = run_seeded_configs(bundles, ABLATION_TILES, quick_configs, phase="step2_quick", seeds=[42], reuse_if_exists=True)
    quick_summary = summarize_seeded_rows(quick_df)
    quick_df.to_csv(SUITE_ROOT / "step2_quick_rows.csv", index=False, encoding="utf-8-sig")
    quick_gate_df.to_csv(SUITE_ROOT / "step2_quick_gate_rows.csv", index=False, encoding="utf-8-sig")
    quick_summary.to_csv(SUITE_ROOT / "step2_quick_summary.csv", index=False, encoding="utf-8-sig")

    print("[step2] three-seed ablations", flush=True)
    final_configs = [configs[name] for name in REQUIRED_ABLATIONS]
    final_df, final_gate_df = run_seeded_configs(bundles, ABLATION_TILES, final_configs, phase="step2_three_seed", seeds=FINAL_SEEDS, reuse_if_exists=True)
    final_summary = summarize_seeded_rows(final_df)
    final_df.to_csv(SUITE_ROOT / "step2_three_seed_rows.csv", index=False, encoding="utf-8-sig")
    final_gate_df.to_csv(SUITE_ROOT / "step2_three_seed_gate_rows.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(SUITE_ROOT / "step2_three_seed_summary.csv", index=False, encoding="utf-8-sig")

    selection = pick_final_gate_variant(final_summary)
    write_json(SUITE_ROOT / "step2_selection.json", selection)
    print(f"[step2] selected_model={selection['selected_model']} delta_c3_minus_c2={selection['delta_c3_minus_c2']:.4f}", flush=True)
    return {
        "quick_rows": quick_df,
        "quick_summary": quick_summary,
        "final_rows": final_df,
        "final_summary": final_summary,
        "selection": selection,
    }


def step1b_if_needed(bundles: dict[str, TileBundle], selection: dict[str, Any]) -> dict[str, Any] | None:
    if selection["selected_model"] != "C2_M1_scalar":
        print("[step1b] skipped because C3 remains selected", flush=True)
        return None
    print("[step1b] running six-tile scalar follow-up", flush=True)
    scalar_config = default_ablation_configs()["C2_M1_scalar"]
    model_df, gate_df = run_seeded_configs(bundles, FULL_TILES, [scalar_config], phase="step1b_scalar_full6", seeds=FINAL_SEEDS, reuse_if_exists=True)
    summary_df = summarize_seeded_rows(model_df)
    gate_dist_df = collect_gate_distribution("step1b_scalar_full6", [scalar_config], FULL_TILES, FINAL_SEEDS)
    model_df.to_csv(SUITE_ROOT / "step1b_scalar_rows.csv", index=False, encoding="utf-8-sig")
    gate_df.to_csv(SUITE_ROOT / "step1b_scalar_gate_rows.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(SUITE_ROOT / "step1b_scalar_summary.csv", index=False, encoding="utf-8-sig")
    gate_dist_df.to_csv(SUITE_ROOT / "step1b_scalar_gate_distribution.csv", index=False, encoding="utf-8-sig")
    verdict = evaluate_step1(summary_df, "C2_M1_scalar")
    write_json(SUITE_ROOT / "step1b_scalar_verdict.json", verdict)
    print(f"[step1b] verdict overall_mean_skill={verdict['overall_mean_skill_mse']:.4f}", flush=True)
    return {
        "model_rows": model_df,
        "model_summary": summary_df,
        "gate_rows": gate_df,
        "gate_distribution": gate_dist_df,
        "verdict": verdict,
    }


def write_report(
    step1: dict[str, Any],
    step2: dict[str, Any],
    step1b: dict[str, Any] | None,
    analysis_model: str,
    gate_horizon: pd.DataFrame,
    gate_regime: pd.DataFrame,
    gate_improvement: pd.DataFrame,
    gate_time_curve: pd.DataFrame,
) -> Path:
    selection = step2["selection"]
    selected_model = selection["selected_model"]
    selected_step = step1b if (selected_model == "C2_M1_scalar" and step1b is not None) else step1
    selected_verdict = selected_step["verdict"]
    report_path = SUITE_ROOT / "act_v2_m1_suite_detailed_report.md"

    def format_tile_table(summary_df: pd.DataFrame, model_name: str) -> list[str]:
        lines: list[str] = []
        sub = summary_df[summary_df["Model"] == model_name]
        for tile in FULL_TILES:
            tile_sub = sub[sub["Tile"] == tile]
            if tile_sub.empty:
                continue
            lines.append(f"- {tile}:")
            for horizon in ["t+1", "t+3", "t+5", "t+10"]:
                row = tile_sub[tile_sub["Horizon"] == horizon]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"  {horizon}: Skill_MSE={r['Skill_MSE mean']:.4f}±{r['Skill_MSE std']:.4f}, "
                    f"RMSE={r['RMSE [mm] mean']:.3f}±{r['RMSE [mm] std']:.3f} mm, "
                    f"MAE={r['MAE [mm] mean']:.3f}±{r['MAE [mm] std']:.3f} mm, "
                    f"Acc@1mm={r['Acc@1mm [%] mean']:.2f}%, "
                    f"Acc@0.5mm={r['Acc@0.5mm [%] mean']:.2f}%, "
                    f"SSIM={r['SSIM mean']:.4f}."
                )
        return lines

    lines = [
        "# ACT-Former v2 M1 六 Tile 主实验、结构消融与 Gate 分析报告",
        "",
        "数据声明：本轮实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是当前数据获取限制下仅这一时间段数据已经完成本地提取与处理。",
        "",
        "## 实验目标",
        "",
        "1. 将当前 PG-RQ-ACT（先按现有主线实现的 patch-wise gate 版本记为 `M1_patch`）扩展到六个 tile、四个 horizon、三随机种子。",
        "2. 通过 C1-C8 消融检验结构贡献，重点比较 scalar gate 与 patch-wise gate，并验证 query token 与 residual formulation 的必要性。",
        "3. 将 gate 行为显式可视化，回答模型何时贴近 persistence、何时启用 residual correction。",
        "",
        "## 关键假设",
        "",
        "- 按照任务说明中“如果 C3 patch-wise gate 比 C2 scalar gate 好就用 C3，否则用更简单的 C2”的文字，我额外假定：若 `C3 - C2` 的平均 Skill_MSE 差值不超过 `0.01`，则视为“差不多”。",
        "- `gate time curve` 为了尽量覆盖 co-seismic 事件前后，使用了同一 checkpoint 在 `train/val/test` 全窗口上重推，而不只看 test split；图中的事件日期由全 tile 平均绝对位移增量峰值自动检测。",
        "- `C8 gate fixed to 0` 与 persistence 在输出上等价，因此只保留 quick 结果，不再做 3 seeds。",
        "",
        "## Step 1：六 Tile 主实验",
        "",
        f"- 主实验汇总表：`{(SUITE_ROOT / 'step1_combined_summary.csv').as_posix()}`",
        f"- 模型原始结果：`{(SUITE_ROOT / 'step1_model_rows.csv').as_posix()}`",
        f"- Gate 分布：`{(SUITE_ROOT / 'step1_gate_distribution.csv').as_posix()}`",
        f"- 主实验判定：`{(SUITE_ROOT / 'step1_verdict.json').as_posix()}`",
        "",
        "### M1_patch 六 tile 结果",
        "",
    ]
    lines.extend(format_tile_table(step1["model_summary"], "M1_patch"))
    lines.extend(
        [
            "",
            f"- 六 tile × 四 horizon 平均 Skill_MSE = {step1['verdict']['overall_mean_skill_mse']:.4f}",
            f"- 非 abrupt regime（E32N34/E32N35/E39N30/E44N23）平均 Skill_MSE = {step1['verdict']['non_abrupt_mean_skill_mse']:.4f}",
            f"- Co-seismic tile（E48N24/E58N17）平均 Skill_MSE = {step1['verdict']['coseismic_mean_skill_mse']:.4f}",
            f"- Co-seismic 最大 RMSE / persistence = {step1['verdict']['coseismic_max_rmse_ratio']:.4f}",
            "",
        ]
    )
    if step1["verdict"]["overall_mean_skill_mse"] > 0 and step1["verdict"]["non_abrupt_mean_skill_mse"] > 0:
        lines.append("- 结论：`M1_patch` 在整体平均和非 abrupt regime 上满足正 skill 目标。")
    else:
        lines.append("- 结论：`M1_patch` 未能在全部主判据上达标，这部分已作为 failure mode 如实保留。")
    if step1["verdict"]["coseismic_mean_skill_mse"] < 0:
        lines.append("- Failure mode：co-seismic tile 仍可能出现负 skill，这表明 abrupt event 仍是当前 M1 的主要薄弱场景。")
    lines.extend(
        [
            "",
            "## Step 2：结构消融",
            "",
            f"- Quick 消融表：`{(SUITE_ROOT / 'step2_quick_summary.csv').as_posix()}`",
            f"- 3-seed 消融表：`{(SUITE_ROOT / 'step2_three_seed_summary.csv').as_posix()}`",
            f"- 结构选择记录：`{(SUITE_ROOT / 'step2_selection.json').as_posix()}`",
            "",
            "### 3-seed 最低要求消融结果",
            "",
        ]
    )
    for model_name in REQUIRED_ABLATIONS:
        model_sub = step2["final_summary"][step2["final_summary"]["Model"] == model_name]
        if model_sub.empty:
            continue
        lines.append(
            f"- {model_name}: 平均 Skill_MSE={model_sub['Skill_MSE mean'].mean():.4f}, "
            f"平均 RMSE={model_sub['RMSE [mm] mean'].mean():.4f} mm."
        )
    lines.extend(
        [
            "",
            f"- C2 平均 Skill_MSE = {selection['c2_mean_skill_mse']:.4f}",
            f"- C3 平均 Skill_MSE = {selection['c3_mean_skill_mse']:.4f}",
            f"- C3 - C2 = {selection['delta_c3_minus_c2']:.4f}",
            f"- 结构选择：`{selection['selected_model']}`",
            f"- 选择规则：{selection['selection_rule']}",
            "",
        ]
    )
    if selection["selected_model"] == "C2_M1_scalar" and step1b is not None:
        lines.extend(
            [
                "### Step 1b：因 C2 被选为最终 gate 变体，追加六 tile × 3 seeds 复验",
                "",
                f"- 复验汇总：`{(SUITE_ROOT / 'step1b_scalar_summary.csv').as_posix()}`",
                f"- 复验判定：`{(SUITE_ROOT / 'step1b_scalar_verdict.json').as_posix()}`",
                f"- C2 六 tile 平均 Skill_MSE = {step1b['verdict']['overall_mean_skill_mse']:.4f}",
                f"- C2 非 abrupt mean Skill_MSE = {step1b['verdict']['non_abrupt_mean_skill_mse']:.4f}",
                f"- C2 co-seismic mean Skill_MSE = {step1b['verdict']['coseismic_mean_skill_mse']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Step 3：Gate 行为分析",
            "",
            f"- 结构选择标签：`{selected_model}`；用于六 tile gate 可视化的对应全量模型：`{analysis_model}`。",
            f"- Gate vs horizon：`{(SUITE_ROOT / 'step3_gate_vs_horizon.csv').as_posix()}` / `step3_gate_vs_horizon.png`",
            f"- Gate vs regime：`{(SUITE_ROOT / 'step3_gate_vs_regime.csv').as_posix()}` / `step3_gate_vs_regime.png`",
            f"- Gate vs improvement：`{(SUITE_ROOT / 'step3_gate_vs_error_improvement.csv').as_posix()}` / `step3_gate_vs_error_improvement.png`",
            f"- Gate maps：`{(SUITE_ROOT / 'step3_gate_maps.png').as_posix()}`",
            f"- Gate time curve：`{(SUITE_ROOT / 'step3_gate_time_curve.csv').as_posix()}` / `step3_gate_time_curve.png`",
            "",
            f"- Horizon 平均 gate：t+1={gate_horizon[gate_horizon['Horizon']=='t+1']['Gate mean'].mean():.4f}, "
            f"t+3={gate_horizon[gate_horizon['Horizon']=='t+3']['Gate mean'].mean():.4f}, "
            f"t+5={gate_horizon[gate_horizon['Horizon']=='t+5']['Gate mean'].mean():.4f}, "
            f"t+10={gate_horizon[gate_horizon['Horizon']=='t+10']['Gate mean'].mean():.4f}.",
        ]
    )
    if not gate_regime.empty:
        regime_means = gate_regime.groupby("Regime", as_index=False)["Gate mean"].mean().sort_values("Gate mean", ascending=False)
        lines.append(
            "- Regime gate 排序：" + "；".join(
                f"{row['Regime']}={row['Gate mean']:.4f}" for _, row in regime_means.iterrows()
            )
            + "。"
        )
    if not gate_improvement.empty:
        imp = gate_improvement.groupby("Gate bin", as_index=False)["Mean error reduction [mm2]"].mean()
        best_bin = imp.sort_values("Mean error reduction [mm2]", ascending=False).iloc[0]
        worst_bin = imp.sort_values("Mean error reduction [mm2]", ascending=True).iloc[0]
        lines.append(
            f"- Gate 与误差改善：平均来看 `{best_bin['Gate bin']}` 桶的误差下降最大（{best_bin['Mean error reduction [mm2]']:.4f} mm²），"
            f"`{worst_bin['Gate bin']}` 桶最弱（{worst_bin['Mean error reduction [mm2]']:.4f} mm²）。"
        )
    if not gate_time_curve.empty:
        for tile in ["E48N24", "E58N17"]:
            sub = gate_time_curve[(gate_time_curve["Tile"] == tile) & (gate_time_curve["Horizon"] == "t+10")]
            if sub.empty:
                continue
            lines.append(
                f"- {tile} 自动事件日期：{sub['Auto event date'].iloc[0]}；t+10 全时段平均 gate = {sub['Gate mean'].mean():.4f}。"
            )
    lines.extend(
        [
            "",
            "## 综合结论",
            "",
            f"- 六 tile 主实验下，当前主线 `M1_patch` 的整体平均 Skill_MSE 为 {step1['verdict']['overall_mean_skill_mse']:.4f}。",
            f"- 消融后选定的最终 gate 变体是 `{selected_model}`。",
            f"- 最终 gate 变体在六 tile 上的主判据平均 Skill_MSE 为 {selected_verdict['overall_mean_skill_mse']:.4f}。",
            "- 如果 co-seismic tile 仍为负 skill，本报告保留其为明确 failure mode，而不做掩盖性总结。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_act_v2_m1_suite() -> dict[str, Any]:
    ensure_dir(SUITE_ROOT)
    ensure_dir(SUITE_CONFIG_ROOT)
    bundles = preprocess_all_tiles()

    print("[suite] step1 start", flush=True)
    step1 = step1_main_experiment(bundles)
    print("[suite] step2 start", flush=True)
    step2 = step2_ablation(bundles)
    step1b = step1b_if_needed(bundles, step2["selection"])

    selected_model = step2["selection"]["selected_model"]
    selected_phase = "step1b_scalar_full6" if (selected_model == "C2_M1_scalar" and step1b is not None) else "step1_main"
    analysis_model = "C2_M1_scalar" if selected_model == "C2_M1_scalar" else "M1_patch"
    if selected_model == "C2_M1_scalar":
        selected_config = default_ablation_configs()["C2_M1_scalar"]
    elif selected_model == "C3_M1_patch":
        selected_config = default_ablation_configs()["C3_M1_patch"]
    else:
        raise ValueError(f"Unexpected selected model: {selected_model}")

    gate_horizon = make_gate_vs_horizon(analysis_model, selected_phase, FULL_TILES, FINAL_SEEDS)
    gate_regime = make_gate_vs_regime(analysis_model, selected_phase, FULL_TILES, FINAL_SEEDS)
    gate_improvement = make_gate_vs_improvement(analysis_model, selected_phase, FULL_TILES, FINAL_SEEDS)
    make_gate_maps(analysis_model, selected_phase, seed=42, tiles=FULL_TILES)
    gate_time_curve = make_gate_time_curve(bundles, analysis_model, selected_config, selected_phase, FINAL_SEEDS)

    gate_horizon.to_csv(SUITE_ROOT / "step3_gate_vs_horizon.csv", index=False, encoding="utf-8-sig")
    gate_regime.to_csv(SUITE_ROOT / "step3_gate_vs_regime.csv", index=False, encoding="utf-8-sig")
    gate_improvement.to_csv(SUITE_ROOT / "step3_gate_vs_error_improvement.csv", index=False, encoding="utf-8-sig")
    gate_time_curve.to_csv(SUITE_ROOT / "step3_gate_time_curve.csv", index=False, encoding="utf-8-sig")

    print("[suite] writing report", flush=True)
    report_path = write_report(step1, step2, step1b, analysis_model, gate_horizon, gate_regime, gate_improvement, gate_time_curve)
    summary = {
        "report": str(report_path),
        "selected_model": selected_model,
        "analysis_model": analysis_model,
        "selected_phase": selected_phase,
        "step1_verdict": step1["verdict"],
        "step2_selection": step2["selection"],
        "step1b_verdict": None if step1b is None else step1b["verdict"],
        "gate_vs_horizon_csv": str(SUITE_ROOT / "step3_gate_vs_horizon.csv"),
        "gate_vs_regime_csv": str(SUITE_ROOT / "step3_gate_vs_regime.csv"),
        "gate_vs_error_improvement_csv": str(SUITE_ROOT / "step3_gate_vs_error_improvement.csv"),
        "gate_time_curve_csv": str(SUITE_ROOT / "step3_gate_time_curve.csv"),
    }
    write_json(SUITE_ROOT / "run_summary.json", summary)
    print(f"[suite] done report={report_path.as_posix()}", flush=True)
    return summary
