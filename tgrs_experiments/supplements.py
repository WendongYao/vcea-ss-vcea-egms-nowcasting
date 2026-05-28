from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    REGIME_TO_ID,
    ModelConfig,
    SequenceDataset,
    TileBundle,
    build_model,
    compute_loss,
    compute_metrics,
    ensure_dir,
    load_model,
    preprocess_all_tiles,
    train_and_eval,
)


SUPP_ROOT = OUTPUT_ROOT / "supplements"
SEEDS = [42, 43, 44]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def flat_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    return {
        "RMSE [mm]": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "MAE [mm]": float(np.mean(abs_err)),
        "Acc@1mm [%]": float(np.mean(abs_err < 1.0) * 100.0),
        "Acc@0.5mm [%]": float(np.mean(abs_err < 0.5) * 100.0),
        "Acc@0.2mm [%]": float(np.mean(abs_err < 0.2) * 100.0),
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_prediction_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name] for name in data.files}


def result_model_config(path: Path) -> ModelConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelConfig(**payload["model_config"])


def a1_three_seed_architecture(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    out_dir = ensure_dir(SUPP_ROOT / "A1_E2_3seeds")
    experiments = {
        "B4_no_query": ModelConfig(kind="act", epochs=10, patience=3, act_use_query=False),
        "B5_no_residual": ModelConfig(kind="act", epochs=10, patience=3, act_residual=False),
        "B7_smoothl1_only": ModelConfig(
            kind="act",
            epochs=10,
            patience=3,
            loss_mode="smoothl1",
            gradient_loss_weight=0.0,
            relative_loss_weight=0.0,
            delta_loss_weight=0.0,
        ),
        "B8_full_act": ModelConfig(kind="act", epochs=10, patience=3),
    }
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for tag, cfg in experiments.items():
            cfg_seed = replace(cfg, seed=seed)
            result = train_and_eval(
                output_dir=out_dir / "runs" / tag / f"seed_{seed}",
                train_bundle=bundles["E32N34"],
                eval_bundle=bundles["E32N34"],
                resolution=64,
                variant="A6",
                model_config=cfg_seed,
                reuse_if_exists=True,
            )
            rows.append(
                {
                    "Tile": "E32N34",
                    "Model": tag,
                    "Seed": seed,
                    "Best epoch": result["best_epoch"],
                    **result["metrics"],
                }
            )
    run_df = pd.DataFrame(rows)
    run_df.to_csv(out_dir / "a1_e2_three_seed_runs.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    for model, group in run_df.groupby("Model"):
        summary_rows.append(
            {
                "Tile": "E32N34",
                "Model": model,
                "RMSE mean [mm]": group["RMSE [mm]"].mean(),
                "RMSE std [mm]": group["RMSE [mm]"].std(ddof=1),
                "MAE mean [mm]": group["MAE [mm]"].mean(),
                "MAE std [mm]": group["MAE [mm]"].std(ddof=1),
                "Acc@1mm mean [%]": group["Acc@1mm [%]"].mean(),
                "Acc@1mm std [%]": group["Acc@1mm [%]"].std(ddof=1),
                "SSIM mean": group["SSIM"].mean(),
                "SSIM std": group["SSIM"].std(ddof=1),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE mean [mm]")
    summary_df.to_csv(out_dir / "a1_e2_three_seed_summary.csv", index=False, encoding="utf-8-sig")
    return {"runs": run_df, "summary": summary_df}


def fit_linear_seasonal_coeffs(bundle: TileBundle, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    tensors = bundle.tensors64 if resolution == 64 else bundle.tensors128
    disp = tensors["disp"].astype(np.float32)
    dates = pd.to_datetime(bundle.dates)
    t_all = ((dates - dates[0]).days.to_numpy(dtype=np.float32)) / 365.25
    t_train = t_all[: bundle.split.train_frame_end]
    y_train = disp[: bundle.split.train_frame_end].reshape(bundle.split.train_frame_end, -1)
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


def build_test_current_indices(bundle: TileBundle, horizons: list[int]) -> list[int]:
    return list(range(bundle.split.val_frame_end, bundle.split.total_frames - max(horizons)))


def baseline_predictions(bundle: TileBundle, resolution: int, horizons: list[int], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = bundle.tensors64 if resolution == 64 else bundle.tensors128
    disp = tensors["disp"].astype(np.float32)
    current_indices = build_test_current_indices(bundle, horizons)
    y_true = np.stack(
        [
            np.stack([disp[current + h] for h in horizons], axis=0)
            for current in current_indices
        ],
        axis=0,
    ).astype(np.float32)
    if method == "persistence":
        y_pred = np.stack(
            [np.repeat(disp[current][None, :, :], len(horizons), axis=0) for current in current_indices],
            axis=0,
        ).astype(np.float32)
        return y_true, y_pred, tensors["valid_mask"].astype(np.float32)
    if method != "linear_seasonal":
        raise ValueError(method)

    coeffs, t_all = fit_linear_seasonal_coeffs(bundle, resolution)
    coeffs = coeffs.reshape(4, -1)
    preds: list[np.ndarray] = []
    for current in current_indices:
        target_t = np.array([t_all[current + h] for h in horizons], dtype=np.float32)
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
        preds.append(pred_flat.reshape(len(horizons), disp.shape[1], disp.shape[2]))
    y_pred = np.stack(preds, axis=0).astype(np.float32)
    return y_true, y_pred, tensors["valid_mask"].astype(np.float32)


def multi_horizon_baselines(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    out_dir = ensure_dir(SUPP_ROOT / "A2_E3_baselines")
    horizons = [1, 3, 5, 10]
    rows: list[dict[str, Any]] = []
    for tile in ["E32N34", "E39N30", "E48N24"]:
        act_result = train_and_eval(
            output_dir=out_dir / "act_recheck" / tile,
            train_bundle=bundles[tile],
            eval_bundle=bundles[tile],
            resolution=64,
            variant="A6",
            model_config=ModelConfig(
                kind="act",
                horizons=horizons,
                dim=64,
                depth=1,
                heads=4,
                mlp_dim=128,
                patch_size=8,
                batch_size=8,
                epochs=15,
                patience=4,
                seed=42,
            ),
            reuse_if_exists=True,
        )
        for idx, horizon in enumerate(horizons):
            rows.append({"Tile": tile, "Horizon": f"t+{horizon}", "Method": "ACT-Former", **act_result["per_horizon_metrics"][f"t+{horizon}"]})
        for method in ["persistence", "linear_seasonal"]:
            y_true, y_pred, valid_mask = baseline_predictions(bundles[tile], 64, horizons, method)
            for idx, horizon in enumerate(horizons):
                metrics = compute_metrics(y_true[:, idx : idx + 1], y_pred[:, idx : idx + 1], valid_mask)
                rows.append({"Tile": tile, "Horizon": f"t+{horizon}", "Method": method, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a2_e3_act_vs_baselines.csv", index=False, encoding="utf-8-sig")
    return {"table": df}


def regimewise_a0_vs_a6(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    out_dir = ensure_dir(SUPP_ROOT / "A3_E1_regimewise")
    per_tile_rows: list[dict[str, Any]] = []
    aggregate_store: dict[str, dict[str, list[np.ndarray]]] = {
        regime: {"A0_true": [], "A0_pred": [], "A6_true": [], "A6_pred": []}
        for regime in REGIME_NAMES
    }
    for tile in bundles:
        for variant in ["A0", "A6"]:
            pred = load_prediction_npz(OUTPUT_ROOT / "E1" / tile / variant / "predictions.npz")
            valid_mask = pred["valid_mask"].astype(np.float32)
            regime_map = pred["regime_map"].astype(np.int16)
            for regime in REGIME_NAMES:
                rid = REGIME_TO_ID[regime]
                mask = valid_mask * (regime_map == rid).astype(np.float32)
                if float(mask.sum()) < 10:
                    continue
                metrics = compute_metrics(pred["y_true_mm"], pred["y_pred_mm"], mask)
                support = int(mask.sum())
                per_tile_rows.append({"Tile": tile, "Variant": variant, "Regime": regime, "Support": support, **metrics})
                true_flat = pred["y_true_mm"][..., mask > 0.5].reshape(-1)
                pred_flat = pred["y_pred_mm"][..., mask > 0.5].reshape(-1)
                aggregate_store[regime][f"{variant}_true"].append(true_flat)
                aggregate_store[regime][f"{variant}_pred"].append(pred_flat)
    per_tile_df = pd.DataFrame(per_tile_rows)
    per_tile_df.to_csv(out_dir / "a3_e1_regimewise_a0_a6_per_tile.csv", index=False, encoding="utf-8-sig")

    aggregate_rows: list[dict[str, Any]] = []
    for regime in REGIME_NAMES:
        if not aggregate_store[regime]["A0_true"]:
            continue
        a0_true = np.concatenate(aggregate_store[regime]["A0_true"])
        a0_pred = np.concatenate(aggregate_store[regime]["A0_pred"])
        a6_true = np.concatenate(aggregate_store[regime]["A6_true"])
        a6_pred = np.concatenate(aggregate_store[regime]["A6_pred"])
        a0_metrics = flat_metrics(a0_true, a0_pred)
        a6_metrics = flat_metrics(a6_true, a6_pred)
        aggregate_rows.append(
            {
                "Regime": regime,
                "A0 RMSE [mm]": a0_metrics["RMSE [mm]"],
                "A6 RMSE [mm]": a6_metrics["RMSE [mm]"],
                "Delta RMSE A6-A0 [mm]": a6_metrics["RMSE [mm]"] - a0_metrics["RMSE [mm]"],
                "A0 Acc@1mm [%]": a0_metrics["Acc@1mm [%]"],
                "A6 Acc@1mm [%]": a6_metrics["Acc@1mm [%]"],
                "Delta Acc@1mm A6-A0 [%]": a6_metrics["Acc@1mm [%]"] - a0_metrics["Acc@1mm [%]"],
                "Support values": len(a0_true),
            }
        )
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("Regime")
    aggregate_df.to_csv(out_dir / "a3_e1_regimewise_a0_a6_aggregate.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(aggregate_df), dtype=np.float32)
    rmse_delta = aggregate_df["Delta RMSE A6-A0 [mm]"].to_numpy(dtype=np.float32)
    acc_delta = aggregate_df["Delta Acc@1mm A6-A0 [%]"].to_numpy(dtype=np.float32)
    colors_rmse = ["#1f77b4" if value <= 0 else "#d62728" for value in rmse_delta]
    colors_acc = ["#1f77b4" if value >= 0 else "#d62728" for value in acc_delta]

    axes[0].bar(x, rmse_delta, color=colors_rmse)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_xticks(x, aggregate_df["Regime"], rotation=30, ha="right")
    axes[0].set_ylabel("Delta RMSE (A6 - A0) [mm]")
    axes[0].set_title("Auxiliary benefit by regime")

    axes[1].bar(x, acc_delta, color=colors_acc)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xticks(x, aggregate_df["Regime"], rotation=30, ha="right")
    axes[1].set_ylabel("Delta Acc@1mm (A6 - A0) [%]")
    axes[1].set_title("Accuracy gain by regime")

    fig.tight_layout()
    fig.savefig(out_dir / "a3_auxiliary_benefit_by_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"per_tile": per_tile_df, "aggregate": aggregate_df}


def redefined_e8_metrics(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    out_dir = ensure_dir(SUPP_ROOT / "A4_E8_redefined")
    rows: list[dict[str, Any]] = []

    for tile in ["E32N34", "E32N35", "E39N30", "E44N23"]:
        bundle = bundles[tile]
        velocity = bundle.tensors64["velocity_full"]
        trend_r2 = bundle.tensors64["trend_r2"]
        valid_mask = bundle.tensors64["valid_mask"]
        moving_mask = (valid_mask > 0.5) & (np.abs(velocity) >= 1.0) & (trend_r2 >= 0.6)
        current_indices = build_test_current_indices(bundle, [1])
        last_disp = bundle.tensors64["disp"][current_indices]
        for variant in ["A0", "A6"]:
            pred = load_prediction_npz(OUTPUT_ROOT / "E1" / tile / variant / "predictions.npz")
            increment_pred = pred["y_pred_mm"][:, 0] - last_disp
            sign_match = (np.sign(increment_pred) == np.sign(velocity)[None, :, :]).astype(np.float32)
            if moving_mask.sum() == 0:
                consistency = float("nan")
            else:
                consistency = float(sign_match[:, moving_mask].mean())
            rows.append(
                {
                    "Metric": "velocity_sign_consistency_redefined",
                    "Tile": tile,
                    "Variant": variant,
                    "Velocity threshold [mm/yr]": 1.0,
                    "Trend R2 threshold": 0.6,
                    "Moving pixel count": int(moving_mask.sum()),
                    "Value": consistency,
                }
            )

    for tile in ["E48N24", "E58N17"]:
        bundle = bundles[tile]
        pred = load_prediction_npz(OUTPUT_ROOT / "E1" / tile / "A6" / "predictions.npz")
        valid_mask = bundle.tensors64["valid_mask"]
        dates = pd.to_datetime(bundle.dates)
        mean_series = bundle.tensors64["disp"].mean(axis=(1, 2))
        event_idx = int(np.argmax(np.abs(np.diff(mean_series))) + 1)
        current_indices = build_test_current_indices(bundle, [1])
        target_indices = [idx + 1 for idx in current_indices]
        offsets = np.array(target_indices, dtype=np.int32) - event_idx
        rmse_series = np.array(
            [compute_metrics(pred["y_true_mm"][i : i + 1], pred["y_pred_mm"][i : i + 1], valid_mask)["RMSE [mm]"] for i in range(pred["y_true_mm"].shape[0])],
            dtype=np.float32,
        )
        post_mask = offsets >= 0
        post_rmse = rmse_series[post_mask]
        early_post = post_rmse[: min(12, len(post_rmse))]
        late_post = post_rmse[-min(12, len(post_rmse)) :]
        peak_post = float(np.max(early_post)) if len(early_post) else float(np.max(rmse_series))
        late_median = float(np.median(late_post)) if len(late_post) else float(np.median(rmse_series))
        pre_mask = offsets < 0
        if pre_mask.any():
            pre_median = float(np.median(rmse_series[pre_mask]))
            recovery_ratio = float(np.clip((peak_post - late_median) / max(1e-6, peak_post - pre_median), 0.0, 1.0))
            definition = "pre-event referenced"
        else:
            pre_median = float("nan")
            recovery_ratio = float(np.clip((peak_post - late_median) / max(1e-6, peak_post), 0.0, 1.0))
            definition = "post-only fallback"
        rows.append(
            {
                "Metric": "post_event_recovery_ratio",
                "Tile": tile,
                "Variant": "A6",
                "Event date": str(dates[event_idx].date()),
                "Peak early-post RMSE [mm]": peak_post,
                "Late-post median RMSE [mm]": late_median,
                "Pre-event median RMSE [mm]": pre_median,
                "Definition": definition,
                "Value": recovery_ratio,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a4_e8_redefined_metrics.csv", index=False, encoding="utf-8-sig")
    return {"table": df}


def model_resource_profile(
    bundle: TileBundle,
    resolution: int,
    variant: str,
    result_json_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_json_path.read_text(encoding="utf-8"))
    cfg = ModelConfig(**result["model_config"])
    train_ds = SequenceDataset(bundle, resolution, variant, "train", cfg.input_len, cfg.horizons)
    test_ds = SequenceDataset(bundle, resolution, variant, "test", cfg.input_len, cfg.horizons)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
    num_channels = train_ds.xs[0].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, cfg, num_channels, resolution, device)
    params = count_params(model)

    train_memory_mb = 0.0
    infer_memory_mb = 0.0
    inference_seconds = 0.0

    if device.type == "cuda":
        batch = next(iter(train_loader))
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = compute_loss(pred, y, batch, cfg)
        loss.backward()
        optimizer.step()
        train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                _ = model(batch["x"].to(device))
        torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - t0
        infer_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                _ = model(batch["x"].to(device))
        inference_seconds = time.perf_counter() - t0

    return {
        "Params": params,
        "Patch size": cfg.patch_size,
        "Tokens per frame": (resolution // cfg.patch_size) ** 2,
        "Batch size": cfg.batch_size,
        "Epoch budget": cfg.epochs,
        "Peak train memory [MiB]": train_memory_mb,
        "Inference time [s]": inference_seconds,
        "Peak inference memory [MiB]": infer_memory_mb,
        "Training time [s]": result["elapsed_seconds"],
    }


def persistence_profile(bundle: TileBundle, resolution: int) -> dict[str, Any]:
    y_true, y_pred, valid_mask = baseline_predictions(bundle, resolution, [1], "persistence")
    t0 = time.perf_counter()
    _ = baseline_predictions(bundle, resolution, [1], "persistence")
    infer_seconds = time.perf_counter() - t0
    metrics = compute_metrics(y_true, y_pred, valid_mask)
    return {
        **metrics,
        "Params": 0,
        "Patch size": np.nan,
        "Tokens per frame": np.nan,
        "Batch size": np.nan,
        "Epoch budget": 0,
        "Peak train memory [MiB]": 0.0,
        "Inference time [s]": infer_seconds,
        "Peak inference memory [MiB]": 0.0,
        "Training time [s]": 0.0,
    }


def e7_compute_table(bundles: dict[str, TileBundle]) -> dict[str, Any]:
    out_dir = ensure_dir(SUPP_ROOT / "A5_E7_compute")
    rows: list[dict[str, Any]] = []
    for tag, resolution, result_dir in [
        ("ACT-64", 64, OUTPUT_ROOT / "E7" / "E32N34" / "full_64"),
        ("ACT-128", 128, OUTPUT_ROOT / "E7" / "E32N34" / "full_128"),
    ]:
        resource = model_resource_profile(
            bundles["E32N34"],
            resolution=resolution,
            variant="A6",
            result_json_path=result_dir / "result.json",
            checkpoint_path=result_dir / "model.pth",
        )
        result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
        rows.append({"Model": tag, "Resolution": resolution, **result["metrics"], **resource})
    for resolution in [64, 128]:
        rows.append({"Model": f"Persistence-{resolution}", "Resolution": resolution, **persistence_profile(bundles["E32N34"], resolution)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "a5_e7_accuracy_compute_table.csv", index=False, encoding="utf-8-sig")
    return {"table": df}


def write_report(a1: dict[str, Any], a2: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any], a5: dict[str, Any]) -> Path:
    report_path = SUPP_ROOT / "TGRS_必须补小实验_实验记录.md"
    a1_df: pd.DataFrame = a1["summary"]
    a2_df: pd.DataFrame = a2["table"]
    a3_df: pd.DataFrame = a3["aggregate"]
    a4_df: pd.DataFrame = a4["table"]
    a5_df: pd.DataFrame = a5["table"]

    lines = [
        "# TGRS 必补小实验执行记录",
        "",
        "数据声明：本补充实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile 数据。",
        "",
        "## A1 E2 关键架构补 3 seeds",
        "",
    ]
    for _, row in a1_df.iterrows():
        lines.append(
            f"- {row['Model']}: RMSE = {row['RMSE mean [mm]']:.3f} ± {row['RMSE std [mm]']:.3f} mm，"
            f"MAE = {row['MAE mean [mm]']:.3f} ± {row['MAE std [mm]']:.3f} mm，"
            f"Acc@1mm = {row['Acc@1mm mean [%]']:.2f} ± {row['Acc@1mm std [%]']:.2f}%。"
        )
    lines.extend(
        [
            "",
            "结论：B5 no-residual 与其余变体之间的差距最大，说明 residual prediction 仍然是 E2 中最关键的结构贡献。B4 no-query 明显劣于 B8 full ACT，说明 query token 具备稳定增益。与此同时，B7 smoothl1-only 在当前轻量训练预算下略优于 B8，表明复合损失的额外收益在本轮 2019-2023 受限数据上并不绝对，需要在正文中避免过度表述。",
            "",
            "## A2 E3 补两个 baseline",
            "",
            "已补 persistence baseline 与 linear+seasonal baseline，覆盖 E32N34、E39N30、E48N24 的 `t+1/t+3/t+5/t+10`。为避免直接复用主实验中预算过轻的多步结果，此处对 ACT-Former 做了 15 epochs 的重跑校验。",
            "",
        ]
    )
    for tile in ["E32N34", "E39N30", "E48N24"]:
        sub = a2_df[a2_df["Tile"] == tile]
        lines.append(f"- {tile}:")
        for horizon in ["t+1", "t+3", "t+5", "t+10"]:
            sub_h = sub[sub["Horizon"] == horizon].sort_values("RMSE [mm]")
            ordered = "；".join(f"{r['Method']} RMSE={r['RMSE [mm]']:.3f} mm" for _, r in sub_h.iterrows())
            lines.append(f"  {horizon}: {ordered}")
    lines.extend(
        [
            "",
            "结论：ACT-Former 的 horizon 优势并非在所有 tile 上都成立。E32N34 上它在 `t+3/t+5/t+10` 优于 persistence；E39N30 上 persistence 在短期更强，而 linear+seasonal 在 `t+5/t+10` 最优；E48N24 上 persistence 全程最强。可以据此说明 ACT-Former 不是简单依赖 persistence，但在 2019-2023 受限数据设定下，短期优势具有明显场景依赖性。",
            "",
            "## A3 E1 做 A0 vs A6 的 regime-wise 分解",
            "",
        ]
    )
    for _, row in a3_df.iterrows():
        lines.append(
            f"- {row['Regime']}: A0 RMSE={row['A0 RMSE [mm]']:.3f} mm，A6 RMSE={row['A6 RMSE [mm]']:.3f} mm，"
            f"ΔRMSE(A6-A0)={row['Delta RMSE A6-A0 [mm]']:.3f} mm；"
            f"ΔAcc@1mm(A6-A0)={row['Delta Acc@1mm A6-A0 [%]']:.2f}% 。"
        )
    lines.extend(
        [
            "",
            "结论：辅助特征 A6 的收益并不具有 universal improvement。它只在 subsidence 与 accelerating 上带来轻微 RMSE 改善，而在 seasonal 与尤其 coseismic 上反而退化。这可以解释为什么 E1 的 tile-level 结果出现明显的 regime-dependent 差异。",
            "",
            "## A4 E8 指标重定义",
            "",
        ]
    )
    for _, row in a4_df[a4_df["Metric"] == "velocity_sign_consistency_redefined"].iterrows():
        lines.append(
            f"- {row['Tile']} {row['Variant']}: moving-pixel velocity-sign consistency = {row['Value']:.4f}，"
            f"阈值为 |velocity| >= {row['Velocity threshold [mm/yr]']} mm/yr 且 trend R² >= {row['Trend R2 threshold']}，"
            f"有效 moving pixels = {int(row['Moving pixel count'])}。"
        )
    for _, row in a4_df[a4_df["Metric"] == "post_event_recovery_ratio"].iterrows():
        lines.append(
            f"- {row['Tile']}: post-event recovery ratio = {row['Value']:.4f}，"
            f"定义 = {row['Definition']}，事件日期 = {row['Event date']}。"
        )
    lines.extend(
        [
            "",
            "结论：重定义后的 velocity-sign consistency 更聚焦于真实移动像元，避免被大量近静止像元稀释；post-event 指标则改为 recovery ratio。对 E48N24 和 E58N17，由于测试窗口几乎没有足够的 pre-event 段，报告中保留了 `post-only fallback` 标记，作为显式假设说明。",
            "",
            "## A5 E7 compute table",
            "",
        ]
    )
    for _, row in a5_df.iterrows():
        if str(row["Model"]).startswith("ACT"):
            lines.append(
                f"- {row['Model']}: RMSE={row['RMSE [mm]']:.3f} mm，SSIM={row['SSIM']:.3f}，"
                f"params={int(row['Params'])}，patch={int(row['Patch size'])}，tokens/frame={int(row['Tokens per frame'])}，"
                f"batch={int(row['Batch size'])}，epochs={int(row['Epoch budget'])}，"
                f"train_time={row['Training time [s]']:.3f}s，infer_time={row['Inference time [s]']:.3f}s，"
                f"peak_train_mem={row['Peak train memory [MiB]']:.1f} MiB。"
            )
        else:
            lines.append(
                f"- {row['Model']}: RMSE={row['RMSE [mm]']:.3f} mm，SSIM={row['SSIM']:.3f}，"
                f"params={int(row['Params'])}，train_time={row['Training time [s]']:.3f}s，"
                f"infer_time={row['Inference time [s]']:.3f}s，peak_train_mem={row['Peak train memory [MiB]']:.1f} MiB。"
            )
    lines.extend(
        [
            "",
            "结论：128×128 的 SSIM 略高，但 RMSE 明显更差，说明更高分辨率在当前设置下更偏向保持整体结构而非精确恢复幅值。与此同时，ACT-128 的参数量、训练时间与推理时间都更高。它的峰值训练显存没有高于 ACT-64，原因不是 128×128 更省显存，而是本轮设置采用了更大的 patch size=16 与更小的 batch size=2，使得每帧 token 数保持为 64、单步显存峰值被压低。因此，A5 应被解释为“当前优化预算下的 accuracy-compute trade-off”，而不是分辨率单变量对比。",
            "",
            "## 假设与说明",
            "",
            "- A2 中的 ACT-Former 结果为补充实验专门重跑，训练预算提升到 15 epochs，以避免直接继承主实验轻量多步设置带来的偏差。",
            "- A4 中 moving-pixel 的判定阈值固定为 `|velocity| >= 1.0 mm/yr` 且 `trend R² >= 0.6`；recovery ratio 的 `post-only fallback` 用于没有充分 pre-event 测试样本的 tile。",
            "- A5 中显存统计是当前硬件与当前 batch 配置下的一次训练 step 峰值，而不是整轮训练累计资源，也不是统一 batch 设定下的严格显存缩放实验。",
            "",
            "## 输出文件",
            "",
            f"- A1 runs: `{(SUPP_ROOT / 'A1_E2_3seeds' / 'a1_e2_three_seed_runs.csv').as_posix()}`",
            f"- A1: `{(SUPP_ROOT / 'A1_E2_3seeds' / 'a1_e2_three_seed_summary.csv').as_posix()}`",
            f"- A2: `{(SUPP_ROOT / 'A2_E3_baselines' / 'a2_e3_act_vs_baselines.csv').as_posix()}`",
            f"- A3 per-tile: `{(SUPP_ROOT / 'A3_E1_regimewise' / 'a3_e1_regimewise_a0_a6_per_tile.csv').as_posix()}`",
            f"- A3: `{(SUPP_ROOT / 'A3_E1_regimewise' / 'a3_e1_regimewise_a0_a6_aggregate.csv').as_posix()}`",
            f"- A3 Figure: `{(SUPP_ROOT / 'A3_E1_regimewise' / 'a3_auxiliary_benefit_by_regime.png').as_posix()}`",
            f"- A4: `{(SUPP_ROOT / 'A4_E8_redefined' / 'a4_e8_redefined_metrics.csv').as_posix()}`",
            f"- A5: `{(SUPP_ROOT / 'A5_E7_compute' / 'a5_e7_accuracy_compute_table.csv').as_posix()}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")
    return report_path


def run_mandatory_supplements() -> dict[str, Any]:
    ensure_dir(SUPP_ROOT)
    bundles = preprocess_all_tiles()
    a1 = a1_three_seed_architecture(bundles)
    a2 = multi_horizon_baselines(bundles)
    a3 = regimewise_a0_vs_a6(bundles)
    a4 = redefined_e8_metrics(bundles)
    a5 = e7_compute_table(bundles)
    report_path = write_report(a1, a2, a3, a4, a5)
    return {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
        "report": str(report_path),
    }
