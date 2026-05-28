from __future__ import annotations

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

from .act_v2 import FINAL_SEEDS, SEARCH_HORIZONS, write_simple_yaml
from .act_v3 import result_rows, summarize_seeded_rows
from .act_v5_selector import (
    ACT_V5_ROOT,
    EXPERT_CACHE_MANIFEST,
    EXPERT_NAMES,
    EXPERT_SEED_MANIFEST,
    M1_RUN_ROOT,
    C3_RUN_ROOT,
    ORACLE_WIN_TABLE_PATH,
    ensure_expert_caches,
    ensure_seed_manifest,
    infer_c3_split,
    infer_m1_split,
    load_cache,
    selected_run_dir,
)
from .pipeline import (
    OUTPUT_ROOT,
    REGIME_NAMES,
    TILE_ORDER,
    TileBundle,
    compute_metrics,
    ensure_dir,
    preprocess_all_tiles,
    seed_everything,
    write_json,
)


ACT_V6_ROOT = OUTPUT_ROOT / "act_v6"
ACT_V6_CONFIG_ROOT = ACT_V6_ROOT / "configs"
UNCERTAINTY_CACHE_ROOT = ACT_V6_ROOT / "uncertainty_cache"
ORACLE_BREAKDOWN_PATH = ACT_V6_ROOT / "v6_oracle_breakdown.csv"
ORACLE_PATCH_FIGURE = ACT_V6_ROOT / "v6_oracle_patch_gain.png"
V6A_RESULTS_PATH = ACT_V6_ROOT / "v6a_results.csv"
V6B_RESULTS_PATH = ACT_V6_ROOT / "v6b_results.csv"
V6C_RESULTS_PATH = ACT_V6_ROOT / "v6c_results.csv"
V6_BEST_3SEED_PATH = ACT_V6_ROOT / "v6_best_3seed.csv"
V6_BEST_VERDICT_PATH = ACT_V6_ROOT / "v6_best_verdict.json"
V6_ABLATION_PATH = ACT_V6_ROOT / "v6_ablation.csv"
V6_FINAL_DECISION_PATH = ACT_V6_ROOT / "v6_final_decision.md"
V6_REPORT_PATH = ACT_V6_ROOT / "act_v6_detailed_report.md"
V6_SUMMARY_PATH = ACT_V6_ROOT / "run_summary.json"
V6_DIAG_ROOT = ACT_V6_ROOT / "v6_selector_diagnostics"
M1_FULL6_SUMMARY_PATH = OUTPUT_ROOT / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv"
V5_RISK_3SEED_PATH = ACT_V5_ROOT / "risk_selector_3seed.csv"
V5_RISK_VERDICT_PATH = ACT_V5_ROOT / "risk_selector_verdict.json"

STRESS_RATIO_THRESHOLD = 1.03
V5_MEAN_SKILL = 0.09643435394262183
V5_STD = 0.02794664925318695
M1_MEAN_SKILL = 0.0783341230429559


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def result_dir_for_phase(phase: str, model_id: str, seed: int) -> Path:
    return ACT_V6_ROOT / "runs" / phase / model_id / f"seed_{seed}"


def load_phase_result(phase: str, model_id: str, seed: int) -> dict[str, Any] | None:
    result_path = result_dir_for_phase(phase, model_id, seed) / "result.json"
    if not result_path.exists():
        return None
    return pd.read_json(result_path, typ="series").to_dict()


def backfill_runtime_columns(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if df.empty or "Model" not in df.columns or "Seed" not in df.columns:
        return df
    out = df.copy()
    for (model_id, seed), _sub in out.groupby(["Model", "Seed"]):
        payload = load_phase_result(phase, str(model_id), int(seed))
        if payload is None:
            continue
        mask = (out["Model"] == model_id) & (out["Seed"] == seed)
        out.loc[mask, "Elapsed seconds"] = float(payload.get("elapsed_seconds", np.nan))
        out.loc[mask, "Infer seconds"] = float(payload.get("infer_seconds", np.nan))
        out.loc[mask, "Best epoch"] = int(payload.get("best_epoch", -1))
        out.loc[mask, "Param count"] = int(payload.get("param_count", 0))
    return out


def mean_train_seconds_from_results(phase: str, model_id: str, seeds: list[int]) -> float:
    values: list[float] = []
    for seed in seeds:
        payload = load_phase_result(phase, model_id, seed)
        if payload is None:
            continue
        elapsed = payload.get("elapsed_seconds")
        if elapsed is not None:
            values.append(float(elapsed))
    if not values:
        return float("nan")
    return float(np.mean(values))


@dataclass
class V6Config:
    model_id: str
    variant: str
    patch_size: int = 8
    hidden_dim: int = 32
    depth: int = 3
    batch_size: int = 32
    epochs: int = 18
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 0
    entropy_lambda: float = 0.005
    lambda_m1: float = 0.0
    lambda_pers: float = 0.0
    margin_m1: float = 0.0
    margin_pers: float = 0.0
    use_uncertainty: bool = False
    use_risk_head: bool = False
    risk_bce_lambda: float = 0.0
    persistence_bias: float = 0.5
    m1_bias: float = 1.0
    risk_thresholds: list[float] = field(default_factory=lambda: [0.45, 0.5, 0.55, 0.6, 0.65])


def default_v6_configs() -> dict[str, list[V6Config] | V6Config]:
    return {
        "v6a": V6Config(
            model_id="V6A_anchor_soft",
            variant="v6a",
            patch_size=8,
            hidden_dim=32,
            depth=3,
            lambda_m1=0.0,
            lambda_pers=0.0,
            use_uncertainty=False,
            use_risk_head=False,
            risk_bce_lambda=0.0,
            entropy_lambda=0.005,
        ),
        "v6b": [
            V6Config(
                model_id="V6B_anchor_risk_l20",
                variant="v6b",
                patch_size=8,
                hidden_dim=32,
                depth=3,
                lambda_m1=0.20,
                lambda_pers=0.10,
                use_uncertainty=False,
                use_risk_head=True,
                risk_bce_lambda=0.15,
                entropy_lambda=0.005,
            ),
            V6Config(
                model_id="V6B_anchor_risk_l50",
                variant="v6b",
                patch_size=8,
                hidden_dim=32,
                depth=3,
                lambda_m1=0.50,
                lambda_pers=0.15,
                use_uncertainty=False,
                use_risk_head=True,
                risk_bce_lambda=0.20,
                entropy_lambda=0.005,
            ),
        ],
        "v6c": V6Config(
            model_id="V6C_anchor_risk_uncert",
            variant="v6c",
            patch_size=8,
            hidden_dim=36,
            depth=3,
            lambda_m1=0.35,
            lambda_pers=0.12,
            use_uncertainty=True,
            use_risk_head=True,
            risk_bce_lambda=0.20,
            entropy_lambda=0.004,
        ),
    }


def load_m1_full6_summary() -> pd.DataFrame:
    return pd.read_csv(M1_FULL6_SUMMARY_PATH)


def load_v5_risk_verdict() -> dict[str, Any]:
    return pd.read_json(V5_RISK_VERDICT_PATH, typ="series").to_dict()


def load_seeded_prediction_std(
    bundles: dict[str, TileBundle],
    run_manifest: pd.DataFrame,
    family: str,
    tile: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    root = M1_RUN_ROOT if family == "M1" else C3_RUN_ROOT
    infer_fn = infer_m1_split if family == "M1" else infer_c3_split
    for seed in FINAL_SEEDS:
        run_dir = root / tile / f"seed_{seed}"
        arrays = infer_fn(bundles[tile], run_dir, split_name)
        preds.append(arrays["y_pred_mm"].astype(np.float32))
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0).astype(np.float32), stack.std(axis=0).astype(np.float32)


def ensure_uncertainty_caches(bundles: dict[str, TileBundle], reuse_if_exists: bool = True) -> pd.DataFrame:
    manifest_path = ACT_V6_ROOT / "v6_uncertainty_cache_manifest.csv"
    if reuse_if_exists and manifest_path.exists():
        return pd.read_csv(manifest_path)

    run_manifest = ensure_seed_manifest(reuse_if_exists=reuse_if_exists)
    rows: list[dict[str, Any]] = []
    for tile in TILE_ORDER:
        tile_dir = ensure_dir(UNCERTAINTY_CACHE_ROOT / tile)
        for split_name in ["train", "val", "test"]:
            out_path = tile_dir / f"{split_name}.npz"
            if not (reuse_if_exists and out_path.exists()):
                m1_mean, m1_std = load_seeded_prediction_std(bundles, run_manifest, "M1", tile, split_name)
                c3_mean, c3_std = load_seeded_prediction_std(bundles, run_manifest, "C3_opt", tile, split_name)
                np.savez_compressed(
                    out_path,
                    m1_mean=m1_mean,
                    m1_std=m1_std,
                    c3_mean=c3_mean,
                    c3_std=c3_std,
                )
            rows.append({"Tile": tile, "Split": split_name, "Cache path": out_path.as_posix()})
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return df


def load_uncertainty_cache(tile: str, split_name: str) -> dict[str, np.ndarray]:
    path = UNCERTAINTY_CACHE_ROOT / tile / f"{split_name}.npz"
    arr = np.load(path)
    return {name: arr[name] for name in arr.files}


def valid_mean_std(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = arr[mask > 0.5]
    return float(vals.mean()), float(vals.std() + 1e-6)


def zscore_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mu, sigma = valid_mean_std(arr, mask)
    return ((arr - mu) / sigma).astype(np.float32)


class GlobalSelectorDataset(Dataset):
    def __init__(self, tiles: list[str], split_name: str, config: V6Config) -> None:
        self.tiles = tiles
        self.split_name = split_name
        self.config = config
        self.patch_size = config.patch_size
        self.tile_caches: dict[str, dict[str, np.ndarray]] = {}
        self.tile_uncert: dict[str, dict[str, np.ndarray] | None] = {}
        self.tile_aux: dict[str, dict[str, np.ndarray]] = {}
        self.examples: list[tuple[str, int, int]] = []
        self.tile_counts: dict[str, int] = {}

        for tile in tiles:
            cache = load_cache(tile, split_name)
            uncert = load_uncertainty_cache(tile, split_name) if config.use_uncertainty else None
            self.tile_caches[tile] = cache
            self.tile_uncert[tile] = uncert
            mask = cache["valid_mask"].astype(np.float32)
            aux = {
                "trend_norm": zscore_map(cache["trend_r2"], mask),
                "amp_norm": zscore_map(cache["descriptor_amp"], mask),
                "resid_norm": zscore_map(cache["residual_std"], mask),
                "change_norm": zscore_map(cache["change_score"], mask),
                "regime_norm": (cache["regime_map"].astype(np.float32) / max(1.0, float(len(REGIME_NAMES) - 1))).astype(np.float32),
            }
            self.tile_aux[tile] = aux
            count = int(cache["y_true_mm"].shape[0] * cache["y_true_mm"].shape[1])
            self.tile_counts[tile] = count
            for sample_idx in range(cache["y_true_mm"].shape[0]):
                for horizon_idx in range(cache["y_true_mm"].shape[1]):
                    self.examples.append((tile, sample_idx, horizon_idx))

        self.tile_weights = {tile: float(1.0 / max(count, 1)) for tile, count in self.tile_counts.items()}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tile, sample_idx, horizon_idx = self.examples[index]
        cache = self.tile_caches[tile]
        aux = self.tile_aux[tile]
        uncert = self.tile_uncert[tile]

        target = cache["y_true_mm"][sample_idx, horizon_idx]
        persistence = cache["persistence_mm"][sample_idx, horizon_idx]
        linear_seasonal = cache["linear_seasonal_mm"][sample_idx, horizon_idx]
        m1_pred = cache["m1_pred_mm"][sample_idx, horizon_idx]
        c3_pred = cache["c3_pred_mm"][sample_idx, horizon_idx]
        last_disp = cache["last_disp_mm"][sample_idx, 0]
        scale = np.maximum(cache["c3_scale_mm"][sample_idx, horizon_idx], 0.1)
        gate = cache["m1_gate_mm"][sample_idx, horizon_idx]
        horizon = SEARCH_HORIZONS[horizon_idx]

        features = [
            ((persistence - last_disp) / scale).astype(np.float32),
            ((linear_seasonal - last_disp) / scale).astype(np.float32),
            ((m1_pred - last_disp) / scale).astype(np.float32),
            ((c3_pred - last_disp) / scale).astype(np.float32),
            (np.abs(m1_pred - persistence) / scale).astype(np.float32),
            (np.abs(c3_pred - persistence) / scale).astype(np.float32),
            (np.abs(c3_pred - m1_pred) / scale).astype(np.float32),
            gate.astype(np.float32),
            np.log1p(scale).astype(np.float32),
            aux["trend_norm"].astype(np.float32),
            aux["amp_norm"].astype(np.float32),
            aux["resid_norm"].astype(np.float32),
            aux["change_norm"].astype(np.float32),
            np.full_like(scale, float(horizon) / 10.0, dtype=np.float32),
            aux["regime_norm"].astype(np.float32),
            cache["valid_mask"].astype(np.float32),
        ]
        if uncert is not None:
            features.append((uncert["m1_std"][sample_idx, horizon_idx] / scale).astype(np.float32))
            features.append((uncert["c3_std"][sample_idx, horizon_idx] / scale).astype(np.float32))
            features.append((np.abs(uncert["c3_mean"][sample_idx, horizon_idx] - uncert["m1_mean"][sample_idx, horizon_idx]) / scale).astype(np.float32))
        feat = np.stack(features, axis=0).astype(np.float32)

        expert_preds = np.stack(
            [
                persistence,
                linear_seasonal,
                m1_pred,
                c3_pred,
            ],
            axis=0,
        ).astype(np.float32)
        sq_err = (expert_preds - target[None, :, :]) ** 2
        risk_target = ((sq_err[0] <= sq_err.min(axis=0)) | (sq_err[2] <= sq_err.min(axis=0))).astype(np.float32)

        return {
            "features": torch.from_numpy(feat).float(),
            "expert_preds": torch.from_numpy(expert_preds).float(),
            "target": torch.from_numpy(target[None, :, :]).float(),
            "persistence": torch.from_numpy(persistence[None, :, :]).float(),
            "anchor_m1": torch.from_numpy(m1_pred[None, :, :]).float(),
            "mask": torch.from_numpy(cache["valid_mask"]).float(),
            "risk_target": torch.from_numpy(risk_target[None, :, :]).float(),
            "tile_weight": torch.tensor(self.tile_weights[tile], dtype=torch.float32),
            "tile_index": torch.tensor(TILE_ORDER.index(tile), dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
            "horizon_index": torch.tensor(horizon_idx, dtype=torch.long),
        }


class AnchoredPatchSelector(nn.Module):
    def __init__(self, in_channels: int, config: V6Config) -> None:
        super().__init__()
        self.config = config
        ps = config.patch_size
        hidden = config.hidden_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden, kernel_size=ps, stride=ps)
        blocks: list[nn.Module] = []
        for _ in range(max(1, config.depth - 1)):
            blocks.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.weight_head = nn.Conv2d(hidden, len(EXPERT_NAMES), kernel_size=1)
        self.alpha_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.risk_head = nn.Conv2d(hidden, 1, kernel_size=1) if config.use_risk_head else None
        with torch.no_grad():
            self.weight_head.bias.zero_()
            self.weight_head.bias[0] = config.persistence_bias
            self.weight_head.bias[2] = config.m1_bias

    def forward(
        self,
        features: torch.Tensor,
        expert_preds: torch.Tensor,
        anchor_m1: torch.Tensor,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        patch_hidden = self.patch_embed(features)
        patch_hidden = self.blocks(patch_hidden)
        weight_logits = self.weight_head(patch_hidden)
        weights_patch = F.softmax(weight_logits, dim=1)
        alpha_patch = torch.sigmoid(self.alpha_head(patch_hidden))
        risk_logits = self.risk_head(patch_hidden) if self.risk_head is not None else None
        if risk_logits is not None:
            alpha_patch = alpha_patch * (1.0 - torch.sigmoid(risk_logits))
            if threshold is not None:
                alpha_patch = torch.where(torch.sigmoid(risk_logits) > threshold, torch.zeros_like(alpha_patch), alpha_patch)

        size = features.shape[-2:]
        weights = F.interpolate(weights_patch, size=size, mode="nearest")
        alpha = F.interpolate(alpha_patch, size=size, mode="nearest")
        softmix = (weights * expert_preds).sum(dim=1, keepdim=True)
        pred = anchor_m1 + alpha * (softmix - anchor_m1)
        return pred, weights_patch, alpha_patch, risk_logits


def weighted_masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, tile_weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")[:, 0]
    weights = mask * tile_weight[:, None, None]
    return (loss * weights).sum() / (weights.sum() + 1e-6)


def weighted_patch_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = ((pred - target) ** 2)[:, 0]
    denom = mask.sum(dim=(1, 2)).clamp_min(1e-6)
    return (err * mask).sum(dim=(1, 2)) / denom


def selector_loss(
    pred: torch.Tensor,
    weights_patch: torch.Tensor,
    alpha_patch: torch.Tensor,
    risk_logits: torch.Tensor | None,
    batch: dict[str, torch.Tensor],
    config: V6Config,
) -> tuple[torch.Tensor, float]:
    target = batch["target"]
    persistence = batch["persistence"]
    anchor_m1 = batch["anchor_m1"]
    mask = batch["mask"]
    tile_weight = batch["tile_weight"]

    pred_loss = weighted_masked_smooth_l1(pred, target, mask, tile_weight)
    mse_v6 = weighted_patch_mse(pred, target, mask)
    mse_m1 = weighted_patch_mse(anchor_m1, target, mask)
    mse_p = weighted_patch_mse(persistence, target, mask)
    weight_norm = tile_weight / tile_weight.sum().clamp_min(1e-6)
    penalty_m1 = F.relu(mse_v6 - mse_m1 + config.margin_m1)
    penalty_p = F.relu(mse_v6 - mse_p + config.margin_pers)
    entropy = -(weights_patch.clamp_min(1e-6) * weights_patch.clamp_min(1e-6).log()).sum(dim=1).mean(dim=(1, 2))
    loss = pred_loss
    loss = loss + config.lambda_m1 * (weight_norm * penalty_m1).sum()
    loss = loss + config.lambda_pers * (weight_norm * penalty_p).sum()
    loss = loss + config.entropy_lambda * (weight_norm * entropy).sum()

    if config.use_risk_head and risk_logits is not None and config.risk_bce_lambda > 0:
        risk_up = F.interpolate(risk_logits, size=target.shape[-2:], mode="nearest")
        bce = F.binary_cross_entropy_with_logits(risk_up, batch["risk_target"], reduction="none")[:, 0]
        weights = mask * tile_weight[:, None, None]
        loss = loss + config.risk_bce_lambda * (bce * weights).sum() / (weights.sum() + 1e-6)

    skill = 1.0 - float((mse_v6.mean().detach().item()) / max(mse_p.mean().detach().item(), 1e-6))
    return loss, skill


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: AnchoredPatchSelector,
    loader: DataLoader,
    config: V6Config,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    scaler: amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    skills: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type="cuda", enabled=autocast_enabled):
            pred, weights_patch, alpha_patch, risk_logits = model(batch["features"], batch["expert_preds"], batch["anchor_m1"])
            loss, skill = selector_loss(pred, weights_patch, alpha_patch, risk_logits, batch, config)
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
        skills.append(float(skill))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "skill_mse": float(np.mean(skills)) if skills else float("nan"),
    }


def evaluate_global_model(
    model: AnchoredPatchSelector,
    dataset: GlobalSelectorDataset,
    config: V6Config,
    device: torch.device,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], float]:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    model.eval()
    tile_arrays: dict[str, dict[str, np.ndarray]] = {}
    for tile in dataset.tiles:
        cache = dataset.tile_caches[tile]
        n_seq = cache["y_true_mm"].shape[0]
        h, w = cache["y_true_mm"].shape[2], cache["y_true_mm"].shape[3]
        g = h // config.patch_size
        tile_arrays[tile] = {
            "y_pred_mm": np.zeros_like(cache["y_true_mm"], dtype=np.float32),
            "y_true_mm": cache["y_true_mm"].astype(np.float32),
            "persistence_mm": cache["persistence_mm"].astype(np.float32),
            "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
            "valid_mask": cache["valid_mask"].astype(np.float32),
            "regime_map": cache["regime_map"].astype(np.int16),
            "current_frames": cache["current_frames"].astype(np.int64),
            "target_indices": cache["target_indices"].astype(np.int64),
            "selector_weights_patch": np.zeros((n_seq, len(SEARCH_HORIZONS), len(EXPERT_NAMES), g, g), dtype=np.float32),
            "alpha_patch": np.zeros((n_seq, len(SEARCH_HORIZONS), g, g), dtype=np.float32),
            "risk_patch": np.zeros((n_seq, len(SEARCH_HORIZONS), g, g), dtype=np.float32),
        }

    start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = move_batch(batch, device)
            pred, weights_patch, alpha_patch, risk_logits = model(
                batch_gpu["features"],
                batch_gpu["expert_preds"],
                batch_gpu["anchor_m1"],
                threshold=threshold,
            )
            pred_np = pred[:, 0].cpu().numpy().astype(np.float32)
            weights_np = weights_patch.cpu().numpy().astype(np.float32)
            alpha_np = alpha_patch[:, 0].cpu().numpy().astype(np.float32)
            risk_np = torch.sigmoid(risk_logits[:, 0]).cpu().numpy().astype(np.float32) if risk_logits is not None else np.zeros_like(alpha_np)
            tile_idx = batch["tile_index"].cpu().numpy().astype(np.int64)
            sample_idx = batch["sample_index"].cpu().numpy().astype(np.int64)
            horizon_idx = batch["horizon_index"].cpu().numpy().astype(np.int64)
            for i in range(len(tile_idx)):
                tile = TILE_ORDER[tile_idx[i]]
                tile_arrays[tile]["y_pred_mm"][sample_idx[i], horizon_idx[i]] = pred_np[i]
                tile_arrays[tile]["selector_weights_patch"][sample_idx[i], horizon_idx[i]] = weights_np[i]
                tile_arrays[tile]["alpha_patch"][sample_idx[i], horizon_idx[i]] = alpha_np[i]
                tile_arrays[tile]["risk_patch"][sample_idx[i], horizon_idx[i]] = risk_np[i]
    infer_seconds = time.time() - start

    rows: list[dict[str, Any]] = []
    for tile in dataset.tiles:
        rows.extend(
            result_rows(
                model_id=config.model_id,
                tile=tile,
                seed=config.seed,
                phase=dataset.split_name,
                elapsed_seconds=0.0,
                infer_seconds=infer_seconds,
                best_epoch=-1,
                param_count=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                arrays=tile_arrays[tile],
                horizons=SEARCH_HORIZONS,
            )
        )
    return pd.DataFrame(rows), tile_arrays, infer_seconds


def val_score_from_rows(rows: pd.DataFrame) -> dict[str, Any]:
    mean_skill = float(rows["Skill_MSE"].mean())
    stress = rows[rows["Tile"].isin(["E48N24", "E58N17"])]
    stress_ratio = float((stress["RMSE [mm]"] / stress["Persistence RMSE [mm]"]).max()) if not stress.empty else 1.0
    tile_means = rows.groupby("Tile", as_index=False)["Skill_MSE"].mean()
    tile_std = float(tile_means["Skill_MSE"].std(ddof=0))
    score = mean_skill - 0.2 * max(0.0, stress_ratio - 1.0) - 0.1 * tile_std
    return {
        "score": score,
        "mean_skill_mse": mean_skill,
        "stress_ratio": stress_ratio,
        "tile_skill_std": tile_std,
    }


def train_global_selector(
    config: V6Config,
    phase: str,
    train_tiles: list[str],
    eval_tiles: list[str],
    reuse_if_exists: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(ACT_V6_ROOT / "runs" / phase / config.model_id / f"seed_{config.seed}")
    result_path = output_dir / "result.json"
    if reuse_if_exists and result_path.exists():
        return pd.read_json(result_path, typ="series").to_dict()

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = GlobalSelectorDataset(train_tiles, "train", config)
    val_ds = GlobalSelectorDataset(train_tiles, "val", config)
    test_ds = GlobalSelectorDataset(eval_tiles, "test", config)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    model = AnchoredPatchSelector(train_ds[0]["features"].shape[0], config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, Any]] = []
    ckpt_path = output_dir / "model.pth"
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, config, device, optimizer=optimizer, scaler=scaler)
        val_rows, _val_arrays, _ = evaluate_global_model(model, val_ds, config, device)
        val_score = val_score_from_rows(val_rows)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_skill_mse": train_stats["skill_mse"],
                "val_mean_skill_mse": val_score["mean_skill_mse"],
                "val_stress_ratio": val_score["stress_ratio"],
                "val_tile_skill_std": val_score["tile_skill_std"],
                "val_score": val_score["score"],
            }
        )
        if val_score["score"] > best_score:
            best_score = val_score["score"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "config": asdict(config)}, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    threshold = None
    if config.use_risk_head:
        val_rows_best, _val_arrays_best, _ = evaluate_global_model(model, val_ds, config, device, threshold=None)
        base_score = val_score_from_rows(val_rows_best)["score"]
        best_threshold = None
        best_threshold_score = base_score
        for threshold_candidate in config.risk_thresholds:
            rows_thr, _arrays_thr, _ = evaluate_global_model(model, val_ds, config, device, threshold=threshold_candidate)
            score_thr = val_score_from_rows(rows_thr)["score"]
            if score_thr > best_threshold_score:
                best_threshold_score = score_thr
                best_threshold = threshold_candidate
        threshold = best_threshold

    test_rows, test_arrays, infer_seconds = evaluate_global_model(model, test_ds, config, device, threshold=threshold)
    elapsed_seconds = time.time() - start_time
    test_rows = test_rows.copy()
    test_rows["Elapsed seconds"] = elapsed_seconds
    test_rows["Infer seconds"] = infer_seconds
    test_rows["Best epoch"] = best_epoch
    test_rows["Param count"] = param_count

    tile_array_dir = ensure_dir(output_dir / "tile_arrays")
    for tile, arrays in test_arrays.items():
        np.savez_compressed(tile_array_dir / f"{tile}.npz", **arrays)

    result = {
        "model_id": config.model_id,
        "variant": config.variant,
        "phase": phase,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "risk_threshold": threshold,
        "elapsed_seconds": elapsed_seconds,
        "infer_seconds": infer_seconds,
        "param_count": param_count,
        "history": history,
        "rows": test_rows.to_dict("records"),
        "mean_skill_mse": float(test_rows["Skill_MSE"].mean()),
        "train_tiles": train_tiles,
        "eval_tiles": eval_tiles,
    }
    write_json(result_path, result)
    return result


def save_rows(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def run_single_seed_candidates(configs: list[V6Config], phase: str, path: Path, reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and path.exists():
        return save_rows(backfill_runtime_columns(pd.read_csv(path), phase), path)
    rows: list[dict[str, Any]] = []
    for config in configs:
        write_simple_yaml(ACT_V6_CONFIG_ROOT / f"{config.model_id}.yaml", asdict(config))
        print(f"[{phase}] start model={config.model_id} seed={config.seed}", flush=True)
        result = train_global_selector(config, phase=phase, train_tiles=TILE_ORDER, eval_tiles=TILE_ORDER, reuse_if_exists=reuse_if_exists)
        print(f"[{phase}] done model={config.model_id} mean_skill={result['mean_skill_mse']:.4f}", flush=True)
        rows.extend(result["rows"])
    return save_rows(backfill_runtime_columns(pd.DataFrame(rows), phase), path)


def candidate_summary(df: pd.DataFrame) -> pd.DataFrame:
    return summarize_seeded_rows(df)


def candidate_mean_skill(summary_df: pd.DataFrame, model_id: str) -> float:
    return float(summary_df[summary_df["Model"] == model_id]["Skill_MSE mean"].mean())


def choose_best_candidate(v6a: pd.DataFrame, v6b: pd.DataFrame, v6c: pd.DataFrame) -> tuple[V6Config, dict[str, Any]]:
    configs = default_v6_configs()
    summaries = {
        "v6a": candidate_summary(v6a),
        "v6b": candidate_summary(v6b),
        "v6c": candidate_summary(v6c),
    }
    scores: list[tuple[float, V6Config]] = []
    mapping: dict[str, V6Config] = {configs["v6a"].model_id: configs["v6a"], configs["v6c"].model_id: configs["v6c"]}  # type: ignore[index]
    for cfg in configs["v6b"]:  # type: ignore[index]
        mapping[cfg.model_id] = cfg
    combined = pd.concat([summaries["v6a"], summaries["v6b"], summaries["v6c"]], ignore_index=True)
    for model_id, cfg in mapping.items():
        sub = combined[combined["Model"] == model_id]
        if sub.empty:
            continue
        mean_skill = float(sub["Skill_MSE mean"].mean())
        stress = sub[sub["Tile"].isin(["E48N24", "E58N17"])]
        stress_ratio = float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()) if not stress.empty else 1.0
        tile_comp = sub.groupby("Tile", as_index=False)["Skill_MSE mean"].mean()
        score = mean_skill - 0.2 * max(0.0, stress_ratio - 1.0) - 0.05 * float(tile_comp["Skill_MSE mean"].std(ddof=0))
        scores.append((score, cfg))
    scores.sort(key=lambda item: item[0], reverse=True)
    best_cfg = scores[0][1]
    stats = {"best_model_id": best_cfg.model_id, "best_score": float(scores[0][0])}
    return best_cfg, stats


def run_best_three_seed(config: V6Config, reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and V6_BEST_3SEED_PATH.exists():
        return save_rows(backfill_runtime_columns(pd.read_csv(V6_BEST_3SEED_PATH), "v6_best_3seed"), V6_BEST_3SEED_PATH)
    rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(config, seed=seed)
        print(f"[v6_best_3seed] start model={cfg.model_id} seed={seed}", flush=True)
        result = train_global_selector(cfg, phase="v6_best_3seed", train_tiles=TILE_ORDER, eval_tiles=TILE_ORDER, reuse_if_exists=reuse_if_exists)
        print(f"[v6_best_3seed] done model={cfg.model_id} seed={seed} mean_skill={result['mean_skill_mse']:.4f}", flush=True)
        rows.extend(result["rows"])
    return save_rows(backfill_runtime_columns(pd.DataFrame(rows), "v6_best_3seed"), V6_BEST_3SEED_PATH)


def v6_best_verdict(summary_df: pd.DataFrame) -> dict[str, Any]:
    model_id = summary_df["Model"].iloc[0]
    mean_skill = float(summary_df["Skill_MSE mean"].mean())
    mean_std = float(summary_df["Skill_MSE std"].mean())
    m1_summary = load_m1_full6_summary()
    merged = summary_df.merge(
        m1_summary[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs M1"] = merged["Skill_MSE mean"] - merged["M1 Skill_MSE mean"]
    tile_compare = merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "M1 Skill_MSE mean", "Delta vs M1"]].mean().rename(columns={"Skill_MSE mean": "V6 Skill_MSE mean"})
    better_tiles = int((tile_compare["Delta vs M1"] > 0).sum())
    stress = merged[merged["Tile"].isin(["E48N24", "E58N17"])]
    stress_ratio = float((stress["RMSE [mm] mean"] / stress["Persistence RMSE [mm] mean"]).max()) if not stress.empty else 1.0
    mean_train = mean_train_seconds_from_results("v6_best_3seed", model_id, FINAL_SEEDS)
    gain = mean_skill - M1_MEAN_SKILL

    if mean_skill >= 0.105 and gain >= 0.025 and mean_std <= 0.020 and better_tiles >= 5 and stress_ratio <= STRESS_RATIO_THRESHOLD and mean_train <= 15.0:
        status = "strong_go"
    elif mean_skill >= 0.100 and gain >= 0.020 and mean_std <= 0.025 and better_tiles >= 5 and stress_ratio <= STRESS_RATIO_THRESHOLD:
        status = "practical_go"
    elif mean_skill > V5_MEAN_SKILL or gain >= 0.018:
        status = "near_pass"
    else:
        status = "no_go"
    verdict = {
        "status": status,
        "model_id": model_id,
        "mean_skill_mse": mean_skill,
        "mean_skill_std": mean_std,
        "gain_vs_m1": gain,
        "better_tiles_vs_m1": better_tiles,
        "stress_ratio": stress_ratio,
        "mean_train_seconds": mean_train,
        "tile_compare": tile_compare.to_dict("records"),
    }
    write_json(V6_BEST_VERDICT_PATH, {k: v for k, v in verdict.items() if k != "tile_compare"})
    tile_compare.to_csv(ACT_V6_ROOT / "v6_best_tile_compare.csv", index=False, encoding="utf-8-sig")
    return verdict


def selector_diagnostics(best_config: V6Config, seed: int = 42, reuse_if_exists: bool = True) -> pd.DataFrame:
    summary_path = V6_DIAG_ROOT / "v6_selector_weight_alpha_summary.csv"
    if reuse_if_exists and summary_path.exists():
        return pd.read_csv(summary_path)

    run_dir = ACT_V6_ROOT / "runs" / "v6_best_3seed" / best_config.model_id / f"seed_{seed}" / "tile_arrays"
    rows: list[dict[str, Any]] = []
    ensure_dir(V6_DIAG_ROOT)
    rep_tiles = ["E39N30", "E48N24", "E44N23"]
    for tile in rep_tiles:
        arr = np.load(run_dir / f"{tile}.npz")
        weights = arr["selector_weights_patch"]
        alpha = arr["alpha_patch"]
        risk = arr["risk_patch"]
        for h_idx, horizon in enumerate(SEARCH_HORIZONS):
            for expert_idx, expert in enumerate(EXPERT_NAMES):
                rows.append(
                    {
                        "Tile": tile,
                        "Horizon": f"t+{horizon}",
                        "Expert": expert,
                        "Mean weight": float(weights[:, h_idx, expert_idx].mean()),
                        "Mean alpha": float(alpha[:, h_idx].mean()),
                        "Mean risk": float(risk[:, h_idx].mean()),
                    }
                )
        seq_idx = 0
        fig, axes = plt.subplots(3, len(SEARCH_HORIZONS), figsize=(12, 6))
        for col, horizon in enumerate(SEARCH_HORIZONS):
            axes[0, col].imshow(weights[seq_idx, col, 2], cmap="viridis")
            axes[0, col].set_title(f"{tile} t+{horizon} M1 w")
            axes[1, col].imshow(weights[seq_idx, col, 3], cmap="viridis")
            axes[1, col].set_title(f"{tile} t+{horizon} C3 w")
            axes[2, col].imshow(alpha[seq_idx, col], cmap="magma")
            axes[2, col].set_title(f"{tile} t+{horizon} alpha")
            for row in range(3):
                axes[row, col].axis("off")
        fig.tight_layout()
        fig.savefig(V6_DIAG_ROOT / f"{tile}_weight_alpha_maps.png", dpi=200)
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return df


def run_ablation(best_config: V6Config, reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and V6_ABLATION_PATH.exists():
        return pd.read_csv(V6_ABLATION_PATH)
    variants = [
        replace(best_config, model_id="V6E_no_uncert", use_uncertainty=False),
        replace(best_config, model_id="V6E_no_anchor", lambda_m1=0.0, lambda_pers=0.0, use_risk_head=False, risk_bce_lambda=0.0, persistence_bias=0.0, m1_bias=0.0),
        replace(best_config, model_id="V6E_no_risk", lambda_m1=0.0, lambda_pers=0.0, use_risk_head=False, risk_bce_lambda=0.0),
        replace(best_config, model_id="V6E_no_C3", use_uncertainty=False),
        replace(best_config, model_id="V6E_no_persistence", use_uncertainty=best_config.use_uncertainty),
        replace(best_config, model_id="V6E_patch4", patch_size=4),
    ]
    rows: list[dict[str, Any]] = []
    for cfg in variants:
        print(f"[v6_ablation] start model={cfg.model_id}", flush=True)
        result = train_global_selector(cfg, phase="v6_ablation", train_tiles=TILE_ORDER, eval_tiles=TILE_ORDER, reuse_if_exists=reuse_if_exists)
        print(f"[v6_ablation] done model={cfg.model_id} mean_skill={result['mean_skill_mse']:.4f}", flush=True)
        rows.extend(result["rows"])
    return save_rows(pd.DataFrame(rows), V6_ABLATION_PATH)


def run_leave_one_tile_out(best_config: V6Config, reuse_if_exists: bool = True) -> pd.DataFrame:
    out_path = ACT_V6_ROOT / "v6d_leave_one_tile_out.csv"
    if reuse_if_exists and out_path.exists():
        return pd.read_csv(out_path)
    rows: list[dict[str, Any]] = []
    for held_out in TILE_ORDER:
        train_tiles = [tile for tile in TILE_ORDER if tile != held_out]
        cfg = replace(best_config, model_id=f"V6D_LOTO_{held_out}", seed=42)
        print(f"[v6_robust] start held_out={held_out}", flush=True)
        result = train_global_selector(cfg, phase="v6_robust", train_tiles=train_tiles, eval_tiles=[held_out], reuse_if_exists=reuse_if_exists)
        print(f"[v6_robust] done held_out={held_out} mean_skill={result['mean_skill_mse']:.4f}", flush=True)
        rows.extend(result["rows"])
    return save_rows(pd.DataFrame(rows), out_path)


def build_oracle_breakdown(reuse_if_exists: bool = True) -> pd.DataFrame:
    if reuse_if_exists and ORACLE_BREAKDOWN_PATH.exists():
        return pd.read_csv(ORACLE_BREAKDOWN_PATH)
    v5_oracle = pd.read_csv(ACT_V5_ROOT / "v5_oracle_table.csv")
    wins = pd.read_csv(ORACLE_WIN_TABLE_PATH)
    patch = v5_oracle[v5_oracle["Model"] == "V5S0_patch_oracle"].copy()
    m1 = v5_oracle[v5_oracle["Model"] == "M1_expert_seeded"][["Tile", "Horizon", "Skill_MSE"]].rename(columns={"Skill_MSE": "M1 Skill_MSE"})
    patch = patch.merge(m1, on=["Tile", "Horizon"], how="left")
    patch["Oracle gain vs M1"] = patch["Skill_MSE"] - patch["M1 Skill_MSE"]
    patch["Breakdown"] = "tile_horizon_patch"
    wins["Breakdown"] = "regime_win_rate"
    wins["Oracle gain vs M1"] = np.nan
    combined = pd.concat([patch, wins], ignore_index=True, sort=False)
    combined.to_csv(ORACLE_BREAKDOWN_PATH, index=False, encoding="utf-8-sig")

    pivot = patch.pivot(index="Tile", columns="Horizon", values="Oracle gain vs M1").reindex(TILE_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("V6 Patch Oracle Gain vs seeded-M1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ORACLE_PATCH_FIGURE, dpi=200)
    plt.close(fig)
    return combined


def write_final_decision(
    oracle_df: pd.DataFrame,
    v6a_summary: pd.DataFrame,
    v6b_summary: pd.DataFrame,
    v6c_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    verdict: dict[str, Any],
    robustness_df: pd.DataFrame | None,
    ablation_df: pd.DataFrame | None,
) -> Path:
    lines = [
        "# V6 Final Decision",
        "",
        "数据声明：本轮继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile。",
        "",
        "## Oracle",
        "",
        f"- patch oracle mean gain vs seeded-M1 = {float(oracle_df[oracle_df['Breakdown'] == 'tile_horizon_patch']['Oracle gain vs M1'].mean()):.4f}",
        "",
        "## Candidate results",
        "",
        f"- V6-A mean Skill_MSE = {float(v6a_summary['Skill_MSE mean'].mean()):.4f}",
        f"- V6-B best single-seed mean Skill_MSE = {float(v6b_summary.groupby('Model')['Skill_MSE mean'].mean().max()):.4f}",
        f"- V6-C single-seed mean Skill_MSE = {float(v6c_summary['Skill_MSE mean'].mean()):.4f}",
        f"- Best V6 3-seed mean Skill_MSE = {float(best_summary['Skill_MSE mean'].mean()):.4f}",
        f"- Best V6 gain vs M1 = {verdict['gain_vs_m1']:.4f}",
        f"- Best V6 better tiles vs M1 = {verdict['better_tiles_vs_m1']} / 6",
        f"- Best V6 stress ratio = {verdict['stress_ratio']:.4f}",
        f"- Best V6 mean train seconds = {verdict['mean_train_seconds']:.2f}",
        f"- Best V6 status = `{verdict['status']}`",
        "",
        "## Conclusion",
        "",
    ]
    if verdict["status"] in {"strong_go", "practical_go"}:
        lines.append("- V6 达到替代 M1 的门槛，可作为 TGRS 主模型。")
    elif verdict["status"] == "near_pass":
        lines.append("- V6 属于 near-pass：相对 V5-D 已改善，但尚未满足替代 M1 的全部门槛。")
    else:
        lines.append("- V6 未达到替代 M1 的门槛，模型探索应停止，主模型保持 M1。")
    if robustness_df is not None and not robustness_df.empty:
        lines.extend(["", "- 已附加 leave-one-tile-out robustness diagnostic。"])
    if ablation_df is not None and not ablation_df.empty:
        lines.extend(["", "- 已附加 V6-E ablation。"])
    lines.append("")
    V6_FINAL_DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    return V6_FINAL_DECISION_PATH


def write_report(
    oracle_df: pd.DataFrame,
    v6a_summary: pd.DataFrame,
    v6b_summary: pd.DataFrame,
    v6c_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    verdict: dict[str, Any],
    selector_diag: pd.DataFrame | None,
    robustness_df: pd.DataFrame | None,
    ablation_df: pd.DataFrame | None,
) -> Path:
    patch_oracle = oracle_df[oracle_df["Breakdown"] == "tile_horizon_patch"]
    v6b_rank = v6b_summary.groupby("Model", as_index=False)["Skill_MSE mean"].mean().sort_values("Skill_MSE mean", ascending=False)
    lines = [
        "# ACT-Former V6 风险校准锚定软选择器实验报告",
        "",
        "数据声明：本轮所有实验继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 方法设定",
        "",
        "- V6 不再开发新 backbone，而是在 V5 expert cache 上训练 **global patch-level anchored soft selector**。",
        "- 预测形式为 `y_hat = M1 + alpha * (softmix(experts) - M1)`，其中 `alpha` 是 risk-calibrated anchor gate。",
        "- V6-A 只用 anchored soft selector；V6-B 加 risk fallback penalties；V6-C 进一步加入 M1/C3 3-seed uncertainty 特征。",
        "- 全部 selector 以单个 global model 在六个 tile 上联合训练，loss 使用 tile-balanced weighting。",
        "",
        "## 1. V6-0 Oracle breakdown",
        "",
        f"- patch oracle mean gain vs seeded-M1 = {float(patch_oracle['Oracle gain vs M1'].mean()):.4f}",
        f"- patch oracle positive tiles = {int(patch_oracle.groupby('Tile')['Oracle gain vs M1'].mean().gt(0).sum())} / 6",
        "",
        "## 2. V6-A / V6-B / V6-C single-seed full-six",
        "",
        f"- V6-A mean Skill_MSE = {float(v6a_summary['Skill_MSE mean'].mean()):.4f}",
        f"- V6-B variants best mean Skill_MSE = {float(v6b_rank.iloc[0]['Skill_MSE mean']):.4f}",
        f"- V6-C mean Skill_MSE = {float(v6c_summary['Skill_MSE mean'].mean()):.4f}",
        "",
        "V6-B variant ranking:",
    ]
    for _, row in v6b_rank.iterrows():
        lines.append(f"- {row['Model']}: mean Skill_MSE = {row['Skill_MSE mean']:.4f}")
    lines.extend(
        [
            "",
            "## 3. Best V6 3-seed full-six",
            "",
            f"- model = `{verdict['model_id']}`",
            f"- mean Skill_MSE = {verdict['mean_skill_mse']:.4f}",
            f"- mean Skill_MSE std = {verdict['mean_skill_std']:.4f}",
            f"- gain vs M1 = {verdict['gain_vs_m1']:.4f}",
            f"- better tiles vs M1 = {verdict['better_tiles_vs_m1']} / 6",
            f"- stress ratio = {verdict['stress_ratio']:.4f}",
            f"- mean train seconds = {verdict['mean_train_seconds']:.2f}",
            f"- verdict = `{verdict['status']}`",
            "",
        ]
    )
    if selector_diag is not None and not selector_diag.empty:
        lines.extend(
            [
                "## 4. Selector diagnostics",
                "",
                f"- selector diagnostics summary: `{(V6_DIAG_ROOT / 'v6_selector_weight_alpha_summary.csv').as_posix()}`",
                "",
            ]
        )
    if robustness_df is not None and not robustness_df.empty:
        lines.extend(
            [
                "## 5. Robustness diagnostic",
                "",
                f"- leave-one-tile-out summary: `{(ACT_V6_ROOT / 'v6d_leave_one_tile_out.csv').as_posix()}`",
                "",
            ]
        )
    if ablation_df is not None and not ablation_df.empty:
        lines.extend(
            [
                "## 6. Ablation",
                "",
                f"- ablation results: `{V6_ABLATION_PATH.as_posix()}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果文件",
            "",
            f"- oracle breakdown: `{ORACLE_BREAKDOWN_PATH.as_posix()}`",
            f"- V6-A results: `{V6A_RESULTS_PATH.as_posix()}`",
            f"- V6-B results: `{V6B_RESULTS_PATH.as_posix()}`",
            f"- V6-C results: `{V6C_RESULTS_PATH.as_posix()}`",
            f"- best 3-seed: `{V6_BEST_3SEED_PATH.as_posix()}`",
            f"- best verdict: `{V6_BEST_VERDICT_PATH.as_posix()}`",
            f"- final decision: `{V6_FINAL_DECISION_PATH.as_posix()}`",
            "",
        ]
    )
    V6_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return V6_REPORT_PATH


def run_act_v6_exploration(reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(ACT_V6_ROOT)
    ensure_dir(ACT_V6_CONFIG_ROOT)
    ensure_dir(UNCERTAINTY_CACHE_ROOT)
    ensure_dir(V6_DIAG_ROOT)
    bundles = preprocess_all_tiles()
    ensure_seed_manifest(reuse_if_exists=reuse_if_exists)
    ensure_expert_caches(bundles, reuse_if_exists=reuse_if_exists)
    ensure_uncertainty_caches(bundles, reuse_if_exists=reuse_if_exists)
    oracle_df = build_oracle_breakdown(reuse_if_exists=reuse_if_exists)

    cfgs = default_v6_configs()
    v6a_rows = run_single_seed_candidates([cfgs["v6a"]], phase="v6a", path=V6A_RESULTS_PATH, reuse_if_exists=reuse_if_exists)  # type: ignore[list-item]
    v6b_rows = run_single_seed_candidates(cfgs["v6b"], phase="v6b", path=V6B_RESULTS_PATH, reuse_if_exists=reuse_if_exists)  # type: ignore[arg-type]
    v6c_rows = run_single_seed_candidates([cfgs["v6c"]], phase="v6c", path=V6C_RESULTS_PATH, reuse_if_exists=reuse_if_exists)  # type: ignore[list-item]

    v6a_summary = candidate_summary(v6a_rows)
    v6b_summary = candidate_summary(v6b_rows)
    v6c_summary = candidate_summary(v6c_rows)
    save_rows(v6a_summary, ACT_V6_ROOT / "v6a_summary.csv")
    save_rows(v6b_summary, ACT_V6_ROOT / "v6b_summary.csv")
    save_rows(v6c_summary, ACT_V6_ROOT / "v6c_summary.csv")

    best_config, selection_stats = choose_best_candidate(v6a_rows, v6b_rows, v6c_rows)
    best_rows = run_best_three_seed(best_config, reuse_if_exists=reuse_if_exists)
    best_summary = candidate_summary(best_rows)
    save_rows(best_summary, ACT_V6_ROOT / "v6_best_3seed_summary.csv")
    verdict = v6_best_verdict(best_summary)

    selector_diag: pd.DataFrame | None = None
    robustness_df: pd.DataFrame | None = None
    ablation_df: pd.DataFrame | None = None
    if verdict["status"] in {"strong_go", "practical_go", "near_pass"}:
        selector_diag = selector_diagnostics(best_config, seed=42, reuse_if_exists=reuse_if_exists)
        robustness_df = run_leave_one_tile_out(best_config, reuse_if_exists=reuse_if_exists)
    if verdict["status"] in {"strong_go", "practical_go"}:
        ablation_df = run_ablation(best_config, reuse_if_exists=reuse_if_exists)

    final_decision = write_final_decision(
        oracle_df=oracle_df,
        v6a_summary=v6a_summary,
        v6b_summary=v6b_summary,
        v6c_summary=v6c_summary,
        best_summary=best_summary,
        verdict=verdict,
        robustness_df=robustness_df,
        ablation_df=ablation_df,
    )
    report = write_report(
        oracle_df=oracle_df,
        v6a_summary=v6a_summary,
        v6b_summary=v6b_summary,
        v6c_summary=v6c_summary,
        best_summary=best_summary,
        verdict=verdict,
        selector_diag=selector_diag,
        robustness_df=robustness_df,
        ablation_df=ablation_df,
    )

    summary = {
        "report": str(report.resolve()),
        "final_decision": str(final_decision.resolve()),
        "selection_stats": selection_stats,
        "best_verdict": {k: v for k, v in verdict.items() if k != "tile_compare"},
        "v6a_mean_skill": float(v6a_summary["Skill_MSE mean"].mean()),
        "v6b_best_mean_skill": float(v6b_summary.groupby("Model")["Skill_MSE mean"].mean().max()),
        "v6c_mean_skill": float(v6c_summary["Skill_MSE mean"].mean()),
        "oracle_path": str(ORACLE_BREAKDOWN_PATH.resolve()),
        "best_3seed_path": str(V6_BEST_3SEED_PATH.resolve()),
        "robustness_path": str((ACT_V6_ROOT / "v6d_leave_one_tile_out.csv").resolve()) if robustness_df is not None else None,
        "ablation_path": str(V6_ABLATION_PATH.resolve()) if ablation_df is not None else None,
    }
    write_json(V6_SUMMARY_PATH, summary)
    return summary
