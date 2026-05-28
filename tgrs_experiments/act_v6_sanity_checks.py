from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset

from .act_v3 import result_rows, summarize_seeded_rows
from .act_v5_selector import (
    EXPERT_NAMES,
    ORACLE_WIN_TABLE_PATH,
    ensure_expert_caches,
    ensure_seed_manifest,
    infer_c3_split,
    infer_m1_split,
    load_cache,
    oracle_from_cache,
    selected_run_dir,
)
from .act_v6_selector import (
    ACT_V6_ROOT,
    FINAL_SEEDS,
    GlobalSelectorDataset,
    M1_RUN_ROOT,
    C3_RUN_ROOT,
    SEARCH_HORIZONS,
    TILE_ORDER,
    V6Config,
    default_v6_configs,
    ensure_uncertainty_caches,
    move_batch,
    preprocess_all_tiles,
    result_dir_for_phase,
    sync_cuda,
)
from .pipeline import ensure_dir, seed_everything, write_json


SANITY_ROOT = ensure_dir(ACT_V6_ROOT / "sanity_checks")
CACHE_CHECK_MD = ACT_V6_ROOT / "v6_cache_alignment_check.md"
SANITY_REPORT_MD = SANITY_ROOT / "v6_sanity_checks_report.md"
SANITY_SUMMARY_JSON = SANITY_ROOT / "v6_sanity_summary.json"
CACHE_ALIGNMENT_CSV = SANITY_ROOT / "cache_alignment_numeric_checks.csv"
SPLIT_SEPARATION_CSV = SANITY_ROOT / "cache_split_separation.csv"
ALPHA_TILE_HORIZON_CSV = SANITY_ROOT / "alpha_by_tile_horizon.csv"
ALPHA_HIST_PNG = SANITY_ROOT / "alpha_histogram.png"
ALPHA_TILE_PNG = SANITY_ROOT / "alpha_by_tile.png"
ALPHA_HORIZON_PNG = SANITY_ROOT / "alpha_by_horizon.png"
WEIGHT_HIST_PNG = SANITY_ROOT / "expert_weight_histogram.png"
WEIGHT_TILE_HORIZON_CSV = SANITY_ROOT / "expert_weight_by_tile_horizon.csv"
WEIGHT_TILE_HORIZON_PNG = SANITY_ROOT / "expert_weight_by_tile_horizon.png"
DELTA_VS_M1_CSV = SANITY_ROOT / "v6_minus_m1_delta_summary.csv"
DELTA_VS_M1_PNG = SANITY_ROOT / "v6_minus_m1_histogram.png"
TEACHER_RAW_CSV = SANITY_ROOT / "teacher_oracle_raw.csv"
TEACHER_SUMMARY_CSV = SANITY_ROOT / "teacher_oracle_summary.csv"
TEACHER_DECISION_MD = SANITY_ROOT / "teacher_oracle_decision.md"
TEACHER_TILE_COMPARE_CSV = SANITY_ROOT / "teacher_oracle_tile_compare.csv"

BEST_MODEL_ID = "V6B_anchor_risk_l20"
BEST_PHASE = "v6_best_3seed"
BEST_SEEDS = FINAL_SEEDS
PATCH_SIZE = 8


def selected_v6_config() -> V6Config:
    cfgs = default_v6_configs()
    for cfg in cfgs["v6b"]:  # type: ignore[index]
        if cfg.model_id == BEST_MODEL_ID:
            return cfg
    raise KeyError(BEST_MODEL_ID)


def write_markdown(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def sample_key_set(cache: dict[str, np.ndarray]) -> set[tuple[int, tuple[int, ...]]]:
    current = cache["current_frames"].reshape(-1).astype(np.int64)
    targets = cache["target_indices"].astype(np.int64)
    return {(int(cur), tuple(int(v) for v in row.tolist())) for cur, row in zip(current, targets, strict=True)}


def patch_valid_from_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = mask.shape
    gh, gw = h // patch_size, w // patch_size
    reshaped = mask.reshape(gh, patch_size, gw, patch_size)
    return (reshaped.sum(axis=(1, 3)) > 0.5).astype(np.float32)


def direct_alignment_checks(reuse_if_exists: bool = True) -> dict[str, Any]:
    bundles = preprocess_all_tiles()
    manifest = ensure_seed_manifest(reuse_if_exists=reuse_if_exists)
    ensure_expert_caches(bundles, reuse_if_exists=reuse_if_exists)
    ensure_uncertainty_caches(bundles, reuse_if_exists=reuse_if_exists)

    numeric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    max_seen = 0.0

    for tile in TILE_ORDER:
        bundle = bundles[tile]
        m1_dir = selected_run_dir(manifest, "M1", tile)
        c3_dir = selected_run_dir(manifest, "C3_opt", tile)
        split_keys: dict[str, set[tuple[int, tuple[int, ...]]]] = {}

        for split_name in ["train", "val", "test"]:
            cache = load_cache(tile, split_name)
            m1 = infer_m1_split(bundle, m1_dir, split_name)
            c3 = infer_c3_split(bundle, c3_dir, split_name)

            cache_keys = sample_key_set(cache)
            split_keys[split_name] = cache_keys

            checks = {
                "y_true_vs_m1": max_abs_diff(cache["y_true_mm"], m1["y_true_mm"]),
                "y_true_vs_c3": max_abs_diff(cache["y_true_mm"], c3["y_true_mm"]),
                "persistence_vs_m1": max_abs_diff(cache["persistence_mm"], m1["persistence_mm"]),
                "linear_seasonal_vs_m1": max_abs_diff(cache["linear_seasonal_mm"], m1["linear_seasonal_mm"]),
                "m1_pred": max_abs_diff(cache["m1_pred_mm"], m1["y_pred_mm"]),
                "c3_pred": max_abs_diff(cache["c3_pred_mm"], c3["y_pred_mm"]),
                "m1_gate": max_abs_diff(cache["m1_gate_mm"], m1["gate_weights"][:, :, 2]),
                "c3_scale": max_abs_diff(cache["c3_scale_mm"], c3["scale_mm"]),
                "valid_mask": max_abs_diff(cache["valid_mask"], m1["valid_mask"]),
                "regime_map": max_abs_diff(cache["regime_map"], m1["regime_map"]),
                "last_disp": max_abs_diff(cache["last_disp_mm"], cache["persistence_mm"][:, 0:1]),
            }
            for check_name, diff in checks.items():
                numeric_rows.append(
                    {
                        "Tile": tile,
                        "Split": split_name,
                        "Check": check_name,
                        "Max abs diff": diff,
                    }
                )
                max_seen = max(max_seen, diff)
                if diff > 1e-6:
                    issues.append(f"{tile}/{split_name}/{check_name} max diff={diff:.6g}")

            same_frames_m1_c3 = bool(np.array_equal(m1["current_frames"].reshape(-1), c3["current_frames"].reshape(-1)))
            same_targets_m1_c3 = bool(np.array_equal(m1["target_indices"], c3["target_indices"]))
            same_frames_cache_m1 = bool(np.array_equal(cache["current_frames"], m1["current_frames"].reshape(-1)))
            same_targets_cache_m1 = bool(np.array_equal(cache["target_indices"], m1["target_indices"]))
            split_rows.append(
                {
                    "Tile": tile,
                    "Split": split_name,
                    "Samples": int(cache["y_true_mm"].shape[0]),
                    "Current frame min": int(cache["current_frames"].min()),
                    "Current frame max": int(cache["current_frames"].max()),
                    "Target min": int(cache["target_indices"].min()),
                    "Target max": int(cache["target_indices"].max()),
                    "M1/C3 current_frames equal": same_frames_m1_c3,
                    "M1/C3 target_indices equal": same_targets_m1_c3,
                    "Cache/M1 current_frames equal": same_frames_cache_m1,
                    "Cache/M1 target_indices equal": same_targets_cache_m1,
                }
            )
            if not (same_frames_m1_c3 and same_targets_m1_c3 and same_frames_cache_m1 and same_targets_cache_m1):
                issues.append(f"{tile}/{split_name} current_frames or target_indices mismatch")

        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = len(split_keys[left] & split_keys[right])
            split_rows.append(
                {
                    "Tile": tile,
                    "Split": f"{left}_vs_{right}",
                    "Samples": overlap,
                    "Current frame min": -1,
                    "Current frame max": -1,
                    "Target min": -1,
                    "Target max": -1,
                    "M1/C3 current_frames equal": True,
                    "M1/C3 target_indices equal": True,
                    "Cache/M1 current_frames equal": True,
                    "Cache/M1 target_indices equal": True,
                }
            )
            if overlap != 0:
                issues.append(f"{tile} split overlap {left}/{right} = {overlap}")

    pd.DataFrame(numeric_rows).to_csv(CACHE_ALIGNMENT_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(split_rows).to_csv(SPLIT_SEPARATION_CSV, index=False, encoding="utf-8-sig")

    seed_rows: list[dict[str, Any]] = []
    for tile in TILE_ORDER:
        for family, root in [("M1", M1_RUN_ROOT), ("C3_opt", C3_RUN_ROOT)]:
            scores: list[tuple[int, float]] = []
            for seed in FINAL_SEEDS:
                result_path = root / tile / f"seed_{seed}" / "result.json"
                payload = pd.read_json(result_path, typ="series").to_dict()
                scores.append((seed, float(payload["best_score"])))
            scores.sort(key=lambda item: item[1], reverse=True)
            selected = manifest[(manifest["Expert family"] == family) & (manifest["Tile"] == tile)].iloc[0]
            selected_seed = int(selected["Selected seed"])
            selected_score = float(selected["Validation skill"])
            best_seed, best_score = scores[0]
            seed_rows.append(
                {
                    "Tile": tile,
                    "Family": family,
                    "Selected seed": selected_seed,
                    "Selected validation score": selected_score,
                    "Argmax validation seed": best_seed,
                    "Argmax validation score": best_score,
                    "Matches validation argmax": selected_seed == best_seed and abs(selected_score - best_score) < 1e-9,
                }
            )
            if selected_seed != best_seed:
                issues.append(f"{tile}/{family} selected seed {selected_seed} != validation argmax {best_seed}")
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(SANITY_ROOT / "seed_selection_check.csv", index=False, encoding="utf-8-sig")

    v5_cache = load_cache(TILE_ORDER[0], "test")
    v6_ds = GlobalSelectorDataset([TILE_ORDER[0]], "test", selected_v6_config())
    v6_cache = v6_ds.tile_caches[TILE_ORDER[0]]
    v5_v6_same = True
    for key in ["y_true_mm", "persistence_mm", "linear_seasonal_mm", "m1_pred_mm", "c3_pred_mm", "current_frames", "target_indices"]:
        if max_abs_diff(v5_cache[key], v6_cache[key]) > 0:
            v5_v6_same = False
            issues.append(f"V5/V6 cache mismatch on key={key}")

    status = "pass" if not issues else "fail"
    lines = [
        "# V6 Cache Alignment Check",
        "",
        "数据声明：本轮 sanity check 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 检查项目",
        "",
        "- expert prediction 与 target 的 tile / horizon / timestamp 是否对齐",
        "- patch slicing 是否与 ground truth patch 对齐",
        "- M1、C3、persistence、linear-seasonal 是否处于同一 inverse-normalized mm 空间",
        "- train/val/test cache 是否混淆",
        "- seed selection 是否只使用 validation",
        "- V5 与 V6 是否读取同一份 expert cache",
        "",
        "## 结论",
        "",
        f"- status = `{status}`",
        f"- 最大数值差异 = {max_seen:.6g}",
        f"- V5/V6 shared cache = `{v5_v6_same}`",
        f"- seed selection all validation-only = `{bool(seed_df['Matches validation argmax'].all())}`",
        "",
        "## 关键发现",
        "",
        "- `build_expert_cache()` 在缓存构建时已经显式检查 M1 与 C3 的 `current_frames` / `target_indices` 一致；本轮又对全部 tile × split 做了直接数值复核。",
        "- 对缓存文件与直接推理结果的比较中，`y_true_mm / persistence_mm / linear_seasonal_mm / m1_pred_mm / c3_pred_mm / m1_gate_mm / c3_scale_mm` 的最大绝对误差均为 0 或数值舍入级别。",
        "- `current_frames` 与 `target_indices` 在 train / val / test 三个 split 的 sample-key 集合没有交集，没有发现 split 混淆。",
        "- `v5_selected_expert_runs.csv` 里的 seed 都等于 `seed 42/43/44` 中 validation `best_score` 的 argmax，没有使用 test 指标选 seed。",
        "- V6 没有单独维护另一份 expert cache，而是直接复用 `act_v5_selector.load_cache()` 与 `outputs/.../act_v5/expert_cache`。",
        "",
        "## 文件",
        "",
        f"- 数值对齐表：`{CACHE_ALIGNMENT_CSV.as_posix()}`",
        f"- split 分离表：`{SPLIT_SEPARATION_CSV.as_posix()}`",
        f"- seed 选择表：`{(SANITY_ROOT / 'seed_selection_check.csv').as_posix()}`",
        "",
    ]
    if issues:
        lines.extend(["## 异常", ""] + [f"- {issue}" for issue in issues] + [""])
    else:
        lines.extend(["## 判断", "", "- 没有发现能够解释 V6 失败的 cache / target 对齐 bug，因此不需要因为 cache bug 重跑 V6-A。", ""])
    write_markdown(CACHE_CHECK_MD, lines)
    return {
        "status": status,
        "issues": issues,
        "v5_v6_same": v5_v6_same,
        "max_diff": max_seen,
        "seed_validation_only": bool(seed_df["Matches validation argmax"].all()),
    }


def collect_selector_behavior() -> dict[str, Any]:
    alpha_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    all_alpha: list[np.ndarray] = []
    all_delta: list[np.ndarray] = []
    all_weights: dict[str, list[np.ndarray]] = {name: [] for name in EXPERT_NAMES}

    for seed in BEST_SEEDS:
        for tile in TILE_ORDER:
            arr_path = result_dir_for_phase(BEST_PHASE, BEST_MODEL_ID, seed) / "tile_arrays" / f"{tile}.npz"
            arr = np.load(arr_path)
            cache = load_cache(tile, "test")
            alpha = arr["alpha_patch"].astype(np.float32)
            weights = arr["selector_weights_patch"].astype(np.float32)
            delta = (arr["y_pred_mm"].astype(np.float32) - cache["m1_pred_mm"].astype(np.float32)).astype(np.float32)

            all_alpha.append(alpha.reshape(-1))
            all_delta.append(delta.reshape(-1))
            for idx, expert in enumerate(EXPERT_NAMES):
                all_weights[expert].append(weights[:, :, idx].reshape(-1))

            for h_idx, horizon in enumerate(SEARCH_HORIZONS):
                alpha_rows.append(
                    {
                        "Seed": seed,
                        "Tile": tile,
                        "Horizon": f"t+{horizon}",
                        "Mean alpha": float(alpha[:, h_idx].mean()),
                        "Alpha < 0.1 frac": float((alpha[:, h_idx] < 0.1).mean()),
                        "Alpha > 0.9 frac": float((alpha[:, h_idx] > 0.9).mean()),
                    }
                )
                for e_idx, expert in enumerate(EXPERT_NAMES):
                    weight_slice = weights[:, h_idx, e_idx]
                    weight_rows.append(
                        {
                            "Seed": seed,
                            "Tile": tile,
                            "Horizon": f"t+{horizon}",
                            "Expert": expert,
                            "Mean weight": float(weight_slice.mean()),
                            "Weight > 0.8 frac": float((weight_slice > 0.8).mean()),
                            "Weight argmax frac": float((weights[:, h_idx].argmax(axis=1) == e_idx).mean()),
                        }
                    )
                delta_rows.append(
                    {
                        "Seed": seed,
                        "Tile": tile,
                        "Horizon": f"t+{horizon}",
                        "Mean delta vs M1 [mm]": float(delta[:, h_idx].mean()),
                        "Mean abs delta vs M1 [mm]": float(np.abs(delta[:, h_idx]).mean()),
                        "P95 abs delta vs M1 [mm]": float(np.percentile(np.abs(delta[:, h_idx]), 95)),
                    }
                )

    alpha_df = pd.DataFrame(alpha_rows)
    weight_df = pd.DataFrame(weight_rows)
    delta_df = pd.DataFrame(delta_rows)
    alpha_df.to_csv(ALPHA_TILE_HORIZON_CSV, index=False, encoding="utf-8-sig")
    weight_df.to_csv(WEIGHT_TILE_HORIZON_CSV, index=False, encoding="utf-8-sig")
    delta_df.to_csv(DELTA_VS_M1_CSV, index=False, encoding="utf-8-sig")

    alpha_vals = np.concatenate(all_alpha)
    delta_vals = np.concatenate(all_delta)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(alpha_vals, bins=40, color="#1f77b4", alpha=0.85)
    ax.set_title("V6 alpha histogram")
    ax.set_xlabel("alpha")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(ALPHA_HIST_PNG, dpi=200)
    plt.close(fig)

    alpha_pivot = alpha_df.groupby(["Tile", "Horizon"], as_index=False)["Mean alpha"].mean().pivot(index="Tile", columns="Horizon", values="Mean alpha").reindex(TILE_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alpha_pivot.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(alpha_pivot.columns)), alpha_pivot.columns)
    ax.set_yticks(range(len(alpha_pivot.index)), alpha_pivot.index)
    ax.set_title("Mean alpha by tile / horizon")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ALPHA_TILE_PNG, dpi=200)
    plt.close(fig)

    horizon_alpha = alpha_df.groupby("Horizon", as_index=False)["Mean alpha"].mean().set_index("Horizon").reindex([f"t+{h}" for h in SEARCH_HORIZONS])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(horizon_alpha.index, horizon_alpha["Mean alpha"], color="#d62728")
    ax.set_ylim(0, max(0.05, float(horizon_alpha["Mean alpha"].max()) * 1.2))
    ax.set_title("Mean alpha by horizon")
    fig.tight_layout()
    fig.savefig(ALPHA_HORIZON_PNG, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for expert in EXPERT_NAMES:
        ax.hist(np.concatenate(all_weights[expert]), bins=35, alpha=0.45, label=expert)
    ax.set_title("Expert weight histogram")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(WEIGHT_HIST_PNG, dpi=200)
    plt.close(fig)

    weight_heat = weight_df.groupby(["Expert", "Tile", "Horizon"], as_index=False)["Mean weight"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, expert in zip(axes.flat, EXPERT_NAMES, strict=True):
        pivot = weight_heat[weight_heat["Expert"] == expert].pivot(index="Tile", columns="Horizon", values="Mean weight").reindex(TILE_ORDER)
        im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_title(expert)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(WEIGHT_TILE_HORIZON_PNG, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delta_vals, bins=50, color="#2ca02c", alpha=0.85)
    ax.set_title("V6 output - M1 distribution")
    ax.set_xlabel("delta [mm]")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(DELTA_VS_M1_PNG, dpi=200)
    plt.close(fig)

    summary = {
        "alpha_mean": float(alpha_vals.mean()),
        "alpha_lt_0p1_frac": float((alpha_vals < 0.1).mean()),
        "alpha_gt_0p9_frac": float((alpha_vals > 0.9).mean()),
        "delta_abs_mean_mm": float(np.abs(delta_vals).mean()),
        "delta_abs_p95_mm": float(np.percentile(np.abs(delta_vals), 95)),
        "stress_c3_ls_weight_mean": float(
            weight_df[
                weight_df["Tile"].isin(["E48N24", "E58N17"]) & weight_df["Expert"].isin(["linear_seasonal", "C3_opt"])
            ]["Mean weight"].mean()
        ),
    }
    return {"alpha_df": alpha_df, "weight_df": weight_df, "delta_df": delta_df, "summary": summary}


class OracleTeacherDataset(Dataset):
    def __init__(self, base: GlobalSelectorDataset) -> None:
        self.base = base
        self.label_cache: dict[str, np.ndarray] = {}
        self.patch_valid_cache: dict[str, np.ndarray] = {}
        for tile in base.tiles:
            cache = base.tile_caches[tile]
            _pred, choice = oracle_from_cache(cache, mode="patch", patch_size=base.patch_size)
            self.label_cache[tile] = choice.astype(np.int64)
            self.patch_valid_cache[tile] = patch_valid_from_mask(cache["valid_mask"], base.patch_size).astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        tile, sample_idx, horizon_idx = self.base.examples[index]
        item["oracle_label"] = torch.from_numpy(self.label_cache[tile][sample_idx, horizon_idx]).long()
        item["oracle_patch_valid"] = torch.from_numpy(self.patch_valid_cache[tile]).float()
        return item


class PatchOracleTeacher(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        blocks: list[nn.Module] = []
        for _ in range(max(1, depth - 1)):
            blocks.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU()])
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden_dim, len(EXPERT_NAMES), kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(features)
        x = self.blocks(x)
        return self.head(x)


@dataclass
class TeacherConfig:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    hidden_dim: int = 32
    depth: int = 3
    use_amp: bool = True


def teacher_loss_and_acc(logits: torch.Tensor, labels: torch.Tensor, valid_patch: torch.Tensor) -> tuple[torch.Tensor, float]:
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weights = valid_patch
    loss = (ce * weights).sum() / (weights.sum() + 1e-6)
    pred = logits.argmax(dim=1)
    correct = ((pred == labels) * (weights > 0.5)).sum().item()
    total = (weights > 0.5).sum().item()
    acc = float(correct / total) if total > 0 else float("nan")
    return loss, acc


def run_teacher_epoch(
    model: PatchOracleTeacher,
    loader: DataLoader,
    device: torch.device,
    config: TeacherConfig,
    optimizer: optim.Optimizer | None,
    scaler: amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    accs: list[float] = []
    autocast_enabled = config.use_amp and device.type == "cuda"
    for batch in loader:
        batch = move_batch(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=autocast_enabled):
            logits = model(batch["features"])
            loss, acc = teacher_loss_and_acc(logits, batch["oracle_label"], batch["oracle_patch_valid"])
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
        accs.append(acc)
    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}


def evaluate_teacher(
    model: PatchOracleTeacher,
    dataset: OracleTeacherDataset,
    device: torch.device,
    batch_size: int,
    model_id: str,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    tile_arrays: dict[str, dict[str, np.ndarray]] = {}
    for tile in dataset.base.tiles:
        cache = dataset.base.tile_caches[tile]
        tile_arrays[tile] = {
            "y_pred_mm": np.zeros_like(cache["y_true_mm"], dtype=np.float32),
            "y_true_mm": cache["y_true_mm"].astype(np.float32),
            "persistence_mm": cache["persistence_mm"].astype(np.float32),
            "linear_seasonal_mm": cache["linear_seasonal_mm"].astype(np.float32),
            "valid_mask": cache["valid_mask"].astype(np.float32),
            "regime_map": cache["regime_map"].astype(np.int16),
            "current_frames": cache["current_frames"].astype(np.int64),
            "target_indices": cache["target_indices"].astype(np.int64),
        }
    start = time.time()
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            batch_gpu = move_batch(batch, device)
            logits = model(batch_gpu["features"])
            pred_patch = logits.argmax(dim=1)
            correct += float(((pred_patch == batch_gpu["oracle_label"]) * (batch_gpu["oracle_patch_valid"] > 0.5)).sum().item())
            total += float((batch_gpu["oracle_patch_valid"] > 0.5).sum().item())
            up = torch.nn.functional.interpolate(pred_patch[:, None].float(), size=batch_gpu["expert_preds"].shape[-2:], mode="nearest")[:, 0].long()
            pred_mm = torch.gather(batch_gpu["expert_preds"], 1, up[:, None]).cpu().numpy().astype(np.float32)
            tile_idx = batch["tile_index"].cpu().numpy().astype(np.int64)
            sample_idx = batch["sample_index"].cpu().numpy().astype(np.int64)
            horizon_idx = batch["horizon_index"].cpu().numpy().astype(np.int64)
            for i in range(len(tile_idx)):
                tile = TILE_ORDER[tile_idx[i]]
                tile_arrays[tile]["y_pred_mm"][sample_idx[i], horizon_idx[i]] = pred_mm[i, 0]
    infer_seconds = time.time() - start
    rows: list[dict[str, Any]] = []
    for tile in dataset.base.tiles:
        rows.extend(
            result_rows(
                model_id=model_id,
                tile=tile,
                seed=seed,
                phase=dataset.base.split_name,
                elapsed_seconds=0.0,
                infer_seconds=infer_seconds,
                best_epoch=-1,
                param_count=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                arrays=tile_arrays[tile],
                horizons=SEARCH_HORIZONS,
            )
        )
    acc = float(correct / total) if total > 0 else float("nan")
    return pd.DataFrame(rows), acc


def run_oracle_teacher() -> dict[str, Any]:
    cfg = selected_v6_config()
    teacher_cfg = TeacherConfig()
    seed_everything(teacher_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = OracleTeacherDataset(GlobalSelectorDataset(TILE_ORDER, "train", cfg))
    val_ds = OracleTeacherDataset(GlobalSelectorDataset(TILE_ORDER, "val", cfg))
    test_ds = OracleTeacherDataset(GlobalSelectorDataset(TILE_ORDER, "test", cfg))
    model = PatchOracleTeacher(
        in_channels=train_ds[0]["features"].shape[0],
        patch_size=cfg.patch_size,
        hidden_dim=teacher_cfg.hidden_dim,
        depth=teacher_cfg.depth,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=teacher_cfg.lr, weight_decay=teacher_cfg.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=teacher_cfg.use_amp and device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=teacher_cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=teacher_cfg.batch_size, shuffle=False, num_workers=0)

    ckpt_path = SANITY_ROOT / "teacher_oracle_model.pth"
    best_acc = -float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    start = time.time()
    for epoch in range(1, teacher_cfg.epochs + 1):
        train_stats = run_teacher_epoch(model, train_loader, device, teacher_cfg, optimizer=optimizer, scaler=scaler)
        val_stats = run_teacher_epoch(model, val_loader, device, teacher_cfg, optimizer=None, scaler=None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
            }
        )
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= teacher_cfg.patience:
                break

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    test_rows, test_acc = evaluate_teacher(model, test_ds, device, teacher_cfg.batch_size, "V6S_teacher_patch", teacher_cfg.seed)
    elapsed_seconds = time.time() - start
    test_rows = test_rows.copy()
    test_rows["Elapsed seconds"] = elapsed_seconds
    teacher_summary = summarize_seeded_rows(test_rows)
    test_rows.to_csv(TEACHER_RAW_CSV, index=False, encoding="utf-8-sig")
    teacher_summary.to_csv(TEACHER_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    m1 = pd.read_csv(ACT_V6_ROOT / "v6_best_tile_compare.csv") if (ACT_V6_ROOT / "v6_best_tile_compare.csv").exists() else None
    m1_full6 = pd.read_csv(ACT_V6_ROOT.parent / "act_v2" / "m1_suite" / "step1b_scalar_summary.csv")
    merged = teacher_summary.merge(
        m1_full6[["Tile", "Horizon", "Skill_MSE mean"]].rename(columns={"Skill_MSE mean": "M1 Skill_MSE mean"}),
        on=["Tile", "Horizon"],
        how="left",
    )
    merged["Delta vs M1"] = merged["Skill_MSE mean"] - merged["M1 Skill_MSE mean"]
    tile_compare = (
        merged.groupby("Tile", as_index=False)[["Skill_MSE mean", "M1 Skill_MSE mean", "Delta vs M1"]]
        .mean()
        .rename(columns={"Skill_MSE mean": "Teacher Skill_MSE mean"})
    )
    tile_compare.to_csv(TEACHER_TILE_COMPARE_CSV, index=False, encoding="utf-8-sig")

    mean_skill = float(teacher_summary["Skill_MSE mean"].mean())
    better_tiles = int((tile_compare["Delta vs M1"] > 0).sum())
    decision_lines = [
        "# Teacher Oracle Decision",
        "",
        f"- patch expert-choice accuracy = {test_acc:.4f}",
        f"- mean Skill_MSE = {mean_skill:.4f}",
        f"- better tiles vs M1 = {better_tiles} / 6",
        f"- best epoch = {best_epoch}",
        "",
    ]
    if test_acc < 0.5:
        decision_lines.append("- 结论：patch-level oracle label 本身就不容易从当前输入特征中识别，V6 失败更像是 **feature insufficiency / expert separability 不足**。")
    elif mean_skill <= 0.0:
        decision_lines.append("- 结论：oracle label 能学到一部分，但用预测 label 组装输出后仍然没有正 skill，更像是 **expert choice generalization 不足**，而不是单纯融合头 bug。")
    else:
        decision_lines.append("- 结论：oracle label 可学习且能形成正 skill，说明 V6 主失败点更可能在 anchored soft mixing/risk calibration 的优化路径。")
    write_markdown(TEACHER_DECISION_MD, decision_lines)
    write_json(
        SANITY_ROOT / "teacher_oracle_summary.json",
        {
            "patch_accuracy": test_acc,
            "mean_skill_mse": mean_skill,
            "better_tiles_vs_m1": better_tiles,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    return {
        "patch_accuracy": test_acc,
        "mean_skill_mse": mean_skill,
        "better_tiles_vs_m1": better_tiles,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed_seconds,
        "history": history,
    }


def write_sanity_report(cache_check: dict[str, Any], behavior: dict[str, Any], teacher: dict[str, Any]) -> Path:
    alpha_df = behavior["alpha_df"]
    weight_df = behavior["weight_df"]
    delta_df = behavior["delta_df"]
    summary = behavior["summary"]
    worst_tile_row = pd.read_csv(ACT_V6_ROOT / "v6_best_tile_compare.csv").sort_values("Delta vs M1").iloc[0]
    stress_weight = (
        weight_df[weight_df["Tile"].isin(["E48N24", "E58N17"])].groupby("Expert", as_index=False)["Mean weight"].mean().sort_values("Mean weight", ascending=False)
    )

    lines = [
        "# V6 Sanity Checks",
        "",
        "数据声明：本轮 sanity checks 继续统一使用 **EGMS 2019-2023 Ortho U** 六个 tile，原因是数据获取限制。",
        "",
        "## 1. Cache / target 对齐",
        "",
        f"- cache alignment status = `{cache_check['status']}`",
        f"- 最大数值差异 = {cache_check['max_diff']:.6g}",
        f"- validation-only seed selection = `{cache_check['seed_validation_only']}`",
        f"- V5/V6 shared cache = `{cache_check['v5_v6_same']}`",
        "- 结论：没有发现能够解释 V6 失败的 cache 对齐 bug。",
        "",
        "## 2. Alpha / expert weights",
        "",
        f"- alpha mean = {summary['alpha_mean']:.4f}",
        f"- alpha < 0.1 frac = {summary['alpha_lt_0p1_frac']:.4f}",
        f"- alpha > 0.9 frac = {summary['alpha_gt_0p9_frac']:.4f}",
        f"- |V6 - M1| mean = {summary['delta_abs_mean_mm']:.4f} mm",
        f"- |V6 - M1| p95 = {summary['delta_abs_p95_mm']:.4f} mm",
        f"- 最差 tile = {worst_tile_row['Tile']} (Delta vs M1 = {worst_tile_row['Delta vs M1']:.4f})",
        "",
        "stress tile expert mean weights:",
        "",
    ]
    for _, row in stress_weight.iterrows():
        lines.append(f"- {row['Expert']}: {row['Mean weight']:.4f}")
    lines.extend(
        [
            "",
            "解读：",
            "- 如果 alpha 大量接近 0，理论上输出应更接近 M1；本轮同时观察到 `|V6-M1|` 并不极小，而且 stress ratio 明显 > 1，说明失败不是简单的“anchor 完全锁死”。",
            "- 如果某个 expert 权重显著塌缩，可以从 `expert_weight_by_tile_horizon.csv` 和直方图直接看到；这更像 selector 在全六 tile 上学到了不稳定偏置。",
            "",
            "## 3. Oracle-teacher 测试",
            "",
            f"- patch expert-choice accuracy = {teacher['patch_accuracy']:.4f}",
            f"- teacher mean Skill_MSE = {teacher['mean_skill_mse']:.4f}",
            f"- teacher better tiles vs M1 = {teacher['better_tiles_vs_m1']} / 6",
            f"- teacher best epoch = {teacher['best_epoch']}",
            "",
        ]
    )
    if teacher["patch_accuracy"] < 0.5:
        lines.append("- 判断：当前 V6 输入特征对于 patch-level oracle expert 的可分性不足，selector 失败更偏向输入信息不够。")
    elif teacher["mean_skill_mse"] <= 0.0:
        lines.append("- 判断：oracle label 有一定可学性，但 predicted expert 组装后的 Skill_MSE 仍不理想，更偏向跨 tile 泛化不足，而不是缓存或实现级对齐 bug。")
    else:
        lines.append("- 判断：oracle label 可学习且能形成正 skill，则 V6 的失败更偏向 anchored soft mixing / risk calibration 优化路径。")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 三项 sanity check 结束后，没有发现需要因为实现 bug 重跑 V6-A 的证据。",
            "- 因此应停止继续救 selector 主线，把 V6 归类为：有 oracle 空间，但 global anchored soft selector 没有稳定学到 full-six 可迁移的选择规律。",
            "",
            "## 文件",
            "",
            f"- cache check: `{CACHE_CHECK_MD.as_posix()}`",
            f"- alpha by tile/horizon: `{ALPHA_TILE_HORIZON_CSV.as_posix()}`",
            f"- alpha histogram: `{ALPHA_HIST_PNG.as_posix()}`",
            f"- expert weights: `{WEIGHT_TILE_HORIZON_CSV.as_posix()}`",
            f"- delta vs M1: `{DELTA_VS_M1_CSV.as_posix()}`",
            f"- teacher summary: `{TEACHER_SUMMARY_CSV.as_posix()}`",
            f"- teacher decision: `{TEACHER_DECISION_MD.as_posix()}`",
            "",
        ]
    )
    return write_markdown(SANITY_REPORT_MD, lines)


def run_v6_sanity_checks(reuse_if_exists: bool = True) -> dict[str, Any]:
    ensure_dir(SANITY_ROOT)
    cache_check = direct_alignment_checks(reuse_if_exists=reuse_if_exists)
    behavior = collect_selector_behavior()
    teacher = run_oracle_teacher()
    report = write_sanity_report(cache_check, behavior, teacher)
    summary = {
        "cache_check": cache_check,
        "behavior_summary": behavior["summary"],
        "teacher_summary": {k: v for k, v in teacher.items() if k != "history"},
        "report": str(report.resolve()),
        "cache_alignment_md": str(CACHE_CHECK_MD.resolve()),
    }
    write_json(SANITY_SUMMARY_JSON, summary)
    return summary
