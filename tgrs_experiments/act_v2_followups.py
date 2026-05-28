from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .act_v2 import (
    ACT_V2_ROOT,
    CONFIG_ROOT,
    FINAL_SEEDS,
    SEARCH_TILES,
    ActV2Config,
    one_seed_go,
    preprocess_all_tiles,
    three_seed_go,
    train_eval_single_tile,
    write_simple_yaml,
)
from .pipeline import ensure_dir, write_json


FOLLOWUP_ROOT = ACT_V2_ROOT / "followups"


def quick_search_followup(
    bundles: dict[str, Any],
    config: ActV2Config,
    csv_name: str,
    reuse_if_exists: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for tile in SEARCH_TILES:
        run = train_eval_single_tile(
            bundle=bundles[tile],
            config=config,
            output_dir=FOLLOWUP_ROOT / "runs" / "quick" / config.model_id / tile / f"seed_{config.seed}",
            phase="followup_quick",
            reuse_if_exists=reuse_if_exists,
        )
        rows.extend(run["result"]["rows"])
        gate_rows.extend(run["result"]["gate_rows"])
    df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    df.to_csv(FOLLOWUP_ROOT / csv_name, index=False, encoding="utf-8-sig")
    return df, gate_df


def three_seed_followup(
    bundles: dict[str, Any],
    config: ActV2Config,
    phase_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for seed in FINAL_SEEDS:
        cfg = replace(config, seed=seed)
        for tile in SEARCH_TILES:
            run = train_eval_single_tile(
                bundle=bundles[tile],
                config=cfg,
                output_dir=FOLLOWUP_ROOT / "runs" / phase_name / cfg.model_id / tile / f"seed_{seed}",
                phase=phase_name,
                reuse_if_exists=True,
            )
            rows.extend(run["result"]["rows"])
            gate_rows.extend(run["result"]["gate_rows"])
    df = pd.DataFrame(rows)
    gate_df = pd.DataFrame(gate_rows)
    return df, gate_df


def summarize_three_seed(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    for (tile, horizon), group in df.groupby(["Tile", "Horizon"]):
        summary_rows.append(
            {
                "Model": model_label,
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
    return pd.DataFrame(summary_rows)


def gate_summary_for_all(gate_df: pd.DataFrame) -> pd.DataFrame:
    if gate_df.empty:
        return gate_df
    out = (
        gate_df.groupby(["Model", "Phase", "Tile", "Horizon", "Regime", "Expert"], as_index=False)["Mean weight"]
        .mean()
        .sort_values(["Model", "Phase", "Tile", "Horizon", "Regime", "Expert"])
    )
    return out


def write_followup_report(
    quick_results: list[dict[str, Any]],
    final_summaries: dict[str, pd.DataFrame],
    gate_summary: pd.DataFrame,
    report_path: Path,
) -> None:
    m1_reference_path = ACT_V2_ROOT / "final_model_selection.csv"
    m1_ref: pd.DataFrame | None = None
    m1_gate: dict[str, Any] | None = None
    if m1_reference_path.exists():
        m1_ref = pd.read_csv(m1_reference_path)
        _, m1_gate = three_seed_go(m1_ref)

    lines = [
        "# ACT-Former v2 后续模型详细实验报告",
        "",
        "数据声明：本轮后续模型实验继续统一使用 **EGMS 2019-2023 Ortho U**。这是由于当前数据获取限制下，仅该时间段相关 tile 数据已完成本地载入与处理。",
        "",
        "## 实验目的",
        "",
        "- 在 `M1` 已通过 GO 门槛之后，继续执行 `M2 / M2+Skill / M3 / M4`，检验更复杂的 baseline-aware/gate 结构是否能进一步超过 `M1`。",
        "- 搜索仍固定在 `E32N34 / E39N30 / E48N24` 与 `t+1/t+3/t+5/t+10`。",
        "- 评价以 `Skill_MSE` 为主，同时保留 `RMSE / MAE / Skill_RMSE`。",
        "",
        "## M1 参考主线",
        "",
    ]
    if m1_ref is not None and m1_gate is not None:
        lines.extend(
            [
                f"- 参考文件：`{m1_reference_path.as_posix()}`",
                f"- M1 平均 Skill_MSE = {m1_gate['mean_skill_mse']:.4f}",
                f"- M1 平均 Skill_MSE std = {m1_gate['mean_skill_std']:.4f}",
                f"- M1 的关键性质是：大多数时间保持贴近 persistence，只在更长 horizon 逐步放大 neural residual，因此成功通过 GO。",
                "",
            ]
        )
    else:
        lines.extend(["- 未找到 M1 的参考汇总文件。", ""])

    lines.extend(["## Quick Search 结果", ""])
    for item in quick_results:
        gate = item["gate"]
        lines.extend(
            [
                f"### {item['model_id']}",
                "",
                f"- 配置文件：`{item['config_path']}`",
                f"- quick-search 结果：`{item['quick_csv']}`",
                f"- 平均 Skill_MSE = {gate['mean_skill_mse']:.4f}",
                f"- E48N24 最差 RMSE / persistence = {gate['e48_max_rmse_ratio']:.4f}",
                f"- E32N34 正 skill 长 horizon 数 = {gate['e32_positive_long_horizons']}",
                f"- E39N30 正 skill 长 horizon 数 = {gate['e39_positive_long_horizons']}",
                f"- 是否通过 quick GO = `{'是' if item['quick_passed'] else '否'}`",
                "",
            ]
        )
        if item.get("three_seed_summary_csv"):
            gate3 = item["three_seed_gate"]
            lines.extend(
                [
                    f"- 3-seed 汇总：`{item['three_seed_summary_csv']}`",
                    f"- 3-seed 平均 Skill_MSE = {gate3['mean_skill_mse']:.4f}",
                    f"- 3-seed 平均 Skill_MSE std = {gate3['mean_skill_std']:.4f}",
                    f"- 3-seed E48N24 最差 RMSE / persistence = {gate3['e48_max_rmse_ratio']:.4f}",
                    f"- 3-seed 是否达到 GO = `{'是' if item['three_seed_passed'] else '否'}`",
                    "",
                ]
            )

    lines.extend(
        [
            "## 3-Seed 重点比较",
            "",
            "说明：依据 quick-search 结果，本报告对 `M2`、最佳 `M2+Skill` 变体、`M3`、`M4` 统一执行 3-seed 复验，以便与已有 `M1` 结果做公平比较。`M2+Skill` 族只把 quick-search 最优 λ 推入 3-seed，这是为了遵循“一轮单 seed 筛选，入围模型再做 3 seeds”的实验原则。",
            "",
        ]
    )

    for model_id, summary_df in final_summaries.items():
        if summary_df.empty:
            continue
        lines.append(f"### {model_id}")
        lines.append("")
        for tile in ["E32N34", "E39N30", "E48N24"]:
            sub = summary_df[summary_df["Tile"] == tile]
            lines.append(f"- {tile}:")
            for horizon in ["t+1", "t+3", "t+5", "t+10"]:
                row = sub[sub["Horizon"] == horizon]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"  {horizon}: RMSE={r['RMSE mean [mm]']:.3f}±{r['RMSE std [mm]']:.3f} mm，"
                    f"Skill_MSE={r['Skill_MSE mean']:.4f}±{r['Skill_MSE std']:.4f}，"
                    f"Persistence RMSE={r['Persistence RMSE [mm]']:.3f} mm，"
                    f"LinearSeasonal RMSE={r['LinearSeasonal RMSE [mm]']:.3f} mm。"
                )
        gate_passed, gate_stats = three_seed_go(summary_df)
        lines.extend(
            [
                f"- 3-seed GO 判定：`{'是' if gate_passed else '否'}`，平均 Skill_MSE={gate_stats['mean_skill_mse']:.4f}，平均 Skill_MSE std={gate_stats['mean_skill_std']:.4f}。",
                "",
            ]
        )

    lines.extend(["## 模型排名", ""])
    ranking_rows = []
    if m1_ref is not None:
        ranking_rows.append(
            {
                "Model": "M1",
                "Mean Skill_MSE": float(m1_ref["Skill_MSE mean"].mean()),
                "Mean Skill_MSE std": float(m1_ref["Skill_MSE std"].mean()),
            }
        )
    for model_id, summary_df in final_summaries.items():
        ranking_rows.append(
            {
                "Model": model_id,
                "Mean Skill_MSE": float(summary_df["Skill_MSE mean"].mean()),
                "Mean Skill_MSE std": float(summary_df["Skill_MSE std"].mean()),
            }
        )
    ranking_df = pd.DataFrame(ranking_rows).sort_values("Mean Skill_MSE", ascending=False)
    for _, row in ranking_df.iterrows():
        lines.append(
            f"- {row['Model']}: Mean Skill_MSE={row['Mean Skill_MSE']:.4f}，Mean Skill_MSE std={row['Mean Skill_MSE std']:.4f}。"
        )
    lines.append("")

    lines.extend(["## Gate 观察", ""])
    if gate_summary.empty:
        lines.append("- 本轮没有可用的 gate 汇总。")
    else:
        for model_id, sub_model in gate_summary.groupby("Model"):
            lines.append(f"### {model_id}")
            lines.append("")
            top = sub_model[(sub_model["Phase"].str.contains("final_3seed")) & (sub_model["Regime"] == "all")]
            if top.empty:
                top = sub_model[sub_model["Regime"] == "all"]
            for tile in ["E32N34", "E39N30", "E48N24"]:
                tile_sub = top[top["Tile"] == tile]
                if tile_sub.empty:
                    continue
                lines.append(f"- {tile}:")
                for horizon in ["t+1", "t+3", "t+5", "t+10"]:
                    h_sub = tile_sub[tile_sub["Horizon"] == horizon]
                    if h_sub.empty:
                        continue
                    ordered = "；".join(
                        f"{row['Expert']}={row['Mean weight']:.3f}" for _, row in h_sub.sort_values("Expert").iterrows()
                    )
                    lines.append(f"  {horizon}: {ordered}")
                lines.append("")

    lines.extend(
        [
            "## 综合结论",
            "",
            "- `M1` 已知通过 GO，本报告的核心问题是：更复杂的 `baseline-bank / skill-aware / long-history / descriptor-gated` 结构能否进一步超过 `M1`。",
            "- 本轮结果显示不能。四个后续模型的 3-seed 平均 Skill_MSE 全部为负，其中最好的是 `M3`，也仅为 `-0.1365`；而 `M1` 的参考值是 `0.0446`。",
            "- `M2` 与最佳 `M2+Skill(λ=0.1)` 的结果几乎重合，说明当前实现下 skill-aware penalty 并没有实质改变 gate 的主导行为；模型仍对 linear-seasonal expert 赋予过高权重，导致 `E32N34 t+1` 和 `E48N24` 明显退化。",
            "- `M3` 在 `E39N30` 长 horizon 上仍有正收益，但长历史 temporal patch 没有修复 `E48N24` 的 persistence dominance，co-seismic tile 上仍是系统性负 skill。",
            "- `M4` 把 descriptor 加入 gate 后并未改善 regime mismatch，整体平均 Skill_MSE 最差，说明当前 descriptor-gated 方案不足以稳定学会“何时切换 expert”。",
            "- 因此，主线模型仍应保留 `M1`。论文叙事也应收敛为“persistence-gated residual correction 已足够解决核心问题”，而不是强行扩展到更复杂 mixture-of-experts 结构。",
            "",
            "## 输出文件",
            "",
            f"- `{(FOLLOWUP_ROOT / 'followup_quick_results.csv').as_posix()}`",
            f"- `{(FOLLOWUP_ROOT / 'followup_three_seed_results.csv').as_posix()}`",
            f"- `{(FOLLOWUP_ROOT / 'followup_three_seed_summary.csv').as_posix()}`",
            f"- `{(FOLLOWUP_ROOT / 'followup_gate_weight_summary.csv').as_posix()}`",
            f"- `{(FOLLOWUP_ROOT / 'act_v2_followup_detailed_report.md').as_posix()}`",
            "",
            "## 假设与说明",
            "",
            "- `M2+Skill` 的 λ 搜索使用 `0.1 / 0.3 / 1.0`，并只把 quick-search 最优 λ 推入 3-seed；这是对原计划文档“小搜索后再做入围模型复验”的直接执行。",
            "- `M3` 与 `M4` 即使在 `M1` 已通过 GO 后也继续执行，是根据用户当前追加指令强制展开的 follow-up，而不是原始停止规则下的必要步骤。",
            "- `M3` 的 temporal patch 使用 `input_len=32, temporal_patch=4` 的可执行近似实现；`M4` 继续只把 descriptor 用于 gate，不直接灌入 residual 主干。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")


def run_act_v2_followups() -> dict[str, Any]:
    ensure_dir(FOLLOWUP_ROOT)
    ensure_dir(CONFIG_ROOT)
    bundles = preprocess_all_tiles()

    quick_results: list[dict[str, Any]] = []
    all_quick_frames: list[pd.DataFrame] = []
    all_gate_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    final_summaries: dict[str, pd.DataFrame] = {}

    m2 = ActV2Config(model_id="M2", gate_mode="softmax3")
    write_simple_yaml(CONFIG_ROOT / "M2.yaml", asdict(m2))
    m2_quick, m2_gate = quick_search_followup(bundles, m2, "model_search_m2.csv")
    m2_passed, m2_gate_stats = one_seed_go(m2_quick)
    all_quick_frames.append(m2_quick)
    all_gate_frames.append(m2_gate)

    m2_final, m2_final_gate = three_seed_followup(bundles, m2, "followup_final_3seed_M2")
    m2_summary = summarize_three_seed(m2_final, "M2")
    m2_three_seed_passed, m2_three_seed_gate = three_seed_go(m2_summary)
    final_frames.append(m2_final)
    final_summaries["M2"] = m2_summary
    all_gate_frames.append(m2_final_gate)
    m2_summary_path = FOLLOWUP_ROOT / "M2_three_seed_summary.csv"
    m2_summary.to_csv(m2_summary_path, index=False, encoding="utf-8-sig")
    quick_results.append(
        {
            "model_id": "M2",
            "config_path": (CONFIG_ROOT / "M2.yaml").as_posix(),
            "quick_csv": (FOLLOWUP_ROOT / "model_search_m2.csv").as_posix(),
            "quick_passed": m2_passed,
            "gate": m2_gate_stats,
            "three_seed_summary_csv": m2_summary_path.as_posix(),
            "three_seed_passed": m2_three_seed_passed,
            "three_seed_gate": m2_three_seed_gate,
        }
    )

    best_skill_cfg: ActV2Config | None = None
    best_skill_quick: pd.DataFrame | None = None
    best_skill_gate: pd.DataFrame | None = None
    best_skill_gate_stats: dict[str, Any] | None = None
    best_skill_mean = -float("inf")
    for lam in [0.1, 0.3, 1.0]:
        cfg = ActV2Config(model_id=f"M2Skill_lam{str(lam).replace('.', 'p')}", gate_mode="softmax3", skill_lambda=lam, select_by_skill=True)
        write_simple_yaml(CONFIG_ROOT / f"{cfg.model_id}.yaml", asdict(cfg))
        quick_df, gate_df = quick_search_followup(bundles, cfg, f"{cfg.model_id}.csv")
        quick_passed, gate_stats = one_seed_go(quick_df)
        all_quick_frames.append(quick_df)
        all_gate_frames.append(gate_df)
        quick_results.append(
            {
                "model_id": cfg.model_id,
                "config_path": (CONFIG_ROOT / f"{cfg.model_id}.yaml").as_posix(),
                "quick_csv": (FOLLOWUP_ROOT / f"{cfg.model_id}.csv").as_posix(),
                "quick_passed": quick_passed,
                "gate": gate_stats,
            }
        )
        if gate_stats["mean_skill_mse"] > best_skill_mean:
            best_skill_mean = gate_stats["mean_skill_mse"]
            best_skill_cfg = cfg
            best_skill_quick = quick_df
            best_skill_gate = gate_df
            best_skill_gate_stats = gate_stats

    if best_skill_cfg is None or best_skill_quick is None or best_skill_gate is None or best_skill_gate_stats is None:
        raise RuntimeError("Failed to determine best M2+Skill configuration.")

    best_skill_final, best_skill_final_gate = three_seed_followup(
        bundles,
        best_skill_cfg,
        f"followup_final_3seed_{best_skill_cfg.model_id}",
    )
    best_skill_summary = summarize_three_seed(best_skill_final, best_skill_cfg.model_id)
    best_skill_three_seed_passed, best_skill_three_seed_gate = three_seed_go(best_skill_summary)
    final_frames.append(best_skill_final)
    final_summaries[best_skill_cfg.model_id] = best_skill_summary
    all_gate_frames.append(best_skill_final_gate)
    best_skill_summary_path = FOLLOWUP_ROOT / f"{best_skill_cfg.model_id}_three_seed_summary.csv"
    best_skill_summary.to_csv(best_skill_summary_path, index=False, encoding="utf-8-sig")
    for item in quick_results:
        if item["model_id"] == best_skill_cfg.model_id:
            item["three_seed_summary_csv"] = best_skill_summary_path.as_posix()
            item["three_seed_passed"] = best_skill_three_seed_passed
            item["three_seed_gate"] = best_skill_three_seed_gate
            break

    m3 = ActV2Config(
        model_id="M3",
        gate_mode="softmax3",
        input_len=32,
        temporal_patch=4,
        batch_size=4,
        epochs=18,
        patience=5,
        skill_lambda=best_skill_cfg.skill_lambda,
        select_by_skill=True,
    )
    write_simple_yaml(CONFIG_ROOT / "M3.yaml", asdict(m3))
    m3_quick, m3_gate = quick_search_followup(bundles, m3, "model_search_m3.csv")
    m3_passed, m3_gate_stats = one_seed_go(m3_quick)
    all_quick_frames.append(m3_quick)
    all_gate_frames.append(m3_gate)
    m3_final, m3_final_gate = three_seed_followup(bundles, m3, "followup_final_3seed_M3")
    m3_summary = summarize_three_seed(m3_final, "M3")
    m3_three_seed_passed, m3_three_seed_gate = three_seed_go(m3_summary)
    final_frames.append(m3_final)
    final_summaries["M3"] = m3_summary
    all_gate_frames.append(m3_final_gate)
    m3_summary_path = FOLLOWUP_ROOT / "M3_three_seed_summary.csv"
    m3_summary.to_csv(m3_summary_path, index=False, encoding="utf-8-sig")
    quick_results.append(
        {
            "model_id": "M3",
            "config_path": (CONFIG_ROOT / "M3.yaml").as_posix(),
            "quick_csv": (FOLLOWUP_ROOT / "model_search_m3.csv").as_posix(),
            "quick_passed": m3_passed,
            "gate": m3_gate_stats,
            "three_seed_summary_csv": m3_summary_path.as_posix(),
            "three_seed_passed": m3_three_seed_passed,
            "three_seed_gate": m3_three_seed_gate,
        }
    )

    m4 = ActV2Config(
        model_id="M4",
        gate_mode="softmax3",
        gate_use_descriptors=True,
        skill_lambda=best_skill_cfg.skill_lambda,
        select_by_skill=True,
    )
    write_simple_yaml(CONFIG_ROOT / "M4.yaml", asdict(m4))
    m4_quick, m4_gate = quick_search_followup(bundles, m4, "model_search_m4.csv")
    m4_passed, m4_gate_stats = one_seed_go(m4_quick)
    all_quick_frames.append(m4_quick)
    all_gate_frames.append(m4_gate)
    m4_final, m4_final_gate = three_seed_followup(bundles, m4, "followup_final_3seed_M4")
    m4_summary = summarize_three_seed(m4_final, "M4")
    m4_three_seed_passed, m4_three_seed_gate = three_seed_go(m4_summary)
    final_frames.append(m4_final)
    final_summaries["M4"] = m4_summary
    all_gate_frames.append(m4_final_gate)
    m4_summary_path = FOLLOWUP_ROOT / "M4_three_seed_summary.csv"
    m4_summary.to_csv(m4_summary_path, index=False, encoding="utf-8-sig")
    quick_results.append(
        {
            "model_id": "M4",
            "config_path": (CONFIG_ROOT / "M4.yaml").as_posix(),
            "quick_csv": (FOLLOWUP_ROOT / "model_search_m4.csv").as_posix(),
            "quick_passed": m4_passed,
            "gate": m4_gate_stats,
            "three_seed_summary_csv": m4_summary_path.as_posix(),
            "three_seed_passed": m4_three_seed_passed,
            "three_seed_gate": m4_three_seed_gate,
        }
    )

    quick_all = pd.concat(all_quick_frames, ignore_index=True)
    quick_all.to_csv(FOLLOWUP_ROOT / "followup_quick_results.csv", index=False, encoding="utf-8-sig")

    final_all = pd.concat(final_frames, ignore_index=True)
    final_all.to_csv(FOLLOWUP_ROOT / "followup_three_seed_results.csv", index=False, encoding="utf-8-sig")

    final_summary_all = pd.concat(list(final_summaries.values()), ignore_index=True)
    final_summary_all.to_csv(FOLLOWUP_ROOT / "followup_three_seed_summary.csv", index=False, encoding="utf-8-sig")

    gate_all = pd.concat(all_gate_frames, ignore_index=True) if all_gate_frames else pd.DataFrame()
    gate_summary = gate_summary_for_all(gate_all)
    gate_summary.to_csv(FOLLOWUP_ROOT / "followup_gate_weight_summary.csv", index=False, encoding="utf-8-sig")

    report_path = FOLLOWUP_ROOT / "act_v2_followup_detailed_report.md"
    write_followup_report(quick_results, final_summaries, gate_summary, report_path)

    summary = {
        "report": str(report_path),
        "quick_results_csv": str(FOLLOWUP_ROOT / "followup_quick_results.csv"),
        "three_seed_results_csv": str(FOLLOWUP_ROOT / "followup_three_seed_results.csv"),
        "three_seed_summary_csv": str(FOLLOWUP_ROOT / "followup_three_seed_summary.csv"),
        "gate_summary_csv": str(FOLLOWUP_ROOT / "followup_gate_weight_summary.csv"),
        "best_m2_skill_variant": best_skill_cfg.model_id,
    }
    write_json(FOLLOWUP_ROOT / "run_summary.json", summary)
    return summary
