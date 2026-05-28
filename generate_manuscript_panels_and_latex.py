from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "tgrs_2019_2023"
MANUSCRIPT = OUT / "manuscript"
FIG_FINAL = MANUSCRIPT / "figures" / "final"
FIG_READY = MANUSCRIPT / "figures" / "ready"
LATEX_PATH = MANUSCRIPT / "tgrs_vcea_ssvcea_figures_tables_draft.tex"

FIGURE_PATHS = {
    "Fig01": "figures/final/Fig01_study_area_map.png",
    "Fig02": "figures/final/Fig02_leakage_safe_pipeline.png",
    "Fig03": "figures/final/Fig03_persistence_and_horizon_difficulty.png",
    "Fig04": "figures/final/Fig04_vcea_ssvcea_method.png",
    "Fig05": "figures/ready/fig_vtsa_mechanism.pdf",
    "Fig06": "figures/final/Fig06_main_six_tile_heatmap_panel.png",
    "Fig07": "figures/final/Fig07_horizon_summary_panel.png",
    "Fig08": "figures/ready/fig_val_test_rank_scatter.png",
    "Fig09": "figures/ready/fig_static_vs_coea_weight_drift.png",
    "Fig10": "figures/ready/fig_stress_skill_pareto.pdf",
    "Fig11": "figures/ready/fig_external_tile_generalization.png",
    "Fig12": "figures/ready/fig_resolution_sensitivity.png",
    "Fig13": "figures/final/Fig13_model_search_audit_timeline.png",
    "Fig14": "figures/final/Fig14_physics_regime_diagnostic_panel.png",
}

TABLE_PATHS = {
    "Table01": "tables/ready/Table01_tile_metadata_main_external.csv",
    "Table02": "tables/ready/Table02_data_protocol.csv",
    "Table03": "tables/ready/Table03_metrics_definition.csv",
    "Table04": "tables/ready/Table04_expert_pool.csv",
    "Table05": "tables/ready/Table05_aggregation_protocol_comparison.csv",
    "Table06": "tables/ready/Table06_main_six_comparison.csv",
    "Table07": "tables/ready/Table07_vtsa_summary.csv",
    "Table08": "tables/ready/Table08_vcea_vs_ssvcea_safety_tradeoff.csv",
    "Table09": "tables/ready/Table09_external_locked_validation.csv",
    "Table10": "tables/ready/Table10_resolution_sensitivity.csv",
    "Table11": "tables/ready/Table11_architecture_search_audit.csv",
    "Table12": "tables/ready/Table12_reproducibility_package.csv",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, stem: str) -> Path:
    ensure_dir(FIG_FINAL)
    ensure_dir(FIG_READY)
    png = FIG_FINAL / f"{stem}.png"
    pdf = FIG_FINAL / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_READY / png.name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def add_value_labels(ax: plt.Axes, data: np.ndarray, fmt: str = ".2f", color_threshold: float | None = None) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                txt = "NA"
                color = "#475569"
            else:
                txt = format(val, fmt)
                color = "white" if (color_threshold is not None and val >= color_threshold) else "#0F172A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.2, color=color)


def style_heatmap(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11.8, pad=8)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Tile")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_fig06() -> Path:
    df = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon.csv")
    tiles = ["E32N34", "E32N35", "E39N30", "E44N23", "E48N24", "E58N17"]
    horizons = ["t+1", "t+3", "t+5", "t+10"]
    model_cols = {
        "M1": "M1 Skill_MSE mean",
        "COEA": "COEA Skill_MSE mean",
        "VCEA": "VCEA Skill_MSE mean",
        "SS-VCEA": "Selected Skill_MSE mean",
    }

    mats = {}
    all_values = []
    for label, col in model_cols.items():
        pivot = df.pivot(index="Tile", columns="Horizon", values=col).reindex(index=tiles, columns=horizons)
        mat = pivot.to_numpy(dtype=float)
        mats[label] = mat
        all_values.append(mat.reshape(-1))
    values = np.concatenate(all_values)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), constrained_layout=True)
    cmap = plt.get_cmap("YlGnBu")
    for ax, (label, mat) in zip(axes.flat, mats.items()):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(horizons)), horizons)
        ax.set_yticks(range(len(tiles)), tiles)
        style_heatmap(ax, label)
        add_value_labels(ax, mat, fmt=".2f", color_threshold=(vmin + vmax) / 2)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("Skill_MSE", fontsize=10.5)
    fig.suptitle("Main-six tile-wise heatmaps across the four main comparison models", fontsize=14.5, y=1.02)
    return save_fig(fig, "Fig06_main_six_tile_heatmap_panel")


def generate_fig07() -> Path:
    df = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon.csv")
    horizons = ["t+1", "t+3", "t+5", "t+10"]
    models = [
        ("M1", "M1 Skill_MSE mean", "M1 stress ratio"),
        ("COEA", "COEA Skill_MSE mean", "COEA stress ratio"),
        ("VCEA", "VCEA Skill_MSE mean", "VCEA stress ratio"),
        ("SS-VCEA", "Selected Skill_MSE mean", "Selected stress ratio"),
    ]

    skill_rows = []
    stress_rows = []
    gain_rows = []
    for label, skill_col, stress_col in models:
        skill_means = df.groupby("Horizon")[skill_col].mean().reindex(horizons)
        stress_means = df.groupby("Horizon")[stress_col].mean().reindex(horizons)
        gain_vs_m1 = (df.groupby("Horizon")[skill_col].mean() - df.groupby("Horizon")["M1 Skill_MSE mean"].mean()).reindex(horizons)
        skill_rows.append(skill_means.to_numpy())
        stress_rows.append(stress_means.to_numpy())
        gain_rows.append(gain_vs_m1.to_numpy())
    skill_mat = np.vstack(skill_rows)
    stress_mat = np.vstack(stress_rows)
    gain_mat = np.vstack(gain_rows)

    fig = plt.figure(figsize=(14.0, 8.2), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax_skill = fig.add_subplot(gs[:, 0])
    ax_stress = fig.add_subplot(gs[0, 1])
    ax_gain = fig.add_subplot(gs[1, 1])

    im1 = ax_skill.imshow(skill_mat, cmap="YlGnBu", aspect="auto")
    ax_skill.set_xticks(range(len(horizons)), horizons)
    ax_skill.set_yticks(range(len(models)), [m[0] for m in models])
    ax_skill.set_title("(a) Mean Skill_MSE by horizon", fontsize=12)
    ax_skill.set_xlabel("Forecast horizon")
    ax_skill.set_ylabel("Model")
    add_value_labels(ax_skill, skill_mat, fmt=".2f", color_threshold=float(np.nanmedian(skill_mat)))
    fig.colorbar(im1, ax=ax_skill, fraction=0.046, pad=0.02, label="Skill_MSE")

    im2 = ax_stress.imshow(stress_mat, cmap="magma_r", aspect="auto", vmin=float(np.nanmin(stress_mat)), vmax=max(1.03, float(np.nanmax(stress_mat))))
    ax_stress.set_xticks(range(len(horizons)), horizons)
    ax_stress.set_yticks(range(len(models)), [m[0] for m in models])
    ax_stress.set_title("(b) Mean stress ratio by horizon", fontsize=12)
    add_value_labels(ax_stress, stress_mat, fmt=".3f", color_threshold=float(np.nanmedian(stress_mat)))
    fig.colorbar(im2, ax=ax_stress, fraction=0.046, pad=0.02, label="Stress ratio")

    im3 = ax_gain.imshow(gain_mat, cmap="PiYG", aspect="auto", norm=Normalize(vmin=float(np.nanmin(gain_mat)), vmax=float(np.nanmax(gain_mat))))
    ax_gain.set_xticks(range(len(horizons)), horizons)
    ax_gain.set_yticks(range(len(models)), [m[0] for m in models])
    ax_gain.set_title("(c) Gain relative to M1", fontsize=12)
    add_value_labels(ax_gain, gain_mat, fmt=".2f", color_threshold=0.0)
    fig.colorbar(im3, ax=ax_gain, fraction=0.046, pad=0.02, label="Skill_MSE delta")

    fig.suptitle("Horizon-wise nowcasting summary: accuracy grows with horizon, while safety remains controlled", fontsize=14.2, y=1.01)
    return save_fig(fig, "Fig07_horizon_summary_panel")


def generate_fig14() -> Path:
    reg = pd.read_csv(OUT / "staticval_ea_study" / "regime_horizon_static_vs_coea.csv")
    phys = pd.read_csv(OUT / "E8" / "e8_physics_diagnostics.csv")
    redef = pd.read_csv(OUT / "supplements" / "A4_E8_redefined" / "a4_e8_redefined_metrics.csv")
    stress_img = Image.open(OUT / "staticval_ea_study" / "stress_tile_regime_panel.png")
    reg["Skill_MSE"] = pd.to_numeric(reg["Skill_MSE"], errors="coerce")
    phys["Value"] = pd.to_numeric(phys["Value"], errors="coerce")
    redef["Value"] = pd.to_numeric(redef["Value"], errors="coerce")

    regimes = ["stable", "seasonal", "subsidence", "accelerating", "coseismic", "mixed"]
    horizons = ["t+1", "t+3", "t+5", "t+10"]

    vcea_reg = (
        reg[reg["Method"] == "VCEA"]
        .groupby(["Regime", "Horizon"], as_index=False)["Skill_MSE"]
        .mean()
        .pivot(index="Regime", columns="Horizon", values="Skill_MSE")
        .reindex(index=regimes, columns=horizons)
    )
    coea_reg = (
        reg[reg["Method"] == "COEA"]
        .groupby(["Regime", "Horizon"], as_index=False)["Skill_MSE"]
        .mean()
        .pivot(index="Regime", columns="Horizon", values="Skill_MSE")
        .reindex(index=regimes, columns=horizons)
    )
    delta_reg = vcea_reg - coea_reg

    sign_df = redef[(redef["Metric"] == "velocity_sign_consistency_redefined") & (redef["Variant"] == "A6")][["Tile", "Value"]].copy()
    sign_df["Metric"] = "Sign consistency"
    recovery_df = redef[redef["Metric"] == "post_event_recovery_ratio"][["Tile", "Value"]].copy()
    recovery_df["Metric"] = "Recovery ratio"
    diag = pd.concat([sign_df, recovery_df], ignore_index=True)

    jump = phys[phys["Metric"] == "unrealistic_jump_rate"].groupby("Regime", as_index=False)["Value"].mean()
    jump = jump.set_index("Regime").reindex(regimes)

    fig = plt.figure(figsize=(15.2, 10.4), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.05, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    im1 = ax1.imshow(vcea_reg.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax1.set_xticks(range(len(horizons)), horizons)
    ax1.set_yticks(range(len(regimes)), [r.title() for r in regimes])
    ax1.set_title("(a) VCEA regime-wise Skill_MSE", fontsize=12)
    add_value_labels(ax1, vcea_reg.to_numpy(dtype=float), fmt=".2f", color_threshold=float(np.nanmedian(vcea_reg.to_numpy(dtype=float))))
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, label="Skill_MSE")

    im2 = ax2.imshow(delta_reg.to_numpy(dtype=float), cmap="PiYG", aspect="auto", norm=Normalize(vmin=float(np.nanmin(delta_reg.to_numpy(dtype=float))), vmax=float(np.nanmax(delta_reg.to_numpy(dtype=float)))))
    ax2.set_xticks(range(len(horizons)), horizons)
    ax2.set_yticks(range(len(regimes)), [r.title() for r in regimes])
    ax2.set_title("(b) VCEA minus COEA by regime", fontsize=12)
    add_value_labels(ax2, delta_reg.to_numpy(dtype=float), fmt=".2f", color_threshold=0.0)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, label="Skill_MSE delta")

    ax3.imshow(stress_img)
    ax3.axis("off")
    ax3.set_title("(c) Stress-tile regime context", fontsize=12)

    x = np.arange(len(diag))
    colors = ["#2563EB" if m == "Sign consistency" else "#C2410C" for m in diag["Metric"]]
    ax4.bar(x, diag["Value"], color=colors, alpha=0.88)
    ax4.set_xticks(x, [f"{t}\n{m}" for t, m in zip(diag["Tile"], diag["Metric"])], rotation=0)
    ax4.set_ylim(0, max(1.0, float(diag["Value"].max()) * 1.15))
    ax4.set_ylabel("Value")
    ax4.set_title("(d) Physics and recovery diagnostics", fontsize=12)
    ax4.grid(axis="y", alpha=0.2)
    for xi, yi in zip(x, diag["Value"]):
        ax4.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax4_2 = ax4.twinx()
    ax4_2.plot(np.arange(len(regimes)), jump["Value"].to_numpy(dtype=float), color="#6B7280", marker="o", linewidth=1.8)
    ax4_2.set_ylim(0, max(0.06, float(jump["Value"].max()) * 1.4))
    ax4_2.set_ylabel("Unrealistic jump rate")
    ax4_2.set_xticks(np.arange(len(regimes)), [r.title() for r in regimes])
    ax4_2.tick_params(axis="x", labelbottom=False)

    fig.suptitle("Reliability and limitation diagnostics across regimes and stress tiles", fontsize=14.5, y=1.01)
    return save_fig(fig, "Fig14_physics_regime_diagnostic_panel")


def update_figure_status() -> None:
    idx = MANUSCRIPT / "manuscript_asset_index.csv"
    df = pd.read_csv(idx)
    mapping = {
        "Fig06": FIGURE_PATHS["Fig06"],
        "Fig07": FIGURE_PATHS["Fig07"],
        "Fig14": FIGURE_PATHS["Fig14"],
    }
    for fig_id, rel_path in mapping.items():
        mask = df["asset_id"] == fig_id
        df.loc[mask, "status"] = "ready"
        df.loc[mask, "manuscript_target"] = rel_path.replace("/", "\\")
        df.loc[mask, "primary_file"] = str(MANUSCRIPT / rel_path)
    df.to_csv(idx, index=False, encoding="utf-8-sig")
    df[df["asset_type"] == "figure"].to_csv(MANUSCRIPT / "figure_asset_status.csv", index=False, encoding="utf-8-sig")
    df[df["asset_type"] == "table"].to_csv(MANUSCRIPT / "table_asset_status.csv", index=False, encoding="utf-8-sig")

    missing = df[(df["asset_type"] == "figure") & (df["status"].isin(["needs_new_draw", "partial_ready"]))]
    lines = [
        "# Manuscript Asset Notes",
        "",
        "本目录已经把大部分表格整理为可直接用于写作的 `csv`，并把所有提纲中的主图位统一整理到了 manuscript 目录。",
        "",
        "## 当前状态",
        "",
    ]
    if missing.empty:
        lines.append("- `Fig01` 到 `Fig14` 当前都已经有可直接引用的成图版本。")
        lines.append("- 若后续继续优化，主要工作将是 caption 微调和 panel 字体/布局的细修，而不是补图。")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['asset_id']}` | `{row['title']}` | 状态 `{row['status']}` | {row['note']}")
    (MANUSCRIPT / "missing_or_redraw_items.md").write_text("\n".join(lines), encoding="utf-8")

    ready_figures = int(((df["asset_type"] == "figure") & (df["status"] == "ready")).sum())
    ready_tables = int(((df["asset_type"] == "table") & (df["status"] == "ready")).sum())
    readme = f"""# TGRS Manuscript Asset Package

生成日期：2026-05-14

目录用途：

- 按 `TGRS_VCEA_SS_VCEA_full_manuscript_writing_outline_final.docx` 统一整理 manuscript 所需图片、表格、源数据和关键支撑文档。
- `figures/final` 下是当前投稿版成图。
- `tables/ready` 下是当前可直接进入写作流程的整理版表格。
- `tgrs_vcea_ssvcea_figures_tables_draft.tex` 提供了 14 图 12 表的统一 caption 草稿与 LaTeX 占位结构。

统计：

- ready figure：{ready_figures}/14
- ready table：{ready_tables}/12
- manuscript asset index：`manuscript_asset_index.csv`

当前结论：

- `Fig01`–`Fig14` 都已经有可直接引用的图文件。
- 如需继续优化，下一步最有价值的是统一 caption、字号、panel 标注风格与正文引用顺序。"""
    (MANUSCRIPT / "README.md").write_text(readme, encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "×": r"$\times$",
        "≤": r"$\leq$",
        "≥": r"$\geq$",
        "²": r"$^2$",
    }
    out = text
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


FIG_CAPTIONS = {
    "Fig01": (
        "Study area and coverage of the unified EGMS 2019--2023 Ortho U benchmark. "
        "(a) The six main-development tiles and four external locked-evaluation tiles are shown with their dominant regime labels. "
        "(b) Temporal span and valid-cell coverage at the primary $64\\times64$ setting. "
        "Main-six tiles were used for model development, whereas the external four tiles remained frozen for locked evaluation."
    ),
    "Fig02": (
        "Leakage-safe preprocessing and split protocol. Raw EGMS point observations were rasterized by point-to-cell mean aggregation, "
        "converted into dense tensors with nearest-neighbor fill for empty cells only, and evaluated with valid-mask weighted metrics. "
        "Chronological train/validation/test splits were assigned by target time, descriptors were computed from train windows or pre-test history only, "
        "and the external locked split remained untouched throughout model search."
    ),
    "Fig03": (
        "Why Skill\\_MSE is required under a strong persistence baseline. "
        "(a) Absolute RMSE grows with horizon, but persistence remains highly competitive, especially at $t{+}1$. "
        "(b) Skill\\_MSE normalizes horizon difficulty and makes relative gains over persistence directly comparable across horizons and models."
    ),
    "Fig04": (
        "Method overview of VCEA and SS-VCEA. VCEA performs validation-only static expert calibration over the expert pool "
        "$\\{\\text{Persistence},\\text{Linear-Seasonal},\\text{M1},\\text{C3\\_opt}\\}$ and produces a tile$\\times$horizon-specific weighted prediction. "
        "SS-VCEA reuses the same expert pool and applies a validation-calibrated hard fallback when the equal-normalized risk score "
        "$r=(d+c+v)/3$ exceeds a horizon-specific threshold."
    ),
    "Fig05": (
        "VTSA mechanism evidence. The figure summarizes validation-to-test rank stability, top-$k$ retention, and the drift behavior of online aggregation, "
        "providing the mechanism-level explanation for why validation-calibrated static aggregation remains competitive on the test period."
    ),
    "Fig06": (
        "Main-six tile-wise heatmaps for the four primary comparison models: M1, COEA, VCEA, and the selected SS-VCEA deployment variant. "
        "Each cell reports mean Skill\\_MSE for one tile-horizon pair. The panel highlights that aggregation gains are spatially heterogeneous, "
        "while SS-VCEA largely preserves the tile-wise benefits of VCEA."
    ),
    "Fig07": (
        "Horizon-wise summary of multi-step nowcasting. "
        "(a) Mean Skill\\_MSE aggregated over the six main tiles. "
        "(b) Mean stress ratio by horizon. "
        "(c) Mean Skill\\_MSE gain relative to M1. "
        "The panel shows that aggregation gains increase with horizon, whereas stress remains controlled for COEA and SS-VCEA."
    ),
    "Fig08": (
        "Validation ranking versus test ranking for expert performance. "
        "The strong validation-to-test agreement supports the use of validation-calibrated static weights in VCEA and helps explain why static aggregation outperforms delayed-feedback online updates."
    ),
    "Fig09": (
        "Weight drift from static VCEA to causal online aggregation (COEA). "
        "The drift pattern illustrates how delayed feedback and repeated online updates move the expert mixture away from the validation-calibrated optimum, "
        "which helps explain the performance gap between COEA and VCEA."
    ),
    "Fig10": (
        "Skill--stress Pareto analysis for VCEA and SS-VCEA variants on the main-six and external locked settings. "
        "The selected SS-VCEA-COEA\\_q97p5 variant sits close to the VCEA accuracy frontier while substantially reducing main-six stress risk."
    ),
    "Fig11": (
        "External locked-evaluation results on four held-out tiles. "
        "VCEA retains positive Skill\\_MSE under the frozen external protocol, and the selected SS-VCEA variant offers a conservative deployment alternative with limited accuracy loss."
    ),
    "Fig12": (
        "Resolution sensitivity of the model families. "
        "Single-model M1 degrades as the spatial grid increases from $64\\times64$ to $512\\times512$, whereas aggregation families continue to benefit from higher spatial resolution in sensitivity experiments."
    ),
    "Fig13": (
        "Model-search audit timeline from E1--E8 through V8 and DSCF hardening. "
        "The final manuscript is supported by a staged architecture search rather than a single tuned model: robust single-model baselines, selector families, online aggregation, static validation aggregation, "
        "and final safety hardening were all evaluated before freezing the mainline."
    ),
    "Fig14": (
        "Reliability and limitation diagnostics across regimes and stress tiles. "
        "(a) Regime-wise Skill\\_MSE of VCEA by horizon. "
        "(b) Regime-wise difference between VCEA and COEA. "
        "(c) Stress-tile regime context. "
        "(d) Physics-oriented diagnostics including sign consistency, post-event recovery ratio, and unrealistic jump behavior. "
        "These panels show that gains are not uniform across regimes and that coseismic cases remain the hardest operating condition."
    ),
}


TABLE_CAPTIONS = {
    "Table01": "Metadata of the six main-development tiles and four external locked-evaluation tiles, including dominant regime labels, temporal span, valid-cell coverage, and raw point counts.",
    "Table02": "Unified data protocol used throughout the manuscript, including the EGMS product version, split strategy, input length, target horizons, primary and sensitivity resolutions, and descriptor policy.",
    "Table03": "Definitions of the primary accuracy, safety, robustness, diagnostic, and reproducibility metrics used in the study.",
    "Table04": "Expert pool used by the aggregation family, including the persistence baseline, the linear-seasonal kinematic expert, the M1 neural expert, and the higher-capacity C3\\_opt expert.",
    "Table05": "Protocol comparison between VCEA, COEA, and SS-VCEA, emphasizing weight source, update mode, safety mechanism, and whether any test-time tuning is allowed.",
    "Table06": "Main-six comparison of the primary baselines, neural experts, and aggregation methods under the exact $64\\times64$ setting.",
    "Table07": "VTSA summary metrics, including rank stability, top-$k$ retention, drift, and blocked-to-test robustness, which explain why validation-calibrated static aggregation remains effective.",
    "Table08": "Safety trade-off between VCEA and the selected SS-VCEA-COEA\\_q97p5 variant on the main-six and external locked settings.",
    "Table09": "Frozen external locked-evaluation summary, reporting the held-out performance of the candidate methods without any external-based retuning.",
    "Table10": "Resolution sensitivity across model families and aggregation types, covering the primary $64\\times64$ setting and the $128\\times128$ and $512\\times512$ sensitivity analyses.",
    "Table11": "Architecture-search audit table summarizing the strongest candidate, best Skill\\_MSE, verdict, and reason for retention or rejection at each major search stage.",
    "Table12": "Reproducibility package and provenance map, listing the manifests, split definitions, cache registries, and claim-to-result tracing artifacts that support the final manuscript.",
}


TABLE_LAYOUTS = {
    "Table01": {"spec": r"lllp{0.13\textwidth}rrrr", "size": r"\scriptsize"},
    "Table02": {"spec": r"p{0.18\textwidth}p{0.18\textwidth}p{0.52\textwidth}", "size": r"\footnotesize"},
    "Table03": {"spec": r"p{0.14\textwidth}p{0.20\textwidth}p{0.56\textwidth}", "size": r"\footnotesize"},
    "Table04": {"spec": r"p{0.14\textwidth}p{0.16\textwidth}p{0.18\textwidth}p{0.42\textwidth}", "size": r"\footnotesize"},
    "Table05": {"spec": r"p{0.10\textwidth}p{0.19\textwidth}p{0.16\textwidth}p{0.16\textwidth}cp{0.21\textwidth}", "size": r"\footnotesize"},
    "Table06": {"spec": r"llrrrrrp{0.22\textwidth}", "size": r"\scriptsize"},
    "Table07": {"spec": r"p{0.22\textwidth}rp{0.54\textwidth}", "size": r"\footnotesize"},
    "Table08": {"spec": r"llrrrp{0.27\textwidth}", "size": r"\footnotesize"},
    "Table09": {"spec": r"lrrrrrrr", "size": r"\scriptsize"},
    "Table10": {"spec": r"lrrrrr", "size": r"\scriptsize"},
    "Table11": {"spec": r"p{0.11\textwidth}p{0.15\textwidth}rrp{0.46\textwidth}", "size": r"\scriptsize"},
    "Table12": {"spec": r"p{0.16\textwidth}p{0.48\textwidth}p{0.25\textwidth}", "size": r"\scriptsize"},
}


HEADER_MAP = {
    "Mean Skill_MSE": r"\shortstack{Mean\\Skill\_MSE}",
    "Gain vs M1": r"\shortstack{Gain\\vs M1}",
    "Better tiles vs M1": r"\shortstack{Better tiles\\vs M1}",
    "Stress ratio": r"\shortstack{Stress\\ratio}",
    "Mean Skill std": r"\shortstack{Mean Skill\\std}",
    "Skill_MSE mean": r"\shortstack{Skill\_MSE\\mean}",
    "Skill_MSE std": r"\shortstack{Skill\_MSE\\std}",
    "RMSE [mm] mean": r"\shortstack{RMSE\\[mm] mean}",
    "Acc@1mm [%] mean": r"\shortstack{Acc@1mm\\[%] mean}",
    "Worst RMSE/persistence": r"\shortstack{Worst RMSE/\\persistence}",
    "Gain/Loss vs VCEA": r"\shortstack{Gain/Loss\\vs VCEA}",
    "Best mean Skill_MSE": r"\shortstack{Best mean\\Skill\_MSE}",
    "Reason not final / role": r"\shortstack{Reason not final\\or role}",
    "Dominant regime": r"\shortstack{Dominant\\regime}",
    "Num points": r"\shortstack{Num\\points}",
    "Training scope": r"\shortstack{Training\\scope}",
    "Role in pool": r"\shortstack{Role in\\pool}",
    "Weight source": r"\shortstack{Weight\\source}",
    "Update mode": r"\shortstack{Update\\mode}",
    "Safety mechanism": r"\shortstack{Safety\\mechanism}",
    "Test tuning": r"\shortstack{Test\\tuning}",
    "Protocol item": r"\shortstack{Protocol\\item}",
    "Metric group": r"\shortstack{Metric\\group}",
}


def build_latex() -> None:
    def fmt_value(table_key: str, col: str, value) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, (np.bool_, bool)):
            return "Yes" if bool(value) else "No"
        if isinstance(value, (np.integer, int)):
            return f"{int(value)}"
        if isinstance(value, (np.floating, float)):
            if math.isfinite(float(value)):
                v = float(value)
                if abs(v) >= 1000:
                    return f"{v:,.0f}"
                if abs(v - round(v)) < 1e-9 and any(tok in col.lower() for tok in ["frames", "tiles", "points", "count"]):
                    return f"{int(round(v))}"
                if any(tok in col.lower() for tok in ["valid", "ratio", "std", "skill", "rmse", "mae", "ssim", "value", "gain"]):
                    return f"{v:.3f}"
                return f"{v:.3f}"
        text = str(value)
        if table_key == "Table12" and col == "Path":
            return r"\path{" + text.replace("\\", "/") + "}"
        return latex_escape(text)

    def fmt_header(col: str) -> str:
        return HEADER_MAP.get(col, latex_escape(col))

    def table_block(table_key: str) -> list[str]:
        df = pd.read_csv(MANUSCRIPT / TABLE_PATHS[table_key])
        layout = TABLE_LAYOUTS[table_key]
        rows: list[str] = []
        rows.extend(
            [
                "\\begin{table*}[t]",
                "  \\centering",
                f"  \\caption{{{TABLE_CAPTIONS[table_key]}}}",
                f"  \\label{{tab:{table_key.lower()}}}",
                f"  {layout['size']}",
                f"  % Embedded from {TABLE_PATHS[table_key]}",
                "  \\begin{adjustbox}{max width=\\textwidth}",
                f"  \\begin{{tabular}}{{{layout['spec']}}}",
                "    \\toprule",
                "    " + " & ".join(fmt_header(c) for c in df.columns) + r" \\",
                "    \\midrule",
            ]
        )
        for _, row in df.iterrows():
            cells = [fmt_value(table_key, col, row[col]) for col in df.columns]
            rows.append("    " + " & ".join(cells) + r" \\")
        rows.extend(
            [
                "    \\bottomrule",
                "  \\end{tabular}",
                "  \\end{adjustbox}",
                "\\end{table*}",
                "",
            ]
        )
        return rows

    lines: list[str] = []
    lines.extend(
        [
            "% Auto-generated manuscript asset skeleton for TGRS VCEA / SS-VCEA writing",
            "% Generated on 2026-05-14",
            "",
            "\\section*{Draft Figure and Table Floats}",
            "\\noindent This file collects the current figure/table draft captions, labels, and source paths for the manuscript asset package.",
            "",
            "% Recommended packages:",
            "% \\usepackage{graphicx}",
            "% \\usepackage{booktabs}",
            "% \\usepackage{adjustbox}",
            "% \\usepackage{threeparttable}",
            "% \\usepackage{float}",
            "% \\usepackage{url}",
            "",
            "\\subsection*{Figures}",
            "",
        ]
    )
    for idx in range(1, 15):
        key = f"Fig{idx:02d}"
        path = FIGURE_PATHS[key]
        caption = FIG_CAPTIONS[key]
        label = f"fig:{key.lower()}"
        lines.extend(
            [
                "\\begin{figure*}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.98\\textwidth]{{{path}}}",
                f"  \\caption{{{caption}}}",
                f"  \\label{{{label}}}",
                "\\end{figure*}",
                "",
            ]
        )

    lines.extend(["\\subsection*{Tables}", ""])
    for idx in range(1, 13):
        key = f"Table{idx:02d}"
        lines.extend(table_block(key))

    lines.extend(
        [
            "\\subsection*{Figure and Table Ordering Notes}",
            "\\begin{itemize}",
            "  \\item Figures 1--4 define the benchmark, leakage-safe protocol, baseline difficulty, and method overview.",
            "  \\item Figures 5--10 cover mechanism, main results, ranking stability, drift, and the safety trade-off.",
            "  \\item Figures 11--14 cover external robustness, resolution sensitivity, model-search audit, and reliability diagnostics.",
            "  \\item Tables 1--5 define data, metrics, experts, and protocols; Tables 6--12 summarize results, safety, robustness, and reproducibility.",
            "\\end{itemize}",
            "",
        ]
    )

    LATEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    generate_fig06()
    generate_fig07()
    generate_fig14()
    update_figure_status()
    build_latex()
    print(LATEX_PATH)


if __name__ == "__main__":
    main()
