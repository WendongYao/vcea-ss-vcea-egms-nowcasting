from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "tgrs_2019_2023"
MANUSCRIPT = OUT / "manuscript"
FIG_FINAL = MANUSCRIPT / "figures" / "final"
FIG_READY = MANUSCRIPT / "figures" / "ready"

FIGURE_NAMES = {
    "Fig01": "Fig01_study_area_map",
    "Fig02": "Fig02_leakage_safe_pipeline",
    "Fig03": "Fig03_persistence_and_horizon_difficulty",
    "Fig04": "Fig04_vcea_ssvcea_method",
    "Fig13": "Fig13_model_search_audit_timeline",
}

REGIME_COLORS = {
    "stable": "#7CB7D4",
    "seasonal": "#2A9D8F",
    "subsidence": "#E76F51",
    "accelerating": "#F4A261",
    "coseismic": "#C1121F",
    "mixed": "#6C757D",
}

VERDICT_COLORS = {
    "pass": "#2A9D8F",
    "fail": "#D97706",
    "no_go": "#C1121F",
    "final_winner": "#264653",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_tile(tile_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"E(\d+)N(\d+)", tile_id)
    if not match:
        raise ValueError(f"Unexpected tile id: {tile_id}")
    return int(match.group(1)), int(match.group(2))


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    ensure_dir(FIG_FINAL)
    ensure_dir(FIG_READY)
    png_path = FIG_FINAL / f"{stem}.png"
    pdf_path = FIG_FINAL / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_READY / png_path.name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.7)


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str, ec: str = "#334155", fontsize: int = 10, roundness: float = 0.03) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={roundness}",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#0F172A", wrap=True)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569", lw: float = 1.8, rad: float = 0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def generate_fig01() -> tuple[Path, Path]:
    main = pd.read_csv(OUT / "Table1_tile_metadata.csv")
    ext = pd.read_csv(OUT / "confirmation" / "external_tiles" / "external_tile_manifest.csv")

    main_df = pd.DataFrame(
        {
            "Tile": main["tile_id"],
            "Group": "Main-six",
            "Dominant regime": main["assigned_regime"],
            "Epochs": main["num_epochs"],
            "Valid64": main["valid_grid_ratio_64"] * 100.0,
            "Num points": main["num_raw_points"],
        }
    )
    ext_df = pd.DataFrame(
        {
            "Tile": ext["Tile"],
            "Group": "External locked",
            "Dominant regime": ext["Dominant regime"].str.lower(),
            "Epochs": ext["Epochs"],
            "Valid64": ext["Valid64"] * 100.0,
            "Num points": ext["Num points"],
        }
    )
    df = pd.concat([main_df, ext_df], ignore_index=True)
    df[["East", "North"]] = df["Tile"].apply(lambda t: pd.Series(parse_tile(t)))

    fig = plt.figure(figsize=(13.5, 7.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_cov = fig.add_subplot(gs[0, 1])

    for _, row in df.iterrows():
        edge_ls = "-" if row["Group"] == "Main-six" else (0, (4, 2))
        rect = Rectangle(
            (row["East"] - 0.48, row["North"] - 0.48),
            0.96,
            0.96,
            facecolor=REGIME_COLORS.get(row["Dominant regime"], "#94A3B8"),
            edgecolor="#0F172A",
            linewidth=2.2 if row["Group"] == "Main-six" else 1.8,
            linestyle=edge_ls,
        )
        ax_map.add_patch(rect)
        tile_label = row["Tile"].replace("N", "\nN")
        ax_map.text(
            row["East"],
            row["North"],
            tile_label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
            fontweight="bold",
            linespacing=0.9,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#0F172A")],
        )

    ax_map.set_xlim(df["East"].min() - 1.0, df["East"].max() + 1.0)
    ax_map.set_ylim(df["North"].min() - 1.0, df["North"].max() + 1.0)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("East tile index")
    ax_map.set_ylabel("North tile index")
    ax_map.set_title("(a) Tile layout and dominant regimes", fontsize=12.5, pad=10)
    ax_map.set_xticks(sorted(df["East"].unique()))
    ax_map.set_yticks(sorted(df["North"].unique()))
    style_axes(ax_map)

    regime_handles = [
        Patch(facecolor=color, edgecolor="#0F172A", label=regime.title())
        for regime, color in REGIME_COLORS.items()
        if regime in set(df["Dominant regime"])
    ]
    role_handles = [
        Line2D([0], [0], color="#0F172A", linewidth=2.2, linestyle="-", label="Main-six"),
        Line2D([0], [0], color="#0F172A", linewidth=1.8, linestyle=(0, (4, 2)), label="External locked"),
    ]
    ax_map.legend(handles=regime_handles + role_handles, loc="upper left", fontsize=8.8, frameon=True, ncol=2)

    markers = {"Main-six": "o", "External locked": "s"}
    for group, group_df in df.groupby("Group"):
        ax_cov.scatter(
            group_df["Epochs"],
            group_df["Valid64"],
            s=np.sqrt(group_df["Num points"].to_numpy()) * 1.2,
            c=[REGIME_COLORS.get(r, "#94A3B8") for r in group_df["Dominant regime"]],
            marker=markers[group],
            edgecolors="#0F172A",
            linewidths=0.8,
            alpha=0.92,
            label=group,
        )
        for _, row in group_df.iterrows():
            ax_cov.text(row["Epochs"] + 1.5, row["Valid64"] + 0.5, row["Tile"], fontsize=8.5, color="#0F172A")

    ax_cov.set_title("(b) Coverage and temporal span", fontsize=12.5, pad=10)
    ax_cov.set_xlabel("Number of epochs")
    ax_cov.set_ylabel("Valid-cell ratio at 64×64 (%)")
    style_axes(ax_cov)
    ax_cov.legend(frameon=True, fontsize=9, loc="lower right")
    ax_cov.text(
        0.02,
        0.02,
        "Circle size ∝ number of raw points\nEGMS Ortho U 2019–2023; 100 km tiles",
        transform=ax_cov.transAxes,
        fontsize=8.8,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", boxstyle="round,pad=0.35"),
    )

    fig.suptitle("Study area: six main tiles and four external locked-evaluation tiles", fontsize=14.5, y=0.99)
    return save_figure(fig, FIGURE_NAMES["Fig01"])


def generate_fig02() -> tuple[Path, Path]:
    split = json.loads((ROOT / "splits" / "main6_train_val_test.json").read_text(encoding="utf-8"))
    example_tile = split["tiles"]["E32N34"]
    train_n = len(example_tile["train_dates"])
    val_n = len(example_tile["val_dates"])
    test_n = len(example_tile["test_dates"])

    fig = plt.figure(figsize=(14.2, 7.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.03, 0.66), (0.16, 0.18), "Raw EGMS Ortho U\n2019–2023 tile CSV/XML", "#DBEAFE")
    draw_box(ax, (0.24, 0.66), (0.18, 0.18), "Point-to-cell mean\nrasterization\n64 / 128 / 512", "#E0F2FE")
    draw_box(ax, (0.47, 0.66), (0.18, 0.18), "Dense tensor build\nnearest-neighbor fill\nfor empty cells only", "#ECFCCB")
    draw_box(ax, (0.70, 0.66), (0.20, 0.18), "Valid-mask metrics\nfill never changes\nvalid-pixel counting", "#FCE7F3")

    draw_arrow(ax, (0.19, 0.75), (0.24, 0.75))
    draw_arrow(ax, (0.42, 0.75), (0.47, 0.75))
    draw_arrow(ax, (0.65, 0.75), (0.70, 0.75))

    draw_box(ax, (0.08, 0.34), (0.23, 0.17), f"Chronological split by target time\nTrain / Val / Test = {train_n} / {val_n} / {test_n}\n(≈70 / 10 / 20)", "#FEF3C7")
    draw_box(ax, (0.39, 0.34), (0.24, 0.17), "Train-window descriptors only\nvelocity, acceleration,\nseasonal amplitude/phase,\nresidual std, trend R², DOY", "#EDE9FE")
    draw_box(ax, (0.71, 0.34), (0.20, 0.17), "Nowcasting windows\ninput_len = 16\nhorizons = 1 / 3 / 5 / 10", "#DCFCE7")

    draw_arrow(ax, (0.20, 0.66), (0.20, 0.51))
    draw_arrow(ax, (0.51, 0.66), (0.51, 0.51))
    draw_arrow(ax, (0.81, 0.66), (0.81, 0.51))
    draw_arrow(ax, (0.31, 0.425), (0.39, 0.425))
    draw_arrow(ax, (0.63, 0.425), (0.71, 0.425))

    draw_box(ax, (0.27, 0.08), (0.46, 0.15), "Training / calibration / locked evaluation under infra_v2\nMain-six used for model development; external four tiles remain frozen for locked evaluation", "#E2E8F0")
    draw_arrow(ax, (0.41, 0.34), (0.41, 0.23))
    draw_arrow(ax, (0.81, 0.34), (0.59, 0.23), rad=-0.08)

    callouts = [
        ((0.06, 0.92), "No future target access", "#FEE2E2"),
        ((0.34, 0.92), "Validation-only calibration", "#FDE68A"),
        ((0.66, 0.92), "External locked split untouched", "#D1FAE5"),
    ]
    for (x, y), text, color in callouts:
        draw_box(ax, (x, y), (0.25, 0.055), text, color, fontsize=10, roundness=0.02)

    ax.text(
        0.08,
        0.27,
        "Rule 1: input windows may cross split boundaries, but targets never do.\n"
        "Rule 2: descriptors are fitted from train windows or pre-test history only.\n"
        "Rule 3: nearest-neighbor fill is tensor construction only; evaluation stays mask-weighted.",
        fontsize=10.2,
        color="#334155",
        va="top",
    )
    ax.set_title("Leakage-safe preprocessing and split protocol", fontsize=15, pad=12)
    return save_figure(fig, FIGURE_NAMES["Fig02"])


def generate_fig03() -> tuple[Path, Path]:
    tile_h = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon.csv")
    m1_suite = pd.read_csv(OUT / "act_v2" / "m1_suite" / "step1_combined_summary.csv")

    horizon_order = ["t+1", "t+3", "t+5", "t+10"]
    horizon_numeric = [1, 3, 5, 10]

    baseline = (
        m1_suite[m1_suite["Model"].isin(["persistence", "linear_seasonal"])]
        .groupby(["Model", "Horizon"], as_index=False)["RMSE [mm] mean"]
        .mean()
    )
    rmse_panel = pd.DataFrame({"Horizon": horizon_order, "HorizonNum": horizon_numeric})
    for model, col in [
        ("persistence", "Persistence"),
        ("linear_seasonal", "Linear-Seasonal"),
    ]:
        s = baseline[baseline["Model"] == model].set_index("Horizon")["RMSE [mm] mean"]
        rmse_panel[col] = rmse_panel["Horizon"].map(s)
    agg_h = tile_h.groupby("Horizon", as_index=False).agg(
        {
            "M1 RMSE [mm] mean": "mean",
            "VCEA RMSE [mm] mean": "mean",
            "M1 Skill_MSE mean": "mean",
            "COEA Skill_MSE mean": "mean",
            "VCEA Skill_MSE mean": "mean",
            "Selected Skill_MSE mean": "mean",
        }
    )
    rmse_panel["M1"] = rmse_panel["Horizon"].map(agg_h.set_index("Horizon")["M1 RMSE [mm] mean"])
    rmse_panel["VCEA"] = rmse_panel["Horizon"].map(agg_h.set_index("Horizon")["VCEA RMSE [mm] mean"])
    skill_panel = agg_h.copy()
    skill_panel["HorizonNum"] = skill_panel["Horizon"].map(dict(zip(horizon_order, horizon_numeric)))

    fig = plt.figure(figsize=(13.5, 6.5))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    line_styles = {
        "Persistence": ("#334155", "o"),
        "Linear-Seasonal": ("#F59E0B", "s"),
        "M1": ("#2563EB", "^"),
        "VCEA": ("#0F766E", "D"),
    }
    for name in ["Persistence", "Linear-Seasonal", "M1", "VCEA"]:
        color, marker = line_styles[name]
        ax1.plot(rmse_panel["HorizonNum"], rmse_panel[name], marker=marker, linewidth=2.2, markersize=6.5, color=color, label=name)
    ax1.set_xticks(horizon_numeric, horizon_order)
    ax1.set_xlabel("Forecast horizon")
    ax1.set_ylabel("Mean RMSE (mm)")
    ax1.set_title("(a) Absolute error grows with horizon,\nbut persistence remains strong", fontsize=12.5)
    style_axes(ax1)
    ax1.legend(frameon=True, fontsize=9.2, loc="upper left")
    ax1.annotate(
        "Persistence is already hard to beat at t+1",
        xy=(1, float(rmse_panel.loc[rmse_panel["HorizonNum"] == 1, "Persistence"].iloc[0])),
        xytext=(2.2, 1.18),
        arrowprops=dict(arrowstyle="->", color="#475569", linewidth=1.2),
        fontsize=9.5,
        color="#334155",
    )

    skill_styles = {
        "M1 Skill_MSE mean": ("M1", "#2563EB", "^"),
        "COEA Skill_MSE mean": ("COEA", "#8B5CF6", "o"),
        "VCEA Skill_MSE mean": ("VCEA", "#0F766E", "D"),
        "Selected Skill_MSE mean": ("SS-VCEA", "#C2410C", "s"),
    }
    for col, (label, color, marker) in skill_styles.items():
        ax2.plot(skill_panel["HorizonNum"], skill_panel[col], marker=marker, linewidth=2.2, markersize=6.5, color=color, label=label)
    ax2.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.1)
    ax2.set_xticks(horizon_numeric, horizon_order)
    ax2.set_xlabel("Forecast horizon")
    ax2.set_ylabel("Mean Skill_MSE")
    ax2.set_title("(b) Skill_MSE normalizes horizon difficulty\nand reveals relative gain over persistence", fontsize=12.5)
    style_axes(ax2)
    ax2.legend(frameon=True, fontsize=9.2, loc="upper left")
    ax2.text(
        0.98,
        0.04,
        "Baseline reference: Persistence = 0 by definition",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.9,
        color="#475569",
        bbox=dict(facecolor="white", edgecolor="#CBD5E1", boxstyle="round,pad=0.3"),
    )

    fig.suptitle("Why Skill_MSE is necessary under a strong persistence baseline", fontsize=14.5, y=0.99)
    return save_figure(fig, FIGURE_NAMES["Fig03"])


def generate_fig04() -> tuple[Path, Path]:
    fig = plt.figure(figsize=(14.5, 7.9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Expert pool", fontsize=12.5, fontweight="bold", color="#0F172A")
    draw_box(ax, (0.03, 0.76), (0.16, 0.11), "Persistence", "#E2E8F0")
    draw_box(ax, (0.03, 0.62), (0.16, 0.11), "Linear-Seasonal", "#FEF3C7")
    draw_box(ax, (0.03, 0.48), (0.16, 0.11), "M1 / C2\nPG-RQ-ACT", "#DBEAFE")
    draw_box(ax, (0.03, 0.34), (0.16, 0.11), "C3_opt\nstate-mixing expert", "#EDE9FE")

    ax.text(0.28, 0.95, "Validation-only static calibration (VCEA)", fontsize=12.5, fontweight="bold", color="#0F172A")
    draw_box(ax, (0.25, 0.70), (0.22, 0.17), "Validation losses per\ntile × horizon\n(no test tuning)", "#ECFCCB")
    draw_box(ax, (0.52, 0.70), (0.22, 0.17), "Static weight table\nw(k | tile, horizon)", "#DCFCE7")
    draw_box(ax, (0.79, 0.70), (0.17, 0.17), "VCEA prediction\nweighted expert mean", "#D1FAE5")
    draw_arrow(ax, (0.19, 0.81), (0.25, 0.81))
    draw_arrow(ax, (0.47, 0.785), (0.52, 0.785))
    draw_arrow(ax, (0.74, 0.785), (0.79, 0.785))

    ax.text(0.53, 0.63, r"$\hat{y}^{\mathrm{VCEA}}_{t+h} = \sum_k w_{k,\mathrm{tile},h}\,\hat{y}^{(k)}_{t+h}$", fontsize=14, ha="center", color="#0F172A")

    ax.text(0.28, 0.50, "Safety wrapper (selected SS-VCEA-COEA_q97p5)", fontsize=12.5, fontweight="bold", color="#0F172A")
    draw_box(ax, (0.24, 0.26), (0.22, 0.16), "Risk components\nexpert disagreement\ncorrection magnitude\nlocal volatility", "#FCE7F3")
    draw_box(ax, (0.50, 0.26), (0.22, 0.16), "Equal-normalized risk\nr = (d + c + v) / 3\nthreshold τ_h from validation", "#FDE68A")
    draw_box(ax, (0.77, 0.26), (0.19, 0.16), "Hard fallback\nif r > τ_h\nuse safer COEA output", "#FEE2E2")
    draw_arrow(ax, (0.875, 0.70), (0.875, 0.42))
    draw_arrow(ax, (0.46, 0.34), (0.50, 0.34))
    draw_arrow(ax, (0.72, 0.34), (0.77, 0.34))

    ax.text(
        0.03,
        0.13,
        "Key design choices:\n"
        "1) VCEA uses validation-only static weights because validation-to-test expert ranking is stable.\n"
        "2) SS-VCEA does not learn a selector; it only applies a validation-calibrated fallback when risk is high.\n"
        "3) The selected deployment variant uses COEA as the safe target and q97.5 thresholding.",
        fontsize=10.3,
        color="#334155",
        va="top",
    )
    ax.text(
        0.62,
        0.13,
        "Operational interpretation:\n"
        "Base mode = accuracy-first aggregation (VCEA)\n"
        "Safe mode = conservative deployment variant (SS-VCEA)\n"
        "Both reuse the same expert pool and leakage-safe calibration protocol.",
        fontsize=10.3,
        color="#334155",
        va="top",
    )
    ax.set_title("VCEA and SS-VCEA: static aggregation with validation-calibrated safety fallback", fontsize=15, pad=12)
    return save_figure(fig, FIGURE_NAMES["Fig04"])


def generate_fig13() -> tuple[Path, Path]:
    audit = pd.read_csv(MANUSCRIPT / "tables" / "ready" / "Table11_architecture_search_audit.csv")
    stages = [
        {
            "Stage": "E1–E8",
            "Best candidate": "Benchmark & diagnostics",
            "Best mean Skill_MSE": np.nan,
            "Verdict": "pass",
            "Reason not final / role": "Established the unified EGMS benchmark, diagnostics, and strong-baseline setting.",
        }
    ] + audit.to_dict(orient="records") + [
        {
            "Stage": "Hardening",
            "Best candidate": "SS-VCEA-COEA_q97p5",
            "Best mean Skill_MSE": 0.1648,
            "Verdict": "pass",
            "Reason not final / role": "Selected as the conservative deployment variant with stress ratio 0.992.",
        }
    ]
    df = pd.DataFrame(stages)
    stage_label_map = {
        "E1–E8": "E1–E8",
        "V2 mainline": "V2\nmainline",
        "V3": "V3",
        "V4": "V4",
        "V4 rescue": "V4\nrescue",
        "V5": "V5",
        "V6": "V6",
        "V7": "V7",
        "StaticVal study": "StaticVal\nstudy",
        "V8": "V8",
        "DSCF": "DSCF",
        "Hardening": "Hardening",
    }
    candidate_label_map = {
        "Benchmark & diagnostics": "benchmark setup",
        "M1 / C2 PG-RQ-ACT": "M1 / C2",
        "V3-C quick": "V3-C",
        "V4-C original": "V4-C",
        "C3_opt": "C3_opt",
        "risk-aware selector": "risk selector",
        "V6B_anchor_risk_l20": "V6B anchor",
        "COEA audited": "COEA",
        "VCEA": "VCEA",
        "STR-Mamba / CoMEx": "STR-Mamba /\nCoMEx",
        "DSCF-Full / VCEA+DSCF": "DSCF-Full /\nVCEA+DSCF",
        "SS-VCEA-COEA_q97p5": "SS-VCEA\nq97.5",
    }
    df["x"] = np.arange(len(df))
    df["y"] = np.where(df.index % 2 == 0, 0.62, 0.28)

    fig = plt.figure(figsize=(18.5, 7.2))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([df["x"].min() - 0.2, df["x"].max() + 0.2], [0.45, 0.45], color="#94A3B8", linewidth=2.2)

    for _, row in df.iterrows():
        color = VERDICT_COLORS.get(row["Verdict"], "#64748B")
        ax.scatter(row["x"], 0.45, s=95, color=color, edgecolor="white", linewidth=1.5, zorder=4)
        box_y = float(row["y"])
        box_h = 0.18
        box = FancyBboxPatch(
            (row["x"] - 0.54, box_y),
            1.08,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=color,
            edgecolor="#0F172A",
            linewidth=1.0,
            alpha=0.94,
        )
        ax.add_patch(box)
        score_text = "benchmark" if pd.isna(row["Best mean Skill_MSE"]) else f"best skill={row['Best mean Skill_MSE']:.3f}"
        verdict_text = row["Verdict"].replace("_", " ")
        stage_text = stage_label_map.get(str(row["Stage"]), str(row["Stage"]))
        cand_text = candidate_label_map.get(str(row["Best candidate"]), str(row["Best candidate"]))
        ax.text(
            row["x"],
            box_y + box_h * 0.66,
            stage_text,
            ha="center",
            va="center",
            fontsize=9.9,
            fontweight="bold",
            color="white",
        )
        ax.text(
            row["x"],
            box_y + box_h * 0.42,
            cand_text,
            ha="center",
            va="center",
            fontsize=8.2,
            color="white",
            wrap=True,
        )
        ax.text(
            row["x"],
            box_y + box_h * 0.17,
            f"{verdict_text} | {score_text}",
            ha="center",
            va="center",
            fontsize=7.8,
            color="white",
        )
        draw_arrow(ax, (row["x"], 0.45), (row["x"], box_y if box_y > 0.45 else box_y + box_h), color="#64748B", lw=1.1)

    ax.text(
        0.02,
        0.96,
        "Model-search audit timeline",
        transform=ax.transAxes,
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        ha="left",
    )
    ax.text(
        0.02,
        0.915,
        "The final manuscript is built on a multi-stage architecture search rather than a single tuned model.",
        transform=ax.transAxes,
        fontsize=10.8,
        color="#334155",
        ha="left",
    )
    ax.text(
        0.02,
        0.04,
        "Search path summary: neural single-model lines (V2–V4) -> selector families (V5–V6) -> online aggregation (V7) -> static validation aggregation (final winner) -> V8/DSCF no-go -> safety hardening.",
        transform=ax.transAxes,
        fontsize=9.7,
        color="#334155",
        ha="left",
    )
    legend_handles = [
        Patch(facecolor=VERDICT_COLORS["pass"], edgecolor="#0F172A", label="Pass / retained"),
        Patch(facecolor=VERDICT_COLORS["fail"], edgecolor="#0F172A", label="Fail / near-pass"),
        Patch(facecolor=VERDICT_COLORS["no_go"], edgecolor="#0F172A", label="No-go"),
        Patch(facecolor=VERDICT_COLORS["final_winner"], edgecolor="#0F172A", label="Final winner"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9.3)
    return save_figure(fig, FIGURE_NAMES["Fig13"])


def refresh_manuscript_indexes(generated: dict[str, Path]) -> None:
    index_path = MANUSCRIPT / "manuscript_asset_index.csv"
    df = pd.read_csv(index_path)
    for fig_id, png_path in generated.items():
        mask = df["asset_id"] == fig_id
        df.loc[mask, "status"] = "ready"
        df.loc[mask, "manuscript_target"] = str(png_path.relative_to(MANUSCRIPT))
        df.loc[mask, "primary_file"] = str(png_path)
        df.loc[mask, "note"] = df.loc[mask, "note"].astype(str) + " Generated manuscript figure is available in figures/final."

    df = df.sort_values(["asset_type", "asset_id"])
    df.to_csv(index_path, index=False, encoding="utf-8-sig")
    df[df["asset_type"] == "figure"].to_csv(MANUSCRIPT / "figure_asset_status.csv", index=False, encoding="utf-8-sig")
    df[df["asset_type"] == "table"].to_csv(MANUSCRIPT / "table_asset_status.csv", index=False, encoding="utf-8-sig")

    missing = df[(df["asset_type"] == "figure") & (df["status"].isin(["needs_new_draw", "partial_ready"]))]
    lines = [
        "# Manuscript Asset Notes",
        "",
        "本目录已经把大部分表格整理为可直接用于写作的 `csv`，并把现有图片、重绘所需源数据和剩余缺口统一整理到一个位置。",
        "",
        "## 仍需新画或二次整理的图片",
        "",
    ]
    if missing.empty:
        lines.append("- 当前 outline 中标记的缺口图已全部补齐。")
    else:
        for _, row in missing.iterrows():
            lines.append(f"- `{row['asset_id']}` | `{row['title']}` | 状态 `{row['status']}` | {row['note']}")
    lines.extend(
        [
            "",
            "## 已可直接使用的核心成图",
            "",
        ]
    )
    ready_rows = df[(df["asset_type"] == "figure") & (df["status"] == "ready")]["asset_id"].tolist()
    for asset_id in ready_rows:
        lines.append(f"- `{asset_id}`")
    (MANUSCRIPT / "missing_or_redraw_items.md").write_text("\n".join(lines), encoding="utf-8")

    ready_tables = int(((df["asset_type"] == "table") & (df["status"] == "ready")).sum())
    ready_figures = int(((df["asset_type"] == "figure") & (df["status"] == "ready")).sum())
    partial_figures = int(((df["asset_type"] == "figure") & (df["status"] == "partial_ready")).sum())
    needs_draw = int(((df["asset_type"] == "figure") & (df["status"] == "needs_new_draw")).sum())
    readme = f"""# TGRS Manuscript Asset Package

生成日期：2026-05-14

目录用途：

- 按 `TGRS_VCEA_SS_VCEA_full_manuscript_writing_outline_final.docx` 统一整理 manuscript 所需图片、表格、源数据和关键支撑文档。
- `tables/ready` 下是当前可直接用于写作的整理版表格。
- `figures/final` 下是已正式生成的成图文件；`figures/source_bundles` 下保留原始 source bundle。
- `source_docs` 下保留提纲和关键总报告，便于写作时快速查证。

统计：

- 资产索引总数：{len(df)}
- ready figure：{ready_figures}
- partial-ready figure：{partial_figures}
- needs-new-draw figure：{needs_draw}
- ready table：{ready_tables}

重点提醒：

- `Fig06 / Fig07 / Fig14` 仍建议按提纲进一步重排 panel。
- `Fig01 / Fig02 / Fig03 / Fig04 / Fig13` 已生成正式版成图，并同步更新到图索引。"""
    (MANUSCRIPT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    generated = {}
    for fig_id, func in [
        ("Fig01", generate_fig01),
        ("Fig02", generate_fig02),
        ("Fig03", generate_fig03),
        ("Fig04", generate_fig04),
        ("Fig13", generate_fig13),
    ]:
        png_path, _pdf_path = func()
        generated[fig_id] = png_path

    refresh_manuscript_indexes(generated)
    print(FIG_FINAL)


if __name__ == "__main__":
    main()
