"""Generate publication-quality 300 dpi figures for the IJRS manuscript.

All data come strictly from ../tables/ and ../../analysis_vcea_comparators_20261104/results/.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent
TABLES = GUIDE / "tables"
ROOT = GUIDE.parent
COMP_RESULTS = ROOT / "analysis_vcea_comparators_20261104" / "results"
API_CATALOG = (
    ROOT
    / "outputs"
    / "tgrs_2019_2023"
    / "vcea_extra_experiments"
    / "E7_pan_europe_locked_audit_strong"
    / "manifests"
    / "table_E7_api_tile_catalog.csv"
)
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_both(fig: plt.Figure, stem: str) -> None:
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path.name} and {pdf_path.name}")


# ---------------------------------------------------------------------------
# Figure 1: Three-tier evaluation design, temporal splits, and pre-registration
# ---------------------------------------------------------------------------
def make_fig1_study_design() -> None:
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.95], hspace=0.28)

    # Panel (a): Three-tier evaluation hierarchy + pre-registration timeline
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 36)
    ax1.axis("off")
    ax1.text(
        0.5, 34.5,
        "(a) Three-tier evaluation hierarchy and pre-registered confirmatory timeline",
        fontsize=11, fontweight="bold", color="#0F172A", ha="left", va="center"
    )

    tiers = [
        {
            "x": 1.0, "y": 10.6, "w": 30.0, "h": 21.0,
            "fc": "#EFF6FF", "ec": "#1D4ED8",
            "title": "Tier 1: Development Set\n(6 tiles, 2019–2023 release)",
            "body": (
                "• E32N34, E32N35, E39N30,\n"
                "  E44N23, E48N24, E58N17\n"
                "• Architecture & hyperparameter\n"
                "  tuning (τ = 4, ρ = 0.1)\n"
                "• Split-protocol leakage audit\n"
                "• Variant screen (V1–V3)"
            ),
        },
        {
            "x": 35.0, "y": 10.6, "w": 30.0, "h": 21.0,
            "fc": "#FFFBEB", "ec": "#B45309",
            "title": "Tier 2: Held-out Audit (E7)\n(14 tiles, 2019–2023 release)",
            "body": (
                "• Stratified across 6 European\n"
                "  regions & 3 coverage classes\n"
                "• Model weights & VCEA frozen\n"
                "• Classical comparators added\n"
                "  post-hoc (exploratory audit)\n"
                "• Includes short-series tiles\n"
                "  (1–4 validation windows)"
            ),
        },
        {
            "x": 69.0, "y": 10.6, "w": 30.0, "h": 21.0,
            "fc": "#ECFDF5", "ec": "#047857",
            "title": "Tier 3: Confirmatory Test\n(24 tiles, 2020–2024 release)",
            "body": (
                "• Stratified 4×3 lat/lon × 2 size\n"
                "  strata (RNG seed 20261104)\n"
                "• ≥2 grid-steps from prior 24 tiles\n"
                "• Pre-registered hypotheses H1–H4\n"
                "• Single-shot test evaluation\n"
                "  after Amendment 01"
            ),
        },
    ]

    for t in tiers:
        box = FancyBboxPatch(
            (t["x"], t["y"]), t["w"], t["h"],
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=t["fc"], edgecolor=t["ec"], linewidth=1.6
        )
        ax1.add_patch(box)
        ax1.text(
            t["x"] + t["w"] / 2, t["y"] + t["h"] - 3.3,
            t["title"], ha="center", va="center",
            fontsize=9.3, fontweight="bold", color=t["ec"]
        )
        ax1.plot(
            [t["x"] + 1.5, t["x"] + t["w"] - 1.5],
            [t["y"] + t["h"] - 6.3, t["y"] + t["h"] - 6.3],
            color=t["ec"], lw=0.8, alpha=0.5
        )
        ax1.text(
            t["x"] + 1.6, t["y"] + t["h"] - 7.0,
            t["body"], ha="left", va="top",
            fontsize=8.1, color="#1E293B", linespacing=1.22
        )

    for x0, x1 in [(31.3, 34.7), (65.3, 68.7)]:
        arr = FancyArrowPatch(
            (x0, 21.5), (x1, 21.5),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.8, color="#475569"
        )
        ax1.add_patch(arr)

    # Pre-registration UTC timeline bar at bottom of Panel (a)
    ax1.plot([4, 96], [4.5, 4.5], color="#334155", lw=2.0, zorder=1)
    milestones = [
        (10, "2026-11-04 12:58 UTC", "Pre-registration & 24-tile list\nlocked (SHA-256 recorded)", "#1D4ED8"),
        (38, "2026-11-04 12:58 UTC", "V3_EvidenceGate frozen\nfrom dev screen only", "#4338CA"),
        (66, "2026-11-05 16:16 UTC", "Amendment 01 (2020–2024 release)\nbefore first download (16:19 UTC)", "#B45309"),
        (91, "2026-11-06 04:23 UTC", "Single confirmatory run\n(Wilcoxon + Holm H1–H4)", "#047857"),
    ]
    for mx, mtime, mdesc, mcol in milestones:
        ax1.scatter([mx], [4.5], s=70, color=mcol, edgecolor="white", linewidth=1.5, zorder=3)
        ax1.text(mx, 6.2, mtime, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=mcol)
        ax1.text(mx, 3.0, mdesc, ha="center", va="top", fontsize=7.6, color="#1E293B", linespacing=1.15)

    # Panel (b): Leakage-safe temporal split and window-containment audit protocols
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 30)
    ax2.axis("off")
    ax2.text(
        0.5, 28.2,
        "(b) Temporal split structure (6-day sampling, lookback L = 16, horizons h ∈ {1, 3, 5, 10}) and split-containment audit",
        fontsize=11, fontweight="bold", color="#0F172A", ha="left", va="center"
    )

    protocols = [
        (
            "target_time_default\n(Baseline 70 / 10 / 20)",
            20.0,
            [(18, 30, "#DBEAFE", "#1E40AF", "Train split (70%)\nFit LS + neural experts"),
             (48, 8.5, "#FEF3C7", "#B45309", "Val (10%)\nCalibrate w"),
             (56.5, 13.5, "#DCFCE7", "#15803D", "Test (20%)\nOut-of-sample")],
            "Split by target t+h; first ~16 windows use lookback across boundary"
        ),
        (
            "window_contained\n(Strict containment)",
            11.5,
            [(18, 27.5, "#DBEAFE", "#1E40AF", "Train split"),
             (45.5, 2.5, "#E2E8F0", "#64748B", "L"),
             (48, 8.5, "#FEF3C7", "#B45309", "Val"),
             (56.5, 2.5, "#E2E8F0", "#64748B", "L"),
             (59, 11.0, "#DCFCE7", "#15803D", "Test")],
            "Entire [t−L+1, t+h] window inside split (+0.079 VCEA gain over M1)"
        ),
        (
            "window_contained\n_gap10 (+60-day buffer)",
            3.0,
            [(18, 26.0, "#DBEAFE", "#1E40AF", "Train split"),
             (44, 4.0, "#FECACA", "#B91C1C", "L+10"),
             (48, 8.0, "#FEF3C7", "#B45309", "Val"),
             (56, 4.0, "#FECACA", "#B91C1C", "L+10"),
             (60, 10.0, "#DCFCE7", "#15803D", "Test")],
            "Adds 10-epoch (60-day) buffer before val & test (+0.085 VCEA gain)"
        ),
    ]

    for label, y0, segs, note in protocols:
        ax2.text(0.8, y0 + 2.6, label, ha="left", va="center", fontsize=8.1, fontweight="bold", color="#0F172A")
        for sx, sw, sfc, sec, stxt in segs:
            rect = FancyBboxPatch(
                (sx, y0), sw, 5.2,
                boxstyle="round,pad=0.1,rounding_size=0.4",
                facecolor=sfc, edgecolor=sec, linewidth=1.2
            )
            ax2.add_patch(rect)
            ax2.text(
                sx + sw / 2, y0 + 2.6, stxt,
                ha="center", va="center", fontsize=7.4, color=sec, fontweight="bold"
            )
        ax2.text(71.5, y0 + 2.6, note, ha="left", va="center", fontsize=7.7, color="#334155", style="italic")

    save_both(fig, "fig1_study_design")


# ---------------------------------------------------------------------------
# Figure 2: Spatial distribution of the three tile sets across Europe
# ---------------------------------------------------------------------------
def make_fig2_tile_map() -> None:
    cat = pd.read_csv(API_CATALOG)
    cat_unique = cat.drop_duplicates(subset=["tile_id"]).copy()

    t1a = pd.read_csv(TABLES / "T1a_e7_tiles.csv").merge(
        cat_unique[["tile_id", "centroid_lon", "centroid_lat"]], on="tile_id", how="left"
    )
    t1b = pd.read_csv(TABLES / "T1b_fresh_tiles.csv")

    dev_tiles = ["E32N34", "E32N35", "E39N30", "E44N23", "E48N24", "E58N17"]
    dev_df = cat_unique[cat_unique["tile_id"].isin(dev_tiles)][["tile_id", "centroid_lon", "centroid_lat"]].copy()

    fig, ax = plt.subplots(figsize=(11.2, 7.6))

    # Plot all 775 EGMS L3 tiles as light land background grid
    ax.scatter(
        cat_unique["centroid_lon"], cat_unique["centroid_lat"],
        s=26, marker="s", color="#E2E8F0", edgecolor="#CBD5E1",
        linewidth=0.4, alpha=0.85, label="All available EGMS L3 tiles (n = 775)", zorder=1
    )

    # Latitude and longitude stratification lines used for confirmatory sampling
    for lat_b in [42.0, 48.0, 55.0]:
        ax.axhline(lat_b, color="#94A3B8", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)
    for lon_b in [5.0, 15.0]:
        ax.axvline(lon_b, color="#94A3B8", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)

    # Plot Tier 1: Development 6 tiles
    ax.scatter(
        dev_df["centroid_lon"], dev_df["centroid_lat"],
        s=115, marker="^", color="#1D4ED8", edgecolor="#0F172A",
        linewidth=1.1, label="Tier 1: Development set (n = 6, 2019–2023)", zorder=5
    )

    # Plot Tier 2: Held-out E7 audit 14 tiles
    ax.scatter(
        t1a["centroid_lon"], t1a["centroid_lat"],
        s=105, marker="s", color="#D97706", edgecolor="#0F172A",
        linewidth=1.1, label="Tier 2: Held-out E7 audit (n = 14, 2019–2023)", zorder=4
    )

    # Plot Tier 3: Pre-registered confirmatory 24 tiles
    ax.scatter(
        t1b["centroid_lon"], t1b["centroid_lat"],
        s=110, marker="o", color="#059669", edgecolor="#0F172A",
        linewidth=1.1, label="Tier 3: Confirmatory test (n = 24, 2020–2024)", zorder=6
    )

    # Highlight Nordic 1-window & low-coverage diagnostic tiles with open outer rings
    special_Ids = {"E40N42", "E41N44", "E42N41", "E46N48", "E32N37", "E44N43"}
    spec_fresh = t1b[t1b["tile_id"].isin(special_Ids)]
    spec_e7 = t1a[t1a["tile_id"].isin(special_Ids)]
    ax.scatter(
        list(spec_fresh["centroid_lon"]) + list(spec_e7["centroid_lon"]),
        list(spec_fresh["centroid_lat"]) + list(spec_e7["centroid_lat"]),
        s=230, facecolors="none", edgecolors="#DC2626",
        linewidth=1.5, linestyle="-",
        label="Boundary cases (1 val window or <15% coverage)", zorder=7
    )

    # Smart label offsets to avoid overlapping text
    offsets = {
        "E32N34": (0.5, -0.65),
        "E32N35": (-2.4, 0.25),
        "E32N37": (-2.4, 0.35),
        "E34N39": (0.45, 0.25),
        "E36N27": (-2.3, -0.45),
        "E36N24": (-2.3, -0.35),
        "E27N24": (-2.3, 0.35),
        "E39N30": (-2.3, 0.25),
        "E40N31": (0.45, 0.25),
        "E39N27": (0.45, -0.45),
        "E39N23": (-2.3, -0.45),
        "E40N21": (0.45, -0.35),
        "E44N23": (-2.3, -0.45),
        "E45N26": (-2.3, 0.25),
        "E48N26": (0.45, 0.25),
        "E48N24": (0.45, -0.45),
        "E49N22": (0.45, -0.45),
        "E47N20": (-2.3, -0.45),
        "E48N32": (0.45, -0.45),
        "E49N35": (0.45, 0.25),
        "E47N35": (-2.3, 0.25),
        "E42N34": (-2.3, -0.45),
        "E44N30": (0.45, -0.45),
        "E52N28": (0.45, 0.25),
        "E53N31": (0.45, 0.25),
        "E53N35": (0.45, 0.25),
        "E54N38": (0.45, 0.25),
        "E57N25": (-2.4, -0.55),
        "E58N26": (0.45, 0.25),
        "E55N22": (0.45, 0.25),
        "E58N17": (0.45, 0.25),
        "E55N14": (0.45, -0.35),
        "E64N14": (-2.3, 0.35),
        "E31N16": (0.45, -0.35),
        "E40N42": (-2.3, 0.25),
        "E41N44": (-2.3, 0.30),
        "E42N41": (0.45, -0.45),
        "E44N43": (0.45, -0.45),
        "E47N43": (0.45, -0.45),
        "E49N44": (0.45, 0.25),
        "E46N48": (0.45, 0.25),
    }

    all_pts = []
    for _, r in dev_df.iterrows():
        all_pts.append((r["tile_id"], r["centroid_lon"], r["centroid_lat"], "#1E3A8A"))
    for _, r in t1a.iterrows():
        all_pts.append((r["tile_id"], r["centroid_lon"], r["centroid_lat"], "#92400E"))
    for _, r in t1b.iterrows():
        all_pts.append((r["tile_id"], r["centroid_lon"], r["centroid_lat"], "#065F46"))

    for tid, lon, lat, col in all_pts:
        dx, dy = offsets.get(tid, (0.42, 0.22))
        ax.text(
            lon + dx, lat + dy, tid,
            fontsize=7.4, fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
            zorder=8
        )

    # Annotate stratification bands
    ax.text(-11.8, 38.0, "34–42°N", fontsize=8.0, color="#64748B", style="italic")
    ax.text(-11.8, 45.0, "42–48°N", fontsize=8.0, color="#64748B", style="italic")
    ax.text(-11.8, 51.5, "48–55°N", fontsize=8.0, color="#64748B", style="italic")
    ax.text(-11.8, 60.5, "55–72°N", fontsize=8.0, color="#64748B", style="italic")
    ax.text(-3.5, 34.0, "West (−11–5°E)", fontsize=8.0, color="#64748B", style="italic")
    ax.text(8.5, 34.0, "Central (5–15°E)", fontsize=8.0, color="#64748B", style="italic")
    ax.text(22.0, 34.0, "East (15–35°E)", fontsize=8.0, color="#64748B", style="italic")

    ax.set_xlim(-12.5, 36.0)
    ax.set_ylim(33.2, 70.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        "Spatial distribution of the 44 evaluated EGMS Ortho Vertical (L3) tiles across Europe",
        fontsize=11, fontweight="bold", pad=10
    )
    ax.grid(True, linestyle=":", alpha=0.45, color="#94A3B8")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.3)

    save_both(fig, "fig2_tile_map")


# ---------------------------------------------------------------------------
# Figure 3: Paired tile-level Skill_MSE differences on 24 confirmatory tiles
# ---------------------------------------------------------------------------
def make_fig3_confirmatory_paired_differences() -> None:
    df = pd.read_csv(TABLES / "skill_by_set_model_tile_horizon.csv")
    fresh = df[df["Set"] == "fresh"].groupby(["Model", "Tile"], as_index=False)["Skill_MSE"].mean()
    piv = fresh.pivot(index="Tile", columns="Model", values="Skill_MSE")

    # Sort tiles by Bates-Granger minus M1
    order = (piv["BatesGranger"] - piv["M1"]).sort_values().index.tolist()
    piv = piv.loc[order]

    panels = [
        ("BatesGranger", "SelectBestVal", "(a) Bates–Granger minus Validation Selection\n(+0.109 mean, 21/24 tiles > 0, $p_{\\mathrm{Holm}} = 3.0\\times 10^{-5}$)"),
        ("BatesGranger", "Uniform4", "(b) Bates–Granger minus Equal Weighting\n(+0.084 mean, 24/24 tiles > 0, $p_{\\mathrm{Holm}} = 2.4\\times 10^{-7}$)"),
        ("BatesGranger", "M1", "(c) Bates–Granger minus M1 (neural backbone)\n(+0.155 mean, 23/24 tiles > 0, $p_{\\mathrm{Holm}} = 5.4\\times 10^{-7}$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.2), sharey=True)
    y = np.arange(len(order))

    for idx, (ax, (a, b, title)) in enumerate(zip(axes, panels)):
        diff = (piv[a] - piv[b]).to_numpy()
        colors = ["#0284C7" if v >= 0 else "#DC2626" for v in diff]
        ax.barh(y, diff, color=colors, edgecolor="#0F172A", linewidth=0.5, height=0.72, alpha=0.9)
        ax.axvline(0.0, color="#0F172A", linewidth=0.9)
        mean_val = float(np.mean(diff))
        ax.axvline(mean_val, color="#0F172A", linestyle="--", linewidth=1.1, label=f"Mean = {mean_val:+.3f}")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
        ax.set_xlabel(r"$\Delta\,\mathrm{Skill}_{\mathrm{MSE}}$")
        ax.grid(axis="x", linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", frameon=True, fontsize=8.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=8.2)
    axes[0].set_ylabel("Pre-registered confirmatory tile (2020–2024 release)")

    fig.tight_layout()
    save_both(fig, "fig3_confirmatory_paired_differences")


# ---------------------------------------------------------------------------
# Figure 4: Horizon profiles across all three tile sets
# ---------------------------------------------------------------------------
def make_fig4_horizon_profiles() -> None:
    t5 = pd.read_csv(TABLES / "T5_horizon_by_set.csv")
    horizons = ["t+1", "t+3", "t+5", "t+10"]
    x = np.array([1, 3, 5, 10])
    x_labels = ["t+1\n(6 d)", "t+3\n(18 d)", "t+5\n(30 d)", "t+10\n(60 d)"]

    sets = [
        ("main6", "(a) Tier 1: Development (6 tiles, 2019–2023)"),
        ("e7", "(b) Tier 2: Held-out audit E7 (14 tiles, 2019–2023)"),
        ("fresh", "(c) Tier 3: Confirmatory (24 tiles, 2020–2024)"),
    ]
    models = [
        ("BatesGranger", "Bates–Granger (τ=1, ρ=0)", "#0284C7", "o", "-"),
        ("VCEA", "VCEA (τ=4, ρ=0.1)", "#059669", "s", "--"),
        ("NNLS_simplex", "Constrained Stacking (NNLS)", "#7C3AED", "^", "-."),
        ("SelectBestVal", "Validation Selection", "#D97706", "D", ":"),
        ("M1", "M1 (gated patch Transformer)", "#DC2626", "v", "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharey=True)
    for ax, (set_key, title) in zip(axes, sets):
        sub = t5[t5["Set"] == set_key].set_index("Model")
        ax.axhline(0.0, color="#64748B", linewidth=0.9, linestyle=":")
        for m_key, m_label, col, marker, ls in models:
            if m_key not in sub.index:
                continue
            vals = [float(sub.loc[m_key, h]) for h in horizons]
            ax.plot(
                x, vals, label=m_label, color=col, marker=marker,
                linestyle=ls, linewidth=1.8, markersize=5.5
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Forecast horizon (6-day epochs / days)")
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.45)

    axes[0].set_ylabel(r"Mean $\mathrm{Skill}_{\mathrm{MSE}}$ vs Persistence")
    axes[0].set_ylim(-0.04, 0.44)
    axes[2].legend(loc="upper left", frameon=True, fontsize=8.0)

    fig.tight_layout()
    save_both(fig, "fig4_horizon_profiles_three_sets")


# ---------------------------------------------------------------------------
# Figure 5: Validation-calibrated expert weights across horizons and tiles
# ---------------------------------------------------------------------------
def make_fig5_expert_weights_heatmap() -> None:
    wdf = pd.read_csv(COMP_RESULTS / "weights_by_tile_seed_horizon.csv")
    vcea_w = wdf[wdf["Model"] == "VCEA"].copy()

    horizons = ["t+1", "t+3", "t+5", "t+10"]
    experts = [
        ("w_P", "Persistence"),
        ("w_LS", "Linear-Seasonal"),
        ("w_M1", "M1 (Transformer)"),
        ("w_C3", "C3_opt (CNN)"),
    ]

    # Order tiles: 6 dev tiles first, then 14 E7 tiles
    dev_order = ["E32N34", "E32N35", "E39N30", "E44N23", "E48N24", "E58N17"]
    e7_order = sorted(vcea_w[vcea_w["Set"] == "e7"]["Tile"].unique().tolist())
    all_tiles = dev_order + e7_order

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 6.4), sharey=True)
    grouped = vcea_w.groupby(["Tile", "Horizon"])[["w_P", "w_LS", "w_M1", "w_C3"]].mean()

    im = None
    for idx, (ax, (w_col, exp_title)) in enumerate(zip(axes, experts)):
        mat = np.zeros((len(all_tiles), len(horizons)))
        for i, t in enumerate(all_tiles):
            for j, h in enumerate(horizons):
                mat[i, j] = grouped.loc[(t, h), w_col]

        im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=0.70, aspect="auto")
        ax.set_title(f"({chr(97+idx)}) {exp_title}", fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(["t+1", "t+3", "t+5", "t+10"])
        ax.set_xlabel("Horizon")

        # Draw horizontal divider between Tier 1 (dev 6) and Tier 2 (E7 14)
        ax.axhline(5.5, color="#DC2626", linewidth=1.5, linestyle="--")

        for i in range(len(all_tiles)):
            for j in range(len(horizons)):
                val = mat[i, j]
                txt_col = "white" if val > 0.40 else "#0F172A"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.4, color=txt_col)

    axes[0].set_yticks(range(len(all_tiles)))
    y_labels = [f"{t} (Dev)" if t in dev_order else f"{t} (E7)" for t in all_tiles]
    axes[0].set_yticklabels(y_labels, fontsize=8.2)
    axes[0].set_ylabel("Tile ID and evaluation tier")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Validation-calibrated weight (mean across 3 seeds)", fontsize=9.0)

    save_both(fig, "fig5_expert_weights_heatmap")


# ---------------------------------------------------------------------------
# Figure 6: Operating boundaries (validation window count & valid coverage)
# ---------------------------------------------------------------------------
def make_fig6_operating_boundaries() -> None:
    t1a = pd.read_csv(TABLES / "T1a_e7_tiles.csv")
    def calc_val_win(T: int) -> int:
        n_tr = int(T * 0.70)
        n_va = int(T * 0.80)
        return max(1, n_va - n_tr - 10)
    t1a["val_windows"] = t1a["frames"].apply(calc_val_win)
    t1b = pd.read_csv(TABLES / "T1b_fresh_tiles.csv")
    skill = pd.read_csv(TABLES / "skill_by_set_model_tile_horizon.csv")

    # Confirmatory tile means
    f_skill = skill[skill["Set"] == "fresh"].groupby(["Model", "Tile"], as_index=False)["Skill_MSE"].mean()
    f_piv = f_skill.pivot(index="Tile", columns="Model", values="Skill_MSE")
    f_df = t1b.set_index("tile_id").join(f_piv)
    f_df["BG_minus_Select"] = f_df["BatesGranger"] - f_df["SelectBestVal"]
    f_df["BG_minus_M1"] = f_df["BatesGranger"] - f_df["M1"]

    # E7 tile means
    e_skill = skill[skill["Set"] == "e7"].groupby(["Model", "Tile"], as_index=False)["Skill_MSE"].mean()
    e_piv = e_skill.pivot(index="Tile", columns="Model", values="Skill_MSE")
    e_df = t1a.set_index("tile_id").join(e_piv)
    e_df["BG_minus_Select"] = e_df["BatesGranger"] - e_df["SelectBestVal"]
    e_df["BG_minus_M1"] = e_df["BatesGranger"] - e_df["M1"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # Panel (a): Combination gain over single-expert validation selection vs validation windows
    ax1.axhline(0.0, color="#0F172A", linewidth=0.9, linestyle=":")
    ax1.axvspan(0.4, 4.6, color="#FEF2F2", alpha=0.8, label="Short-series Nordic regime (1–4 val windows)")

    ax1.scatter(
        e_df["val_windows"], e_df["BG_minus_Select"],
        s=75, marker="s", color="#D97706", edgecolor="#0F172A", linewidth=0.9,
        alpha=0.9, label="Tier 2: Held-out E7 audit (14 tiles)"
    )
    ax1.scatter(
        f_df["val_windows"], f_df["BG_minus_Select"],
        s=80, marker="o", color="#059669", edgecolor="#0F172A", linewidth=0.9,
        alpha=0.9, label="Tier 3: Confirmatory test (24 tiles)"
    )

    # Annotate key tiles in Panel (a)
    annot_a = [
        ("E41N44 (+0.200)", f_df.loc["E41N44", "val_windows"], f_df.loc["E41N44", "BG_minus_Select"], (1.4, 0.035)),
        ("E40N42 (+0.143)", f_df.loc["E40N42", "val_windows"], f_df.loc["E40N42", "BG_minus_Select"], (1.4, 0.025)),
        ("E42N41 (-0.011)", f_df.loc["E42N41", "val_windows"], f_df.loc["E42N41", "BG_minus_Select"], (1.4, -0.045)),
        ("E46N48 (-0.106)", f_df.loc["E46N48", "val_windows"], f_df.loc["E46N48", "BG_minus_Select"], (1.4, 0.02)),
        ("E44N43 (E7, -0.110)", e_df.loc["E44N43", "val_windows"], e_df.loc["E44N43", "BG_minus_Select"], (5.2, -0.015)),
        ("E47N43 (4 win)", f_df.loc["E47N43", "val_windows"], f_df.loc["E47N43", "BG_minus_Select"], (2.5, -0.045)),
        ("E32N37 (-0.025)", f_df.loc["E32N37", "val_windows"], f_df.loc["E32N37", "BG_minus_Select"], (-5.5, -0.065)),
    ]
    for txt, xv, yv, (dx, dy) in annot_a:
        ax1.annotate(
            txt, xy=(xv, yv), xytext=(xv + dx, yv + dy),
            fontsize=7.6, fontweight="bold", color="#0F172A",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#475569")
        )

    ax1.set_xlabel("Number of validation windows (val_windows)")
    ax1.set_ylabel(r"$\Delta\,\mathrm{Skill}_{\mathrm{MSE}}$ (Bates–Granger minus Validation Selection)")
    ax1.set_title("(a) Sensitivity to validation sample size", fontsize=10.2, fontweight="bold", pad=10)
    ax1.set_xlim(0, 22.5)
    ax1.set_ylim(-0.16, 0.56)
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, fontsize=7.9)

    # Panel (b): Absolute Bates-Granger Skill_MSE vs valid 64x64 grid coverage
    ax2.axhline(0.0, color="#0F172A", linewidth=0.9, linestyle=":")
    ax2.axvspan(8.0, 20.0, color="#FFFBEB", alpha=0.85, label="Sparse coastal/island coverage (<20% valid cells)")

    ax2.scatter(
        e_df["valid_grid_ratio_64"] * 100.0, e_df["BatesGranger"],
        s=75, marker="s", color="#D97706", edgecolor="#0F172A", linewidth=0.9,
        alpha=0.9, label="Tier 2: Held-out E7 audit (Bates–Granger)"
    )
    ax2.scatter(
        f_df["valid_grid_ratio_64"] * 100.0, f_df["BatesGranger"],
        s=80, marker="o", color="#059669", edgecolor="#0F172A", linewidth=0.9,
        alpha=0.9, label="Tier 3: Confirmatory test (Bates–Granger)"
    )

    annot_b = [
        ("E32N37 (11.8% valid,\nBG=+0.043, Unif=-0.390)", f_df.loc["E32N37", "valid_grid_ratio_64"] * 100.0, f_df.loc["E32N37", "BatesGranger"], (6.0, 0.14)),
        ("E58N26 (E7, 16.8%)", e_df.loc["E58N26", "valid_grid_ratio_64"] * 100.0, e_df.loc["E58N26", "BatesGranger"], (6.0, -0.055)),
        ("E49N22 (E7, 28.0%)", e_df.loc["E49N22", "valid_grid_ratio_64"] * 100.0, e_df.loc["E49N22", "BatesGranger"], (5.0, -0.05)),
        ("E41N44 (38.8%, 1 win)", f_df.loc["E41N44", "valid_grid_ratio_64"] * 100.0, f_df.loc["E41N44", "BatesGranger"], (-12.0, 0.085)),
        ("E44N43 (E7, 1 win)", e_df.loc["E44N43", "valid_grid_ratio_64"] * 100.0, e_df.loc["E44N43", "BatesGranger"], (-32.0, -0.02)),
    ]
    for txt, xv, yv, (dx, dy) in annot_b:
        ax2.annotate(
            txt, xy=(xv, yv), xytext=(xv + dx, yv + dy),
            fontsize=7.6, fontweight="bold", color="#0F172A",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#475569")
        )

    ax2.set_xlabel("Valid 64×64 grid cell coverage (%)")
    ax2.set_ylabel(r"Tile-level $\mathrm{Skill}_{\mathrm{MSE}}$ (Bates–Granger)")
    ax2.set_title("(b) Sensitivity to valid spatial grid coverage", fontsize=10.2, fontweight="bold", pad=10)
    ax2.set_xlim(8, 104)
    ax2.set_ylim(-0.04, 0.70)
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.9)

    fig.tight_layout()
    save_both(fig, "fig6_operating_boundaries")


if __name__ == "__main__":
    make_fig1_study_design()
    make_fig2_tile_map()
    make_fig3_confirmatory_paired_differences()
    make_fig4_horizon_profiles()
    make_fig5_expert_weights_heatmap()
    make_fig6_operating_boundaries()
    print("All 6 figures generated successfully.")
