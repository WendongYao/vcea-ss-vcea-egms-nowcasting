from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
EXTRAS_ROOT = PROJECT_ROOT / "outputs" / "tgrs_2019_2023" / "vcea_extra_experiments"
STRONG_ROOT = EXTRAS_ROOT / "E7_pan_europe_locked_audit_strong"
OUT_ROOT = PROJECT_ROOT / "paper_assets_vcea_extras"


@dataclass(frozen=True)
class AssetSpec:
    key: str
    experiment_id: str
    source_path: Path
    target_subdir: str
    asset_type: str
    intended_location: str
    short_description: str


E1_E6_TABLES = [
    AssetSpec("table_E1_leakage_audit.csv", "E1", EXTRAS_ROOT / "tables" / "table_E1_leakage_audit.csv", "tables_csv_original/E1_E6", "table_csv", "main_text", "Leakage audit protocols and resulting VCEA/M1 skill comparison."),
    AssetSpec("table_E2_expert_ablation_external.csv", "E2", EXTRAS_ROOT / "tables" / "table_E2_expert_ablation_external.csv", "tables_csv_original/E1_E6", "table_csv", "appendix", "Expert-pool ablation on the original external evaluation set."),
    AssetSpec("table_E2_expert_ablation_main6.csv", "E2", EXTRAS_ROOT / "tables" / "table_E2_expert_ablation_main6.csv", "tables_csv_original/E1_E6", "table_csv", "main_text", "Expert-pool ablation on the main-six setting."),
    AssetSpec("table_E3_calibration_ablation.csv", "E3", EXTRAS_ROOT / "tables" / "table_E3_calibration_ablation.csv", "tables_csv_original/E1_E6", "table_csv", "main_text", "Calibration ablation across uniform, global, and tile-horizon weighting strategies."),
    AssetSpec("table_E4_blocked_validation.csv", "E4", EXTRAS_ROOT / "tables" / "table_E4_blocked_validation.csv", "tables_csv_original/E1_E6", "table_csv", "appendix", "Blocked-validation results by block, tile, and seed."),
    AssetSpec("table_E5_external_regime_audit.csv", "E5", EXTRAS_ROOT / "tables" / "table_E5_external_regime_audit.csv", "tables_csv_original/E1_E6", "table_csv", "appendix", "Original external audit grouped by coverage and regime notes."),
    AssetSpec("table_E6_ssvcea_threshold_sweep.csv", "E6", EXTRAS_ROOT / "tables" / "table_E6_ssvcea_threshold_sweep.csv", "tables_csv_original/E1_E6", "table_csv", "main_text", "Stress-safe threshold sweep for SS-VCEA."),
]

E1_E6_FIGURES = [
    AssetSpec("fig_E1_protocol_timeline.png", "E1", EXTRAS_ROOT / "figures" / "fig_E1_protocol_timeline.png", "figures/E1", "figure_png", "appendix", "Schematic comparison of target-time, containment, and gap-buffer split protocols."),
    AssetSpec("fig_E1_skill_delta_by_tile.png", "E1", EXTRAS_ROOT / "figures" / "fig_E1_skill_delta_by_tile.png", "figures/E1", "figure_png", "main_text", "Tile-wise VCEA gain over M1 under stricter leakage-audit protocols."),
    AssetSpec("fig_E2_expert_weights_heatmap.png", "E2", EXTRAS_ROOT / "figures" / "fig_E2_expert_weights_heatmap.png", "figures/E2", "figure_png", "main_text", "Validation-calibrated expert-weight heatmap over tile and horizon."),
    AssetSpec("fig_E2_remove_expert_delta.png", "E2", EXTRAS_ROOT / "figures" / "fig_E2_remove_expert_delta.png", "figures/E2", "figure_png", "appendix", "Skill and stress deltas after removing experts from the VCEA pool."),
    AssetSpec("fig_E3_calibration_barplot.png", "E3", EXTRAS_ROOT / "figures" / "fig_E3_calibration_barplot.png", "figures/E3", "figure_png", "main_text", "Calibration ablation summary across main-six and external settings."),
    AssetSpec("fig_E3_val_vs_test_weights.png", "E3", EXTRAS_ROOT / "figures" / "fig_E3_val_vs_test_weights.png", "figures/E3", "figure_png", "appendix", "Relationship between validation-derived and oracle test weights."),
    AssetSpec("fig_E4_test_skill_by_block.png", "E4", EXTRAS_ROOT / "figures" / "fig_E4_test_skill_by_block.png", "figures/E4", "figure_png", "appendix", "Test skill distribution under early, middle, and late validation blocks."),
    AssetSpec("fig_E4_weight_stability.png", "E4", EXTRAS_ROOT / "figures" / "fig_E4_weight_stability.png", "figures/E4", "figure_png", "appendix", "Weight drift from full-validation calibration under blocked validation."),
    AssetSpec("fig_E5_external_map_regime.png", "E5", EXTRAS_ROOT / "figures" / "fig_E5_external_map_regime.png", "figures/E5", "figure_png", "appendix", "Original external locked tiles mapped with dominant-regime annotations."),
    AssetSpec("fig_E5_skill_vs_coverage.png", "E5", EXTRAS_ROOT / "figures" / "fig_E5_skill_vs_coverage.png", "figures/E5", "figure_png", "appendix", "External skill versus valid coverage for the original external audit."),
    AssetSpec("fig_E6_fallback_rate_by_tile.png", "E6", EXTRAS_ROOT / "figures" / "fig_E6_fallback_rate_by_tile.png", "figures/E6", "figure_png", "appendix", "Fallback rate by tile and threshold under SS-VCEA."),
    AssetSpec("fig_E6_skill_stress_frontier.png", "E6", EXTRAS_ROOT / "figures" / "fig_E6_skill_stress_frontier.png", "figures/E6", "figure_png", "main_text", "Skill-stress trade-off frontier for SS-VCEA threshold selection."),
]

E7_STRONG_TABLES = [
    AssetSpec("table_E7_pan_europe_failure_cases.csv", "E7_strong", STRONG_ROOT / "csv" / "table_E7_pan_europe_failure_cases.csv", "tables_csv_original/E7_strong", "table_csv", "main_text", "Strong E7 failure cases under the frozen pan-European audit."),
    AssetSpec("table_E7_pan_europe_group_summary.csv", "E7_strong", STRONG_ROOT / "csv" / "table_E7_pan_europe_group_summary.csv", "tables_csv_original/E7_strong", "table_csv", "main_text", "Strong E7 group summary by region, regime, and coverage strata."),
    AssetSpec("table_E7_pan_europe_locked_results_by_tile.csv", "E7_strong", STRONG_ROOT / "csv" / "table_E7_pan_europe_locked_results_by_tile.csv", "tables_csv_original/E7_strong", "table_csv", "appendix", "Full by-tile by-horizon results for the strong pan-European frozen audit."),
    AssetSpec("table_E7_api_tile_catalog.csv", "E7_strong", STRONG_ROOT / "manifests" / "table_E7_api_tile_catalog.csv", "tables_csv_original/E7_strong", "table_csv", "appendix", "Official EGMS API tile catalog harvested for strong E7 selection."),
    AssetSpec("table_E7_pan_europe_tile_metadata.csv", "E7_strong", STRONG_ROOT / "manifests" / "table_E7_pan_europe_tile_metadata.csv", "tables_csv_original/E7_strong", "table_csv", "appendix", "Metadata and regime/coverage audit for the 14 held-out strong E7 tiles."),
    AssetSpec("table_E7_selected_tiles_prefetch.csv", "E7_strong", STRONG_ROOT / "manifests" / "table_E7_selected_tiles_prefetch.csv", "tables_csv_original/E7_strong", "table_csv", "appendix", "Prefetch list and selection notes for the 14 strong E7 held-out tiles."),
    AssetSpec("threshold_diag_main6_validation.csv", "E7_strong", STRONG_ROOT / "manifests" / "threshold_diag_main6_validation.csv", "tables_csv_original/E7_strong", "table_csv", "appendix", "Main-six validation diagnostics used to calibrate SS-VCEA thresholds for strong E7."),
]

E7_STRONG_FIGURES = [
    AssetSpec("fig_E7_failure_case_overview.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_failure_case_overview.png", "figures/E7_strong", "figure_png", "appendix", "Heatmap overview of strong E7 gains and stress ratios across held-out tiles and horizons."),
    AssetSpec("fig_E7_pan_europe_tile_map.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_pan_europe_tile_map.png", "figures/E7_strong", "figure_png", "main_text", "Map of development, original external, and 14 strong E7 held-out tiles."),
    AssetSpec("fig_E7_skill_by_regime.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_skill_by_regime.png", "figures/E7_strong", "figure_png", "appendix", "Strong E7 gains and stress metrics grouped by audit-regime class."),
    AssetSpec("fig_E7_skill_by_region.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_skill_by_region.png", "figures/E7_strong", "figure_png", "main_text", "Strong E7 VCEA gain and SS-VCEA fallback rate by broad European region."),
    AssetSpec("fig_E7_skill_vs_coverage.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_skill_vs_coverage.png", "figures/E7_strong", "figure_png", "main_text", "Strong E7 gain versus valid coverage over the 14 held-out tiles."),
    AssetSpec("fig_E7_stress_by_region.png", "E7_strong", STRONG_ROOT / "figures" / "fig_E7_stress_by_region.png", "figures/E7_strong", "figure_png", "appendix", "Strong E7 SS-VCEA stress ratio and fallback rate by region."),
]

ALL_SPECS = E1_E6_TABLES + E1_E6_FIGURES + E7_STRONG_TABLES + E7_STRONG_FIGURES


MAIN_TEXT_TABLE_KEYS = [
    "table_E1_leakage_audit.csv",
    "table_E2_expert_ablation_main6.csv",
    "table_E3_calibration_ablation.csv",
    "table_E6_ssvcea_threshold_sweep.csv",
    "table_E7_pan_europe_group_summary.csv",
    "table_E7_pan_europe_failure_cases.csv",
]
MAIN_TEXT_FIG_KEYS = [
    "fig_E1_skill_delta_by_tile.png",
    "fig_E2_expert_weights_heatmap.png",
    "fig_E3_calibration_barplot.png",
    "fig_E6_skill_stress_frontier.png",
    "fig_E7_pan_europe_tile_map.png",
    "fig_E7_skill_by_region.png",
    "fig_E7_skill_vs_coverage.png",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_with_versioning(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    target = dst_dir / src.name
    if target.exists():
        if sha256_file(target) == sha256_file(src):
            return target
        stem, suffix = src.stem, src.suffix
        index = 2
        while True:
            candidate = dst_dir / f"{stem}_v{index}{suffix}"
            if not candidate.exists():
                target = candidate
                break
            if sha256_file(candidate) == sha256_file(src):
                return candidate
            index += 1
    shutil.copy2(src, target)
    return target


LATEX_SPECIALS = {
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
}


def latex_escape(text: Any) -> str:
    if text is None:
        return "--"
    if isinstance(text, float) and math.isnan(text):
        return "--"
    s = str(text)
    if not s.strip() or s.lower() == "nan":
        return "--"
    out = []
    for ch in s:
        out.append(LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def format_float(value: Any, decimals: int | None) -> str:
    if value is None:
        return "--"
    try:
        if pd.isna(value):
            return "--"
    except Exception:
        pass
    if decimals is None:
        return latex_escape(value)
    if isinstance(value, str):
        try:
            numeric = float(value)
        except Exception:
            return latex_escape(value)
    else:
        numeric = float(value)
    return f"{numeric:.{decimals}f}"


def infer_column_decimals(df: pd.DataFrame, compact: bool) -> dict[str, int | None]:
    mapping: dict[str, int | None] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mapping[col] = 3 if compact else None
        else:
            mapping[col] = None
    return mapping


def compact_column_name(name: str) -> str:
    compact_map = {
        "Mean Skill_MSE": r"Mean Skill$_{\mathrm{MSE}}$",
        "Mean Skill_MSE std": r"Mean Skill$_{\mathrm{MSE}}$ std",
        "Main-six Skill": r"Main-six Skill$_{\mathrm{MSE}}$",
        "External Skill": r"External Skill$_{\mathrm{MSE}}$",
        "VCEA Skill": r"VCEA Skill$_{\mathrm{MSE}}$",
        "M1 Skill": r"M1 Skill$_{\mathrm{MSE}}$",
        "Skill MSE": r"Skill$_{\mathrm{MSE}}$",
        "Stress ratio": "Stress ratio",
        "Fallback rate": "Fallback rate",
        "Gain/Loss vs VCEA": "Gain/loss vs VCEA",
        "Mean Gain/Loss vs VCEA": "Mean gain/loss vs VCEA",
        "Mean Gain vs M1": "Gain vs M1",
        "Mean RMSE [mm]": "RMSE [mm]",
        "Mean MAE [mm]": "MAE [mm]",
        "Better tiles vs M1": "Better tiles vs M1",
        "Better tiles vs VCEA": "Better tiles vs VCEA",
        "Num tile-horizon rows": "Tile-horizon rows",
    }
    return compact_map.get(name, name.replace("_", r"\_").replace("%", r"\%"))


def latex_tabular(df: pd.DataFrame, compact: bool, col_subset: list[str] | None = None) -> str:
    data = df.copy()
    if col_subset is not None:
        data = data[col_subset].copy()
    dec_map = infer_column_decimals(data, compact=compact)
    headers = [compact_column_name(col) for col in data.columns]
    colspec = "l" * len(headers)
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for row in data.itertuples(index=False, name=None):
        cells = []
        for col, value in zip(data.columns, row):
            if dec_map[col] is None:
                cells.append(latex_escape(value))
            else:
                cells.append(format_float(value, dec_map[col]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def split_dataframe_columns(df: pd.DataFrame, key_cols: list[str], max_extra_cols: int) -> list[pd.DataFrame]:
    extra_cols = [c for c in df.columns if c not in key_cols]
    chunks: list[pd.DataFrame] = []
    for idx in range(0, len(extra_cols), max_extra_cols):
        cols = key_cols + extra_cols[idx : idx + max_extra_cols]
        chunks.append(df[cols].copy())
    return chunks


def latex_longtable(
    df: pd.DataFrame,
    compact: bool,
    landscape: bool,
    caption: str,
    label: str,
    key_cols: list[str] | None = None,
    max_extra_cols: int = 9,
) -> str:
    dataframes = [df]
    if key_cols and len(df.columns) > len(key_cols) + max_extra_cols:
        dataframes = split_dataframe_columns(df, key_cols, max_extra_cols)

    parts: list[str] = []
    for part_idx, part in enumerate(dataframes, start=1):
        panel_label = label if part_idx == 1 else f"{label}_part{part_idx}"
        panel_caption = caption if len(dataframes) == 1 else f"{caption} (column panel {part_idx}/{len(dataframes)})."
        if landscape:
            parts.append(r"\begin{landscape}")
        parts.append(r"\scriptsize")
        parts.append(r"\setlength{\LTleft}{0pt}")
        parts.append(r"\setlength{\LTright}{0pt}")
        headers = [compact_column_name(c) for c in part.columns]
        colspec = "l" * len(headers)
        parts.append(rf"\begin{{longtable}}{{{colspec}}}")
        parts.append(rf"\caption{{{panel_caption}}}\label{{{panel_label}}}\\")
        parts.append(r"\toprule")
        parts.append(" & ".join(headers) + r" \\")
        parts.append(r"\midrule")
        parts.append(r"\endfirsthead")
        parts.append(r"\toprule")
        parts.append(" & ".join(headers) + r" \\")
        parts.append(r"\midrule")
        parts.append(r"\endhead")
        parts.append(r"\midrule")
        parts.append(rf"\multicolumn{{{len(headers)}}}{{r}}{{\textit{{Continued on next page}}}} \\")
        parts.append(r"\endfoot")
        parts.append(r"\bottomrule")
        parts.append(r"\endlastfoot")
        for row in part.itertuples(index=False, name=None):
            row_cells = []
            for col, value in zip(part.columns, row):
                if pd.api.types.is_numeric_dtype(part[col]) and compact:
                    row_cells.append(format_float(value, 3))
                else:
                    row_cells.append(latex_escape(value))
            parts.append(" & ".join(row_cells) + r" \\")
        parts.append(r"\bottomrule")
        parts.append(r"\end{longtable}")
        if landscape:
            parts.append(r"\end{landscape}")
        if part_idx != len(dataframes):
            parts.append("")
    return "\n".join(parts)


def figure_block(rel_path: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=0.95\textwidth]{{{rel_path}}}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{figure*}",
        ]
    )


def table_block(body: str, caption: str, label: str, env: str = "table*", size_cmd: str = r"\footnotesize") -> str:
    return "\n".join(
        [
            rf"\begin{{{env}}}[t]",
            r"\centering",
            size_cmd,
            body,
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\end{{{env}}}",
        ]
    )


def caption_map_tables() -> dict[str, tuple[str, str]]:
    return {
        "table_E1_leakage_audit.csv": (
            "Leakage-audit comparison across the target-time assignment, full-window containment, and gap-buffer protocols. VCEA retains a positive gain over M1 under stricter split rules, although the number of admissible windows decreases and the margin is reduced. This supports the interpretation that the aggregation advantage is not an artifact of target-time assignment.",
            "tab:e1_leakage_audit",
        ),
        "table_E2_expert_ablation_main6.csv": (
            "Expert-pool ablation on the main-six setting. The full VCEA pool outperforms removal variants, while uniform averaging is not a reliable substitute. The ablation supports the interpretation that persistence, linear-seasonal structure, and neural experts provide complementary information under strong persistence baselines.",
            "tab:e2_expert_ablation_main6",
        ),
        "table_E2_expert_ablation_external.csv": (
            "Expert-pool ablation on the original external locked evaluation. The table reports how removing individual experts changes transfer skill and stress relative to the full VCEA pool.",
            "tab:e2_expert_ablation_external",
        ),
        "table_E3_calibration_ablation.csv": (
            "Calibration ablation for validation-based expert aggregation. Full tile+horizon validation calibration yields the strongest main-six result and the smallest oracle gap, whereas uniform averaging is insufficient under the strong persistence baseline.",
            "tab:e3_calibration_ablation",
        ),
        "table_E4_blocked_validation.csv": (
            "Blocked-validation audit by calibration block, tile, and seed. The table shows that short blocked windows remain informative but are less stable than the full validation period, which supports retaining the full validation period for the frozen calibration protocol.",
            "tab:e4_blocked_validation_full",
        ),
        "table_E5_external_regime_audit.csv": (
            "Original external audit grouped by coverage and dominant-regime notes. The gains are not explained solely by valid coverage, supporting the interpretation that external transfer is not purely a coverage artifact.",
            "tab:e5_external_audit",
        ),
        "table_E6_ssvcea_threshold_sweep.csv": (
            "Threshold sweep for SS-VCEA. The selected $q97.5$ operating point preserves skill close to VCEA while reducing stress below one with a modest fallback rate, making it the preferred conservative operating point.",
            "tab:e6_threshold_sweep",
        ),
        "table_E7_pan_europe_group_summary.csv": (
            "Compact summary of the geographically stratified pan-European frozen audit on 14 held-out EGMS 2019--2023 tiles, corresponding to 56 tile-horizon evaluations. The table aggregates performance over six broad European regions, four audit-regime groups, and three coverage groups. It supports positive average transfer beyond the development and original external sets, while remaining a frozen audit rather than an exhaustive population estimate over all European deformation conditions.",
            "tab:e7_strong_group_summary",
        ),
        "table_E7_pan_europe_failure_cases.csv": (
            "Failure cases in the strong pan-European frozen audit. The negative or high-fallback cases are localized, notably including the short-temporal-span E44N43 cases and the low-coverage E49N22 case, rather than indicating a systematic collapse of VCEA across held-out regions.",
            "tab:e7_strong_failure_cases",
        ),
        "table_E7_pan_europe_locked_results_by_tile.csv": (
            "Full by-tile by-horizon results for the strong pan-European frozen audit on 14 held-out EGMS 2019--2023 tiles (56 tile-horizon evaluations). For IEEEtran two-column final version, this full table is intended for appendix/supplement; use the compact summary table in the main text.",
            "tab:e7_strong_by_tile_full",
        ),
        "table_E7_api_tile_catalog.csv": (
            "Official EGMS API tile catalog used to define the candidate pool for the strong pan-European frozen audit. This appendix table documents the searchable held-out universe from which the 14 geographically stratified tiles were selected after the model and threshold settings were frozen.",
            "tab:e7_strong_api_catalog",
        ),
        "table_E7_pan_europe_tile_metadata.csv": (
            "Metadata for the 14 held-out EGMS 2019--2023 tiles in the strong pan-European frozen audit, including broad-region assignment, audit-regime grouping, valid coverage, and raw support. This table documents the stratified audit design rather than an exhaustive population census.",
            "tab:e7_strong_tile_metadata",
        ),
        "table_E7_selected_tiles_prefetch.csv": (
            "Prefetch list and selection notes for the 14 held-out tiles used in the strong pan-European frozen audit. The selection is geographically stratified and fixed after model, calibration, and threshold choices were frozen.",
            "tab:e7_strong_selected_tiles",
        ),
        "threshold_diag_main6_validation.csv": (
            "Main-six validation diagnostics used to calibrate the SS-VCEA threshold vector later transferred to the strong pan-European frozen audit. The table provides traceability for the equal-normalized risk calibration without reusing held-out labels.",
            "tab:e7_strong_threshold_diag",
        ),
    }


def caption_map_figures() -> dict[str, tuple[str, str]]:
    return {
        "fig_E1_protocol_timeline.png": (
            "Timeline illustration of the three E1 split protocols. The figure contrasts the original target-time assignment, the stricter full-window containment rule, and the gap-buffer protocol used to reduce near-boundary leakage risk.",
            "fig:e1_protocol_timeline",
        ),
        "fig_E1_skill_delta_by_tile.png": (
            "Tile-wise VCEA gain over M1 under the E1 leakage-audit protocols. The gains remain positive under stricter protocols, supporting the interpretation that VCEA does not rely on target-time assignment artifacts.",
            "fig:e1_skill_delta_by_tile",
        ),
        "fig_E2_expert_weights_heatmap.png": (
            "Validation-calibrated expert weights by tile and horizon. The heatmap shows that VCEA does not reduce to a uniform average and instead shifts trust across persistence, linear-seasonal, and neural experts according to tile-horizon conditions.",
            "fig:e2_expert_weights",
        ),
        "fig_E2_remove_expert_delta.png": (
            "Performance change after removing individual experts from the VCEA pool. The figure indicates that the linear-seasonal and neural experts contribute complementary gains, while removing persistence can sharply worsen stress.",
            "fig:e2_remove_expert_delta",
        ),
        "fig_E3_calibration_barplot.png": (
            "Calibration-ablation summary across aggregation strategies. Full tile+horizon validation calibration achieves the strongest main-six skill and the smallest oracle gap, whereas uniform averaging is insufficient under the strong persistence baseline.",
            "fig:e3_calibration_barplot",
        ),
        "fig_E3_val_vs_test_weights.png": (
            "Comparison between validation-derived weights and oracle test weights. The agreement suggests that validation ranking stability is strong enough to support the static validation-calibrated aggregation scheme.",
            "fig:e3_val_vs_test_weights",
        ),
        "fig_E4_test_skill_by_block.png": (
            "Blocked-validation test skill by calibration block. Positive transfer remains visible across blocks, but short blocks are less stable than the full validation period, which supports using the full validation span in the frozen protocol.",
            "fig:e4_test_skill_by_block",
        ),
        "fig_E4_weight_stability.png": (
            "Weight stability under blocked validation. The figure shows how far blocked-validation weights drift from the full-validation weights used in the frozen protocol.",
            "fig:e4_weight_stability",
        ),
        "fig_E5_external_map_regime.png": (
            "Map of the original external locked tiles with dominant-regime annotations. The finite external set is geographically useful but not sufficient for a Europe-wide population claim on its own.",
            "fig:e5_external_map_regime",
        ),
        "fig_E5_skill_vs_coverage.png": (
            "External gain versus valid coverage in the original external audit. The positive transfer is not purely a valid-coverage artifact, because gains remain visible across different support levels.",
            "fig:e5_skill_vs_coverage",
        ),
        "fig_E6_fallback_rate_by_tile.png": (
            "Fallback rate by tile under the SS-VCEA threshold sweep. The figure helps interpret where conservative fallback is triggered and how risk control varies across tiles.",
            "fig:e6_fallback_rate_by_tile",
        ),
        "fig_E6_skill_stress_frontier.png": (
            "Skill-stress frontier for the SS-VCEA threshold sweep. The selected $q97.5$ operating point keeps skill close to VCEA while reducing stress below one, supporting its use as the main conservative deployment setting.",
            "fig:e6_skill_stress_frontier",
        ),
        "fig_E7_failure_case_overview.png": (
            "Overview of strong E7 gains and stress across the 14 held-out EGMS 2019--2023 tiles and 56 tile-horizon evaluations. The remaining failures are localized rather than systematic, which supports a restrained claim of geographically stratified transfer.",
            "fig:e7_failure_case_overview",
        ),
        "fig_E7_pan_europe_tile_map.png": (
            "Geographically stratified pan-European frozen audit design. The map shows the development tiles, the original external tiles, and the 14 held-out EGMS 2019--2023 tiles used in the strong audit. This is a frozen audit rather than an exhaustive population estimate over all European deformation conditions.",
            "fig:e7_pan_europe_tile_map",
        ),
        "fig_E7_skill_by_regime.png": (
            "Strong E7 performance grouped by the four audit-regime groups. The figure suggests that VCEA transfer extends beyond the original evaluation regions while still exhibiting regime-dependent limits that should be discussed as remaining failure modes.",
            "fig:e7_skill_by_regime",
        ),
        "fig_E7_skill_by_region.png": (
            "Strong E7 skill by broad European region. Over the 14 held-out tiles and 56 tile-horizon evaluations, VCEA supports positive average transfer beyond the development and original external sets, while SS-VCEA preserves most of the gain with more conservative behavior.",
            "fig:e7_skill_by_region",
        ),
        "fig_E7_skill_vs_coverage.png": (
            "Strong E7 VCEA gain versus valid coverage across the 14 held-out tiles. The trend suggests that the transfer gain is not reducible to a simple valid-coverage effect, although lower-support cases still define important limitations.",
            "fig:e7_skill_vs_coverage",
        ),
        "fig_E7_stress_by_region.png": (
            "Strong E7 stress ratio and fallback rate by region for SS-VCEA. The figure supports the interpretation that the selected conservative wrapper reduces over-correction risk below one on average while retaining most of the VCEA gain.",
            "fig:e7_stress_by_region",
        ),
    }


def compute_target_path(spec: AssetSpec) -> Path:
    return OUT_ROOT / spec.target_subdir / spec.key


def discover_assets() -> tuple[list[AssetSpec], list[str]]:
    missing: list[str] = []
    found: list[AssetSpec] = []
    for spec in ALL_SPECS:
        if spec.source_path.exists():
            found.append(spec)
        else:
            matches = list(EXTRAS_ROOT.rglob(spec.key)) + list(STRONG_ROOT.rglob(spec.key))
            if matches:
                found.append(AssetSpec(spec.key, spec.experiment_id, matches[0], spec.target_subdir, spec.asset_type, spec.intended_location, spec.short_description))
            else:
                missing.append(spec.key)
    return found, missing


def make_fragment_name(prefix: str, key: str) -> str:
    stem = key.rsplit(".", 1)[0]
    return f"{prefix}_{stem}.tex"


def compact_e7_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "Group type",
        "Group name",
        "Model",
        "Num tiles",
        "Num tile-horizon rows",
        "Mean Skill_MSE",
        "Mean RMSE [mm]",
        "Mean MAE [mm]",
        "Mean Gain vs M1",
        "Better tiles vs M1",
        "Stress ratio",
        "Mean fallback rate",
    ]
    filtered = df[df["Group type"].isin(["overall", "broad_region", "audit_regime_group", "coverage_group"])].copy()
    return filtered[keep_cols]


def generate_table_content(key: str, copied_csv: Path, fragment_dir: Path, manifest_records: list[dict[str, Any]]) -> tuple[str, list[str]]:
    captions = caption_map_tables()
    caption, label = captions[key]
    df = pd.read_csv(copied_csv)
    labels = [label]
    pieces: list[str] = []
    rel_csv = copied_csv.relative_to(OUT_ROOT).as_posix()

    if key == "table_E7_pan_europe_group_summary.csv":
        compact_df = compact_e7_group_summary(df)
        compact_body = "\n".join(
            [
                f"% Source CSV: {rel_csv}",
                r"\begin{table*}[t]",
                r"\centering",
                r"\footnotesize",
                latex_tabular(compact_df, compact=True),
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{table*}",
            ]
        )
        compact_fragment = fragment_dir / make_fragment_name("main", key)
        compact_fragment.write_text(compact_body + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, compact_fragment, "tex_fragment", "E7_strong", "main_text", "Compact main-text summary derived from the full E7 group summary CSV."))
        pieces.append(compact_body)

        full_label = "tab:e7_strong_group_summary_full"
        labels.append(full_label)
        full_caption = "Full appendix version of the strong E7 group summary corresponding to Table~\\ref{tab:e7_strong_group_summary}. The complete table retains all grouping rows and columns from the original CSV for appendix/supplement use."
        full_body = "\n".join(
            [
                f"% Source CSV: {rel_csv}",
                latex_longtable(df, compact=False, landscape=True, caption=full_caption, label=full_label, key_cols=["Group type", "Group name", "Model"], max_extra_cols=8),
            ]
        )
        full_fragment = fragment_dir / "appendix_table_E7_pan_europe_group_summary_full.tex"
        full_fragment.write_text(full_body + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, full_fragment, "tex_fragment", "E7_strong", "appendix", "Full appendix version of the E7 group summary."))
        pieces.append(full_body)
        return "\n\n".join(pieces), labels

    if key == "table_E7_pan_europe_locked_results_by_tile.csv":
        body = latex_longtable(df, compact=False, landscape=True, caption=caption, label=label, key_cols=["Model", "Tile", "Horizon"], max_extra_cols=9)
        content = "\n".join(
            [
                f"% Source CSV: {rel_csv}",
                "% For IEEEtran two-column final version, this full table is intended for appendix/supplement; use the compact summary table in the main text.",
                body,
            ]
        )
        fragment = fragment_dir / make_fragment_name("appendix", key)
        fragment.write_text(content + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, fragment, "tex_fragment", "E7_strong", "appendix", "Split longtable appendix rendering of the full E7 by-tile results."))
        return content, labels

    if key == "table_E4_blocked_validation.csv":
        body = latex_longtable(df, compact=False, landscape=False, caption=caption, label=label, key_cols=["Calibration block", "Time range", "Validation windows", "Tile", "Seed"], max_extra_cols=6)
        content = "\n".join(
            [
                f"% Source CSV: {rel_csv}",
                "% For IEEEtran two-column final version, this full table is intended for appendix/supplement; use the compact summary discussion in the main text.",
                body,
            ]
        )
        fragment = fragment_dir / make_fragment_name("appendix", key)
        fragment.write_text(content + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, fragment, "tex_fragment", "E4", "appendix", "Appendix rendering of the blocked-validation full table."))
        return content, labels

    if key in {"table_E7_api_tile_catalog.csv", "table_E7_pan_europe_tile_metadata.csv", "table_E7_selected_tiles_prefetch.csv", "threshold_diag_main6_validation.csv"}:
        key_cols_map = {
            "table_E7_api_tile_catalog.csv": ["tile_id"],
            "table_E7_pan_europe_tile_metadata.csv": ["tile_id"],
            "table_E7_selected_tiles_prefetch.csv": ["tile_id"],
            "threshold_diag_main6_validation.csv": ["Safe target", "Percentile", "Tile", "Seed", "Horizon"],
        }
        body = latex_longtable(df, compact=False, landscape=True, caption=caption, label=label, key_cols=key_cols_map[key], max_extra_cols=8)
        content = "\n".join(
            [
                f"% Source CSV: {rel_csv}",
                "% Appendix metadata / manifest table.",
                body,
            ]
        )
        fragment = fragment_dir / make_fragment_name("appendix", key)
        fragment.write_text(content + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, fragment, "tex_fragment", "E7_strong", "appendix", "Appendix rendering of a strong-E7 metadata or manifest table."))
        return content, labels

    if key == "table_E7_pan_europe_failure_cases.csv":
        body = latex_tabular(df, compact=True)
        content = "\n".join([f"% Source CSV: {rel_csv}", table_block(body, caption, label, env="table*", size_cmd=r"\footnotesize")])
        fragment = fragment_dir / make_fragment_name("main", key)
        fragment.write_text(content + "\n", encoding="utf-8")
        manifest_records.append(build_manifest_record(copied_csv, fragment, "tex_fragment", "E7_strong", "main_text", "Main-text failure-case table for strong E7."))
        return content, labels

    compact = key in MAIN_TEXT_TABLE_KEYS
    if df.shape[0] > 25 or df.shape[1] > 10:
        body = latex_longtable(df, compact=compact, landscape=df.shape[1] > 10, caption=caption, label=label, key_cols=list(df.columns[: min(3, len(df.columns))]), max_extra_cols=8)
        content = "\n".join([f"% Source CSV: {rel_csv}", body])
    else:
        env = "table*"
        size_cmd = r"\footnotesize" if df.shape[1] <= 8 else r"\scriptsize"
        content = "\n".join([f"% Source CSV: {rel_csv}", table_block(latex_tabular(df, compact=compact), caption, label, env=env, size_cmd=size_cmd)])
    fragment_type = "main" if key in MAIN_TEXT_TABLE_KEYS else "appendix"
    fragment = fragment_dir / make_fragment_name(fragment_type, key)
    fragment.write_text(content + "\n", encoding="utf-8")
    manifest_records.append(build_manifest_record(copied_csv, fragment, "tex_fragment", "E7_strong" if key.startswith("table_E7") or key == "threshold_diag_main6_validation.csv" else key.split("_")[1], "main_text" if key in MAIN_TEXT_TABLE_KEYS else "appendix", f"LaTeX fragment generated from {key}."))
    return content, labels


def generate_figure_content(key: str, copied_png: Path, manifest_records: list[dict[str, Any]]) -> tuple[str, str]:
    captions = caption_map_figures()
    caption, label = captions[key]
    rel_path = copied_png.relative_to(OUT_ROOT).as_posix()
    content = figure_block(rel_path, caption, label)
    return content, label


def build_manifest_record(source_path: Path, target_path: Path, asset_type: str, experiment_id: str, intended_location: str, short_description: str) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "sha256": sha256_file(target_path),
        "file_size": target_path.stat().st_size,
        "asset_type": asset_type,
        "experiment_id": experiment_id,
        "intended_location": intended_location,
        "short_description": short_description,
    }


def build_main_tex(table_sections: list[str], figure_sections: list[str]) -> str:
    lines = [
        "% Suggested packages for the parent manuscript preamble:",
        "% \\usepackage{graphicx}",
        "% \\usepackage{booktabs}",
        "% \\usepackage{multirow}",
        "% \\usepackage{array}",
        "% \\usepackage{tabularx}",
        "% \\usepackage{adjustbox}",
        "% \\usepackage{threeparttable}",
        "% \\usepackage{pdflscape}",
        "% \\usepackage{longtable}",
        "",
        "% ============================================================",
        "% Main-text candidate tables and figures",
        "% ============================================================",
        r"\section*{Main-text Candidate Tables and Figures}",
        "",
    ]
    lines.extend(table_sections[:6])
    lines.append("")
    lines.extend(figure_sections[:7])
    lines.extend(
        [
            "",
            "% ============================================================",
            "% Appendix / supplementary full tables and figures",
            "% ============================================================",
            r"\section*{Appendix / Supplementary Full Tables and Figures}",
            "",
        ]
    )
    lines.extend(table_sections[6:])
    lines.append("")
    lines.extend(figure_sections[7:])
    return "\n\n".join(lines) + "\n"


def recommended_sections(table_content_map: dict[str, str], figure_content_map: dict[str, str]) -> tuple[list[str], list[str]]:
    main_table_order = [
        "table_E1_leakage_audit.csv",
        "table_E2_expert_ablation_main6.csv",
        "table_E3_calibration_ablation.csv",
        "table_E6_ssvcea_threshold_sweep.csv",
        "table_E7_pan_europe_group_summary.csv",
        "table_E7_pan_europe_failure_cases.csv",
    ]
    appendix_table_order = [
        "table_E2_expert_ablation_external.csv",
        "table_E4_blocked_validation.csv",
        "table_E5_external_regime_audit.csv",
        "table_E7_pan_europe_locked_results_by_tile.csv",
        "table_E7_api_tile_catalog.csv",
        "table_E7_pan_europe_tile_metadata.csv",
        "table_E7_selected_tiles_prefetch.csv",
        "threshold_diag_main6_validation.csv",
    ]
    main_fig_order = [
        "fig_E1_skill_delta_by_tile.png",
        "fig_E2_expert_weights_heatmap.png",
        "fig_E3_calibration_barplot.png",
        "fig_E6_skill_stress_frontier.png",
        "fig_E7_pan_europe_tile_map.png",
        "fig_E7_skill_by_region.png",
        "fig_E7_skill_vs_coverage.png",
    ]
    appendix_fig_order = [
        "fig_E1_protocol_timeline.png",
        "fig_E2_remove_expert_delta.png",
        "fig_E3_val_vs_test_weights.png",
        "fig_E4_test_skill_by_block.png",
        "fig_E4_weight_stability.png",
        "fig_E5_external_map_regime.png",
        "fig_E5_skill_vs_coverage.png",
        "fig_E6_fallback_rate_by_tile.png",
        "fig_E7_skill_by_regime.png",
        "fig_E7_stress_by_region.png",
        "fig_E7_failure_case_overview.png",
    ]
    table_sections = [table_content_map[key] for key in main_table_order if key in table_content_map] + [table_content_map[key] for key in appendix_table_order if key in table_content_map]
    figure_sections = [figure_content_map[key] for key in main_fig_order if key in figure_content_map] + [figure_content_map[key] for key in appendix_fig_order if key in figure_content_map]
    return table_sections, figure_sections


def write_readme(missing_files: list[str], compile_warning: str) -> None:
    lines = [
        "# paper_assets_vcea_extras",
        "",
        "This folder contains LaTeX-ready assets for the VCEA extra experiments (E1--E6) and the Strong E7 pan-European frozen audit.",
        "",
        "## Contents",
        "- `vcea_extra_tables_and_figures.tex`: self-contained LaTeX file containing all table contents, captions, labels, and figure blocks.",
        "- `figures/`: copied PNG figures grouped by experiment (`E1` ... `E7_strong`).",
        "- `tables_csv_original/`: copied original CSV tables grouped into `E1_E6` and `E7_strong`.",
        "- `tables_tex_fragments/`: per-table LaTeX fragments generated from the CSV sources.",
        "- `manifest_assets.json`: asset manifest with source/target paths, hashes, sizes, and intended manuscript placement.",
        "- `compile_check/test_compile.tex`: minimal syntax-check document that inputs the master TeX file.",
        "",
        "## How to include from the main paper",
        "In the parent manuscript directory, load the required packages in the preamble and then either:",
        "1. `\\input{paper_assets_vcea_extras/vcea_extra_tables_and_figures.tex}` directly, or",
        "2. Copy selected table/figure environments from this file into the main manuscript.",
        "",
        "## Recommended placement",
        "- Main text: E1 leakage audit, E2 main-six expert ablation, E3 calibration ablation, E6 threshold sweep, Strong E7 compact group summary, Strong E7 failure cases, and the selected main-text figures.",
        "- Appendix / supplement: full E4 blocked-validation table, full Strong E7 by-tile table, metadata/catalog/prefetch tables, threshold diagnostics, and the remaining figures.",
        "",
        "## IEEEtran compile notes",
        "- `figure*` and `table*` environments are used for main-text candidate assets.",
        "- Very wide or long appendix tables use `longtable` and `pdflscape`.",
        "- For the IEEEtran two-column final version, the full Strong E7 by-tile table and the full E4 blocked-validation table are intended for appendix/supplement use.",
        f"- Compile check status: {compile_warning}",
    ]
    if missing_files:
        lines.extend(["", "## Missing files", *[f"- `{name}`" for name in missing_files]])
    else:
        lines.extend(["", "## Missing files", "- None."])
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compile_check(master_tex_rel: str) -> Path:
    compile_dir = ensure_dir(OUT_ROOT / "compile_check")
    tex_path = compile_dir / "test_compile.tex"
    content = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs}",
            r"\usepackage{multirow}",
            r"\usepackage{array}",
            r"\usepackage{tabularx}",
            r"\usepackage{adjustbox}",
            r"\usepackage{threeparttable}",
            r"\usepackage{pdflscape}",
            r"\usepackage{longtable}",
            r"\begin{document}",
            rf"\input{{{master_tex_rel}}}",
            r"\end{document}",
            "",
        ]
    )
    tex_path.write_text(content, encoding="utf-8")
    return tex_path


def try_compile(tex_path: Path) -> str:
    compile_dir = tex_path.parent
    log_path = compile_dir / "compile.log"
    engines = ["latexmk", "pdflatex", "xelatex", "tectonic"]
    chosen = None
    for engine in engines:
        probe = shutil.which(engine)
        if probe:
            chosen = engine
            break
    if not chosen:
        msg = "No LaTeX engine (`latexmk`, `pdflatex`, `xelatex`, or `tectonic`) is available in PATH; syntax check file was generated but not compiled."
        log_path.write_text(msg + "\n", encoding="utf-8")
        return msg
    if chosen == "latexmk":
        cmd = [chosen, "-pdf", "-interaction=nonstopmode", tex_path.name]
    elif chosen == "tectonic":
        cmd = [chosen, tex_path.name]
    else:
        cmd = [chosen, "-interaction=nonstopmode", tex_path.name]
    result = subprocess.run(cmd, cwd=str(compile_dir), capture_output=True, text=True, check=False)
    log_text = f"COMMAND: {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n"
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        return f"Compile attempted with `{chosen}` but returned a non-zero exit status; see `compile_check/compile.log`."
    return f"Compile succeeded with `{chosen}`."


def quality_checks(master_tex: str, required_table_keys: list[str], required_fig_keys: list[str]) -> tuple[list[str], list[str]]:
    missing_tables = [key for key in required_table_keys if key not in master_tex]
    missing_figs = [key for key in required_fig_keys if key not in master_tex]
    return missing_tables, missing_figs


def build_assets() -> dict[str, Any]:
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_ROOT / "figures")
    ensure_dir(OUT_ROOT / "tables_csv_original" / "E1_E6")
    ensure_dir(OUT_ROOT / "tables_csv_original" / "E7_strong")
    ensure_dir(OUT_ROOT / "tables_tex_fragments" / "E1_E6")
    ensure_dir(OUT_ROOT / "tables_tex_fragments" / "E7_strong")

    specs, missing_files = discover_assets()
    manifest_records: list[dict[str, Any]] = []
    table_content_map: dict[str, str] = {}
    figure_content_map: dict[str, str] = {}
    labels: list[str] = []
    tables_processed = 0
    figures_processed = 0

    for spec in specs:
        target_dir = OUT_ROOT / spec.target_subdir
        copied_path = copy_with_versioning(spec.source_path, target_dir)
        manifest_records.append(build_manifest_record(spec.source_path, copied_path, spec.asset_type, spec.experiment_id, spec.intended_location, spec.short_description))

        if spec.asset_type == "table_csv":
            fragment_dir = OUT_ROOT / "tables_tex_fragments" / ("E1_E6" if spec.experiment_id != "E7_strong" else "E7_strong")
            content, new_labels = generate_table_content(spec.key, copied_path, fragment_dir, manifest_records)
            table_content_map[spec.key] = content
            labels.extend(new_labels)
            tables_processed += 1
        else:
            content, label = generate_figure_content(spec.key, copied_path, manifest_records)
            figure_content_map[spec.key] = content
            labels.append(label)
            figures_processed += 1

    table_sections, figure_sections = recommended_sections(table_content_map, figure_content_map)
    master_tex = build_main_tex(table_sections, figure_sections)
    master_tex_path = OUT_ROOT / "vcea_extra_tables_and_figures.tex"
    master_tex_path.write_text(master_tex, encoding="utf-8")

    unique_labels = set(labels)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    missing_tables_in_tex, missing_figs_in_tex = quality_checks(
        master_tex,
        [spec.key for spec in E1_E6_TABLES + E7_STRONG_TABLES],
        [spec.key for spec in E1_E6_FIGURES + E7_STRONG_FIGURES],
    )

    compile_tex = write_compile_check("../vcea_extra_tables_and_figures.tex")
    compile_warning = try_compile(compile_tex)
    write_readme(missing_files + missing_tables_in_tex + missing_figs_in_tex, compile_warning)

    manifest_payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "output_root": str(OUT_ROOT),
        "tables_processed": tables_processed,
        "figures_processed": figures_processed,
        "missing_files": missing_files,
        "missing_tables_in_tex": missing_tables_in_tex,
        "missing_figures_in_tex": missing_figs_in_tex,
        "duplicate_labels": duplicate_labels,
        "compile_warning": compile_warning,
        "assets": manifest_records,
    }
    (OUT_ROOT / "manifest_assets.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_root": OUT_ROOT,
        "tables_processed": tables_processed,
        "figures_processed": figures_processed,
        "tex_path": master_tex_path,
        "readme_path": OUT_ROOT / "README.md",
        "missing_files": sorted(set(missing_files)),
        "missing_tables_in_tex": missing_tables_in_tex,
        "missing_figures_in_tex": missing_figs_in_tex,
        "duplicate_labels": duplicate_labels,
        "compile_warning": compile_warning,
    }


def main() -> None:
    summary = build_assets()
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
