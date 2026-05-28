from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "tgrs_2019_2023"
MANUSCRIPT = OUT / "manuscript"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    try:
        if src.resolve() == dst.resolve():
            return
    except FileNotFoundError:
        pass
    shutil.copy2(src, dst)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


@dataclass
class AssetRow:
    asset_id: str
    asset_type: str
    title: str
    status: str
    manuscript_target: str
    primary_file: str
    source_files: str
    note: str


def build_table01() -> Path:
    main6 = pd.read_csv(OUT / "Table1_tile_metadata.csv")
    external = pd.read_csv(OUT / "confirmation" / "external_tiles" / "external_tile_manifest.csv")
    main_df = pd.DataFrame(
        {
            "Tile": main6["tile_id"],
            "Group": "main6",
            "Role": main6["role"],
            "Dominant regime": main6["assigned_regime"],
            "Frames": main6["num_epochs"],
            "Valid64": main6["valid_grid_ratio_64"],
            "Valid128": main6["valid_grid_ratio_128"],
            "Num points": main6["num_raw_points"],
        }
    )
    ext_df = pd.DataFrame(
        {
            "Tile": external["Tile"],
            "Group": "external_locked",
            "Role": external["Role"],
            "Dominant regime": external["Dominant regime"],
            "Frames": external["Epochs"],
            "Valid64": external["Valid64"],
            "Valid128": external["Valid128"],
            "Num points": external["Num points"],
        }
    )
    out = MANUSCRIPT / "tables" / "ready" / "Table01_tile_metadata_main_external.csv"
    write_csv(pd.concat([main_df, ext_df], ignore_index=True), out)
    return out


def build_table02() -> Path:
    rows = [
        ("Product", "EGMS 2019-2023 Ortho U", "All formal results use the unified 2019-2023 Ortho U version."),
        ("Main tiles", "6", "E32N34, E32N35, E39N30, E44N23, E48N24, E58N17."),
        ("External locked tiles", "4", "E37N42, E45N51, E30N35, E42N21."),
        ("Main split", "70/10/20 chronological", "Leakage-safe train/val/test split."),
        ("Input length", "16 frames", "Unified for E1-E8 and all later mainline experiments."),
        ("Horizons", "t+1, t+3, t+5, t+10", "Multi-horizon nowcasting."),
        ("Main resolution", "64×64", "Primary regional nowcasting setting."),
        ("Sensitivity resolutions", "128×128, 512×512", "Sensitivity/robustness only, not the main task."),
        ("Descriptors", "velocity, acceleration, seasonal amplitude, seasonal phase, residual std, trend R², DOY sin/cos", "Fitted from train-window history only."),
        ("Masking", "valid-mask weighted metrics", "Nearest-neighbor fill is for dense tensor construction only."),
        ("Cross-site normalization", "source-tile stats", "Target descriptors still computed from target history."),
    ]
    df = pd.DataFrame(rows, columns=["Protocol item", "Setting", "Note"])
    out = MANUSCRIPT / "tables" / "ready" / "Table02_data_protocol.csv"
    write_csv(df, out)
    return out


def build_table03() -> Path:
    rows = [
        ("Accuracy", "RMSE", "Primary magnitude error"),
        ("Accuracy", "MAE", "Robust average absolute error"),
        ("Accuracy", "Skill_MSE", "1 - MSE_model / MSE_persistence"),
        ("Accuracy", "Skill_RMSE", "Relative RMSE improvement over persistence"),
        ("Accuracy", "Acc@1mm / Acc@0.5mm", "Thresholded accuracy"),
        ("Accuracy", "SSIM", "Structural similarity"),
        ("Safety", "stress ratio", "Worst-case risk / overcorrection safety measure"),
        ("Safety", "fallback rate", "Fraction of safe fallback in SS-VCEA"),
        ("Safety", "correction magnitude", "Magnitude of residual correction / risk"),
        ("Robustness", "better tiles", "How many tiles outperform the baseline"),
        ("Robustness", "seed std", "Stability across random seeds"),
        ("Robustness", "external locked gain", "Generalization beyond main-six"),
        ("Diagnostics", "velocity-sign consistency", "Sign consistency on moving pixels"),
        ("Diagnostics", "jump rate", "Unrealistic temporal jump frequency"),
        ("Diagnostics", "post-event recovery ratio", "Post-event stabilization diagnostic"),
        ("Diagnostics", "regime-wise metrics", "Stable/seasonal/subsidence/accelerating/coseismic/mixed diagnostics"),
        ("Efficiency/Repro", "train time / inference time / peak memory", "Engineering cost"),
        ("Efficiency/Repro", "lineage / exact-proxy-transfer flags", "Reproducibility provenance"),
    ]
    df = pd.DataFrame(rows, columns=["Metric group", "Metric", "Purpose"])
    out = MANUSCRIPT / "tables" / "ready" / "Table03_metrics_definition.csv"
    write_csv(df, out)
    return out


def build_table04() -> Path:
    rows = [
        ("Persistence", "strong baseline", "No training", "Short-horizon baseline and skill reference"),
        ("Linear-Seasonal", "kinematic baseline", "Per-tile fit", "Trend + seasonal explicit expert"),
        ("M1 / C2", "neural expert", "Full-six exact training", "Best robust single-model expert with gated residual correction"),
        ("C3_opt", "state-space-inspired neural expert", "Full-six exact training", "Higher-capacity expert retained for complementarity"),
    ]
    df = pd.DataFrame(rows, columns=["Expert", "Category", "Training scope", "Role in pool"])
    out = MANUSCRIPT / "tables" / "ready" / "Table04_expert_pool.csv"
    write_csv(df, out)
    return out


def build_table05() -> Path:
    rows = [
        ("VCEA", "Validation-only static calibration", "Tile×horizon static weights", "None", "No", "Best-accuracy mainline"),
        ("COEA", "Validation warm start + online update", "Delayed-feedback online update", "None", "No", "Online aggregation baseline"),
        ("SS-VCEA", "Reuse VCEA + validation-calibrated safety threshold", "Static VCEA output", "Risk-triggered hard fallback", "No", "Conservative deployment variant"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["Method", "Weight source", "Update mode", "Safety mechanism", "Test tuning", "Role"],
    )
    out = MANUSCRIPT / "tables" / "ready" / "Table05_aggregation_protocol_comparison.csv"
    write_csv(df, out)
    return out


def build_table06() -> Path:
    final_cmp = pd.read_csv(OUT / "staticval_ea_study" / "final_aggregation_comparison.csv")
    lin = pd.read_csv(OUT / "act_v7" / "finalization" / "v7_ablation_supplementary.csv")
    lin_row = lin[lin["Model"] == "Linear-Seasonal"].iloc[0]
    ss = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_main6_summary.csv").iloc[0]
    c3_verdict = read_json(OUT / "act_v4" / "v4c_efficiency_rescue" / "c3_candidate" / "c3_full6_verdict.json")
    c3_summary = pd.read_csv(OUT / "act_v4" / "v4c_efficiency_rescue" / "c3_candidate" / "c3_full6_summary.csv")
    c3_std = float(c3_summary["Skill_MSE std"].mean())
    rows = []
    for _, r in final_cmp.iterrows():
        rows.append(
            {
                "Model": r["Model"],
                "Category": "aggregation" if r["Model"] in {"VCEA", "COEA", "CRCF-COEA"} else "baseline",
                "Mean Skill_MSE": r["Mean Skill_MSE"],
                "Gain vs M1": r["Gain vs M1"],
                "Better tiles vs M1": r["Better tiles"],
                "Stress ratio": r["Stress ratio"],
                "Mean Skill std": r["Mean Skill std"],
                "Note": "main-six exact comparison",
            }
        )
    rows.append(
        {
            "Model": "SS-VCEA-COEA_q97p5",
            "Category": "safe aggregation",
            "Mean Skill_MSE": ss["Mean Skill_MSE"],
            "Gain vs M1": ss["Gain vs M1"],
            "Better tiles vs M1": ss["Better tiles vs M1"],
            "Stress ratio": ss["Stress ratio"],
            "Mean Skill std": ss["Seed std"],
            "Note": "selected conservative deployment variant",
        }
    )
    rows.append(
        {
            "Model": "C3_opt",
            "Category": "neural expert",
            "Mean Skill_MSE": c3_verdict["c3_overall_mean_skill_mse"],
            "Gain vs M1": c3_verdict["overall_skill_delta_vs_m1"],
            "Better tiles vs M1": c3_verdict["better_tiles_vs_m1"],
            "Stress ratio": c3_verdict["stress_max_rmse_ratio"],
            "Mean Skill std": c3_std,
            "Note": "full-six near-pass expert",
        }
    )
    rows.append(
        {
            "Model": "Linear-Seasonal",
            "Category": "kinematic baseline",
            "Mean Skill_MSE": lin_row["Mean Skill_MSE"],
            "Gain vs M1": lin_row["Gain vs M1"],
            "Better tiles vs M1": lin_row["Better tiles"],
            "Stress ratio": lin_row["Stress ratio"],
            "Mean Skill std": lin_row["Mean Skill std"],
            "Note": "strong explicit kinematic baseline",
        }
    )
    df = pd.DataFrame(rows)
    order = ["VCEA", "SS-VCEA-COEA_q97p5", "COEA", "CRCF-COEA", "C3_opt", "M1", "Persistence", "Linear-Seasonal"]
    df["sort_order"] = df["Model"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("sort_order").drop(columns="sort_order")
    out = MANUSCRIPT / "tables" / "ready" / "Table06_main_six_comparison.csv"
    write_csv(df, out)
    return out


def build_table07() -> Path:
    run = read_json(OUT / "hardening_package" / "run_summary.json")
    vtsa = run["vtsa"]
    rows = [
        ("Mean Spearman", vtsa["mean_spearman"], "Validation-to-test expert ranking stability"),
        ("Mean Kendall", vtsa["mean_kendall"], "Rank consistency"),
        ("Top1 retention", vtsa["top1_ratio"], "Validation top-1 remains top-1 on test"),
        ("Top2 retention", vtsa["top2_ratio"], "Validation top-1 remains within test top-2"),
        ("COEA mean L1 drift", vtsa["mean_drift"], "Online weight drift relative to static weights"),
        ("Blocked-to-test skill", vtsa["blocked_to_test_skill"], "Static blocked validation robustness"),
    ]
    df = pd.DataFrame(rows, columns=["VTSA metric", "Value", "Interpretation"])
    out = MANUSCRIPT / "tables" / "ready" / "Table07_vtsa_summary.csv"
    write_csv(df, out)
    return out


def build_table08() -> Path:
    ss_main = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_main6_summary.csv").iloc[0]
    ss_ext = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_external_summary.csv").iloc[0]
    run = read_json(OUT / "hardening_package" / "run_summary.json")
    rows = [
        ("VCEA", "main-six", run["ss_vcea"]["vcea_main_skill"], None, None, "best-accuracy reference in hardening package"),
        ("SS-VCEA-COEA_q97p5", "main-six", ss_main["Mean Skill_MSE"], ss_main["Stress ratio"], ss_main["Gain/Loss vs VCEA"], "selected main-six safety trade-off"),
        ("VCEA", "external_locked", 0.317246347403093, 0.9741283274440994, None, "frozen external exact evaluation"),
        ("SS-VCEA-COEA_q97p5", "external_locked", ss_ext["Mean Skill_MSE"], ss_ext["Stress ratio"], ss_ext["Gain/Loss vs VCEA"], ss_ext["Deployment note"]),
    ]
    df = pd.DataFrame(rows, columns=["Model", "Scope", "Mean Skill_MSE", "Stress ratio", "Gain/Loss vs VCEA", "Note"])
    out = MANUSCRIPT / "tables" / "ready" / "Table08_vcea_vs_ssvcea_safety_tradeoff.csv"
    write_csv(df, out)
    return out


def build_table09() -> Path:
    ext = pd.read_csv(OUT / "confirmation" / "external_tiles" / "external_summary.csv")
    stress = pd.read_csv(OUT / "confirmation" / "external_tiles" / "external_stress_report.csv")
    agg = ext.groupby("Model", as_index=False).agg(
        {
            "Skill_MSE mean": "mean",
            "Skill_MSE std": "mean",
            "RMSE [mm] mean": "mean",
            "Acc@1mm [%] mean": "mean",
            "SSIM mean": "mean",
        }
    )
    worst = stress.groupby("Model", as_index=False)["Worst RMSE/persistence"].max()
    merged = agg.merge(worst, on="Model", how="left")
    if "VCEA" in merged["Model"].values:
        vcea_skill = float(merged.loc[merged["Model"] == "VCEA", "Skill_MSE mean"].iloc[0])
    else:
        vcea_skill = 0.317246347403093
    merged["Gain/Loss vs VCEA"] = merged["Skill_MSE mean"] - vcea_skill
    ss_ext = pd.read_csv(OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_external_summary.csv")
    ss_row = pd.DataFrame(
        [
            {
                "Model": "SS-VCEA-COEA_q97p5",
                "Skill_MSE mean": float(ss_ext.iloc[0]["Mean Skill_MSE"]),
                "Skill_MSE std": float(ss_ext.iloc[0]["Seed std"]),
                "RMSE [mm] mean": None,
                "Acc@1mm [%] mean": None,
                "SSIM mean": None,
                "Worst RMSE/persistence": float(ss_ext.iloc[0]["Stress ratio"]),
                "Gain/Loss vs VCEA": float(ss_ext.iloc[0]["Gain/Loss vs VCEA"]),
            }
        ]
    )
    merged = pd.concat([merged, ss_row], ignore_index=True)
    out = MANUSCRIPT / "tables" / "ready" / "Table09_external_locked_validation.csv"
    write_csv(merged.sort_values("Skill_MSE mean", ascending=False), out)
    return out


def build_table10() -> Path:
    family = pd.read_csv(OUT / "confirmation" / "model_family_resolution_extension" / "family_resolution_ranking.csv")
    out = MANUSCRIPT / "tables" / "ready" / "Table10_resolution_sensitivity.csv"
    write_csv(family, out)
    return out


def build_table11() -> Path:
    rows = [
        ("V2 mainline", "M1 / C2 PG-RQ-ACT", 0.0783, "pass", "Best robust single-model line; kept as neural expert/baseline"),
        ("V3", "V3-C quick", 0.0135, "fail", "Too weak versus M1 despite acceptable complexity"),
        ("V4", "V4-C original", 0.0911, "fail", "Accuracy-credible but compute-no-go"),
        ("V4 rescue", "C3_opt", 0.0796, "fail", "Runtime fixed but full-six generalization insufficient"),
        ("V5", "risk-aware selector", 0.0964, "fail", "Oracle space exists but learned selector under-delivers on full-six"),
        ("V6", "V6B_anchor_risk_l20", -0.1041, "no_go", "Anchored soft selector unstable and stress-unsafe"),
        ("V7", "COEA audited", 0.1322, "pass", "First successful aggregation family beyond M1"),
        ("StaticVal study", "VCEA", 0.1776, "final_winner", "Best overall accuracy trade-off"),
        ("V8", "STR-Mamba / CoMEx", 0.0740, "no_go", "New architectures did not beat VCEA mainline"),
        ("DSCF", "DSCF-Full / VCEA+DSCF", 0.0082, "no_go", "Task-native consistency direction did not pass quick gate"),
    ]
    df = pd.DataFrame(rows, columns=["Stage", "Best candidate", "Best mean Skill_MSE", "Verdict", "Reason not final / role"])
    out = MANUSCRIPT / "tables" / "ready" / "Table11_architecture_search_audit.csv"
    write_csv(df, out)
    return out


def build_table12() -> Path:
    rows = [
        ("Product manifest", r"data/manifests/product_manifest.csv", "Data product provenance"),
        ("Tile manifest", r"data/manifests/tile_manifest.csv", "Main/external tile provenance"),
        ("Resolution manifest", r"data/manifests/resolution_manifest.csv", "64/128/512 scope freeze"),
        ("Main split", r"splits/main6_train_val_test.json", "Formal main-six split"),
        ("External locked split", r"splits/external4_locked_eval.json", "Frozen external evaluation"),
        ("Expert cache registry", r"cache/expert_cache_registry.yaml", "Expert prediction cache provenance"),
        ("Aggregation weight registry", r"cache/aggregation_weight_registry.yaml", "Aggregation calibration provenance"),
        ("Lineage table", r"outputs/tgrs_2019_2023/hardening_package/reproducibility/result_lineage_table.csv", "Experiment-to-artifact mapping"),
        ("Claim-to-result map", r"outputs/tgrs_2019_2023/hardening_package/reproducibility/claim_to_result_map.md", "Claim audit"),
        ("Paper table source map", r"outputs/tgrs_2019_2023/hardening_package/reproducibility/paper_table_source_map.csv", "Table-to-source provenance"),
        ("Figure source map", r"outputs/tgrs_2019_2023/hardening_package/reproducibility/figure_source_map.csv", "Figure-to-source provenance"),
        ("Exact/proxy/transfer flags", r"outputs/tgrs_2019_2023/hardening_package/reproducibility/exact_proxy_transfer_flags.csv", "Historical artifact status"),
    ]
    df = pd.DataFrame(rows, columns=["Artifact", "Path", "Role"])
    out = MANUSCRIPT / "tables" / "ready" / "Table12_reproducibility_package.csv"
    write_csv(df, out)
    return out


def make_derived_tables() -> dict[str, Path]:
    return {
        "Table01": build_table01(),
        "Table02": build_table02(),
        "Table03": build_table03(),
        "Table04": build_table04(),
        "Table05": build_table05(),
        "Table06": build_table06(),
        "Table07": build_table07(),
        "Table08": build_table08(),
        "Table09": build_table09(),
        "Table10": build_table10(),
        "Table11": build_table11(),
        "Table12": build_table12(),
    }


def register_copy(src: Path, dst: Path, assets: list[AssetRow], **meta) -> None:
    copy_file(src, dst)
    assets.append(
        AssetRow(
            asset_id=meta["asset_id"],
            asset_type=meta["asset_type"],
            title=meta["title"],
            status=meta["status"],
            manuscript_target=str(dst.relative_to(MANUSCRIPT)),
            primary_file=str(src),
            source_files=meta.get("source_files", str(src)),
            note=meta.get("note", ""),
        )
    )


def make_assets(derived_tables: dict[str, Path]) -> list[AssetRow]:
    assets: list[AssetRow] = []
    figs_ready = ensure_dir(MANUSCRIPT / "figures" / "ready")
    figs_source = ensure_dir(MANUSCRIPT / "figures" / "source_bundles")
    tables_ready = ensure_dir(MANUSCRIPT / "tables" / "ready")
    tables_source = ensure_dir(MANUSCRIPT / "tables" / "source_bundles")
    source_docs = ensure_dir(MANUSCRIPT / "source_docs")

    # outline and key reference docs
    for src in [
        ROOT / "TGRS_VCEA_SS_VCEA_full_manuscript_writing_outline_final.docx",
        OUT / "hardening_package" / "tgrs_vcea_hardening_detailed_report.md",
        OUT / "staticval_ea_study" / "staticval_ea_why_stronger_full_report.md",
        OUT / "final_summary_package" / "TGRS_2019_2023_全流程实验总结与完整试验记录.docx",
    ]:
        copy_file(src, source_docs / src.name)

    # copy derived tables as official manuscript tables
    for tid, path in derived_tables.items():
        register_copy(
            path,
            tables_ready / path.name,
            assets,
            asset_id=tid,
            asset_type="table",
            title=path.stem,
            status="ready",
            note="curated manuscript-ready table",
        )

    # figure source bundles
    figure_bundles = {
        "Fig01": {
            "title": "Study area map: main-six + external tiles + regimes",
            "status": "needs_new_draw",
            "files": [
                OUT / "Table1_tile_metadata.csv",
                OUT / "confirmation" / "external_tiles" / "external_tile_manifest.csv",
            ],
            "note": "No final study-area figure exists yet; use these metadata files to redraw Fig.1.",
        },
        "Fig02": {
            "title": "Leakage-safe preprocessing pipeline",
            "status": "needs_new_draw",
            "files": [
                OUT / "leakage_check_report.md",
                OUT / "normalization_policy.md",
                OUT / "preprocess_config.yaml",
                OUT / "infrastructure_v2" / "scope_decision_apply_report.md",
            ],
            "note": "Pipeline figure should be redrawn from protocol docs; not copied as final image.",
        },
        "Fig03": {
            "title": "Strong persistence baseline and horizon difficulty",
            "status": "needs_new_draw",
            "files": [
                OUT / "supplements" / "A2_E3_baselines" / "a2_e3_act_vs_baselines.csv",
                OUT / "act_v2" / "m1_suite" / "step1_combined_summary.csv",
                OUT / "act_v7" / "finalization" / "audited_full6_summary.csv",
            ],
            "note": "No dedicated final figure exists; redraw from baseline and full-six horizon summaries.",
        },
        "Fig04": {
            "title": "VCEA / SS-VCEA method diagram",
            "status": "needs_new_draw",
            "files": [
                OUT / "staticval_ea_study" / "methods_static_online_aggregation.md",
                OUT / "act_v7" / "v7_final_model_definition.md",
                OUT / "hardening_package" / "hardening_paper_insert.md",
            ],
            "note": "Method figure should be newly drawn from these method definitions.",
        },
        "Fig05": {
            "title": "VTSA mechanism diagram",
            "status": "ready",
            "files": [
                OUT / "hardening_package" / "vtsa" / "fig_vtsa_mechanism.pdf",
                OUT / "hardening_package" / "vtsa" / "vtsa_protocol_comparison.csv",
                OUT / "hardening_package" / "vtsa" / "vtsa_rank_stability.csv",
            ],
            "note": "Existing ready figure plus supporting CSV.",
        },
        "Fig06": {
            "title": "Main-six tile-wise heatmap",
            "status": "partial_ready",
            "files": [
                OUT / "act_v7" / "finalization" / "v7_figure4_skill_heatmaps.png",
                OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon.csv",
            ],
            "note": "Existing heatmap can be adapted; manuscript may split/relabel panels.",
        },
        "Fig07": {
            "title": "Horizon-wise heatmap",
            "status": "partial_ready",
            "files": [
                OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon_heatmaps.pdf",
                OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_tile_horizon.csv",
            ],
            "note": "Existing selected tile-horizon heatmap asset; likely needs panel extraction.",
        },
        "Fig08": {
            "title": "Validation ranking vs test ranking",
            "status": "ready",
            "files": [
                OUT / "staticval_ea_study" / "fig_val_test_rank_scatter.png",
                OUT / "staticval_ea_study" / "expert_rank_stability.csv",
            ],
            "note": "Direct VTSA mechanism evidence.",
        },
        "Fig09": {
            "title": "COEA drift vs static VCEA weights",
            "status": "ready",
            "files": [
                OUT / "staticval_ea_study" / "fig_static_vs_coea_weight_drift.png",
                OUT / "staticval_ea_study" / "weight_drift_metrics.csv",
                OUT / "hardening_package" / "vtsa" / "vtsa_coea_drift.csv",
            ],
            "note": "Direct evidence for why VCEA outperforms COEA.",
        },
        "Fig10": {
            "title": "Skill-stress Pareto curve",
            "status": "ready",
            "files": [
                OUT / "hardening_package" / "ss_vcea" / "fig_stress_skill_pareto.pdf",
                OUT / "hardening_package" / "ss_vcea" / "ss_vcea_skill_stress_pareto.csv",
            ],
            "note": "Ready safety trade-off figure.",
        },
        "Fig11": {
            "title": "External locked tile results",
            "status": "ready",
            "files": [
                OUT / "confirmation" / "figures" / "fig_external_tile_generalization.png",
                OUT / "confirmation" / "external_tiles" / "external_summary.csv",
                OUT / "confirmation" / "external_tiles" / "external_tile_horizon.csv",
            ],
            "note": "Ready external locked robustness figure.",
        },
        "Fig12": {
            "title": "Resolution sensitivity curves",
            "status": "ready",
            "files": [
                OUT / "confirmation" / "figures" / "fig_resolution_sensitivity.png",
                OUT / "hardening_package" / "resolution_aware" / "resolution_family_heatmap.pdf",
                OUT / "confirmation" / "model_family_resolution_extension" / "family_resolution_ranking.csv",
            ],
            "note": "Ready resolution sensitivity visual assets.",
        },
        "Fig13": {
            "title": "Model search audit timeline",
            "status": "needs_new_draw",
            "files": [
                MANUSCRIPT / "tables" / "ready" / "Table11_architecture_search_audit.csv",
                OUT / "act_v2" / "go_no_go_act_v2.md",
                OUT / "act_v7" / "finalization" / "v7_finalization_report.md",
                OUT / "staticval_ea_study" / "final_aggregation_decision.md",
            ],
            "note": "No existing timeline figure; redraw from the curated audit table and decision reports.",
        },
        "Fig14": {
            "title": "Physics / regime diagnostic panel",
            "status": "partial_ready",
            "files": [
                OUT / "staticval_ea_study" / "stress_tile_regime_panel.png",
                OUT / "E6" / "e6_regime_out.csv",
                OUT / "E8" / "e8_physics_diagnostics.csv",
                OUT / "supplements" / "A4_E8_redefined" / "a4_e8_redefined_metrics.csv",
            ],
            "note": "Existing diagnostic panel plus regime and physics CSVs; manuscript may recompose this into the final panel.",
        },
    }

    for fig_id, bundle in figure_bundles.items():
        bundle_dir = figs_source / fig_id
        ensure_dir(bundle_dir)
        copied_paths: list[str] = []
        primary_file = ""
        for idx, src in enumerate(bundle["files"]):
            if src.exists():
                dst = bundle_dir / src.name
                copy_file(src, dst)
                copied_paths.append(str(src))
                if idx == 0:
                    primary_file = str(src)
        write_text(bundle_dir / "README.txt", bundle["note"])
        assets.append(
            AssetRow(
                asset_id=fig_id,
                asset_type="figure",
                title=bundle["title"],
                status=bundle["status"],
                manuscript_target=str(bundle_dir.relative_to(MANUSCRIPT)),
                primary_file=primary_file,
                source_files="; ".join(copied_paths),
                note=bundle["note"],
            )
        )

    # Direct ready copies for convenience
    ready_direct = {
        "Fig05_ready": OUT / "hardening_package" / "vtsa" / "fig_vtsa_mechanism.pdf",
        "Fig08_ready": OUT / "staticval_ea_study" / "fig_val_test_rank_scatter.png",
        "Fig09_ready": OUT / "staticval_ea_study" / "fig_static_vs_coea_weight_drift.png",
        "Fig10_ready": OUT / "hardening_package" / "ss_vcea" / "fig_stress_skill_pareto.pdf",
        "Fig11_ready": OUT / "confirmation" / "figures" / "fig_external_tile_generalization.png",
        "Fig12_ready": OUT / "confirmation" / "figures" / "fig_resolution_sensitivity.png",
    }
    for asset_id, src in ready_direct.items():
        copy_file(src, figs_ready / src.name)

    # source bundles for tables
    table_sources = {
        "Table01_source": [OUT / "Table1_tile_metadata.csv", OUT / "confirmation" / "external_tiles" / "external_tile_manifest.csv"],
        "Table06_source": [OUT / "staticval_ea_study" / "final_aggregation_comparison.csv", OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_main6_summary.csv", OUT / "act_v4" / "v4c_efficiency_rescue" / "c3_candidate" / "c3_full6_verdict.json"],
        "Table07_source": [OUT / "hardening_package" / "vtsa" / "vtsa_protocol_comparison.csv", OUT / "hardening_package" / "vtsa" / "vtsa_rank_stability.csv", OUT / "hardening_package" / "vtsa" / "vtsa_topk_retention.csv"],
        "Table08_source": [OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_main6_summary.csv", OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_external_summary.csv"],
        "Table09_source": [OUT / "confirmation" / "external_tiles" / "external_summary.csv", OUT / "confirmation" / "external_tiles" / "external_stress_report.csv", OUT / "hardening_package" / "ss_vcea" / "ss_vcea_selected_external_summary.csv"],
        "Table10_source": [OUT / "confirmation" / "model_family_resolution_extension" / "family_resolution_ranking.csv"],
        "Table11_source": [OUT / "act_v3" / "final_model_decision.md", OUT / "act_v4" / "v4_final_decision.md", OUT / "act_v5" / "v5_final_decision.md", OUT / "act_v6" / "v6_final_decision.md", OUT / "act_v8" / "v8_final_decision.md", OUT / "dscf" / "dscf_final_decision.md"],
        "Table12_source": [OUT / "hardening_package" / "reproducibility" / "result_lineage_table.csv", OUT / "hardening_package" / "reproducibility" / "figure_source_map.csv", OUT / "hardening_package" / "reproducibility" / "paper_table_source_map.csv", OUT / "hardening_package" / "reproducibility" / "exact_proxy_transfer_flags.csv"],
    }
    for table_id, srcs in table_sources.items():
        d = tables_source / table_id
        ensure_dir(d)
        for src in srcs:
            if src.exists():
                copy_file(src, d / src.name)

    return assets


def make_manifest(assets: list[AssetRow]) -> None:
    df = pd.DataFrame([a.__dict__ for a in assets])
    df = df.sort_values(["asset_type", "asset_id"])
    write_csv(df, MANUSCRIPT / "manuscript_asset_index.csv")
    write_csv(df[df["asset_type"] == "figure"], MANUSCRIPT / "figure_asset_status.csv")
    write_csv(df[df["asset_type"] == "table"], MANUSCRIPT / "table_asset_status.csv")

    missing = df[df["status"].isin(["needs_new_draw", "partial_ready"])]
    lines = [
        "# Manuscript Asset Notes",
        "",
        "本目录已把大部分表格整理为可直接写作使用的 `csv`，并把现有图、图源数据和缺口整理到了统一位置。",
        "",
        "## 仍需新画或二次整理的图片",
        "",
    ]
    for _, row in missing[missing["asset_type"] == "figure"].iterrows():
        lines.append(f"- `{row['asset_id']}` | `{row['title']}` | 状态=`{row['status']}` | {row['note']}")
    lines.extend(
        [
            "",
            "## 已直接可用的核心成图",
            "",
            "- `Fig05` VTSA mechanism",
            "- `Fig08` validation ranking vs test ranking",
            "- `Fig09` COEA drift vs static VCEA",
            "- `Fig10` skill-stress Pareto",
            "- `Fig11` external locked generalization",
            "- `Fig12` resolution sensitivity",
        ]
    )
    write_text(MANUSCRIPT / "missing_or_redraw_items.md", "\n".join(lines))

    ready_fig = int((df["asset_type"] == "figure").sum())
    ready_tbl = int((df["asset_type"] == "table").sum())
    note = f"""# TGRS Manuscript Asset Package

生成日期：2026-05-14

目录用途：

- 按 `TGRS_VCEA_SS_VCEA_full_manuscript_writing_outline_final.docx` 统一整理 manuscript 所需图片、表格、源数据和关键支撑文档。
- `tables/ready` 下是当前可直接写稿的整理版表格。
- `figures/source_bundles` 下每个 `FigXX` 子目录包含现有图或重画所需源数据。
- `source_docs` 下保留提纲和关键总报告，便于写作时快速查证。

统计：

- 资产索引总数：{len(df)}
- figure 项：{len(df[df['asset_type'] == 'figure'])}
- table 项：{len(df[df['asset_type'] == 'table'])}
- 可直接写作的整理表：{ready_tbl}

重点提醒：

- `Fig01 / Fig02 / Fig03 / Fig04 / Fig13` 仍需要新画。
- `Fig06 / Fig07 / Fig14` 已有现成资产，但更适合作为 panel source，投稿前建议按提纲重排。
- `Table06-Table12` 已经整理成 manuscript-friendly `csv`，可以直接作为正文/附录表草稿底稿。
"""
    write_text(MANUSCRIPT / "README.md", note)


def main() -> None:
    if MANUSCRIPT.exists():
        shutil.rmtree(MANUSCRIPT)
    ensure_dir(MANUSCRIPT)
    derived = make_derived_tables()
    assets = make_assets(derived)
    make_manifest(assets)
    print(MANUSCRIPT)


if __name__ == "__main__":
    main()
