#!/usr/bin/env python
"""Build every table/figure the IJRS manuscript needs from existing result files.

Read-only with respect to experiment outputs; writes only to ./tables and ./figures.
Run:  python make_tables.py
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PR = HERE.parent                                   # pythonProject
AN = PR / "analysis_vcea_comparators_20261104"
OUT = PR / "outputs" / "tgrs_2019_2023"
E7 = OUT / "vcea_extra_experiments" / "E7_pan_europe_locked_audit_strong"
T, F = HERE / "tables", HERE / "figures"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
H = ["t+1", "t+3", "t+5", "t+10"]
sys.path.insert(0, str(AN))
import run_vcea_comparators as rc  # noqa: E402

# ---------------- skill tables per set
dev_e7 = pd.read_csv(AN / "results" / "skill_by_tile_horizon.csv")              # main6 + e7, seed-averaged
fresh = pd.read_csv(AN / "results" / "fresh" / "skill_by_tile_seed_horizon.csv")
fresh = fresh.groupby(["Model", "Tile", "Horizon"], as_index=False)["Skill_MSE"].mean().assign(Set="fresh")
allsk = pd.concat([dev_e7, fresh], ignore_index=True)
allsk.to_csv(T / "skill_by_set_model_tile_horizon.csv", index=False)

order = ["VCEA", "BatesGranger", "NNLS_simplex", "V3_EvidenceGate", "GlobalVal", "SelectBestVal", "DampedLS_val",
         "Uniform4", "VCEA_noLS", "VCEA_noNeural", "M1", "C3_opt", "Persistence", "Linear-Seasonal", "OracleSingle_test"]
tile_mean = allsk.groupby(["Set", "Model", "Tile"])["Skill_MSE"].mean(numeric_only=True)
t2 = tile_mean.groupby(["Set", "Model"]).mean().unstack(0).reindex(order)[["main6", "e7", "fresh"]]
t2.columns = ["Development (6 tiles, 2019-2023)", "Held-out audit E7 (14 tiles, 2019-2023)", "Confirmatory (24 tiles, 2020-2024)"]
t2.loc["V3_EvidenceGate"] = np.nan   # NaN-skipping mean is misleading; see T3b (20 defined tiles)
t2.round(4).to_csv(T / "T2_main_comparison_three_sets.csv")
(T / "T2_notes.md").write_text(
    "- Values are means over tiles of (mean over 4 horizons and 3 seeds) Skill_MSE.\n"
    "- GlobalVal, DampedLS_val and OracleSingle_test were not part of the pre-registered confirmatory evaluation (blank for 2020-2024).\n"
    "- V3_EvidenceGate is blank: it is undefined on 4 confirmatory tiles with a single validation window. On the 20 tiles where it\n"
    "  is defined: V3 0.1909, VCEA 0.1909, Bates-Granger 0.1829 (T3b_posthoc_diagnostics.json).\n"
    "- Uniform4 on development tiles is dominated by unbounded linear-seasonal extrapolation (LS mean Skill -36.1).\n"
    "- OracleSingle_test uses test labels (diagnostic upper bound for any selector).\n", encoding="utf-8")

# ---------------- ablation and horizon tables
abl = ["VCEA", "BatesGranger", "VCEA_noLS", "VCEA_noNeural", "M1"]
t2.loc[abl].round(4).to_csv(T / "T4_expert_contribution.csv")
hz = allsk[allsk.Model.isin(["VCEA", "BatesGranger", "NNLS_simplex", "M1"])].groupby(["Set", "Model", "Horizon"])["Skill_MSE"].mean().unstack(2)[H]
hz.round(4).to_csv(T / "T5_horizon_by_set.csv")

# ---------------- confirmatory (pre-registered) + post-hoc
rep = json.loads((AN / "results" / "fresh" / "confirmatory_report.json").read_text())
prim = pd.DataFrame(rep["primary"]); prim["family"] = "primary (pre-registered)"
sec = pd.DataFrame(rep["secondary"]); sec["family"] = "secondary (pre-registered)"
conf = pd.concat([prim, sec], ignore_index=True)
conf["note"] = ""
conf.loc[conf.contrast.str.startswith("V3_EvidenceGate"), "note"] = (
    "V3 undefined (NaN) on 4 tiles with 1 validation window; reported p_holm is an artefact of NaN handling; verdict = not supported")
conf.to_csv(T / "T3_confirmatory_preregistered.csv", index=False)
post = json.loads((AN / "results" / "fresh" / "posthoc_diagnostics.json").read_text())
(T / "T3b_posthoc_diagnostics.json").write_text(json.dumps(post, indent=2))

# ---------------- static ceilings (diagnostic upper bounds, test-fitted)
ceil = pd.read_csv(AN / "results" / "oracle_static_simplex_test_upper_bound.csv").groupby("Set").Skill.mean()
def regime_oracle_e7() -> float:
    rows = []
    for t in rc.E7_TILES:
        for s in rc.SEEDS:
            reg = np.load(rc.E7_RUNS / "M1" / t / f"seed_{s}" / "predictions.npz")["regime_map"]
            ct = rc._pair(rc.CACHE / "e7" / t / f"M1_seed_{s}_test.npz", rc.CACHE / "e7" / t / f"C3_seed_{s}_test.npz")
            v = ct["valid"]; rr = reg[v]
            for h in range(4):
                yy = ct["y"][:, h][:, v].astype(np.float64); XX = ct["X"][:, :, h][:, :, v].astype(np.float64)
                pred = np.zeros_like(yy)
                for k in np.unique(rr):
                    m = rr == k
                    sub = {"y": yy[:, m][:, None, :], "X": XX[:, :, m][:, :, None, :], "valid": np.ones(m.sum(), bool)}
                    X2 = np.stack([XX[e][:, m].ravel() for e in range(4)], 1); y2 = yy[:, m].ravel()
                    G, b = X2.T @ X2, X2.T @ y2; best, bo = None, np.inf
                    for r in range(1, 5):
                        for S in itertools.combinations(range(4), r):
                            S = list(S); K = np.zeros((r + 1, r + 1)); K[:r, :r] = 2 * G[np.ix_(S, S)]; K[:r, r] = 1; K[r, :r] = 1
                            w_s = np.linalg.lstsq(K, np.r_[2 * b[S], 1.0], rcond=None)[0][:r]
                            if np.all(w_s >= -1e-9):
                                w = np.zeros(4); w[S] = np.clip(w_s, 0, None); w /= w.sum(); o = w @ G @ w - 2 * b @ w
                                if o < bo: best, bo = w, o
                    pred[:, m] = np.tensordot(best, XX[:, :, m], axes=1)
                rows.append(1 - np.mean((pred - yy) ** 2) / np.mean((XX[0] - yy) ** 2))
    return float(np.mean(rows))
t6 = pd.DataFrame({
    "Bates-Granger (validation)": [t2.loc["BatesGranger"].iloc[0], t2.loc["BatesGranger"].iloc[1]],
    "VCEA (validation)": [t2.loc["VCEA"].iloc[0], t2.loc["VCEA"].iloc[1]],
    "Oracle static per tile x horizon (test-fitted)": [ceil["main6"], ceil["e7"]],
    "Oracle per regime class (test-fitted)": [np.nan, regime_oracle_e7()],
}, index=["main6", "e7"]).round(4)
t6.to_csv(T / "T6_static_ceiling_diagnostic.csv")

# ---------------- strict split protocols (development tiles)
pd.read_csv(OUT / "vcea_extra_experiments" / "tables" / "table_E1_leakage_audit.csv").to_csv(T / "T7_split_protocol_audit_main6.csv", index=False)

# ---------------- compute
rows = []
for fam, nparam in (("M1", 190213), ("C3_opt", 69929)):
    el = [json.loads(p.read_text()).get("elapsed_seconds") for p in (OUT / "vcea_fresh_confirmatory_audit" / "runs" / fam).glob("*/seed_*/result.json")]
    rows.append({"Expert": fam, "Parameters": nparam, "Train s per tile-seed (median, GPU)": round(float(np.median(el)), 1),
                 "Train s (max)": round(float(np.max(el)), 1), "Runs": len(el)})
rows.append({"Expert": "Linear-Seasonal", "Parameters": "per-pixel least squares", "Train s per tile-seed (median, GPU)": "negligible"})
rows.append({"Expert": "Combiners (BG / VCEA / NNLS)", "Parameters": "4 weights per tile x horizon", "Train s per tile-seed (median, GPU)": "< 1 s (CPU)"})
pd.DataFrame(rows).to_csv(T / "T8_compute.csv", index=False)

# ---------------- data / tiles
e7meta = pd.read_csv(E7 / "csv" / "table_E7_pan_europe_locked_results_by_tile.csv").groupby("tile_id")[["frames", "valid_grid_ratio_64", "temporal_span", "broad_region", "audit_regime_group"]].first()
ft = pd.read_csv(AN / "prereg" / "fresh_tiles.csv")
fr = []
for t in ft.tile_id:
    z = np.load(AN / "cache" / "fresh" / t / "M1_seed_42_val.npz"); zt = np.load(AN / "cache" / "fresh" / t / "M1_seed_42_test.npz")
    hdr = pd.read_csv(next((PR / "data" / "extracted_2020_2024" / t).glob("*.csv")), nrows=0).columns
    fr.append({"tile_id": t, "frames": sum(len(c) == 8 and c.isdigit() for c in hdr), "val_windows": z["y_true_mm"].shape[0],
               "test_windows": zt["y_true_mm"].shape[0], "valid_grid_ratio_64": float((z["valid_mask"] > 0.5).mean())})
fr = pd.DataFrame(fr).merge(ft[["tile_id", "centroid_lat", "centroid_lon", "lat_band", "lon_band", "size_stratum"]], on="tile_id")
fr.round(3).to_csv(T / "T1b_fresh_tiles.csv", index=False)
e7meta.to_csv(T / "T1a_e7_tiles.csv")

# ---------------- operating boundaries
fb = fr[fr.val_windows < 10][["tile_id", "frames", "val_windows", "valid_grid_ratio_64"]]
ftile = fresh.groupby(["Model", "Tile"]).Skill_MSE.mean().unstack(0)
fb = fb.merge(ftile[["VCEA", "BatesGranger", "M1", "Uniform4"]].round(3), left_on="tile_id", right_index=True)
fb.to_csv(T / "T9_operating_boundaries_fresh.csv", index=False)

# ---------------- dev selection of the variant
fz = json.loads((AN / "prereg" / "frozen_variant.json").read_text())
pd.DataFrame(fz["dev_by_horizon"]).T.assign(mean=pd.Series(fz["dev_mean_nested_suffix_skill"])).round(4).to_csv(T / "T10_variant_dev_selection.csv")

# ---------------- figures (draft quality)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)   # each panel sorted separately -> own tile labels
for a, c in zip(ax, ["SelectBestVal", "Uniform4", "M1"]):
    d = (ftile["BatesGranger"] - ftile[c]).sort_values()
    a.barh(range(len(d)), d.values, color=["#3b75af" if x > 0 else "#c44e52" for x in d.values])
    a.axvline(0, color="k", lw=0.8); a.set_title(f"Bates-Granger minus {c}", fontsize=9); a.set_yticks(range(len(d))); a.set_yticklabels(d.index, fontsize=6)
    a.set_xlabel("Tile-level Skill_MSE difference")
fig.tight_layout(); fig.savefig(F / "fig_confirmatory_paired_differences.png", dpi=200)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
for a, s, ttl in zip(ax, ["main6", "e7", "fresh"], ["Development", "Held-out audit (E7)", "Confirmatory (2020-2024)"]):
    for m, st in (("VCEA", "-o"), ("BatesGranger", "-s"), ("NNLS_simplex", "--^"), ("M1", ":x")):
        a.plot(H, hz.loc[(s, m)].values, st, label=m, ms=4)
    a.set_title(ttl, fontsize=9); a.set_xlabel("Horizon")
ax[0].set_ylabel("Mean Skill_MSE"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / "fig_horizon_profiles_three_sets.png", dpi=200)
print(t2.round(4).to_string()); print(t6.to_string()); print(pd.read_csv(T / "T8_compute.csv").to_string())
