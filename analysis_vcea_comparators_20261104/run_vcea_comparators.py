#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VCEA comparator audit (R1): VCEA vs pre-specified simple forecast combiners.

This script never edits existing experiment outputs. Everything it writes goes
under the folder that contains this file (cache/ and results/).

Usage (from anywhere):
    python run_vcea_comparators.py infer      # needs project env + torch (GPU optional)
    python run_vcea_comparators.py compare    # numpy / pandas / scipy only
    python run_vcea_comparators.py all

Stage `infer`
    Re-runs inference (no training, no tuning) of the frozen per-tile M1 and C3_opt
    checkpoints of the E7 14-tile audit on the validation and test splits and caches
    the expert arrays in cache/e7/<tile>/. The re-inferred test arrays are checked
    against runs/*/predictions.npz.
Stage `compare`
    Evaluates every comparator below on
      (a) main6: the six development tiles (act_v7 seed_expert_cache, val + test)
      (b) e7:    the 14-tile frozen audit (cache/e7)
    and writes CSV tables plus results/summary.md.

Comparators are fixed here, before any result is inspected. All weights are
estimated on the validation split only; test labels are used for scoring only.

  Single experts : Persistence, Linear-Seasonal, M1, C3_opt
  VCEA (paper)   : inverse-loss, tau=4, rho=0.1, no floor (replica of
                   StaticVal_inverse_temp4p0_sh0p1_floor0p0)
  Simple combiners (the reference class for the decision rule):
    Uniform4      : equal weights over P, LS, M1, C3
    BatesGranger  : inverse validation MSE, tau=1, rho=0 (classic)
    SelectBestVal : per tile x horizon, the single expert with lowest val MSE
    NNLS_simplex  : per tile x horizon, least squares on val with w>=0, sum w=1
    GlobalVal     : VCEA transform (tau=4, rho=0.1) on relative val losses
                    averaged over all tiles and horizons of the set (one weight
                    vector per seed, not tile- or horizon-aware)
    DampedLS_val  : P + phi*(LS-P), phi in {0,.25,.5,.75,1} chosen per
                    tile x horizon on val (a bounded kinematic baseline)
  VCEA ablations : VCEA_rho0 (tau=4, rho=0), VCEA_noLS (P,M1,C3),
                   VCEA_noNeural (P,LS)
  Diagnostic only: OracleSingle_test (best single expert chosen on TEST; upper
                   bound for any selector, not an operational method)

Scoring: Skill_MSE = 1 - MSE(model)/MSE(persistence) pooled over windows and valid
cells, per tile x horizon x seed (same as act_v2.skill_scores); averaged over
seeds, then over horizons for the tile-level unit used in the paired tests.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT = PROJECT_ROOT / "outputs" / "tgrs_2019_2023"
MAIN6_CACHE = OUT / "act_v7" / "seed_expert_cache"
E7_ROOT = OUT / "vcea_extra_experiments" / "E7_pan_europe_locked_audit_strong"
E7_RUNS = E7_ROOT / "runs"
E7_TABLE = E7_ROOT / "csv" / "table_E7_pan_europe_locked_results_by_tile.csv"
CACHE = HERE / "cache"
RES = HERE / "results"

MAIN6 = ["E32N34", "E32N35", "E39N30", "E44N23", "E48N24", "E58N17"]
E7_TILES = ["E27N24", "E31N16", "E34N39", "E39N23", "E40N31", "E44N30", "E44N43",
            "E48N26", "E48N32", "E49N22", "E49N44", "E55N14", "E57N25", "E58N26"]
SEEDS = [42, 43, 44]
HORIZONS = ["t+1", "t+3", "t+5", "t+10"]
ENAMES = ["P", "LS", "M1", "C3"]
DAMP_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
KEEP = ["y_pred_mm", "y_true_mm", "persistence_mm", "linear_seasonal_mm",
        "valid_mask", "current_frames", "target_indices"]
SIMPLE = ["Uniform4", "BatesGranger", "SelectBestVal", "NNLS_simplex", "GlobalVal", "DampedLS_val"]
DECISION_MARGIN = 0.02  # pre-stated: VCEA must beat the best simple combiner by >= 0.02


# --------------------------------------------------------------------------- io
def _pair(m1_path: Path, c3_path: Path) -> dict[str, np.ndarray]:
    a, b = np.load(m1_path), np.load(c3_path)
    if not np.array_equal(a["current_frames"].reshape(-1), b["current_frames"].reshape(-1)):
        raise ValueError(f"frame mismatch {m1_path}")
    if not np.array_equal(a["target_indices"], b["target_indices"]):
        raise ValueError(f"target mismatch {m1_path}")
    return {
        "y": a["y_true_mm"].astype(np.float32),
        "X": np.stack([a["persistence_mm"], a["linear_seasonal_mm"], a["y_pred_mm"], b["y_pred_mm"]], 0).astype(np.float32),
        "valid": a["valid_mask"] > 0.5,
    }


def load_set(name: str) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    data = {}
    if name == "main6":
        for t in MAIN6:
            for s in SEEDS:
                for sp in ("val", "test"):
                    data[(t, s, sp)] = _pair(MAIN6_CACHE / "M1" / t / f"seed_{s}_{sp}.npz",
                                             MAIN6_CACHE / "C3_opt" / t / f"seed_{s}_{sp}.npz")
    elif name == "e7":
        for t in E7_TILES:
            for s in SEEDS:
                for sp in ("val", "test"):
                    data[(t, s, sp)] = _pair(CACHE / "e7" / t / f"M1_seed_{s}_{sp}.npz",
                                             CACHE / "e7" / t / f"C3_seed_{s}_{sp}.npz")
    else:
        raise KeyError(name)
    return data


# ------------------------------------------------------------------ inference
def _enable_lowmem_csv() -> None:
    """Low-RAM EGMS CSV reader used only by the bundle builder.

    build_bundle_from_tile_id only uses the easting/northing columns, the YYYYMMDD
    displacement columns and len(frame); all of them are cast to float32 right after
    reading. This reader loads exactly those columns as float32 into one preallocated
    block, so the values seen by the pipeline are identical to the original path.
    """
    import pandas as _pd
    orig = _pd.read_csv

    def lowmem_read_csv(path, *args, **kwargs):
        if args or kwargs or not str(path).lower().endswith(".csv"):
            return orig(path, *args, **kwargs)
        header = orig(path, nrows=0).columns.tolist()
        if "easting" not in header or "northing" not in header:  # not an EGMS point CSV
            return orig(path)
        keep = ["easting", "northing"] + [c for c in header if len(str(c)) == 8 and str(c).isdigit()]
        with open(path, "rb") as fh:
            n_rows = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 24), b"")) - 1
        block = np.empty((n_rows + 1, len(keep)), dtype=np.float32)
        pos = 0
        for ch in orig(path, usecols=keep, dtype={c: np.float32 for c in keep}, chunksize=100_000):
            arr = ch[keep].to_numpy(dtype=np.float32)
            block[pos:pos + len(arr)] = arr
            pos += len(arr)
        return _pd.DataFrame(block[:pos], columns=keep, copy=False)

    _pd.read_csv = lowmem_read_csv


def _chunked_fit_descriptors(orig_fit, chunk: int = 100_000):
    """Row-chunked wrapper of pipeline.fit_descriptors (low-RAM mode only).

    Every descriptor is per point except change_score, which is normalised by the
    median and MAD over all points; that normalisation is applied once on the full
    vector here, so the output equals the original function.
    """
    def wrapped(values: np.ndarray, time_years: np.ndarray) -> dict[str, np.ndarray]:
        n = values.shape[0]
        if n <= chunk:
            return orig_fit(values, time_years)
        half = len(time_years) // 2
        outs, diffs = [], []
        for i in range(0, n, chunk):
            v = values[i:i + chunk]
            o = orig_fit(v, time_years)
            o.pop("change_score")
            outs.append(o)
            diffs.append(np.abs(v[:, half:].mean(axis=1) - v[:, :half].mean(axis=1)))
        res = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        diff = np.concatenate(diffs)
        med = np.median(diff)
        mad = np.median(np.abs(diff - med)) + 1e-6
        res["change_score"] = ((diff - med) / mad).astype(np.float32)
        return res

    return wrapped


def stage_infer(tiles: list[str], lowmem: bool = False) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch  # noqa: F401
    if lowmem:
        _enable_lowmem_csv()
        import tgrs_experiments.dscf_confirmation as _dc
        _dc.fit_descriptors = _chunked_fit_descriptors(_dc.fit_descriptors)
    from tgrs_experiments.act_v5_selector import infer_c3_split, infer_m1_split
    from tgrs_experiments.dscf_confirmation import build_bundle_from_tile_id
    from tgrs_experiments.pipeline import EXTRACTED_DIR, RAW_DIR

    log = []
    for t in tiles:
        out = CACHE / "e7" / t
        want = [out / f"{fam}_seed_{s}_{sp}.npz" for fam in ("M1", "C3") for s in SEEDS for sp in ("val", "test")]
        if all(p.exists() for p in want):
            print(f"[infer] {t}: cached, skip", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        d = EXTRACTED_DIR / t
        raw_files = {t: {"zip": next(RAW_DIR.glob(f"EGMS_L3_{t}_*_2019_2023_*.zip")),
                         "csv": next(d.glob("*.csv")), "xml": next(d.glob("*.xml")), "dir": d}}
        bundle = build_bundle_from_tile_id(t, raw_files)
        t_bundle = time.time() - t0
        for s in SEEDS:
            for sp in ("val", "test"):
                m1 = infer_m1_split(bundle, E7_RUNS / "M1" / t / f"seed_{s}", sp)
                c3 = infer_c3_split(bundle, E7_RUNS / "C3_opt" / t / f"seed_{s}", sp)
                np.savez_compressed(out / f"M1_seed_{s}_{sp}.npz", **{k: m1[k] for k in KEEP})
                np.savez_compressed(out / f"C3_seed_{s}_{sp}.npz", **{k: c3[k] for k in KEEP})
                if sp == "test":  # reproducibility check against the archived audit run
                    for fam, arr in (("M1", m1), ("C3_opt", c3)):
                        ref = np.load(E7_RUNS / fam / t / f"seed_{s}" / "predictions.npz")
                        diff = float(np.max(np.abs(ref["y_pred_mm"] - arr["y_pred_mm"])))
                        log.append({"tile": t, "seed": s, "family": fam, "max_abs_diff_test_pred_mm": diff})
        del bundle
        gc.collect()
        print(f"[infer] {t}: done in {time.time() - t0:.1f}s (bundle {t_bundle:.1f}s)", flush=True)
    if log:
        RES.mkdir(parents=True, exist_ok=True)
        p = RES / "infer_test_reproduction_check.csv"
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(log)], ignore_index=True).drop_duplicates(["tile", "seed", "family"], keep="last").to_csv(p, index=False)


# ------------------------------------------------------------------ methods
def val_losses(c: dict, h: int) -> np.ndarray:
    v, y = c["valid"], c["y"][:, h][:, c["valid"]].astype(np.float64)
    return np.array([np.mean((c["X"][m][:, h][:, v] - y) ** 2) for m in range(4)])


def inv_weights(losses: np.ndarray, tau: float, rho: float) -> np.ndarray:
    rel = losses / max(losses[0], 1e-8)
    raw = 1.0 / np.power(np.maximum(rel, 1e-8), tau)
    raw = raw / max(raw.sum(), 1e-8)
    raw = (1 - rho) * raw + rho / len(raw)
    raw = np.clip(raw, 1e-8, None)
    return raw / raw.sum()


def subset_inv(losses: np.ndarray, idx: list[int], tau: float, rho: float) -> np.ndarray:
    w = np.zeros(4)
    w[idx] = inv_weights(losses[idx], tau, rho)  # idx[0] must be persistence
    return w


def simplex_ls(c: dict, h: int) -> np.ndarray:
    v = c["valid"]
    X = np.stack([c["X"][m][:, h][:, v].reshape(-1) for m in range(4)], 1).astype(np.float64)
    y = c["y"][:, h][:, v].reshape(-1).astype(np.float64)
    G, b = X.T @ X, X.T @ y
    best, best_obj = None, np.inf
    for r in range(1, 5):
        for S in itertools.combinations(range(4), r):
            S = list(S)
            K = np.zeros((r + 1, r + 1))
            K[:r, :r] = 2 * G[np.ix_(S, S)]
            K[:r, r] = 1
            K[r, :r] = 1
            rhs = np.r_[2 * b[S], 1.0]
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            w_s = sol[:r]
            if np.all(w_s >= -1e-9):
                w = np.zeros(4)
                w[S] = np.clip(w_s, 0, None)
                w /= w.sum()
                obj = w @ G @ w - 2 * b @ w
                if obj < best_obj:
                    best, best_obj = w, obj
    return best


def damped_phi(c: dict, h: int) -> float:
    v = c["valid"]
    y = c["y"][:, h][:, v].astype(np.float64)
    P, LS = c["X"][0][:, h][:, v].astype(np.float64), c["X"][1][:, h][:, v].astype(np.float64)
    errs = [np.mean((P + phi * (LS - P) - y) ** 2) for phi in DAMP_GRID]
    return DAMP_GRID[int(np.argmin(errs))]


def skill(pred: np.ndarray, c: dict, h: int) -> float:
    v, y = c["valid"], c["y"][:, h][:, c["valid"]].astype(np.float64)
    mse = np.mean((pred[:, v] - y) ** 2)
    pm = np.mean((c["X"][0][:, h][:, v] - y) ** 2)
    return float(1 - mse / max(pm, 1e-6))


def evaluate_set(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_set(name)
    tiles = MAIN6 if name == "main6" else E7_TILES
    rows, wrows = [], []
    for s in SEEDS:
        # GlobalVal: one weight vector per seed from relative losses pooled over tiles x horizons
        rel = np.mean([val_losses(data[(t, s, "val")], h) / max(val_losses(data[(t, s, "val")], h)[0], 1e-8)
                       for t in tiles for h in range(4)], axis=0)
        w_global = inv_weights(rel, 4.0, 0.1)
        for t in tiles:
            cv, ct = data[(t, s, "val")], data[(t, s, "test")]
            for h, hn in enumerate(HORIZONS):
                L = val_losses(cv, h)
                W = {
                    "Persistence": np.eye(4)[0], "Linear-Seasonal": np.eye(4)[1],
                    "M1": np.eye(4)[2], "C3_opt": np.eye(4)[3],
                    "VCEA": inv_weights(L, 4.0, 0.1),
                    "VCEA_rho0": inv_weights(L, 4.0, 0.0),
                    "VCEA_noLS": subset_inv(L, [0, 2, 3], 4.0, 0.1),
                    "VCEA_noNeural": subset_inv(L, [0, 1], 4.0, 0.1),
                    "Uniform4": np.full(4, 0.25),
                    "BatesGranger": inv_weights(L, 1.0, 0.0),
                    "SelectBestVal": np.eye(4)[int(np.argmin(L))],
                    "NNLS_simplex": simplex_ls(cv, h),
                    "GlobalVal": w_global,
                }
                phi = damped_phi(cv, h)
                W["DampedLS_val"] = np.array([1 - phi, phi, 0.0, 0.0])
                Xt = ct["X"][:, :, h]  # (4, N, 64, 64)
                for model, w in W.items():
                    pred = np.tensordot(w, Xt, axes=1)
                    rows.append({"Set": name, "Model": model, "Tile": t, "Seed": s, "Horizon": hn, "Skill_MSE": skill(pred, ct, h)})
                    if model not in ("Persistence", "Linear-Seasonal", "M1", "C3_opt"):
                        wrows.append({"Set": name, "Model": model, "Tile": t, "Seed": s, "Horizon": hn,
                                      **{f"w_{e}": float(x) for e, x in zip(ENAMES, w)}})
                single = [skill(Xt[m], ct, h) for m in range(4)]
                rows.append({"Set": name, "Model": "OracleSingle_test", "Tile": t, "Seed": s, "Horizon": hn, "Skill_MSE": max(single)})
    return pd.DataFrame(rows), pd.DataFrame(wrows)


# ------------------------------------------------------------------ statistics
def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    from scipy.stats import wilcoxon
    d = a - b
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(1)
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {"n_tiles": len(d), "mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
            "wins": int((d > 0).sum()), "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)), "wilcoxon_p_two_sided": p}


def stage_compare(sets: list[str]) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    all_rows, all_w, tests, repro = [], [], [], {}
    for name in sets:
        df, wdf = evaluate_set(name)
        all_rows.append(df)
        all_w.append(wdf)
        th = df.groupby(["Set", "Model", "Tile", "Horizon"], as_index=False)["Skill_MSE"].mean()
        tile = th.groupby(["Model", "Tile"])["Skill_MSE"].mean().unstack(0)
        for comp in [m for m in tile.columns if m != "VCEA"]:
            tests.append({"Set": name, "Comparator": comp, **paired(tile["VCEA"].values, tile[comp].values, rng)})
        repro[name] = {"VCEA_mean": float(th[th.Model == "VCEA"].Skill_MSE.mean()),
                       "M1_mean": float(th[th.Model == "M1"].Skill_MSE.mean())}
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(RES / "skill_by_tile_seed_horizon.csv", index=False)
    pd.concat(all_w, ignore_index=True).to_csv(RES / "weights_by_tile_seed_horizon.csv", index=False)
    th = df.groupby(["Set", "Model", "Tile", "Horizon"], as_index=False)["Skill_MSE"].mean()
    th.to_csv(RES / "skill_by_tile_horizon.csv", index=False)
    summ = th.groupby(["Set", "Model"])["Skill_MSE"].mean().unstack(0).sort_values(sets[-1], ascending=False)
    summ.to_csv(RES / "mean_skill_by_model.csv")
    hz = th.groupby(["Set", "Model", "Horizon"])["Skill_MSE"].mean().unstack(2)[HORIZONS]
    hz.to_csv(RES / "mean_skill_by_model_horizon.csv")
    tdf = pd.DataFrame(tests)
    tdf.to_csv(RES / "vcea_vs_comparators_tile_level.csv", index=False)

    # reproduction check against the paper's numbers
    if "e7" in repro and E7_TABLE.exists():
        ref = pd.read_csv(E7_TABLE)
        repro["e7"]["paper_VCEA_mean"] = float(ref[ref.Model == "VCEA"]["Skill_MSE mean"].mean())
        repro["e7"]["paper_M1_mean"] = float(ref[ref.Model == "M1"]["Skill_MSE mean"].mean())
    if "main6" in repro:
        repro["main6"]["paper_VCEA_mean"] = 0.1776
        repro["main6"]["paper_M1_mean"] = 0.0783
    (RES / "reproduction_check.json").write_text(json.dumps(repro, indent=2))

    # summary
    lines = ["# VCEA comparator audit (R1)", "",
             "All weights from validation only; test used for scoring only. Comparator list fixed in the script header.", "",
             "## Reproduction check", "", "```", json.dumps(repro, indent=2), "```", "",
             "## Mean Skill_MSE by model", "", summ.round(4).to_markdown(), "",
             "## VCEA minus comparator, tile-level (mean over horizons and seeds)", ""]
    for name in sets:
        sub = tdf[tdf.Set == name].set_index("Comparator")
        lines += [f"### {name}", "", sub.drop(columns="Set").round(4).to_markdown(), ""]
        simple = sub.loc[[c for c in SIMPLE if c in sub.index]]
        best = summ.loc[SIMPLE, name].idxmax()
        delta = float(sub.loc[best, "mean_delta"])
        lines += [f"- Best simple combiner on {name}: **{best}** (mean {summ.loc[best, name]:.4f}); "
                  f"VCEA - best = {delta:+.4f}; pre-stated margin {DECISION_MARGIN:.2f} -> "
                  f"{'MET' if delta >= DECISION_MARGIN else 'NOT MET'} (exploratory; e7 test labels were already seen).", ""]
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["infer", "compare", "all"])
    ap.add_argument("--tiles", nargs="*", default=E7_TILES)
    ap.add_argument("--sets", nargs="*", default=["main6", "e7"])
    ap.add_argument("--lowmem", action="store_true", help="chunked float32 CSV reading for low-RAM machines")
    a = ap.parse_args()
    if a.stage in ("infer", "all"):
        stage_infer(a.tiles, lowmem=a.lowmem)
    if a.stage in ("compare", "all"):
        stage_compare(a.sets)
