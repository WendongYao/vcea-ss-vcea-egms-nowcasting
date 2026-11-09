#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-registered VCEA-variant protocol (one hypothesis, one confirmatory test).

Hypothesis (written before any variant was scored)
    VCEA's tau=4 sharpening amplifies validation noise when expert losses are close
    (the "forecast combination puzzle": estimation error erodes the benefit of
    optimised weights; Smith & Wallis 2009; Claeskens et al. 2016). Making the
    sharpness depend on how reliable the validation evidence is should keep VCEA's
    gains at long horizons without losing to Bates-Granger at short horizons.

Pre-declared variants (no others will be tried)
    V1_HorizonTau   tau_h in {1, 2, 4} per horizon, rho = 0.1; tau_h is a global
                    hyper-parameter chosen on main-six validation only.
    V2_BootBlend    w = s * w_VCEA + (1 - s) * w_BG, where s in [0, 1] is the stability
                    of the VCEA weights under a circular block bootstrap of validation
                    windows (B = 200, block = ceil(n^(1/3))). Parameter-free.
    V3_EvidenceGate g = clip(t / 2, 0, 1), t = block-corrected t-statistic of the
                    per-window loss gain of the best non-persistence expert over
                    persistence; w = inverse-loss(tau = 1 + 3g, rho = 0.1g).
                    g = 0 gives exactly Bates-Granger, g = 1 exactly VCEA. Parameter-free.

Stages
    dev            main-six VALIDATION ONLY (prefix -> suffix nested split). Never opens
                   a test file. Chooses one variant by the rule below and freezes it.
    select-fresh   pre-registers fresh confirmatory tiles from API-catalog metadata only.
    fresh-run      (run on the GPU machine with EGMS credentials) downloads the fresh
                   tiles, trains M1 / C3_opt with the frozen configs, caches val/test.
    eval           confirmatory test on the fresh tiles. Runs once (lock file).

Selection rule (dev)
    Pick the variant with the highest mean nested-suffix Skill_MSE over main-six
    tiles x seeds x horizons. It is carried forward only if it beats BOTH VCEA and
    Bates-Granger on that score; otherwise no variant is carried and the confirmatory
    test runs VCEA vs Bates-Granger only. V1 chooses its tau_h with an inner split of
    the prefix, so no variant sees the suffix before it is scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_vcea_comparators as rc  # noqa: E402

PREREG = HERE / "prereg"
FREEZE = PREREG / "frozen_variant.json"
FRESH_TILES = PREREG / "fresh_tiles.csv"
FRESH_CACHE = rc.CACHE / "fresh"
FRESH_RES = rc.RES / "fresh"
FRESH_ROOT = rc.OUT / "vcea_fresh_confirmatory_audit"
CATALOG = rc.E7_ROOT / "manifests" / "table_E7_api_tile_catalog.csv"
TAU_GRID = [1.0, 2.0, 4.0]
B_BOOT = 200
USED_TILES = rc.MAIN6 + ["E30N35", "E37N42", "E42N21", "E45N51"] + rc.E7_TILES


def code_hash() -> str:
    """Hash of every function that decides weights or scores (not of download/IO code),
    so plumbing can be fixed on the GPU machine without breaking the pre-registration."""
    import inspect
    mod = sys.modules[__name__]
    fns = [getattr(mod, n) for n in ("window_losses", "block_len", "boot_indices", "w_vcea", "w_bg",
                                      "w_v1", "w_v2", "w_v3", "variant_weights", "stage_eval")]
    fns += [rc.inv_weights, rc.subset_inv, rc.simplex_ls, rc.skill, rc._pair]
    h = hashlib.sha256()
    for f in fns:
        h.update(inspect.getsource(f).encode("utf-8"))
    h.update(repr((TAU_GRID, B_BOOT, rc.SEEDS, rc.HORIZONS)).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------- method pieces
def window_losses(c: dict, h: int) -> np.ndarray:
    v = c["valid"]
    y = c["y"][:, h][:, v].astype(np.float64)
    return np.stack([((c["X"][m][:, h][:, v] - y) ** 2).mean(1) for m in range(4)])  # (4, n)


def block_len(n: int) -> int:
    return max(2, int(np.ceil(n ** (1 / 3))))


def boot_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    b = block_len(n)
    starts = rng.integers(0, n, size=int(np.ceil(n / b)))
    return np.concatenate([(s + np.arange(b)) % n for s in starts])[:n]


def w_vcea(L):
    return rc.inv_weights(L, 4.0, 0.1)


def w_bg(L):
    return rc.inv_weights(L, 1.0, 0.0)


def w_v2(wl: np.ndarray, seed: int) -> np.ndarray:
    L = wl.mean(1)
    full = w_vcea(L)
    rng = np.random.default_rng(seed)
    n = wl.shape[1]
    dist = [0.5 * np.abs(w_vcea(wl[:, boot_indices(n, rng)].mean(1)) - full).sum() for _ in range(B_BOOT)]
    s = float(np.clip(1.0 - np.mean(dist), 0.0, 1.0))
    return s * full + (1 - s) * w_bg(L)


def w_v3(wl: np.ndarray) -> np.ndarray:
    L = wl.mean(1)
    m_star = 1 + int(np.argmin(L[1:]))
    g = 0.0
    if L[m_star] < L[0]:
        d = wl[0] - wl[m_star]
        n = len(d)
        n_eff = n / block_len(n)
        se = d.std(ddof=1) / np.sqrt(n_eff) + 1e-12
        g = float(np.clip((d.mean() / se) / 2.0, 0.0, 1.0))
    return rc.inv_weights(L, 1.0 + 3.0 * g, 0.1 * g)


def w_v1(L: np.ndarray, h: int, tau_h: list[float]) -> np.ndarray:
    return rc.inv_weights(L, tau_h[h], 0.1)


def variant_weights(name: str, wl: np.ndarray, h: int, params: dict, seed: int) -> np.ndarray:
    if name == "V1_HorizonTau":
        return w_v1(wl.mean(1), h, params["tau_h"])
    if name == "V2_BootBlend":
        return w_v2(wl, seed)
    if name == "V3_EvidenceGate":
        return w_v3(wl)
    raise KeyError(name)


def subset(c: dict, idx: np.ndarray) -> dict:
    return {"y": c["y"][idx], "X": c["X"][:, idx], "valid": c["valid"]}


def halves(n: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(4, min(n - 4, int(round(n * 0.5))))
    return np.arange(k), np.arange(k, n)


# ------------------------------------------------------------------------ dev
def load_main6_val() -> dict:
    out = {}
    for t in rc.MAIN6:
        for s in rc.SEEDS:
            p1 = rc.MAIN6_CACHE / "M1" / t / f"seed_{s}_val.npz"
            p2 = rc.MAIN6_CACHE / "C3_opt" / t / f"seed_{s}_val.npz"
            assert p1.name.endswith("_val.npz") and p2.name.endswith("_val.npz")
            out[(t, s)] = rc._pair(p1, p2)
    return out


def choose_tau_h(pairs: list[tuple[dict, dict]]) -> list[float]:
    """pairs of (estimation cache, scoring cache); pooled over tiles x seeds."""
    tau_h = []
    for h in range(4):
        scores = []
        for tau in TAU_GRID:
            sc = [rc.skill(np.tensordot(rc.inv_weights(window_losses(e, h).mean(1), tau, 0.1), s["X"][:, :, h], axes=1), s, h)
                  for e, s in pairs]
            scores.append(np.mean(sc))
        tau_h.append(TAU_GRID[int(np.argmax(scores))])
    return tau_h


def stage_dev() -> None:
    if FREEZE.exists():
        raise SystemExit(f"{FREEZE} exists: the variant is already frozen. Dev stage cannot be rerun.")
    PREREG.mkdir(parents=True, exist_ok=True)
    data = load_main6_val()
    splits = {}
    for k, c in data.items():
        pre, suf = halves(c["y"].shape[0])
        splits[k] = (subset(c, pre), subset(c, suf))
    # V1: inner split of the prefix only
    inner = []
    for pre, _ in splits.values():
        a, b = halves(pre["y"].shape[0])
        inner.append((subset(pre, a), subset(pre, b)))
    tau_inner = choose_tau_h(inner)

    rows = []
    for (t, s), (pre, suf) in splits.items():
        for h, hn in enumerate(rc.HORIZONS):
            wl = window_losses(pre, h)
            L = wl.mean(1)
            seed = 1000 * s + h
            W = {"VCEA": w_vcea(L), "BatesGranger": w_bg(L),
                 "V1_HorizonTau": w_v1(L, h, tau_inner),
                 "V2_BootBlend": w_v2(wl, seed), "V3_EvidenceGate": w_v3(wl)}
            for m, w in W.items():
                rows.append({"Tile": t, "Seed": s, "Horizon": hn, "Method": m,
                             "Skill_suffix": rc.skill(np.tensordot(w, suf["X"][:, :, h], axes=1), suf, h)})
    df = pd.DataFrame(rows)
    df.to_csv(PREREG / "dev_nested_scores.csv", index=False)
    mean = df.groupby("Method").Skill_suffix.mean()
    by_h = df.groupby(["Method", "Horizon"]).Skill_suffix.mean().unstack()[rc.HORIZONS]
    variants = ["V1_HorizonTau", "V2_BootBlend", "V3_EvidenceGate"]
    best = mean[variants].idxmax()
    carried = bool(mean[best] > mean["VCEA"] and mean[best] > mean["BatesGranger"])

    params = {}
    if carried and best == "V1_HorizonTau":  # final tau_h from full prefix -> suffix, like the original tau selection
        params["tau_h"] = choose_tau_h(list(splits.values()))
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": best if carried else None,
        "carried_forward": carried,
        "params": params,
        "tau_h_inner_dev": tau_inner,
        "dev_mean_nested_suffix_skill": mean.round(6).to_dict(),
        "dev_by_horizon": by_h.round(6).to_dict(orient="index"),
        "selection_rule": "best of V1-V3 by mean nested-suffix Skill on main-six validation; carried only if > VCEA and > BatesGranger",
        "data_used": "main-six validation windows only (act_v7 seed_expert_cache *_val.npz)",
        "code_sha256": code_hash(),
    }
    FREEZE.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


# --------------------------------------------------------------- fresh tiles
def tile_xy(tid: str) -> tuple[int, int]:
    e, n = tid[1:].split("N")
    return int(e), int(n)


def stage_select_fresh(n_per_cell: int = 2, seed: int = 20261104) -> None:
    if FRESH_TILES.exists():
        raise SystemExit(f"{FRESH_TILES} exists: fresh tiles already pre-registered.")
    PREREG.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(CATALOG)
    used = [tile_xy(t) for t in USED_TILES]
    def near_used(tid):
        x, y = tile_xy(tid)
        return any(abs(x - ux) <= 1 and abs(y - uy) <= 1 for ux, uy in used)
    c = cat[(cat.filesize_mb >= 5) & (cat.filesize_mb <= 270)
            & cat.centroid_lon.between(-11, 35) & cat.centroid_lat.between(34, 72)].copy()
    c = c[~c.tile_id.apply(near_used)].copy()
    c["lat_band"] = pd.cut(c.centroid_lat, [34, 42, 48, 55, 72], labels=["<42", "42-48", "48-55", ">=55"], include_lowest=True)
    c["lon_band"] = pd.cut(c.centroid_lon, [-11, 5, 15, 35], labels=["<5", "5-15", ">=15"], include_lowest=True)
    rng = np.random.default_rng(seed)
    chosen = []
    for (la, lo), g in c.groupby(["lat_band", "lon_band"], observed=True):
        med = g.filesize_mb.median()
        for part, label in ((g[g.filesize_mb < med], "below-cell-median size"), (g[g.filesize_mb >= med], "above-cell-median size")):
            part = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            for _, r in part.iterrows():
                x, y = tile_xy(r.tile_id)
                if all(abs(x - cx) >= 2 or abs(y - cy) >= 2 for cx, cy in [tile_xy(t["tile_id"]) for t in chosen]):
                    chosen.append({**r[["tile_id", "centroid_lat", "centroid_lon", "filesize_mb", "filename"]].to_dict(),
                                   "lat_band": la, "lon_band": lo, "size_stratum": label})
                    break
    out = pd.DataFrame(chosen).sort_values("tile_id").reset_index(drop=True)
    out.to_csv(FRESH_TILES, index=False)
    digest = hashlib.sha256(FRESH_TILES.read_bytes()).hexdigest()
    (PREREG / "fresh_tiles.sha256").write_text(digest + "\n")
    print(out.to_string(index=False))
    print("sha256", digest, "n =", len(out))


# ------------------------------------------------------- EGMS API plumbing
API = "https://egms.land.copernicus.eu/insar-api/archive"
WANT_YEARS = ("2019", "2023")


def _api_get(session, what: str):
    r = session.get(f"{API}/{what}", timeout=120)
    r.raise_for_status()
    return r.json()


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _resolve_release(session) -> str:
    """Return the API identifier of the 2019-2023 release; stop if it is not served."""
    payload = _api_get(session, "releases")
    cands = sorted({s for s in _strings(payload) if all(y in s for y in WANT_YEARS)})
    if len(cands) != 1:
        raise SystemExit(f"2019-2023 release not uniquely available via the API. /releases returned: {payload}")
    return cands[0]


def _download(session, tile_id: str, expected_filename: str, release: str, raw_dir: Path) -> Path:
    dest = raw_dir / expected_filename
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    query = {"id": None, "tileId": tile_id, "releases": [release], "levels": ["L3"], "productType": "ORTHO-UP"}
    r = session.post(f"{API}/search", data=json.dumps(query), timeout=120)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("status") or not payload.get("hits"):
        raise SystemExit(f"EGMS search failed for {tile_id} with release {release!r}: {payload}")
    hit = payload["hits"][0]
    if hit["filename"] != expected_filename:
        raise SystemExit(f"{tile_id}: API returned {hit['filename']!r}, pre-registered file is {expected_filename!r}. Stop and review.")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(f"{API}/download/{hit['filename']}?id={payload['id']}", stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    if tmp.stat().st_size <= 1024:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded file looks invalid for {tile_id}")
    tmp.replace(dest)
    return dest


def _session():
    sys.path.insert(0, str(rc.PROJECT_ROOT))
    from tgrs_experiments import vcea_e7_pan_europe_audit_strong as e7
    return e7.build_session(e7.load_service_key())


def stage_check_api() -> None:
    session = _session()
    for what in ("releases", "levels", "product_types"):
        print(f"== {what}:", json.dumps(_api_get(session, what))[:2000])
    try:
        print("resolved 2019-2023 release id:", _resolve_release(session))
    except SystemExit as e:
        print(e)


AMENDMENT = PREREG / "amendment_01_release.md"


def _write_amendment(release: str) -> None:
    if AMENDMENT.exists():
        if f"Release used for fresh tiles: {release}" not in AMENDMENT.read_text(encoding="utf-8"):
            raise SystemExit(f"{AMENDMENT.name} already fixes a different release; stop.")
        return
    if any(FRESH_CACHE.glob("*/*.npz")) or FRESH_RES.exists():
        raise SystemExit("Fresh data already exist; an amendment can no longer be made blind.")
    raw_new = [p.name for p in (rc.PROJECT_ROOT / "data" / "raw").glob("*.zip") if "2019_2023" not in p.name]
    text = f"""# Amendment 01: EGMS release for the fresh confirmatory tiles

Written {datetime.now(timezone.utc).isoformat()}, BEFORE any fresh-tile data were downloaded or inspected
(fresh cache empty; non-2019-2023 zips in data/raw at this time: {raw_new or 'none'}).

## Reason
The EGMS archive API no longer serves the 2019-2023 release. `/releases` returned only
["2020-2024", "2020-2025"] on 2026-11-05 (the 2019-2023 search failed with "Invalid release").

## Change
- Release used for fresh tiles: {release}
- Chosen because it is the official annual five-year release closest to the 2019-2023 product used for development and
  for the E7 audit (four overlapping years, same span length). 2020-2025 was not used.
- Only the data release (time span and EGMS reprocessing) changes. Tile IDs (fresh_tiles.csv, hash unchanged), expert
  configs, split rule, preprocessing, metrics, frozen variant (V3_EvidenceGate, code hash unchanged), hypotheses,
  statistics and decision rules are unchanged.

## Consequence for interpretation
All hypotheses are paired within-tile comparisons with hyper-parameters frozen on 2019-2023 main-six validation data, so
the test remains valid; it additionally tests transfer to a newer EGMS release. Absolute Skill values are not directly
comparable with the 2019-2023 E7 audit.
"""
    AMENDMENT.write_text(text, encoding="utf-8")
    (PREREG / "amendment_01_release.sha256").write_text(hashlib.sha256(AMENDMENT.read_bytes()).hexdigest() + "\n")
    print(f"[amendment] written {AMENDMENT}", flush=True)


def _download_release(session, tile_id: str, release: str, raw_dir: Path) -> Path:
    import re
    y1, y2 = release.split("-")
    pat = re.compile(rf"EGMS_L3_{tile_id}_100km_U_{y1}_{y2}_\d+\.zip")
    have = [p for p in raw_dir.glob(f"EGMS_L3_{tile_id}_100km_U_{y1}_{y2}_*.zip") if p.stat().st_size > 1024]
    if have:
        return have[0]
    query = {"id": None, "tileId": tile_id, "releases": [release], "levels": ["L3"], "productType": "ORTHO-UP"}
    r = session.post(f"{API}/search", data=json.dumps(query), timeout=120)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("status") or not payload.get("hits"):
        raise SystemExit(f"EGMS search failed for {tile_id} with release {release!r}: {payload}")
    name = payload["hits"][0]["filename"]
    if not pat.fullmatch(name):
        raise SystemExit(f"{tile_id}: unexpected filename {name!r} for release {release}; stop and review.")
    dest = raw_dir / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(f"{API}/download/{name}?id={payload['id']}", stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    if tmp.stat().st_size <= 1024:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded file looks invalid for {tile_id}")
    tmp.replace(dest)
    return dest


def _extract_release(zip_path: Path, tile_id: str, out_root: Path) -> dict:
    import zipfile
    d = out_root / tile_id
    d.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = d / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        dst.write(buf)
    return {"zip": zip_path, "csv": next(d.glob("*.csv")), "xml": next(d.glob("*.xml")), "dir": d}


# ------------------------------------------------------------- fresh-run (GPU)
def stage_fresh_run(lowmem: bool, skip_download: bool, release: str | None = None) -> None:
    tiles = pd.read_csv(FRESH_TILES)
    if release:
        _write_amendment(release)
    sys.path.insert(0, str(rc.PROJECT_ROOT))
    if lowmem:
        rc._enable_lowmem_csv()
    from tgrs_experiments.act_v2 import train_eval_single_tile
    from tgrs_experiments.act_v4c_c3_candidate import c3_runtime_optimized_config, m1_scalar_config
    from tgrs_experiments.act_v4c_efficiency import train_eval_rescue_single_tile
    from tgrs_experiments.act_v5_selector import infer_c3_split, infer_m1_split
    import tgrs_experiments.dscf_confirmation as dc
    from tgrs_experiments.pipeline import EXTRACTED_DIR, RAW_DIR, extract_raw_archives
    if lowmem:
        dc.fit_descriptors = rc._chunked_fit_descriptors(dc.fit_descriptors)
    if release:
        y1, y2 = release.split("-")
        if not skip_download:
            session = _session()
            served = _strings(_api_get(session, "releases"))
            if release not in served:
                raise SystemExit(f"Release {release!r} not served by the API: {served}")
        raw_files, manifest = {}, []
        out_root = EXTRACTED_DIR.parent / f"extracted_{y1}_{y2}"
        for _, r in tiles.iterrows():
            if skip_download:
                zp = next(RAW_DIR.glob(f"EGMS_L3_{r.tile_id}_100km_U_{y1}_{y2}_*.zip"))
            else:
                print(f"[download] {r.tile_id} ({release})", flush=True)
                zp = _download_release(session, r.tile_id, release, RAW_DIR)
            raw_files[r.tile_id] = _extract_release(zp, r.tile_id, out_root)
            manifest.append({"tile_id": r.tile_id, "release": release, "zip": zp.name,
                             "zip_sha256": hashlib.sha256(zp.read_bytes()).hexdigest()})
        FRESH_CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest).to_csv(FRESH_CACHE.parent / "fresh_download_manifest.csv", index=False)
    else:
        if not skip_download:
            session = _session()
            rel = _resolve_release(session)
            print(f"[download] using release id {rel!r}", flush=True)
            for _, r in tiles.iterrows():
                print(f"[download] {r.tile_id}", flush=True)
                _download(session, r.tile_id, r.filename, rel, RAW_DIR)
        raw_files = extract_raw_archives()
    phase = "fresh_confirmatory_audit"
    for _, r in tiles.iterrows():
        t = r.tile_id
        out = FRESH_CACHE / t
        want = [out / f"{fam}_seed_{s}_{sp}.npz" for fam in ("M1", "C3") for s in rc.SEEDS for sp in ("val", "test")]
        if all(p.exists() for p in want):
            print(f"[fresh] {t}: cached, skip", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        bundle = dc.build_bundle_from_tile_id(t, raw_files)
        for s in rc.SEEDS:
            m1_dir = FRESH_ROOT / "runs" / "M1" / t / f"seed_{s}"
            c3_dir = FRESH_ROOT / "runs" / "C3_opt" / t / f"seed_{s}"
            train_eval_single_tile(bundle, replace(m1_scalar_config(), seed=s), m1_dir, phase, resolution=64, reuse_if_exists=True)
            train_eval_rescue_single_tile(bundle, replace(c3_runtime_optimized_config(), seed=s), c3_dir, phase, reuse_if_exists=True)
            for sp in ("val", "test"):
                m1 = infer_m1_split(bundle, m1_dir, sp)
                c3 = infer_c3_split(bundle, c3_dir, sp)
                np.savez_compressed(out / f"M1_seed_{s}_{sp}.npz", **{k: m1[k] for k in rc.KEEP})
                np.savez_compressed(out / f"C3_seed_{s}_{sp}.npz", **{k: c3[k] for k in rc.KEEP})
        del bundle
        print(f"[fresh] {t}: done in {time.time() - t0:.1f}s", flush=True)


# ------------------------------------------------------------------------ eval
def stage_eval(force: bool) -> None:
    from scipy.stats import wilcoxon
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["code_sha256"] != code_hash():
        raise SystemExit("Code changed after freezing; the confirmatory test is no longer pre-registered.")
    digest = hashlib.sha256(FRESH_TILES.read_bytes()).hexdigest()
    if digest != (PREREG / "fresh_tiles.sha256").read_text().strip():
        raise SystemExit("fresh_tiles.csv changed after pre-registration.")
    FRESH_RES.mkdir(parents=True, exist_ok=True)
    lock = FRESH_RES / "EVAL_DONE"
    if lock.exists() and not force:
        raise SystemExit(f"Confirmatory eval already run ({lock.read_text().strip()}). Use --force only to re-render; it is logged.")
    tiles = pd.read_csv(FRESH_TILES).tile_id.tolist()
    missing = [t for t in tiles if not (FRESH_CACHE / t / "C3_seed_44_test.npz").exists()]
    if missing:
        raise SystemExit(f"Missing fresh caches: {missing}")
    var = freeze["selected_variant"]
    rows = []
    for t in tiles:
        for s in rc.SEEDS:
            cv = rc._pair(FRESH_CACHE / t / f"M1_seed_{s}_val.npz", FRESH_CACHE / t / f"C3_seed_{s}_val.npz")
            ct = rc._pair(FRESH_CACHE / t / f"M1_seed_{s}_test.npz", FRESH_CACHE / t / f"C3_seed_{s}_test.npz")
            for h, hn in enumerate(rc.HORIZONS):
                wl = window_losses(cv, h)
                L = wl.mean(1)
                W = {"Persistence": np.eye(4)[0], "Linear-Seasonal": np.eye(4)[1], "M1": np.eye(4)[2], "C3_opt": np.eye(4)[3],
                     "VCEA": w_vcea(L), "BatesGranger": w_bg(L), "NNLS_simplex": rc.simplex_ls(cv, h),
                     "Uniform4": np.full(4, 0.25), "SelectBestVal": np.eye(4)[int(np.argmin(L))],
                     "VCEA_noLS": rc.subset_inv(L, [0, 2, 3], 4.0, 0.1), "VCEA_noNeural": rc.subset_inv(L, [0, 1], 4.0, 0.1)}
                if var:
                    W[var] = variant_weights(var, wl, h, freeze["params"], 1000 * s + h)
                for m, w in W.items():
                    rows.append({"Tile": t, "Seed": s, "Horizon": hn, "Model": m,
                                 "Skill_MSE": rc.skill(np.tensordot(w, ct["X"][:, :, h], axes=1), ct, h)})
    df = pd.DataFrame(rows)
    df.to_csv(FRESH_RES / "skill_by_tile_seed_horizon.csv", index=False)
    tile = df.groupby(["Model", "Tile"]).Skill_MSE.mean().unstack(0)
    tile.to_csv(FRESH_RES / "tile_means.csv")

    def test(a, b, alt="greater", shift=0.0):
        d = tile[a] - tile[b] + shift
        p = float(wilcoxon(d, alternative=alt).pvalue) if np.any(d != 0) else 1.0
        return {"contrast": f"{a} - {b}" + (f" + {shift}" if shift else ""), "mean_delta": float((tile[a] - tile[b]).mean()),
                "wins": int(((tile[a] - tile[b]) > 0).sum()), "n": len(d), "p_one_sided": p}
    primary = [test("BatesGranger", "SelectBestVal"), test("BatesGranger", "Uniform4"), test("BatesGranger", "M1")]
    if var:
        primary.append(test(var, "BatesGranger"))
    ps = np.array([r["p_one_sided"] for r in primary])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        holm[i] = min(1.0, running)
    for r, hp in zip(primary, holm):
        r["p_holm"] = float(hp)
    secondary = [test("VCEA", "BatesGranger", shift=0.01), test("VCEA", "VCEA_noLS"), test("VCEA", "VCEA_noNeural"),
                 test("VCEA", "NNLS_simplex", alt="two-sided")]
    verdict = {}
    if var:
        r = primary[-1]
        verdict["method_claim_supported"] = bool(r["mean_delta"] >= 0.02 and r["p_holm"] < 0.05)
    verdict["soft_combination_claim_supported"] = bool(all(r["p_holm"] < 0.05 and r["mean_delta"] > 0 for r in primary[:3]))
    verdict["vcea_noninferior_to_bg_margin_0.01"] = bool(secondary[0]["p_one_sided"] < 0.05)
    report = {"run_at_utc": datetime.now(timezone.utc).isoformat(), "forced_rerun": bool(lock.exists()),
              "variant": var, "primary": primary, "secondary": secondary, "verdict": verdict,
              "mean_skill": tile.mean().sort_values(ascending=False).round(4).to_dict()}
    (FRESH_RES / "confirmatory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lock.write_text(report["run_at_utc"] + ("  (forced)" if report["forced_rerun"] else "") + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["dev", "select-fresh", "check-api", "fresh-run", "eval"])
    ap.add_argument("--lowmem", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--release", default=None, help="EGMS release for fresh tiles, e.g. 2020-2024 (writes amendment_01)")
    a = ap.parse_args()
    {"dev": stage_dev, "select-fresh": stage_select_fresh, "check-api": stage_check_api,
     "fresh-run": lambda: stage_fresh_run(a.lowmem, a.skip_download, a.release),
     "eval": lambda: stage_eval(a.force)}[a.stage]()
