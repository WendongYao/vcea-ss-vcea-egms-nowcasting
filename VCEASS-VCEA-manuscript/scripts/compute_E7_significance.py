from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover
    wilcoxon = None


BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260528
MODELS = ("M1", "VCEA")


def two_sided_sign_test_pvalue(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    probs = [math.comb(n, i) * (0.5 ** n) for i in range(n + 1)]
    observed = probs[k]
    p = sum(prob for prob in probs if prob <= observed + 1e-15)
    return float(min(1.0, p))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_ITERS, len(values)))
    sampled = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(sampled, 2.5)),
        float(np.percentile(sampled, 97.5)),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_csv = root / "paper_assets_vcea_extras" / "tables_csv_original" / "E7_strong" / "table_E7_pan_europe_locked_results_by_tile.csv"
    out_table = root / "tables" / "table_E7_statistical_significance.csv"
    out_horizon = root / "tables" / "table_E7_horizon_gain_ci.csv"
    out_diag = root / "diagnostics" / "E7_significance_audit.json"

    df = pd.read_csv(source_csv)
    part = df[df["Model"].isin(MODELS)].copy()
    part["Skill_MSE mean"] = part["Skill_MSE mean"].astype(float)

    pivot = (
        part.pivot_table(
            index=["Tile", "Horizon"],
            columns="Model",
            values="Skill_MSE mean",
            aggfunc="mean",
        )
        .reset_index()
        .dropna(subset=list(MODELS))
    )
    pivot["gain_vs_m1"] = pivot["VCEA"] - pivot["M1"]

    tile_summary = (
        pivot.groupby("Tile", as_index=False)
        .agg(
            tile_gain_mean=("gain_vs_m1", "mean"),
            horizon_count=("gain_vs_m1", "size"),
            vcea_skill_mean=("VCEA", "mean"),
            m1_skill_mean=("M1", "mean"),
        )
        .sort_values("Tile")
    )

    gains = tile_summary["tile_gain_mean"].to_numpy(dtype=float)
    wins = int((gains > 0).sum())
    losses = int((gains < 0).sum())
    ties = int((gains == 0).sum())
    n_tiles = int(len(gains))
    sign_n = n_tiles - ties
    sign_p = two_sided_sign_test_pvalue(wins, sign_n)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_gain, ci_low, ci_high = bootstrap_mean_ci(gains, rng)

    wilcoxon_stat = None
    wilcoxon_p = None
    if wilcoxon is not None and sign_n > 0:
        result = wilcoxon(gains, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_stat = float(result.statistic)
        wilcoxon_p = float(result.pvalue)

    summary_df = pd.DataFrame(
        [
            {
                "comparison": "VCEA_vs_M1_tile_mean_gain",
                "n_tiles": n_tiles,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_n": sign_n,
                "sign_test_pvalue_two_sided": sign_p,
                "mean_gain": mean_gain,
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "bootstrap_iterations": BOOTSTRAP_ITERS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_pvalue_two_sided": wilcoxon_p,
                "source_csv": str(source_csv.relative_to(root)).replace("\\", "/"),
            }
        ]
    )
    summary_df.to_csv(out_table, index=False)

    horizon_rows: list[dict[str, object]] = []
    for horizon, hdf in pivot.groupby("Horizon", sort=False):
        h_gains = hdf["gain_vs_m1"].to_numpy(dtype=float)
        h_rng = np.random.default_rng(BOOTSTRAP_SEED + len(horizon_rows) + 1)
        h_mean, h_low, h_high = bootstrap_mean_ci(h_gains, h_rng)
        horizon_rows.append(
            {
                "horizon": horizon,
                "n_tiles": int(len(h_gains)),
                "wins": int((h_gains > 0).sum()),
                "losses": int((h_gains < 0).sum()),
                "ties": int((h_gains == 0).sum()),
                "mean_gain": h_mean,
                "bootstrap_ci95_low": h_low,
                "bootstrap_ci95_high": h_high,
                "bootstrap_iterations": BOOTSTRAP_ITERS,
                "bootstrap_seed": BOOTSTRAP_SEED + len(horizon_rows) + 1,
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(out_horizon, index=False)

    diag = {
        "source_csv": str(source_csv.relative_to(root)).replace("\\", "/"),
        "comparison": "VCEA vs M1",
        "tile_level": {
            "n_tiles": n_tiles,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_n": sign_n,
            "sign_test_pvalue_two_sided": sign_p,
            "mean_gain": mean_gain,
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "bootstrap_iterations": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "wilcoxon_statistic": wilcoxon_stat,
            "wilcoxon_pvalue_two_sided": wilcoxon_p,
        },
        "per_horizon": horizon_rows,
        "tile_summary": tile_summary.to_dict(orient="records"),
        "method_note": "Descriptive robustness audit over held-out tiles; not a population-level inference over all European deformation conditions.",
    }
    out_diag.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(json.dumps({"summary_csv": str(out_table), "horizon_csv": str(out_horizon), "diagnostics_json": str(out_diag)}, indent=2))


if __name__ == "__main__":
    main()
