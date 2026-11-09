# VCEA comparator audit (R1), 2026-11-04

Question: does VCEA beat simple, validation-only forecast combiners on the frozen
14-tile E7 audit (and on the main-six development tiles)?

- Script: `run_vcea_comparators.py` (comparator list fixed in its header).
- Nothing outside this folder is modified. Expert checkpoints are only loaded for
  inference; no training or tuning.
- `cache/e7/`: re-inferred M1 / C3_opt val+test arrays for the 14 audit tiles.
  Re-inferred test predictions match `runs/*/predictions.npz`
  (max abs diff M1 3e-5 mm, C3_opt 0.013 mm; GPU AMP vs CPU).
- `results/summary.md`: headline tables; `results/*.csv`: full tables.
- Reproduction: VCEA 0.2601 / M1 0.1128 on E7 and 0.1776 / 0.0783 on main-six,
  identical to the paper tables.

Rerun (project env, GPU optional):

    python run_vcea_comparators.py all            # infer skips cached tiles
    python run_vcea_comparators.py infer --lowmem # only for machines with < 8 GB RAM
    python run_vcea_comparators.py compare

Caveat: the E7 test labels had already been seen when this analysis was run, so
these results are exploratory. A confirmatory test needs fresh tiles.

## Pre-registered variant test (added 2026-11-04)

Files in `prereg/` (do not edit; hashes recorded alongside):
- `preregistration.md`: hypotheses, decision rules and disclosure.
- `fresh_tiles.csv`: 24 fresh confirmatory tiles chosen from catalog metadata only.
- `frozen_variant.json`: dev result on main-six validation only. Selected and frozen: **V3_EvidenceGate**.
  `variant_protocol.py eval` refuses to run if the method code changes.

Steps to run (GPU recommended; PowerShell syntax shown for the key variable):

    cd analysis_vcea_comparators_20261104
    $env:EGMS_SERVICE_KEY_PATH = "<path to your EGMS service key json>"
    python variant_protocol.py fresh-run     # download 24 tiles, train M1/C3_opt (3 seeds), cache val/test
    python variant_protocol.py eval          # runs ONCE; writes results/fresh/confirmatory_report.json

`fresh-run` can be restarted; finished tiles are skipped. Add `--lowmem` on a machine with < 8 GB RAM.
Do not open or inspect results/fresh before `eval` has written its lock file.
