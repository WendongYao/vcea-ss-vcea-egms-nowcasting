# Pre-registration: VCEA confirmatory audit on fresh EGMS tiles

Written 2026-11-04T12:58Z, before the dev stage scored any variant and before any fresh tile was downloaded.

## Disclosure
- Main-six test results and the 14-tile E7 test results were already known when this protocol was written.
- The variant hypothesis was motivated by a test-set observation (Bates-Granger beat VCEA at t+1 on main-six and E7).
  E7 and main-six test data are therefore NOT used as evidence for any variant.

## Fresh tiles
- 24 tiles in fresh_tiles.csv (sha256 in fresh_tiles.sha256), chosen by variant_protocol.py select-fresh from the
  EGMS API catalog metadata only: zip size 5-270 MB, centroid lon -11..35 and lat 34..72, not within one tile of any of the
  24 previously used tiles, at most one tile per 2-tile neighbourhood, 4 latitude x 3 longitude bands x 2 size strata,
  fixed RNG seed 20261104.
- Experts: M1 and C3_opt trained per tile with the frozen configs (seeds 42, 43, 44), as in E7. Split, preprocessing and
  metrics unchanged (target-time 70/10/20, 64x64, valid-mask Skill_MSE).

## Unit and statistics
- Unit = tile. Tile score = mean Skill_MSE over 4 horizons and 3 seeds.
- One-sided Wilcoxon signed-rank on tile-level differences; Holm correction across the primary family (alpha = 0.05).

## Primary hypotheses
- P1a Bates-Granger > SelectBestVal; P1b Bates-Granger > Uniform4; P1c Bates-Granger > M1
  (claim: validation-calibrated soft combination beats selection, equal weighting and the best neural expert).
- P2 (only if dev carries a variant): Variant - Bates-Granger >= 0.02 mean and Holm-significant -> method claim.

## Secondary
- VCEA non-inferior to Bates-Granger with margin 0.01; VCEA > VCEA_noLS; VCEA > VCEA_noNeural; VCEA vs NNLS (two-sided).

## Decision rules
- P2 met: present the frozen variant as the paper's method.
- P2 not met or no variant carried: present validation-calibrated soft combination as the finding, VCEA as a regularised
  Bates-Granger variant; no further variant iteration.
- P1a-c not all met: downgrade to a short benchmark note.
- The eval stage runs once (lock file). All results are reported.

## Variants
Defined in the docstring of variant_protocol.py (V1_HorizonTau, V2_BootBlend, V3_EvidenceGate) with the dev selection rule.
