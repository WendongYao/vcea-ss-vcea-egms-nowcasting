# Amendment 01: EGMS release for the fresh confirmatory tiles

Written 2026-11-05T16:16:59.661320+00:00, BEFORE any fresh-tile data were downloaded or inspected
(fresh cache empty; non-2019-2023 zips in data/raw at this time: none).

## Reason
The EGMS archive API no longer serves the 2019-2023 release. `/releases` returned only
["2020-2024", "2020-2025"] on 2026-11-05 (the 2019-2023 search failed with "Invalid release").

## Change
- Release used for fresh tiles: 2020-2024
- Chosen because it is the official annual five-year release closest to the 2019-2023 product used for development and
  for the E7 audit (four overlapping years, same span length). 2020-2025 was not used.
- Only the data release (time span and EGMS reprocessing) changes. Tile IDs (fresh_tiles.csv, hash unchanged), expert
  configs, split rule, preprocessing, metrics, frozen variant (V3_EvidenceGate, code hash unchanged), hypotheses,
  statistics and decision rules are unchanged.

## Consequence for interpretation
All hypotheses are paired within-tile comparisons with hyper-parameters frozen on 2019-2023 main-six validation data, so
the test remains valid; it additionally tests transfer to a newer EGMS release. Absolute Skill values are not directly
comparable with the 2019-2023 E7 audit.
