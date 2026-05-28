# Hardening Paper Insert

## VTSA
We explain the strength of VCEA through validation-to-test stability analysis rather than through learned expert selection. Across the six EGMS tiles, validation expert ranks remain strongly aligned with test ranks, while delayed-feedback online updates exhibit substantial weight drift. This supports static validation-calibrated aggregation as a more reliable mechanism than online adaptation under strong persistence baselines.

## SS-VCEA
To address the stress-tile caveat of VCEA, we introduce SS-VCEA, a validation-calibrated fallback wrapper that reverts to a safer predictor only when a normalized risk score exceeds a threshold calibrated on validation data. This design does not learn a selector and performs no test-time threshold tuning.

## Resolution-aware robustness
Resolution sensitivity shows that higher resolution does not uniformly benefit single backbones, but aggregation remains robust and can improve with higher-resolution expert diversity. We therefore present 64x64 as the main regional nowcasting setting, while retaining 128/512 only as robustness evidence with explicit transfer/calibration caveats.

## Reproducibility
The final workflow is backed by infra_v2 with frozen manifests, chronological split registries, expert-cache traceability, aggregation calibration provenance, and exact/proxy/transfer flags for result-level lineage.
