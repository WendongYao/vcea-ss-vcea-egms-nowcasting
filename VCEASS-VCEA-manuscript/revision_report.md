# Revision Report

## 1. Files modified

- `bare_jrnl_new_sample4.tex`
- `supplement/README.md`
- `supplement/reproducibility_manifest.json`
- `supplement/assets/manifests/audit_E7_manifest.json` (path sanitization only)
- `supplement/assets/manifests/manifest_assets.json` (path sanitization only)
- `supplement/assets/results/table_E7_pan_europe_locked_results_by_tile.csv` (path sanitization only)
- additional copied supplement text assets sanitized to `${PROJECT_ROOT}` / `${MANUSCRIPT_ROOT}` placeholders where applicable

## 2. Files created

### Statistical audit outputs

- `scripts/compute_E7_significance.py`
- `tables/table_E7_statistical_significance.csv`
- `tables/table_E7_horizon_gain_ci.csv`
- `diagnostics/E7_significance_audit.json`

### Supplement documentation

- `supplement/selection_notes.md`
- `supplement/README.md`
- `supplement/environment.txt`
- `supplement/seeds.csv`
- `supplement/claim_to_evidence_map.md`
- `supplement/baseline_extension_todo.md`
- `supplement/ai_disclosure_check.md`
- `supplement/reproducibility_manifest.json`
- `supplement/reproducibility/claim_to_result_map.md`
- `supplement/reproducibility/paper_table_source_map.csv`
- `supplement/reproducibility/figure_source_map.csv`
- `supplement/reproducibility/result_lineage_table.csv`
- `supplement/reproducibility/exact_proxy_transfer_flags.csv`

## 3. Manuscript sections modified

- Abstract
- Introduction contribution framing
- Data / Evaluation Protocol
- Methods: VCEA calibration source
- Methods: SS-VCEA threshold source
- Methods: stress-ratio definition
- Results: original four-tile external audit
- Results: geographically stratified 14-tile European audit
- Discussion: external-to-European audit boundary
- Discussion: stress-ratio interpretation
- Conclusion
- Appendix: calibration-source table
- Appendix: statistical audit subsection and tables
- Appendix: claim-to-evidence map
- Appendix: reproducibility traceability notes

## 4. New statistical audit outputs

The new statistical audit is computed only from archived by-tile strong-audit results and does not rerun any model.

- Tile-level sign test over 14 held-out tiles: 14 wins, two-sided `p = 0.0001220703125`
- Paired tile-level bootstrap mean gain (VCEA - M1): `0.1473563070590386`
- Bootstrap 95% CI: `[0.08873811539872542, 0.21656354029640754]`
- Per-horizon gain summary and bootstrap intervals archived in `tables/table_E7_horizon_gain_ci.csv`

## 5. Protocol wording changes made

- Replaced wording that could imply calibration-free external transfer.
- Standardized the protocol to:
  - `tile-local historical validation calibration followed by frozen future evaluation`
- Clarified that held-out external and 14-tile audit tiles:
  - are held out for expert search, threshold selection, future-test tuning, and final configuration choice;
  - still use their own historical validation segments for aggregation-weight estimation.
- Clarified that SS-VCEA q97.5 thresholds:
  - are per-horizon;
  - come from pooled main-six validation risk distributions;
  - remain fixed for main test, original external audit, and the 14-tile European audit.
- Clarified that stress ratio is:
  - an operational over-correction / reliability indicator;
  - not a physical hazard metric.

## 6. Claims downgraded or clarified

- The 14-tile European audit is no longer described in wording that suggests calibration-free or zero-shot transfer.
- The original four-tile external evaluation is clarified as:
  - held-out future evaluation with tile-local historical validation calibration.
- The 14-tile European audit is clarified as:
  - a metadata-stratified held-out European future evaluation;
  - not exhaustive validation over all European deformation conditions.
- The statistical appendix is described as:
  - a descriptive robustness audit over held-out tiles;
  - not a population-level inference over all European deformation conditions.

## 7. Remaining unresolved issues

- The compiled PDF still contains non-blocking layout warnings:
  - `Unused global option(s): [lettersize]`
  - several `Underfull \\hbox` warnings
  - small `Overfull \\vbox` warnings on float-heavy pages
- Archived raw supplement result CSVs and the helper significance script retain native source column names such as `Skill_MSE`. The manuscript prose and captions have been normalized to `Skill$_{\\mathrm{MSE}}$`, but the frozen archival CSV headers were left unchanged to preserve source compatibility.
- `selection_tile_pool.csv` is a reconstructed candidate-pool register. The complete nonselected candidate pool was not archived with full regime/coverage annotations, and this caveat remains documented in `supplement/selection_notes.md`.

## 8. Compilation status

- `pdflatex` compilation succeeded.
- Updated PDF: `bare_jrnl_new_sample4.pdf`
- Final build status:
  - no unresolved reference warnings remained after the final pass;
  - no missing-figure errors were found;
  - no LaTeX control-sequence errors remained after fixing the appendix statistical-caption issue.

## 9. Suggested next human review checklist

- Verify that the revised calibration wording matches the intended protocol exactly in the Abstract, Methods, Results, and Conclusion.
- Skim the new calibration-source table for width/readability in the final PDF.
- Check the appendix statistical-audit tables for acceptable readability in two-column format.
- Decide whether to keep or remove the non-blocking `lettersize` option warning before submission.
- Confirm whether the archived supplement should preserve native CSV headers (`Skill_MSE`) or whether a paper-facing alias layer is desired.
- Confirm with the authors/advisor whether an IEEE AI disclosure statement is required using `supplement/ai_disclosure_check.md`.
- Run the final submission checklist and cover-letter pass after this revision.

## 10. Final QA summary

- Manuscript now avoids calibration-free / zero-shot wording in its active claims.
- External and 14-tile audit calibration wording now matches `vcea_protocol_audit_report.md`.
- Supplement paths have been sanitized away from local absolute machine paths.
- No numerical manuscript result was changed except for adding the lightweight statistical audit computed from existing CSV outputs.
