# Final Hardening Revision Report

## 1. Files modified

- `bare_jrnl_new_sample4.tex`
- `supplement/ai_disclosure_check.md`
- `supplement/claim_to_evidence_map.md`
- `supplement/reproducibility/figure_source_map.csv`
- `supplement/assets/manifests/manifest_assets.json`

## 2. Main-text compression summary

- The main narrative was compressed without changing numerical results or removing core evidence.
- Approximate compression:
  - `Abstract`: shortened and de-duplicated.
  - `Introduction`: contribution bullets tightened and made less repetitive.
  - `Methods`: repeated fallback / threshold / calibration wording reduced after the protocol-hardening pass.
  - `Results`: repeated audit caveats and deployment-style phrasing shortened; section headings cooled.
  - `Discussion` / `Conclusion`: repeated limitations and geographic caveats compressed.
- Overall, this pass removed or shortened roughly `25--35` prose lines and several long captions while preserving all core result blocks.

## 3. Terminology cooling summary

### Replacements performed

- `Stress-Safe` in the manuscript title was removed in favor of:
  - `Conservative Fallback Control`
- prose `stress-safe` / `safety-oriented` / `safe aggregation` were cooled to:
  - `conservative fallback`
  - `conservative operating variant`
  - `fallback-controlled aggregation`
  - `operational control`
- `deployment` wording was replaced with:
  - `operating`
  - `operational use`
  - `operational setting`
- `pan-European` wording in the active manuscript was replaced with:
  - `geographically stratified 14-tile European frozen audit`
  - `14-tile European frozen audit`
- `positive transfer` in the external-audit interpretation was replaced with:
  - `positive future-evaluation performance`

### Formal names retained

- `VCEA`
- `SS-VCEA`
- `SS-VCEA q97.5`

These formal names were retained because they are part of the established method vocabulary.

## 4. Baseline / expert transparency changes

- The early appendix expert-pool table was expanded into a reviewer-facing summary of evaluated experts and baselines.
- Updated location:
  - `Appendix A`, Table `XII`
- New rows now cover:
  - Persistence
  - Linear-Seasonal
  - M1 / C2
  - C3_opt
  - COEA
  - VCEA
  - SS-VCEA q97.5
- Added a clarifying paragraph in Methods:
  - VCEA and SS-VCEA are not new backbone architectures;
  - they operate over a fixed pool of persistence, kinematic, online, and neural experts.

No new baseline was added because the final-hardening pass avoids new experiments.

## 5. Appendix / supplementary-material organization changes

- The appendix now explicitly states that it is supplementary material within the same PDF.
- Opening note added at the start of `Appendix A`.
- Section naming was cooled and organized more clearly:
  - `Appendix C` renamed to `Conservative Operating-Point Reference and Resolution Sensitivity`
- Appendix content order remains compatible with the existing manuscript structure:
  1. protocol / metric details
  2. blocked-validation / VTSA diagnostics
  3. matched operating-point reference + resolution sensitivity
  4. architecture-search / expert-lineage audit
  5. reproducibility / claim-to-evidence traceability

## 6. Skill notation cleanup status

- Active manuscript prose, captions, and table headers now use:
  - `Skill$_{\mathrm{MSE}}$`
  - `\mathrm{Skill}_{\mathrm{MSE}}`
- No active `SkillMSE`, `Skill_MSE`, or `Skill-MSE` strings remain in the compiled manuscript source.
- Frozen archival CSVs and helper scripts still retain native source headers such as `Skill_MSE`; these were intentionally not renamed to preserve compatibility with archived result files and scripts.

## 7. Figure / caption issues remaining

- The compiled manuscript captions were cooled and synchronized with the revised terminology.
- No figure regeneration was required in this pass.
- Remaining non-blocking issues:
  - some legacy, non-submission asset sources in `paper_assets_vcea_extras/`, `figures/source_bundles/`, and `*_fig04check.*` still contain older wording such as `pan-European`, `Skill_MSE`, or older draft language;
  - these files are not the active manuscript source used by the compiled PDF.

## 8. AI disclosure check

- Updated file:
  - `supplement/ai_disclosure_check.md`

The file now records:
- possible AI-assisted figure/layout work;
- possible AI-assisted text polishing;
- possible AI-assisted utility-code drafting;
- the IEEE disclosure reminder;
- the recommendation to confirm with all authors/advisor before submission.

## 9. Reproducibility / path hygiene status

- Submission-facing supplement files were re-checked for absolute local paths.
- No absolute local paths were found in `supplement/` after this pass.
- Raw EGMS redistribution remains correctly bounded in the supplement README.
- A few non-submission internal files elsewhere in the repository still use older wording or archived path conventions; these were left untouched because they are not part of the paper-facing supplement package.

## 10. LaTeX compilation status

- `pdflatex` compilation succeeded.
- Updated PDF:
  - `bare_jrnl_new_sample4.pdf`
- Current page count:
  - total PDF pages: `31`
  - references begin on page `14`
  - appendix begins on page `16`
- This means the main narrative plus references occupies pages `1--15`, while the appendix/supplementary material occupies pages `16--31`.

### Non-blocking warnings that remain

- `Unused global option(s): [lettersize]`
- several `Underfull \hbox` warnings from dense tables, paths, and bibliography URLs
- a few `Overfull \vbox` warnings on float-heavy appendix pages
- `Text page 7 contains only floats`

These warnings are layout-related and did not block compilation.

## 11. Remaining TODOs before submission

- Human-skim Appendix Table `XII` to confirm the expanded baseline/expert table remains readable.
- Decide whether to keep the `lettersize` documentclass option or remove it for a cleaner build log.
- Decide whether to create a paper-facing alias layer for archived `Skill_MSE` CSV headers if the full repository, not just the supplement, will be shared.
- If the broader repo will be released, optionally clean legacy non-submission asset sources that still contain older wording such as `pan-European` or draft-only terminology.
- Confirm with all authors/advisor whether IEEE AI disclosure is required.

## 12. Human review checklist

- Check the revised title and confirm it matches the intended submission framing.
- Confirm that the main message still reads as:
  - validation-calibrated expert aggregation under strong persistence baselines;
  - conservative fallback control as a secondary operating variant.
- Skim Abstract, Methods, and Discussion for tone consistency after the cooling pass.
- Verify that the baseline / expert summary table helps rather than slows the reading flow.
- Confirm that appendix-as-supplement wording is acceptable for the target submission package.
- Recheck final PDF figures on pages 7--13 and appendix pages 16--20 for readability after the float reflow.
