# paper_assets_vcea_extras

This folder contains LaTeX-ready assets for the VCEA extra experiments (E1--E6) and the Strong E7 pan-European frozen audit.

## Contents
- `vcea_extra_tables_and_figures.tex`: self-contained LaTeX file containing all table contents, captions, labels, and figure blocks.
- `figures/`: copied PNG figures grouped by experiment (`E1` ... `E7_strong`).
- `tables_csv_original/`: copied original CSV tables grouped into `E1_E6` and `E7_strong`.
- `tables_tex_fragments/`: per-table LaTeX fragments generated from the CSV sources.
- `manifest_assets.json`: asset manifest with source/target paths, hashes, sizes, and intended manuscript placement.
- `compile_check/test_compile.tex`: minimal syntax-check document that inputs the master TeX file.

## How to include from the main paper
In the parent manuscript directory, load the required packages in the preamble and then either:
1. `\input{paper_assets_vcea_extras/vcea_extra_tables_and_figures.tex}` directly, or
2. Copy selected table/figure environments from this file into the main manuscript.

## Recommended placement
- Main text: E1 leakage audit, E2 main-six expert ablation, E3 calibration ablation, E6 threshold sweep, Strong E7 compact group summary, Strong E7 failure cases, and the selected main-text figures.
- Appendix / supplement: full E4 blocked-validation table, full Strong E7 by-tile table, metadata/catalog/prefetch tables, threshold diagnostics, and the remaining figures.

## IEEEtran compile notes
- `figure*` and `table*` environments are used for main-text candidate assets.
- Very wide or long appendix tables use `longtable` and `pdflscape`.
- For the IEEEtran two-column final version, the full Strong E7 by-tile table and the full E4 blocked-validation table are intended for appendix/supplement use.
- Compile check status: No LaTeX engine (`latexmk`, `pdflatex`, `xelatex`, or `tectonic`) is available in PATH; syntax check file was generated but not compiled.

## Missing files
- None.
