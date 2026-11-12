# Reproducible analysis code

This directory contains the code used to generate the current revision's
tables and figures. The numbered scripts are ordered by workflow.

## Layout

- `FM_embedding_extraction/`: five flat extraction entry points plus
  `data_loader.py`. The private model-source snapshots are consolidated under
  one local-only `local_model_code/` directory and are excluded by
  `.gitignore`.
- `radiomics_baseline_extraction/`: two independent extraction scripts in one
  folder: PyRadiomics features and the mean-HU/spleen-volume baselines.
- The remaining Python files are flat analysis and figure/table-generation
  scripts. 

## Relative workflow order

The leading position is an internal workflow order beginning at 00; it is
independent of manuscript Figure/Table numbering. `A` denotes the analysis or
table-producing step and `B` denotes its paired plotting step when the two are
kept separate.

- `01A` `01A_unified_linear_probe.py` — unified frozen-representation linear probes.
- `02A` `02A_discrimination_and_signed_delong.py` — mean seed-level AUC and signed DeLong results.
- `02B` `02B_plot_pairwise_auc_comparisons.py` — pairwise AUC comparison plot.
- `03` `03_threshold_metrics_bootstrap.py` — threshold-derived metrics and their 1,000-resample patient-bootstrap CIs. The bootstrap seed is fixed in the script at `9715` and cannot be overridden from the command line.
- `04A` `04A_permutation_computation.py` — generates all 18 fixed-OOF permutation cells.
- `04B` `04B_permutation_summary_and_plot.py` — validates/merges the 18 cells, applies Holm correction, and plots the null distributions.
- `05A` `05A_nested_calibration_threshold_predictions.py` — nested Platt calibration, threshold predictions, and error-overlap analysis.
- `05B` `05B_plot_calibration_curves.py` — calibration plot with bootstrap confidence bands. Requires separate figure and analysis output directories; only PDF/PNG are written to the figure directory.
- `06A` `06A_simple_imaging_baselines.py` — HU/spleen-volume baselines.
- `06B` `06B_plot_simple_imaging_baselines.py` — simple-imaging-baseline plot.
- `07` `07_mask_agreement.py` — Dice/surface-distance analysis and summary.
  It accepts either two paired NIfTI mask directories or the prepared
  `Data/mask_agreement_per_case.csv`.
- `08A` `08A_totalfm_roi_sensitivity.py` — TotalFM ROI sensitivity analysis.
- `08B` `08B_plot_roi_sensitivity.py` — ROI sensitivity composite.
- `09` `09_radiomics_roi_sensitivity.py` — three-radiomics-ROI sensitivity analysis.
- `10` `10_nas3_in_steatosis_positive.py` — steatosis-positive NAS≥3 analysis. Requires separate figure and analysis output directories.
- `11A` `11A_chemotherapy_subgroup_analysis.py` — chemotherapy subgroup analysis.
- `11B` `11B_plot_chemotherapy_subgroups.py` — chemotherapy subgroup plot.
- `12A` `12A_complementarity_and_fusion.py` — exploratory complementarity/fusion analysis; placed last because it is exploratory.
- `12B` `12B_plot_complementarity_and_fusion.py` — complementarity/fusion plot.

TotalFM embedding extraction is handled by
`FM_embedding_extraction/extract_totalfm_embeddings.py`.
It accepts CT and liver-mask paths directly; resampling, liver-bounding-box
extraction, windowing, and resizing are performed internally by the TotalFM
preprocessing code. No separate patch archive is required.

## Reproducibility notes

- The final analysis uses 1,000 bootstrap resamples and 1,000 permutations.
- Model order is Merlin, RADAR, TotalFM, Jolia, OrganLens, Radiomics.
- Figures are exported as vector PDF by default; raster PNG is provided for
  convenient review.
- Analysis scripts use CLI paths so the package can be moved to a clean checkout.

## One-command replay of the reported downstream analysis

The complete downstream replay begins from the de-identified, precomputed
files in `Data/`; it does not require image volumes, masks, model checkpoints,
or any prior analysis output. It regenerates the computational manuscript
figures (Figures 2–5 and Figures S1–S6) and the table CSVs defined by the
current table-packaging script.
Figure 1 and Tables 1–2/S1–S2 are author-prepared descriptive materials and
are intentionally outside this computational replay.

1. Create a clean environment with Python 3.13 and install the core packages:

   ```sh
   python -m pip install -r requirements.txt
   ```

2. Place the supplied `Data/` directory beside `Results/` in the project root.
   The runner validates all required feature matrices, the clinical workbook,
   baseline CSV, and mask-agreement CSV before any computation begins.

3. From this directory, run:

   ```sh
   bash run_all.sh --revision-root /path/to/myRevision \
     --output-root /path/to/myRevision/Results/replay
   ```

   If the desired Python interpreter is not the shell default, use either
   `--python /path/to/python` or `PYTHON_BIN=/path/to/python`.

The target output directory must be absent or empty. The runner never
overwrites an existing analysis. It creates `analysis/`, `figures/`, and
`table/`; the temporary 18 permutation-cell files and Matplotlib cache are
removed automatically after a successful or failed run. `PERMUTATION_WORKERS`
defaults to 6 and may be reduced on lower-resource machines.

All stochastic procedures use the prespecified seeds embedded in the scripts,
including 1,000 class-stratified bootstrap resamples and 1,000 label
permutations. Do not add command-line seed overrides when reproducing the
reported results.

## Extraction scripts

`FM_embedding_extraction/` and `radiomics_baseline_extraction/` are retained
to document upstream feature creation. They are not invoked by `run_all.sh`.
FM extraction additionally requires the respective licensed/public model code,
weights, and model-specific environment; private model snapshots and weights
are excluded by `.gitignore`. The published replay is therefore designed to
start with the supplied de-identified feature matrices in `Data/`.

The scripts validate case IDs, labels, feature dimensions, and finite values
while reading `Data` directly. No undocumented staging command is required.
