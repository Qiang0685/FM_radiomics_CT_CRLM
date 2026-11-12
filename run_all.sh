#!/usr/bin/env bash
# Replay all downstream analyses from precomputed Data matrices.
# This script never overwrites an existing non-empty output directory.
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REVISION_ROOT="$(cd "$CODE_DIR/../.." && pwd)"
OUTPUT_ROOT="$REVISION_ROOT/Results/reproduce"
PYTHON_BIN="${PYTHON_BIN:-python}"
PERMUTATION_WORKERS="${PERMUTATION_WORKERS:-6}"

usage() {
  cat <<'EOF'
Usage:
  bash run_all.sh [--revision-root PATH] [--output-root PATH] [--python PATH]

Examples:
  bash run_all.sh
  PYTHON_BIN=/path/to/python bash run_all.sh --output-root /path/to/replay

The output directory must not already contain files. The script writes three
subdirectories: analysis/, figures/, and table/.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --revision-root) REVISION_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REVISION_ROOT="$(cd "$REVISION_ROOT" && pwd)"
DATA="$REVISION_ROOT/Data"
OUTPUT_ROOT="${OUTPUT_ROOT/#\~/$HOME}"

if [[ ! -x "$(command -v "$PYTHON_BIN")" && ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$DATA" ]]; then
  echo "Missing input directory: $DATA" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -not -name '._*' -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $OUTPUT_ROOT" >&2
  exit 2
fi

required_inputs=(
  "01_Merlin_197x2048.csv"
  "02_RADAR_liver_embeddings_197x256.csv"
  "03_TotalFM_liver_TS_197x768.csv"
  "03b_totalfm_resected_liver_197x768.csv"
  "03c_TotalFM_TCIA_liver_197x768.csv"
  "04_Jolia_liver_197x576.csv"
  "05_OrganLens_liver_197x1024.csv"
  "06_radiomics_resected_TumorFree_197x1130.csv"
  "06b_radiomics_wholeLiver_tumorFree_197x1130.csv"
  "06c_radiomics_wholeLiver_197x1130.csv"
  "Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx"
  "simple_baseline_SpleenLiverHU.csv"
  "mask_agreement_per_case.csv"
)
for file in "${required_inputs[@]}"; do
  [[ -f "$DATA/$file" ]] || { echo "Missing required input: $DATA/$file" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT"
ANALYSIS="$OUTPUT_ROOT/analysis"
FIGURES="$OUTPUT_ROOT/figures"
TABLES="$OUTPUT_ROOT/table"
mkdir -p "$ANALYSIS" "$FIGURES" "$TABLES"

# Avoid machine-dependent BLAS oversubscription and disposable matplotlib state.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/fattyliver-replay.XXXXXX")"
export MPLCONFIGDIR="$TMP_ROOT/matplotlib"
PERMUTATION_CELLS="$TMP_ROOT/permutation_cells"
trap 'rm -rf "$TMP_ROOT"' EXIT

run() {
  echo
  echo "==> $1"
  shift
  "$@"
}

cd "$CODE_DIR"

run "01A: unified linear probes" "$PYTHON_BIN" 01A_unified_linear_probe.py \
  --input-root "$DATA" --output-dir "$ANALYSIS/01_unified_linear_probe" --bootstrap 1000

run "05A: nested calibration, thresholds, and error overlap" "$PYTHON_BIN" 05A_nested_calibration_threshold_predictions.py \
  --data-root "$DATA" --oof-dir "$ANALYSIS/01_unified_linear_probe" --out "$ANALYSIS/05_calibration"

run "02A: discrimination and signed DeLong contrasts" "$PYTHON_BIN" 02A_discrimination_and_signed_delong.py \
  --oof "$ANALYSIS/01_unified_linear_probe" \
  --seed-performance "$ANALYSIS/05_calibration/seed_level_performance_nested_calibration.csv" \
  --out "$ANALYSIS/02_delong" --figure-dir "$FIGURES"

run "03: threshold-derived metrics" "$PYTHON_BIN" 03_threshold_metrics_bootstrap.py \
  --oof-dir "$ANALYSIS/05_calibration" --out "$ANALYSIS/03_threshold_metrics" --figure-dir "$FIGURES" --bootstrap 1000

run "04A: permutation cells" "$PYTHON_BIN" 04A_permutation_computation.py \
  --oof-dir "$ANALYSIS/01_unified_linear_probe" --output-dir "$PERMUTATION_CELLS" \
  --n-permutations 1000 --workers "$PERMUTATION_WORKERS"
run "04B: permutation summary and figure" "$PYTHON_BIN" 04B_permutation_summary_and_plot.py \
  --cell-dir "$PERMUTATION_CELLS" --oof-dir "$ANALYSIS/01_unified_linear_probe" \
  --output-dir "$ANALYSIS/04_permutation_fixed_summary" --figure-dir "$FIGURES" --n-permutations 1000

run "05B: calibration figure" "$PYTHON_BIN" 05B_plot_calibration_curves.py \
  --oof-dir "$ANALYSIS/05_calibration" --output-dir "$FIGURES" --bootstrap 1000

run "06A: simple imaging baselines" "$PYTHON_BIN" 06A_simple_imaging_baselines.py \
  --inputs "$DATA/simple_baseline_SpleenLiverHU.csv" --oof "$ANALYSIS/01_unified_linear_probe" \
  --out "$ANALYSIS/06_simple_baselines" --bootstrap 1000
run "06B: imaging-baseline figure" "$PYTHON_BIN" 06B_plot_simple_imaging_baselines.py \
  --table "$ANALYSIS/06_simple_baselines/simple_baseline_comparison_auc_ci.csv" --out-dir "$FIGURES"
run "02B: pairwise AUC-comparison figure" "$PYTHON_BIN" 02B_plot_pairwise_auc_comparisons.py \
  --fm-rad "$ANALYSIS/02_delong/delong_signed_auc_contrasts.csv" \
  --baseline "$ANALYSIS/06_simple_baselines/simple_baseline_vs_models_delong.csv" --out-dir "$FIGURES"

run "07: mask agreement" "$PYTHON_BIN" 07_mask_agreement.py \
  --per-case-input "$DATA/mask_agreement_per_case.csv" --output-dir "$ANALYSIS/07_mask_agreement"

run "08A: TotalFM VOI sensitivity" "$PYTHON_BIN" 08A_totalfm_roi_sensitivity.py \
  --data "$DATA" --tcia-embedding "$DATA/03c_TotalFM_TCIA_liver_197x768.csv" \
  --oof "$ANALYSIS/01_unified_linear_probe" --out "$ANALYSIS/08_totalfm_roi" --bootstrap 1000
run "09: radiomics VOI sensitivity" "$PYTHON_BIN" 09_radiomics_roi_sensitivity.py \
  --data-root "$DATA" --code-root "$CODE_DIR" --out "$ANALYSIS/09_radiomics_roi" --bootstrap 1000
run "08B: VOI-sensitivity figure" "$PYTHON_BIN" 08B_plot_roi_sensitivity.py \
  --totalfm-performance "$ANALYSIS/08_totalfm_roi/totalfm_roi_performance_raw.csv" \
  --totalfm-delong "$ANALYSIS/08_totalfm_roi/totalfm_roi_pairwise_delong_raw.csv" \
  --radiomics-performance "$ANALYSIS/09_radiomics_roi/radiomics_roi_performance_raw.csv" \
  --radiomics-delong "$ANALYSIS/09_radiomics_roi/radiomics_roi_pairwise_delong_raw.csv" --out-dir "$FIGURES"

run "10: NAS >=3 in steatosis-positive cases" "$PYTHON_BIN" 10_nas3_in_steatosis_positive.py \
  --data "$DATA" --oof "$ANALYSIS/01_unified_linear_probe" \
  --analysis-dir "$ANALYSIS/10_nas3_positive" --figure-dir "$FIGURES" --bootstrap 1000

run "11A: chemotherapy subgroup analysis" "$PYTHON_BIN" 11A_chemotherapy_subgroup_analysis.py \
  --revision-root "$REVISION_ROOT" --cohort-manifest "$DATA/Colorectal-Liver-Metastases-Clinical-data-April-2023.xlsx" \
  --output-dir "$ANALYSIS/11_chemotherapy" --oof-dir "$ANALYSIS/01_unified_linear_probe" --bootstrap 1000
run "11B: chemotherapy subgroup figure" "$PYTHON_BIN" 11B_plot_chemotherapy_subgroups.py \
  --subgroup "$ANALYSIS/11_chemotherapy/subgroup_auc_raw.csv" --out-dir "$FIGURES"

run "12A: exploratory complementarity and fusion" "$PYTHON_BIN" 12A_complementarity_and_fusion.py \
  --oof-dir "$ANALYSIS/01_unified_linear_probe" --out "$ANALYSIS/12_fusion" --bootstrap 1000
run "12B: complementarity and fusion figure" "$PYTHON_BIN" 12B_plot_complementarity_and_fusion.py \
  --fusion-dir "$ANALYSIS/12_fusion" --overlap "$ANALYSIS/05_calibration/current_main_threshold_error_overlap.csv" \
  --oof-dir "$ANALYSIS/01_unified_linear_probe" --out-dir "$FIGURES"

run "13: manuscript table packaging" "$PYTHON_BIN" 13_package_final_tables.py \
  --analysis "$ANALYSIS" --data "$DATA" --out "$TABLES"

echo
echo "Completed successfully. Outputs:"
echo "  $ANALYSIS"
echo "  $FIGURES"
echo "  $TABLES"
