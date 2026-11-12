# Radiomics and imaging-baseline extraction

This folder contains two independent extraction scripts. They share a folder
for organization, but the radiomics feature extraction and simple imaging
baseline extraction are not combined.

## Radiomics features

`extract_radiomics.py` extracts the 1,130 PyRadiomics features used in the
study. It supports the three liver-mask sets:

```bash
python extract_radiomics.py \
  --image-dir /path/to/Img_nii \
  --mask-root /path/to/Raw_msk \
  --output-dir /path/to/output \
  --label resected_noVesTumor \
  --workers 4
```

Valid labels are `whole_noVes`, `whole_noVesTumor`, and
`resected_noVesTumor`. The settings are 1-mm isotropic resampling, B-spline
interpolation, 25-HU bin width, no normalization, Original/Wavelet/LoG image
types, and first-order, shape, GLCM, GLRLM, GLSZM, GLDM, and NGTDM features.

## Simple imaging baselines

`extract_liver_hu_spleen_volume.py` independently extracts only the two prespecified simple
imaging baselines:

- mean liver parenchyma HU from the resected tumour-free liver mask;
- spleen volume in mL from the spleen mask.

Example:

```bash
python extract_liver_hu_spleen_volume.py \
  --image-dir /path/to/Img_nii \
  --liver-mask-dir /path/to/Msk_Liver_resected_noVesTumor \
  --spleen-mask-dir /path/to/Msk_spleen_lcc_TS \
  --exclude-spleen-cases 1040 1139 \
  --output /path/to/simple_baseline_SpleenLiverHU.csv
```

The script checks matched 197-case CT/mask cohorts, binary masks, and image–mask
geometry. It does not resample, clip, filter, or modify source files. Missing
or QC-excluded spleen volumes remain missing; median imputation belongs to the
downstream analysis script, not extraction.

This is the clean, path-parameterized extraction implementation added for the
GitHub-ready package. The earlier analysis package contained the resulting
measurement tables and their definitions, but no standalone raw-image
extraction script. The definitions were taken from
`simple_baselines/source_data/Liver_HU_README.txt` and
`simple_baselines/source_data/SpleenLiv_HUVol_README.txt`; the downstream
evaluation remains in `Results/code/06A_simple_imaging_baselines.py`.
