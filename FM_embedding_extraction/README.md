# Local embedding-extraction archive for the five foundation models

This is a **local-only** archive for reproducing the embeddings used in this study. It contains English-language extraction entry points and the corresponding model-source snapshots. It is not a public-release package: do not upload `local_model_code/`, private images, masks, clinical tables, checkpoints, or generated embeddings to GitHub.

## Scope

The archive covers the five representation arms used here: Merlin, RADAR, OrganLens, Jolia (liver embedding), and TotalFM. Zero-shot analyses are deliberately outside the present organisation task. The legacy RADAR and TotalFM extractors retain certain zero-shot implementation objects because they are intertwined with their original inference pipelines; these objects are not part of the study's current modelling scope and should not be run or reported as study results.

## Layout

- `extract_merlin_embeddings.py`, `extract_radar_embeddings.py`, `extract_totalfm_embeddings.py`, `extract_jolia_liver_embeddings.py`, and `extract_organlens_liver_embeddings.py`: the five study extraction entry points.
- `data_loader.py`: shared cohort-file pairing and validation utility.
- `local_model_code/`: one private folder containing the model-source snapshots (`merlin`, `radar`, `totalfm`, `jolia`, and `organlens`). **Excluded from public release.**
- `MODEL_VERSION_RECORD.md`: model/checkpoint identity evidence, including the Merlin determination.
- `LOCAL_ONLY_NOTICE.md`: public-release boundary and required external inputs.

TotalFM uses one extraction entry point: `extract_totalfm_embeddings.py`.
To generate embeddings for a different liver ROI, provide the corresponding
mask directory through `--liver-mask-dir`; the model and feature dimension stay
the same while the input mask/ROI changes.

## Merlin version used for this study

The study extractor calls `Merlin(ImageEmbedding=True)`. In the archived loader this resolves to the `default` image-embedding configuration, not any report-generation or five-year-prediction configuration. The exact checkpoint filename is:

`i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt`

The loader identifies the upstream Hugging Face repository as `stanfordmimi/Merlin`. No checkpoint file or immutable upstream revision was present in the supplied source folder, so the checkpoint checksum and revision remain `AUTHOR_INPUT_NEEDED`; the loader source checksum is recorded in `MODEL_VERSION_RECORD.md`.

## Before a rerun

1. Keep the data, checkpoints, and output directory outside this archive.
2. Set `FATTYLIVER_DATA_ROOT` to the directory containing `Img_nii/` and the liver-mask folders. Clinical labels are not required for embedding extraction.
3. Record the exact checkpoint path and SHA-256 digest before execution.
4. Use the original model code snapshot under the matching `local_model_code/<model>/` folder, rather than an unverified updated upstream release.
5. First run a single-case smoke test; then run the full cohort. Preserve standard output, output hashes, and the command used.

This package is currently **TRACEABLE**, not replay-certified: code and model-source snapshots are present, but the original checkpoint identities, complete environment records, and a fresh run receipt have not been verified here.
