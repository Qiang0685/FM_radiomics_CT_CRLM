# Model and checkpoint identity record

| Representation | Extraction route | Identity evidenced in supplied materials | Status |
|---|---|---|---|
| Merlin | `Merlin(ImageEmbedding=True)` | Loader default configuration; Hugging Face repository `stanfordmimi/Merlin`; checkpoint `i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt`; loader SHA-256 `f4370dde919cad09141abf5438c485ef197003337fcb8b11e848288fd43cad09` | `IDENTITY_VERIFIED` for code route and checkpoint filename; `AUTHOR_INPUT_NEEDED` for checkpoint digest and immutable upstream revision |
| RADAR | visual branch plus organ-query heads | Script expects `checkpoint_radar_pretrain.pth` and `infer_text_embedding_radar.pt` | `PRESENT_UNVERIFIED` |
| OrganLens | liver embedding | Script expects `teacher_checkpoint.pth` | `PRESENT_UNVERIFIED` |
| Jolia | liver embedding via `encode_organs(..., organs=["liver"])` | local Hugging Face-compatible repository snapshot | `PRESENT_UNVERIFIED` |
| TotalFM | liver embedding | script expects `totalfm_en_checkpoint_best_loss.pt` | `PRESENT_UNVERIFIED` |

## Merlin determination

The Merlin constructor receives `ImageEmbedding=True` only. In `merlin/models/load.py`, all image-embedding calls without `RadiologyReport`, `RadiologyReportStage1_Stage2`, or `FiveYearPred` select `self.task = "default"`. `MODEL_CONFIGS["default"]` supplies the checkpoint above. Thus the study used the **default Merlin image-embedding checkpoint**, producing a 2,048-dimensional vector; it did not use the report-generation, report-generation stage-1/stage-2, or five-year disease-prediction variants.

The supplied directory contains no checkpoint file, package tag, Git commit, or pinned Hugging Face revision. Do not infer those missing identifiers from the checkpoint filename. Add a checkpoint SHA-256 and resolved revision before claiming a replayed result.
