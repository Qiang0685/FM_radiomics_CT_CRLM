# Local-only material — do not publish

The `local_model_code/` directories are kept solely to allow the study team to recreate its own feature-extraction environment. They must not be included in a public GitHub repository without a separate licence and redistribution review. They may contain upstream source code governed by their own licences and are not represented as original study code.

Before any public code release, retain only the study-authored extraction wrappers, configuration templates, dependency records, and instructions; replace all study data, paths, checkpoints, and generated embeddings with documented access routes or synthetic fixtures.

Chinese text in model-output vocabularies (for example, RADAR organ/finding keys) is intentionally retained where it forms a checkpoint/text-embedding lookup key. Translating those literals would change model behaviour. All archive names, entry-point documentation, comments, and user-facing messages have otherwise been converted to English.
