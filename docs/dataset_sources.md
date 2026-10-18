# Dataset source and privacy

[Data production repository](https://github.com/Peanut-butter116/uk-insurance-QA-data) owns collection, generation, validation, leakage screening and exports. Restricted records remain in its private local storage.

[Main research repository](https://github.com/Peanut-butter116/uk-motor-insurance-dslm) owns model experiments, Kaggle integration, aggregate evaluation and presentations.

Workflow: resolve the production final-data pointer once; verify its export hash; create an immutable private snapshot; obtain approval before uploading new restricted data to a private Kaggle dataset; verify dataset privacy and exact version; run GPU jobs; download and verify receipts and outputs; publish only code, schemas, manifests and aggregate summaries.

The 18 October snapshot contains 409 accepted records (384 distinct question texts), 135 inherited and 274 generated. It covers six insurers, 27 documents and 501 evidence sections. All carry passed protected-question screens. Human review is zero and the source sets training_ready=false. The 25 repeated texts are deliberate answerable/insufficient-evidence pairs with different contexts, with no exact model-visible duplicates. Keep each pair grouped and review label contrast before SFT. Source versions are content hashes, not verified printed editions.

The canonical CSV JSON-encodes every cell for exact equivalence with JSONL. A private display CSV uses readable strings, JSON for nested fields and spreadsheet formula escaping. Field definitions are listed in the public snapshot manifest. No private records or real policy excerpts are included here. The presentation's example is fictional and explicitly labelled.

Private experiment root: sibling `uk-motor-insurance-dslm-private-data/experiments/2026-10-20/`. Owner approved the 409-record snapshot upload in this continuation. Private Kaggle dataset: https://www.kaggle.com/datasets/scarletthe0116/uk-motor-interim-20261020-409 (version 1). Kaggle confirms privacy. Owner per-file download integrity is recorded in the access receipt; supervisor invitations await usernames. Existing private Kaggle references are documented in the weekly experiment README.
