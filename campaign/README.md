# Joint adaptive tokenization campaign

Current worker scope: A1 UNITE-B reconstruction/generation and Section 5.11 essential adapter checks. No A2 or training.
Immutable plan SHA256: 71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b.
Coordinator: research-image-tokenization. Source/artifact-lock, submission, terminal and UNITE canonical notebook owner: unite-reproduce-20261023.
Initial allowance: 19.2 H100 GPU-hours; 4.8 held by coordinator. Legacy spending is recorded separately.

Untouched source snapshots live in third_party. Their commits and byte hashes are in campaign/locks/source-imports.json.
Wrappers, patches, configurations and evidence live in campaign. Full stages use reproduction/tools/stage_and_run.sh.
Retained outputs live in outputs. Full immutable plan and administrative receipts are retained there and are not published with code.
Use literal sbatch --export=NIL and define every required environment value after all directives. Do not use ALL,variables or NONE.

Reused inputs are verified by hash, not inferred from filenames. A1 uses seed 20261023, torch-fidelity 0.3.0, native M14 checkpoint settings, source interval [0,1], 5000 fixed validation reconstructions and independent balanced 50000 generation.
Saved CFG 2.0 must be reverified. Gate: valid reconstruction and strict loading plus gFID <=3.1, labeled CHECKPOINT_EFFECT_REPRODUCTION.
The 2.12 headline and 2.735 appendix values remain distinct. Old Large and alternate-interval runs are outside current scope.
