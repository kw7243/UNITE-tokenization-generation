## 1 Objective

Prepare only Section 5.4 ImageNet100 training splits.
Execution used version 1.0, SHA256 71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b.
The current plan is version 1.1, SHA256 4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5.
Section 5.4 is text-identical; these completed artifacts are compatible. This was not a new v1.1 execution.
Node PILOT_DATA; method SPLIT_V1; deterministic, no random seed.
The predeclared integrity gate is exact class coverage, split membership, byte identity, and ALLOC containment.

## 2 Setup

Source/stage: /data/scratch-fast/kwen1/jat-pilot-data-20261024-stages/20261024_025819-4842a9-4929e73-code.
Output: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z/PILOT_DATA__SPLIT_V1__snone__fd1bec3f630d.
Source commits: UNITE 2389f2c65fedbd43ada851e69b2b810efd23af7d; KARL 5c1a5693edf87228f988f363b2acdb7eb826a889.
Original executed config SHA256: fd1bec3f630d0da887bcab20cc8562b8ababf66aab5da4c8552e10eac6b0f00b.
CPU environment and exact argv: environment.json. Source/config/script locks: provenance.json and source_hashes.json.
Data root: /data/vision/torralba/datasets/imagenet_pytorch_new/train; resolved identity and symlinks: dataset_identity.json.
Class mapping sorts synsets as ImageFolder does. Full stage includes .git and untouched official sources under third_party.
Preprocessing is recorded only by pinned source pointers and hashes in interface.json; no implementation was added.

## 3 Baseline

No model or baseline comparison is run. Existing KARL content hashes cover validation only and cannot identify training files.
A1 validation and generation-label preparation are separate. No A1 workload is duplicated.

## 4 Results

Verified counts: {"ALLOC": 3200, "CONFIRM": 3200, "CONTENT": 126689, "DEV": 3200, "FIT": 120289}.
Exactly 100 official classes; DEV, CONFIRM and ALLOC each have 32 rows per class.
FIT preserves every remaining training image. Per-class actual counts: checks.json.
All integrity checks passed. Raw bytes hashed once: 14495992270 bytes.
Manifest SHA256 values: manifest_hashes.json. Row interface: interface.json.
Model metrics and outcome_class are null with reasons. Training/GPU time is zero.
One CPU attempt completed 0:0 on groenig-1, with 4 allocated CPU cores and 4 GB requested RAM.
Elapsed time: 816 seconds; allocated CPU core-hours: 0.906666667; recorded total CPU: 08:40.931.
Terminal accounting and both script markers: receipts/terminal-receipt.json and retained stdout/stderr.
No failed Slurm attempt or rerun. All owned jobs are terminal; submission ownership is released.

| Method | Seed | Data preparation status | Training outcome | GPU-hours |
|---|---|---|---|---:|
| SPLIT_V1 | Not applicable | Complete; integrity verified | Not applicable | 0 |

## 5 Interpretation

The pilot data artifacts satisfy the prescribed split and byte-identity rules.
This supports data preparation only. It does not establish A1, A2, training correctness, or any scientific hypothesis.

## 6 Limitations

The original archive hash is unavailable. Extracted file hashes identify the observed bytes; no archive hash is inferred.
CONFIRM remains sealed for scientific use until D1 method freeze. No images were decoded, visualized, scored, or ranked by difficulty.
The shared dataset can change later; consumers must require these frozen content identities.
Content identity does not test JPEG decoding; images were not decoded in this preparation.
No training sampler, selector, model, feature statistics, or full ImageNet1K split was built.

## 7 Next action

Version 1.1 controls future work. GPU-hour figures are optional guides, with no fixed reserve or estimate-triggered stop.
B1 phase and checkpoint fields do not apply to data preparation; no compute extension occurred.
Original execution evidence is preserved in receipts/execution-v1.0-* and the unchanged stage/config.

PASS for bounded data preparation only. Coordinator must accept interface.json, checks.json and exact manifest hashes.
The canonical campaign writer owns the artifact lock. A2 still requires A1 reference and adapter checks.
