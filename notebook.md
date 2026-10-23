# UNITE research notebook

## Current state

- Date: 2026-10-23 UTC.
- Role: Released-checkpoint reproduction under way; sole owner unite-reproduce-20261023.
- Official upstream: https://github.com/ShivamDuggal4/UNITE-tokenization-generation.
- Public fork: https://github.com/kw7243/UNITE-tokenization-generation.
- Pinned upstream source: `2389f2c65fedbd43ada851e69b2b810efd23af7d`.
- Canonical retained clone: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-tokenization-generation`.
- SSH access: `kwen1@slurm-login.csail.mit.edu` (observed `slurm-login-0.csail.mit.edu`).
- The authenticated GitHub account is `kw7243`. GitHub API confirms the fork parent is `ShivamDuggal4/UNITE-tokenization-generation`.
- Initial source setup changed only this notebook. Current CPU-only setup and later research changes are recorded in dated entries below; no numerical result or GPU allocation exists yet.

## Branch index

- `main`: canonical source plus this documentation-only setup commit; fork and upstream default branches remain `main`.
- `upstream/main`: official source at setup revision `2389f2c65fedbd43ada851e69b2b810efd23af7d`.
- `research/unite-released-20261023`: independent released-checkpoint branch, parent de27e4a, implementation receipt dfc1e30.
- No existing divergent branch was changed. No force push was used.

## Storage and next step

- Retained source stays under `/data/vision/torralba/u/kwen1/image-tokenization-20261023`.
- Performance copies, environments, and caches belong under `/data/scratch-fast/kwen1`.
- Stage a complete repository copy before any later experiment, as required by the research reproducibility skill.
- The next reproduction worker must inspect the pinned README, configuration, training, and evaluation source before selecting the scientific protocol. No experiments have run.
- Start with `README.md` at the pinned source. Checkpoint URLs and evaluation notes are source references, not verified downloads.
- CSAIL GitHub SSH authentication was unavailable during setup. This notebook was committed through the authenticated GitHub API and then fetched into the retained clone. No credential was copied to CSAIL.


## 2026-10-23 — Released-checkpoint setup and staging recovery

- Sole submitter, terminal-result owner, and canonical notebook writer: assignment `unite-reproduce-20261023`.
- Independent research clone: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023`.
- Branch index: `research/unite-released-20261023` starts at main `de27e4a7084e631452b27d84d3e92aadec815ea7`, whose parent is official source `2389f2c65fedbd43ada851e69b2b810efd23af7d`.
- Backed-up setup source: `59227b6c796bbb0b937495986deaa86e0394caf4`, containing the CPU setup, metadata inspection and archived paper v1.
- Existing CSAIL HTTPS authentication works through the installed GitHub credential provider. Only the two UNITE clones received a local credential-helper setting. No credential changed.
- Targets: released Base FID-50K 2.12 (240 epochs), Large 1.73 (120 epochs). Both README checkpoint identities use 14 flow steps per reconstruction.
- Paper Appendix B specifies 50,000 class-balanced samples, torch-fidelity Inception, dopri5, and CFG 1.0–4.0 by 0.2. Appendix Table 4 instead gives 2.735 / IS 268.1 for nominally the same Base checkpoint. Preserve this discrepancy.
- Released integration interval is [0, 1]; the paper states [0.1, 1.0]. The released CFG interval [0.1, 1.0] is a separate setting. Exact evaluation fidelity is unresolved.
- Shared ImageNet is present at `/data/vision/torralba/datasets/imagenet_pytorch_new`; use it read-only. Exact counts, label-map hash and checkpoint hashes await the CPU job.
- No package install, checkpoint download or experimental command has run yet.
- Two full-copy attempts failed with rsync exit 23 while preserving unsupported setgid attributes. Failed copies remain at scratch `unite-released-20261023-stages/20261023_174022-6d0230-59227b6-code` and retained root `UNITE-released-20261023-stages/20261023_174047-241426-59227b6-code`.
- Firstmate resolved the repeated-obstacle stop on 2026-10-23. Prior proven asset helper differs from the installed helper only by `rsync -rlt --no-perms --no-owner --no-group`.
- Small copy counterfactual: archive options exit 23; compatibility options exit 0. No contrary copy evidence observed. Logs are in research `reproduction/evidence/staging-*-probe.log`.
- The project-local helper is copied from installed helper SHA256 `44e5dc6f1a958b1f4b32e8dceeb49814885ad8dcca59716090ca87b1033731fa`. Proven prior helper SHA256: `f69e0ffd7b8129302952d13d62d658645b8a83f26af9c10e68f8abeed024ae3b`. Installed skills and shared Firstmate files remain unchanged.
- Permission-policy difference: copy bytes, symlinks, times and meaningful execute bits; omit unsupported owner/group/setgid preservation. A complete included-file hash manifest and Git-identity comparison must pass before each submitted command. Root generated-output exclusions match the installed helper.
- Prepared CPU setup: `reproduction/setup.sbatch`, 4 CPUs / 24 GB RAM / 1 hour / zero GPUs on `csail/tig-cpu/tig-main`. Environment, checkpoints and validation inventory will be inspected before a GPU request.
- Advertised Base full training scale is 8 H200 140GB GPUs, approximately 26 min/epoch: 240 epochs estimate 104 node-hours / 832 GPU-hours before evaluation. No training is submitted or authorized by this estimate.
- No measured FID, IS or rFID is available; scientific status remains untested.

- Recovery check on 2026-10-23: first compatible full copy passed file/hash/execute-bit equality, then the Git comparison correctly detected generated provenance as an extra untracked file. No command ran. The comparison now excludes only the helper-generated REPRODUCIBILITY_METADATA.json and STAGING_MANIFEST.json; all source and untracked research files remain covered. Failed verification stage: /data/scratch-fast/kwen1/unite-released-20261023-stages/20261023_174439-cdf80b-37fa69c-code.


## 2026-10-23 — CPU setup accepted

- Job `2826271` accepted on `csail/tig-cpu/tig-main`: 4 CPUs, 24 GB RAM, 1 hour, zero GPUs; observed running on `groenig-2`.
- Full stage: `/data/scratch-fast/kwen1/unite-released-20261023-stages/20261023_174513-6fa88a-66211fb-code`, research commit `66211fb`.
- Stage verified 103 files/links, meaningful executable bits and Git identity before submission. Exact receipts: `REPRODUCIBILITY_METADATA.json` and `STAGING_MANIFEST.json` in that stage.
- Output root: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-results/cpu-setup-2826271`; log beside it: `cpu-setup-2826271.log`.
- Sole submission, terminal-result and notebook owner: `unite-reproduce-20261023`.
- This job prepares dependencies, downloads released artifacts, inspects checkpoint metadata and inventories the validation filenames/class map. It produces no paper metric.


## 2026-10-23 — Standalone evaluation entrypoint prepared

- Research implementation receipt: `dfc1e30` on `research/unite-released-20261023`, backed up to the authorized fork.
- `reproduction/protocol.md` records Table 1 targets Base FID 2.12 / IS 294.1 and Large FID 1.73 / IS 296.0, exact 50K balanced sampling, metric implementation, seed policy and unresolved source/paper differences.
- `reproduction/evaluate_released.py` loads the exact hash-locked EMA with strict state matching and invokes released `engines.eval.evaluate_fid`. It retains Inception features/logits without replacing metric functions. It is not yet numerically validated.
- CPU-only source readiness is prepared in `reproduction/readiness.sbatch`. It is not submitted before setup job 2826271 supplies checkpoint hashes and configuration evidence.
- Source trace: training evaluation hooks run outside BF16 autocast; decoder compilation uses default mode for the YAML gradient accumulation of two. The entrypoint restores those settings.
- Source concern: CFG=1.0 follows a separate normalization branch. This is recorded, not silently changed.
- Full paper reconstruction Table 2 uses Base at 120 epochs; released Base is advertised at 240 epochs. Do not claim that table is replicated by evaluating the wrong checkpoint.
- CPU job 2826271 is still installing dependencies. No GPU is requested and no numerical claim is made.
