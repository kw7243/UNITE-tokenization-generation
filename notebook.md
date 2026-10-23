# Source setup notebook

## Current state

- Date: 2026-10-23 UTC.
- Role: Source prepared for a separate reproduction worker.
- Official upstream: https://github.com/ShivamDuggal4/UNITE-tokenization-generation.
- Public fork: https://github.com/kw7243/UNITE-tokenization-generation.
- Pinned upstream source: `2389f2c65fedbd43ada851e69b2b810efd23af7d`.
- Canonical retained clone: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-tokenization-generation`.
- SSH access: `kwen1@slurm-login.csail.mit.edu` (observed `slurm-login-0.csail.mit.edu`).
- The authenticated GitHub account is `kw7243`. GitHub API confirms the fork parent is `ShivamDuggal4/UNITE-tokenization-generation`.
- This notebook is the only setup change. No environment, checkpoint download, experiment, or GPU allocation was started.

## Branch index

- `main`: canonical source plus this documentation-only setup commit; fork and upstream default branches remain `main`.
- `upstream/main`: official source at setup revision `2389f2c65fedbd43ada851e69b2b810efd23af7d`.
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
