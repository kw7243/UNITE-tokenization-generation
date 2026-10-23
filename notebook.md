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
- Initial source setup changed only this notebook. CPU setup has completed and input hashes are locked. Later research changes are recorded in dated entries below; no numerical result or GPU allocation exists yet.

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


## 2026-10-23 — CPU setup completed; released inputs locked

- Job 2826271 COMPLETED 0:0 on groenig-2: 1635 seconds / 6540 allocated CPU-core-seconds / zero GPU-seconds; MaxRSS 18412356 KB.
- Exact retained run evidence: `UNITE-released-20261023-results/cpu-setup-2826271/report.md`, `results.json`, `checkpoint-metadata.json`, `packages.txt`. All paths are under the retained image-tokenization root.
- Installed Python 3.10.22, Torch 2.10.0+cu128, torchvision 0.25.0+cu128, torch-fidelity 0.4.0, torchdiffeq 0.2.5, timm 0.9.16, transformers 4.56.2, NumPy 1.26.0 and SciPy 1.15.3, plus remaining source requirements.
- Base SHA256: `7b7f40dd37111196cad6e4401bf9737156268612e001c29cb0a760ae416bff4b`; bytes 2745102286; epoch 240, step 316000, saved CFG 2.0.
- Large SHA256: `5b15fe343c4b288d1e0231d04af4c38de126bd1609ac8b4155a5c147720d7a40`; bytes 6687027302; epoch 113, step 150000, saved CFG 1.8. This is the actual release, not proof of the advertised 120-epoch checkpoint.
- Inception SHA256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`; stats SHA256 `412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c`.
- Validation inventory: 50,000 images, 1,000 classes, 50 images per class; sorted synsets match all map indices. Class-map SHA256 `a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc`. Image content is not individually hashed.
- Both checkpoints contain raw and EMA weights, optimizer/scheduler states and compiled decoder keys. Complete scientific configurations are absent. The source/README remain the authority for non-weight settings, including 14 training flow iterations.
- This is a successful setup check, not a paper result. Next: strict CPU model loading and source readiness, then one attended GPU diagnosis. UNITE ownership remains with this assignment.


## 2026-10-23 — CPU readiness passed; first GPU request proposed

- CPU job 2826684 COMPLETED 0:0 on groenig-2: 189 seconds / 756 allocated core-seconds / zero GPU-seconds; MaxRSS 2889700 KB. These totals apply to the shared job, not separately to each model.
- Stage `/data/scratch-fast/kwen1/unite-released-20261023-stages/20261023_181429-bbd908-f744bc9-code` passed 130-file/link, execute-bit and Git-identity checks.
- Both Base and Large load strictly after exact normalization of 152 compiled decoder key names. Prescribed Inception and package checks passed. Per-model reports/results are in retained `cpu-readiness-2826684-B` and `cpu-readiness-2826684-L`.
- Actual encoder/decoder counts: Base 130789920 / 86435328; Large 458477824 / 86435328. Total including frozen LPIPS: Base 232951392, Large 560967488. Preserve these measured source counts instead of silently repeating advertised 217.6M/589.0M identities.
- GPU smoke implementation `1f780e8` records two calls at the selected evaluation batch size, warm/steady timing, observed encoder calls and peak memory. It has passed AST checks, not GPU execution.
- First GPU proposal: one RTX 3080, expected 10 GB, 4 CPUs / 16 GB RAM / 45 minutes, vision-torralba-urops-meng / vision-shared-rtx3080 / lab-free. Static FP32 model storage is below 2.3 GB for Large; activation/compile peak remains to be measured.
- Current all-partition/node, association, QoS and own-account job inventory is retained in research `reproduction/evidence/resources-20261023-2219-*`. No account GPU job is running; B1's H100 jobs remain pending and are untouched.
- Native TF32 support excludes older Turing/Pascal choices from matching the released evaluation backend. RTX 3080 is the smallest currently listed compatible-memory class. Verify actual VRAM/driver/CUDA/mounts after allocation.
- Diagnostic target: Base CFG 2.0 and Large CFG 1.8, batch 16, seed zero, compiled default decoder, FP32 with source TF32 settings, source dopri5 interval [0,1], CFG interval [0.1,1]. These are smoke calls, not paper metrics.
- Release the attended allocation after diagnosis. Estimate final 50K evaluation resources from observed steady timing, not the cold compile call.
