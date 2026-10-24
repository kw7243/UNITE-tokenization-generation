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


## 2026-10-23 — First GPU preflight stopped on a node driver mismatch

- Allocation 2826772 selected isola-3080-1, one RTX3080-class GPU / 4 CPUs / 16 GB / 45-minute limit. No model command ran.
- Full source stage: `/data/scratch-fast/kwen1/unite-released-20261023-stages/20261023_182231-62ac98-3259811-code`; 161-file/link, execute-bit and Git proof passed.
- NVML failed with exit 18. A read-only cause check found loaded kernel driver 580.178.04 versus system NVML/libcuda 580.196.07, with empty LD_LIBRARY_PATH. This is independent of Torch or model code.
- The allocation was released immediately after cause inspection. Authoritative accounting: 22:22:53–22:24:55 UTC, 122 allocated GPU-seconds / 488 allocated CPU-core-seconds. Allocation exit 0:0 does not mean the preflight passed: step 2826772.0 failed 18:0.
- Retained report/results/logs: `UNITE-released-20261023-results/gpu-diagnostic-2826772/`; no raw file or failed attempt was removed.
- Scientific status remains untested. No system driver, shared library or scheduler state was changed.
- Prior authorized records and existing project evidence were checked; no matching workaround was found. Do not repeat the same probe on this proven unhealthy node.
- The 22:25 UTC full partition/node/account/QoS inventory remains in research `reproduction/evidence/resources-20261023-2225-*`. No account GPUs are running; B1's H100 jobs remain pending and untouched.
- Corrected request proposal: one RTX3090 24GB / 4 CPUs / 16 GB RAM / 45 minutes on vision-torralba-rtx3090 with vision-torralba-interactive QoS, same account and same diagnostic. This is the next-smallest viable native-TF32 class after the sole RTX3080 node failed its prerequisite. Do not change scientific settings to address a node fault.


## 2026-10-23 — Healthy GPU allocation and TF32 compatibility proof

- Corrected allocation 2826865 selected torralba-3090-3: RTX3090 UUID `8db9c33b-38d7-ff85-9dc2-8a8d5cee5ae1`, driver 580.178.04, native SM8.6, CUDA 12.8, one visible GPU, GCC/G++ 11.4, allocated 4 CPUs / 16 GB RAM / 45 minutes.
- Allocated-node mounts and package identity passed before model execution. Allocation stage: `20261023_182812-8e2145-c8bfa48-code`; separately staged preflight: `20261023_182929-a26954-c8bfa48-code`, under the UNITE-specific scratch staging root.
- First Base smoke `base-smoke` failed at decoder compilation due to Torch 2.10's legacy TF32 getter conflicting with the source's newer setters. Step 2826865.1 failed 1:0 after 34 seconds. Preserve its log/report/results; no paper metric or decoded image was produced.
- [Torch 2.10 CUDA documentation](https://docs.pytorch.org/docs/2.10/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) describes both APIs and disallows mixing them. Installed Inductor's warning path reads the legacy getter.
- Adapter compatibility code `1db5adf` uses one legacy API family while preserving TF32-enabled matmul/cuDNN and default decoder compilation. Core source model, sampler, metric implementation and checkpoint tensors remain unchanged.
- Staged primitive counterfactuals in the same allocation passed. Source new-API eager versus legacy-only eager matrix and convolution outputs were bitwise equal, maximum absolute difference zero. Compiled legacy matrix versus eager was also exactly equal.
- This verifies the selected native-API compatibility change for the fixed primitives; it does not establish full-model or paper equivalence.
- Retained diagnostic root: `UNITE-released-20261023-results/gpu-diagnostic-2826865/`, with separate report/results for every completed or failed command.
- Base retry `base-smoke-r1` completed finite-image/feature checks. It generated 32 smoke images, not FID-50K: cold/warm calls 17.7716 / 10.8931 seconds, encoder evaluations 104 / 110, peak allocated/reserved 1463837184 / 1853882368 bytes.
- Large smoke is running in the same attended allocation. No additional GPU reservation or dependency change was made for the precision API fix.
- Steady Base sampling alone projects about 9.46 hours per 50K on this RTX3090 before feature extraction/scoring and runtime variation. Do not treat this two-call projection as a measured paper-run duration.


## 2026-10-23 — GPU diagnosis terminal; interval and launch checks passed

- Allocation 2826865 COMPLETED 0:0 and released at 22:55:30 UTC: 1616 allocated GPU-seconds / 6464 allocated core-seconds, charged once.
- Source Base / Large steady batch-16 sampling: 10.8931 / 35.9369 seconds; peak allocated 1463837184 / 2923552256 bytes.
- Explicit paper-text Base / Large steady sampling: 8.5596 / 28.4953 seconds. Both passed finite image/feature checks; actual 50-point grids are saved.
- Source ODE interval is [0,1]. Adapter c3a9dd8 changes only the paper-text grid to 0.1 + 0.9 * shifted_grid, retaining source Gaussian start and drift. Unpublished intended settings remain uncertain.
- Adapter a1b4484 reuses native distributed setup with per-rank receipts and source seed rule global_seed * world_size + rank. One-rank torchrun images, labels, features and logits match direct Base launch bitwise.
- Every child has report.md and results.json in retained UNITE-released-20261023-results/gpu-diagnostic-2826865. accounting.txt includes the initial compile failure.
- Branch index: research/unite-released-20261023 at a1b4484 backed up to kw7243; parent de27e4a7084e631452b27d84d3e92aadec815ea7, official pin 2389f2c65fedbd43ada851e69b2b810efd23af7d.
- Paper scores remain unmeasured. Source-path single-RTX3090 sampling projects Base 9.46 hours / Large 31.2 hours per 50K, before extraction/scoring and adaptive variation.
- Next: prepare source-engine batch evaluation, inspect current permitted resources/account use, report requests and submit complete stages. UNITE job/result/notebook ownership remains active.


## 2026-10-23 — First full evaluation externally cancelled before confirmed batch execution

- Frozen initial plan fce4491 requests Base release 1 RTX3090, Large release 4, Base paper-text 1 and Large paper-text 3; batch 16 per rank, zero global seed with native seed = global_seed * world_size + rank.
- The measured single-steady-call sampling projections are 9.46 / 7.80 / 7.43 / 8.25 hours respectively, excluding extraction/scoring and adaptive variation. All limits are 12 hours. Exact plan: reproduction/locks/final-plan.json in the research branch.
- IS uses all 50,000 images and the first 1,000 class dimensions per image from 1,008 logits, then ten contiguous splits. It is not a 1,000-sample evaluation.
- Actual paper batch/world size and RNG state are unspecified. Saved CFG Base 2.0 / Large 1.8 remains separate from the unrun 16-point paper sweep. Large interval comparisons use different world sizes and do not isolate a causal interval effect.
- Slurm accepted Base-release job 2827364 from full stage /data/scratch-fast/kwen1/unite-released-20261023-stages/20261023_230229-7f3b99-fce4491-code (236 verified files/links).
- Job 2827364 then CANCELLED by UID 0 after 2 seconds, 23:02:44–23:02:46 UTC; Reason=None, PreemptTime=None, Restarts=0. No initial batch receipt, stdout or GPU preflight log exists. No model execution or metric is confirmed.
- Retained report/results/accounting: UNITE-released-20261023-results/fid-Base-release-2827364. Allocated cost is 2 GPU-seconds / 8 core-seconds; this is an operational cancellation, not a scientific failure.
- Read-only association, QoS, partition, public job-submit/prolog and node/controller checks give no cause. Controller log is unavailable on login. No system file changed, no admin privilege used, and no speculative resubmission or other final request made.
- Source unequal-count logic already gathers actual counts, pads to max and slices g[:count]. CPU proof gives [16667,16667,16666] for world 3 and exactly 50,000 labels, each class 50 times.
- Update 68d1c7d prepares native allocated-node GPU binding, NCCL all-reduce and the original source evaluator's unequal-count gather on 2,000 synthetic rows before long inference. No model or scientific metric is used in this test. Actual multi-rank GPU execution remains unverified.
- Future preflight and full evaluation each receive a new complete stage inside the submitted immutable stage; inherited skip-stage state is explicitly cleared.
- Next: obtain the controller-side cancellation reason or a verified remedy. Research/terminal/notebook ownership remains with unite-reproduce-20261023; no owned job is active.


## 2026-10-24 — A1 campaign scope and TIG-safe launch validation

- Immutable joint adaptive tokenization plan SHA256 71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b is verified and retained privately in campaign outputs.
- New independent plain source /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024, branch research/joint-adaptive-tokenization-20261024, parent a19c1501ed9d28e251a90950349bf2a739c9a9f3. Setup8bc4374 and environment preparationea61231 are backed up to the approved UNITE fork.
- Untouched pinned sources are imported under third_party with byte manifests: UNITE2389f2c65fedbd43ada851e69b2b810efd23af7d; KARL5c1a5693edf87228f988f363b2acdb7eb826a889. No KARL research copy or notebook changed.
- Current worker scope is A1 Base5000 fixed validation reconstructions, independent balanced50000 generation and the three Section5.11 adapter checks. No A2, training, Large or alternate-interval campaign is started.
- A1 evaluation seed20261023 and torch-fidelity0.3.0 supersede old seed0/fidelity0.4.0. Preserve native M14 checkpoint identity and sampling. Saved CFG2.0 must be reverified. Practical gate is gFID<=3.1 plus valid reconstruction/strict load, labeled CHECKPOINT_EFFECT_REPRODUCTION. Headline2.12 and appendix2.735 remain distinct.
- Supplied TIG evidence identifies ALL,variables export as unsupported. The effective2827364 argv matches it exactly. Historical inherited SBATCH_EXPORT is unrecorded, current unset. This is a documented matching cause/remedy, not new controller-log proof.
- Literal-NIL CPU canary2830341 COMPLETED0:0 ongroenig-3,06:21:25–06:21:30 UTC,5CPU-core-seconds,zeroGPUs. Both first-script markers exist; submit sentinel and inherited export value are absent. Required environment is set after every directive; Git fsck passed.
- Canary full stage /data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_062106-ed7829-8bc4374-code;252-file/link/Git verification. Retained run outputs/A1__LAUNCH_NIL_CPU__s20261023__0d8d0039dc3d includes exact argv, latch, owner log, resolved config, report, results and accounting.
- Initial A1 allowance19.2H100-hours; coordinator retains4.8 reserve and owns gates. Legacy0.483333device-hours remain separate. This worker is sole campaign source/artifact-lock, UNITE canonical notebook, submission and terminal owner.
- CPU environment/artifact reconciliation is starting from a new complete stage. Actual GPU software identity and every A1 scientific result remain unverified.

## 2026-10-24: A1 launch validation and prepared essential checks

- A1 launch canary `2830341` completed `0:0` on `groenig-3`, 06:21:25–06:21:30 UTC. One CPU, 1 GiB, zero GPUs; MaxRSS 39952 KB.
- Its full stage is `/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_062106-ed7829-8bc4374-code`. Both first-script markers, stripped sentinel, explicit environment and Git fsck passed.
- Retained canary report: `outputs/A1__LAUNCH_NIL_CPU__s20261023__0d8d0039dc3d/report.md`; terminal accounting and submission receipts are alongside it. This validates a CPU launch only.
- CPU environment/artifact job `2830382` was accepted from `/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_062617-3317c5-ea61231-code`. Request: four CPUs, 24 GiB, one hour, csail/tig-cpu/tig-main, literal NIL, attached sbatch --wait owner.
- Setup output: `outputs/A1__ENVIRONMENT_LOCK__s20261023__ac02febdf65c/`. It remains queued at 06:49 UTC; no duplicate was submitted. The compute-node environment and hashes must pass before any GPU request.
- Local wrappers reuse the untouched source. The isolated register hook removes patch outputs for counts below256; physical count groups restore image order. Reconstruction uses source VGG loss; denoising detaches target/input and sums three draws per image.
- `campaign/scripts/check_adapter.py` prepares only Section5.11 checks: 256-register identity, inactive/grouped/per-image reduction checks, and gradient/generation boundaries. The FP32 fixture disables TF32/compile and makes no training claim. These checks are not yet executed.
- `campaign/scripts/evaluate_a1.py` delegates 50K generation to the unchanged training evaluator, retaining its features/logits and actual NFE. It prepares the required fixed5K reconstruction L1/AlexLPIPS/rFID workload. Validation and full-reference configs have different method IDs.
- `campaign/scripts/resolve_reference.py` requires completed CPU accounting and fixed artifact/environment hashes. Native M14/256 reference settings remain distinct from the isolated M3/64 correctness fixture.
- The coordinator accepted the separate KARL reference. Read-only retained pointers/hashes are in `outputs/karl-accepted-pointers.json`. No KARL repository was changed; this does not complete A1.
- A separate CPU lane owns pilot FIT/ALLOC/DEV/sealed-CONFIRM construction. This source prepares only A1 five-per-class validation and balanced50K labels.
- New GPU use remains zero. Initial allowance is19.2 H100 hours; coordinator reserve4.8. Legacy0.483333 RTX-device hours remain separate.
- Next action: finish CPU preparation, inspect current compatible resources/account usage, then submit the smallest useful GPU validation. Full A1 remains unrun; A2/training are outside this worker scope.

## 2026-10-24: CPU environment failure and one diagnosed retry

- Job2830382 ran06:53:30–06:54:56UTC on groenig-2;FAILED1:0,86reserved seconds,four CPUs,24GiB,MaxRSS120536KB,zero GPU-hours.
- The first stdout/stderr markers appeared. This was a package-install failure after successful literal-NIL launch.
- The new venv seeded pip23.0.1; it rejected typing_extensions wheel metadata and then could not resolve flit_core from the CUDA-only index. The prior working environment has pip26.2.1.
- Preserve the failed environment and stage. The one diagnosed CPU retry uses a separate -r1 environment,pinned pip25.3 with published wheelSHA9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd,PyPI dependency resolution and explicit cu128 Torch/torchvision identities.
- Release evidence: https://pypi.org/project/pip/25.3/ . The bootstrap wheel report, package reports and final freeze will be retained.
- A1 reference and correctness checks remain unrun; no scientific negative is inferred. GPU use remains zero.

## 2026-10-24: version1.1 adopted; A1 gates unchanged

- Read all1227lines of the replacement plan and its handoff. Current plan SHA256 is `4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5`.
- Current retained plan bytes at `outputs/immutable-plan.md` now match v1.1. Original v1.0 bytes remain at `outputs/historical-plan-v1.0-71728da427a6.md`; completed/submitted stages and configurations retain their execution identity.
- `campaign/locks/plan.json`, the current source-import binding and future GPU configuration resolver now bind v1.1. This supersedes the earlier19.2+4.8split: all GPU-hour figures are advisory, with no hard cap, fixed reserve, estimate-triggered stop or approval.
- Section15.2 permits evidence-based extensions/targeted diagnostics. Record reason, original/revised config hashes/endpoints, affected controls, forecast/actual costs and actual constraints. No estimate alone changes status.
- CPU retry `2830643` continues unchanged under original config `da7c044db1c64d6480ab2a1dc0c0f8a892968ff7c255305ac2303fff7fd097e3`, source `6d07622c4d5e17b4fd4b57651ee948ee255db8b3`, and its full stage. The v1.1 environment/reference requirements and Section5.11 checks are unchanged. Do not cancel, duplicate or relabel that execution.
- Job2830382 remains a preserved pip-bootstrap failure. Retry2830643 uses a separate environment and verified pip25.3 bootstrap; its native literal-NIL launch and package resolution passed the prior boundary. No GPU has been submitted.
- Accepted KARL evidence remains read-only and accepted under unchanged Section6 gates, with its original execution identity. Source, terminal, artifact-lock and canonical UNITE notebook ownership remain with this worker; coordinator owns acceptance/next-node decisions.
- Later work now distinguishes B1 SCREEN/conditional PAIRED/RATE, SCREEN_PREFIX versus budget-ready COMMON_PREFIX, and the5.12/16.2 records. This does not expand this worker beyond A1 or enable A2/training.
- Exact before/after hashes and compatibility evidence: `outputs/revision-v1.1-adoption-csail.json`. The separate pilot-data lane remains independent.
- Next action: complete CPU2830643, freeze compatible artifacts and run the staged A1 validation/essential checks, then the complete5K reconstruction and independent50K generation.

## 2026-10-24: fixed pilot data preparation

- Prepared Section 5.4 only; no A1/A2 or scientific pass claimed.
- Source branch: research/jat-pilot-data-20261024; stage: /data/scratch-fast/kwen1/jat-pilot-data-20261024-stages/20261024_025819-4842a9-4929e73-code.
- CPU job: 2830620; host: groenig-1.novalocal; GPU-hours: zero.
- Retained output: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z/PILOT_DATA__SPLIT_V1__snone__fd1bec3f630d.
- Counts: {"ALLOC": 3200, "CONFIRM": 3200, "CONTENT": 126689, "DEV": 3200, "FIT": 120289}. All integrity checks passed; exact 100-class set and actual FIT counts retained.
- Manifest hashes: manifest_hashes.json; consumer contract: interface.json; report: report.md.
- CONFIRM remains sealed until D1 method freeze. Only IDs, hashes and integrity counts were prepared.
- Coordinator accepts these exact files; the sole canonical campaign writer updates its lock and main notebook.
- Terminal: COMPLETED 0:0, 816 seconds, 0.906666667 allocated CPU core-hours, no failed Slurm attempts. Ownership released.
- SHA256 ALLOC.tsv: 8e8754ef26e985c927e8b72fcef842d7f911678bb7c13d58469d6430b11ac126.
- SHA256 CONFIRM.tsv: dad77476e3a70426f01d3592b72ccdf65cf2bdb2b45c83891903f65dfba7b882.
- SHA256 CONTENT.tsv: 90a710bf273c7bee7818f629044221828b560dd20e85e0b3eef23f5444665f09.
- SHA256 DEV.tsv: bd5bbfb6bff9c38a07b4d7743f0b59369ef7ff220434712a68864576e0a85d3a.
- SHA256 FIT.tsv: e61a2f3994ef5d28a1c05df68f425ce705dcaf38ba423b0a4710f1f1d700b0c5.
- SHA256 class_to_idx.json: cad280772fed54b561c06ab6d3bb43f9c27d99ecd79c13878054147e38eb4115.
- SHA256 classes.json: 4bdfa8a1e607cc5cff7498aa87fcdf755f175853954b18846eaa77ce9cbfb846.

- Revision adoption: current plan v1.1 SHA256 4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5; full 1,227-line read completed.
- Execution remains v1.0 with original config/stage hashes. Section 5.4 matches exactly; no cancellation, rerun, data hash change, or relabeling.
- B1 phase/extension fields do not apply. Future GPU estimates are advisory; this lane remains CPU-only.

## 2026-10-24: CPU prerequisite complete; bounded A1 GPU validation prepared

- CPU retry2830643 completed0:0 on groenig-2,07:02:22–07:27:29UTC,1507reserved seconds,fourCPUs,24GiB,MaxRSS12752276KB,zero GPU-hours. Source/stage/config remain their original v1.0 identities and are compatible with v1.1.
- Retained run: `outputs/A1__ENVIRONMENT_LOCK_R1__s20261023__da7c044db1c6/`; terminal accounting, original payload reports, installed-wheel reports, pip inspect and final package freeze are present.
- Actual compute environment: Python3.10.22,Torch2.10.0+cu128,torchvision0.25.0+cu128,torch-fidelity0.3.0 and the remaining scientific pins. Package lock SHA256 `fbc44e755ea7ab9c1fb95f88783678b7c57fc8a367086054903fbfc2617ef253`.
- BaseEMA strictly loaded with no missing/unexpected keys after152 exact compiled-key normalizations. Epoch240,step316000,savedCFG2.0,256registers,d32,M14 reference identity. Calibration is unnecessary.
- Required5000validation-image manifest SHA256 `d31e573bba422ea6626552e1b2e717bffbabf2a9f6adea777514e5ff85085b0d`; generation-label manifest `8feedadddbffa7f8b868dc0df60afc2a8c4846376aa717e40623822cb97043eb`.
- Metric protocol SHA256 `693aa2cca24f2ae03238cef3453b9b6f95f39120da9a2575b04415afc87e0971`. Prescribed Inception, externalAlexLPIPS and sourceVGG loads passed; JiT statistics match Gitblob3c94ea961be94d06efa9e5bf4da71eff52deaebb and their byteSHA.
- Additional read-only CPU imports of the source evaluation engine and essential checker passed. No GPU or numerical quality metric has run.
- Current artifact lock is `campaign/locks/artifacts.json`; it links immutable accepted KARL/pilot evidence while keeping A1's1000-class protocol separate from the pilot100-class map.
- Before first GPU request, all permitted partition/association and compatible node allocations were inspected. No user GPU is running. RTX4090/A100/H200 healthy nodes are fully GPU allocated; one shared H100NVL slot is unallocated at the snapshot. The separate preconditioning A100/4090 requests remain untouched.
- Proposed validation: oneH100,4CPUs,24GiB,30minutes,vision-shared-h100/shared-if-available,full-stage literalNIL and attached sbatch--wait. Expected5–15minutes is an unmeasured forecast, with requested0.5device-hour. Actual GPU/VRAM/CUDA/build/mounts must pass before model loops.
- Validation configuration `campaign/configs/validation-2bc8cb64303e.yaml`, SHA256 `2bc8cb64303eaea18ef5a2ebca7ad3937b1ce6482bf65ff40074c73757e892a0`, preserves the native M14 reference and isolated M3 correctness fixture. It is not a paper-score run.
- Next action: run the staged validation once, inspect all5.11 proofs and actual H100 throughput, then forecast and submit the complete5000reconstruction/independent50000generation reference. A2/training remain outside this worker.

## 2026-10-24: first GPU launch reached the node; parser defect corrected

- GPU validation2830897 was accepted from `/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_073714-f082c3-c2bd3a9-code`, config `2bc8cb64303eaea18ef5a2ebca7ad3937b1ce6482bf65ff40074c73757e892a0`.
- It ran on andreas-h100-1,07:38:05–07:38:17UTC:FAILED1:0,12reserved H100 seconds,4CPUs,24GiB,MaxRSS3417868KB. Actual CUDA device/VRAM were not queried because configuration validation failed first.
- Both first-script markers and mounts passed; literalNIL worked on the GPU node. No model sample,5.11 proof,FID,IS or rFID was produced. This is operational failure, not a scientific negative.
- CPU reproduction identified exactly one sampling-field difference: yaml.safe_load read JSON exponent1e-06 as a string. The generated files are JSON/YAML1.2. Use json.loads to retain the declared numeric type; the unchanged sampling guard passes on CPU through the actual source reader.
- Preserve the failed stage, config, logs, original payload reports and terminal accounting in `outputs/A1__UNITE_B_VALIDATE_ESSENTIAL__s20261023__2bc8cb64303e/`.
- Section15.2 targeted rerun is predeclared in `campaign/configs/diagnostic-2830897.json`. The revised config is `campaign/configs/validation-400d3bce3a02.yaml`, SHA256 `400d3bce3a02dee9d7ad6e1e8290f0cbfd59a0fcecc5c33b09d38bd5c8fce0f4`.
- The same checkpoint,seed,source sampler,32native generation-smoke samples,3native reconstructions and three essential checks remain fixed. No comparison/training arm is affected. Expected additional reservation is at most the requested0.5H100-hour; actual cost will be recorded. No estimate-based approval applies.
- Current new GPU reservation cost is12/3600=0.003333H100-hours; legacy RTX spend stays separate. Full5000/50000reference remains unrun;noA2/training.

## 2026-10-24 A1 essential checks measured and full reference prepared
GPU2830982 completed0:0 in37reserved seconds on NVIDIA H100 NVL,CUDA12.8,Torch2.10.0+cu128.
Execution source:e80c229dc4bd494e0eabf0c6bdeb4dc936fdbf19;config400d3bce3a02dee9d7ad6e1e8290f0cbfd59a0fcecc5c33b09d38bd5c8fce0f4.
Stage:/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_074956-0912fa-e80c229-code.
Output:/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/A1__UNITE_B_VALIDATE_ESSENTIAL__s20261023__400d3bce3a02.
All three Section5.11 correctness checks pass in FP32,TF32 off,uncompiled.
Identity and inactive-tail relative errors are0;group/single decode6.712846986881186e-7 and denoise maximum4.7122170492796403e-7.
Generation target encoding is detached;reconstruction and generation shared-block gradients are nonzero;generation uses class/count/noise/settings without source image access.
Fixture uses native256 identity then isolated64/M3 with zero optimizer updates.
Proof:campaign/locks/essential-checks.json,SHA256664e75b27d64241135078dbc8b6a78d9046a90cfa80743ecd74a0c535d0a0368.
Validation results SHA256ccbcc70b2c5d809ae888047c9f38c040c30834c4a7aaf766f75a025f48a31426.
Native steady batch16 took2.91663317498751s;50000-generation-only projection2.531799631065547h,complete forecast2.5–3.5h.
Full fixture peak allocation1956945920bytes.
Prepared full reference:campaign/configs/reference-c89e740d3d08.yaml,SHA256c89e740d3d085426b1a75878c5da0d2bfa7db39d4e5d3c5aa370f2c1d81fc517.
Request1H100,4CPU,24GiB,4h;forecast advisory under v1.1.
Endpoint remains5000native reconstruction with rFID/L1/AlexLPIPS plus independent balanced50000generation with unchanged source FID/IS.
No full quality metrics exist yet;this is correctness and throughput evidence.
Actual new reserved GPU cost49/3600device-hours includes failed12s parser attempt plus37s validation;legacy RTX cost remains separate.
No A2 training harness exists or is being written.
PrefixAdapter correctness methods and check_adapter.py are available for independent CPU-only preparation in a separate source/branch.

## 2026-10-24 A1 full-reference submission
Job2831095 accepted under attached sbatch--wait ownership by unite-reproduce-20261023;automatic requeue disabled.
Execution source51a29b63aaba705a23a9de0e39ed8818b74a77ec;configSHA256c89e740d3d085426b1a75878c5da0d2bfa7db39d4e5d3c5aa370f2c1d81fc517.
Stage:/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_080942-76bfe1-51a29b6-code.
Output:/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/A1__UNITE_B_REFERENCE__s20261023__c89e740d3d08.
Request1H100,4CPU,24GiB,4h;vision-shared-h100,shared-if-available,accountvision-torralba-urops-meng,literal--export=NIL.
Fresh inventory:outputs/gpu-inventory-20261024T080912Z.json;all permitted partitions/compatible nodes and own account queue inspected.
No own GPU job running at submission;other B2/C1 jobs pending on A100/RTX4090 were left untouched.
Native generation uses unchanged third_party/unite/engines/eval.py:evaluate_fid and source utils/fid.py feature/FID/IS functions.
Instrumentation retains feature/logit arrays,NFE,timing and progress;it does not replace the source metric implementation.
5000 fixed reconstructions with same-subset rFID/L1/AlexLPIPS precede independently reseeded50000generation.
Submission is not a quality result or A1 acceptance;all numerical quality values and actual reservation cost await terminal evidence.

## 2026-10-24 Section5.11 component accepted
Coordinator independently reviewed complete proof/source/result hashes and accepted Section5.11 only.
Acceptance SHA2563288aec8958acd9b9153b2ddc7e32444703abec693a0edc40decc9b768dddaa0;retained copy:outputs/adapter-acceptance.json in the campaign source.
Producer job2830982,sourcee80c229dc4bd494e0eabf0c6bdeb4dc936fdbf19,resultsSHA256ccbcc70b2c5d809ae888047c9f38c040c30834c4a7aaf766f75a025f48a31426.
The FP32,TF32-off fixture has zero optimizer updates;it does not establish BF16 training or full A1 reference quality.
Full-reference job2831095 remains this worker's sole submission and terminal responsibility.
Independent CPU-only A2 preparation was assigned from stable51a29b63aaba705a23a9de0e39ed8818b74a77ec in retained joint-adaptive-tokenization-a2-20261024,branchresearch/jat-a2-20261024.
That independent preparation permits no GPU/main updates and cannot modify this source or canonical notebook.
No A2 harness implementation is needed or underway in this A1 lane.

## 2026-10-24 Expanded resource assessment and controlled replacement prepared
The first occupancy statement covered only newer classes;the expanded permitted node inventory also covers RTX3090,RTX2080Ti,TitanRTX and V100.
Retained inventory:outputs/gpu-older-inventory-018.json;assessment:outputs/gpu-reassessment-018.json.
One healthy RTX3090 GPU was free on torralba-3090-3;18RTX2080Ti,11TitanRTX and44V100 slots were also free.
Older Turing/Volta have installed CUDA kernels but cannot execute nativeTF32;their full reference runtime is unmeasured.
Select one RTX3090,SM86,24GB,nativeFP32/TF32,with measured source Base batch16 timing10.893116650986485s.
Retained measured receipt:UNITE-released-20261023-results/gpu-diagnostic-2826865/base-smoke-r1/results.json,SHA2564432d4abd8f9a4a4b6be9e237fe0467197e06350a302df8c57567ac79ec470a8.
That smoke used seed0,current run uses20261023;it is timing evidence,not a quality result. Torch/CUDA/source/checkpoint/CFG2/M14/native interval/batch16 match;metric version differs but is not invoked in the timed generation call.
Generation-only projection9.455830426203546h;conditional128NFE sensitivity11.003148132309581h;complete forecast9.8–11.5h;request12h.
No theoretical device conversion is used. Forecasts remain advisory.
Prepared configuration:campaign/configs/reference-c43d8bbaa2d5.yaml,SHA256c43d8bbaa2d5bd2e936590e9d9f439f16a1fa34f49b8307d4132776bede7795b.
Request1RTX3090,4CPU,24GiB,12h,vision-torralba-rtx3090,account/QoSvision-torralba-urops-meng,literalNIL,attachedsbatch--wait,no-requeue.
Checkpoint,seed,sampling,reconstruction,data,metrics,environment and gate are exactly equal to the H100 configuration;tested execution files are unchanged.
Only metadata resolver adds the RTX3090 hardware enum and24GB mapping.
Section15.2 reason,old/new endpoint/config references,unaffected controls,forecasts and terminal-cost placeholders are retained in outputs/controlled-replacement-018-plan.json and the new config.
H100 job2831095 remains pending until a fresh state check;cancel only this owned pending job,record terminal zero-GPU accounting and owner exit,then stage and submit one replacement.
Do not submit duplicate jobs or run a new benchmark campaign.

## 2026-10-24 H100 request terminal and separate phase memory accounting
Owned H100 job2831095 was confirmedPENDING and cancelled with --state=PENDING --user=kwen1.
Terminal sacct:CANCELLED by28131,0:0,ElapsedRaw0,AllocCPUS0,no AllocTRES,no assigned node;0GPU-hours and0samples.
The attached owner exited0 before replacement submission;original stage/submission/report/results are preserved.
Coordinator019 requested separate actual reconstruction and generation memory peaks.
The evaluator adds two torch.cuda allocator high-water reads and one stats reset between those phases.
Exact reversal of these three metadata insertions restores the accepted evaluator bytes;AST parses;no numerical,RNG,model,sampling or metric change.
Provenance delta:campaign/locks/reference-instrumentation.json;accepted evaluator3df4f29314f049b2fb731bb10e0da34657dd098b7c86642d71aa3afb4ee5c519;instrumented5cc773112e67b70c7215b050b6922fee8ac27c63d12bfe40129a23df48bb9a0c.
The accepted adapter/check source files remain byte-identical. Original Section5.11 proof keeps its producer identity.
Final prepared RTX3090 config:campaign/configs/reference-aeafc85486b8.yaml,SHA256aeafc85486b84d51295c99e8b3d83aa7f50695759c98045e6762a7730983b94f.
The prior preparedc43d8bbaa2d5 config was never submitted;it remains retained.
After terminal execution,compute overall allocator peak as the maximum of the recorded reconstruction and generation phase peaks;the payload's post-reset unqualified peak covers generation.
No A2 training or new GPU benchmark is added.

## 2026-10-24 Full reference2831212 running;5000 reconstruction complete
Replacement job2831212 accepted and running since2026-10-24T08:46:36Z;attached owner remains unite-reproduce-20261023.
Stage:/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_084608-ede7a1-8b3c14d-code;execution source8b3c14d132768edf4c760fba3e0c28e5bf3cdec9.
Configaeafc85486b84d51295c99e8b3d83aa7f50695759c98045e6762a7730983b94f;resources1RTX3090,4CPU,24GiB,12h.
Actual node:torralba-3090-3;NVIDIA GeForce RTX3090,UUIDb731d69e-543b-e2ec-1b49-6a4c2ba63c59,VRAM25296044032bytes,SM86,CUDA12.8,driver580.178.04,Torch2.10.0+cu128,fidelity0.3.0.
Actual environment/build,RAM allowance and retained storage mounts passed before model execution.
Runtime record:outputs/A1__UNITE_B_REFERENCE__s20261023__aeafc85486b8/runtime.json,SHA256a190e00e172dc1647faa848a3862cbe16025edb22484cddf7d1a735e43c14995.
BaseEMA strict load passed:152 compiled-key normalizations,no missing/unexpected keys,epoch240,step316000,CFG2.0.
5000 prescribed reconstructions finished in114.07195800699992s.
rFID4.15449338310998,L1mean0.0339586476677563,L1p900.056375064328312886,externalAlexLPIPSmean0.08381803412204608.
Real/reconstruction Inception feature SHA256:20ddeca1b2463c27594e9bb4b07c762941a98eeaf1ee2ff26d1bfcc20ec409c7/d2edd761664656f48215b3d59cf306fdd6afbe33e67d1d171664a63b7a874013.
Reconstruction phase peak allocated2792183808bytes,reserved4357881856bytes;79native reconstruction batches,one GE forward each.
Phase snapshot:outputs/A1__UNITE_B_REFERENCE__s20261023__aeafc85486b8/reconstruction-phase-snapshot.json,SHA2563bfb90f21b8ae041fd20631199dc02abb1403121d894c56a1c7439999bfe611b.
First16 reconstruction pairs and first16 independent generation samples were visually inspected through in-memory previews;reconstructions preserve the shown objects with some fine-detail smoothing.
This limited grid is not a quantitative full-dataset quality claim.
The5Ksubset has no matching published rFID value;paper120epoch reconstruction differs from released240epoch Base.
50000 independent balanced generation remains active;FID/IS and actual terminal reservation cost are pending.
No full A1 pass or training result is claimed.

## 2026-10-24 A2 implementation preparation: PREPARATION_ONLY

- Branch research/jat-a2-20261024; parent51a29b63aaba705a23a9de0e39ed8818b74a77ec; tested code a367fca5a1cb9b597826d5ffda26d5d1d0c13cca.
- Source /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024. Approved fork https://github.com/kw7243/UNITE-tokenization-generation; official upstream2389f2c65fedbd43ada851e69b2b810efd23af7d is an ancestor.
- Plan1.1 SHA256 4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5.
- Report /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/report.md; results /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/results.json; terminal receipts /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/receipts/terminal-summary.json; final backup receipt /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/receipts/backup.json.
- Final stage /data/scratch-fast/kwen1/jat-a2-20261024-stages/20261024_044035-153234-a367fca-code. All earlier stages and outputs are preserved.
- Implemented native16/64 joint and conditional RECON recipes,exact class-balanced paired streams,source M3 loss/optimizer partition,BF16/FP32 protocol,update LR/EMA,complete optimizer-boundary snapshots,monitoring,DEV evaluation,and last-block probes.
- No selector,prefix training,count embeddings,B1 converter,KARL/ALIT changes,scientific features,or quality scoring.
- Nine focused CPU check groups passed on three completed4-core jobs:2831152,2831187,2831201. Total 0.423333333 actual CPU core-hours. No GPU submissions or main-model updates.
- One normal-QoS CPU request was rejected before allocation; csail/tig-main/tig-cpu matches partition policy and the completed pilot-data precedent. All accepted requests used literal --export=NIL. No allocated job failed.
- Read-only environment /data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1; package-lock SHA256 fbc44e755ea7ab9c1fb95f88783678b7c57fc8a367086054903fbfc2617ef253; freeze--all SHA256 41b5c4ee15094b54d9151687fb091d654ed6ca9d0adc53612d6e46218668010f. No environment mutation.
- Resolved config hashes: {"NATIVE16_JOINT": "261f77693650343762115c0aa37fb11343e78e1df4a587189e71bbd6b4c4ccbb", "NATIVE16_RECON": "19be684784a5639c729b81031fbb28a2a949e7f80ad6236086be567a8466bdb8", "NATIVE64_JOINT": "68c8a5bdd512f79815862854eea5a0945ffaccee2e594b82c65f521188396c17", "NATIVE64_RECON": "92b3c45664d57db62768dc5b269c837fed1f9fe8f4f58101db9bc5f49982481c"}.
- Code bundle 2599d8857a1a2914f77525a59db782a7796445cfc2898143bd6d03e84b080a55; tested patch 7642a8d589cd841fe93a82f930d989c88acb35e26854e09ab2eb6530395f9dff; metric protocol 20acc43b9f3498e092590daddbb56184a7d09e2fe3b4fe7f9812cd36189bd873.
- Interfaces/schema /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/campaign/a2/interface.json; launch/remaining checks /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/campaign/a2/README.md.
- A1 full-reference acceptance still belongs to unite-reproduce-20261023. Repeat affected native GPU correctness/BF16/DDP/memory/full-checkpoint checks before the separate training assignment. The unchanged accepted256 proof remains reusable until its affected source changes.
- Scientific A2 is unrun,not passed or negative. CONFIRM is sealed. B1 still needs a future stable NATIVE64_JOINT complete checkpoint. Pilot FIT features/hash remain an endpoint-metric prerequisite.
- Canonical notebook writer: unite-reproduce-20261023. This entry is a handoff only; this worker did not modify main or its canonical notebook.

- Accepted preparation backup: [c80973b1b3f30a139c6a633325e0ec23c1173db2](https://github.com/kw7243/UNITE-tokenization-generation/commit/c80973b1b3f30a139c6a633325e0ec23c1173db2); tested code: [a367fca5a1cb9b597826d5ffda26d5d1d0c13cca](https://github.com/kw7243/UNITE-tokenization-generation/commit/a367fca5a1cb9b597826d5ffda26d5d1d0c13cca).
- Accepted report SHA256 9110425f5c34b20c53ca5600303cbca3e2c10b6354c7442b39876d1a10dbfff1; results SHA256 8e8845c6dadb702713c91fdf1895aa8a18747e58566e11940435460b988ab1f0; terminal summary SHA256 601e9f786d13f9d5f8367010c1d1bf78a276aa339c45fea58cb608d51342a221.
- Coordinator acceptance is PREPARATION_ONLY, scientific_a2_pass=false. Retained administrative copy in the A1 campaign: outputs/a2-preparation-acceptance.json, SHA256 8047616f53a138ee790b710fcdfb5e8db44409f436741a753766504f774d5f31.

## 2026-10-24 Full UNITE-B reference complete: native FID-50K 2.103550117192

This entry supersedes the earlier nonterminal2831212 snapshot.
The complete5000reconstruction and independent50000generation workload completed0:0; no training or A2 scientific result is claimed.
The predeclared practical CHECKPOINT_EFFECT_REPRODUCTION candidate passes gFID<=3.1, strict loading and intact finite reconstruction.
Coordinator acceptance is recorded separately from the producer evidence.

| Metric | Actual | Paper Table1 | Absolute difference | Paper appendix | Absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| FID-50K | 2.1035501171916167 | 2.12 | 0.016449882808383443 | 2.735 | 0.6314498828083832 |
| IS | 295.81353759765625 | 294.1 | 1.7135375976562273 | 268.1 | 27.713537597656227 |

IS split standard deviation3.021472454071045 describes ten contiguous splits; it is not a confidence interval or training-seed uncertainty.
No author RNG state, matching paper uncertainty or additional fidelity threshold is supplied.
This supports the released-checkpoint reference effect under the native protocol; it does not settle the headline/appendix discrepancy.

- Job2831212: COMPLETED0:0,2026-10-24T08:46:36Z–19:19:42Z,torralba-3090-3.
- Sole attached owner exited0; all this assignment's jobs are terminal.
- Resources:1RTX3090,4CPU,24GiB,12h; actual reservation37986s=10.551666666666666RTX3090-hours and42.20666666666666CPUcore-hours.
- Payload runtime37976.45966713503s; generation pipeline37850.255699014s; native sampling/decode37785.58664617827s.
- Generation at batch16:755.7117329235655ms/image,1.3232558877065133images/s,meanNFE121.99424,total381232.
- Host MaxRSS10014372KiB.
- Reconstruction CUDA peak allocated2792183808B,reserved4357881856B.
- Generation CUDA peak allocated1895939584B,reserved2774532096B.
- Whole-run allocator peak=max(phases)=2792183808B. Original unqualified final payload peaks covered generation only after the declared reset; original results.payload.json/report.payload.md remain byte-preserved. This metadata correction changes no metric.
- New A1 allocations including setup/validation:10.565277777777778heterogeneousGPU-hours and44.0325CPUcore-hours; legacy1740GPU-seconds remain separate. No hardware conversion or sibling/A2 cost is added.
- Reconstruction5000:rFID4.15449338310998,meanL10.0339586476677563,p90L10.056375064328312886,meanexternalAlexLPIPS0.08381803412204608;114.07195800699992s.
- Same-subset rFID has no matching published target; paper120epoch reconstruction differs from releasedBase240epoch.
- All5000per-image rows,50000features/logits,3125generation batches and50labels/class are verified complete and finite.
- Section5.11 FP32 correctness is accepted independently; zerooptimizerupdates. Its original proof source remains e80c229dc4bd494e0eabf0c6bdeb4dc936fdbf19.

Execution uses releasedBaseEMA256/d32/M14,epoch240,step316000,savedCFG2.0,seed20261023,FP32/TF32 and compiled default decoder.
Native dopri5[0,1],atol1e-6,rtol1e-3,50outputtimes,shift0.5,norm_first,CFGinterval[0.1,1],batch16/world1.
Reconstruction uses first5lexicographicvalidationfilenames/class,sourcecrop/no flip/clean path.
The unchanged training evaluation engine and source FID/IS implement the numerical path.
Source[0,1] differs from paper prose; only savedCFG2.0 was evaluated.
No shortened sample count, alternate FID implementation or from-scratch training is substituted.

Executed source: [8b3c14d132768edf4c760fba3e0c28e5bf3cdec9](https://github.com/kw7243/UNITE-tokenization-generation/commit/8b3c14d132768edf4c760fba3e0c28e5bf3cdec9).
Official upstream:2389f2c65fedbd43ada851e69b2b810efd23af7d; research parent:a19c1501ed9d28e251a90950349bf2a739c9a9f3.
Stage:/data/scratch-fast/kwen1/joint-adaptive-tokenization-20261024-stages/20261024_084608-ede7a1-8b3c14d-code.
ConfigSHA256:aeafc85486b84d51295c99e8b3d83aa7f50695759c98045e6762a7730983b94f.
BasecheckpointSHA256:7b7f40dd37111196cad6e4401bf9737156268612e001c29cb0a760ae416bff4b.
ManifestSHA256:d31e573bba422ea6626552e1b2e717bffbabf2a9f6adea777514e5ff85085b0d;metricprotocolSHA256:693aa2cca24f2ae03238cef3453b9b6f95f39120da9a2575b04415afc87e0971.
GenerationfeaturesSHA256:d743a2a201da79174fa41c6d4d01cf00a44625219fd66916fdd0d4d9e39feecd;logitsSHA256:bb6f7a592aa312b3d3e2eb2621919fd62137346fd4a95c7ddf780daed5ad923d.
Terminal payload resultsSHA256:ead270fb289596be14b5de28c2ee191cdec8c60b5b445b2dde52dfc0f6e644db.
Run reportSHA256:572c26746cddf4a344142357d6c05ee6530c9b3ec865b4271023e00161096e49;resultsSHA256:2251d31767c06e98622fb4942ff753d81240b39d74d18b6edeb917490663012d;artifactindexSHA256:efcf7e0e92c6deb1e68e74ddf6034500b08bfabf01681415cd564d95e284b682.
A1 aggregate reportSHA256:7493d5c3d3e4e4c1167a18b3be679bb58117d45cb26d34cd2f516a80ba64f0dd;resultsSHA256:925a099bc4aa3e2cabdb33fb8a8a7a1a50a95388f1b014136f028ba13c344113;artifactindexSHA256:9e4f8e6392654d39a45f2605b469766743a2ccd34f3fe990a4357a825c47e66f.

Retained complete node bundle:/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/A1.
Retained full run:/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/A1__UNITE_B_REFERENCE__s20261023__aeafc85486b8.
Exact commands and resources:submission.json,argv.shell and campaign/scripts/reference.sbatch.
Per-image values,feature/logit arrays,grids,failed attempts,terminal-accounting.txt,terminal-receipt.json,terminal-verification.json and artifacts.json are retained.
Environment: /data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1;Torch2.10.0+cu128,CUDA12.8,torch-fidelity0.3.0;completepackage/build/metricweight locks remain linked.
Paper: https://arxiv.org/html/2603.22283v1.

| Branch index | Scope | Retained result |
| --- | --- | --- |
| research/joint-adaptive-tokenization-20261024 | This A1 reference and accepted adapter checks; parent a19c1501ed9d28e251a90950349bf2a739c9a9f3 | /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024/outputs/A1 |
| main | Canonical notebook-only publication | Notebook; no experiment-code merge |
| research/jat-a2-20261024 | Separate accepted CPU preparation; backup c80973b1b3f30a139c6a633325e0ec23c1173db2 | Separate A2 source/outputs; scientific_a2_pass=false |

KARL and pilot data remain separate accepted evidence; CONFIRM stays sealed.
Large checkpoint inspection remains recorded; its actual113epochs differ from advertised120. FullLarge50K is outside this later boundedA1-B scope.
The full-training scale remains8H200140GB at26minutes/epoch,about832H200GPU-hours for240epochs; no such run was started.
Next action: coordinator acceptance of A1, then scope a separate A2 GPU preparation/training assignment.
This worker submits no A2 job and imports no A2 experiment code.
