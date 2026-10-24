## 1 Objective

Preparation PASS. Scientific B1 SCREEN is NOT_STARTED.
Prepare PREFIX16, PREFIX64 and NESTED_RANDOM for the approved plan v1.1, without starting experiments.
H_NESTED remains untested. Scientific acceptance requires Section 8.1 stability and per-count reconstruction/gFID usability margins, with KID and tail-gain evidence as specified in the plan.
No SCREEN_PREFIX or COMMON_PREFIX exists.

## 2 Setup

- Retained repository: `/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024`.
- Research branch: `research/jat-b1-screen-prepare-20261024`.
- Exact parent: `7a492f54a1974931d5d603b14b150263a8891f44`. Parent code was tested at `c3eafe8ffa8b809a92b18af429003ff199bffe00`; only notebook.md changed afterward.
- Tested and backed-up implementation revision: `1b0f251ef3552592d4f407cb156562c07354d71f`. The later delivery commit adds evidence/notebook text only.
- Source-manifest patch SHA256: `4a5060a678f63046994c4abd3c9e5f4fb3ba0dbd027a287bf74d0e2c64a61213`.
- Literal implementation patch SHA256: `8f7922192dd9a00c1072c108d2b2175c1cefcb0897b9e492bcee2d663510ffd6` at `/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024/outputs/b1-preparation/implementation.patch`.
- Fork parent verified with gh-axi: ShivamDuggal4/UNITE-tokenization-generation. Origin/upstream are recorded in remotes.txt. Imported official bytes, A2 sources, A2 outputs, canonical clone and main notebook remain unchanged.
- No installation. Locked environment read-only at `/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1`; lock SHA256 `fbc44e755ea7ab9c1fb95f88783678b7c57fc8a367086054903fbfc2617ef253`. The final compute-node package freeze matches the locked content after excluding the two already-present bootstrap tools from --all.
- Data: existing FIT/DEV/ALLOC manifests and100-class mapping were hashed/read. No image bytes, CONFIRM rows or quality features were read.
- CPU-only Slurm jobs used groenig-3; no GPU allocation. Final request:2 CPUs, 2 GiB, 5 minutes, tig-cpu, account csail, QoS tig-main. The final measured batch MaxRSS was352012K.
- Every stage is a full repository copy, including Git state, made with the unchanged project-local rsync-compatible helper. Stages and all failed outputs are retained.

| Method | Config SHA256 | Scientific status | Quality |
|---|---|---|---|
| PREFIX16 | `ef900bdb78f6cc933859fedf727739f74013bc1a0487554cfa62fd5ca607bdcd` | NOT_STARTED | null |
| PREFIX64 | `25a032b18a2469e27fa04bfcfaeeb62018162a4208f5517483df1bd4a2985572` | NOT_STARTED | null |
| NESTED_RANDOM | `d89834e31cc21d944b240a32580b49c9a4ddbdd5c502272413e3aa3551c6e6cb` | NOT_STARTED | null |

Trained A2 input, its actual SHA256, accepted configuration/history, acceptance and GPU receipts are null with reasons in results.json.
Pilot FIT feature identities also remain null pending later manifest verification; no extraction was duplicated.

## 3 Baseline

All arms encode each image once with 64 candidate registers. PREFIX16 is not A2 native16.
All receive the same three zero length tables in [4,16,64] order, source Muon/AdamW partition, zero new moments, EMA capacity and inherited A2 state.
The implementation preserves full 64-position random draws, image/flip order, M3 source-autograd draw sum, per-image token/channel normalization, corruption and detach boundaries, original positions/context and physical square-grid interpolation.
NESTED_RANDOM permutes[0,32,32]within each 64-image distributed microbatch, independently of image content. Its optimizer batch has[0,128,128],E[K]=40,E[K²]=2176.
Fixed controls and nested inference counts are separate matched-exposure comparisons. No equal-rate or compute advantage is established here.

## 4 Results

The final source passed seven focused check groups in job2834485. All own CPU jobs are terminal.

- `configs_sources_data_sealed_confirm`: PASS. 
- `exact_quota_independent_stream_replay`: PASS. 
- `synthetic_grouping_order_loss_detach`: PASS. Synthetic blocks, not full-model numerical proof
- `synthetic_length_class_dropout_cfg_boundary`: PASS. 
- `synthetic_ddp_image_weighted_math`: PASS. 
- `full_model_meta_optimizer_ema_capacity`: PASS. Metadata tensors and frozen LPIPS fixture; no full-model CPU/GPU forward
- `synthetic_strict_conversion_refusals`: PASS. 

| CPU job | Terminal state | Exit | Wall seconds | CPUs | Requested memory | Batch MaxRSS | Reserved core-hours |
|---|---|---|---:|---:|---|---|---:|
| 2834299 | FAILED | 1:0 | 80 | 2 | 8G | 825644K | 0.044444 |
| 2834302 | COMPLETED | 0:0 | 60 | 2 | 8G | 351284K | 0.033333 |
| 2834485 | COMPLETED | 0:0 | 53 | 2 | 2G | 352012K | 0.029444 |

Total reserved CPU cost: 0.107222222 core-hours over 193 allocated job-seconds, including the failed attempt.
TotalCPU and full Slurm accounting remain in each attempt directory. GPU-hours=0; scientific updates=0; scientific images_seen=0.
Every quality metric is null with a reason. Synthetic fixtures are not model experiments.

Issue/fix: attempt1 stopped on absent unused KARL files. The validator was narrowed to the applicable UNITE source, following A2's accepted boundary; inherited files were preserved.
The first separate submission receipt was written about 4 seconds after batch start, with source/environment/script bytes already staged. Attempts2/3 saved exact argv, environment and artifact identities before submission. Final PASS uses attempt3.

## 5 Interpretation

The CPU evidence supports implementation preparation: exact grouping/order/reduction, quota replay, nonzero length routing through dropout/CFG and decoder projections, source boundaries, explicit parameter registration, and strict synthetic named-state conversion/refusal.
The full Base/Base metadata check confirms source parameter membership plus 6912 new parameters, identical across the three arms.
The converter preserves old named state and adds zero new AdamW moments and EMA tables. A generic COMPLETE role alone is refused.
This evidence does not establish H_NESTED, real training stability, generation quality, GPU correctness, memory fit or real checkpoint resume.
The synthetic fixture deliberately contains manually constructed 5000-update fields. Its role and artifacts are explicitly SYNTHETIC; they cannot enter the real scientific loader or resume path.

## 6 Limitations

No coordinator-accepted trained native64 checkpoint was provided to this preparation. There is no actual A2 input hash to report and no scientific conversion was performed.
Tiny synthetic blocks and metadata tensors replace expensive numerical model execution. They do not replace full-model FP32/BF16, shared-gradient, actual ODE, DDP or resume checks.
The CPU CFG fixture calls the source CFG control flow through a one-call sampler stub; it does not integrate an ODE.
Only manifest identities and synthetic tensors were used. No scientific image metric, training cost forecast, GPU memory fit or throughput measurement exists.
All draft configs are execution-disabled. Missing input and evidence fields are null with reasons, not invented hashes.
PAIRED, RATE, selector, C2 and later nodes remain outside this implementation. CONFIRM remains sealed.

## 7 Next action

Preparation PASS; scientific B1 NOT_STARTED. Hand ownership back to the coordinator for trained-input acceptance, remaining GPU checks and any later experiment assignment.

1. Coordinator accept stable NATIVE64_JOINT at 5000 completed updates, hash actual checkpoint bytes, bind exact trained source/config history and complete model/optimizer/EMA/names/RNG/sampler/exposure state.
2. Real Base/Base register identity and no patch leakage; retain accepted unchanged256identity proof, and exercise64candidate image pass.
3. Actual mixed 16/64 grouped/single-image and inactive-slot FP32 relative L2<=1e-5; BF16<=5e-3 with FP32 rerun on failure; unequal and empty groups, exact image reduction.
4. Actual shared-GE reconstruction and denoising gradients; detached clean/noised denoising target/input; M3 sum and k*32 normalization.
5. Actual source ODE and both CFG halves retain length after class dropout; no image/patch/feature inputs during generation; original positions/context and both decoder table placements.
6. Four-rank DDP with uneven/empty count groups, accumulation4, EMA inclusion, finite gradients/states and memory fit on declared hardware.
7. Accepted real A2-to-B1 conversion and complete optimizer-boundary resume: exact old named moments, zero new moments/EMA tables, next FIT IDs/flips/full64noise/source RNG and count-stream replay.
8. Coordinator bind new executable configs and exact GPU evidence/authorization receipt; inspect current permitted inventory; stage final full source before separately authorized launches.
9. Verify existing pilot FIT feature manifest/hash before endpoint metrics; do not replace with standard 1K stats or duplicate extraction here.
10. After launch only: matched 5000 initial continuation, scheduled 512 DEV/1000, common 10000 extension if Section 8.1requires, later matched Section 15.2 extensions, full 3200 DEV and required four paired 5K fixed-count sets. CONFIRM remains sealed.

Exact CPU reproduction and future disabled launch/evaluation commands are in `campaign/b1/README.md`.
Reproduce from the final recorded stage using the exact argv in `/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024/outputs/b1-preparation/attempt-3/submission.json` and a fresh retained output path.
The actual final test command was:

```bash
cd /data/scratch-fast/kwen1/jat-b1-screen-prepare-20261024-stages/20261024_170251-75d085-1b0f251-code
env -u SBATCH_EXPORT sbatch --export=NIL --parsable \
  --output=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024/outputs/b1-preparation/attempt-3/slurm-%j.out \
  --error=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024/outputs/b1-preparation/attempt-3/slurm-%j.err \
  campaign/b1/cpu.sbatch /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-b1-screen-prepare-20261024/outputs/b1-preparation/attempt-3/checks
```

The final output path exists and must be preserved; use a new output path for a repeat.
No experiment merge, PR, no-mistakes, force push, deletion or scratch cleanup occurred.
