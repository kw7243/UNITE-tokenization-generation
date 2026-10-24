# B1 SCREEN preparation

The implementation covers PREFIX16, PREFIX64, and NESTED_RANDOM only.
B1 is NOT_STARTED. No SCREEN_PREFIX or COMMON_PREFIX exists.
All three checked-in configurations are disabled drafts. Trained input paths, hashes, and acceptance are null with reasons.
No GPU allocation, scientific conversion, scientific optimizer update, feature extraction, or quality evaluation is part of this preparation.

## Source and interfaces

- Parent: `7a492f54a1974931d5d603b14b150263a8891f44`; accepted code test: `c3eafe8ffa8b809a92b18af429003ff199bffe00`. Only notebook.md changed between them.
- `screen.py` extends the accepted project prefix adapter. Official imported sources and the A2 implementation stay unchanged.
- `Screen.encode(images,noise,times)` has no selected-count argument. It processes64registers once.
- `Screen.forward(images,labels,counts,draws)` physically groups16/64prefixes and restores image order. Losses are per image; M3 draws are summed.
- Three zero tables use rows[4,16,64]. Denoiser length is added to the source time/class vector, independently of class dropout. Context stays class-derived. A constant group count reaches both source CFG halves. Decoder tables follow both source projections. Square-grid interpolation is unchanged.
- Source Muon/AdamW selectors supply every old parameter in the same order. Exactly three new tables are appended to main AdamW, with zero moments/step and EMA copies. Frozen source VGG LPIPS stays in eval mode after train().
- `counts.py` restores every A2 noise stream and adds only `b1_screen_assignment`. Each global64image microbatch independently permutes32counts16and32counts64. The optimizer batch has[0,128,128]; E[K]=40,E[K²]=2176. All arms consume matched full64noise before grouping.
- `convert.py` validates coordinator acceptance of actual hashed A2 bytes, a stable trained native64 joint control at exactly5000completed updates, source/config history, model/EMA/named optimizer/RNG/sampler/exposure state, and finite tensors. Engineering COMPLETE labels alone are refused.
- The pure converter can process an explicitly synthetic fixture in the CPU test. Its output role is SYNTHETIC_B1_SCREEN_START. It cannot pass the real input loader or scientific resume path.

## Reproduce the CPU preparation

Keep the locked environment read-only. From the retained clone:

```bash
python3 campaign/b1/resolve.py
reproduction/tools/stage_and_run.sh \
  --staging-parent /data/scratch-fast/kwen1/jat-b1-screen-prepare-20261024-stages \
  --stage-only
```

Use the printed stage path. Record its STAGING_MANIFEST.json and REPRODUCIBILITY_METADATA.json hashes, the source revision, environment lock, selected configuration hashes, exact command and resource request before submission.
From that stage, with a new retained output path:

```bash
env -u SBATCH_EXPORT sbatch --export=NIL --parsable \
  --output="$NEW_RETAINED_OUTPUT/slurm-%j.out" \
  --error="$NEW_RETAINED_OUTPUT/slurm-%j.err" \
  campaign/b1/cpu.sbatch "$NEW_RETAINED_OUTPUT/checks"
```

The CPU request is2CPUs,8GiB,10minutes,tig-cpu,accountcsail,QoStig-main,and no GPU.
The job records the node, visible devices, storage, environment versions and unchanged package content.
Retain every stage and attempt. Reconcile sacct terminal state, exit code, AllocTRES, elapsed time and MaxRSS.
The synthetic optimizer fixtures do not call optimizer.step(). No main model update is permitted.

## Required later binding

The coordinator must supply a new resolved configuration for each arm. Keep these drafts intact.
Bind `input.checkpoint`, its actual SHA256, and `input.acceptance` to the same accepted A2 artifact across all arms.
The acceptance JSON must contain:

```json
{
  "synthetic_fixture": false,
  "a2_native64_accepted": true,
  "stable_control": true,
  "scientific_training_completed": true,
  "checkpoint_sha256": null,
  "a2_configuration_sha256": null,
  "a2_patch_sha256": null,
  "source_config_history": null
}
```

The null fields above must become actual identities, not placeholder strings.
Each history row contains source_revision,config_sha256,patch_sha256; the final row must match the snapshot.
The source config and complete state must match the proposed B1 source, optimizer partition and64candidate model.
Set execution_authorized=true only in the new accepted configs. Hash their final bytes and prepare an exact execution receipt:

```json
{
  "allow_b1_screen": true,
  "configuration_sha256": null,
  "patch_sha256": null,
  "plan_sha256": "4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5",
  "gpu_check_source_patch_sha256": null,
  "gpu_checks": {
    "register_identity": false,
    "mixed_count_fp32": false,
    "mixed_count_bf16": false,
    "gradient_boundary": false,
    "ode_cfg_image_free": false,
    "ddp_image_reduction": false,
    "real_checkpoint_resume": false
  }
}
```

No current receipt or trained checkpoint exists. The tables above are schema examples only.
The GPU checks must exercise real Base/Base mixed16/64 FP32/BF16 paths, unequal/empty groups, original register indices and source context, detached target/input and both shared GE gradient paths, actual ODE/CFG with image inputs forbidden, four-rank accumulation, all new parameters in AdamW/EMA, and complete accepted-A2-to-B1 resume including next images/flips/noises/count stream.
Use FP32 relative L2<=1e-5; BF16<=5e-3. Repeat a failed BF16 check in FP32.
A real-checkpoint conversion is not proved by the synthetic fixture.

## Future training commands - disabled until binding and GPU proof

Inspect current permitted Slurm inventory, memory fit, homogeneous GPU availability and exact environment before allocating.
The drafts state the default4H10080GB layout. An alternate device/layout needs a new matched configuration and affected checks; do not silently use A2's earlier RTX3090 setup.
Stage the complete final source again with the project helper. In the separately authorized allocation, from the stage, run each arm with its own bound config and receipt:

```bash
torchrun --standalone --nproc_per_node=4 campaign/b1/train.py \
  --config "${B1_BOUND_CONFIG:?}" --authorization "${B1_EXECUTION_RECEIPT:?}" --execute
```

The command refuses every checked-in draft, even if --execute is supplied.
The initial endpoint is5000continuation updates at executed LR1e-4. A2 warmup is not restarted.
Record actual throughput/memory over the first200real updates. CPU preparation establishes neither fit nor throughput.
All arms share the exact A2 snapshot and image/augmentation streams.
Scheduled512DEV checks run every1000updates at both16and64. The coordinator selects the common endpoint under Section8.1.
If required, continue all three to10000with new resolved configs, explicit original/revised hashes/endpoints, affected controls, scientific reason and additional expected cost. Preserve original results.
Use `--resume "$OWN_COMPLETE_B1_SNAPSHOT"` with the bound extension config/receipt. The previous config hash must equal extension.original_config_sha256.
If an interrupted attempt exceeds the saved exposure, `--ack-replay-through N` must name the exact highest attempted B1 update.
Further matched extensions follow Section15.2. Estimated GPU-hours do not stop useful work.

## Future endpoint commands - disabled until accepted common endpoint

Run full3200DEV at the selected common endpoint with EMA and the same seed. Add `endpoint_checkpoint_sha256` to the exact execution receipt:

```bash
python campaign/b1/evaluate.py --config "${B1_BOUND_CONFIG:?}" \
  --checkpoint "${OWN_COMPLETE_B1_SNAPSHOT:?}" --authorization "${B1_ENDPOINT_RECEIPT:?}" \
  --execute --mode reconstruction --output "${NEW_FULL_DEV_OUTPUT:?}"
```

For endpoint generation, also require common_screen_endpoint_accepted=true and fixed count16or64:

```bash
python campaign/b1/evaluate.py --config "${B1_BOUND_CONFIG:?}" \
  --checkpoint "${OWN_COMPLETE_B1_SNAPSHOT:?}" --authorization "${B1_ENDPOINT_RECEIPT:?}" \
  --execute --mode generation --count "${FIXED_COUNT:?}" --output "${NEW_5K_OUTPUT:?}"
```

PREFIX16 generates at16,PREFIX64at64,and NESTED_RANDOM separately at16and64: four paired5K sets.
There is no natural-mixture count-prior fit. There is no selector. Count4is not a trained operating point.
The implementation writes truncating uint8pixels, common input identities and actual source forward counts. Feature extraction/scoring remains a separate endpoint workload using metric-protocol.json, the exact locked Inception extractor and verified pilot FIT feature manifest.
Reuse the existing pilot FIT artifact after manifest/hash verification. Do not substitute standard1Kstatistics or duplicate FIT extraction here.
Use shared gFID/KID features, paired KID subset streams, clamped external AlexNet LPIPS, N-1/FP64 source FID, and the locked dopri5/CFG2sampler.
The coordinator applies Section8.1 gates, including stability, per-count10%usability margins, KID direction and positive tail gain. A passing usability result alone does not prove H_NESTED.
Any required compute-matched fixed-control extension is a separate result; it cannot alter the saved equal-exposure comparison.
Only the coordinator can assign SCREEN_PREFIX after scientific acceptance. COMMON_PREFIX additionally requires the later RATE phase, which is outside this implementation.
CONFIRM remains sealed. PAIRED,RATE,C2,and later nodes are not implemented.
