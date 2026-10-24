# A2 native-count preparation

PREPARATION_ONLY. No GPU job or main-model optimizer update ran in this assignment.
A1 full-reference acceptance remains a prerequisite.
The auxiliary branch does not modify the canonical notebook or A1 source.

## Interfaces

- `data.py`: immutable FIT/DEV/ALLOC TSV loaders, supplied 100-class map, exact FIT queues, independent paired random streams, per-image evaluation noise.
- `native.py`: native16/64 image encoders and denoisers, source decoder, source Muon/AdamW partition, explicit update-based LR, image-weighted accumulation.
- `train.py`: four-rank BF16 training path with FP32 parameters/losses, monitoring, EMA, complete snapshots, and scheduled DEV evaluation.
- `evaluate.py`: clean EMA reconstruction, a32-image last-block gradient probe, the conditional interference ratio, generation eligibility, and common endpoint labels/noise.
- `metric-protocol.json`: exact later reconstruction/generation/feature/statistical contract. No scientific features or scores were computed here.
- `configs/index.json`: immutable resolved configuration paths, SHA256 values, and run IDs. Files are JSON, a YAML1.2 subset, and are read with `json.loads` to retain exponent types.
- `optimizer-structure.json`: exact ordered parameter names from source selectors, checked with full-size metadata-only model structures.
- `asset-lock.json`: existing source VGG, external AlexNet, and Inception weight identities. The released M14 UNITE checkpoint is not loaded for A2 initialization.

Every arm starts from the same seeded64-register construction.
Native16 slices only the trainable register positional table, then processes16 registers in every encoder/denoiser forward.
Both counts retain the same deterministic RoPE cache capacity; unused capacity does not process extra registers.
No latent prefix truncation, selector, or length embedding is used.
The accepted prefix adapter is unchanged; only its register-output correction is reused.

The default future comparison is4 H100 GPUs, microbatch16 per rank, accumulation4, global256, seed0, and5000 updates.
Warmup is500 optimizer updates, then constant executed LR1e-4.
Muon keeps its prescribed `match_rms_adamw` adjustment; the entry point adds no batch-size LR multiplier.
Reconstruction-only arms consume the same image/flip/noise streams and are conditional on the paired joint results.

## Resume and complete checkpoint

A snapshot is written after each1000 completed optimizer updates.
It contains live/EMA model state, both optimizers, ordered parameter names, schedule position, FIT queues/cursors/data RNG, every paired stream state, per-rank Python/NumPy/Torch CPU/CUDA RNG, evaluation history, exposure counters, configuration hash, and resolved configuration.
Snapshots are new files, with SHA256 sidecars; existing snapshots are preserved.
They are written only at a complete optimizer boundary.

`checkpoint.py` owns schema `jat-a2-checkpoint-v1`.
The64-token role is `A2_NATIVE64_COMPLETE`.
It contains the full state needed for a later B1 converter, but it is scientifically usable only after A2 has a valid stable64-token control.
No B1 converter or B1 model is implemented here.

Resume requires exactly the same configuration and four-rank layout.
It restores the same next image batch and streams.
The append-only exposure record retains every attempted/completed update and image ID, flip, noise digest, and exposure-chain digest.
If recorded attempts extend beyond the saved snapshot, resume refuses until `--ack-replay-through N` names the exact highest recorded attempt.
This makes repeated exposure explicit; it does not silently skip or repeat images.
Failed or interrupted attempts must also be counted in terminal cost/exposure reports.

## Checks before later training

The existing A1 Section5.11 proof applies to the unchanged prefix fixture, not to this new training path.
Before training, repeat the affected native16/64 paths in FP32:

1. Verify output shape `[B,k,32]` and no patch outputs. Retain the accepted256-token identity proof while its source remains unchanged; repeat it if the correction or source changes.
2. Verify batched/single-image decode/denoise/loss agreement at both counts, correct physical count and image reduction, and identical common initialized bytes with only the register table sliced. The mixed-count inactive-slot check stays in the unchanged prefix fixture; A2 has no inactive latent slots.
3. Verify detached targets/noised inputs, intended shared-block gradients, and image-free class/noise generation.

FP32 tolerance is relative L2<=1e-5.
For BF16 use5e-3 and repeat any failed comparison in FP32.
Check one real microbatch at the declared BF16 precision, memory fit, four-rank reduction, EMA, and full checkpoint round trip before committing to the paired run.
No such GPU check ran during preparation.
Measure throughput/memory from the first200 real authorized optimizer updates, not a separate benchmark.
The CPU tests use metadata-only full-size structures and tiny tensor fixtures; they do not establish large-model numerical correctness or quality.

## Later launch

These commands document the interface. This assignment does not authorize executing them.
The coordinator must first accept A1 and the native GPU checks, then issue the separate A2 assignment.
A later authorization receipt has these required fields:

```json
{
  "allow_a2_training": true,
  "a1_unite_pass": true,
  "gpu_native_checks_passed": true,
  "configuration_sha256": "the exact selected config SHA256",
  "patch_sha256": "source.patch_sha256 from that config",
  "plan_sha256": "4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5",
  "conditional_reconstruction_arms_authorized": false
}
```

The reconstruction-only configurations additionally require the last field to be true.
Keep source environments and canonical notebooks read-only.
Use the existing auxiliary asset links below as TORCH_HOME so source VGG lock files are local to this branch's output directory.

```bash
export PATH=/data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-assets/torch
export LPIPS_CACHE_DIR=/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs/lpips
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
```

Stage the full auxiliary repository with `reproduction/tools/stage_and_run.sh --staging-parent /data/scratch-fast/kwen1/jat-a2-20261024-stages --stage-only`.
Run from that new stage inside the later approved four-GPU allocation:

```bash
torchrun --standalone --nproc_per_node=4 campaign/a2/train.py \
  --config campaign/a2/configs/NATIVE64_JOINT.yaml \
  --execute --authorization /absolute/path/to/coordinator-receipt.json
```

Use the exact16-token joint configuration for its paired arm.
The code stops unless the configured default H100/BF16 layout is available.
An alternate device or distributed layout needs a declared matched revision and relevant checks; it is never a silent fallback.
Before any later submission, record exact argv/source/config/stage/environment/resources and inspect permitted inventory.
For `sbatch`, clear inherited `SBATCH_EXPORT`, use literal `--export=NIL`, and place environment assignments after every directive.
No GPU submission script was executed or submitted in this task.

## Endpoint protocol

- Monitor losses, preclip gradient norm, latent variance/RMS, affine normalization scales, and per-draw flow loss/time bins every100 updates.
- Evaluate the same512DEV images every1000 updates and all3200DEV images at5000. Save per-image clean/clamped L1, external AlexNet LPIPS, shared distortion, and p90.
- Probe the same32DEV images at t=.25,.5,.75 with fixed noise and class dropout disabled. Report per-image last-block norms/cosines and their relation to reconstruction difficulty. Generation uses one draw at each t; the reported convention is mean per draw.
- The conditional A2 ratio uses all four methods' shared distortion on the same DEV rows, with1000 paired within-class bootstrap resamples, seed917.
- Generate5K samples for the two joint arms only after both are finite and improving. The common random input artifacts are in the CPU evidence directory. Their creation did not run a model.
- Pilot generation uses the native source dopri5 sampler, fixed CFG2, interval[.1,1], source ODE[0,1], tolerances1e-6/1e-3, manual shift.5, and actual NFE counting. No guidance search.
- FIT Inception reference features still need a separate authorized extraction and immutable hash before endpoint gFID/KID. Never substitute full1K published statistics for pilot FIT statistics.
- Missing measures are null with reasons. Initial estimates are advisory. Any Section15.2 extension must record original/revised config hashes/endpoints, affected controls, reason, expected extra cost, and actual constraints before execution.

CONFIRM stays sealed until D1 method freeze.
