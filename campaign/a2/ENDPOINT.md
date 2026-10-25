# A2 endpoint work

This completes the bounded A2 measurement path. It adds no training trial or later-node run.
The initial matched endpoint is5000 updates. A justified extension needs a new matched protocol/configuration and preserves these results.

`endpoint_features.py` accepts both complete joint arms together, checks their fixed512DEV generation condition and uses EMA.
It checks complete native checkpoint schema, role, counters, schedule, live/EMA layouts, ordered optimizer names, FIT queues, stream state and all four rank RNG records.
The source native sampler receives only labels and common full64-position Gaussian noise, sliced at the native boundary.
It keeps the fixed source dopri5/CFG2 recipe, actual GE NFEs, sampling/decode/conversion timing, full generated uint8 images and common input draws.
DEV real/reconstruction and generated Inception features use the pinned source extractor and pixel conversion.
It does not compute FID/KID while reserving a GPU. The two joint arms run sequentially on the same selected GPU and use batch16.
This quality-workload timing is not a D1 batch64 cost comparison.

`aggregate.py` runs in a CPU-only Slurm allocation after the relevant input jobs are terminal.
It joins reconstruction rows by image ID, verifies the exact fixed DEV set and all5000 exposure records, and requires identical exposure-file hashes across compared arms.
It uses the source FID implementation and the package's unbiased KID kernel with100 paired subsets of1000, without replacement, seed917.
Reconstruction differences use1000 paired within-class bootstraps, then equal class means.
It retains full per-image error pairs, gradient probes, monitoring, readable curves and a seven-section report.
Conditional RECON input requires exact coordinator per-arm release receipts and unchanged controls beyond the objective.
The ratio screen remains a mechanism hypothesis, not proof. The script does not issue coordinator acceptance or dispatch B1.

The three current r2 configuration templates are disabled; the original r1 drafts are preserved byte-for-byte as unused history. Missing future endpoint, terminal and review/test evidence is a prerequisite, not a scientific failure.
Before use, bind actual immutable input hashes and the completed source/configuration to the required local NoMistakes evidence.
Do not treat a reviewed disabled template as proof that a later changed scientific recipe was reviewed.
The supported path skips only push,pr,ci, keeps every required local stage, recovers offered fixes and preserves source custody.
Freeze a fresh full repository stage and exact resolved configuration after relevant actual-environment checks. Keep old configurations and outputs.
No new numerical analysis, scientific launch or guidance search is authorized merely by this source preparation.

The original training environment stays read-only.
The separate `jat-a2-postprocess-20261024-r1` environment adds the eight pinned plotting wheels and reads original scientific packages through a `.pth` file.
`postprocess-environment.json` binds its metadata; runtime checks enforce the base scientific versions and all new plotting versions.
The concrete defect was the absence of Matplotlib and native plot tools. Package metadata/installation checks are not a successful plot or scientific test.

Prepared checks, not yet run:

- `tests/check_endpoint_cpu.py`: synthetic row alignment/rejection, class-balanced paired bootstrap, matched-config rejection and paired KID/package parity. It proves no CUDA or scientific result.
- `tests/check_endpoint_gpu.py`: full initialized native64/16 BF16 batch16 source ODE, physical native CFG batches, actual NFEs, no image-encoding input, unchanged common noise, source feature parity, memory fit and rejection of the retained engineering snapshot as a complete scientific checkpoint.

Run these only on the required reviewed/final source in full CSAIL stages and the appropriate Slurm allocation.
The GPU fixture uses random initialization and the role `A2_ENDPOINT_ENGINEERING_CHECK`; its images never count toward scientific5K generation.
Training/DDP/EMA/resume and unchanged source numerical checks inherit the exact accepted GPU evidence only while affected bytes/semantics remain unchanged.
Any relevant review fix invalidates inherited proof and needs the affected real checks. CPU fixtures alone do not supply that proof.
Keep all test outputs, terminal receipts, source/config hashes, costs and notebook records on CSAIL.

Concrete launch payloads are endpoint.sbatch and aggregate.sbatch; each receives an already verified full-stage path and a frozen resolved-config path. Resource/account/partition/time requests are explicit caller arguments after current inventory. Clear inherited SBATCH_EXPORT and pass literal --export=NIL. The GPU payload uses the original environment and source one-GPU visibility; CPU aggregation uses the separate plot environment and CPU-only visibility. These payloads have syntax checks only and have never been submitted.
