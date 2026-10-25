## 1 Objective

Prepare the same native64 arm to continue from checkpoint4000 to5000 effective optimizer updates.
This is recovery within the original A2 plan, selected by coordinator message036, not a new scientific branch.
The configuration is disabled and has no launch command or authorization receipt.
No recovery model load, model-byte hash, test, analysis, allocation or submission ran.

## 2 Setup

Inspected research revision: `8287b073fca65956b51bf308283ce61716189ebf` on `research/jat-a2-20261024`.
Original r6 config: `campaign/a2/configs/r6-source-ddp-unused/NATIVE64_JOINT.yaml`, SHA256 `115e15e348d2c928aaba88a4c2654cb37cfd8d19d22580117bac9213b525c7b6`.
Prepared specification: `campaign/a2/configs/native64-continuation-r1.disabled.json`, SHA256 `e0dee53858796806760395e15b1774349d6a02d5a57c0a53e4c97ecc96b6814d`.
It embeds the original resolved configuration without changes, and binds original non-model logs by digest.
It is a preparation schema, not an accepted `train.py` config.
The original immutable stage is `/data/scratch-fast/kwen1/jat-a2-20261024-stages/20261024_164742-bce608-7a492f5-code`.
The future continuation uses a new output parent `outputs/A2/native64-continuation-r1` and a fresh full stage after final review/freeze.
Neither future output nor stage has been created.
All original stages/outputs, r2 templates, native16, FIT and the read-only environment remain unchanged.

## 3 Baseline

The target stays5000 effective updates and1,280,000 images, seed0, physical native64, PILOT M3 sum, the supplied100-class FIT and the original matched native16 control.
Preserve global256, four ranks, micro16, accumulation4,32classes x8, the same images/flips/full64 random draws before native slicing, detached clean/noised inputs, source loss scaling, LR1e-4/warmup500/constant, clip3, EMA.9978, BF16/FP32, TF32off/compileoff and frozen source VGG LPIPS.
No data queue/RNG reseed, LR warmup restart or partial-weight restore is valid.

Reuse the accepted actual four-rank engineering proof `/data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/A2/engineering/native-checks-complete-20261024/results.json`, SHA256 `1d481e6f509919cb649597d8c32995f429e95d4d81e5d4a35b80c0b0781c2a45`.
It restored the saved complete state exactly at the engineering boundary, then reproduced the next update numerically.
Native64 relativeL2 maxima were model1.4865179113412888e-08, EMA3.2129695176305284e-10 and optimizer2.412110108857699e-06.
Next-update bytes were not bitwise equal; no bitwise guarantee is claimed.
This checkpoint1 proof does not inspect checkpoint4000 or the revised production resume path.

## 4 Results

Original job2834281 ended PREEMPTED at2026-10-25T13:36:48Z; it was first observed at17:01:53Z.
Parentexit0:0 is not a successful endpoint; its batch/numbered steps were cancelled/failed.
Saved results contain4266 completed updates, while retained checkpoint4000 has original recorded SHA256 `9fb9c31deb782cb845ef34c20d539dda10651e720304f26339d24467be74ab7d`.
That model file was not independently rehashed or loaded in this preparation.

| Exposure boundary | Updates | Image exposures or selected slots |
| --- | ---: | ---: |
| Retained effective prefix | 4000 | 1024000 |
| Abandoned completed4001..4266 | 266 | 68096 |
| Last attempted4267, completion absent | 1 | 256 selected; actual processed count unknown |
| Planned new effective suffix4001..5000 | 1000 | 256000 |
| Planned effective endpoint | 5000 | 1280000 |
| Physical completed total if one recovery succeeds | 5266 | 1348096 |
| Attempted batch total under that assumption | 5267 | 1348352 selected slots, not a claim all were processed |

The original ledger has8533 rows:4267 attempts and4266 completions.
The last attempt chain is `9f8f1a105809fe7b2d2c62b576beeaa5b3306b998f5dfae5f98d3c8c984b338c`.
Any replay acknowledgement must cover highest attempted4267, not merely4266.
Future engineering attempts are additional physical work with zero effective scientific exposure.

Source inspection identified seven concrete gaps, detailed in the specification:

1. New config/output identity conflicts with the saved original config digest required by `checkpoint.load`.
2. The replay guard reads only destination logs and cannot see abandoned parent work in a new output.
3. The resumed results writer expects a local checkpoint4000 hash sidecar and prefix monitor/DEV files.
4. Inherited counters need separate physical/effective/cost accounting to avoid double counting.
5. Endpoint matching and current-ROOT source locks need explicit original-stage and recovery lineage validation.
6. Endpoint consumers expect `job_state`; existing terminal receipts use `state`.
7. Production loads before DDP construction; accepted engineering validation loads after DDP construction.

No source guard has been removed and no executable code was changed to hide these gaps.
The minimum reviewed revision must validate exact original configuration/checkpoint bytes and explicitly enumerate new code/output/lineage fields, while rejecting any scientific-control change.
Derived effective exposure/monitor histories must bind original prefix1..4000 plus new suffix4001..5000, verify IDs/flips/noise against unchanged native16 and retain all raw physical/replayed records.
A derived effective ledger is not an original physical ledger.

## 5 Interpretation

Restoring full checkpoint4000 state and replaying the same logical sequence can recover the original finite5000 experiment.
It does not erase the266 abandoned completed updates or unknown partial4267 work from cost/exposure accounting.
A role string such as A2_NATIVE64_COMPLETE is insufficient: the existing saver emits that role at partial boundaries too.
Require strict model/EMA/named optimizers/schedule/FIT queues/all streams/all four rank RNG/counters/history validation at exactly4000 after a completed update.

For a proposed later check, isolate restore/update4001/reload/repeat from science and compare sampler/streams/RNG/exposure exactly.
Use FP32 relativeL2 tolerance1e-5 and BF16 tolerance5e-3; repeat failed BF16 comparisons in FP32 before defect diagnosis.
Check the actual production entry order, four-rank BF16 micro16/accum4, clipping, EMA, frozen VGG and memory.
Reuse unaffected accepted checks; do not start another broad test campaign.

Saved first200 timing predicts4.045111 wall-hours for1000 remaining updates on the original four RTX3090 layout, or16.180443 RTX3090 device-hours and64.721771 allocatedCPU core-hours at16 CPUs.
Planning proxies add a2CPU/32GiB/15minute trusted checkpoint check, a5–15minute real four-rank recovery check (0.333333–1RTX3090h), and0.1–0.5RTX3090h for endpoint evaluation/probes/checkpoint.
The latter proxies are not measured native64 continuation costs; allocation setup/tail and I/O remain unknown.
The proposed continuation request is four RTX3090s, two ranks per node across two nodes,16CPUs total,32GiB per node and6hours, subject to current permitted-resource inspection and actual fit.
No device or allocation is selected now.
The64H100-hour plan figure is advisory, not a cap, reserve, cross-device conversion or reason to stop.
Count original native64 allocation67.12666666666667RTX3090h/268.50666666666666CPUcoreh once, and original native16 allocation17.996666666666666RTX3090h separately; do not add inherited prefix process time twice.

## 6 Limitations

No usable substantive-review route is released and the primary launcher decision remains pending.
No new runnable training config exists until G1..G7 have a reviewed minimal implementation and new source/config/patch hashes.
Checkpoint4000 bytes/state, the trained production resume path and future allocation fit remain untested here.
The prepared r2 endpoint code has no substantive-review acceptance; r2 bytes are preserved.
Native64 has no completed5000 endpoint/full3200DEV/probes/checkpoint5000.
Paired scientific outcome is null; valid64, generation and conditional RECON remain unreleased.
Only accepted256 A1 reference quality is established.
This report is a worker preparation receipt, not coordinator acceptance or final A2 completion.

## 7 Next action

Include the required G1..G7 changes and existing unreviewed training/checkpoint/endpoint helpers in the one substantive review unit when the supported route is released.
Bind the final runtime config, immutable parent, scientific-control equivalence, tests and source/patch digests before requesting exact coordinator acceptance.
Only then run authorized checkpoint/actual-environment checks, freeze a fresh complete source stage, verify allocated Slurm/node/GPU/CUDA/memory/mounts, and return the measured launch forecast/check evidence.
A later continuation submission needs the coordinator acceptance tied to those exact identities.
Use literal --export=NIL, clear inherited SBATCH_EXPORT and put required environment assignments after all directives.
Do not append to the preempted attempt or change native16/FIT.

After any later authorized submission, the same terminal owner must retain live monitoring between DEV observations, through the existing supported job monitor or bounded foreground polling of the exact owned job, proposed at most60seconds between checks.
Use explicitUTC accounting and raw stdout/stderr when the job disappears or a terminal/preemption/failure state appears; reconcile all child steps and record end and first-observed timestamps separately.
Scheduled four-hour DEV observations supplement that monitoring; they cannot replace it.
Preserve failures and report before any further recovery attempt. No new monitoring service or wrapper is requested or created.
All9owned jobs are currently terminal, so no current job monitor or resource is running.
