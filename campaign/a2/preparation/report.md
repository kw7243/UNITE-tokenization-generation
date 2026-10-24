# A2 preparation: PREPARATION_ONLY

## 1 Objective

Prepare the exact A2 native16/64 implementation and CPU evidence for plan1.1, without starting training.
A2 asks whether the16-token problem is a capacity tradeoff or joint-task pressure; H_CONFLICT is a mechanism hypothesis.
The scientific next-node gate is a stable valid NATIVE64_JOINT control.
The conditional reconstruction ratio must be at least1.10 with a paired95% interval lower endpoint above1.0.
This preparation does not evaluate either gate.

## 2 Setup

- Plan SHA256: 4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5.
- Approved fork: https://github.com/kw7243/UNITE-tokenization-generation.
- Official upstream: https://github.com/ShivamDuggal4/UNITE-tokenization-generation at2389f2c65fedbd43ada851e69b2b810efd23af7d.
- Parent:51a29b63aaba705a23a9de0e39ed8818b74a77ec. Official upstream ancestry and forge commit/fork records are retained in receipts/provenance.json.
- Auxiliary branch: research/jat-a2-20261024. Tested code revision: a367fca5a1cb9b597826d5ffda26d5d1d0c13cca.
- Source: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024.
- Final CPU stage: /data/scratch-fast/kwen1/jat-a2-20261024-stages/20261024_044035-153234-a367fca-code.
- Code bundle SHA256: 2599d8857a1a2914f77525a59db782a7796445cfc2898143bd6d03e84b080a55.
- Exact tested Git patch SHA256: 7642a8d589cd841fe93a82f930d989c88acb35e26854e09ab2eb6530395f9dff; artifact tested-code.patch.
- Resolved config index SHA256: 5cac5fe2f9d54afc86faa246e3fe9826bcfef402cca39255a1b60763dfe4e6a8.
- Metric protocol SHA256: 20acc43b9f3498e092590daddbb56184a7d09e2fe3b4fe7f9812cd36189bd873.
- Environment: /data/scratch-fast/kwen1/envs/joint-adaptive-tokenization-20261024-r1, reused read-only. Torch2.10.0+cu128/CUDA12.8; proposed package pins matched on CPU compute nodes.
- Accepted package-lock SHA256: fbc44e755ea7ab9c1fb95f88783678b7c57fc8a367086054903fbfc2617ef253. Actual freeze--all SHA256: 41b5c4ee15094b54d9151687fb091d654ed6ca9d0adc53612d6e46218668010f. Only Pip25.3 and Setuptools79.0.1 distinguish the two listing modes; no environment change occurred.
- Data: immutable FIT120289,DEV3200,ALLOC3200 and the supplied100-class map. Hashes are in results.json and each resolved YAML. The consumer never opens CONFIRM or CONTENT.
- Accepted adapter SHA256 remains f7c661c7904023c790d8e2e8cfa4e1d6591481286155f354ae2613a7979cdff9. Official third-party snapshots are unchanged.

The default future run is random initialization,seed0,5000 updates,global256,4 H100 GPUs,micro16,accumulation4.
Both objectives are active from update1 in the joint arms.
The implementation keeps source M3 summed denoising loss,detach boundaries,source VGG training loss,and source optimizer partition.
It logs generation loss divided by3 separately.
Source VGG stays frozen and in eval mode after model.train; reporting uses clean/clamped external AlexNet distortion.
LR warms up for500 optimizer updates to1e-4,then stays constant; clipping is global norm3 and EMA decay is.9978.
Weights and loss/normalization reductions are FP32; model autocast is BF16; TF32 and compile are off.
Resolved files also record the source model topology,optimizer defaults/ordered parameter names,noise laws,preprocessing,and sampler/metric settings.

## 3 Baseline

The four prepared methods share images,flips,initialized common bytes,independent random streams,and optimizer-update exposure.
Each native encoder processes its own register count; only the initialized register positional table is sliced.
The common RoPE cache keeps spare capacity but does not process extra registers.
There is no selector,latent prefix truncation,or added length conditioning.
RECON methods are conditional mechanism probes,with cheaper updates; they are not compute-matched generation baselines.
The released M14 checkpoint remains an A1 reference and is not an A2 initialization.
This follows the pinned source rather than substituting the paper's simplified reconstruction-noise/loss description: https://arxiv.org/html/2603.22283v1.

## 4 Results

Zero GPU submissions,zero GPU-hours,zero main-model updates,zero model image exposure,zero scientific feature extraction,and zero quality scores.
All scientific measures are null with reasons in results.json.

| Method | Configuration SHA256 | Main updates | Outcome class |
|---|---|---:|---|
| NATIVE64_JOINT | 68c8a5bdd512f79815862854eea5a0945ffaccee2e594b82c65f521188396c17 | 0 | null: unrun |
| NATIVE64_RECON | 92b3c45664d57db62768dc5b269c837fed1f9fe8f4f58101db9bc5f49982481c | 0 | null: unrun |
| NATIVE16_JOINT | 261f77693650343762115c0aa37fb11343e78e1df4a587189e71bbd6b4c4ccbb | 0 | null: unrun |
| NATIVE16_RECON | 19be684784a5639c729b81031fbb28a2a949e7f80ad6236086be567a8466bdb8 | 0 | null: unrun |

The final CPU checks passed all9 focused groups: emitted configs/schedule; immutable interfaces/class balance; queue exhaustion/resume exposure; paired full-noise streams; tiny common-init byte mapping; full-size metadata parameter membership/default optimizer construction; tiny loss/detach/frozen-LPIPS behavior; image-weighted DDP accumulation/nonfinite-gradient rejection; and scientific/execution boundaries.
No real full model was numerically evaluated on CPU.
Meta-only source structures use a frozen LPIPS placeholder only for shape/membership checks.
The actual pretrained asset bytes were independently hashed in the CPU allocations.

| CPU job | Terminal result | Reserved seconds | CPU core-hours |
|---|---|---:|---:|
| 2831152 | COMPLETED 0:0 | 145 | 0.161111111 |
| 2831187 | COMPLETED 0:0 | 124 | 0.137777778 |
| 2831201 | COMPLETED 0:0 | 112 | 0.124444444 |

Total actual CPU allocation: 0.423333333 core-hours.
There were no failed allocated jobs.
One earlier request was rejected before allocation because normal QoS is invalid for tig-cpu.
The retained partition policy and completed pilot-data precedent selected csail/tig-main/tig-cpu; all later requests used that policy.
Every request used literal --export=NIL with inherited SBATCH_EXPORT cleared and environment assignments after directives.
The initial GitHub clone wrapper did not honor the destination; the approved plain HTTPS clone succeeded with existing credentials,without global credential changes.

The first CPU pass supplied exact optimizer names.
The second checked final configuration/default/asset serialization and saved common generation noise.
The last pass covered added time-bin monitoring,explicit unavailable RECON flow measures,and numerical-gradient failure labeling.
All stages and earlier receipts are preserved.
Common5K labels/noise are random-input artifacts only: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/cpu-check-release/common-generation-inputs.json.
Their creation did not generate model images.

## 5 Interpretation

The implementation and CPU evidence remove bounded engineering work from the later A2 critical path.
They support the sampling,reduction,membership,configuration,and resume-exposure contracts tested here.
They do not establish full-model GPU numerical correctness,stability,quality,task conflict,or a scientific A2 pass.
No hypothesis is classified as negative,and no estimate-based INCONCLUSIVE_RESOURCES outcome is assigned.
The stable64-token checkpoint required by B1 does not exist yet; the complete schema is prepared for its later creation.

## 6 Limitations

A1 full-reference acceptance remains pending under its separate owner.
The accepted prefix correctness fixture is unchanged,but it does not cover the new native loss/training paths.
Repeat the affected Section5.11 native16/64 checks on GPU: correct register outputs,batched/single physical counts and losses,detach/shared-gradient boundaries,and image-free generation.
Retain the accepted256-token proof while its affected source stays unchanged; repeat that identity comparison if it changes.
Also verify real BF16 micro16,four-rank accumulation,memory,EMA,and full checkpoint resume before training.
Full-size common initialized byte equality remains a GPU check; the CPU common-init byte test used a tiny mapping fixture.

Exact checkpoint fields and tensor interfaces are in /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/campaign/a2/interface.json.
Full CPU evidence is at /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation.
Source and unique results stay on the vision filesystem; stages/environments stay on scratch-fast.
The canonical A1 source/branch and main notebook are unchanged.
No scientific checkpoint,trained result,CONFIRM score,or FIT feature cache was produced.
Pilot FIT reference features and their hash remain required before endpoint gFID/KID.
The final Git backup receipt is /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/outputs/a2-preparation/receipts/backup.json; it is written after the report commit to avoid a self-referential commit hash.

## 7 Next action

PREPARATION_ONLY is complete.
Scientific A2 remains BLOCKED_REFERENCE until the coordinator accepts A1; this is not a scientific negative.
The coordinator owns the later GPU check/training assignment.
Use /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-a2-20261024/campaign/a2/README.md for exact launch,checkpoint/resume,and remaining-GPU-check instructions.
Start only the two joint arms after those prerequisites; decide RECON execution from their paired results.
Use fixed512DEV every1000 updates,full3200DEV at5000,and the32-image gradient probe at.25/.5/.75.
Generate5K samples for the joint arms only when both are finite and improving,with common labels/noise and the native pilot sampler/metric protocol.
Then apply A2's gate before B1.
All GPU-hour estimates remain advisory; Section15.2 owns any later matched extension record.
