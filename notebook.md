# Joint adaptive tokenization pilot data notebook

## 2026-10-24: preparation setup

- Scope: Section 5.4 pilot FIT/ALLOC/DEV/CONFIRM manifests and content identity only.
- Source: this distinct plain clone of kw7243/UNITE-tokenization-generation.
- Branch: research/jat-pilot-data-20261024, directly descended from official UNITE 2389f2c65fedbd43ada851e69b2b810efd23af7d.
- Exact KARL class-list source: 5c1a5693edf87228f988f363b2acdb7eb826a889, verified byte for byte against its retained source file.
- Approved data: /data/vision/torralba/datasets/imagenet_pytorch_new/train; shared source remains read-only.
- Existing 5,000-row KARL manifest contains validation only and cannot identify pilot training bytes.
- No model work, A1 duplication, GPU allocation, canonical campaign edits, or canonical notebook edits.
- CONFIRM stays sealed until D1 method freeze; only IDs, byte hashes and integrity counts are prepared.
- Clone attempt through gh-axi ignored the destination argument and refused the existing default destination; plain git clone into the exact assigned path succeeded. No source was overwritten.

## 2026-10-24: CPU submission

- Full stage: /data/scratch-fast/kwen1/jat-pilot-data-20261024-stages/20261024_025819-4842a9-4929e73-code.
- Stage verified 118 files/links, execute bits and Git identity at source commit 4929e73.
- Config SHA256: fd1bec3f630d0da887bcab20cc8562b8ababf66aab5da4c8552e10eac6b0f00b.
- CPU job 2830620; one owner and one submitted attempt; 4 CPU cores, 4 GB RAM, 1-hour ceiling, no GPU.
- Submission used literal --export=NIL with SBATCH_EXPORT removed; all required environment assignments follow the SBATCH directives.
- Submission/effective script: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z/attempt-1.
- Output parent: /data/vision/torralba/u/kwen1/image-tokenization-20261023/joint-adaptive-tokenization-20261024-data/pilot-splits-v1/20261024T065816Z.
- Terminal result and CPU core-hours remain pending. No A2 pass is claimed.
