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

## 2026-10-24: version 1.1 adoption

- Active source plan replaced with exact v1.1 bytes, SHA256 4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5.
- Section 5.4 text equality verified; completed CPU2830620 remains a v1.0 execution compatible with v1.1.
- Full source stage, executed config fd1bec3f630d0da887bcab20cc8562b8ababf66aab5da4c8552e10eac6b0f00b, original result snapshots and old source plan retained.
- Current mutable config binds v1.1 but was not executed; exact bindings are in pilot_data/current-plan-binding.json.
- No GPU authorization, B1 phase, compute extension, cancellation or duplicate hashing job. Canonical notebook owner stays unite-reproduce-20261023.

## 2026-10-24: artifact sealing

- Data job and all integrity checks completed successfully; no new Slurm attempt.
- Administrative sealing initially failed when chmod tried to preserve the run directory's inherited vision-torralba setgid bit. The login's supplementary groups do not include that owning group.
- Corrective action: remove write bits and the inherited setgid bit only on this run's owned directories; keep all artifact bytes and dataset/source paths unchanged. Retain sealing-receipt.json with the exact failure and resolution.
