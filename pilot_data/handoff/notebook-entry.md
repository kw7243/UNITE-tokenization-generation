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
