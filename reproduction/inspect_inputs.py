"""Read released checkpoint metadata and the permitted validation inventory on CPU."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

inputs, out = map(Path, sys.argv[1:])
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

receipt = {"node": platform.node(), "job_id": os.getenv("SLURM_JOB_ID"), "stage": str(Path.cwd()),
           "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "files": {}, "checkpoints": {}}
for path in sorted(inputs.iterdir()):
    if path.is_file():
        receipt["files"][path.name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
for name in ["UNITE-B.pt", "UNITE-L.pt"]:
    with FakeTensorMode():
        ckpt = torch.load(inputs / name, map_location="cpu", weights_only=True, mmap=True)
    def describe(value):
        if torch.is_tensor(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return {"type": str(type(value))}
    receipt["checkpoints"][name] = describe(ckpt)
    (out / "checkpoint-metadata.json").write_text(json.dumps(receipt, indent=2))
    print(name, "top-level keys", list(ckpt), flush=True)
    for key, value in ckpt.items():
        if not isinstance(value, (dict, list, tuple)) and not torch.is_tensor(value):
            print(key, value, flush=True)
stats = np.load(inputs / "jit_in256_stats.npz")
receipt["fid_stats"] = {k: {"shape": list(stats[k].shape), "dtype": str(stats[k].dtype)} for k in stats.files}
data = Path("/data/vision/torralba/datasets/imagenet_pytorch_new")
class_map_path = data / "imagenet_class_index.json"
class_map = json.loads(class_map_path.read_text())
classes = sorted(p.name for p in (data / "val").iterdir() if p.is_dir())
assert len(classes) == 1000
assert classes == [class_map[str(i)][0] for i in range(1000)]
records = []
for index, synset in enumerate(classes):
    files = sorted((data / "val" / synset).iterdir())
    images = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    records.append({"index": index, "synset": synset, "count": len(images), "first": images[0].name if images else None})
assert sum(x["count"] for x in records) == 50000
receipt["dataset"] = {"root": str(data), "validation_count": 50000, "class_map_sha256": sha(class_map_path),
                      "class_count": len(classes), "classes": records, "read_only": True,
                      "inventory_scope": "validation filenames and class-map bytes; image content not hashed"}
(out / "checkpoint-metadata.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps({k: v for k, v in receipt.items() if k not in {"checkpoints", "dataset"}}, indent=2), flush=True)
print("input metadata inspection passed", flush=True)
