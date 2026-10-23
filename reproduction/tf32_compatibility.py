"""Bounded native-API counterfactual; no checkpoint or paper metric is evaluated."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser()
p.add_argument("--api",choices=["source","legacy"],required=True)
p.add_argument("--output",type=Path,required=True)
p.add_argument("--reference",type=Path)
a=p.parse_args()
assert Path("STAGING_MANIFEST.json").is_file() and os.getenv("SLURM_JOB_ID")
a.output.mkdir(parents=True,exist_ok=True)
if a.api=="source":
    torch.backends.cuda.matmul.fp32_precision="tf32"
    torch.backends.cudnn.conv.fp32_precision="tf32"
else:
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
torch.manual_seed(17)
x=torch.randn(64,768).cuda()
w=torch.randn(768,768).cuda()
cx=torch.randn(2,64,32,32).cuda()
cw=torch.randn(64,64,3,3).cuda()
with torch.no_grad():
    linear=F.linear(x,w)
    convolution=F.conv2d(cx,cw,padding=1)
arrays={"linear":linear.cpu().numpy(),"convolution":convolution.cpu().numpy()}
record={"run_id":a.output.name,"node_id":"TF32_API_COMPATIBILITY","api":a.api,"status":"PASS",
        "paper_metric":False,"job_id":os.environ["SLURM_JOB_ID"],"stage":str(Path.cwd()),
        "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
        "gpu_uuid":str(torch.cuda.get_device_properties(0).uuid)}
if a.api=="legacy":
    compiled=torch.compile(lambda x,w:F.linear(x,w),mode="default")
    with torch.no_grad():
        actual=compiled(x,w)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    record["compiled_linear_max_abs_vs_eager"]=float((actual-linear).abs().max())
    record["legacy_getters"]={"matmul":torch.backends.cuda.matmul.allow_tf32,"cudnn":torch.backends.cudnn.allow_tf32}
if a.reference:
    ref=np.load(a.reference)
    record["source_eager_comparison"]={k:{"bitwise_equal":bool(np.array_equal(v,ref[k])),
                    "max_absolute_difference":float(np.max(np.abs(v-ref[k])))} for k,v in arrays.items()}
    assert all(v["bitwise_equal"] for v in record["source_eager_comparison"].values())
np.savez(a.output/"outputs.npz",**arrays)
(a.output/"results.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
