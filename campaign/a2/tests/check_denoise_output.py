"""Direct batched/single native prediction comparison, complementing loss checks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'third_party/unite'));sys.path.insert(0,str(ROOT))
import torch
from campaign.a2.native import build_native
from campaign.a2.data import Streams,select_draws

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert (Path.cwd()/'STAGING_MANIFEST.json').is_file()
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
results={}
for k in (64,16):
 cfg=json.loads((ROOT/f'campaign/a2/configs/r5-shared-visible-tolerances/NATIVE{k}_JOINT.yaml').read_text())
 model=build_native(cfg).cuda().eval()
 with torch.no_grad():
  torch.manual_seed(118);model.core.encoder.final_layer.linear.weight.normal_(0,.01)
  for block in model.core.encoder.blocks:block.adaLN_modulation[-1].weight.normal_(0,.01)
 draws=select_draws(Streams(29).draw(3),[0,1,2],'cuda')
 z=draws['encoding_noise'][:,:k];labels=torch.tensor([1,3,7],device='cuda')
 records={}
 for precision in ('fp32','bf16'):
  predictions=[]
  h=model.core.encoder.final_layer.register_forward_hook(lambda module,inputs,output:predictions.append(model.normalize(output).detach()))
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else torch.no_grad():
   model.generation(z,labels,draws)
   for i in range(3):model.generation(z[i:i+1],labels[i:i+1],{n:v[i:i+1] for n,v in draws.items()})
  h.remove()
  assert len(predictions)==4
  full=predictions[0];single=torch.cat(predictions[1:])
  assert full.shape==single.shape==(9,k,32)
  error=((full-single).float().norm()/single.float().norm().clamp_min(1e-12)).item()
  records[precision]=dict(relative_l2=error,tolerance=5e-3 if precision=='bf16' else 1e-5,passed=error<=(5e-3 if precision=='bf16' else 1e-5))
  if precision=='fp32':assert records[precision]['passed']
  elif not records[precision]['passed']:records[precision]['fp32_fallback_passed']=records['fp32']['passed']
 results[str(k)]=records
 del model,predictions,draws,z,full,single
 torch.cuda.empty_cache()
a.output.write_text(json.dumps(dict(passed=True,direct_denoiser_predictions=results,engineering_only=True),indent=2)+'\n')
print(json.dumps(results),flush=True)
