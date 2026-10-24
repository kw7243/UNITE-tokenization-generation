"""Future clean per-count SCREEN evaluation; no quality workload during preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/unite'))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from campaign.a2.data import load_images, per_image_draws, eval_rows, read_interface, sha
from campaign.a2.evaluate import generation_inputs
from campaign.b1.convert import require, SCHEMA
from campaign.b1.screen import build_screen


@torch.no_grad()
def reconstruction(model, rows, device, seed, out, counts=(16,64), batch_size=16):
    import lpips
    alex=lpips.LPIPS(net='alex',version='0.1').to(device).eval().requires_grad_(False)
    records={str(k):[] for k in counts}
    sums={str(k):[] for k in counts}
    for start in range(0,len(rows),batch_size):
        part=rows[start:start+batch_size]
        images=load_images(part,[False]*len(part),device)
        draws={k:v.to(device) for k,v in per_image_draws(part,seed).items()}
        with torch.autocast(device.type,dtype=torch.bfloat16):
            z=model.encode(images,draws['encoding_noise'],draws['encoding_time'])
            for k in counts:
                decoded=model.decode(z,torch.full((len(part),),k,dtype=torch.long,device=device)).float().clamp(0,1)
                with torch.autocast(device.type,enabled=False):
                    l1=(decoded-images).abs().flatten(1).mean(1)
                    perceptual=alex(2*decoded-1,2*images-1).flatten()
                sums[str(k)].append(z[:,:k].double().cpu())
                for row,a,b in zip(part,l1.tolist(),perceptual.tolist()):
                    records[str(k)].append(dict(image_id=row['image_id'],class_index=row['class_index'],
                        recon_l1=a,recon_lpips=b,allocation_distortion=a+b))
    summary={}
    for k in counts:
        r=records[str(k)]; z=torch.cat(sums[str(k)])
        summary[str(k)]={name:float(np.mean([v[name] for v in r])) for name in ('recon_l1','recon_lpips','allocation_distortion')}
        summary[str(k)].update(recon_l1_p90=float(np.quantile([v['recon_l1'] for v in r],.9)),
            latent_image_variance=z.var(0,unbiased=False).mean().item(),mean_tokens=k,mean_tokens_sq=k*k,
            images=len(r),rfid_eval=None,rfid_eval_reason='Separate endpoint feature workload required')
    Path(out).write_text(json.dumps(dict(summary=summary,per_count=records),indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--mode',choices=['reconstruction','generation'],required=True)
    p.add_argument('--count',type=int,choices=[16,64])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text()); receipt=json.loads(args.authorization.read_text())
    from campaign.b1.train import verify_config, authorize
    digest=sha(args.config)
    authorize(cfg,receipt,digest,args.execute)
    verify_config(cfg)
    require(receipt.get('endpoint_checkpoint_sha256')==sha(args.checkpoint),'Endpoint acceptance/hash required')
    require(torch.cuda.is_available() and (Path.cwd()/'STAGING_MANIFEST.json').is_file(),'Staged GPU evaluation required')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    require(state['schema']==SCHEMA and not state.get('synthetic_fixture',False)
            and state['configuration_sha256']==digest,'Wrong B1 endpoint')
    model=build_screen(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['ema'],strict=True)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.mode=='reconstruction':
        reconstruction(model,read_interface(cfg['data'])['DEV'],torch.device('cuda'),cfg['evaluation']['seed'],args.output/'dev-full.json')
    else:
        require(receipt.get('common_screen_endpoint_accepted') is True and args.count is not None,'Common endpoint and fixed count required')
        if cfg['method_id'] in ('PREFIX16','PREFIX64'):
            require(args.count==int(cfg['method_id'][6:]),'Fixed arm must use its own count')
        labels,noise=generation_inputs(cfg['evaluation']['seed'])
        # Exact same full64 random tensor is sliced by the source physical generation path.
        nfe=[0]
        def count_forward(module,inputs):
            nfe[0]+=1
        handle=model.core.encoder.t_embedder.register_forward_pre_hook(count_forward)
        try:
            for start in range(0,5000,64):
                y=labels[start:start+64].cuda(); e=noise[start:start+64].cuda()
                counts=torch.full((len(y),),args.count,dtype=torch.long,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    images=model.generate(y,counts,e).float().clamp(0,1)
                pixels=(images*255).clamp(0,255).to(torch.uint8)
                np.save(args.output/f'pixels-{start:04d}.npy',pixels.cpu().numpy())
        finally:
            handle.remove()
        (args.output/'generation-inputs.json').write_text(json.dumps(dict(seed=cfg['evaluation']['seed'],
            labels_sha256=hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
            full64_noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
            count=args.count,samples=5000,actual_nfe=nfe[0],metrics=None,
            metrics_reason='Use verified pilot FIT features and locked metric protocol; no standard1K substitution'),indent=2)+'\n')

if __name__=='__main__':
    main()
