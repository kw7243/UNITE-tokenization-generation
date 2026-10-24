"""Real four-rank BF16 accumulation, EMA and checkpoint continuation checks."""
import argparse
import copy
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'third_party/unite'))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from campaign.a2 import checkpoint
from campaign.a2.data import FitQueue,Streams,load_images,local_indices,read_interface,select_draws,sha
from campaign.a2.native import build_native,backward_mean,clip_gradients,executed_lr,make_optimizers
from campaign.a2.train import verify_config


def fingerprint(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous()
            h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):add(key);add(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(str(type(v)).encode())
            for part in v:add(part)
        else:h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def floating_difference(expected, actual):
    """Compare floating state as one ordered vector; require other state exactly."""
    difference=0.;reference=0.;maximum=0.
    def visit(a,b):
        nonlocal difference,reference,maximum
        if torch.is_tensor(a):
            assert torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype
            x,y=a.detach().cpu(),b.detach().cpu()
            if x.is_floating_point():
                delta=x.double()-y.double()
                difference+=delta.square().sum().item();reference+=x.double().square().sum().item()
                maximum=max(maximum,delta.abs().max().item() if delta.numel() else 0.)
            else:assert torch.equal(x,y)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:visit(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert type(a)==type(b) and len(a)==len(b)
            for x,y in zip(a,b):visit(x,y)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        else:assert a==b
    visit(expected,actual)
    return dict(relative_l2=(difference/max(reference,1e-30))**.5,maximum_absolute=maximum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file() and os.environ.get('SLURM_JOB_ID')
    cfg=json.loads(args.config.read_text());verify_config(cfg)
    digest=sha(args.config)
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    print(json.dumps(dict(preinit_rank=os.environ['RANK'], local_device=str(device),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), device_count=torch.cuda.device_count(),
        uuid=str(torch.cuda.get_device_properties(device).uuid))),flush=True)
    dist.init_process_group('nccl', device_id=device)
    rank=dist.get_rank();assert dist.get_world_size()==4
    assert torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
    out=args.output
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    runtime=dict(rank=rank,job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
        gpu=torch.cuda.get_device_name(device),vram_bytes=torch.cuda.get_device_properties(device).total_memory,
        uuid=str(torch.cuda.get_device_properties(device).uuid),stage=str(Path.cwd()),torch=torch.__version__,cuda=torch.version.cuda,
        configuration_sha256=digest,engineering_only=True)
    (out/f'runtime-rank{rank}.json').write_text(json.dumps(runtime,indent=2)+'\n')
    splits=read_interface(cfg['data']);sampler=FitQueue(splits['FIT'],0);streams=Streams(0)
    model=build_native(cfg).to(device).train()
    ema=build_native(cfg).to(device).eval().requires_grad_(False);ema.load_state_dict(model.state_dict(),strict=True)
    opts,names=make_optimizers(model,cfg['training'])
    assert names==cfg['training']['optimizer_parameter_names']
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False, find_unused_parameters=True)
    params=[p for p in model.parameters() if p.requires_grad]
    counters=dict(main_updates=0,images_seen=0);history=[];records=[]
    def state():
        return dict(model=model.state_dict(),ema=ema.state_dict(),optimizers={k:o.state_dict() for k,o in opts.items()},
            names=names,sampler=sampler.state_dict(),streams=streams.state_dict(),rng=checkpoint.process_rng(),
            counters=counters,history=history)
    def update(number,verify_reduction=False):
        assert sampler.updates+1==number
        indices=sampler.next();draws=streams.draw(256);rows=[splits['FIT'][i] for i in indices]
        inputs=[]
        for acc in range(4):
            local=local_indices(rank,acc)
            inputs.append((load_images([rows[i] for i in local],draws['flip'][local],device),
                torch.tensor([rows[i]['class_index'] for i in local],device=device),select_draws(draws,local,device)))
        for opt in opts.values():
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=executed_lr(number)
        losses=[]
        for acc,(images,labels,selected) in enumerate(inputs):
            with ddp.no_sync() if acc<3 else nullcontext():
                with torch.autocast('cuda',dtype=torch.bfloat16):values=ddp(images,labels,selected)
                assert torch.isfinite(values['per_image']).all()
                backward_mean(values['per_image']).backward()
            losses.append(values['per_image'].detach().mean().item())
        unused=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert set(unused)<=set(cfg['distributed']['expected_unused_trainable_parameters']),unused
        reduction_error=None
        if verify_reduction:
            expected=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            for images,labels,selected in inputs:
                with ddp.no_sync():
                    with torch.autocast('cuda',dtype=torch.bfloat16):values=ddp(images,labels,selected)
                    backward_mean(values['per_image']).backward()
            difference=torch.zeros(2,dtype=torch.float64,device=device)
            for p,reference in zip(params,expected):
                assert (p.grad is None)==(reference is None)
                if reference is None:continue
                dist.all_reduce(p.grad);p.grad.div_(4)
                ref=reference.to(device)
                difference[0]+=(p.grad.double()-ref.double()).square().sum()
                difference[1]+=ref.double().square().sum()
            reduction_error=(difference[0]/difference[1].clamp_min(1e-30)).sqrt().item()
            assert reduction_error<=5e-3,reduction_error
            del expected
        norm=clip_gradients(params)
        clipped=torch.stack([p.grad.float().square().sum() for p in params if p.grad is not None]).sum().sqrt().item()
        assert clipped<=3.0001 and torch.isfinite(norm)
        norms=[None]*4;dist.all_gather_object(norms,norm.item());assert max(norms)-min(norms)<1e-5
        before={n:p.detach().cpu().clone() for n,p in ema.named_parameters() if dict(model.named_parameters())[n].requires_grad}
        for opt in opts.values():opt.step()
        live=dict(model.named_parameters());ema_error=0.
        with torch.no_grad():
            for name,param in ema.named_parameters():
                if live[name].requires_grad:
                    param.mul_(.9978).add_(live[name],alpha=1-.9978)
                    expected=before[name].to(device).mul_(.9978).add_(live[name],alpha=1-.9978)
                    ema_error=max(ema_error,(param-expected).abs().max().item())
        assert ema_error==0
        assert not model.core.lpips_loss.training and all(not p.requires_grad for p in model.core.lpips_loss.parameters())
        for p in model.parameters():assert torch.isfinite(p).all()
        for o in opts.values():
            for s in o.state.values():
                for v in s.values():
                    if torch.is_tensor(v):assert torch.isfinite(v).all()
        counters.update(main_updates=number,images_seen=number*256)
        result=dict(update=number,preclip_norm=norm.item(),postclip_norm=clipped,
            ddp_vs_explicit_mean_relative_l2=reduction_error,ema_exact_error=ema_error,
            local_micro16_losses=losses,unused_parameter_names=unused,exposure_chain=sampler.exposure_chain,draw_sha256=fingerprint(draws))
        del inputs,before,values
        return result
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    records.append(update(1,verify_reduction=True))
    boundary=fingerprint(state())
    checkpoint.save(out/'engineering-update1.pt',model,ema,opts,names,sampler,streams,digest,cfg,counters,history)
    saved=checkpoint.process_rng()
    torch.cuda.synchronize(device); update2_started=time.perf_counter()
    records.append(update(2))
    torch.cuda.synchronize(device); update2_seconds=time.perf_counter()-update2_started
    expected=cpu_copy(state())
    # Destroy optimizer state and mutate all state, then restore the actual snapshot.
    with torch.no_grad():next(model.parameters()).add_(1)
    sampler.next();streams.draw(256);torch.rand(3,device=device)
    restored_counters,restored_history=checkpoint.load(out/'engineering-update1.pt',model,ema,opts,names,sampler,streams,digest)
    counters.clear();counters.update(restored_counters);history[:]=restored_history
    assert fingerprint(state())==boundary,'Round-trip state changed'
    assert fingerprint(checkpoint.process_rng())==fingerprint(saved)
    replay=update(2)
    assert replay['update']==records[-1]['update']
    assert replay['exposure_chain']==records[-1]['exposure_chain'] and replay['draw_sha256']==records[-1]['draw_sha256']
    for key in ('preclip_norm','postclip_norm','local_micro16_losses'):
        a,b=torch.as_tensor(records[-1][key]),torch.as_tensor(replay[key])
        assert ((a-b).norm()/a.norm().clamp_min(1e-12)).item()<=5e-3,key
    actual=state()
    for key in ('names','sampler','streams','rng','counters','history'):
        assert fingerprint(expected[key])==fingerprint(actual[key]),key
    continuation={key:floating_difference(expected[key],actual[key]) for key in ('model','ema','optimizers')}
    assert all(value['relative_l2']<=1e-5 for value in continuation.values()),continuation
    next_update_exact=fingerprint(expected)==fingerprint(actual)
    model_hash=fingerprint(model.state_dict());all_hashes=[None]*4
    dist.all_gather_object(all_hashes,model_hash)
    rank_agreement_error=0.
    if len(set(all_hashes))!=1:
        sums=torch.zeros(2,dtype=torch.float64,device=device)
        for parameter in model.parameters():
            reference=parameter.detach().clone();dist.broadcast(reference,src=0)
            sums[0]+=(parameter.double()-reference.double()).square().sum()
            sums[1]+=reference.double().square().sum()
        local_error=(sums[0]/sums[1].clamp_min(1e-30)).sqrt()
        dist.all_reduce(local_error,op=dist.ReduceOp.MAX);rank_agreement_error=local_error.item()
        assert rank_agreement_error<=1e-5,rank_agreement_error
    result=dict(passed=True,rank=rank,world_size=4,microbatch_per_gpu=16,accumulation=4,global_batch=256,
        physical_native_count=model.count,records=records,replay=replay,all_rank_model_sha256=all_hashes,
        complete_roundtrip_exact=True,next_update_exact=next_update_exact,continuation=continuation,
        continuation_fp32_state_tolerance=1e-5,replayed_bf16_loss_tolerance=5e-3,all_rank_parameter_relative_l2=rank_agreement_error,
        all_rng_sampler_optimizer_names_verified=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.perf_counter()-started,engineering_update2_seconds=update2_seconds,
        timing_scope='One complete engineering update including CPU image loads and explicit EMA verification copies; not a real200-update training forecast.',
        engineering_only=True,scientific_updates=0,
        engineering_checkpoint_sha256=(out/'engineering-update1.pt.sha256').read_text().strip(),
        checkpoint_warning='Engineering update1 only; its library role field does not establish a stable scientific A2 handoff.')
    (out/f'results-rank{rank}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if rank==0:print(json.dumps(result,allow_nan=False),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
