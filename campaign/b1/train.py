"""Future SCREEN continuation. Draft configs and absent trained inputs cannot execute."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'third_party/unite'))
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from campaign.a2.data import FitQueue, read_interface, local_indices, select_draws, load_images, eval_rows, sha
from campaign.a2.native import backward_mean, clip_gradients
from campaign.a2.train import append
from campaign.b1.convert import require, load_accepted, convert_snapshot, SCHEMA
from campaign.b1.counts import ScreenStreams
from campaign.b1.screen import build_screen, make_optimizers
from campaign.b1.evaluate import reconstruction
from campaign.b1 import checkpoint


def authorize(cfg, receipt, digest, execute):
    require(execute and cfg['execution_authorized'] is True, 'Draft preparation config is disabled')
    require(cfg['input']['checkpoint'] and cfg['input']['checkpoint_sha256']
            and cfg['input']['acceptance'], 'Unresolved trained A2 input')
    require(receipt.get('allow_b1_screen') is True and receipt.get('configuration_sha256')==digest
            and receipt.get('patch_sha256')==cfg['source']['patch_sha256']
            and receipt.get('plan_sha256')==cfg['plan_sha256'], 'Exact coordinator execution receipt required')
    required=('register_identity','mixed_count_fp32','mixed_count_bf16','gradient_boundary',
              'ode_cfg_image_free','ddp_image_reduction','real_checkpoint_resume')
    require(all(receipt.get('gpu_checks',{}).get(key) is True for key in required), 'Affected Section5.11/GPU checks unresolved')
    require(receipt.get('gpu_check_source_patch_sha256')==cfg['source']['patch_sha256'], 'GPU proof belongs to another patch')


def verify_config(cfg):
    require(cfg['node_id']=='B1' and cfg['phase_id']=='SCREEN' and cfg['method_id'] in ('PREFIX16','PREFIX64','NESTED_RANDOM'), 'Only SCREEN is implemented')
    require(cfg['seed']==0 and cfg['config_family']=='PILOT_M3', 'Wrong causal pilot recipe')
    require(cfg['plan_sha256']=='4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5','Wrong plan')
    require(cfg['model']['candidate_count']==64 and cfg['model']['num_classes']==100
            and cfg['model']['length_count_order']==[4,16,64] and not cfg['model']['selector'], 'Wrong prefix model')
    t=cfg['training']
    require((t['global_batch'],t['gpu_count'],t['microbatch_per_gpu'],t['accumulation'])==(256,4,16,4),'Revised layout requires matched protocol')
    require(t['main_lr']==1e-4 and t['warmup_updates']==0 and t['denoising_draws']==t['flow_chunk_size']==3,'Continuation schedule/loss mismatch')
    require(t['main_updates']>=5000,'Under-length SCREEN is ineligible')
    if t['main_updates']!=5000:
        extension=cfg['extension']
        require(extension['revised_endpoint']==t['main_updates'] and extension['extension_reason']
                and set(extension['affected_control_runs'])=={'PREFIX16','PREFIX64','NESTED_RANDOM'},'Matched recorded extension required')
    for name,expected in cfg['source']['code_sha256'].items():
        require(sha(ROOT/name)==expected,'Source hash mismatch: '+name)
    imports=json.loads((ROOT/'campaign/locks/source-imports.json').read_text())
    for name,expected in imports['files']['unite'].items():
        require(sha(ROOT/'third_party/unite'/name)==expected,'Imported UNITE source changed')
    require(sha(cfg['environment']['packages'])==cfg['environment']['packages_sha256'],'Environment lock mismatch')
    for package,version in cfg['environment']['package_versions'].items():
        require(importlib.metadata.version(package)==version,'Package version mismatch: '+package)
    require(sha(ROOT/'campaign/b1/metric-protocol.json')==cfg['evaluation']['metric_protocol_sha256'],'Metric protocol changed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--authorization',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--ack-replay-through',type=int)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text()); digest=sha(args.config)
    receipt=json.loads(args.authorization.read_text()) if args.authorization else {}
    authorize(cfg,receipt,digest,args.execute)
    verify_config(cfg)
    require(os.environ.get('SLURM_JOB_ID') and (Path.cwd()/'STAGING_MANIFEST.json').is_file(),'Staged allocated run required')
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device)
    rank=dist.get_rank()
    require(dist.get_world_size()==4 and torch.cuda.is_bf16_supported(),'Four BF16 ranks required')
    props=torch.cuda.get_device_properties(device)
    require(cfg['hardware']['device_name_substring'] in props.name
            and props.total_memory>=cfg['hardware']['minimum_vram_bytes'],'Undeclared hardware')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=build_screen(cfg).to(device).train()
    ema=build_screen(cfg).to(device).eval().requires_grad_(False)
    optimizers,names=make_optimizers(model,cfg['training'])
    require(names==cfg['training']['optimizer_parameter_names'],'Optimizer capacity differs from resolved config')
    splits=read_interface(cfg['data'])
    sampler,streams=FitQueue(splits['FIT'],0),ScreenStreams(0)
    acceptance=json.loads(Path(cfg['input']['acceptance']).read_text())
    require(acceptance['checkpoint_sha256']==cfg['input']['checkpoint_sha256'],'Bound input hash differs')
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        require(state['schema']==SCHEMA and not state.get('synthetic_fixture',False),'Not a scientific B1 continuation')
        old_digest=state['configuration_sha256']
        require(old_digest==digest or cfg.get('extension',{}).get('original_config_sha256')==old_digest,'Resume configuration not declared')
        provenance=state['provenance']
    else:
        source,actual=load_accepted(cfg['input']['checkpoint'],acceptance)
        require(source['resolved_config']['model']['gen_tok']==cfg['model']['gen_tok'],
                'Accepted A2 model semantics differ from B1 configuration')
        # Validate real FIT queue membership before accepting the inherited exposure.
        sampler.load_state_dict(source['sampler'])
        state=convert_snapshot(source,acceptance,actual,model.state_dict(),names)
        provenance=dict(input_a2_sha256=actual,acceptance=acceptance)
    checkpoint.restore(state,model,ema,optimizers,names,sampler,streams,rank)
    completed=state['b1_completed_updates']
    require(sampler.updates==5000+completed,'Resume exposure mismatch')
    history=state['evaluation_history'] if args.resume else []
    costs=state.get('costs',dict(training_seconds=0.,evaluation_seconds=0.))
    out=Path(cfg['output_parent'])/f"B1__{cfg['method_id']}__s0__{digest[:12]}"
    if rank==0:
        out.mkdir(parents=True,exist_ok=bool(args.resume))
        (out/'resolved.yaml').write_bytes(args.config.read_bytes())
        append(out/'runtime.jsonl',dict(job_id=os.environ['SLURM_JOB_ID'],stage=str(Path.cwd()),
            source_revision=os.popen('git rev-parse HEAD').read().strip(),gpu=props.name,
            cuda=torch.version.cuda,torch=torch.__version__,config_sha256=digest,authorization_sha256=sha(args.authorization)))
        ledger=(args.resume.parent if args.resume else out)/'exposure.jsonl'
        records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        highest=max([r.get('b1_update_attempted',0) for r in records]+[0])
        require(highest<=completed or args.ack_replay_through==highest,'Interrupted exposure needs exact replay acknowledgement')
        if highest>completed:
            append(out/'exposure.jsonl',dict(resume_discontinuity=True,checkpoint_update=completed,replay_through=highest))
    dist.barrier()
    ddp=DDP(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True)
    try:
        for update in range(completed+1,cfg['training']['main_updates']+1):
            start=time.perf_counter()
            indices=sampler.next(); draws=streams.draw(256); counts,count_record=streams.counts(cfg['method_id'])
            rows=[splits['FIT'][i] for i in indices]
            if rank==0:
                append(out/'exposure.jsonl',dict(b1_update_attempted=update,cumulative_update=5000+update,
                    image_ids=[r['image_id'] for r in rows],flips=draws['flip'].tolist(),exposure_chain=sampler.exposure_chain,
                    encoding_noise_sha256=hashlib.sha256(draws['encoding_noise'].numpy().tobytes()).hexdigest(),counts=count_record))
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:
                    group['lr']=1e-4
            totals=torch.zeros(2,dtype=torch.float64,device=device)
            for acc in range(4):
                local=local_indices(rank,acc)
                images=load_images([rows[i] for i in local],draws['flip'][local],device)
                labels=torch.tensor([rows[i]['class_index'] for i in local],device=device)
                with ddp.no_sync() if acc<3 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        value=ddp(images,labels,counts[local].to(device),select_draws(draws,local,device))
                    finite=torch.isfinite(value['per_image']).all().int()
                    dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                    require(finite.item(),'Nonfinite loss before optimizer update')
                    backward_mean(value['per_image']).backward()
                totals+=torch.stack([value['reconstruction'].detach().double().sum(),value['generation_sum'].detach().double().sum()])
            norm=clip_gradients([p for p in model.parameters() if p.requires_grad])
            for opt in optimizers.values():
                opt.step()
            with torch.no_grad():
                live=dict(model.named_parameters())
                for name,param in ema.named_parameters():
                    if live[name].requires_grad:
                        param.mul_(.9978).add_(live[name],alpha=.0022)
                finite=all(torch.isfinite(p).all() for p in model.parameters())
                finite=finite and all(torch.isfinite(v).all() for o in optimizers.values() for st in o.state.values()
                                     for v in st.values() if torch.is_tensor(v))
                require(finite,'Nonfinite parameter or optimizer state')
            torch.cuda.synchronize()
            costs['training_seconds']+=time.perf_counter()-start
            if rank==0:
                append(out/'exposure.jsonl',dict(b1_update_completed=update))
            if update%100==0:
                dist.all_reduce(totals)
                if rank==0:
                    append(out/'monitor.jsonl',dict(update=update,reconstruction_loss=(totals[0]/256).item(),
                        flow_loss_sum=(totals[1]/256).item(),flow_loss_mean_draw=(totals[1]/768).item(),
                        gradient_norm_before_clip=norm.item(),lr=1e-4,count_histogram=count_record['histogram']))
            if update%1000==0:
                dist.barrier(); ev=time.perf_counter()
                if rank==0:
                    summary=reconstruction(ema,eval_rows(splits['DEV'],512,cfg['evaluation']['seed']),device,
                        cfg['evaluation']['seed'],out/f'dev-{update}.json')
                    history.append(dict(update=update,per_count=summary))
                shared=[history if rank==0 else None]; dist.broadcast_object_list(shared,src=0); history=shared[0]
                dist.barrier(); costs['evaluation_seconds']+=time.perf_counter()-ev
                checkpoint.save(out/f'checkpoint-{update}.pt',model,ema,optimizers,names,sampler,streams,cfg,digest,
                                update,history,provenance,costs)
                if rank==0:
                    (out/'results.json').write_text(json.dumps(dict(node_id='B1',phase_id='SCREEN',method_id=cfg['method_id'],
                        seed=0,node_status='AWAITING_COMMON_ENDPOINT',outcome_class=None,config_sha256=digest,
                        checkpoint_sha256=sha(out/f'checkpoint-{update}.pt'),main_updates=update,images_seen=update*256,
                        training_gpu_hours=costs['training_seconds']*4/3600,gpu_hours=None,
                        gpu_hours_reason='Terminal allocation accounting required',training=dict(history=history,costs=costs),
                        reconstruction={},generation={},checkpoint_roles=dict(SCREEN_PREFIX=None,COMMON_PREFIX=None)),indent=2)+'\n')
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
