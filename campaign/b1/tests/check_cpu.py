"""Focused synthetic/meta CPU proof. No scientific updates, images, or GPU proof."""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'third_party/unite'))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from modules.encoder import Encoder
from modules.decoder import Decoder, DecoderConfig
from models.unite import UNITE
from campaign.a2.tests.check_cpu import FrozenFixture
from campaign.a2.native import backward_mean, build_native, parameter_groups as native_groups
from campaign.a2.data import Streams, FitQueue, local_indices, sha, read_interface
from campaign.b1.screen import Screen, build_screen, parameter_groups, make_optimizers, NEW_NAMES
from campaign.b1.counts import ScreenStreams
from campaign.b1.convert import convert_snapshot, validate_input
from campaign.b1.train import authorize, verify_config


class RecordBlock(nn.Module):
    def forward(self,x,c,rope):
        self.condition=c.detach().clone(); self.tokens=x.detach().clone()
        return x+c[:,None]*.01


class TinyCore(nn.Module):
    """Official encoder control flow and decoder interpolation; blocks are synthetic."""
    decode=UNITE.decode
    diffusion_generate=UNITE.diffusion_generate
    def __init__(self):
        super().__init__()
        self.num_latent_tokens=64; self.num_classes=100; self.diffusion_input_dim=32
        self.flow_steps_per_recon=self.flow_mini_batch=3
        self.perceptual_weight=self.gen_loss_weight=1.
        self.encoder=Encoder(hidden_size=768,depth=0,num_heads=12,in_channels=32,
            max_tokens=320,in_context_start=4,in_context_len=32)
        self.encoder.blocks=nn.ModuleList([RecordBlock() for _ in range(5)])
        nn.init.normal_(self.encoder.final_layer.linear.weight,std=.01)
        self.encoder_layer_norm=nn.LayerNorm(32)
        self.latent_tokens=nn.Parameter(torch.randn(1,64,768)*.01)
        self.patch_embed=PatchFixture()
        self.up_sample_decoder=nn.Linear(32,768)
        self.decoder=Decoder(DecoderConfig(hidden_size=768,patch_size=1,decoder_hidden_size=768,
            decoder_num_hidden_layers=0,decoder_num_attention_heads=12),num_patches=4)
        self.lpips_loss=FrozenFixture()
        self.cfg_norm_order='norm_first'; self.timestep_shift_alpha=.5
        self.transport=type('TransportFixture',(),dict(sample_eps=.05))()
        self.flow_sampler=SamplerFixture()


class PatchFixture(nn.Module):
    def __init__(self):
        super().__init__(); self.projection=nn.Linear(12,2*768); self.calls=0
    def forward(self,images):
        self.calls+=1
        return self.projection(images.flatten(1)).reshape(len(images),2,768)


class SamplerFixture:
    """One explicit source CFG call; no ODE integration is claimed."""
    def sample_ode(self,**kwargs):
        def sample(noise,model_fn):
            return [model_fn(noise,torch.full((len(noise),),.5,device=noise.device))]
        return sample


def equal_tree(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_tree(x,y)
    else: assert a==b


def synthetic_checkpoint(cfg):
    cfg=copy.deepcopy(cfg)
    cfg.update(node_id='A2',method_id='NATIVE64_JOINT',objective='joint')
    cfg['model'].update(native_count=64,candidate_count=64)
    names=dict(muon=['core.encoder.weight'],adamw=['core.encoder.bias'])
    cfg['training']['optimizer_parameter_names']=names
    cfg['distributed']['expected_unused_trainable_parameters']=[]
    cfgsha=hashlib.sha256((json.dumps(cfg,indent=2,sort_keys=True)+'\n').encode()).hexdigest()
    state=dict(schema='jat-a2-checkpoint-v1',checkpoint_role='A2_NATIVE64_COMPLETE',synthetic_fixture=True,
        boundary='AFTER_COMPLETE_OPTIMIZER_UPDATE',world_size=4,resolved_config=cfg,configuration_sha256=cfgsha,
        model={'core.encoder.weight':torch.ones(2,2),'core.encoder.bias':torch.ones(2)},
        ema={'core.encoder.weight':torch.ones(2,2)*.9,'core.encoder.bias':torch.ones(2)*.9},
        counters=dict(main_updates=5000,images_seen=1280000),scheduler=dict(completed_updates=5000),
        sampler=dict(queues={0:[0]},cursor={0:0},rng=__import__('random').Random(0).getstate(),
                     updates=5000,images_seen=1280000,exposure_chain='a'*64),
        streams=Streams(0).state_dict(),process_rng_by_rank=[dict(python=(1,),numpy=(1,),
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.zeros(8,dtype=torch.uint8)) for _ in range(4)],
        evaluation_history=[dict(update=5000,synthetic=True)],optimizer_parameter_names=names,
        optimizers=dict(muon=dict(state={0:dict(momentum_buffer=torch.ones(2,2)*.3)},
            param_groups=[dict(params=[0],lr=1e-4,momentum=.95)]),
            adamw=dict(state={0:dict(step=torch.tensor(5000.),exp_avg=torch.ones(2)*.2,exp_avg_sq=torch.ones(2)*.4)},
                param_groups=[dict(params=[0],lr=1e-4,betas=(.9,.999),eps=1e-6,weight_decay=0.)])))
    receipt=dict(synthetic_fixture=True,checkpoint_sha256=None,a2_native64_accepted=True,stable_control=True,
        scientific_training_completed=True,a2_configuration_sha256=cfgsha,a2_patch_sha256=cfg['source']['patch_sha256'],
        source_config_history=[dict(source_revision='synthetic_fixture',config_sha256=cfgsha,patch_sha256=cfg['source']['patch_sha256'])])
    return state,receipt


def run(output):
    start=time.perf_counter()
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    checks={}
    index=json.loads((ROOT/'campaign/b1/configs/index.json').read_text())
    configs={name:json.loads((ROOT/e['path']).read_text()) for name,e in index.items()}
    for name,cfg in configs.items():
        assert sha(ROOT/index[name]['path'])==index[name]['sha256']
        verify_config(cfg)
        assert cfg['execution_authorized'] is False and cfg['input']['checkpoint_sha256'] is None
        for execute in (False,True):
            try: authorize(cfg,{},index[name]['sha256'],execute)
            except ValueError: pass
            else: raise AssertionError('Disabled config executed')
    cfg=configs['NESTED_RANDOM']
    splits=read_interface(cfg['data'])
    checks['configs_sources_data_sealed_confirm']=dict(passed=True,configs=index,rows={k:len(v) for k,v in splits.items()},confirm_opened=False)

    streams=ScreenStreams(0); source=Streams(0)
    x=streams.draw(256); y=Streams(0).draw(256)
    equal_tree(x,y)
    saved=copy.deepcopy(streams.state_dict())
    c,r=streams.counts('NESTED_RANDOM')
    assert r['histogram']==[0,128,128] and r['distributed_microbatch_histograms']==[[0,32,32]]*4
    assert r['mean_tokens']==40 and r['mean_tokens_sq']==2176
    replay=ScreenStreams(91); replay.load_state_dict(saved)
    c2,r2=replay.counts('NESTED_RANDOM'); equal_tree(c,c2); assert r==r2
    changed=ScreenStreams(0); changed.counts('PREFIX16'); equal_tree(changed.draw(256),x)
    for arm,k in [('PREFIX16',16),('PREFIX64',64)]:
        a,_=ScreenStreams(0).counts(arm); assert (a==k).all()
    queue=FitQueue(splits['FIT'],0); queue.next(); queue_state=queue.state_dict()
    clone=FitQueue(splits['FIT'],99); clone.load_state_dict(queue_state); assert queue.next()==clone.next()
    (output/'quota-stream.json').write_text(json.dumps(r,indent=2)+'\n')
    checks['exact_quota_independent_stream_replay']=dict(passed=True,mean_tokens=40,mean_tokens_sq=2176)

    torch.manual_seed(19)
    model=Screen(TinyCore()).train()
    assert not model.core.lpips_loss.training and all(not p.requires_grad for p in model.core.lpips_loss.parameters())
    for name in NEW_NAMES:
        assert not torch.count_nonzero(dict(model.named_parameters())[name])
    with torch.no_grad():
        for name in NEW_NAMES:
            dict(model.named_parameters())[name].normal_(0,.03)
    images=torch.rand(3,3,2,2); labels=torch.tensor([3,4,5]); counts=torch.tensor([16,64,16]); draws=Streams(0).draw(3)
    z=model.encode(images,draws['encoding_noise'],draws['encoding_time'])
    assert z.shape==(3,64,32) and model.core.patch_embed.calls==1
    rec,_=model.reconstruction(z,images,counts,draws)
    gen=model.generation(z,labels,counts,draws)
    loopr=[];loopg=[]
    for i in range(3):
        local={k:v[i:i+1] for k,v in draws.items()}
        a,_=model.reconstruction(z[i:i+1],images[i:i+1],counts[i:i+1],local);loopr.append(a)
        loopg.append(model.generation(z[i:i+1],labels[i:i+1],counts[i:i+1],local))
    torch.testing.assert_close(rec,torch.cat(loopr),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(gen,torch.cat(loopg),rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(model.image_mean(rec+gen.sum(1)),(torch.cat(loopr)+torch.cat(loopg).sum(1)).sum()/3)
    assert not torch.isclose((rec[counts==16].mean()+rec[counts==64].mean())/2,rec.mean(),atol=1e-8,rtol=0)
    assert torch.autograd.grad(gen.sum(),z,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(rec.sum(),z,retain_graph=True)[0].norm()>0
    assert torch.autograd.grad(gen.sum(),model.core.encoder.final_layer.linear.weight,retain_graph=True)[0].norm()>0
    # Perturb discarded positions, then compare both loss and decoded output.
    changed=z.detach().clone(); changed[0,16:]+=100; changed[2,16:]-=100
    torch.testing.assert_close(model.decode(z,counts),model.decode(changed,counts),rtol=0,atol=0)
    torch.testing.assert_close(gen,model.generation(changed,labels,counts,draws),rtol=0,atol=0)
    for cts in (torch.tensor([16,16,16]),torch.tensor([64,64,64])):
        assert model.decode(z,cts).shape==(3,3,2,2)
    try: model.decode(z[:0],counts[:0])
    except ValueError: pass
    else: raise AssertionError('Empty logical batch not refused')
    # Explicit equation: k*d normalization, time weight, and sum over three draws.
    expected=[]
    for i,k in enumerate(counts.tolist()):
        clean=z[i:i+1,:k].detach().expand(3,-1,-1)
        t=draws['diffusion_time'][i]; xt=t[:,None,None]*clean+(1-t[:,None,None])*draws['diffusion_noise'][i,:,:k]
        with model._length(k):
            pred=model.normalize(model.core.encoder(xt,t=t,y=labels[i].expand(3),
                pos_embed=model.core.latent_tokens[:,:k].expand(3,-1,-1),
                force_drop_ids_y_embedder=draws['class_dropout'][i].long()))
        expected.append((pred-clean).square().mean((1,2))/(1-t).clamp_min(.05).square())
    torch.testing.assert_close(gen,torch.stack(expected),rtol=2e-5,atol=2e-5)
    checks['synthetic_grouping_order_loss_detach']=dict(passed=True,counts=[16,64,16],empty_count_groups_skipped=True,
        empty_logical_batch_refused=True,draws_summed=3,encode_calls=1,limitation='Synthetic blocks, not full-model numerical proof')

    # Time+class+length condition; context remains class-only even with class dropout.
    model.eval(); noised=draws['encoding_noise'][:1,:16]; t=torch.tensor([.4]); y=torch.tensor([2])
    with torch.no_grad(),model._length(16):
        model.core.encoder(noised,t=t,y=y,pos_embed=model.core.latent_tokens[:,:16],force_drop_ids_y_embedder=torch.ones(1,dtype=torch.long))
        context=model.core.encoder.blocks[4].tokens[:,:32].clone()
        cond=model.core.encoder.blocks[0].condition.clone()
        model.encoder_length.weight[1]+=1
        model.core.encoder(noised,t=t,y=y,pos_embed=model.core.latent_tokens[:,:16],force_drop_ids_y_embedder=torch.ones(1,dtype=torch.long))
        torch.testing.assert_close(model.core.encoder.blocks[0].condition-cond,torch.ones_like(cond),rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(context,model.core.encoder.blocks[4].tokens[:,:32],rtol=0,atol=0)
    with torch.no_grad(),model._length(16):
        xx=noised.expand(2,-1,-1); tt=t.expand(2); yy=torch.tensor([2,100])
        model.core.encoder.forward_with_cfg(xx,tt,yy,model.core.latent_tokens[:,:16].expand(2,-1,-1),2.,
            bn_func=model.normalize,cfg_norm_order='norm_first')
        c=model.core.encoder.blocks[0].condition
        base=model.core.encoder.t_embedder(tt)-model.encoder_length.weight[1]
        yc=model.core.encoder.y_embedder(yy,False)
        torch.testing.assert_close(c-base-yc,model.encoder_length.weight[1].expand(2,-1),rtol=1e-5,atol=1e-6)
    # Encoding has no length condition, including after a generation call.
    before=model.encode(images,draws['encoding_noise'],draws['encoding_time'])
    with torch.no_grad(): model.encoder_length.weight.add_(17)
    after=model.encode(images,draws['encoding_noise'],draws['encoding_time'])
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert list(inspect.signature(model.generate).parameters)==['labels','counts','noise','cfg_scale','cfg_interval']
    def forbid(*unused): raise AssertionError('Generation used image patches')
    hook=model.core.patch_embed.register_forward_pre_hook(forbid)
    model.generate(labels,counts,draws['encoding_noise']); hook.remove()
    checks['synthetic_length_class_dropout_cfg_boundary']=dict(passed=True,table_order=[4,16,64],
        cfg_halves_same_length=True,encoder_selected_count_absent=True,source_ode_exercised=False)

    x=torch.linspace(.1,2,256); w=torch.tensor(.4,requires_grad=True)
    baseline=torch.autograd.grad(((w*x-1)**2).mean(),w)[0]
    grads=[]
    for rank in range(4):
        w=torch.tensor(.4,requires_grad=True)
        losses=[backward_mean((w*x[local_indices(rank,a)]-1)**2) for a in range(4)]
        grads.append(torch.autograd.grad(sum(losses),w)[0])
    torch.testing.assert_close(torch.stack(grads).mean(),baseline)
    checks['synthetic_ddp_image_weighted_math']=dict(passed=True,distributed_runtime_exercised=False)

    # Full Base/Base metadata, source optimizer partition, and identical arm capacity.
    structures={}
    for arm in configs:
        with patch('models.unite.LPIPS',FrozenFixture),torch.device('meta'):
            full=build_screen(configs[arm]).to('meta')
        groups,names=parameter_groups(full)
        assert names==configs[arm]['training']['optimizer_parameter_names']
        assert all(p.is_meta for p in full.parameters())
        opts,_=make_optimizers(full,configs[arm]['training']); assert all(not o.state for o in opts.values())
        structures[arm]=dict(names=names,trainable_parameters=sum(p.numel() for p in full.parameters() if p.requires_grad),
            new_parameters=sum(dict(full.named_parameters())[n].numel() for n in NEW_NAMES))
        del full,opts
    assert len({v['trainable_parameters'] for v in structures.values()})==1
    (output/'optimizer-structure.json').write_text(json.dumps(structures,indent=2)+'\n')
    checks['full_model_meta_optimizer_ema_capacity']=dict(passed=True,new_parameters=3*3*768,
        limitation='Metadata tensors and frozen LPIPS fixture; no full-model CPU/GPU forward')

    state,receipt=synthetic_checkpoint(cfg)
    fixture=output/'EXPLICITLY_SYNTHETIC_A2_FIXTURE.pt'; torch.save(state,fixture)
    digest=sha(fixture); receipt['checkpoint_sha256']=digest
    target=copy.deepcopy(state['model']);target.update({n:torch.zeros(3,768) for n in NEW_NAMES})
    names=copy.deepcopy(state['optimizer_parameter_names']); names['adamw']+=list(NEW_NAMES)
    converted=convert_snapshot(state,receipt,digest,target,names,synthetic=True)
    equal_tree(state['optimizers']['muon'],converted['optimizers']['muon'])
    equal_tree(state['optimizers']['adamw']['state'][0],converted['optimizers']['adamw']['state'][0])
    for key in ('model','ema'):
        for n,value in state[key].items(): equal_tree(value,converted[key][n])
    for slot in (1,2,3):
        assert converted['optimizers']['adamw']['state'][slot]['step']==0
        assert not converted['optimizers']['adamw']['state'][slot]['exp_avg'].count_nonzero()
        assert not converted['optimizers']['adamw']['state'][slot]['exp_avg_sq'].count_nonzero()
    for name in NEW_NAMES: assert not converted['ema'][name].count_nonzero()
    assert converted['checkpoint_roles']==dict(SCREEN_PREFIX=None,COMMON_PREFIX=None)
    # Load real AdamW state mechanics on synthetic tensors; no optimizer.step call.
    params=[nn.Parameter(torch.zeros(2))]+[nn.Parameter(torch.zeros(3,768)) for _ in NEW_NAMES]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-6,weight_decay=0)
    opt.load_state_dict(converted['optimizers']['adamw'])
    assert opt.state[params[0]]['step']==5000 and opt.state[params[1]]['step']==0
    failures=[]
    mutations={
        'engineering_complete_role':lambda s:s.update(engineering_only=True),
        'generic_complete_role_2_updates':lambda s:s['counters'].update(main_updates=2),
        'missing_optimizer':lambda s:s['optimizers'].pop('muon'),
        'wrong_native':lambda s:s['resolved_config']['model'].update(native_count=16),
        'missing_ema':lambda s:s['ema'].pop('core.encoder.bias'),
        'missing_rng':lambda s:s['process_rng_by_rank'][0].update(torch_cuda=None),
        'missing_names':lambda s:s['optimizer_parameter_names']['adamw'].clear(),
        'missing_moment':lambda s:s['optimizers']['adamw']['state'][0].pop('exp_avg'),
        'wrong_schema':lambda s:s.update(schema='jat-b1-screen-checkpoint-v1'),
        'wrong_boundary':lambda s:s.update(boundary='MID_UPDATE'),
        'incomplete_exposure':lambda s:s['sampler'].update(images_seen=1),
        'nonfinite':lambda s:s['model']['core.encoder.bias'].fill_(float('nan'))}
    for name,mutate in mutations.items():
        bad=copy.deepcopy(state); mutate(bad)
        try: validate_input(bad,receipt,digest,synthetic=True)
        except (ValueError,KeyError): failures.append(name)
        else: raise AssertionError('Bad checkpoint accepted: '+name)
    for key,value in [('checkpoint_sha256','b'*64),('source_config_history',[])]:
        bad=copy.deepcopy(receipt);bad[key]=value
        try: validate_input(state,bad,digest,synthetic=True)
        except ValueError: failures.append(key)
        else: raise AssertionError('Bad acceptance accepted')
    try: validate_input(state,receipt,digest,synthetic=False)
    except ValueError: failures.append('synthetic_cannot_be_scientific')
    else: raise AssertionError('Synthetic fixture accepted as scientific')
    torch.save(converted,output/'EXPLICITLY_SYNTHETIC_B1_START.pt')
    checks['synthetic_strict_conversion_refusals']=dict(passed=True,rejected=failures,
        fixture_sha256=digest,scientific_conversion=False,real_resume_proven=False,optimizer_updates=0)
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    (output/'packages.txt').write_text(packages)
    locked=Path(cfg['environment']['packages']).read_text().splitlines()
    assert [line for line in packages.splitlines() if not line.startswith(('pip==','setuptools=='))]==locked
    for path in (ROOT/'campaign/b1').rglob('*.py'):
        compile(path.read_text(),str(path),'exec')
    record=dict(passed=True,preparation_status='PASS',b1_status='NOT_STARTED',checks=checks,
        job_id=os.environ['SLURM_JOB_ID'],stage=str(Path.cwd()),node=platform.node(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        elapsed_seconds=time.perf_counter()-start,main_updates=0,images_seen=0,gpu_hours=0.,
        environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,visible_gpus=torch.cuda.device_count(),
            locked_package_sha256=cfg['environment']['packages_sha256'],frozen_packages_sha256=sha(output/'packages.txt'),unchanged=True))
    (output/'checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),output=str(output),seconds=record['elapsed_seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
