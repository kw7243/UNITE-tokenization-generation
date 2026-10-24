"""Result-critical CPU structure, sampler, and tiny-tensor checks. No main updates."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from campaign.a2.data import FitQueue, Streams, local_indices, read_interface, sha
from campaign.a2.native import NativeA2, backward_mean, build_native, executed_lr, make_optimizers, parameter_groups, slice_register_table
from campaign.a2.train import verify_config
from campaign.a2.evaluate import generation_eligible, paired_interference_interval, generation_inputs


class FrozenFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)
        self.dropout = nn.Dropout()
    def forward(self, real, decoded, reduction='none'):
        value = (real-decoded).square().flatten(1).mean(1)
        return value[:, None, None, None]


class TinyFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
    def forward(self, x, condition):
        return self.linear(x)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = TinyFinal()
    def forward(self, x, t, y, pos_embed, img_patch_embed=None, **unused):
        if img_patch_embed is not None:
            x = x+img_patch_embed.mean(1, keepdim=True)
            x = torch.cat([x, img_patch_embed], 1)
        return self.final_layer(x, t)


class TinyCore(nn.Module):
    """A tensor boundary fixture, not a scaled UNITE model or quality test."""
    def __init__(self, count=16):
        super().__init__()
        self.num_latent_tokens = count
        self.flow_steps_per_recon = self.flow_mini_batch = 3
        self.perceptual_weight = self.gen_loss_weight = 1.
        self.encoder = TinyEncoder()
        self.encoder_layer_norm = nn.LayerNorm(32)
        self.latent_tokens = nn.Parameter(torch.randn(1, count, 32))
        self.patch_projection = nn.Linear(12, 64)
        self.decoder = nn.Linear(32, 12)
        self.lpips_loss = FrozenFixture()
    def patch_embed(self, images):
        return self.patch_projection(images.flatten(1)).reshape(len(images), 2, 32)
    def decode(self, z, representation_phase=False):
        return self.decoder(z.mean(1)).reshape(len(z), 3, 2, 2)


def run(output):
    start = time.perf_counter()
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file()
    torch.set_num_threads(4)
    results = {}
    index = json.loads((ROOT/'campaign/a2/configs/index.json').read_text())
    configs = {method: json.loads((ROOT/entry['path']).read_text()) for method, entry in index.items()}
    for method, cfg in configs.items():
        assert sha(ROOT/index[method]['path']) == index[method]['sha256']
        verify_config(cfg)
        assert cfg['execution_authorized'] is False
        assert cfg['training']['main_lr'] == 1e-4
        assert not cfg['model']['selector'] and not cfg['model']['length_embeddings']
        assert cfg['conditional'] == method.endswith('RECON')
    assert abs(executed_lr(1)-2e-7)<1e-20 and executed_lr(500) == executed_lr(5000) == 1e-4
    results['resolved_configs_and_schedule'] = dict(passed=True, configs=index)

    cfg = configs['NATIVE64_JOINT']
    splits = read_interface(cfg['data'])
    sampler = FitQueue(splits['FIT'], 0)
    for _ in range(20):
        indices = sampler.next()
        labels = [splits['FIT'][i]['class_index'] for i in indices]
        assert sorted(Counter(labels).values()) == [8]*32
        for acc in range(4):
            block = labels[acc*64:(acc+1)*64]
            assert sorted(Counter(block).values()) == [8]*8
            for rank in range(4):
                local = [labels[i] for i in local_indices(rank, acc)]
                assert sorted(Counter(local).values()) == [8, 8]
    results['immutable_fit_dev_alloc_and_balance'] = dict(passed=True, counts={k: len(v) for k,v in splits.items()},
                                                        dry_updates=20, confirm_opened=False)

    tiny_rows = [dict(image_id=f'{c}/{i}', class_index=c) for c in range(100) for i in range(13)]
    a, streams = FitQueue(tiny_rows, 0), Streams(0)
    for _ in range(19):
        a.next()
        streams.draw(256)
    state, random_state = a.state_dict(), copy.deepcopy(streams.state_dict())
    b, streams_b = FitQueue(tiny_rows, 999), Streams(999)
    b.load_state_dict(state)
    streams_b.load_state_dict(random_state)
    for _ in range(12):
        assert a.next() == b.next()
        x, y = streams.draw(256), streams_b.draw(256)
        assert all(torch.equal(x[k],y[k]) for k in x)
        assert a.exposure_chain == b.exposure_chain
    assert a.images_seen == b.images_seen == 31*256
    results['queue_exhaustion_and_resume_exposure'] = dict(passed=True, images_seen=a.images_seen, exposure_chain=a.exposure_chain)

    x, y = Streams(0).draw(256), Streams(0).draw(256)
    assert all(torch.equal(x[k], y[k]) for k in x)
    assert not torch.equal(x['encoding_noise'], x['recon_noise'])
    assert x['diffusion_noise'].shape == (256,3,64,32)
    for rank in range(4):
        for acc in range(4):
            local = local_indices(rank, acc)
            for k in (16,64):
                assert torch.equal(x['encoding_noise'][local,:k], y['encoding_noise'][local,:k])
                assert torch.equal(x['diffusion_noise'][local,:,:k], y['diffusion_noise'][local,:,:k])
    s = Streams(0)
    original=s.draw(4)
    changed=Streams(0)
    torch.rand(400, generator=changed.generators['assignment_shuffle'])
    after = changed.draw(4)
    assert all(torch.equal(original[k], after[k]) for k in original)
    results['paired_full_noise_and_independent_streams'] = dict(passed=True, full_image_order=True, full_count=64, draws=3)

    core64 = SimpleNamespace(num_latent_tokens=64, latent_tokens=nn.Parameter(torch.randn(1,64,8)),
                              common=nn.Parameter(torch.randn(8,8)))
    core16 = copy.deepcopy(core64)
    common = core16.common.detach().clone()
    slice_register_table(core16,16)
    assert torch.equal(core16.common,common) and torch.equal(core16.common,core64.common)
    assert torch.equal(core16.latent_tokens, core64.latent_tokens[:,:16])
    assert core64.latent_tokens.shape == (1,64,8)
    results['common_init_mapping_tiny_bytes'] = dict(passed=True, only_sliced_parameter='latent_tokens',
        limitation='Full-size initialized byte equality remains a GPU prerequisite; no large CPU numerical model is constructed.')

    meta_shapes = {}
    for method in ('NATIVE64_JOINT','NATIVE16_JOINT'):
        with patch('models.unite.LPIPS', FrozenFixture), torch.device('meta'):
            model = build_native(configs[method]).to('meta')
        groups,names = parameter_groups(model)
        assert names == configs[method]['training']['optimizer_parameter_names']
        optimizers,_ = make_optimizers(model, configs[method]['training'])
        for key, optimizer in optimizers.items():
            for name, expected in configs[method]['training']['optimizer_internal_defaults'][key].items():
                actual = optimizer.param_groups[0][name]
                assert (list(actual) if isinstance(actual, tuple) else actual) == expected, (key,name,actual,expected)
            assert not optimizer.state, 'No main optimizer update is permitted'
        del optimizers
        assert all(p.is_meta for p in model.parameters())
        assert not any('lpips_loss' in name for group in names.values() for name in group)
        assert model.core.latent_tokens.shape[1] == configs[method]['model']['native_count']
        assert not any('length' in name for name,_ in model.named_parameters())
        meta_shapes[method] = dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  group_sizes={k: len(v) for k,v in names.items()},
                                  group_numel={k: sum(p.numel() for p in v) for k,v in groups.items()},
                                  names=names)
        del model
    results['full_model_meta_parameter_membership'] = dict(passed=True, methods=meta_shapes,
        limitation='Meta tensors validate names/shapes and exact source partition only; frozen LPIPS replaced only in this structural fixture.')

    torch.manual_seed(31)
    model=NativeA2(TinyCore()).train()
    assert not model.core.lpips_loss.training and not any(p.requires_grad for p in model.core.lpips_loss.parameters())
    images=torch.rand(3,3,2,2)
    draws=Streams(0).draw(3)
    labels=torch.tensor([0,1,2])
    z=model.encode(images,draws['encoding_noise'],draws['encoding_time'])
    rec,_=model.reconstruction(z,images,draws)
    gen=model.generation(z,labels,draws)
    loop_rec=[];loop_gen=[]
    for i in range(3):
        d={k:v[i:i+1] for k,v in draws.items()}
        r,_=model.reconstruction(z[i:i+1],images[i:i+1],d)
        loop_rec.append(r)
        loop_gen.append(model.generation(z[i:i+1],labels[i:i+1],d))
    torch.testing.assert_close(rec,torch.cat(loop_rec),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(gen,torch.cat(loop_gen),atol=1e-5,rtol=1e-6)
    assert torch.autograd.grad(gen.sum(),z,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(rec.sum(),z,retain_graph=True)[0].norm() > 0
    encoder_grad=torch.autograd.grad(gen.sum(),list(model.core.encoder.parameters()),allow_unused=True)
    assert any(g is not None and g.norm()>0 for g in encoder_grad)
    target=z.detach()
    noise=draws['diffusion_noise'][:,:,:16].reshape(9,16,32)
    t=draws['diffusion_time'].flatten()
    clean=target[:,None].expand(3,3,16,32).reshape(9,16,32)
    noised=t[:,None,None]*clean+(1-t[:,None,None])*noise
    prediction=model.normalize(model.core.encoder(noised,t=t,y=labels.repeat_interleave(3),
        pos_embed=model.core.latent_tokens.expand(9,-1,-1)))
    expected=((prediction-clean).square().mean((1,2))/(1-t).clamp_min(.05).square()).reshape(3,3).sum(1)
    torch.testing.assert_close(gen,expected,atol=1e-5,rtol=1e-6)
    results['tiny_loss_sum_detach_and_frozen_lpips'] = dict(passed=True, source_draw_sum=3, main_optimizer_updates=0)

    x=torch.linspace(.1,2,256)
    w=torch.tensor(.4,requires_grad=True)
    global_grad=torch.autograd.grad(((w*x-1)**2).mean(),w)[0]
    rank_grad=[]
    for rank in range(4):
        w=torch.tensor(.4,requires_grad=True)
        loss=sum(backward_mean((w*x[local_indices(rank,acc)]-1)**2) for acc in range(4))
        rank_grad.append(torch.autograd.grad(loss,w)[0])
    torch.testing.assert_close(torch.stack(rank_grad).mean(),global_grad,atol=1e-7,rtol=1e-6)
    results['ddp_accumulation_image_mean'] = dict(passed=True,world_size=4,accumulation=4,micro=16)

    assert not generation_eligible({})
    assert not generation_eligible(dict(NATIVE16_JOINT=[2.,1.],NATIVE64_JOINT=[2.,float('nan')]))
    assert generation_eligible(dict(NATIVE16_JOINT=[2.,1.],NATIVE64_JOINT=[3.,1.]))
    interval=paired_interference_interval(np.ones(3200)*2,np.ones(3200),np.ones(3200),np.ones(3200),np.repeat(np.arange(100),32))
    assert interval['ratio']==2 and interval['joint_pressure_screen']
    guard=subprocess.run([sys.executable,str(ROOT/'campaign/a2/train.py'),'--config',str(ROOT/index['NATIVE64_JOINT']['path'])],capture_output=True,text=True)
    assert guard.returncode==2 and 'Training requires later coordinator authorization' in guard.stderr
    results['scientific_gates_and_execution_boundary'] = dict(passed=True, training_refused_without_later_authorization=True)

    output.mkdir(parents=True, exist_ok=False)
    labels, noise = generation_inputs()
    assert torch.equal(labels, torch.arange(100).repeat_interleave(50))
    np.save(output/'generation-labels.npy',labels.numpy())
    np.save(output/'generation-noise.npy',noise.numpy())
    (output/'common-generation-inputs.json').write_text(json.dumps(dict(seed=20261023,labels_sha256=sha(output/'generation-labels.npy'),noise_sha256=sha(output/'generation-noise.npy'),shape=list(noise.shape),dtype='float32',scope='Random inputs only;zero model generation or feature extraction'),indent=2)+'\n')
    versions = {name:importlib.metadata.version(name) for name in configs['NATIVE64_JOINT']['environment']['package_versions']}
    record=dict(preparation_status='PREPARATION_ONLY',checks=results,passed=True,main_updates=0,images_seen=0,
                gpu_hours=0.,cpu_core_hours=None,cpu_core_hours_reason='Awaiting terminal Slurm allocation accounting.',
                job_id=os.environ['SLURM_JOB_ID'],node=platform.node(),stage=str(Path.cwd()),
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                environment=dict(python=sys.version,versions=versions,torch_cuda=torch.version.cuda,
                                 visible_gpus=torch.cuda.device_count(),cpus=os.environ.get('SLURM_CPUS_PER_TASK')),
                elapsed_seconds=time.perf_counter()-start,
                scope='CPU structural checks,immutable manifest and sampler dry runs,tiny synthetic tensors;no image decoding or scientific scoring',
                a1_prerequisite='Still pending under separate owner;A2 not scientifically passed.')
    (output/'optimizer-structure.json').write_text(json.dumps({k: v['names'] for k,v in meta_shapes.items()},indent=2)+'\n')
    (output/'checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
    (output/'packages.txt').write_text(packages)
    locked=Path(cfg['environment']['packages']).read_text().splitlines()
    normalized=[line for line in packages.splitlines() if not line.startswith(('pip==','setuptools=='))]
    assert normalized == locked, 'Package content changed beyond freeze --all bootstrap tools'
    (output/'environment-sha256.json').write_text(json.dumps(dict(packages_sha256=sha(output/'packages.txt'),
                            exact_locked_package_content_match=True, freeze_difference='--all additionally records pip25.3 and setuptools79.0.1; no environment mutation',
                            locked_packages_sha256=cfg['environment']['packages_sha256']),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(results),output=str(output),elapsed_seconds=record['elapsed_seconds'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
