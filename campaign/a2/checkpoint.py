"""Optimizer-boundary snapshots for A2, with exact configuration and exposure checks."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

SCHEMA = 'jat-a2-checkpoint-v1'


def process_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda'])


def save(path, model, ema, optimizers, names, sampler, streams, config_sha, cfg, counters, history):
    assert sampler.updates == counters['main_updates']
    assert sampler.images_seen == counters['images_seen'] == sampler.updates*256
    local_rng = process_rng()
    rank_rng = [None]*dist.get_world_size()
    dist.all_gather_object(rank_rng, local_rng)
    if dist.get_rank() == 0:
        snapshot = dict(schema=SCHEMA, configuration_sha256=config_sha, resolved_config=cfg,
                        model=model.state_dict(), ema=ema.state_dict(),
                        optimizers={k: o.state_dict() for k, o in optimizers.items()},
                        optimizer_parameter_names=names, sampler=sampler.state_dict(),
                        streams=streams.state_dict(), process_rng_by_rank=rank_rng,
                        counters=counters, evaluation_history=history,
                        checkpoint_role='A2_NATIVE64_COMPLETE' if model.count == 64 else 'A2_NATIVE16_COMPLETE',
                        scheduler=dict(completed_updates=sampler.updates, warmup=500, after_warmup='constant'),
                        boundary='AFTER_COMPLETE_OPTIMIZER_UPDATE', world_size=4,
                        b1_conversion_implemented=False)
        path = Path(path)
        assert not path.exists(), f'Preserve existing checkpoint: {path}'
        temporary = path.with_name(path.name+f'.partial-{os.getpid()}')
        assert not temporary.exists()
        with temporary.open('xb') as stream:
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(8 << 20), b''):
                h.update(part)
        path.with_suffix(path.suffix+'.sha256').write_text(h.hexdigest()+'\n')
    dist.barrier()


def validate_state(state, cfg, config_sha, expected_update=None):
    """Check the complete saved contract before mutating live training state."""
    from campaign.a2.data import Streams
    assert state['schema'] == SCHEMA and state['configuration_sha256'] == config_sha
    assert state['resolved_config'] == cfg and state['world_size'] == 4
    assert state['boundary'] == 'AFTER_COMPLETE_OPTIMIZER_UPDATE'
    update = state['counters']['main_updates']
    if expected_update is not None:
        assert update == expected_update
    assert state['sampler']['updates'] == update
    assert state['counters']['images_seen'] == state['sampler']['images_seen'] == update*256
    assert state['scheduler'] == dict(completed_updates=update, warmup=500, after_warmup='constant')
    if update == 5000:
        assert state['checkpoint_role'] == f'A2_NATIVE{cfg["model"]["native_count"]}_COMPLETE'
    assert set(state['model']) == set(state['ema'])
    for name, value in state['model'].items():
        other = state['ema'][name]
        assert value.shape == other.shape and value.dtype == other.dtype
        assert torch.isfinite(value).all() and torch.isfinite(other).all()
    assert state['optimizer_parameter_names'] == cfg['training']['optimizer_parameter_names']
    assert set(state['optimizers']) == {'muon', 'adamw'}
    assert set(state['sampler']) == {'queues', 'cursor', 'rng', 'updates', 'images_seen', 'exposure_chain'}
    assert set(state['sampler']['queues']) == set(state['sampler']['cursor']) == set(range(100))
    assert len(state['process_rng_by_rank']) == 4
    for rng in state['process_rng_by_rank']:
        assert set(rng) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
        for name in ('torch_cpu', 'torch_cuda'):
            assert torch.is_tensor(rng[name]) and rng[name].dtype == torch.uint8 and rng[name].numel() > 0
    assert set(state['streams']) == {'generators', 'times'}
    assert set(state['streams']['generators']) == set(Streams.names)
    assert all(torch.is_tensor(v) and v.dtype == torch.uint8 and v.numel() > 0
               for v in state['streams']['generators'].values())
    history = state['evaluation_history']
    assert [r['update'] for r in history] == list(range(1000, update+1, 1000))
    assert all(r['images'] == 512 for r in history)
    assert np.isfinite([r[k] for r in history for k in ('recon_l1', 'recon_lpips',
        'allocation_distortion', 'latent_image_variance', 'latent_rms')]).all()


def load(path, model, ema, optimizers, names, sampler, streams, config_sha,
         *, cfg=None, expected_update=None, expected_sha=None):
    # Only load this research branch's own trusted snapshots, never arbitrary uploads.
    from campaign.a2.data import sha
    from campaign.a2.native import executed_lr
    if expected_sha is not None:
        assert sha(path) == expected_sha, 'Parent checkpoint bytes changed'
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_state(state, state['resolved_config'] if cfg is None else cfg, config_sha, expected_update)
    assert state['world_size'] == dist.get_world_size() == 4
    assert state['optimizer_parameter_names'] == names
    for module, saved in ((model, state['model']), (ema, state['ema'])):
        live = module.state_dict()
        assert live.keys() == saved.keys()
        assert all(live[k].shape == saved[k].shape and live[k].dtype == saved[k].dtype for k in live)
    for key, optimizer in optimizers.items():
        saved = state['optimizers'][key]
        groups = optimizer.state_dict()['param_groups']
        assert len(saved['param_groups']) == len(groups)
        parameters = [p for g in saved['param_groups'] for p in g['params']]
        assert len(set(parameters)) == len(parameters) and set(saved['state']) <= set(parameters)
        for group, reference in zip(saved['param_groups'], groups):
            assert {k:v for k,v in group.items() if k != 'lr'} == {k:v for k,v in reference.items() if k != 'lr'}
            assert group['lr'] == executed_lr(state['counters']['main_updates'])
        expected_ids = {i for i, name in zip(parameters, names[key])
                        if name not in state['resolved_config']['distributed']['expected_unused_trainable_parameters']}
        assert len(parameters) == len(names[key]) and set(saved['state']) == expected_ids
        live_parameters = [p for group in optimizer.param_groups for p in group['params']]
        by_id = dict(zip(parameters, live_parameters))
        for parameter_id, slot in saved['state'].items():
            assert set(slot) == ({'momentum_buffer'} if key == 'muon' else {'step', 'exp_avg', 'exp_avg_sq'})
            for slot_name, value in slot.items():
                assert torch.is_tensor(value) and torch.isfinite(value).all()
                if slot_name == 'step':
                    assert value.numel() == 1 and value.item() == state['counters']['main_updates']
                else:
                    assert value.shape == by_id[parameter_id].shape and value.dtype == by_id[parameter_id].dtype
    model.load_state_dict(state['model'], strict=True)
    ema.load_state_dict(state['ema'], strict=True)
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(state['optimizers'][key])
    sampler.load_state_dict(state['sampler'])
    streams.load_state_dict(state['streams'])
    assert sampler.updates == state['counters']['main_updates']
    assert sampler.images_seen == state['counters']['images_seen'] == sampler.updates*256
    restore_rng(state['process_rng_by_rank'][dist.get_rank()])
    return state['counters'], state['evaluation_history']
