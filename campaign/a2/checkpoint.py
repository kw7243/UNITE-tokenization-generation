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


def load(path, model, ema, optimizers, names, sampler, streams, config_sha):
    # Only load this research branch's own trusted snapshots, never arbitrary uploads.
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['schema'] == SCHEMA and state['configuration_sha256'] == config_sha
    assert state['world_size'] == dist.get_world_size() == 4
    assert state['optimizer_parameter_names'] == names
    assert state['boundary'] == 'AFTER_COMPLETE_OPTIMIZER_UPDATE'
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
