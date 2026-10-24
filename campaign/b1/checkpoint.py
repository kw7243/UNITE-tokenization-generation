"""B1 SCREEN optimizer-boundary snapshots; source RNG and sampler are preserved."""
import os
from pathlib import Path
import torch
import torch.distributed as dist
from campaign.a2.checkpoint import process_rng, restore_rng
from campaign.a2.data import sha
from campaign.b1.convert import SCHEMA, require


def save(path, model, ema, optimizers, names, sampler, streams, cfg, digest, updates, history, provenance, costs):
    rng = [None]*dist.get_world_size()
    dist.all_gather_object(rng, process_rng())
    if dist.get_rank() == 0:
        state = dict(schema=SCHEMA, checkpoint_role='B1_SCREEN_CONTINUATION', phase_id='SCREEN',
            allowed_counts=[16,64], checkpoint_roles=dict(SCREEN_PREFIX=None, COMMON_PREFIX=None),
            model=model.state_dict(), ema=ema.state_dict(), optimizers={k:v.state_dict() for k,v in optimizers.items()},
            optimizer_parameter_names=names, sampler=sampler.state_dict(), streams=streams.state_dict(),
            process_rng_by_rank=rng, configuration_sha256=digest, resolved_config=cfg,
            b1_completed_updates=updates, main_updates=5000+updates, images_seen=(5000+updates)*256,
            scheduler=dict(completed_updates=5000+updates, warmup_restarted=False, lr=1e-4),
            evaluation_history=history, provenance=provenance, costs=costs,
            boundary='AFTER_COMPLETE_OPTIMIZER_UPDATE', world_size=4)
        path=Path(path)
        with path.open('xb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        path.with_suffix(path.suffix+'.sha256').write_text(sha(path)+'\n')
    dist.barrier()


def restore(state, model, ema, optimizers, names, sampler, streams, rank):
    require(state['schema'] == SCHEMA and state['optimizer_parameter_names'] == names, 'B1 schema or optimizer names differ')
    model.load_state_dict(state['model'], strict=True)
    ema.load_state_dict(state['ema'], strict=True)
    for key, opt in optimizers.items():
        opt.load_state_dict(state['optimizers'][key])
    sampler.load_state_dict(state['sampler'])
    source = state['streams']
    if 'b1_screen_assignment' not in source['generators']:
        # The single new independent stream starts at seed0; every source stream is restored exactly.
        source = dict(generators=dict(source['generators'],
            b1_screen_assignment=streams.generators['b1_screen_assignment'].get_state()), times=source['times'])
    streams.load_state_dict(source)
    restore_rng(state['process_rng_by_rank'][rank])
