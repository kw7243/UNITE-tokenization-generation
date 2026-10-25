"""Focused continuation metadata contracts; synthetic checks are not science."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))
from campaign.a2 import continuation as contract


def rejects(function):
    try:
        function()
    except (AssertionError, KeyError):
        return
    raise AssertionError('Incompatible recovery metadata was accepted')


def write_lock(path, value, rows=False):
    text = ''.join(json.dumps(v)+'\n' for v in value) if rows else json.dumps(value)
    path.write_text(text)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def event(number):
    return dict(optimizer_update_attempted=number, image_ids=['synthetic-fixture']*256,
                flips=[False]*256, exposure_chain='a'*64, encoding_noise_sha256='b'*64)


def ledger(first, last):
    return [v for n in range(first, last+1) for v in (event(n), {'optimizer_update_completed': n})]


def check_snapshot_contract(parent_config):
    """Tiny tensor/optimizer fixtures; CUDA RNG is validated separately on real GPUs."""
    import random
    import numpy as np
    import torch
    from campaign.a2 import checkpoint
    from campaign.a2.data import FitQueue, Streams
    cfg = copy.deepcopy(parent_config)
    cfg['training']['optimizer_parameter_names'] = {'muon':['matrix'], 'adamw':['bias']}
    rows = [dict(image_id=f'fixture-{c}-{n}', class_index=c) for c in range(100) for n in range(8)]
    sampler = FitQueue(rows, 0); sampler.updates = 4000; sampler.images_seen = 1024000
    streams = Streams(0)
    def live():
        model = torch.nn.Module()
        model.register_parameter('matrix', torch.nn.Parameter(torch.ones(2, 2)))
        model.register_parameter('bias', torch.nn.Parameter(torch.ones(2)))
        ema = copy.deepcopy(model)
        opts = dict(muon=torch.optim.Muon([model.matrix], lr=1e-4),
                    adamw=torch.optim.AdamW([model.bias], lr=1e-4, fused=True))
        return model, ema, opts
    model, ema, opts = live()
    states = {k:copy.deepcopy(v.state_dict()) for k,v in opts.items()}
    states['muon']['state'] = {0:dict(momentum_buffer=torch.zeros(2, 2))}
    states['adamw']['state'] = {0:dict(step=torch.tensor(4000.), exp_avg=torch.zeros(2), exp_avg_sq=torch.zeros(2))}
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
               torch_cuda=torch.zeros(16, dtype=torch.uint8))
    history = [dict(update=n, images=512, recon_l1=1., recon_lpips=1., allocation_distortion=2.,
                    latent_image_variance=1., latent_rms=1.) for n in (1000, 2000, 3000, 4000)]
    state = dict(schema=checkpoint.SCHEMA, configuration_sha256='fixture-digest', resolved_config=cfg,
        world_size=4, boundary='AFTER_COMPLETE_OPTIMIZER_UPDATE',
        counters=dict(main_updates=4000, images_seen=1024000), sampler=sampler.state_dict(),
        model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(ema.state_dict()),
        optimizer_parameter_names=cfg['training']['optimizer_parameter_names'], optimizers=states,
        scheduler=dict(completed_updates=4000, warmup=500, after_warmup='constant'),
        process_rng_by_rank=[copy.deepcopy(rng) for _ in range(4)],
        streams=streams.state_dict(), evaluation_history=history,
        checkpoint_role='A2_NATIVE64_COMPLETE')
    def restore(value):
        model, ema, optimizers = live()
        with patch.object(checkpoint.torch, 'load', return_value=value), \
             patch.object(checkpoint.dist, 'get_world_size', return_value=4), \
             patch.object(checkpoint.dist, 'get_rank', return_value=0), \
             patch.object(checkpoint, 'restore_rng'), \
             patch.object(checkpoint, 'process_rng', return_value=value['process_rng_by_rank'][0]):
            checkpoint.load('synthetic-fixture-only.pt', model, ema, optimizers,
                cfg['training']['optimizer_parameter_names'], FitQueue(rows, 0), Streams(0),
                'fixture-digest', cfg=cfg, expected_update=4000)
    restore(state)
    mutations = [
        lambda v:v.update(configuration_sha256='wrong'),
        lambda v:v['scheduler'].update(completed_updates=3999),
        lambda v:v['counters'].update(images_seen=1),
        lambda v:v['process_rng_by_rank'].pop(),
        lambda v:v['streams']['generators'].pop('probe'),
        lambda v:v['optimizers']['adamw']['state'][0].pop('exp_avg'),
        lambda v:v['optimizers']['adamw']['state'][0].update(step=torch.tensor(3999.)),
        lambda v:v['optimizers']['muon']['state'][0].update(momentum_buffer=torch.zeros(3, 2)),
        lambda v:v['ema'].update(matrix=torch.ones(2, 2, dtype=torch.float64)),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(state); mutate(changed)
        rejects(lambda: restore(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and (ROOT/'STAGING_MANIFEST.json').is_file()
    assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Use a CPU-only allocation'
    cfg = json.loads(args.config.read_text())
    parent = contract.continuation(cfg)  # Real metadata only; never opens the model file.
    assert len(parent['exposure']) == 8533
    assert parent['attempts'][4267]['optimizer_update_attempted'] == 4267
    for field in ('training', 'data', 'model'):
        wrong = copy.deepcopy(cfg)
        wrong[field]['undeclared-change'] = True
        rejects(lambda: contract.continuation(wrong))
    wrong = copy.deepcopy(cfg); wrong['continuation']['parent_attempted_update'] = 4266
    rejects(lambda: contract.continuation(wrong))
    wrong = copy.deepcopy(cfg); wrong['continuation']['parent_checkpoint']['sha256'] = '0'*64
    rejects(lambda: contract.continuation(wrong))
    wrong = copy.deepcopy(cfg); wrong['continuation']['parent_exposure']['sha256'] = '0'*64
    rejects(lambda: contract.continuation(wrong))
    changed_stage = dict(cfg['continuation']['parent_stage'], source_revision='0'*40)
    rejects(lambda: contract.stage_root(changed_stage))
    terminal = dict(parent['terminal'])
    rejects(lambda: contract.terminal_state(terminal, cfg['continuation']['parent_config']['sha256']))
    terminal.update(state='COMPLETED', exit_code='0:0', all_recorded_rows_success=True)
    contract.terminal_state(terminal, cfg['continuation']['parent_config']['sha256'])
    rejects(lambda: contract.terminal_state(dict(terminal, job_state='PREEMPTED'), terminal['config_sha256']))
    rejects(lambda: contract.terminal_state(dict(terminal, all_recorded_rows_success=False), terminal['config_sha256']))
    rejects(lambda: contract.exposure_rows(ledger(1, 2)[:-1], 1, 2))
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    fixture = dict(parent)
    fixture['exposure'] = ledger(1, 4266)+[event(4267)]
    fixture['attempts'] = {n:event(n) for n in range(1, 4268)}
    fixture['monitor'] = [dict(update=n) for n in range(100, 4201, 100)]
    suffix = ledger(4001, 5000)
    arm = dict(exposure=write_lock(out/'synthetic-suffix.jsonl', suffix, rows=True),
               monitor=write_lock(out/'synthetic-monitor.jsonl',
                   [dict(update=n) for n in range(4100, 5001, 100)], rows=True))
    ordinary = dict(exposure=write_lock(out/'synthetic-full.jsonl', ledger(1, 5000), rows=True),
                    monitor=write_lock(out/'synthetic-full-monitor.jsonl',
                        [dict(update=n) for n in range(100, 5001, 100)], rows=True))
    with patch.object(contract, 'continuation', return_value=fixture):
        monitor, digest, accounting = contract.effective_histories(cfg, arm)
        _, expected_digest, _ = contract.effective_histories({}, ordinary)
        assert digest == expected_digest and len(monitor) == 50
        assert accounting['effective_updates'] == 5000 and accounting['physical_completed_updates'] == 5266
        assert accounting['attempted_global_batches'] == 5267 and accounting['partial_processed_images'] is None
        suffix[0]['flips'][0] = True
        bad = dict(arm, exposure=write_lock(out/'synthetic-wrong-flip.jsonl', suffix, rows=True))
        rejects(lambda: contract.effective_histories(cfg, bad))
        newer = dict(fixture['terminal'], job_id='new-fixture', state='COMPLETED')
        costs = contract.allocation_costs(cfg, newer)
        assert len(costs['allocations']) == 2
        assert costs['allocated_gpu_hours_by_class']['RTX3090'] == 2*newer['allocated_gpu_hours']
        rejects(lambda: contract.allocation_costs(cfg, fixture['terminal']))
    check_snapshot_contract(parent['config'])
    result = dict(schema='jat-a2-continuation-cpu-check-v1', engineering_only=True,
        scientific_updates=0, groups=['immutable_parent_metadata', 'scientific_control_rejection',
        'attempt4267_and_truncation', 'stage_identity', 'terminal_state_adapter',
        'effective_vs_physical_history', 'replay_flip_mismatch', 'unique_allocation_costs', 'complete_snapshot_and_optimizer_slot_rejections'],
        passed=True, trained_checkpoint_loaded=False, distributed_validation=False,
        limitation='Synthetic effective-history fixtures and real non-model metadata do not validate checkpoint bytes or numerical continuation.')
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
