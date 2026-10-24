"""Strict A2-complete -> B1 SCREEN-start interface; never accepts a role alone."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from campaign.b1.screen import NEW_NAMES
from campaign.a2.data import FitQueue, Streams

SCHEMA = 'jat-b1-screen-checkpoint-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def named_slots(state, names):
    ids = [i for group in state['param_groups'] for i in group['params']]
    require(len(ids) == len(names) == len(set(ids)) == len(set(names)), 'Optimizer names/slots are not bijective')
    require(set(state['state']) <= set(ids), 'Unknown optimizer slots')
    return dict(zip(names, ids))


def validate_input(snapshot, acceptance, actual_sha256, *, synthetic=False):
    require(isinstance(actual_sha256, str) and len(actual_sha256) == 64
            and all(c in '0123456789abcdef' for c in actual_sha256), 'Actual checkpoint SHA256 required')
    require(acceptance.get('synthetic_fixture', False) is synthetic, 'Synthetic/scientific acceptance mismatch')
    require(acceptance.get('checkpoint_sha256') == actual_sha256, 'Checkpoint bytes do not match acceptance')
    require(acceptance.get('a2_native64_accepted') is True and acceptance.get('stable_control') is True
            and acceptance.get('scientific_training_completed') is True, 'Coordinator-accepted trained stable native64 required')
    require(snapshot.get('schema') == 'jat-a2-checkpoint-v1', 'Not an A2 snapshot')
    require(snapshot.get('checkpoint_role') == 'A2_NATIVE64_COMPLETE', 'Wrong native checkpoint role')
    require(not snapshot.get('engineering_only', False) and snapshot.get('synthetic_fixture', False) is synthetic,
            'Engineering or synthetic input is ineligible')
    require(snapshot.get('boundary') == 'AFTER_COMPLETE_OPTIMIZER_UPDATE' and snapshot.get('world_size') == 4,
            'Wrong optimizer boundary or rank layout')
    cfg = snapshot['resolved_config']
    require(cfg['node_id'] == 'A2' and cfg['method_id'] == 'NATIVE64_JOINT'
            and cfg['objective'] == 'joint' and cfg['model']['native_count'] == cfg['model']['candidate_count'] == 64,
            'Not trained native64 joint A2')
    config_hash = hashlib.sha256((json.dumps(cfg, indent=2, sort_keys=True)+'\n').encode()).hexdigest()
    require(snapshot['configuration_sha256'] == acceptance['a2_configuration_sha256'] == config_hash,
            'Source configuration identity mismatch')
    require(cfg['source']['patch_sha256'] == acceptance['a2_patch_sha256'], 'A2 patch mismatch')
    require(acceptance.get('source_config_history') and all(set(row) >= {'source_revision','config_sha256','patch_sha256'}
            for row in acceptance['source_config_history']), 'Recorded source/config history required')
    require(acceptance['source_config_history'][-1]['config_sha256'] == config_hash
            and acceptance['source_config_history'][-1]['patch_sha256'] == cfg['source']['patch_sha256'], 'History endpoint mismatch')
    counters, sampler = snapshot['counters'], snapshot['sampler']
    require(counters['main_updates'] == snapshot['scheduler']['completed_updates'] == sampler['updates'] == 5000,
            'Exactly 5000 completed updates required; COMPLETE role text is insufficient')
    require(counters['images_seen'] == sampler['images_seen'] == 5000*256, 'Incomplete image exposure')
    require(len(sampler['exposure_chain']) == 64 and all(c in '0123456789abcdef' for c in sampler['exposure_chain']), 'Missing exposure chain')
    require(set(sampler) >= {'queues','cursor','rng','updates','images_seen','exposure_chain'}, 'Incomplete sampler')
    require(set(snapshot['streams']) == {'generators','times'} and set(snapshot['streams']['generators']) == set(Streams.names), 'Incomplete source streams')
    probe = Streams(0)
    probe.load_state_dict(snapshot['streams'])
    require(len(snapshot['process_rng_by_rank']) == 4 and all(set(s) == {'python','numpy','torch_cpu','torch_cuda'}
        and all(s[k] is not None for k in s) for s in snapshot['process_rng_by_rank']), 'Incomplete per-rank RNG')
    require(snapshot['evaluation_history'], 'Missing A2 evaluation history')
    require(snapshot['optimizer_parameter_names'] == cfg['training']['optimizer_parameter_names'], 'Unexpected source optimizer names')
    require(set(snapshot['optimizers']) == {'muon','adamw'}, 'Missing source optimizer')
    require(set(snapshot['model']) == set(snapshot['ema']), 'Incomplete EMA')
    for key in ('model','ema'):
        for tensor in snapshot[key].values():
            require(torch.is_tensor(tensor) and (not tensor.is_floating_point() or torch.isfinite(tensor).all().item()), 'Nonfinite or invalid model/EMA')
    unused = set(cfg['distributed'].get('expected_unused_trainable_parameters', []))
    for key in ('muon','adamw'):
        optimizer = snapshot['optimizers'][key]
        require(len(optimizer['param_groups']) == 1, 'Source optimizer must retain its one exact group')
        slots = named_slots(optimizer, snapshot['optimizer_parameter_names'][key])
        for name, slot in slots.items():
            require(name in snapshot['model'], 'Optimizer parameter absent from model')
            require(slot in optimizer['state'] or name in unused, 'Missing existing optimizer state: '+name)
            if slot not in optimizer['state']:
                continue
            state = optimizer['state'][slot]
            required = {'momentum_buffer'} if key == 'muon' else {'step','exp_avg','exp_avg_sq'}
            require(set(state) >= required, 'Incomplete optimizer moments: '+name)
            for field, tensor in state.items():
                if torch.is_tensor(tensor):
                    require(torch.isfinite(tensor).all().item(), 'Nonfinite optimizer state')
                    require(field == 'step' or tensor.shape == snapshot['model'][name].shape, 'Moment shape mismatch')
            if key == 'adamw':
                require(float(state['step']) == 5000, 'AdamW update count mismatch')
        require(all(group['lr'] == 1e-4 for group in optimizer['param_groups']), 'Executed continuation LR differs')
    return cfg


def convert_snapshot(snapshot, acceptance, actual_sha256, target_state, target_names, *, synthetic=False):
    """Pure conversion; callers must hash trusted bytes and bind acceptance first."""
    validate_input(snapshot, acceptance, actual_sha256, synthetic=synthetic)
    require(set(target_state)-set(snapshot['model']) == set(NEW_NAMES)
            and set(snapshot['model']) <= set(target_state), 'Unexpected model capacity change')
    for name, value in snapshot['model'].items():
        require(target_state[name].shape == value.shape and target_state[name].dtype == value.dtype, 'Existing tensor shape/dtype changed')
    for name in NEW_NAMES:
        require(target_state[name].shape == (3,768) and not torch.count_nonzero(target_state[name]).item(), 'Length tables must start at zero in [4,16,64] order')
    out = copy.deepcopy(snapshot)
    old_names = snapshot['optimizer_parameter_names']
    require(target_names['muon'] == old_names['muon'] and target_names['adamw'] == old_names['adamw']+list(NEW_NAMES), 'Source named optimizer order changed')
    for key in ('model','ema'):
        out[key].update({name: target_state[name].detach().cpu().clone() for name in NEW_NAMES})
    # Append to the existing AdamW group; keep every old group, ID, state, and hyperparameter.
    adam = out['optimizers']['adamw']
    ids = [i for group in adam['param_groups'] for i in group['params']]
    next_id = max(ids, default=-1)+1
    for offset, name in enumerate(NEW_NAMES):
        slot = next_id+offset
        adam['param_groups'][0]['params'].append(slot)
        adam['state'][slot] = dict(step=torch.tensor(0.), exp_avg=torch.zeros_like(out['model'][name]),
                                  exp_avg_sq=torch.zeros_like(out['model'][name]))
    out['schema'] = SCHEMA
    out['checkpoint_role'] = 'SYNTHETIC_B1_SCREEN_START' if synthetic else 'B1_SCREEN_START'
    out['synthetic_fixture'] = synthetic
    out['optimizer_parameter_names'] = copy.deepcopy(target_names)
    out['source_a2_configuration_sha256'] = out.pop('configuration_sha256')
    out['source_a2_resolved_config'] = out.pop('resolved_config')
    out['input_a2_sha256'] = actual_sha256
    out['acceptance'] = copy.deepcopy(acceptance)
    out['b1_completed_updates'] = 0
    out['phase_id'], out['allowed_counts'] = 'SCREEN', [16,64]
    out['checkpoint_roles'] = dict(SCREEN_PREFIX=None, COMMON_PREFIX=None)
    out['b1_conversion_implemented'] = True
    return out


def load_accepted(path, acceptance):
    from campaign.a2.data import sha
    actual = sha(path)
    require(actual == acceptance.get('checkpoint_sha256') and not acceptance.get('synthetic_fixture', False), 'Real accepted checkpoint bytes required')
    # Only the named trusted project checkpoint is deserialized, after byte acceptance.
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    validate_input(snapshot, acceptance, actual)
    return snapshot, actual
