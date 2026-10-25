"""Bounded A2 checkpoint4000 lineage; physical attempts remain immutable."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from campaign.a2.resolve import sha


def locked_json(lock):
    assert sha(lock['path']) == lock['sha256'], lock['path']
    return json.loads(Path(lock['path']).read_text())


def locked_rows(lock):
    assert sha(lock['path']) == lock['sha256'], lock['path']
    return [json.loads(line) for line in Path(lock['path']).read_text().splitlines()]


def stage_root(lock, cfg=None):
    root = Path(lock['path']).resolve()
    manifest = root/'STAGING_MANIFEST.json'
    assert sha(manifest) == lock['staging_manifest_sha256']
    saved = json.loads(manifest.read_text())
    assert Path(saved['stage']).resolve() == root
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == lock['source_revision']
    assert any(v['command'] == ['git', 'rev-parse', 'HEAD'] and v['value'].strip() == revision
               for v in saved['git_identity'])
    if cfg is not None:
        for name, digest in cfg['source']['code_sha256'].items():
            assert saved['files'][name]['kind'] == 'file' and not (root/name).is_symlink()
            assert (root/name).stat().st_mode & 0o111 == saved['files'][name]['execute_bits']
            assert saved['files'][name]['sha256'] == sha(root/name) == digest, name
    return root


def terminal_state(receipt, config_sha, completed=True):
    states = [receipt[k] for k in ('state', 'job_state') if k in receipt]
    assert states and len(set(states)) == 1, 'Missing or contradictory terminal state'
    assert receipt['config_sha256'] == config_sha
    if completed:
        assert states[0] == 'COMPLETED' and receipt['exit_code'] == '0:0'
        assert (receipt.get('all_recorded_steps_terminal_success') is True or
                (receipt.get('all_recorded_rows_terminal') is True and
                 receipt.get('all_recorded_rows_success') is True)), 'Child-step success is required'
    else:
        assert states[0] == 'PREEMPTED' and receipt['all_recorded_rows_terminal'] is True
    return states[0]


def exposure_rows(rows, first, last_complete, last_attempt=None):
    """Require alternating attempt/completion rows, with one possible final attempt."""
    last_attempt = last_complete if last_attempt is None else last_attempt
    assert last_attempt in (last_complete, last_complete+1)
    assert len(rows) == 2*(last_complete-first+1)+(last_attempt-last_complete)
    attempts = {}
    cursor = 0
    for number in range(first, last_attempt+1):
        event = rows[cursor]; cursor += 1
        assert set(event) == {'optimizer_update_attempted', 'image_ids', 'flips',
                              'exposure_chain', 'encoding_noise_sha256'}
        assert event['optimizer_update_attempted'] == number
        assert len(event['image_ids']) == len(event['flips']) == 256
        assert all(type(v) is bool for v in event['flips'])
        assert all(len(event[k]) == 64 for k in ('exposure_chain', 'encoding_noise_sha256'))
        attempts[number] = event
        if number <= last_complete:
            assert rows[cursor] == {'optimizer_update_completed': number}
            cursor += 1
    assert cursor == len(rows)
    return attempts


def continuation(cfg):
    """Validate exact parent bytes and the small operational-only config difference."""
    item = cfg['continuation']
    assert item['schema'] == 'jat-a2-native64-continuation-v1'
    assert item['parent_config']['sha256'] == '115e15e348d2c928aaba88a4c2654cb37cfd8d19d22580117bac9213b525c7b6'
    assert item['parent_checkpoint']['sha256'] == '9fb9c31deb782cb845ef34c20d539dda10651e720304f26339d24467be74ab7d'
    assert str(item['parent_job_id']) == '2834281'
    parent = locked_json(item['parent_config'])
    assert parent['method_id'] == cfg['method_id'] == 'NATIVE64_JOINT'
    assert 'continuation' not in parent
    assert set(cfg) == set(parent) | {'continuation', 'execution_enabled', 'disabled_reason'}
    candidate = copy.deepcopy(cfg)
    for key in ('continuation', 'execution_enabled', 'disabled_reason'):
        candidate.pop(key)
    # Only these existing metadata paths may change; every scientific field is exact.
    candidate['output_parent'] = parent['output_parent']
    candidate['source']['code_sha256'] = parent['source']['code_sha256']
    candidate['source']['patch_sha256'] = parent['source']['patch_sha256']
    assert candidate == parent, 'Continuation changed scientific controls or undeclared metadata'
    assert Path(cfg['output_parent']).resolve() != Path(parent['output_parent']).resolve()
    assert item['retained_update'] == 4000 and item['parent_completed_update'] == 4266
    assert item['parent_attempted_update'] == 4267
    terminal = locked_json(item['parent_terminal'])
    terminal_state(terminal, item['parent_config']['sha256'], completed=False)
    assert str(terminal['job_id']) == str(item['parent_job_id'])
    assert terminal['last_saved_main_updates'] == 4266
    assert terminal['stage'] == item['parent_stage']
    assert stage_root(item['parent_stage'], parent) == Path(item['parent_stage']['path']).resolve()
    assert item['parent_stage']['config_sha256'] == item['parent_config']['sha256']
    source = locked_json(item['parent_result'])
    assert source['config_sha256'] == item['parent_config']['sha256']
    assert source['main_updates'] == 4266 and source['images_seen'] == 4266*256
    assert source['checkpoint_sha256'] == item['parent_checkpoint']['sha256']
    rows = locked_rows(item['parent_exposure'])
    attempts = exposure_rows(rows, 1, 4266, 4267)
    monitor = locked_rows(item['parent_monitor'])
    assert [r['update'] for r in monitor] == list(range(100, 4201, 100))
    ids = locked_json(item['parent_evaluation_ids'])
    assert len(ids['dev512']) == 512 and len(ids['probe32']) == 32 and len(ids['dev_full']) == 3200
    history = source['reconstruction']['history']
    assert [v['update'] for v in history] == [1000, 2000, 3000, 4000]
    assert set(item['parent_dev']) == {'1000', '2000', '3000', '4000'}
    for number, lock in item['parent_dev'].items():
        summary = locked_json(lock)['summary']
        assert dict(update=int(number), **summary) == next(v for v in history if v['update'] == int(number))
    checkpoint = item['parent_checkpoint']
    assert Path(checkpoint['path']).parent == Path(item['parent_exposure']['path']).parent
    assert Path(checkpoint['path']).name == 'checkpoint-4000.pt'
    # Sidecars are metadata; the actual model-byte hash is required by checkpoint.load.
    assert Path(checkpoint['path']+'.sha256').read_text().strip() == checkpoint['sha256']
    return dict(config=parent, exposure=rows, attempts=attempts, monitor=monitor,
                history=history, evaluation_ids=ids, terminal=terminal)


def original_config(cfg):
    return continuation(cfg)['config'] if 'continuation' in cfg else cfg


def check_replay(attempt, lineage):
    number = attempt['optimizer_update_attempted']
    if number in lineage['attempts']:
        assert attempt == lineage['attempts'][number], f'Parent replay mismatch at{number}'


def effective_histories(cfg, arm):
    physical = locked_rows(arm['exposure'])
    monitor = locked_rows(arm['monitor'])
    if 'continuation' in cfg:
        parent = continuation(cfg)
        exposure_rows(physical, 4001, 5000)
        for event in physical:
            if 'optimizer_update_attempted' in event:
                check_replay(event, parent)
        assert [v['update'] for v in monitor] == list(range(4100, 5001, 100))
        effective = parent['exposure'][:8000]+physical
        monitor = [v for v in parent['monitor'] if v['update'] <= 4000]+monitor
        accounting = dict(parent_completed_updates=4266, parent_attempted_updates=4267,
            retained_updates=4000, abandoned_completed_updates=266,
            partial_attempt=4267, partial_processed_images=None,
            partial_processed_images_reason='The attempt record precedes processing; actual partial work is unknown.',
            new_completed_updates=1000, effective_updates=5000, effective_images=1280000,
            physical_completed_updates=5266, physical_completed_images=1348096,
            attempted_global_batches=5267, selected_slots=1348352,
            parent_exposure=cfg['continuation']['parent_exposure'], new_exposure=arm['exposure'])
    else:
        effective = physical
        accounting = dict(effective_updates=5000, effective_images=1280000,
                          physical_completed_updates=5000, physical_completed_images=1280000,
                          attempted_global_batches=5000, selected_slots=1280000,
                          new_exposure=arm['exposure'])
    exposure_rows(effective, 1, 5000)
    assert [v['update'] for v in monitor] == list(range(100, 5001, 100))
    encoded = ''.join(json.dumps(v, sort_keys=True, separators=(',', ':'))+'\n' for v in effective).encode()
    return monitor, hashlib.sha256(encoded).hexdigest(), accounting


def allocation_costs(cfg, terminal):
    receipts = [terminal]
    if 'continuation' in cfg:
        receipts.insert(0, continuation(cfg)['terminal'])
    ids = [str(r['job_id']) for r in receipts]
    assert len(set(ids)) == len(ids), 'Do not count a parent allocation twice'
    for row in receipts:
        assert all(math.isfinite(row[k]) and row[k] >= 0 for k in
                   ('allocated_gpu_hours', 'allocated_cpu_core_hours'))
        assert math.isclose(row['allocated_gpu_hours'],
                            row['allocated_elapsed_seconds']*row['gpu_count']/3600, rel_tol=1e-12)
    return dict(allocations=receipts, allocated_gpu_hours_by_class={kind:sum(
        r['allocated_gpu_hours'] for r in receipts if r['gpu_class'] == kind)
        for kind in sorted({r['gpu_class'] for r in receipts})},
        allocated_cpu_core_hours=sum(r['allocated_cpu_core_hours'] for r in receipts),
        basis='Unique parent Slurm allocations, including preempted work; never add nested steps or inherited process counters.')
