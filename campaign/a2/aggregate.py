"""CPU-only A2 endpoint metrics, paired reconstruction intervals and curves."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch_fidelity.metric_kid import polynomial_mmd
from campaign.a2.data import eval_rows, read_interface, sha
from campaign.a2.endpoint_features import load_ema, locked_json, paired_configs, save_json
from campaign.a2.evaluate import generation_eligible, paired_interference_interval
from campaign.a2.train import verify_config
from utils.fid import compute_fid_from_features


METRICS = ('recon_l1', 'recon_lpips', 'allocation_distortion')


def aligned_records(records, rows):
    expected = {r['image_id']: r['class_index'] for r in rows}
    indexed = {r['image_id']: r for r in records}
    assert len(expected) == len(rows) == len(indexed) == len(records) == 3200
    assert set(indexed) == set(expected)
    ordered = [indexed[k] for k in sorted(expected)]
    assert all(r['class_index'] == expected[r['image_id']] for r in ordered)
    return ordered


def complete_arm(arm, method, rows):
    train, result = locked_json(arm['training_config']), locked_json(arm['result'])
    verify_config(train)
    terminal = locked_json(arm['terminal_receipt'])
    assert terminal['job_state'] == 'COMPLETED' and terminal['exit_code'] == '0:0'
    assert terminal['config_sha256'] == arm['training_config']['sha256'] == result['config_sha256']
    assert train['method_id'] == method and result['method_id'] == method
    assert result['main_updates'] == 5000 and result['images_seen'] == 1280000
    assert result['checkpoint_sha256'] == arm['checkpoint']['sha256']
    assert Path(arm['checkpoint']['path']+'.sha256').read_text().strip() == arm['checkpoint']['sha256']
    full, probe = locked_json(arm['dev_full']), locked_json(arm['gradient_probe'])
    assert full['summary']['images'] == 3200
    ordered = aligned_records(full['per_image'], rows)
    values = np.asarray([[r[k] for k in METRICS] for r in ordered], dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.allclose(values[:, 0]+values[:, 1], values[:, 2], rtol=1e-10, atol=1e-12)
    history = result['reconstruction']['history']
    assert [r['update'] for r in history] == [1000, 2000, 3000, 4000, 5000]
    assert all(r['images'] == 512 for r in history)
    assert np.isfinite([r[k] for r in history for k in METRICS]).all()
    assert sha(arm['monitor']['path']) == arm['monitor']['sha256']
    records = [json.loads(line) for line in Path(arm['monitor']['path']).read_text().splitlines()]
    assert sha(arm['exposure']['path']) == arm['exposure']['sha256']
    attempted, completed = [], []
    with Path(arm['exposure']['path']).open() as stream:
        for line in stream:
            event = json.loads(line)
            assert 'resume_discontinuity' not in event, 'Replayed exposure needs a declared matched revision'
            if 'optimizer_update_attempted' in event:
                attempted.append(event['optimizer_update_attempted'])
                assert len(event['image_ids']) == len(event['flips']) == 256
            else:
                completed.append(event['optimizer_update_completed'])
    assert attempted == completed == list(range(1, 5001))
    assert [r['update'] for r in records] == list(range(100, 5001, 100))
    assert np.isfinite([r[k] for r in records for k in
                       ('reconstruction_loss', 'gradient_norm_before_clip', 'latent_image_variance', 'latent_rms')]).all()
    if train['objective'] == 'joint':
        assert np.isfinite([r['flow_loss_sum'] for r in records]).all()
    assert len(probe['per_image']) == 96
    expected_probe = {r['image_id'] for r in eval_rows(rows, 32, train['evaluation']['seed'])}
    assert {r['image_id'] for r in probe['per_image']} == expected_probe
    assert {(r['image_id'], r['t']) for r in probe['per_image']} == {(i, t) for i in expected_probe for t in (.25, .5, .75)}
    assert np.isfinite([r[k] for r in probe['per_image'] for k in
        ('reconstruction_gradient_norm', 'generation_gradient_norm', 'grad_cos_last_block')]).all()
    assert np.allclose(values.mean(0), [full['summary'][k] for k in METRICS], rtol=1e-10, atol=1e-12)
    return dict(config=train, result=result, full=full, probe=probe, rows=ordered, values=values,
                terminal=terminal, monitor=records, input=arm)


def paired_difference(short, long, classes):
    assert short.shape == long.shape == (3200, 3)
    groups = [np.flatnonzero(classes == c) for c in range(100)]
    assert all(len(g) == 32 for g in groups)
    delta = short-long
    rng = np.random.RandomState(917)
    draws = np.empty((1000, 3), dtype=np.float64)
    for i in range(1000):
        # Every class has equal weight; image indices are paired across arms/metrics.
        draws[i] = np.mean([delta[rng.choice(g, len(g), replace=True)].mean(0) for g in groups], axis=0)
    interval = np.quantile(draws, [.025, .975], axis=0)
    return {k: dict(short_minus_long_mean=float(delta[:, j].mean()), ci95=interval[:, j].tolist(),
                   relative_loss=float(delta[:, j].mean()/max(long[:, j].mean(), 1e-8)))
            for j, k in enumerate(METRICS)}


def features(lock, count):
    assert sha(lock['path']) == lock['sha256']
    value = np.load(lock['path'], mmap_mode='r', allow_pickle=False)
    assert value.shape == (count, 2048) and value.dtype == np.float32
    assert np.isfinite(value).all()
    return value


def paired_kid(real, generated):
    assert set(generated) == {'64', '16'}
    rng = np.random.RandomState(917)
    samples = {k: [] for k in generated}
    for _ in range(100):
        real_indices = rng.choice(len(real), 1000, replace=False)
        gen_indices = rng.choice(5000, 1000, replace=False)
        for k, value in generated.items():
            samples[k].append(float(polynomial_mmd(real[real_indices], value[gen_indices],
                                 degree=3, gamma=1/2048, coef0=1)))
    assert np.isfinite([v for values in samples.values() for v in values]).all()
    return {k: dict(kid_5k=float(np.mean(v)), kid_5k_subset_std=float(np.std(v)),
                   kid_subset_values=v, kid_seed=917, kid_subsets=100, kid_subset_size=1000,
                   kid_definition='torch-fidelity unbiased polynomial MMD; paired subsets without replacement; degree3,gamma1/2048,coef1.')
            for k, v in samples.items()}


def score_features(lock, arms, out):
    endpoint = locked_json(lock)
    assert endpoint['phase_status'] == 'COMPLETED'
    assert endpoint['inputs'] == {str(k): arms[f'NATIVE{k}_JOINT']['input'] for k in (64, 16)}
    real = features({'path': endpoint['dev_real']['features_path'],
                     'sha256': endpoint['dev_real']['features_sha256']}, 3200)
    statistics = out/'dev-real-statistics.npz'
    real64 = np.asarray(real, dtype=np.float64)
    np.savez(statistics, mu=real64.mean(0), sigma=np.cov(real64, rowvar=False))
    del real64
    fit = features(endpoint['fit_reference']['features'], 120289)
    fit_stats = endpoint['fit_reference']['statistics']
    assert sha(fit_stats['path']) == fit_stats['sha256']
    generated = {k: features(v['files']['generation-features.npy'], 5000) for k, v in endpoint['arms'].items()}
    assert set(generated) == {'64', '16'}
    kid = paired_kid(fit, generated)
    scores = {}
    for k, values in endpoint['arms'].items():
        rec = features(values['files']['reconstruction-features.npy'], 3200)
        scores[k] = dict(rfid_eval=compute_fid_from_features(torch.from_numpy(np.array(rec)), str(statistics)),
                        gfid_5k=compute_fid_from_features(torch.from_numpy(np.array(generated[k])), fit_stats['path']),
                        **kid[k], sampling=values)
        assert np.isfinite([scores[k]['rfid_eval'], scores[k]['gfid_5k']]).all()
    return scores


def curves(arms, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('allocation_distortion', 'Fixed512 DEV shared image distortion', 'dev'),
              ('recon_l1', 'Fixed512 DEV L1', 'dev'), ('recon_lpips', 'Fixed512 DEV AlexNet LPIPS', 'dev'),
              ('reconstruction_loss', 'Source reconstruction loss (logged global256 batch)', 'train'),
              ('flow_loss_sum', 'Source M3 denoising loss sum (logged global256 batch)', 'train'),
              ('gradient_norm_before_clip', 'Global gradient norm before clipping', 'train')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for ax, (key, label, scope) in zip(axes.flat, panels):
        for method, arm in arms.items():
            rows = arm['result']['reconstruction']['history'] if scope == 'dev' else arm['monitor']
            available = [r for r in rows if r.get(key) is not None]
            if available:
                ax.plot([r['update'] for r in available], [r[key] for r in available], label=method)
        ax.set(xlabel='Completed optimizer updates', ylabel=label)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(out/'matched-curves.png', dpi=180)
    fig.savefig(out/'matched-curves.pdf')
    plt.close(fig)
    with (out/'matched-curves.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['method', 'scope', 'update', 'metric', 'value'])
        for method, arm in arms.items():
            for key, _, scope in panels:
                rows = arm['result']['reconstruction']['history'] if scope == 'dev' else arm['monitor']
                writer.writerows((method, scope, r['update'], key, r[key]) for r in rows if r.get(key) is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes(); cfg = json.loads(raw)
    assert cfg['schema'] == 'jat-a2-aggregate-config.v1' and cfg['execution_enabled'] is True
    assert os.environ.get('SLURM_JOB_ID') and (ROOT/'STAGING_MANIFEST.json').is_file()
    assert Path.cwd().resolve() == ROOT and not torch.cuda.is_available(), 'Use a CPU-only Slurm allocation'
    assert cfg['bootstrap_seed'] == 917 and cfg['bootstrap_resamples'] == 1000
    assert cfg['metric_protocol_sha256'] == sha(ROOT/'campaign/a2/metric-protocol.json')
    for name, digest in cfg['source']['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert cfg['source']['code_sha256']['campaign/a2/aggregate.py'] == sha(__file__)
    stage = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['source']['implementation_revision'], stage], check=True)
    torch.set_num_threads(cfg['cpu_threads'])
    import matplotlib
    plot_env = cfg['plot_environment']
    assert Path(sys.prefix).resolve() == Path(plot_env['path']).resolve()
    assert sha(plot_env['freeze_all']) == plot_env['freeze_all_sha256']
    assert sha(plot_env['base_pth']) == plot_env['base_pth_sha256']
    assert matplotlib.__version__ == plot_env['matplotlib_version']
    first = locked_json(cfg['arms']['NATIVE64_JOINT']['training_config'])
    rows = read_interface(first['data'])['DEV']
    assert set(cfg['arms']) in ({'NATIVE64_JOINT', 'NATIVE16_JOINT'},
                               {'NATIVE64_JOINT', 'NATIVE16_JOINT', 'NATIVE64_RECON', 'NATIVE16_RECON'})
    arms = {k: complete_arm(v, k, rows) for k, v in cfg['arms'].items()}
    paired_configs({k: arms[f'NATIVE{k}_JOINT']['config'] for k in (64, 16)})
    assert len({v['input']['exposure']['sha256'] for v in arms.values()}) == 1, 'Unmatched image/flip/full-noise exposure logs'
    assert all(v['probe']['parameter_names'] == arms['NATIVE64_JOINT']['probe']['parameter_names'] for v in arms.values())
    checked64 = load_ema(arms['NATIVE64_JOINT']['input'], arms['NATIVE64_JOINT']['config'],
                         arms['NATIVE64_JOINT']['result'], torch.device('cpu'))
    del checked64
    classes = np.array([r['class_index'] for r in arms['NATIVE64_JOINT']['rows']])
    paired = paired_difference(arms['NATIVE16_JOINT']['values'], arms['NATIVE64_JOINT']['values'], classes)
    eligible = generation_eligible({k: [r['allocation_distortion'] for r in v['result']['reconstruction']['history']]
                                    for k, v in arms.items() if k.endswith('JOINT')})
    digest = sha(args.config)
    out = Path(cfg['output_parent'])/f'A2__ENDPOINT_AGGREGATION__s917__{digest[:12]}'
    out.mkdir(parents=True, exist_ok=False); (out/'resolved.yaml').write_bytes(raw)
    if cfg['endpoint_features']:
        assert eligible, 'Generation metrics require the original paired finite/improving condition'
    scores = score_features(cfg['endpoint_features'], arms, out) if cfg['endpoint_features'] else None
    with (out/'paired-errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'class_index']+[f'{m}_{k}' for k in (64, 16) for m in METRICS]+['shared_distortion_16_minus_64'])
        for i, row in enumerate(arms['NATIVE64_JOINT']['rows']):
            a, b = arms['NATIVE64_JOINT']['values'][i], arms['NATIVE16_JOINT']['values'][i]
            writer.writerow([row['image_id'], row['class_index'], *a, *b, b[2]-a[2]])
    interference = None
    if 'NATIVE64_RECON' in arms:
        releases = cfg['conditional_reconstruction_releases']
        assert set(releases) == {'NATIVE64_RECON', 'NATIVE16_RECON'}
        for method, lock in releases.items():
            release = locked_json(lock)
            assert release['allow_a2_training'] is True
            assert release['conditional_reconstruction_arms_authorized'] is True
            assert release['configuration_sha256'] == arms[method]['input']['training_config']['sha256']
            assert release['plan_sha256'] == arms[method]['config']['plan_sha256']
        recon = {k: arms[f'NATIVE{k}_RECON']['config'] for k in (64, 16)}
        paired_configs(recon, objective='reconstruction_only')
        for count in (64, 16):
            a = json.loads(json.dumps(arms[f'NATIVE{count}_JOINT']['config']))
            b = json.loads(json.dumps(arms[f'NATIVE{count}_RECON']['config']))
            for item in (a, b):
                for key in ('method_id', 'objective', 'conditional'):
                    item.pop(key)
                item['evaluation']['generated_samples'] = 0
            assert a == b, 'Joint/RECON controls differ beyond the declared objective'
        interference = paired_interference_interval(*(arms[k]['values'][:, 2] for k in
            ('NATIVE16_JOINT', 'NATIVE64_JOINT', 'NATIVE16_RECON', 'NATIVE64_RECON')), classes)
    delta = arms['NATIVE16_JOINT']['values'][:, 2]-arms['NATIVE64_JOINT']['values'][:, 2]
    positive = np.sort(np.maximum(delta, 0))
    recon = {k: dict(v['full']['summary'], history=v['result']['reconstruction']['history'],
                     per_image_path=v['input']['dev_full']['path'], per_image_sha256=v['input']['dev_full']['sha256'])
             for k, v in arms.items()}
    generation = {}
    for method in arms:
        count = str(arms[method]['config']['model']['native_count'])
        if scores is not None and method.endswith('JOINT'):
            recon[method].update(rfid_eval=scores[count]['rfid_eval'], rfid_eval_reason=None)
            generation[method] = {k: v for k, v in scores[count].items() if k != 'rfid_eval'}
        else:
            generation[method] = dict(gfid_5k=None, kid_5k=None, reason='RECON is a mechanism probe without generation.' if method.endswith('RECON') else 'No completed endpoint feature workload is bound.')
    h64 = np.array([r['allocation_distortion'] for r in arms['NATIVE64_JOINT']['result']['reconstruction']['history']])
    result = dict(schema='jat-a2-aggregate-result.v1', node_id='A2', phase_status='RUNNING',
        node_status='IN_PROGRESS' if eligible and scores is None else 'AWAITING_COORDINATOR',
        outcome_class=None, outcome_class_reason='Coordinator classifies the paired scientific evidence and conditional next action.',
        config_sha256=digest, source_revision=stage, inputs=cfg['arms'], metric_protocol_sha256=cfg['metric_protocol_sha256'],
        training={k: dict(v['result']['training'], monitor_history=v['monitor']) for k, v in arms.items()},
        reconstruction=recon, generation=generation,
        generation_eligible=eligible, paired_reconstruction=paired,
        gradient_probes={k: v['probe'] for k, v in arms.items()},
        error_concentration=dict(fraction_images_16_higher=float(np.mean(delta > 0)),
            positive_gap_fraction_in_top10pct_images=float(positive[-320:].sum()/positive.sum()) if positive.sum() else None,
            reason=None if positive.sum() else 'No positive per-image short-count gap.'),
        joint_pressure=interference, joint_pressure_reason=None if interference is not None else 'Conditional paired RECON results are not bound; no mechanism screen is claimed.',
        valid64_gate=dict(complete5000_checkpoint=True, finite_logged_training=True,
            improving_fixed512DEV=bool(np.isfinite(h64).all() and h64[-1] < min(h64[0], h64[-2])),
            full_snapshot_schema_and_ema_strict_load_verified=True,
            coordinator_acceptance=None, reason='Worker evidence is not coordinator acceptance or downstream release.'),
        stable64_checkpoint=cfg['arms']['NATIVE64_JOINT']['checkpoint'],
        costs={k: v['terminal'] for k, v in arms.items()},
        additional_cost_receipts=[locked_json(lock) for lock in cfg['additional_cost_receipts']],
        analysis_job_terminal_cost=None, analysis_job_terminal_cost_reason='Reconcile this job after its Slurm terminal state.',
        figures=None, figures_reason='Plot generation pending.', confirm_consumed=False)
    save_json(out/'results.json', result)
    try:
        curves(arms, out)
        result.update(phase_status='COMPLETED', figures={name: dict(path=str(out/name), sha256=sha(out/name))
                      for name in ('matched-curves.csv', 'matched-curves.png', 'matched-curves.pdf', 'paired-errors.csv')},
                      figures_reason=None)
    except Exception as exc:
        result.update(phase_status='FAILED', error=repr(exc), figures_reason='Required plotting failed; computed metrics retained without a completed report.')
        raise
    finally:
        save_json(out/'results.json', result)
    sections=[('1 Objective', 'Classify the fixed16 pilot and establish the evidence for a stable local native64 control.'),
        ('2 Setup', 'The resolved.yaml, input hashes and immutable original run artifacts bind the actual matched5000-update experiment. This CPU analysis does not train or change a model.'),
        ('3 Baseline', 'Native counts, common seed0 initialization, same FIT/flips/full64-position draws, global256 exposures, source M3 sum and fixed protocol. RECON arms, if present, are mechanism probes rather than compute-matched generation baselines.'),
        ('4 Results', json.dumps(result, indent=2, allow_nan=False)),
        ('5 Interpretation', 'Use paired reconstruction intervals, readable matched curves and the32-image gradient probes together. A pilot gFID difference is a screen, not a statistical guarantee; a joint-pressure ratio is a mechanism hypothesis, not proof.'),
        ('6 Limitations', 'One training seed. No CONFIRM, guidance search or later-node run. Different training node pairs limit speed attribution. Full terminal owner-ledger costs and local review/test identities remain part of the worker handoff. Scientific acceptance remains with the coordinator.'),
        ('7 Next action', 'Coordinator reviews the joint endpoint, decides conditional RECON if still pending and accepts the valid64 handoff before dispatching a later node. This script releases no downstream run.')]
    (out/'report.md').write_text('\n\n'.join('## '+k+'\n\n'+v for k, v in sections)+'\n')


if __name__ == '__main__':
    main()
