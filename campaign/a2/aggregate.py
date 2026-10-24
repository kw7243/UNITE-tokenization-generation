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
from campaign.a2.data import read_interface, sha
from campaign.a2.endpoint_features import locked_json, paired_configs, save_json
from campaign.a2.evaluate import generation_eligible, paired_interference_interval
from campaign.a2.train import verify_config
from utils.fid import compute_fid_from_features


METRICS = ('recon_l1', 'recon_lpips', 'allocation_distortion')


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
    expected = {r['image_id']: r['class_index'] for r in rows}
    indexed = {r['image_id']: r for r in full['per_image']}
    assert len(indexed) == len(full['per_image']) == 3200 and set(indexed) == set(expected)
    ordered = [indexed[k] for k in sorted(expected)]
    assert all(r['class_index'] == expected[r['image_id']] for r in ordered)
    values = np.asarray([[r[k] for k in METRICS] for r in ordered], dtype=np.float64)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.allclose(values[:, 0]+values[:, 1], values[:, 2], rtol=1e-10, atol=1e-12)
    history = result['reconstruction']['history']
    assert [r['update'] for r in history] == [1000, 2000, 3000, 4000, 5000]
    assert all(r['images'] == 512 for r in history)
    assert np.isfinite([r[k] for r in history for k in METRICS]).all()
    monitor = Path(arm['result']['path']).parent/'monitor.jsonl'
    records = [json.loads(line) for line in monitor.read_text().splitlines()]
    assert [r['update'] for r in records] == list(range(100, 5001, 100))
    assert np.isfinite([r[k] for r in records for k in
                       ('reconstruction_loss', 'gradient_norm_before_clip', 'latent_image_variance', 'latent_rms')]).all()
    if train['objective'] == 'joint':
        assert np.isfinite([r['flow_loss_sum'] for r in records]).all()
    assert len(probe['per_image']) == 96
    assert len({r['image_id'] for r in probe['per_image']}) == 32
    assert {r['t'] for r in probe['per_image']} == {.25, .5, .75}
    assert np.isfinite([r[k] for r in probe['per_image'] for k in
        ('reconstruction_gradient_norm', 'generation_gradient_norm', 'grad_cos_last_block')]).all()
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
    assert matplotlib.__version__ == cfg['plot_environment']['matplotlib_version']
    first = locked_json(cfg['arms']['NATIVE64_JOINT']['training_config'])
    rows = read_interface(first['data'])['DEV']
    assert set(cfg['arms']) in ({'NATIVE64_JOINT', 'NATIVE16_JOINT'},
                               {'NATIVE64_JOINT', 'NATIVE16_JOINT', 'NATIVE64_RECON', 'NATIVE16_RECON'})
    arms = {k: complete_arm(v, k, rows) for k, v in cfg['arms'].items()}
    paired_configs({k: arms[f'NATIVE{k}_JOINT']['config'] for k in (64, 16)})
    classes = np.array([r['class_index'] for r in arms['NATIVE64_JOINT']['rows']])
    paired = paired_difference(arms['NATIVE16_JOINT']['values'], arms['NATIVE64_JOINT']['values'], classes)
    eligible = generation_eligible({k: [r['allocation_distortion'] for r in v['result']['reconstruction']['history']]
                                    for k, v in arms.items() if k.endswith('JOINT')})
    digest = sha(args.config)
    out = Path(cfg['output_parent'])/f'A2__ENDPOINT_AGGREGATION__s917__{digest[:12]}'
    out.mkdir(parents=True, exist_ok=False); (out/'resolved.yaml').write_bytes(raw)
    scores = score_features(cfg['endpoint_features'], arms, out) if cfg['endpoint_features'] else None
    with (out/'paired-errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'class_index']+[f'{m}_{k}' for k in (64, 16) for m in METRICS]+['shared_distortion_16_minus_64'])
        for i, row in enumerate(arms['NATIVE64_JOINT']['rows']):
            a, b = arms['NATIVE64_JOINT']['values'][i], arms['NATIVE16_JOINT']['values'][i]
            writer.writerow([row['image_id'], row['class_index'], *a, *b, b[2]-a[2]])
    interference = None
    if 'NATIVE64_RECON' in arms:
        assert cfg['conditional_reconstruction_release'], 'Bind coordinator release before using RECON results'
        locked_json(cfg['conditional_reconstruction_release'])
        recon = {k: arms[f'NATIVE{k}_RECON']['config'] for k in (64, 16)}
        paired_configs(recon, objective='reconstruction_only')
        interference = paired_interference_interval(*(arms[k]['values'][:, 2] for k in
            ('NATIVE16_JOINT', 'NATIVE64_JOINT', 'NATIVE16_RECON', 'NATIVE64_RECON')), classes)
    curves(arms, out)
    delta = arms['NATIVE16_JOINT']['values'][:, 2]-arms['NATIVE64_JOINT']['values'][:, 2]
    positive = np.sort(np.maximum(delta, 0))
    result = dict(schema='jat-a2-aggregate-result.v1', node_id='A2', phase_status='COMPLETED',
        node_status='IN_PROGRESS' if eligible and scores is None else 'AWAITING_COORDINATOR',
        outcome_class=None, outcome_class_reason='Coordinator classifies the paired scientific evidence and conditional next action.',
        config_sha256=digest, source_revision=stage, inputs=cfg['arms'], metric_protocol_sha256=cfg['metric_protocol_sha256'],
        training={k: v['result']['training'] for k, v in arms.items()},
        reconstruction={k: v['full']['summary'] for k, v in arms.items()},
        generation=scores, generation_reason=None if scores is not None else 'No completed endpoint feature workload is bound; eligibility is reported separately.',
        generation_eligible=eligible, paired_reconstruction=paired,
        gradient_probes={k: v['probe'] for k, v in arms.items()},
        error_concentration=dict(fraction_images_16_higher=float(np.mean(delta > 0)),
            positive_gap_fraction_in_top10pct_images=float(positive[-320:].sum()/positive.sum()) if positive.sum() else None,
            reason=None if positive.sum() else 'No positive per-image short-count gap.'),
        joint_pressure=interference, joint_pressure_reason=None if interference is not None else 'Conditional paired RECON results are not bound; no mechanism screen is claimed.',
        valid64_gate=dict(complete5000_checkpoint=True, finite_logged_training=True,
            improving_fixed512DEV=generation_eligible({'NATIVE64_JOINT': [r['allocation_distortion'] for r in arms['NATIVE64_JOINT']['result']['reconstruction']['history']],
                                                     'NATIVE16_JOINT': [r['allocation_distortion'] for r in arms['NATIVE64_JOINT']['result']['reconstruction']['history']]}),
            coordinator_acceptance=None, reason='Worker evidence is not coordinator acceptance or downstream release.'),
        stable64_checkpoint=cfg['arms']['NATIVE64_JOINT']['checkpoint'],
        costs={k: v['terminal'] for k, v in arms.items()}, confirm_consumed=False)
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
