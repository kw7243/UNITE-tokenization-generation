"""Focused synthetic CPU checks; no scientific data, GPU or trained64 proof."""
import argparse
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch_fidelity.metric_kid import kid_features_to_metric
from campaign.a2.aggregate import aligned_records, paired_difference, paired_kid
from campaign.a2.endpoint_features import paired_configs, save_json


def rejects(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError('Invalid paired input was accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and (ROOT/'STAGING_MANIFEST.json').is_file()
    assert not torch.cuda.is_available(), 'This fixture is CPU only'
    torch.set_num_threads(4)
    rows = [dict(image_id=f'image-{c:03d}-{i:02d}', class_index=c) for c in range(100) for i in range(32)]
    shuffled = [dict(r, reconstruction_error=i) for i, r in enumerate(rows)][::-1]
    aligned = aligned_records(shuffled, rows)
    assert [r['image_id'] for r in aligned] == [r['image_id'] for r in rows]
    rejects(lambda: aligned_records(shuffled[:-1]+[shuffled[0]], rows))
    wrong_class = copy.deepcopy(shuffled); wrong_class[0]['class_index'] = 999
    rejects(lambda: aligned_records(wrong_class, rows))
    wrong_id = copy.deepcopy(shuffled); wrong_id[0]['image_id'] = 'not-a-dev-image'
    rejects(lambda: aligned_records(wrong_id, rows))
    classes = np.array([r['class_index'] for r in rows])
    baseline = np.full((3200, 3), 2., dtype=np.float64)
    interval = paired_difference(baseline+.25, baseline, classes)
    for result in interval.values():
        assert np.allclose(result['ci95'], [.25, .25], rtol=0, atol=1e-12)
        assert abs(result['short_minus_long_mean']-.25) < 1e-12
        assert abs(result['relative_loss']-.125) < 1e-12
    rejects(lambda: paired_difference(baseline, baseline, np.zeros(3200, dtype=int)))
    configs = {k: dict(method_id=f'NATIVE{k}_JOINT', objective='joint', seed=0,
                      model=dict(active_count=k, native_count=k, candidate_count=k), lr=1e-4)
               for k in (64, 16)}
    paired_configs(configs)
    changed = copy.deepcopy(configs); changed[16]['lr'] = 2e-4
    rejects(lambda: paired_configs(changed))
    # Four-dimensional synthetic arrays test the resampling/kernel contract cheaply.
    # The production feature reader separately requires2048-dimensional Inception arrays.
    rng = np.random.RandomState(7)
    real = rng.normal(size=(1200, 4)).astype(np.float32)
    generated = rng.normal(size=(5000, 4)).astype(np.float32)
    kid = paired_kid(real, {'64': generated, '16': generated.copy()})
    assert kid['64']['kid_subset_values'] == kid['16']['kid_subset_values']
    reference = kid_features_to_metric(torch.from_numpy(real), torch.from_numpy(generated),
        kid_subsets=100, kid_subset_size=1000, kid_degree=3, kid_gamma=1/2048,
        kid_coef0=1., rng_seed=917, verbose=False)
    assert abs(kid['64']['kid_5k']-reference['kernel_inception_distance_mean']) < 1e-12
    assert abs(kid['64']['kid_5k_subset_std']-reference['kernel_inception_distance_std']) < 1e-12
    save_json(args.output, dict(phase='SYNTHETIC_CPU_CHECK', groups=dict(
        image_id_alignment='PASS', duplicate_missing_class_rejection='PASS',
        paired_within_class_bootstrap='PASS', matched_config_rejection='PASS',
        paired_kid_subsets_and_package_parity='PASS'), scientific_updates=0,
        gpu_validation=False, trained64_checkpoint_validated=False,
        limitation='Synthetic CPU checks do not prove CUDA/BF16, full-model generation, distributed execution or a scientific result.'))


if __name__ == '__main__':
    main()
