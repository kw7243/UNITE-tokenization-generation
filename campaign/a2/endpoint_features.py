"""Retain A2 endpoint features and paired samples; CPU scoring is separate."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from campaign.a2.checkpoint import SCHEMA
from campaign.a2.data import load_images, per_image_draws, read_interface, sha
from campaign.a2.evaluate import generation_eligible, generation_inputs
from campaign.a2.native import build_native, seed_all
from campaign.a2.train import verify_config
from utils.fid import extract_inception_features, get_inception_model


def locked_json(lock):
    assert sha(lock['path']) == lock['sha256'], lock['path']
    return json.loads(Path(lock['path']).read_text())


def paired_configs(configs, objective='joint'):
    assert objective in ('joint', 'reconstruction_only')
    suffix = 'JOINT' if objective == 'joint' else 'RECON'
    controls = []
    for count in (64, 16):
        cfg = copy.deepcopy(configs[count])
        assert cfg.pop('method_id') == f'NATIVE{count}_{suffix}'
        assert cfg['objective'] == objective
        for key in ('active_count', 'candidate_count', 'native_count'):
            assert cfg['model'].pop(key) == count
        controls.append(cfg)
    assert controls[0] == controls[1], 'Unmatched joint configurations'


def complete_inputs(cfg):
    configs, results = {}, {}
    for count in (64, 16):
        arm = cfg['inputs'][str(count)]
        train = locked_json(arm['training_config'])
        verify_config(train)
        result = locked_json(arm['result'])
        terminal = locked_json(arm['terminal_receipt'])
        assert terminal['job_state'] == 'COMPLETED' and terminal['exit_code'] == '0:0'
        assert terminal['config_sha256'] == arm['training_config']['sha256']
        assert result['config_sha256'] == arm['training_config']['sha256']
        assert result['main_updates'] == 5000 and result['images_seen'] == 1280000
        assert result['checkpoint_sha256'] == arm['checkpoint']['sha256']
        history = result['reconstruction']['history']
        assert [r['update'] for r in history] == [1000, 2000, 3000, 4000, 5000]
        assert all(r['images'] == 512 for r in history)
        assert np.isfinite([r[k] for r in history for k in
                            ('recon_l1', 'recon_lpips', 'allocation_distortion',
                             'latent_image_variance', 'latent_rms')]).all()
        assert result['metric_protocol_sha256'] == cfg['metric_protocol_sha256']
        configs[count], results[count] = train, result
    paired_configs(configs)
    assert generation_eligible({f'NATIVE{k}_JOINT': [r['allocation_distortion']
               for r in results[k]['reconstruction']['history']] for k in (64, 16)}), \
        'Both complete joint arms must pass the finite/improving generation condition'
    return configs, results


def load_ema(arm, train, result, device):
    # Only the owner's hash-locked, complete research snapshots are unpickled.
    assert sha(arm['checkpoint']['path']) == arm['checkpoint']['sha256']
    state = torch.load(arm['checkpoint']['path'], map_location='cpu', weights_only=False)
    assert state['schema'] == SCHEMA and state['world_size'] == 4
    assert state['boundary'] == 'AFTER_COMPLETE_OPTIMIZER_UPDATE'
    assert state['configuration_sha256'] == arm['training_config']['sha256']
    assert state['resolved_config'] == train
    assert state['checkpoint_role'] == f'A2_NATIVE{train["model"]["native_count"]}_COMPLETE'
    assert state['counters']['main_updates'] == state['sampler']['updates'] == 5000
    assert state['counters']['images_seen'] == state['sampler']['images_seen'] == 1280000
    assert state['scheduler']['completed_updates'] == 5000
    assert state['optimizer_parameter_names'] == train['training']['optimizer_parameter_names']
    assert set(state['optimizers']) == {'muon', 'adamw'}
    assert len(state['process_rng_by_rank']) == 4 and state['streams']
    assert state['evaluation_history'] == result['reconstruction']['history']
    model = build_native(train).eval().requires_grad_(False)
    model.load_state_dict(state['ema'], strict=True)
    del state
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model.to(device)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def retained_features(path, images, inception, start):
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        features, logits = extract_inception_features(images.float(), inception, batch_size=len(images))
    assert features.shape == (len(images), 2048) and features.dtype == torch.float32
    assert logits.shape == (len(images), 1008)
    assert torch.isfinite(features).all() and torch.isfinite(logits).all()
    path[start:start+len(images)] = features.numpy()
    return logits.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    cfg = json.loads(raw)
    assert cfg['schema'] == 'jat-a2-endpoint-features-config.v1' and cfg['execution_enabled'] is True
    assert cfg['node_id'] == 'A2' and cfg['seed'] == 20261023
    assert cfg['batch_size'] == 16 and cfg['sample_count'] == 5000
    assert cfg['precision'] == 'BF16_AUTOCAST_FP32_WEIGHTS_TF32_OFF'
    assert cfg['metric_protocol_sha256'] == sha(ROOT/'campaign/a2/metric-protocol.json')
    assert cfg['sampling'] == dict(solver='dopri5', interval=[0., 1.], atol=1e-6,
        rtol=1e-3, output_grid_points=50, shift=.5, cfg=2., cfg_interval=[.1, 1.], cfg_norm_order='norm_first')
    assert os.environ.get('SLURM_JOB_ID') and (ROOT/'STAGING_MANIFEST.json').is_file()
    assert Path.cwd().resolve() == ROOT
    for name, digest in cfg['source']['code_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert cfg['source']['code_sha256']['campaign/a2/endpoint_features.py'] == sha(__file__)
    stage_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', cfg['source']['implementation_revision'], stage_revision], check=True)
    configs, results = complete_inputs(cfg)
    assert cfg['plan_sha256'] == configs[64]['plan_sha256']
    reference = locked_json(cfg['fit_reference']['result'])
    assert reference['phase_status'] == 'COMPLETED' and reference['completed_images'] == 120289
    assert reference['metric_protocol_sha256'] == cfg['metric_protocol_sha256']
    assert reference['data_manifest_sha256'] == configs[64]['data']['hashes']
    for key in ('features', 'statistics', 'image_ids'):
        item = cfg['fit_reference'][key]
        assert sha(item['path']) == item['sha256']
        field = {'features': 'features_sha256', 'statistics': 'statistics_sha256',
                 'image_ids': 'image_ids_sha256'}[key]
        assert reference[field] == item['sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    assert cfg['hardware']['name_substring'] in props.name
    assert props.total_memory >= cfg['hardware']['minimum_vram_bytes']
    assert torch.cuda.is_bf16_supported()
    assert (torch.ones(1, device=device)+1).item() == 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    seed_all(cfg['seed'])
    os.environ['INCEPTION_WEIGHTS'] = configs[64]['evaluation']['metric_weights']['path']
    assert sha(os.environ['INCEPTION_WEIGHTS']) == configs[64]['evaluation']['metric_weights']['sha256']
    rows = read_interface(configs[64]['data'])['DEV']
    assert len(rows) == 3200
    for count in (64, 16):
        full = locked_json(cfg['inputs'][str(count)]['dev_full'])
        assert full['summary']['images'] == 3200
        assert [r['image_id'] for r in full['per_image']] == [r['image_id'] for r in rows]
        assert [r['class_index'] for r in full['per_image']] == [r['class_index'] for r in rows]
    digest = sha(args.config)
    out = Path(cfg['output_parent'])/f'A2__PAIRED_ENDPOINT_FEATURES__s{cfg["seed"]}__{digest[:12]}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'resolved.yaml').write_bytes(raw)
    runtime = dict(job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename, gpu=props.name,
        uuid=str(props.uuid), vram_bytes=props.total_memory, torch=torch.__version__, cuda=torch.version.cuda,
        capability=list(torch.cuda.get_device_capability()), stage=str(ROOT), source_revision=stage_revision,
        mounts={p: subprocess.check_output(['df', '-PT', p], text=True).splitlines()[-1]
                for p in (str(ROOT), str(out), configs[64]['data']['root'])})
    save_json(out/'runtime.json', runtime)
    record = dict(schema='jat-a2-endpoint-features-result.v1', node_id='A2', phase_status='RUNNING',
        node_status='IN_PROGRESS', config_sha256=digest, runtime=runtime, inputs=cfg['inputs'],
        fit_reference=cfg['fit_reference'], metric_protocol_sha256=cfg['metric_protocol_sha256'], arms={},
        scientific_outcome=None, scientific_outcome_reason='Coordinator owns scientific classification.',
        confirm_consumed=False, scoring=None, scoring_reason='FID/KID and final aggregation use a separate CPU workload.')
    save_json(out/'results.json', record)
    tick = time.perf_counter()
    try:
        inception = get_inception_model(device).requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(device)
        real = np.lib.format.open_memmap(out/'dev-real-features.npy', mode='w+', dtype=np.float32, shape=(3200, 2048))
        for start in range(0, len(rows), 16):
            images = load_images(rows[start:start+16], [False]*min(16, len(rows)-start), device)
            retained_features(real, images, inception, start)
        real.flush()
        save_json(out/'dev-image-ids.json', [r['image_id'] for r in rows])
        record['dev_real'] = dict(features_path=str(out/'dev-real-features.npy'), features_sha256=sha(out/'dev-real-features.npy'),
            image_ids_path=str(out/'dev-image-ids.json'), image_ids_sha256=sha(out/'dev-image-ids.json'), shape=[3200, 2048])
        labels, noise = generation_inputs(cfg['seed'])
        assert labels.shape == (5000,) and noise.shape == (5000, 64, 32)
        np.savez(out/'generation-inputs.npz', labels=labels.numpy(), noise=noise.numpy())
        record['common_generation_inputs'] = dict(path=str(out/'generation-inputs.npz'), sha256=sha(out/'generation-inputs.npz'),
            labels='class index repeated50', noise_shape=[5000, 64, 32], native_slice_only=True)
        for count in (64, 16):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            arm = out/f'NATIVE{count}_JOINT'
            arm.mkdir()
            model = load_ema(cfg['inputs'][str(count)], configs[count], results[count], device)
            rec = np.lib.format.open_memmap(arm/'reconstruction-features.npy', mode='w+', dtype=np.float32, shape=(3200, 2048))
            for start in range(0, len(rows), 16):
                part = rows[start:start+16]
                images = load_images(part, [False]*len(part), device)
                draws = {k: v.to(device) for k, v in per_image_draws(part, cfg['seed']).items()}
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model.encode(images, draws['encoding_noise'], draws['encoding_time'])
                    decoded = model.core.decode(z, representation_phase=False).float().clamp(0, 1)
                assert decoded.shape == images.shape and torch.isfinite(decoded).all()
                retained_features(rec, decoded, inception, start)
            rec.flush()
            gen = np.lib.format.open_memmap(arm/'generation-features.npy', mode='w+', dtype=np.float32, shape=(5000, 2048))
            logits_file = np.lib.format.open_memmap(arm/'generation-logits.npy', mode='w+', dtype=np.float32, shape=(5000, 1008))
            pixels_file = np.lib.format.open_memmap(arm/'generation-images.npy', mode='w+', dtype=np.uint8, shape=(5000, 3, 256, 256))
            batches, nfe = [], [0]
            hook = model.core.encoder.blocks[0].register_forward_pre_hook(lambda module, inputs: nfe.__setitem__(0, nfe[0]+1))
            try:
                for start in range(0, 5000, 16):
                    part_labels = labels[start:start+16].to(device)
                    part_noise = noise[start:start+16].to(device)
                    previous = nfe[0]
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        images = model.generate(part_labels, part_noise).float().clamp(0, 1)
                    assert images.shape == (len(part_labels), 3, 256, 256) and torch.isfinite(images).all()
                    pixels = (images*255).clamp(0, 255).to(torch.uint8)
                    torch.cuda.synchronize(device)
                    seconds = time.perf_counter()-begin
                    evaluations = nfe[0]-previous
                    assert evaluations > 0
                    batches.append(dict(start=start, samples=len(part_labels), seconds=seconds, nfe=evaluations))
                    pixels_file[start:start+len(part_labels)] = pixels.cpu().numpy()
                    logits_file[start:start+len(part_labels)] = retained_features(gen, images, inception, start)
                    if start == 0 or len(batches) % 20 == 0 or start+len(part_labels) == 5000:
                        for value in (gen, logits_file, pixels_file):
                            value.flush()
                        save_json(arm/'generation-batches.json', batches)
                        save_json(out/'progress.json', dict(arm=count, completed_generated_images=start+len(part_labels),
                            sampling_decode_seconds=sum(b['seconds'] for b in batches),
                            projected_sampling_decode_device_hours=sum(b['seconds'] for b in batches)/(start+len(part_labels))*5000/3600))
            finally:
                hook.remove()
            files = {name: dict(path=str(arm/name), sha256=sha(arm/name)) for name in
                     ('reconstruction-features.npy', 'generation-features.npy', 'generation-logits.npy',
                      'generation-images.npy', 'generation-batches.json')}
            seconds = sum(b['seconds'] for b in batches)
            record['arms'][str(count)] = dict(files=files, generated_samples=5000, native_count=count,
                expected_mean_tokens=float(count), mean_tokens=float(count), mean_tokens_sq=count**2,
                count_histogram_by_class={str(c): {str(count): 50} for c in range(100)},
                nfe=sum(b['samples']*b['nfe'] for b in batches)/5000,
                nfe_mean_per_batch=float(np.mean([b['nfe'] for b in batches])), nfe_total_calls=nfe[0],
                nfe_definition='Actual GE block0 calls; conditional/null CFG halves share each forward batch.',
                sampling_decode_seconds=seconds, gen_ms_per_image=1000*seconds/5000, gen_images_per_s=5000/seconds,
                timing_batch_size=16, timing_scope='Sampling, decode, clamp and uint8 conversion; excludes features, disk and scoring. No batch64 cost comparison.',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            save_json(out/'results.json', record)
            del model, rec, gen, logits_file, pixels_file
            torch.cuda.empty_cache()
        record.update(phase_status='COMPLETED', node_status='AWAITING_CPU_SCORING', elapsed_seconds=time.perf_counter()-tick)
    except Exception as exc:
        record.update(phase_status='FAILED', error=repr(exc), elapsed_seconds=time.perf_counter()-tick)
        raise
    finally:
        save_json(out/'results.json', record)


if __name__ == '__main__':
    main()
