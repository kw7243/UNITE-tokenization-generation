"""Extract an immutable pilot FIT reference through the pinned source metric path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from campaign.a2.data import load_images, read_interface, sha
from campaign.a2.train import verify_config
from utils.fid import get_inception_model, extract_inception_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    cfg = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert cfg['schema'] == 'jat-a2-fit-features-config.v1'
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file() and os.environ.get('SLURM_JOB_ID')
    assert sha(ROOT/'campaign/a2/fit_features.py') == cfg['source']['script_sha256']
    refpath = ROOT/cfg['source']['reference_config']
    assert sha(refpath) == cfg['source']['reference_config_sha256']
    reference = json.loads(refpath.read_text())
    verify_config(reference)
    assert cfg['plan_sha256'] == reference['plan_sha256']
    assert cfg['metric_protocol_sha256'] == reference['evaluation']['metric_protocol_sha256']
    assert sha(cfg['coordinator_acceptance']['path']) == cfg['coordinator_acceptance']['sha256']
    acceptance = json.loads(Path(cfg['coordinator_acceptance']['path']).read_text())
    assert acceptance['phase_status'] == 'PASS' and 'NATIVE64_JOINT' in acceptance['release']
    assert cfg['batch_size'] == 64 and cfg['seed'] == 20261023
    assert cfg['precision'] == 'FP32_TF32_OFF' and cfg['split'] == 'FIT'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    assert cfg['hardware']['name_substring'] in props.name
    assert props.total_memory >= cfg['hardware']['minimum_vram_bytes']
    capability = torch.cuda.get_device_capability(device)
    assert f'sm_{capability[0]}{capability[1]}' in torch.cuda.get_arch_list()
    assert (torch.ones(1, device=device)+1).item() == 2
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['INCEPTION_WEIGHTS'] = reference['evaluation']['metric_weights']['path']
    assert sha(os.environ['INCEPTION_WEIGHTS']) == reference['evaluation']['metric_weights']['sha256']
    rows = read_interface(reference['data'])['FIT']
    assert len(rows) == 120289 and len({row['image_id'] for row in rows}) == len(rows)
    out = Path(cfg['output_parent'])/f'PILOT_FIT_INCEPTION__s{cfg["seed"]}__{digest[:12]}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'resolved.json').write_bytes(raw)
    runtime = dict(job_id=os.environ['SLURM_JOB_ID'], node=platform.node(), gpu=props.name,
                   uuid=str(props.uuid), vram_bytes=props.total_memory, capability=list(capability),
                   torch=torch.__version__, cuda=torch.version.cuda, compiled_arches=torch.cuda.get_arch_list(),
                   stage=str(Path.cwd()), source_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                   cpu_allowed=Path('/proc/self/status').read_text().split('Cpus_allowed_list:')[1].splitlines()[0].strip(),
                   mounts={p: subprocess.check_output(['df','-PT',p],text=True).splitlines()[-1]
                           for p in [str(Path.cwd()),str(out),reference['data']['root'],'/data/vision/torralba/datasets/imagenet_pytorch_new/train']})
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (out/'image-ids.json').write_text(json.dumps([row['image_id'] for row in rows])+'\n')
    features = np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(rows),2048))
    record = dict(schema='jat-a2-fit-features-result.v1',node_id='A2',phase='PILOT_FIT_REFERENCE',
                  phase_status='RUNNING',node_status='IN_PROGRESS',config_sha256=digest,
                  plan_sha256=cfg['plan_sha256'],metric_protocol_sha256=cfg['metric_protocol_sha256'],
                  data_manifest_sha256=reference['data']['hashes'],runtime=runtime,
                  completed_images=0,training_updates=0,features_sha256=None,
                  features_sha256_reason='Complete extraction and terminal validation pending.',
                  scientific_outcome=None,scientific_outcome_reason='Reference extraction alone is not a learned-model result.',
                  confirm_consumed=False,pixel_conversion='Pinned source: (x*255).clamp(0,255).to(uint8); truncate',
                  logits='Computed and checked for finiteness; reference FID/KID do not need retained real logits.')
    def save():
        (out/'results.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    save()
    start = time.perf_counter()
    try:
        inception = get_inception_model(device).requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(device)
        for offset in range(0,len(rows),cfg['batch_size']):
            batch = rows[offset:offset+cfg['batch_size']]
            with torch.no_grad():
                images = load_images(batch,[False]*len(batch),device)
                f,logits = extract_inception_features(images,inception,batch_size=len(batch))
            assert f.shape == (len(batch),2048) and f.dtype == torch.float32
            assert logits.shape[0] == len(batch) and torch.isfinite(f).all() and torch.isfinite(logits).all()
            features[offset:offset+len(batch)] = f.numpy()
            record['completed_images'] = offset+len(batch)
            if offset == 0 or (offset//cfg['batch_size']+1)%100 == 0 or record['completed_images'] == len(rows):
                torch.cuda.synchronize(device)
                seconds = time.perf_counter()-start
                record.update(extraction_seconds=seconds,images_per_second=record['completed_images']/seconds,
                              projected_extraction_device_hours=seconds/record['completed_images']*len(rows)/3600,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                features.flush(); save()
                print('FIT_FEATURE_PROGRESS',record['completed_images'],seconds,flush=True)
        assert record['completed_images'] == len(rows)
        features.flush()
        values = np.asarray(features,dtype=np.float64)
        np.savez(out/'statistics.npz',mu=np.mean(values,axis=0),sigma=np.cov(values,rowvar=False))
        del values
        record.update(phase_status='COMPLETED',features_sha256=sha(out/'features.npy'),
                      features_sha256_reason=None,statistics_sha256=sha(out/'statistics.npz'),
                      image_ids_sha256=sha(out/'image-ids.json'),feature_shape=list(features.shape),
                      feature_dtype=str(features.dtype),elapsed_seconds=time.perf_counter()-start)
        save()
        sections=[('1 Objective','Lock the supplied 100-class FIT Inception reference for paired A2 metrics.'),
                  ('2 Setup',json.dumps(dict(config_sha256=digest,source=cfg['source'],runtime=runtime),indent=2)),
                  ('3 Baseline','Pinned source Inception and image loader; manifest row order; center crop 256; no flips; FP32 with TF32 off. No learned checkpoint or CONFIRM input.'),
                  ('4 Results',json.dumps(record,indent=2)),
                  ('5 Interpretation','These features define the pilot reference. Extraction is independent of learned weights and does not release generation.'),
                  ('6 Limitations','This is pilot FIT, not the published full 1000-class reference. Terminal allocated-time accounting remains required.'),
                  ('7 Next action','Verify terminal 0:0 and feature identities. Use this reference only after both joint arms pass the finite/improving generation condition.')]
        (out/'report.md').write_text('\n\n'.join('## '+title+'\n\n'+body for title,body in sections)+'\n')
    except Exception as exc:
        record.update(phase_status='FAILED',error=repr(exc),elapsed_seconds=time.perf_counter()-start)
        save()
        raise


if __name__ == '__main__':
    main()
