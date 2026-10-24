"""Resolve four exact initial A2 configurations without constructing a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reference = json.loads((ROOT/'campaign/configs/validation-400d3bce3a02.yaml').read_text())
    pilot = json.loads((ROOT/'campaign/locks/pilot-data.json').read_text())['consumer_interface']
    files = sorted((ROOT/'campaign/a2').glob('*.py'))+sorted((ROOT/'campaign/a2/tests').glob('*.py'))
    files += [ROOT/'campaign/adapters/prefix.py', ROOT/'campaign/a2/metric-protocol.json', ROOT/'campaign/a2/cpu.sbatch',
              ROOT/'reproduction/tools/stage_and_run.sh', ROOT/'reproduction/tools/verify_stage.py']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    patch = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    gen = reference['model']['gen_tok'].copy()
    gen.update(flow_steps_per_recon=3, flow_mini_batch=3, gradient_checkpointing_denoiser=False)
    index = {}
    for count in (64, 16):
        for joint in (True, False):
            method = f'NATIVE{count}_{"JOINT" if joint else "RECON"}'
            cfg = dict(node_id='A2', method_id=method, config_family='PILOT_M3', seed=0,
                       plan_version='1.1', plan_sha256='4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5',
                       objective='joint' if joint else 'reconstruction_only',
                       execution_authorized=False, execution_authorized_reason='PREPARATION_ONLY. Later coordinator receipt required.',
                       conditional=not joint, prerequisite='A1 UNITE reference acceptance and new native-path GPU checks',
                       source=dict(parent_commit='51a29b63aaba705a23a9de0e39ed8818b74a77ec',
                                   unite_commit='2389f2c65fedbd43ada851e69b2b810efd23af7d',
                                   branch='research/jat-a2-20261024', code_sha256=hashes, patch_sha256=patch,
                                   origin='https://github.com/kw7243/UNITE-tokenization-generation.git',
                                   upstream='https://github.com/ShivamDuggal4/UNITE-tokenization-generation'),
                       input=dict(initialization='random common64 construction, seed0, slice only latent_tokens table', checkpoint=None,
                                  checkpoint_reason='From random initialization; released M14 reference weights are not loaded.'),
                       data=dict(root=pilot['artifact_root'], hashes={k:v for k,v in pilot['hashes'].items()
                                 if k in ['FIT.tsv','DEV.tsv','ALLOC.tsv','class_to_idx.json']},
                                 rows={k:pilot['rows'][k] for k in ['FIT','DEV','ALLOC']}, resolution=256,
                                 preprocessing='pinned CenterCropTransform256,ToTensor[0,1],train flip.5;eval no flip',
                                 sampler='32 distinct classes x8 queued FIT images;four64-image distributed microbatches;two class groups/rank',
                                 confirm='sealed; consumer never opens CONFIRM or CONTENT',
                                 mapping='supplied100-class map; never full-root ImageFolder'),
                       model=dict(family='UNITE_BASE_BASE', num_classes=100, native_count=count, candidate_count=count,
                                  active_count=count, latent_width=32, selector=False, prefix_truncation=False,
                                  length_embeddings=False, decoder_interface='source grid interpolation,256image tokens plusCLS',
                                  gen_tok=gen, latent_layer_norm=dict(eps=1e-5, elementwise_affine=True, variance='population')),
                       training=dict(main_updates=5000, global_batch=256, gpu_count=4, microbatch_per_gpu=16, accumulation=4,
                                     denoising_draws=3, flow_chunk_size=3, main_lr=1e-4, warmup_updates=500,
                                     schedule='p*u/500 foru1..500;constant1e-4 after', implicit_lr_multiplier=1.,
                                     muon=dict(type='muon', lr=1e-4, momentum=.95, eps=1e-6,
                                               muon_adjust_lr_fn='match_rms_adamw', muon_weight_decay=0.),
                                     adamw=dict(type='adamw', lr=1e-4, betas=[.9,.999], eps=1e-6, weight_decay=0.),
                                     parameter_partition='source get_learnable_params;exactly once;frozen LPIPS excluded',
                                     gradient_clip_global_norm=3., ema_decay=.9978, mixed_precision='bf16',
                                     parameter_precision='fp32', reduction_precision='fp32', reporting_sums_precision='fp64',
                                     tf32=False, compile=False, checkpoint_every_updates=1000,
                                     source_vgg_lpips='frozen,eval after every model.train,unclamped[0,1],weight1',
                                     reconstruction_corruption=dict(mask_probability=.5,time_uniform=[.7,1.],noise='independent Gaussian'),
                                     generation_loss='sum3draws,mean token/channel squared velocity error,denominator max(1-t,.05)',
                                     generation_time=dict(logit_normal_mu=0.,logit_normal_sigma=1.,shift=.5),
                                     denoising_class_dropout=.1, encoding_class_dropout=1.,
                                     streams=['data_order','flip','encoding_noise','encoding_time','recon_noise','recon_time',
                                              'recon_mask','diffusion_noise','time_draws','class_dropout','assignment_shuffle','probe'],
                                     stream_layout='draw global B256 at64positions before rank/microbatch/native slice;consume all streams for every arm',
                                     monitor_every_updates=100, throughput_first_real_updates=200),
                       evaluation=dict(seed=20261023, bootstrap_seed=917, weights='ema', fixed_dev=512,
                                       every_updates=1000, endpoint_dev=3200, probe_images=32, probe_times=[.25,.5,.75],
                                       generated_samples=5000 if joint else 0, generation_gate='both joint arms finite/improving',
                                       cfg=2.,cfg_interval=[.1,1.],ode='dopri5',ode_interval=[0.,1.],
                                       ode_atol=1e-6,ode_rtol=1e-3,output_grid_points=50,output_grid_shift=.5,
                                       cfg_norm_order='norm_first', guidance_search=False,
                                       metric_protocol_sha256=sha(ROOT/'campaign/a2/metric-protocol.json'),
                                       metric_weights=reference['metrics']['inception'],
                                       pilot_fit_features=None,pilot_fit_features_reason='Separate future feature extraction required'),
                       environment=reference['environment'], hardware=dict(model='H100 80GB minimum',gpu_count=4,
                       alternate='Declare and resolve a revised matched protocol for all arms before execution'),
                       distributed=dict(backend='nccl',world_size=4,gradient_reduction='DDP average',
                                        accumulation_loss='local image sum/(16*4)',microbatch_order='acc64 +rank16,each rank two whole classes'),
                       output_parent=str(ROOT/'outputs/A2'),
                       compute_guidance=dict(advisory_gpu_hours=64,hard_cap=None,fixed_reserve=None,
                                             extensions='Section15.2;no estimate-based scientific negative'))
            raw = (json.dumps(cfg, indent=2, sort_keys=True)+'\n').encode()
            path = ROOT/'campaign/a2/configs'/f'{method}.yaml'
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            index[method] = dict(path=str(path.relative_to(ROOT)),sha256=digest,
                                 run_id=f'A2__{method}__s0__{digest[:12]}')
    (ROOT/'campaign/a2/configs/index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(patch_sha256=patch, configs=index),indent=2))

if __name__ == '__main__':
    main()
