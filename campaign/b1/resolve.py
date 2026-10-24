"""Write three full disabled SCREEN drafts; no model import or experiment launch."""
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    base=json.loads((ROOT/'campaign/a2/configs/r6-source-ddp-unused/NATIVE64_JOINT.yaml').read_text())
    metric=json.loads((ROOT/'campaign/a2/metric-protocol.json').read_text())
    metric['scope']='B1 SCREEN only; all qualities unmeasured during CPU preparation'
    metric['generation'].update(fixed_count='PREFIX16 at16,PREFIX64 at64,NESTED_RANDOM separately at16 and64',
        labels='identical paired class labels/full64 Gaussian prefixes across arms',
        gate='coordinator selects common5000 or10000 continuation endpoint; finite endpoint required; no guidance search')
    metric.pop('a2_interference_gate')
    metric['screen_gate']=dict(stable_progress_required=True,max_distortion_relative_loss=.10,max_gfid_relative_loss=.10,
        positive_nested_gfid_improvement=.05,kid_same_direction=True,positive_tail_gain_required_for_hierarchy=True,
        failed_prefix16='stability result only; own-initial D_img improvement>=.05,finite generation without collapse,ordinary64gates',
        endpoint='initial5000;matched10000 if Section8.1 requires;further matched Section15.2 extensions preserve original results')
    metric['features']['pilot_reference_features_sha256_reason']='Bind existing pilot FIT feature artifact by verified manifest later; no duplicate extraction in B1 preparation'
    metric_path=ROOT/'campaign/b1/metric-protocol.json'
    metric_path.write_text(json.dumps(metric,indent=2)+'\n')
    files=sorted((ROOT/'campaign/b1').rglob('*.py'))+[metric_path,ROOT/'campaign/b1/cpu.sbatch']
    inherited=base['source']['code_sha256']
    files+= [ROOT/name for name in inherited]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
    patch=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    names=copy.deepcopy(base['training']['optimizer_parameter_names'])
    names['adamw']+=['encoder_length.weight','decoder_length_after_up.weight','decoder_length_after_embed.weight']
    index={}
    for method,quotas in [('PREFIX16',[0,64,0]),('PREFIX64',[0,0,64]),('NESTED_RANDOM',[0,32,32])]:
        cfg=copy.deepcopy(base)
        cfg.update(node_id='B1',phase_id='SCREEN',method_id=method,conditional=False,protocol_revision='b1-screen-prepare-v1',
            prerequisite='Coordinator accepted trained A2 native64 checkpoint at5000 plus affected Section5.11 GPU/DDP/resume checks',
            execution_authorized=False,execution_authorized_reason='PREPARATION_ONLY. Trained input and GPU evidence absent.',
            output_parent=str(ROOT/'outputs/B1/SCREEN'))
        cfg['input']=dict(checkpoint=None,checkpoint_sha256=None,acceptance=None,
            checkpoint_reason='No coordinator-accepted trained native64 checkpoint exists; engineering COMPLETE snapshots are ineligible',
            checkpoint_sha256_reason='No trained bytes yet; do not fabricate a hash',
            acceptance_reason='Coordinator must bind actual trained bytes,stable5000updates,complete state,and source/config history')
        cfg['source']=dict(origin=base['source']['origin'],upstream=base['source']['upstream'],
            unite_commit=base['source']['unite_commit'],parent_commit='7a492f54a1974931d5d603b14b150263a8891f44',
            parent_tested_commit='c3eafe8ffa8b809a92b18af429003ff199bffe00',
            parent_patch_sha256='f18eb9c75f00da255b0940e6b2bacda1474e5d9ff36fb328036c3873618745ea',
            branch='research/jat-b1-screen-prepare-20261024',code_sha256=hashes,patch_sha256=patch,
            patch_definition='SHA256 canonical sorted compact JSON of exact source-path/SHA256 map')
        cfg['model'].pop('native_count',None)
        cfg['model'].update(candidate_count=64,active_count=None,length_embeddings=True,length_count_order=[4,16,64],
            length_table_shapes=[[3,768],[3,768],[3,768]],length_initialization='zero for all arms',
            length_interfaces=['denoiser c=t+y+length; class-only context; same length in both CFG halves',
                'after up_sample_decoder','after decoder.decoder_embed'],
            prefix_truncation=True,decoder_interface='unchanged physical sqrt(k) square-grid bilinear interpolation to16x16 then256tokens+CLS')
        cfg['allocation']=dict(budget_id='NESTED40' if method=='NESTED_RANDOM' else f'FIXED{16 if method=="PREFIX16" else 64}',
            allowed_counts=[16,64],count_order=[4,16,64],distributed_microbatch=64,microbatch_quotas=quotas,
            optimizer_batch_quotas=[q*4 for q in quotas],target_mean=sum(k*q for k,q in zip([4,16,64],quotas))/64,
            target_mean_sq=sum(k*k*q for k,q in zip([4,16,64],quotas))/64,
            rule='image-independent multiset permutation; never image-specific selection',stream='b1_screen_assignment')
        cfg['training'].update(main_updates=5000,warmup_updates=0,schedule='constant_continuation1e-4;A2warmup not restarted',
            optimizer_parameter_names=names,parameter_partition='reuse source Muon/AdamW partition;append exactly3length tables to AdamW',
            new_optimizer_state='zero step,exp_avg,exp_avg_sq;exact named old state retained',
            prefix_loss_reduction='one active prefix per image;image mean outside M3 sum',
            stream_layout='inherited A2 streams restored at5000;B256 full64 draws before count grouping;new independent count stream',
            streams=base['training']['streams']+['b1_screen_assignment'])
        cfg['evaluation'].update(metric_protocol_sha256=sha(metric_path),generation_gate='selected common SCREEN endpoint',
            fixed_counts=[16] if method=='PREFIX16' else [64] if method=='PREFIX64' else [16,64],
            pilot_fit_features_manifest=None,pilot_fit_features_manifest_sha256=None,
            pilot_fit_features_reason='Reuse existing pilot FIT feature extraction by verified immutable manifest later; no extraction here',
            common_endpoint_policy=dict(initial_updates=5000,matched_extension_updates=10000,scheduled_dev_images=512,
                schedule_every=1000,full_dev_images=3200,endpoint_selection='Section8.1 reconstruction criteria;coordinator across all3arms',
                further_extensions='Section15.2;record hashes,endpoints,controls,reason,cost;preserve originals'),
            confirm='sealed')
        cfg['hardware']=dict(model='NVIDIA H100 80GB',device_name_substring='H100',minimum_vram_bytes=80000000000,
            gpu_count=4,nodes=1,ranks_per_node=4,selection_reason='Future default only;inspect actual inventory and fit before binding matched launch')
        cfg['distributed'].update(nodes=1,ranks_per_node=4,slurm_launcher_tasks=1,slurm_cpus_per_launcher=16,
            launcher='torchrun four ranks;declare any alternate matched layout before execution')
        cfg['compute_guidance'].update(advisory_gpu_hours=96,hard_cap=None)
        raw=(json.dumps(cfg,indent=2,sort_keys=True)+'\n').encode()
        path=ROOT/'campaign/b1/configs'/f'{method}.yaml';path.write_bytes(raw)
        digest=hashlib.sha256(raw).hexdigest()
        index[method]=dict(path=str(path.relative_to(ROOT)),sha256=digest,run_id=f'B1__{method}__s0__{digest[:12]}')
    (ROOT/'campaign/b1/configs/index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(patch_sha256=patch,configs=index),indent=2))

if __name__=='__main__':main()
