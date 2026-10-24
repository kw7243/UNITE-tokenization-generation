"""Future four-GPU A2 execution. This preparation does not authorize execution."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from campaign.a2 import checkpoint
from campaign.a2.data import FitQueue, Streams, eval_rows, load_images, local_indices, read_interface, select_draws, sha
from campaign.a2.evaluate import reconstruction, gradient_probe
from campaign.a2.native import build_native, executed_lr, make_optimizers, backward_mean, clip_gradients


def append(path, record):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_config(cfg):
    assert cfg['node_id'] == 'A2' and cfg['config_family'] == 'PILOT_M3' and cfg['seed'] == 0
    assert cfg['plan_sha256'] == '4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5'
    assert cfg['training']['global_batch'] == 256
    assert cfg['training']['gpu_count'] == 4 and cfg['training']['microbatch_per_gpu'] == 16
    assert cfg['training']['accumulation'] == 4 and cfg['training']['main_updates'] == 5000
    assert cfg['model']['num_classes'] == 100
    assert cfg['model']['native_count'] in (16, 64)
    assert cfg['model']['candidate_count'] == cfg['model']['native_count']
    for name, expected in cfg['source']['code_sha256'].items():
        assert sha(ROOT/name) == expected, name
    imports = json.loads((ROOT/'campaign/locks/source-imports.json').read_text())
    for name, expected in imports['files']['unite'].items():
        assert sha(ROOT/'third_party/unite'/name) == expected, name
    assert sha(cfg['environment']['packages']) == cfg['environment']['packages_sha256']
    for package, version in cfg['environment']['package_versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert sha(ROOT/'campaign/a2/metric-protocol.json') == cfg['evaluation']['metric_protocol_sha256']
    assets = cfg['evaluation']['asset_lock']
    for value in assets['metric_weight_files'].values():
        assert sha(value['path']) == value['sha256']
    for path, expected in assets['model_cache_files'].items():
        assert sha(path) == expected


def save_report(out, cfg, record):
    sections = [
        ('1 Objective', 'A2 fixed-count diagnosis. A stable64-token joint control is required. The conditional reconstruction ratio threshold is1.10 with paired95% lower bound>1.0.'),
        ('2 Setup', f"{cfg['method_id']};seed0;resolved.yaml and optimizer-parameter-names.json fix the recipe. ConfigSHA256 {record['config_sha256']}. Runtime, source, and authorization receipts are in runtime.jsonl."),
        ('3 Baseline', 'Paired random initialization,images,flips,full64-position noise,updates,and losses. Native encoders process only their stated count. Reconstruction-only arms are conditional mechanism probes, not compute-matched generation baselines.'),
        ('4 Results', json.dumps(record,indent=2,allow_nan=False)),
        ('5 Interpretation', 'This arm cannot establish the paired A2 conclusion alone. Coordinator classification and the exact sharedDEV comparisons are required.'),
        ('6 Limitations', 'Endpoint generation requires both joint arms to be finite and improving. Feature extraction and quality scoring are separate workloads. Terminal allocation time must reconcile all phase times,failures,and overhead. An unrun or incomplete comparison is not a scientific negative.'),
        ('7 Next action', 'Await coordinator comparison. PASS to B1 requires a valid stable NATIVE64_JOINT complete snapshot; otherwise STOP and diagnose the common setup or record the actual reference/resource constraint.')]
    (out/'report.md').write_text('\n\n'.join('## '+title+'\n\n'+body for title,body in sections)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--ack-replay-through', type=int)
    args = parser.parse_args()
    if not args.execute or args.authorization is None:
        parser.error('Training requires later coordinator authorization and --execute. Preparation only.')
    raw = args.config.read_bytes()
    cfg = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    receipt = json.loads(args.authorization.read_text())
    assert receipt['allow_a2_training'] is True and receipt['a1_unite_pass'] is True
    assert receipt['gpu_native_checks_passed'] is True
    assert receipt['configuration_sha256'] == digest
    assert receipt['patch_sha256'] == cfg['source']['patch_sha256']
    assert receipt['plan_sha256'] == cfg['plan_sha256']
    if cfg['objective'] == 'reconstruction_only':
        assert receipt['conditional_reconstruction_arms_authorized'] is True
    assert (Path.cwd()/'STAGING_MANIFEST.json').is_file()
    assert os.environ.get('SLURM_JOB_ID')
    verify_config(cfg)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    assert torch.cuda.is_bf16_supported()
    props = torch.cuda.get_device_properties(device)
    assert cfg['hardware']['device_name_substring'] in props.name
    assert props.total_memory >= cfg['hardware']['minimum_vram_bytes']
    # Different hardware needs a revised matched configuration, never a silent fallback.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = Path(cfg['output_parent'])/f"A2__{cfg['method_id']}__s0__{digest[:12]}"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'resolved.yaml').exists():
            assert args.resume and (out/'resolved.yaml').read_bytes() == raw
        else:
            (out/'resolved.yaml').write_bytes(raw)
        runtime = dict(node=os.uname().nodename, job_id=os.environ['SLURM_JOB_ID'],
                       gpu=props.name, uuid=str(props.uuid), vram_bytes=props.total_memory,
                       torch=torch.__version__, cuda=torch.version.cuda, world_size=world,
                       stage=str(Path.cwd()), source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       authorization_sha256=sha(args.authorization), configuration_sha256=digest)
        append(out/'runtime.jsonl', runtime)
    dist.barrier()
    rank_runtime = dict(rank=rank, node=os.uname().nodename, gpu=props.name,
                        uuid=str(props.uuid), vram_bytes=props.total_memory,
                        local_rank=int(os.environ['LOCAL_RANK']))
    rank_runtimes = [None]*world
    dist.all_gather_object(rank_runtimes, rank_runtime)
    if rank == 0:
        append(out/'rank-runtime.jsonl', dict(job_id=os.environ['SLURM_JOB_ID'], ranks=rank_runtimes))
    process_training_start = time.perf_counter()
    splits = read_interface(cfg['data'])
    sampler, streams = FitQueue(splits['FIT'], 0), Streams(0)
    model = build_native(cfg).to(device).train()
    # Build independently; copying a Python closure installed on encoder.forward would bind the live model.
    ema = build_native(cfg).to(device).eval().requires_grad_(False)
    ema.load_state_dict(model.state_dict(), strict=True)
    optimizers, names = make_optimizers(model, cfg['training'])
    if rank == 0:
        (out/'optimizer-parameter-names.json').write_text(json.dumps(names, indent=2)+'\n')
    counters = dict(main_updates=0, images_seen=0, training_seconds=0., evaluation_seconds=0.,
                    first200_update_seconds=0., attempted_images_this_process=0)
    history = []
    prior_training_seconds = 0.
    prior_evaluation_seconds = 0.
    if args.resume:
        counters, history = checkpoint.load(args.resume, model, ema, optimizers, names, sampler, streams, digest)
        assert sampler.updates < 5000
        prior_training_seconds = counters['training_seconds']
        prior_evaluation_seconds = counters['evaluation_seconds']
    if rank == 0:
        ledger = out/'exposure.jsonl'
        previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        highest = max([r.get('optimizer_update_attempted', 0) for r in previous]+[0])
        if highest > sampler.updates:
            assert args.ack_replay_through == highest, 'Resume would repeat exposure; acknowledge its exact recorded extent.'
            append(ledger, dict(resume_discontinuity=True, checkpoint_update=sampler.updates,
                                replay_through=highest, reason='Interrupted work since the last complete snapshot; replay is explicit.'))
        elif not args.resume:
            assert not previous, 'Existing exposure cannot be overwritten.'
    dist.barrier()
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    dev512 = eval_rows(splits['DEV'], 512, cfg['evaluation']['seed'])
    probe32 = eval_rows(splits['DEV'], 32, cfg['evaluation']['seed'])
    if rank == 0:
        (out/'evaluation-image-ids.json').write_text(json.dumps(dict(
            dev512=[r['image_id'] for r in dev512], probe32=[r['image_id'] for r in probe32],
            dev_full=[r['image_id'] for r in splits['DEV']]), indent=2)+'\n')
    torch.cuda.reset_peak_memory_stats(device)
    first200_start = time.perf_counter()
    try:
        for update in range(sampler.updates+1, 5001):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            indices, draws = sampler.next(), streams.draw(256)
            rows = [splits['FIT'][i] for i in indices]
            if rank == 0:
                append(out/'exposure.jsonl', dict(optimizer_update_attempted=update,
                       image_ids=[r['image_id'] for r in rows], flips=draws['flip'].tolist(),
                       exposure_chain=sampler.exposure_chain,
                       encoding_noise_sha256=hashlib.sha256(draws['encoding_noise'].numpy().tobytes()).hexdigest()))
            counters['attempted_images_this_process'] += 256
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = executed_lr(update)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            zsum = torch.zeros((model.count, 32), dtype=torch.float64, device=device)
            zsquare = torch.zeros_like(zsum)
            flow_bins = torch.zeros((4,2), dtype=torch.float64, device=device)
            for acc in range(4):
                local = local_indices(rank, acc)
                images = load_images([rows[i] for i in local], draws['flip'][local], device)
                labels = torch.tensor([rows[i]['class_index'] for i in local], device=device)
                selected = select_draws(draws, local, device)
                context = ddp.no_sync() if acc < 3 else nullcontext()
                with context:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        values = ddp(images, labels, selected)
                    finite = torch.isfinite(values['per_image']).all().int()
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError('Nonfinite loss before update')
                    backward_mean(values['per_image']).backward()
                totals += torch.stack([values['reconstruction'].detach().double().sum(), values['generation_sum'].detach().double().sum()])
                if model.joint:
                    times = selected['diffusion_time'].flatten()
                    bins = torch.bucketize(times, torch.tensor([.25,.5,.75], device=device))
                    flow = values['generation_draws'].detach().double().flatten()
                    for noise_bin in range(4):
                        mask = bins == noise_bin
                        flow_bins[noise_bin,0] += flow[mask].sum()
                        flow_bins[noise_bin,1] += mask.sum()
                z = values['latents'].double()
                zsum += z.sum(0)
                zsquare += z.square().sum(0)
            norm = clip_gradients(trainable)
            for optimizer in optimizers.values():
                optimizer.step()
            with torch.no_grad():
                # All trainable GE/decoder/normalization parameters, once per optimizer update.
                live = dict(model.named_parameters())
                for name, param in ema.named_parameters():
                    if live[name].requires_grad:
                        param.mul_(.9978).add_(live[name], alpha=1-.9978)
                parameter_norms = torch.stack([p.detach().float().norm() for p in trainable])
                opt_values = [v for optimizer in optimizers.values() for state in optimizer.state.values()
                              for v in state.values() if torch.is_tensor(v) and v.is_floating_point()]
                state_norms = torch.stack([v.float().norm().to(device) for v in opt_values])
                finite = (torch.isfinite(parameter_norms).all() & torch.isfinite(state_norms).all()).int()
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite parameter/optimizer state')
            counters['main_updates'], counters['images_seen'] = update, update*256
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter()-start, device=device, dtype=torch.float64)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            counters['training_seconds'] = prior_training_seconds + time.perf_counter()-process_training_start-(counters['evaluation_seconds']-prior_evaluation_seconds)
            if update <= 200:
                counters['first200_update_seconds'] += elapsed.item()
            if update == 200 and not args.resume:
                first200 = torch.tensor([time.perf_counter()-first200_start,
                        torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                        dtype=torch.float64, device=device)
                dist.all_reduce(first200, op=dist.ReduceOp.MAX)
                counters['first200_update_seconds'] = first200[0].item()
                counters['first200_peak_allocated_bytes'] = int(first200[1].item())
                counters['first200_peak_reserved_bytes'] = int(first200[2].item())
                counters['first200_projected_5000_training_seconds'] = first200[0].item()*25
                if rank == 0:
                    (out/'first200-forecast.json').write_text(json.dumps(dict(
                        completed_updates=200, images_seen=51200,
                        seconds=first200[0].item(), images_per_s=51200/first200[0].item(),
                        peak_allocated_bytes=int(first200[1].item()), peak_reserved_bytes=int(first200[2].item()),
                        projected_5000_training_device_hours=first200[0].item()*25*4/3600,
                        gpu_model=props.name, gpu_count=world,
                        scope='Measured first200 real updates, including synchronization and training logging; endpoint evaluation is additional.',
                        uncertainty='Initial warmup, evolving training, data stalls, and endpoint evaluation can change this forecast.'),
                        indent=2)+'\n')
            if rank == 0:
                append(out/'exposure.jsonl', dict(optimizer_update_completed=update))
            if update % 100 == 0:
                dist.all_reduce(totals)
                dist.all_reduce(zsum)
                dist.all_reduce(zsquare)
                dist.all_reduce(flow_bins)
                if rank == 0:
                    append(out/'monitor.jsonl', dict(update=update, images_seen=update*256,
                           reconstruction_loss=(totals[0]/256).item(), flow_loss_sum=(totals[1]/256).item() if model.joint else None,
                           flow_loss_mean_draw=(totals[1]/768).item() if model.joint else None,
                           flow_loss_reason=None if model.joint else 'Reconstruction-only objective; denoising is not evaluated.',
                           native_count=model.count,
                           flow_loss_mean_draw_by_noise_level=[dict(bin=b,mean=(flow_bins[b,0]/flow_bins[b,1]).item() if flow_bins[b,1]>0 else None,
                                 draws=int(flow_bins[b,1].item()),reason=None if flow_bins[b,1]>0 else 'No denoising draws in this bin.') for b in range(4)],
                           gradient_norm_before_clip=norm.item(),
                           latent_image_variance=(zsquare/256-(zsum/256).square()).mean().item(),
                           latent_rms=(zsquare/256).mean().sqrt().item(), lr=executed_lr(update),
                           norm_gamma_rms=model.core.encoder_layer_norm.weight.float().square().mean().sqrt().item(),
                           norm_beta_rms=model.core.encoder_layer_norm.bias.float().square().mean().sqrt().item()))
            if update % 1000 == 0:
                dist.barrier()
                ev = time.perf_counter()
                if rank == 0:
                    model.eval()
                    summary = reconstruction(ema, dev512, device,
                                             cfg['evaluation']['seed'], out/f'dev-{update}.json')
                    history.append(dict(update=update, **summary))
                    if update == 5000:
                        reconstruction(ema, splits['DEV'], device, cfg['evaluation']['seed'], out/'dev-full-5000.json')
                        gradient_probe(model, probe32, device, cfg['evaluation']['seed'], out/'gradient-probe.json')
                    model.train()
                dist.barrier()
                ev_time = torch.tensor(time.perf_counter()-ev, device=device, dtype=torch.float64)
                dist.all_reduce(ev_time, op=dist.ReduceOp.MAX)
                counters['evaluation_seconds'] += ev_time.item()
                shared = [history if rank == 0 else None]
                dist.broadcast_object_list(shared, src=0)
                history = shared[0]
            if update % 1000 == 0:
                checkpoint.save(out/f'checkpoint-{update}.pt', model, ema, optimizers, names, sampler, streams,
                                digest, cfg, counters, history)
            if rank == 0:
                result = dict(node_id='A2', method_id=cfg['method_id'], seed=0, config_sha256=digest,
                              checkpoint_sha256=(out/f'checkpoint-{update//1000*1000}.pt.sha256').read_text().strip() if update >= 1000 else None,
                              checkpoint_sha256_reason='No optimizer-boundary snapshot yet.' if update < 1000 else None,
                              data_manifest_sha256=cfg['data']['hashes'], metric_protocol_sha256=cfg['evaluation']['metric_protocol_sha256'],
                              node_status='AWAITING_COORDINATOR' if update == 5000 else 'IN_PROGRESS',
                              outcome_class=None, outcome_class_reason='Coordinator must classify paired scientific trajectories.',
                              main_updates=update, images_seen=update*256, gpu_hours=None,
                              gpu_hours_reason='Terminal allocated-time accounting is required, including failures.',
                              training_gpu_hours=counters['training_seconds']*4/3600,
                              evaluation_gpu_hours=counters['evaluation_seconds']*4/3600,
                              cost_basis='Process wall time minus evaluation, times four GPUs; terminal allocation setup/tail overhead must also be reconciled.',
                              training=dict(**counters, first200_images_per_s=51200/counters['first200_update_seconds'] if update >= 200 else None),
                              reconstruction=dict(history=history), generation=dict(gfid_5k=None, kid_5k=None,
                              reason='Both joint arms must pass the finite/improving gate before a separate endpoint workload.'))
                (out/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                save_report(out, cfg, result)
    except Exception as exc:
        if rank == 0:
            append(out/'failures.jsonl', dict(main_updates=counters['main_updates'], attempted_update=sampler.updates,
                   error=repr(exc), outcome_class='NUMERICAL_FAILURE' if isinstance(exc, FloatingPointError) else None,
                   reason='Technical failure is not a scientific negative; retain exposure and terminal receipts.'))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
