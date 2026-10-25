"""Actual production checkpoint4000 restore/update comparison; engineering only."""
import argparse
import gc
import json
import os
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from campaign.a2 import checkpoint
from campaign.a2.continuation import continuation, check_replay
from campaign.a2.data import FitQueue, Streams, read_interface, sha
from campaign.a2.native import build_native, make_optimizers
from campaign.a2.train import verify_config, restore_and_wrap, optimizer_update
from campaign.a2.tests.check_ddp_gpu import fingerprint, cpu_copy, floating_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and Path.cwd().resolve() == ROOT
    assert (ROOT/'STAGING_MANIFEST.json').is_file()
    cfg = json.loads(args.config.read_text()); digest = sha(args.config)
    authorization = json.loads(args.authorization.read_text())
    assert authorization['allow_a2_recovery_checks'] is True
    assert authorization['configuration_sha256'] == digest
    assert authorization['patch_sha256'] == cfg['source']['patch_sha256']
    assert authorization['parent_checkpoint_sha256'] == cfg['continuation']['parent_checkpoint']['sha256']
    verify_config(cfg); lineage = continuation(cfg)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device); dist.init_process_group('nccl', device_id=device)
    rank = dist.get_rank(); assert dist.get_world_size() == 4
    props = torch.cuda.get_device_properties(device)
    assert cfg['hardware']['device_name_substring'] in props.name
    assert props.total_memory >= cfg['hardware']['minimum_vram_bytes']
    assert torch.cuda.is_bf16_supported()
    assert (torch.ones(1, device=device)+1).item() == 2
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = args.output
    if rank == 0: out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    runtime = dict(job_id=os.environ['SLURM_JOB_ID'], rank=rank, node=os.uname().nodename,
        gpu=props.name, uuid=str(props.uuid), memory_bytes=props.total_memory,
        torch=torch.__version__, cuda=torch.version.cuda, stage=str(ROOT), config_sha256=digest,
        precision=args.precision, engineering_only=True, scientific_updates=0)
    runtime['mounts'] = {p:subprocess.check_output(['df', '-PT', p], text=True).splitlines()[-1]
                         for p in (str(ROOT), str(out), cfg['data']['root'])}
    ranks = [None]*4; dist.all_gather_object(ranks, runtime)
    assert len({v['uuid'] for v in ranks}) == 4
    assert len({v['gpu'] for v in ranks}) == 1
    assert len({v['node'] for v in ranks}) == 2
    assert all(sum(v['node'] == node for v in ranks) == 2 for node in {v['node'] for v in ranks})
    (out/f'runtime-rank{rank}.json').write_text(json.dumps(runtime, indent=2)+'\n')
    splits = read_interface(cfg['data']); reference = None; records = []; baseline = None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for trial in range(2):
            sampler, streams = FitQueue(splits['FIT'], 0), Streams(0)
            model = build_native(cfg).to(device).train()
            ema = build_native(cfg).to(device).eval().requires_grad_(False)
            ema.load_state_dict(model.state_dict(), strict=True)
            optimizers, names = make_optimizers(model, cfg['training'])
            ddp, counters, history = restore_and_wrap(model, ema, optimizers, names, sampler,
                streams, cfg, digest, device, cfg['continuation']['parent_checkpoint']['path'], lineage)
            def state():
                return dict(model=model.state_dict(), ema=ema.state_dict(),
                    optimizers={k:o.state_dict() for k,o in optimizers.items()}, names=names,
                    sampler=sampler.state_dict(), streams=streams.state_dict(),
                    rng=checkpoint.process_rng(), counters=counters, history=history)
            boundary = fingerprint(state())
            if baseline is None: baseline = boundary
            else: assert boundary == baseline, 'Actual restore-before-DDP boundary changed'
            # Preview and restore only dedicated streams to check all full64 draws.
            before_streams = streams.state_dict(); preview = streams.draw(256)
            draw_digest = fingerprint(preview); streams.load_state_dict(before_streams); del preview
            old_ema = {n:p.detach().cpu().clone() for n,p in ema.named_parameters()
                       if dict(model.named_parameters())[n].requires_grad}
            attempts = []
            def attempted(event):
                check_replay(event, lineage); attempts.append(event)
            values = optimizer_update(ddp, model, ema, optimizers, sampler, streams,
                splits, device, rank, 4001, attempted, bf16=args.precision == 'bf16')
            assert len(attempts) == 1
            counters.update(main_updates=4001, images_seen=4001*256, attempted_images_this_process=256)
            clipped = torch.stack([p.grad.float().square().sum() for p in model.parameters()
                                   if p.grad is not None]).sum().sqrt().item()
            assert clipped <= 3.0001
            assert not model.core.lpips_loss.training
            assert all(not p.requires_grad for p in model.core.lpips_loss.parameters())
            live = dict(model.named_parameters())
            for name, before in old_ema.items():
                wanted = before.to(device).mul_(.9978).add_(live[name], alpha=1-.9978)
                assert torch.equal(wanted, dict(ema.named_parameters())[name])
            del old_ema, live, wanted
            record = dict(trial=trial, draw_sha256=draw_digest, attempt=attempts[0],
                          preclip_norm=values[-1].item(), postclip_norm=clipped,
                          loss_sums=values[0].detach().cpu().tolist())
            if trial == 0:
                checkpoint.save(out/'engineering-update4001.pt', model, ema, optimizers, names,
                    sampler, streams, digest, cfg, counters, history)
                saved = fingerprint(state())
                with torch.no_grad():
                    next(model.parameters()).add_(1)
                restored, loaded_history = checkpoint.load(out/'engineering-update4001.pt',
                    model, ema, optimizers, names, sampler, streams, digest, cfg=cfg,
                    expected_update=4001, expected_sha=(out/'engineering-update4001.pt.sha256').read_text().strip())
                counters.clear(); counters.update(restored); history[:] = loaded_history
                assert fingerprint(state()) == saved, 'New-identity complete-state round trip changed'
                reference = cpu_copy(state())
            else:
                for key in ('names', 'sampler', 'streams', 'rng', 'counters', 'history'):
                    assert fingerprint(reference[key]) == fingerprint(state()[key]), key
                errors = {key:floating_difference(reference[key], state()[key])
                          for key in ('model', 'ema', 'optimizers')}
                assert all(v['relative_l2'] <= 1e-5 for v in errors.values()), errors
                tolerance = 5e-3 if args.precision == 'bf16' else 1e-5
                for key in ('preclip_norm', 'postclip_norm', 'loss_sums'):
                    a, b = torch.as_tensor(records[0][key]), torch.as_tensor(record[key])
                    assert ((a-b).norm()/a.norm().clamp_min(1e-12)).item() <= tolerance
                assert draw_digest == records[0]['draw_sha256']
                record['continuation_relative_l2'] = errors
            hashes = [None]*4
            dist.all_gather_object(hashes, fingerprint(model.state_dict()))
            rank_error = 0.
            if len(set(hashes)) != 1:
                sums = torch.zeros(2, dtype=torch.float64, device=device)
                for parameter in model.parameters():
                    other = parameter.detach().clone(); dist.broadcast(other, src=0)
                    sums[0] += (parameter.double()-other.double()).square().sum()
                    sums[1] += other.double().square().sum()
                error = (sums[0]/sums[1].clamp_min(1e-30)).sqrt()
                dist.all_reduce(error, op=dist.ReduceOp.MAX); rank_error = error.item()
                assert rank_error <= 1e-5, rank_error
            record['all_rank_model_relative_l2'] = rank_error
            records.append(record)
            del ddp, model, ema, optimizers, values
            gc.collect(); torch.cuda.empty_cache(); dist.barrier()
        result = dict(passed=True, records=records, engineering_updates=2,
            engineering_images=512, scientific_updates=0, world_size=4, micro16=True, accumulation=4,
            global_batch=256, actual_production_restore_before_DDP=True,
            exact_two_restore_boundaries=True, new_config_checkpoint_roundtrip=True,
            parent_checkpoint_sha256=cfg['continuation']['parent_checkpoint']['sha256'],
            configuration_sha256=digest, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), precision=args.precision,
            limitation='Engineering comparison only; checkpoint4001 is not a stable64 endpoint or a scientific training result.')
        (out/f'results-rank{rank}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except Exception as exc:
        (out/f'failure-rank{rank}.json').write_text(json.dumps(dict(error=repr(exc),
            precision=args.precision, fp32_rerun_required_before_defect_diagnosis=args.precision == 'bf16',
            scientific_updates=0), indent=2)+'\n')
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
