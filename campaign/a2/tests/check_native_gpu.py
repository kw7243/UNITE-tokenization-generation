"""Focused full native checks. Engineering state is never a scientific initialization."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))
import torch
from campaign.a2.data import Streams, load_images, read_interface, select_draws, sha
from campaign.a2.native import build_native, make_optimizers, backward_mean, clip_gradients
from campaign.a2.train import verify_config


def relative(a, b):
    return ((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).item()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def emit(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert os.environ.get('SLURM_JOB_ID') and (Path.cwd()/'STAGING_MANIFEST.json').is_file()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    assert torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(4)
    cfgs = {k: json.loads((ROOT/f'campaign/a2/configs/NATIVE{k}_JOINT.yaml').read_text()) for k in (64,16)}
    for c in cfgs.values():
        verify_config(c)
    props = torch.cuda.get_device_properties(device)
    runtime = dict(job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
        gpu=props.name, vram_bytes=props.total_memory, capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__, cuda=torch.version.cuda, compiled_arches=torch.cuda.get_arch_list(),
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        stage=str(Path.cwd()), config_sha256={k:sha(ROOT/f'campaign/a2/configs/NATIVE{k}_JOINT.yaml') for k in (64,16)},
        scope='Native correctness and one engineering microbatch; no scientific A2 training',
        source_torch_home=os.environ['TORCH_HOME'])
    emit(out/'runtime.json', runtime)
    splits = read_interface(cfgs[64]['data'])
    rows = splits['DEV'][:16]
    # Compare full initialized parameter bytes before any perturbation or optimizer update.
    a, b = build_native(cfgs[64]), build_native(cfgs[16])
    left = dict(a.named_parameters())
    hashes = {}
    for name, value in b.named_parameters():
        expected = left[name][:,:16] if name == 'core.latent_tokens' else left[name]
        assert torch.equal(value, expected), name
        hashes[name] = dict(native64=tensor_hash(left[name]), native16=tensor_hash(value),
            native64_shape=list(left[name].shape), native16_shape=list(value.shape))
    emit(out/'common-initialization.json', dict(passed=True, parameters=hashes,
        exception='Only core.latent_tokens is sliced to16; all common parameter bytes equal.'))
    del left, a, b
    gc.collect()
    all_results = {}
    for count in (64,16):
        model = build_native(cfgs[count]).to(device).eval()
        assert model.count == model.core.num_latent_tokens == count
        assert model.core.latent_tokens.shape == (1,count,768)
        # The source zeroes final output and block gates. Nonzero engineering-only
        # values prevent trivial all-zero equality/gradient checks.
        with torch.no_grad():
            torch.manual_seed(118)
            model.core.encoder.final_layer.linear.weight.normal_(0,.01)
            for block in model.core.encoder.blocks:
                block.adaLN_modulation[-1].weight.normal_(0,.01)
        images = load_images(rows[:3], [False]*3, device)
        labels = torch.tensor([r['class_index'] for r in rows[:3]],device=device)
        draws = select_draws(Streams(19).draw(3), list(range(3)), device)
        physical = []
        def final_input(module, inputs):
            assert inputs[0].shape[1] == count, inputs[0].shape
            physical.append(list(inputs[0].shape))
        handle = model.core.encoder.final_layer.linear.register_forward_pre_hook(final_input)
        results = {}
        for precision in ('fp32','bf16'):
            context = lambda: torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with torch.no_grad(), context():
                z = model.encode(images, draws['encoding_noise'],draws['encoding_time'])
                rec, decoded = model.reconstruction(z,images,draws)
                gen = model.generation(z,labels,draws,return_draws=True)
                singles = []
                for i in range(3):
                    d = {k:v[i:i+1] for k,v in draws.items()}
                    zi = model.encode(images[i:i+1],d['encoding_noise'],d['encoding_time'])
                    ri, xi = model.reconstruction(zi,images[i:i+1],d)
                    gi = model.generation(zi,labels[i:i+1],d,return_draws=True)
                    singles.append((zi,ri,xi,gi))
                errs = dict(encoding=relative(z,torch.cat([v[0] for v in singles])),
                    reconstruction_loss=relative(rec,torch.cat([v[1] for v in singles])),
                    decode=relative(decoded,torch.cat([v[2] for v in singles])),
                    generation_draw_loss=relative(gen,torch.cat([v[3] for v in singles])))
                tol = 5e-3 if precision=='bf16' else 1e-5
                results[precision] = dict(relative_l2=errs,tolerance=tol,all_passed=all(x<=tol for x in errs.values()),
                    shape=list(z.shape), source_sum_error=relative(gen.sum(1),model.generation(z,labels,draws)))
                assert z.shape == (3,count,32)
                assert results[precision]['source_sum_error'] == 0
            if precision=='fp32':
                assert results[precision]['all_passed'], results
            elif not results[precision]['all_passed']:
                # The immediately preceding FP32 run is the required fallback.
                assert results['fp32']['all_passed']
                results[precision]['fp32_fallback_passed'] = True
            del z,rec,decoded,gen,singles
        # Actual gradient boundaries; the generation graph cannot reach encoded z.
        z = model.encode(images[:1],draws['encoding_noise'][:1],draws['encoding_time'][:1])
        d = {k:v[:1] for k,v in draws.items()}
        params = list(model.core.encoder.blocks[-1].parameters())
        gen = model.generation(z,labels[:1],d).sum()
        assert torch.autograd.grad(gen,z,allow_unused=True,retain_graph=True)[0] is None
        gg = torch.autograd.grad(gen,params,allow_unused=True)
        rec,_ = model.reconstruction(z,images[:1],d)
        gr = torch.autograd.grad(rec.sum(),[z]+params,allow_unused=True)
        gn = sum(g.float().square().sum() for g in gg if g is not None).sqrt().item()
        rn = sum(g.float().square().sum() for g in gr[1:] if g is not None).sqrt().item()
        assert gn>0 and rn>0 and gr[0].norm()>0
        results['gradients'] = dict(generation_to_encoding_is_none=True,generation_last_block=gn,
            reconstruction_last_block=rn,reconstruction_to_encoding=gr[0].norm().item())
        del z,gen,rec,gg,gr,params
        # Unequal native groups are reduced by images, not by count groups.
        results['image_weighting'] = dict(local_loss_divisor=16*4,
            explicit_error=abs(backward_mean(torch.tensor([1.,2.,6.],device=device)).item()-(1+2+6)/(3*4)))
        # Exercise the actual source ODE/CFG path while forbidding all image access.
        def forbidden(*unused):
            raise AssertionError('Generation attempted to access image patches')
        h = model.core.patch_embed.register_forward_pre_hook(forbidden)
        nfes = []
        h2 = model.core.encoder.up_sample.register_forward_pre_hook(lambda module,inputs:nfes.append(list(inputs[0].shape)))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            sample = model.generate(labels[:1],draws['encoding_noise'][:1])
        h.remove(); h2.remove()
        assert sample.shape == (1,3,256,256) and torch.isfinite(sample).all()
        assert all(shape[1]==count for shape in nfes)
        results['generation'] = dict(signature=str(inspect.signature(model.generate)),image_access_forbidden=True,
            ode_encoder_calls=len(nfes),physical_count=count,finite=True)
        handle.remove()
        results['physical_final_projection_shapes'] = sorted({tuple(x) for x in physical})
        del sample, model,images,labels,draws,d
        gc.collect(); torch.cuda.empty_cache()
        # Real BF16 micro16 with unmodified seed0 scientific initialization, but
        # engineering-only optimizer state. The four-rank proof is separate.
        model = build_native(cfgs[count]).to(device).train()
        assert not model.core.lpips_loss.training
        assert all(not p.requires_grad for p in model.core.lpips_loss.parameters())
        images = load_images(rows,[False]*16,device)
        labels = torch.tensor([r['class_index'] for r in rows],device=device)
        draws = select_draws(Streams(0).draw(16),list(range(16)),device)
        opts,names = make_optimizers(model,cfgs[count]['training'])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            values = model(images,labels,draws)
        assert torch.isfinite(values['per_image']).all()
        backward_mean(values['per_image']).backward()
        norm = clip_gradients([p for p in model.parameters() if p.requires_grad])
        for o in opts.values(): o.step()
        torch.cuda.synchronize()
        results['bf16_micro16'] = dict(passed=True, images=16,denoising_images=48,
            gradient_norm_before_clip=norm.item(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.perf_counter()-started,
            optimizer_parameter_names=names,engineering_only=True,scientific_updates=0)
        all_results[str(count)] = results
        emit(out/f'native{count}.json',results)
        del model,opts,images,labels,draws,values
        gc.collect(); torch.cuda.empty_cache()
    emit(out/'results.json',dict(native_checks=all_results,passed=True,
        four_rank_ddp=None,four_rank_ddp_reason='Separate real four-rank check still required.',
        scientific_a2_pass=False,training_authorized_by_this_receipt=False))

if __name__=='__main__':
    main()
