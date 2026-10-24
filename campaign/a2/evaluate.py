"""A2 reconstruction and gradient probes; quality features remain a later workload."""
import json
from pathlib import Path

import numpy as np
import torch
from campaign.a2.data import eval_rows, load_images, per_image_draws


@torch.no_grad()
def reconstruction(model, rows, device, seed, out, *, batch_size=16):
    assert not model.training
    records, latents = [], []
    # Reporting LPIPS is separate from source training VGG LPIPS.
    import lpips
    alex = lpips.LPIPS(net='alex', version='0.1').to(device).eval().requires_grad_(False)
    for start in range(0, len(rows), batch_size):
        part = rows[start:start+batch_size]
        images = load_images(part, [False]*len(part), device)
        draws = {k: v.to(device) for k, v in per_image_draws(part, seed).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16):
            z = model.encode(images, draws['encoding_noise'], draws['encoding_time'])
            decoded = model.core.decode(z, representation_phase=False).float().clamp(0, 1)
        with torch.autocast(device.type, enabled=False):
            l1 = (decoded-images).abs().flatten(1).mean(1)
            perceptual = alex(2*decoded-1, 2*images-1).flatten()
        latents.append(z.float().cpu())
        for row, a, b in zip(part, l1.tolist(), perceptual.tolist()):
            records.append(dict(image_id=row['image_id'], class_index=row['class_index'],
                                recon_l1=a, recon_lpips=b, allocation_distortion=a+b))
    z = torch.cat(latents).double()
    result = {name: float(np.mean([r[name] for r in records]))
              for name in ('recon_l1', 'recon_lpips', 'allocation_distortion')}
    result.update(recon_l1_p90=float(np.quantile([r['recon_l1'] for r in records], .9)),
                  latent_image_variance=z.var(dim=0, unbiased=False).mean().item(),
                  latent_rms=z.square().mean().sqrt().item(),
                  norm_gamma_rms=model.core.encoder_layer_norm.weight.float().square().mean().sqrt().item(),
                  norm_beta_rms=model.core.encoder_layer_norm.bias.float().square().mean().sqrt().item(),
                  images=len(rows), mean_tokens=model.count, mean_tokens_sq=model.count**2,
                  rfid_eval=None, rfid_eval_reason='Feature extraction/scoring is a separate authorized endpoint workload.')
    Path(out).write_text(json.dumps(dict(summary=result, per_image=records), indent=2)+'\n')
    del alex
    return result


def gradient_probe(model, rows, device, seed, output):
    """32 per-image last-block gradients; three fixed t; one draw, no class drop."""
    assert len(rows) == 32 and not model.training
    named = sorted(model.core.encoder.blocks[-1].named_parameters())
    parameters = [p for _, p in named if p.requires_grad]
    records = []
    for row in rows:
        images = load_images([row], [False], device)
        labels = torch.tensor([row['class_index']], device=device)
        draws = {k: v.to(device) for k, v in per_image_draws([row], seed).items()}
        # Repeat the first draw so sum/3 is exactly the one-draw convention.
        draws['diffusion_noise'] = draws['diffusion_noise'][:, :1].expand(-1, 3, -1, -1)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            z = model.encode(images, draws['encoding_noise'], draws['encoding_time'])
            rec, _ = model.reconstruction(z, images, draws)
        gr = torch.autograd.grad(rec.sum(), parameters, allow_unused=True)
        gr = torch.cat([(g.float() if g is not None else torch.zeros_like(p)).flatten()
                        for g, p in zip(gr, parameters)])
        for t in (.25, .5, .75):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                gen = model.generation(z.detach(), labels, draws, fixed_time=t, disable_dropout=True)/3
            gg = torch.autograd.grad(gen.sum(), parameters, allow_unused=True)
            gg = torch.cat([(g.float() if g is not None else torch.zeros_like(p)).flatten()
                            for g, p in zip(gg, parameters)])
            cosine = (torch.dot(gr, gg)/(gr.norm()*gg.norm()+1e-12)).item()
            records.append(dict(image_id=row['image_id'], t=t, reconstruction_gradient_norm=gr.norm().item(),
                                generation_gradient_norm=gg.norm().item(), grad_cos_last_block=cosine))
    result = dict(parameter_names=[n for n, p in named if p.requires_grad], per_image=records,
                  by_time={str(t): float(np.mean([r['grad_cos_last_block'] for r in records if r['t'] == t]))
                           for t in (.25, .5, .75)})
    result['grad_cos_last_block'] = float(np.mean(list(result['by_time'].values())))
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    return result


def generation_eligible(histories):
    """Both joint arms need finite, improving observed DEV distortion."""
    for name in ('NATIVE64_JOINT', 'NATIVE16_JOINT'):
        values = histories.get(name, [])
        if len(values) < 2 or any(not np.isfinite(v) for v in values):
            return False
        if values[-1] >= values[0] or values[-1] >= values[-2]:
            return False
    return True


def paired_interference_interval(joint16, joint64, recon16, recon64, classes, seed=917):
    """A2 Section7 ratio of mean shared distortions; within-class paired bootstrap."""
    arrays = [np.asarray(a, dtype=np.float64) for a in (joint16, joint64, recon16, recon64)]
    classes = np.asarray(classes)
    def score(indices):
        means = [a[indices].mean() for a in arrays]
        return (means[0]/max(means[1], 1e-8))/max(means[2]/max(means[3], 1e-8), 1e-8)
    rng = np.random.RandomState(seed)
    groups = [np.flatnonzero(classes == c) for c in sorted(set(classes))]
    assert len(set(map(len, groups))) == 1, 'DEV must be balanced'
    samples = [score(np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]))
               for _ in range(1000)]
    estimate = score(np.arange(len(classes)))
    low, high = np.quantile(samples, [.025, .975])
    return dict(ratio=estimate, ci95=[float(low), float(high)],
                joint_pressure_screen=bool(estimate >= 1.10 and low > 1.0))
