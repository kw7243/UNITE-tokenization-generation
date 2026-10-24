"""Native 16/64 GE paths; source model, optimizer selectors, and decoder retained."""
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from models.unite import UNITE
from utils.optim import build_optimizer, build_optimizer_muon
from campaign.adapters.prefix import install_register_only


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slice_register_table(core, count):
    assert count in (16, 64) and core.num_latent_tokens == 64
    # All common initialized tensors are preserved, including the 64-path RoPE cache.
    # The cache has spare capacity; the forward processes exactly count registers.
    if count != 64:
        core.latent_tokens = nn.Parameter(core.latent_tokens.detach()[:, :count].clone())
    core.num_latent_tokens = count
    return core


def build_native(cfg):
    seed_all(cfg['seed'])
    core = UNITE(cfg['model']['gen_tok'], num_latent_tokens=64, num_classes=100)
    core = slice_register_table(core, cfg['model']['native_count'])
    return NativeA2(core, joint=cfg['objective'] == 'joint')


class NativeA2(nn.Module):
    def __init__(self, core, joint=True):
        super().__init__()
        self.core = core
        self.joint = joint
        self.count = core.num_latent_tokens
        assert self.count in (16, 64)
        assert core.flow_steps_per_recon == core.flow_mini_batch == 3
        assert core.perceptual_weight == core.gen_loss_weight == 1
        install_register_only(core.encoder)
        core.lpips_loss.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.core.lpips_loss.eval()
        return self

    def normalize(self, value):
        with torch.autocast(value.device.type, enabled=False):
            return self.core.encoder_layer_norm(value.float())

    def encode(self, images, noise, times):
        assert noise.shape == (len(images), 64, 32)
        labels = torch.zeros(len(images), device=images.device, dtype=torch.long)
        z = self.core.encoder(noise[:, :self.count], t=times, y=labels,
                              pos_embed=self.core.latent_tokens.expand(len(images), -1, -1),
                              img_patch_embed=self.core.patch_embed(images),
                              force_drop_ids_y_embedder=torch.ones_like(labels))
        assert z.shape == (len(images), self.count, 32)
        return self.normalize(z)

    def reconstruction(self, z, images, draws, corrupt=True):
        if corrupt:
            t = draws['recon_time'][:, None, None]
            mixed = t*z+(1-t)*draws['recon_noise'][:, :self.count]
            z = torch.where(draws['recon_mask'][:, None, None], mixed, z)
        decoded = self.core.decode(z, representation_phase=False)
        with torch.autocast(images.device.type, enabled=False):
            l1 = (decoded.float()-images.float()).abs().flatten(1).mean(1)
            perceptual = self.core.lpips_loss(images.float(), decoded.float(), reduction='none').flatten()
        return l1+perceptual, decoded

    def generation(self, z, labels, draws, fixed_time=None, disable_dropout=False, return_draws=False):
        b, k, d = z.shape
        target = z.detach()[:, None].expand(b, 3, k, d).reshape(b*3, k, d)
        noise = draws['diffusion_noise'][:, :, :k].reshape(b*3, k, d)
        times = draws['diffusion_time'].reshape(-1)
        if fixed_time is not None:
            times = torch.full_like(times, fixed_time)
        noised = times[:, None, None]*target+(1-times[:, None, None])*noise
        assert not target.requires_grad and not noised.requires_grad
        drop = draws['class_dropout'].reshape(-1).long()
        if disable_dropout:
            drop = torch.zeros_like(drop)
        prediction = self.core.encoder(noised, t=times, y=labels.repeat_interleave(3),
                                       pos_embed=self.core.latent_tokens.expand(b*3, -1, -1),
                                       force_drop_ids_y_embedder=drop,
                                       checkpoint_blocks=False)
        prediction = self.normalize(prediction)
        with torch.autocast(z.device.type, enabled=False):
            xt = noised.float()
            denom = (1-times.float()).clamp_min(.05)[:, None, None]
            # Preserve the source velocity difference and the SUM of M=3 draws.
            draw_loss = (((prediction.float()-xt)/denom-(target.float()-xt)/denom)**2).mean((1, 2))
        per_draw = draw_loss.reshape(b, 3)
        return per_draw if return_draws else per_draw.sum(1)

    def forward(self, images, labels, draws):
        z = self.encode(images, draws['encoding_noise'], draws['encoding_time'])
        rec, _ = self.reconstruction(z, images, draws)
        per_draw = self.generation(z, labels, draws, return_draws=True) if self.joint else None
        gen = per_draw.sum(1) if self.joint else torch.zeros_like(rec)
        return dict(per_image=rec+gen, reconstruction=rec, generation_sum=gen,
                    generation_draws=per_draw, latents=z.detach())

    @torch.no_grad()
    def generate(self, labels, noise, *, cfg_scale=2., cfg_interval=(.1, 1.)):
        assert not self.training and noise.shape == (len(labels), 64, 32)
        # The native source CFG/ODE path has no image input or length embedding.
        # Count is a model property, not a source-image property.
        return self.core.diffusion_generate(device=labels.device, num_visuals=len(labels),
                                            cfg_scale=cfg_scale, cfg_interval=cfg_interval,
                                            cfg_norm_order='norm_first', y_given=labels,
                                            noise=noise[:, :self.count])


def parameter_groups(model):
    core = model.core
    muon = core.get_learnable_params(muon=True)
    adamw = core.get_learnable_params(non_muon=True)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for p in muon+adamw]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    names = {id(p): name for name, p in model.named_parameters()}
    return {'muon': muon, 'adamw': adamw}, {key: [names[id(p)] for p in values]
                                        for key, values in [('muon', muon), ('adamw', adamw)]}


def make_optimizers(model, cfg):
    groups, names = parameter_groups(model)
    optimizers = {}
    optimizers['muon'], _ = build_optimizer_muon(groups['muon'], {'optimizer': cfg['muon']})
    optimizers['adamw'], _ = build_optimizer(groups['adamw'], {'optimizer': cfg['adamw']})
    return optimizers, names


def executed_lr(update):
    assert update >= 1
    return 1e-4*min(update/500., 1.)


def backward_mean(per_image, accumulation=4):
    # DDP averages ranks. The local image SUM / (n*A) yields global image mean.
    return per_image.float().sum()/(len(per_image)*accumulation)


def clip_gradients(parameters):
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, 3., error_if_nonfinite=True)
    except RuntimeError as exc:
        if 'non-finite' in str(exc) or 'nonfinite' in str(exc):
            raise FloatingPointError('Nonfinite gradient norm before clipping') from exc
        raise
