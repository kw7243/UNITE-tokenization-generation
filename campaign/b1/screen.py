"""SCREEN only: source prefix adapter, M3 autograd, and explicit AdamW tables."""
import torch
from models.unite import UNITE
from campaign.adapters.prefix import PrefixAdapter
from campaign.a2.native import seed_all
from utils.optim import build_optimizer, build_optimizer_muon

NEW_NAMES = ('encoder_length.weight', 'decoder_length_after_up.weight',
             'decoder_length_after_embed.weight')


def build_screen(cfg):
    seed_all(cfg['seed'])
    return Screen(UNITE(cfg['model']['gen_tok'], num_latent_tokens=64, num_classes=100))


class Screen(PrefixAdapter):
    allowed_counts = (4, 16, 64)

    def __init__(self, core):
        super().__init__(core)
        self.core.lpips_loss.requires_grad_(False)

    def normalize(self, value):
        with torch.autocast(value.device.type, enabled=False):
            return self.core.encoder_layer_norm(value.float())

    def encode(self, images, noise, modulation_times):
        # No selected count is accepted or passed into the image encoding.
        if noise.shape != (len(images), 64, 32):
            raise ValueError('Encoding requires 64 candidate registers')
        labels = torch.zeros(len(images), dtype=torch.long, device=images.device)
        value = self.core.encoder(noise, t=modulation_times, y=labels,
            pos_embed=self.core.latent_tokens.expand(len(images), -1, -1),
            img_patch_embed=self.core.patch_embed(images),
            force_drop_ids_y_embedder=torch.ones_like(labels))
        if value.shape != noise.shape:
            raise ValueError('Encoder returned non-register positions')
        return self.normalize(value)

    def _groups(self, counts):
        if counts.numel() == 0:
            raise ValueError('A local logical batch must contain images')
        yield from super()._groups(counts)

    def reconstruction(self, z, images, counts, draws, corrupt=True):
        if corrupt:
            t = draws['recon_time'][:, None, None]
            mixed = t*z+(1-t)*draws['recon_noise']
            z = torch.where(draws['recon_mask'][:, None, None], mixed, z)
        decoded = self.decode(z, counts)
        with torch.autocast(images.device.type, enabled=False):
            l1 = (decoded.float()-images.float()).abs().flatten(1).mean(1)
            perceptual = self.core.lpips_loss(images.float(), decoded.float(), reduction='none').flatten()
        return l1+perceptual, decoded

    def generation(self, z, labels, counts, draws):
        target = z.detach()
        values, order = [], []
        for k, indices in self._groups(counts):
            b = len(indices)
            clean = target[indices, :k][:, None].expand(b, 3, k, 32).reshape(b*3, k, 32)
            noise = draws['diffusion_noise'][indices, :, :k].reshape(b*3, k, 32)
            times = draws['diffusion_time'][indices].reshape(-1)
            noised = times[:, None, None]*clean+(1-times[:, None, None])*noise
            if clean.requires_grad or noised.requires_grad:
                raise RuntimeError('Denoising target or input escaped the detach boundary')
            with self._length(k):
                predicted = self.core.encoder(noised, t=times, y=labels[indices].repeat_interleave(3),
                    pos_embed=self.core.latent_tokens[:, :k].expand(b*3, -1, -1),
                    force_drop_ids_y_embedder=draws['class_dropout'][indices].reshape(-1).long(),
                    checkpoint_blocks=False)
            with torch.autocast(z.device.type, enabled=False):
                prediction = self.normalize(predicted)
                denominator = (1-times.float()).clamp_min(.05)[:, None, None]
                # Source velocity difference, per-image token/channel normalization.
                per_draw = (((prediction-noised.float())/denominator
                             -(clean.float()-noised.float())/denominator)**2).mean((1, 2))
            values.append(per_draw.reshape(b, 3))
            order.append(indices)
        return torch.cat(values).index_select(0, torch.argsort(torch.cat(order)))

    def forward(self, images, labels, counts, draws):
        if not torch.all((counts == 16) | (counts == 64)):
            raise ValueError('SCREEN trains only 16/64; count4 stays untrained')
        z = self.encode(images, draws['encoding_noise'], draws['encoding_time'])
        rec, _ = self.reconstruction(z, images, counts, draws)
        per_draw = self.generation(z, labels, counts, draws)
        return dict(per_image=rec+per_draw.sum(1), reconstruction=rec,
                    generation_sum=per_draw.sum(1), generation_draws=per_draw, latents=z.detach())


def parameter_groups(model):
    muon = model.core.get_learnable_params(muon=True)
    adamw = model.core.get_learnable_params(non_muon=True)
    named = dict(model.named_parameters())
    adamw += [named[n] for n in NEW_NAMES]
    ids = [id(p) for p in muon+adamw]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Each trainable parameter must occur exactly once')
    reverse = {id(p): n for n, p in named.items()}
    groups = dict(muon=muon, adamw=adamw)
    return groups, {key: [reverse[id(p)] for p in group] for key, group in groups.items()}


def make_optimizers(model, cfg):
    groups, names = parameter_groups(model)
    return dict(muon=build_optimizer_muon(groups['muon'], {'optimizer': cfg['muon']})[0],
                adamw=build_optimizer(groups['adamw'], {'optimizer': cfg['adamw']})[0]), names
