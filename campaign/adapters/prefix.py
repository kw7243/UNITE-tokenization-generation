"""Minimal retained-prefix paths for the plan's essential A1 correctness checks."""
from contextlib import contextmanager

import torch
from torch import nn


def install_register_only(encoder):
    """Reuse the source forward and slice its pre-projection register positions."""
    original = encoder.forward

    def forward(x, *args, **kwargs):
        register_count = x.shape[1]
        def keep_registers(module, inputs):
            hidden, condition = inputs
            assert hidden.shape[1] >= register_count
            return hidden[:, :register_count], condition
        handle = encoder.final_layer.register_forward_pre_hook(keep_registers)
        try:
            return original(x, *args, **kwargs)
        finally:
            handle.remove()
    encoder.forward = forward


class PrefixAdapter(nn.Module):
    """One image encoding, physical count grouping, explicit per-image losses."""

    allowed_counts = (4, 16, 64)

    def __init__(self, core):
        super().__init__()
        if core.num_latent_tokens != 64:
            raise ValueError("The prefix adapter requires exactly 64 candidates")
        self.core = core
        install_register_only(self.core.encoder)
        width = self.core.encoder.hidden_size
        self.encoder_length = nn.Embedding(3, width)
        self.decoder_length_after_up = nn.Embedding(3, 768)
        self.decoder_length_after_embed = nn.Embedding(3, 768)
        for embedding in (self.encoder_length, self.decoder_length_after_up,
                          self.decoder_length_after_embed):
            nn.init.zeros_(embedding.weight)
        self._active_length = None

    def train(self, mode=True):
        super().train(mode)
        self.core.lpips_loss.eval()
        return self

    def _groups(self, counts):
        if counts.ndim != 1 or counts.dtype != torch.long:
            raise ValueError("counts must be a one-dimensional int64 tensor")
        if not all(int(k) in self.allowed_counts for k in counts.tolist()):
            raise ValueError("Unsupported retained count")
        for count in self.allowed_counts:
            indices = torch.nonzero(counts == count, as_tuple=True)[0]
            if len(indices):
                yield count, indices

    @contextmanager
    def _length(self, count):
        if self._active_length is not None:
            raise RuntimeError("Nested length contexts are not supported")
        self._active_length = count
        row = self.allowed_counts.index(count)
        handles = [
            self.core.encoder.t_embedder.register_forward_hook(
                lambda module, inputs, output: output + self.encoder_length.weight[row]),
            self.core.up_sample_decoder.register_forward_hook(
                lambda module, inputs, output: output + self.decoder_length_after_up.weight[row]),
            self.core.decoder.decoder_embed.register_forward_hook(
                lambda module, inputs, output: output + self.decoder_length_after_embed.weight[row]),
        ]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active_length = None

    def encode(self, images, noise, modulation_times):
        if noise.shape != (len(images), 64, 32):
            raise ValueError("Encoding noise must cover all64 candidate positions")
        patches = self.core.patch_embed(images)
        labels = torch.full((len(images),), 7, device=images.device, dtype=torch.long)
        dropped = torch.ones_like(labels)
        positions = self.core.latent_tokens[:, :64].expand(len(images), -1, -1)
        registers = self.core.encoder(
            noise, t=modulation_times, y=labels, pos_embed=positions,
            img_patch_embed=patches, force_drop_ids_y_embedder=dropped)
        assert registers.shape == noise.shape
        return self.core.encoder_layer_norm(registers)

    def decode(self, full_latents, counts):
        outputs, order = [], []
        for count, indices in self._groups(counts):
            retained = full_latents.index_select(0, indices)[:, :count]
            with self._length(count):
                outputs.append(self.core.decode(retained, representation_phase=False))
            order.append(indices)
        permutation = torch.argsort(torch.cat(order))
        return torch.cat(outputs).index_select(0, permutation)

    def denoise(self, noised_full, counts, times, labels, class_drop_mask):
        """Return physical retained outputs, keeping each group's image indices."""
        outputs = []
        for count, indices in self._groups(counts):
            retained = noised_full.index_select(0, indices)[:, :count]
            positions = self.core.latent_tokens[:, :count].expand(len(indices), -1, -1)
            with self._length(count):
                prediction = self.core.encoder(
                    retained, t=times.index_select(0, indices),
                    y=labels.index_select(0, indices), pos_embed=positions,
                    force_drop_ids_y_embedder=class_drop_mask.index_select(0, indices))
            outputs.append((count, indices, self.core.encoder_layer_norm(prediction)))
        return outputs

    def reconstruction_losses(self, latents, images, counts, noise, times, mask):
        mixed = times[:, None, None] * latents + (1-times[:, None, None]) * noise
        noised = torch.where(mask[:, None, None], mixed, latents)
        decoded = self.decode(noised, counts)
        per_image = (decoded-images).abs().flatten(1).mean(1)
        per_image = per_image + self.core.lpips_loss(images, decoded, reduction="none").flatten()
        return per_image, decoded

    def generation_losses(self, latents, counts, labels, noises, times, class_drop_mask):
        target = latents.detach()
        draws = []
        for noise, time in zip(noises, times):
            noised = time[:, None, None] * target + (1-time[:, None, None]) * noise
            assert not noised.requires_grad and not target.requires_grad
            values, order = [], []
            for count, indices, prediction in self.denoise(noised, counts, time, labels, class_drop_mask):
                clean = target.index_select(0, indices)[:, :count]
                xt = noised.index_select(0, indices)[:, :count]
                denom = (1-time.index_select(0, indices)).clamp_min(.05)[:, None, None]
                velocity_gt = (clean-xt)/denom
                velocity_prediction = (prediction-xt)/denom
                values.append((velocity_prediction-velocity_gt).square().mean((1, 2)))
                order.append(indices)
            draws.append(torch.cat(values).index_select(0, torch.argsort(torch.cat(order))))
        if not draws:
            raise ValueError("At least one denoising draw is required")
        return torch.stack(draws).sum(0)

    @staticmethod
    def image_mean(per_image):
        if per_image.ndim != 1 or not len(per_image):
            raise ValueError("Expected one scalar per image")
        return per_image.sum()/len(per_image)

    @torch.no_grad()
    def generate(self, labels, counts, noise, *, cfg_scale=2.0, cfg_interval=(.1, 1.0)):
        """Generation has no image/patch/encoding-feature input."""
        if self.training:
            raise RuntimeError("Use eval mode for generation")
        if noise.shape != (len(labels), 64, 32):
            raise ValueError("Generation noise must cover all64 positions")
        outputs, order = [], []
        for count, indices in self._groups(counts):
            original_count = self.core.num_latent_tokens
            try:
                self.core.num_latent_tokens = count
                with self._length(count):
                    outputs.append(self.core.diffusion_generate(
                        device=labels.device, num_visuals=len(indices),
                        cfg_scale=cfg_scale, cfg_interval=cfg_interval,
                        cfg_norm_order="norm_first",
                        y_given=labels.index_select(0, indices),
                        noise=noise.index_select(0, indices)[:, :count]))
            finally:
                self.core.num_latent_tokens = original_count
            order.append(indices)
        return torch.cat(outputs).index_select(0, torch.argsort(torch.cat(order)))
