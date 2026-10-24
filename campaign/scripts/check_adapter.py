"""Only the three plan-critical checks. No optimizer updates or training campaign."""
from pathlib import Path
import inspect
import json
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from campaign.adapters.prefix import PrefixAdapter, install_register_only


def relative_l2(actual, expected):
    return ((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-12)).item()


def check_essential(core, images, seed=20261023):
    """Use the loaded Base EMA as a nondegenerate correctness fixture."""
    start=time.perf_counter()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core.eval()
    device=images.device
    proof={"precision":"FP32","tf32":False,"tolerance":1e-5,"fixture":"BaseEMA checkpoint, no optimizer updates",
           "test_batch_size":3,"counts":[16,64,16],"generation_counts":[4,16],
           "core_fixture_num_classes":core.num_classes}
    with torch.no_grad():
        torch.cuda.manual_seed_all(seed)
        native=core.encode(images[:1])
        original_forward=install_register_only(core.encoder)
        torch.cuda.manual_seed_all(seed)
        fixed=core.encode(images[:1])
        error=relative_l2(fixed,native)
        assert fixed.shape==(1,256,32)
        assert error<=1e-5,error
        proof["register_identity_256_relative_l2"]=error
        proof["register_shape_256"]=list(fixed.shape)
    # The source has been checked at256; now construct the isolated64-candidate fixture.
    # Restore the original encoder method before the PrefixAdapter installs one hook.
    core.encoder.forward=original_forward
    core.num_latent_tokens=64
    core.latent_tokens=torch.nn.Parameter(core.latent_tokens.detach()[:,:64].clone())
    core.flow_steps_per_recon=3
    core.flow_mini_batch=3
    core.gradient_checkpointing_denoiser=False
    adapter=PrefixAdapter(core).to(device).eval()
    with torch.no_grad():
        for table in (adapter.encoder_length,adapter.decoder_length_after_up,adapter.decoder_length_after_embed):
            assert torch.count_nonzero(table.weight)==0
            table.weight.copy_(torch.randn_like(table.weight)*.01)
    proof["length_embedding_fixture"]="Zero initialization verified, then nonzero seeded0.01-scale values exercise all length paths."
    noise=torch.randn(3,64,32,device=device)
    tau=torch.rand(3,device=device)*.01
    counts=torch.tensor([16,64,16],device=device)
    labels=torch.tensor([0,1,2],device=device)
    drop=torch.zeros(3,device=device,dtype=torch.long)
    recon_noise=torch.randn_like(noise)
    recon_t=torch.tensor([.72,.83,.91],device=device)
    recon_mask=torch.tensor([True,False,True],device=device)
    flow_noise=torch.randn(3,3,64,32,device=device)
    flow_t=torch.tensor([[.2,.5,.8],[.3,.6,.7],[.4,.45,.75]],device=device)
    with torch.no_grad():
        latents=adapter.encode(images,noise,tau)
        assert latents.shape==(3,64,32)
        proof["register_shape_64"]=list(latents.shape)
        decoded=adapter.decode(latents,counts)
        changed=latents.clone()
        inactive=torch.arange(64,device=device)[None,:]>=counts[:,None]
        changed[inactive]=torch.randn_like(changed[inactive])*100
        changed_decoded=adapter.decode(changed,counts)
        proof["inactive_decode_relative_l2"]=relative_l2(changed_decoded,decoded)
        assert proof["inactive_decode_relative_l2"]<=1e-5
        times=flow_t[0]
        grouped=adapter.denoise(latents,counts,times,labels,drop)
        changed_groups=adapter.denoise(changed,counts,times,labels,drop)
        errors=[]
        for (_,indices,pred),(_,changed_indices,changed_pred) in zip(grouped,changed_groups):
            assert torch.equal(indices,changed_indices)
            errors.append(relative_l2(changed_pred,pred))
        proof["inactive_denoise_relative_l2_max"]=max(errors)
        assert max(errors)<=1e-5
        single_decoded=torch.cat([adapter.decode(latents[i:i+1],counts[i:i+1]) for i in range(3)])
        proof["group_vs_single_decode_relative_l2"]=relative_l2(decoded,single_decoded)
        assert proof["group_vs_single_decode_relative_l2"]<=1e-5
        denoise_errors=[]
        for count,indices,pred in grouped:
            single=torch.cat([adapter.denoise(latents[i:i+1],counts[i:i+1],times[i:i+1],
                                             labels[i:i+1],drop[i:i+1])[0][2] for i in indices.tolist()])
            denoise_errors.append(relative_l2(pred,single))
        proof["group_vs_single_denoise_relative_l2_max"]=max(denoise_errors)
        assert max(denoise_errors)<=1e-5
        rec,_=adapter.reconstruction_losses(latents,images,counts,recon_noise,recon_t,recon_mask)
        gen=adapter.generation_losses(latents,counts,labels,flow_noise,flow_t,drop)
        loop_rec=[];loop_gen=[]
        for i in range(3):
            local_rec,_=adapter.reconstruction_losses(latents[i:i+1],images[i:i+1],counts[i:i+1],
                recon_noise[i:i+1],recon_t[i:i+1],recon_mask[i:i+1])
            local_gen=adapter.generation_losses(latents[i:i+1],counts[i:i+1],labels[i:i+1],
                flow_noise[:,i:i+1],flow_t[:,i:i+1],drop[i:i+1])
            loop_rec.append(local_rec[0]);loop_gen.append(local_gen[0])
        proof["per_image_reconstruction_loss_relative_l2"]=relative_l2(rec,torch.stack(loop_rec))
        proof["per_image_generation_loss_relative_l2"]=relative_l2(gen,torch.stack(loop_gen))
        for key in ("per_image_reconstruction_loss_relative_l2","per_image_generation_loss_relative_l2"):
            assert proof[key]<=1e-5,(key,proof[key])
        explicit=sum(loop_rec[i]+loop_gen[i] for i in range(3))/3
        reduced=adapter.image_mean(rec+gen)
        proof["image_weighted_scalar_relative_error"]=relative_l2(reduced,explicit)
        assert proof["image_weighted_scalar_relative_error"]<=1e-5
    # Targets and noised inputs detach the encoding graph, while the second GE pass is trainable.
    # Do not freeze inherited trainable parameters or enable frozen positions/LPIPS parameters.
    latents=adapter.encode(images[:1],noise[:1],tau[:1])
    rec,_=adapter.reconstruction_losses(latents,images[:1],counts[:1],recon_noise[:1],recon_t[:1],recon_mask[:1])
    gen=adapter.generation_losses(latents,counts[:1],labels[:1],flow_noise[:,:1],flow_t[:,:1],drop[:1])
    shared=[p for p in core.encoder.blocks[-1].parameters() if p.requires_grad]
    gen_grad=torch.autograd.grad(gen.mean(),[latents]+shared,retain_graph=True,allow_unused=True)
    rec_grad=torch.autograd.grad(rec.mean(),[latents]+shared,allow_unused=True)
    assert gen_grad[0] is None
    assert rec_grad[0] is not None and rec_grad[0].norm().item()>0
    gen_norm=sum(g.float().square().sum() for g in gen_grad[1:] if g is not None).sqrt().item()
    rec_norm=sum(g.float().square().sum() for g in rec_grad[1:] if g is not None).sqrt().item()
    assert np.isfinite(gen_norm) and gen_norm>0
    assert np.isfinite(rec_norm) and rec_norm>0
    proof.update(generation_encoding_gradient_is_none=True,
                 reconstruction_encoding_gradient_norm=rec_grad[0].norm().item(),
                 generation_last_shared_block_gradient_norm=gen_norm,
                 reconstruction_last_shared_block_gradient_norm=rec_norm)
    del latents,rec,gen,gen_grad,rec_grad
    adapter.train()
    assert not core.lpips_loss.training
    adapter.eval()
    signature=list(inspect.signature(adapter.generate).parameters)
    assert signature==["labels","counts","noise","cfg_scale","cfg_interval"],signature
    small_counts=torch.tensor([4,16],device=device)
    small_noise=noise[:2].clone()
    def forbid_images(module,inputs):
        raise AssertionError("Generation used image patch embedding")
    guard=core.patch_embed.register_forward_pre_hook(forbid_images)
    try:
        first=adapter.generate(labels[:2],small_counts,small_noise)
    finally:
        guard.remove()
    with torch.no_grad():
        adapter.encode(images.flip(0),noise,tau)
    guard=core.patch_embed.register_forward_pre_hook(forbid_images)
    try:
        second=adapter.generate(labels[:2],small_counts,small_noise)
    finally:
        guard.remove()
    proof["generation_after_changed_image_relative_l2"]=relative_l2(first,second)
    assert proof["generation_after_changed_image_relative_l2"]<=1e-5
    proof.update(real_image_encoding_inputs=9,generated_samples=4,generation_signature=signature,source_images_forbidden_during_generation=True,
                 frozen_lpips_eval_after_train=True,status="PASS",elapsed_seconds=time.perf_counter()-start)
    return proof
