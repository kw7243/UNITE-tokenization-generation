"""Standalone entrypoint to the released training evaluation; core source is unchanged."""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml

from models.unite import UNITE
from engines import eval as evaluation
from utils.fid import get_inception_model, extract_inception_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--checkpoint-sha256", required=True)
parser.add_argument("--encoder", choices=["Base", "Large"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=["check", "smoke", "fid"], required=True)
parser.add_argument("--protocol", choices=["release", "paper-text"], required=True)
parser.add_argument("--cfg-scale", type=float, default=2.4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()
if args.protocol == "paper-text":
    from modules import denoiser
    from modules.denoiser_utils.integrators import ode
    class PaperTextODE(ode):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.t = 0.1 + 0.9 * self.t
    denoiser.ode = PaperTextODE
args.output.mkdir(parents=True, exist_ok=True)
source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert (Path.cwd() / "REPRODUCIBILITY_METADATA.json").is_file(), "Run only from a complete source stage"
assert (Path.cwd() / "STAGING_MANIFEST.json").is_file(), "A verified staging manifest is required"
assert os.getenv("SLURM_JOB_ID"), "Run only in a Slurm allocation"
assert args.batch_size > 0
assert 1.0 <= args.cfg_scale <= 4.0
start = time.perf_counter()
record = {"run_id": args.output.name, "node_id": "RELEASED_CHECKPOINT_" + args.mode.upper(),
          "status": "PLANNED", "mode": args.mode, "protocol": args.protocol,
          "owner": "unite-reproduce-20261023", "job_id": os.environ["SLURM_JOB_ID"],
          "node": platform.node(), "stage": str(Path.cwd()), "seed": args.seed,
          "checkpoint": str(args.checkpoint), "checkpoint_sha256_expected": args.checkpoint_sha256,
          "encoder": args.encoder, "batch_size": args.batch_size,
          "source_revision": source_revision, "upstream_source_revision": "2389f2c65fedbd43ada851e69b2b810efd23af7d",
          "package_lock_sha256": hashlib.sha256(Path("reproduction/locks/packages-cpu-setup-2826271.txt").read_bytes()).hexdigest(),
          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
          "python": sys.version, "cfg_scale": args.cfg_scale,
          "cfg_interval": [0.1, 1.0], "cfg_norm_order": "norm_first",
          "ode_interval": [0.1 if args.protocol == "paper-text" else 0.0, 1.0], "ode_solver": "dopri5", "atol": 1e-6, "rtol": 1e-3,
          "flow_steps_per_recon": 14, "timestep_shift_alpha": 0.5,
          "paper_interval_difference": "Explicit paper-text interval variant; source Gaussian initial state and drift retained." if args.protocol == "paper-text" else "Paper states integration [0.1,1.0]; this run follows released source integration [0,1]."}
def save():
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
save()
try:
    h = hashlib.sha256()
    with args.checkpoint.open("rb") as f:
        for data in iter(lambda: f.read(8 << 20), b""):
            h.update(data)
    assert h.hexdigest() == args.checkpoint_sha256, "Checkpoint bytes changed"
    cfg = yaml.safe_load(Path("configs/imagenet.yaml").read_text())["gen_tok"]
    cfg.update(encoder_model_size=args.encoder, decoder_model_size="Base", flow_steps_per_recon=14)
    record["gen_tok_config"] = cfg
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    assert "ema" in checkpoint and "model" in checkpoint
    state = checkpoint["ema"]
    renamed = {key: key.replace("._orig_mod.", ".") for key in state}
    assert len(set(renamed.values())) == len(state), "Compiled-key normalization collides"
    record["compiled_key_renames"] = {k: v for k, v in renamed.items() if k != v}
    model = UNITE(cfg, num_latent_tokens=256, num_classes=1000).eval().requires_grad_(False)
    model.load_state_dict({renamed[k]: v for k, v in state.items()}, strict=True)
    record["checkpoint_epoch"] = checkpoint.get("epoch")
    record["checkpoint_step"] = checkpoint.get("step")
    record["checkpoint_cfg_scale"] = checkpoint.get("current_best_cfg_scale")
    record["parameter_count_including_frozen_loss"] = sum(p.numel() for p in model.parameters())
    record["strict_state_load"] = True
    record["model_dtype"] = str(next(model.parameters()).dtype)
    record["encoder_parameters"] = sum(p.numel() for p in model.encoder.parameters())
    record["decoder_parameters"] = sum(p.numel() for p in model.decoder.parameters())
    del state, checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # PyTorch 2.10 Inductor reads legacy getters; use one API family consistently.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    record["tf32_configuration_api"] = "legacy_only_compatibility_with_pytorch_2_10_inductor"
    if args.mode == "check":
        inception = get_inception_model(torch.device("cpu"))
        record["inception_loaded_on_cpu"] = True
        del inception
    else:
        assert torch.cuda.is_available()
        device = torch.device("cuda", 0)
        torch.cuda.manual_seed_all(args.seed)
        model = model.to(device)
        model.decoder = torch.compile(model.decoder, mode="default")
        record["decoder_compile_mode"] = "default"
        record["gpu_model"] = torch.cuda.get_device_name(0)
        record["gpu_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        record["driver_version"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0]
        record["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        record["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        nfe_count = [0]
        def count_encoder_evaluation(module, inputs):
            nfe_count[0] += 1
        nfe_hook = model.encoder.blocks[0].register_forward_pre_hook(count_encoder_evaluation)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        if args.mode == "smoke":
            with torch.no_grad():
                labels = torch.linspace(0, 999, args.batch_size, device=device).to(torch.long)
                batches, timings, nfe_per_call = [], [], []
                for smoke_index in range(2):
                    torch.cuda.synchronize()
                    call_start = time.perf_counter()
                    previous_nfe = nfe_count[0]
                    images = model.diffusion_generate(device=device, num_visuals=args.batch_size, cfg_scale=args.cfg_scale,
                        cfg_interval=(0.1, 1.0), cfg_norm_order="norm_first", y_given=labels).clamp(0, 1)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter() - call_start)
                    nfe_per_call.append(nfe_count[0] - previous_nfe)
                    assert tuple(images.shape) == (args.batch_size, 3, 256, 256) and torch.isfinite(images).all()
                    batches.append(images.cpu().numpy())
                record["sampling_seconds_per_call"] = timings
                record["encoder_nfe_per_call"] = nfe_per_call
                inception = get_inception_model(device)
                features, logits = extract_inception_features(images, inception, batch_size=args.batch_size)
                assert tuple(features.shape) == (args.batch_size, 2048) and torch.isfinite(features).all()
                assert torch.isfinite(logits).all()
                np.savez(args.output / "smoke-images.npz", images=np.stack(batches),
                         labels=labels.cpu().numpy(), features=features.numpy(), logits=logits.numpy())
                record["sample_count"] = 2 * args.batch_size
                record["warmup_sample_count"] = args.batch_size
                record["steady_sample_count"] = args.batch_size
                record["batch_size"] = args.batch_size
                record["paper_metric"] = False
        else:
            def capture_fid(features, ref):
                value = source_fid(features, ref)
                np.save(args.output / "inception-features.npy", features.numpy())
                record["fid_full_precision"] = value
                return value
            def capture_is(logits):
                result = source_is(logits)
                np.save(args.output / "inception-logits.npy", logits.numpy())
                record["inception_score"], record["inception_score_split_std"] = result
                return result
            source_fid, source_is = evaluation.compute_fid_from_features, evaluation.compute_inception_score
            evaluation.compute_fid_from_features, evaluation.compute_inception_score = capture_fid, capture_is
            _, fid = evaluation.evaluate_fid(model, device, global_step=0,
                ema_decay=0.9978, image_size=256, cfg_scales=[args.cfg_scale],
                cfg_intervals=[(0.1, 1.0)], cfg_norm_orders=["norm_first"],
                num_images=50000, num_classes=1000, batch_size=args.batch_size,
                log_to_wandb=False, log_fid_best=False, current_best_cfg_scale=args.cfg_scale)
            record.update(fid_release_return=fid, sample_count=50000, images_per_class=50,
                          paper_metric=False, comparison_scope="Released-code numerical comparison with declared paper differences")
        torch.cuda.synchronize()
        record["encoder_nfe_total"] = nfe_count[0]
        nfe_hook.remove()
        record["evaluation_elapsed_seconds_including_first_compile"] = time.perf_counter() - evaluation_start
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    record["status"] = "PASS"
except Exception as exc:
    record.update(status="BLOCKED_IMPLEMENTATION", error=repr(exc))
    raise
finally:
    record["elapsed_seconds"] = time.perf_counter() - start
    save()
