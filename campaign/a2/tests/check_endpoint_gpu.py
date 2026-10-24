"""Initialized native ODE/feature integration check; never a scientific endpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'third_party/unite'))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from campaign.a2.data import sha
from campaign.a2.endpoint_features import load_ema, retained_features, save_json
from campaign.a2.evaluate import generation_inputs
from campaign.a2.native import build_native
from campaign.a2.train import verify_config
from utils.fid import extract_inception_features, get_inception_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-name', required=True)
    parser.add_argument('--engineering-receipt', type=Path, required=True)
    parser.add_argument('--engineering-receipt-sha', required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and (ROOT/'STAGING_MANIFEST.json').is_file()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    device = torch.device('cuda', 0); torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    assert args.gpu_name in props.name and props.total_memory >= 24000000000
    assert (torch.ones(1, device=device)+1).item() == 2
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True, exist_ok=False)
    configs = {k: json.loads((ROOT/f'campaign/a2/configs/r6-source-ddp-unused/NATIVE{k}_JOINT.yaml').read_text()) for k in (64, 16)}
    for cfg in configs.values():
        verify_config(cfg)
    os.environ['INCEPTION_WEIGHTS'] = configs[64]['evaluation']['metric_weights']['path']
    assert sha(os.environ['INCEPTION_WEIGHTS']) == configs[64]['evaluation']['metric_weights']['sha256']
    inception = get_inception_model(device).requires_grad_(False)
    _, common = generation_inputs()
    labels = torch.tensor([0, 99]*8, device=device)
    noise = common[:16].to(device)
    original = noise.clone()
    records = {}
    for count in (64, 16):
        model = build_native(configs[count]).to(device).eval().requires_grad_(False)
        # The image-encoding method must not be consulted by generation.
        def reject_image_encoding(*args, **kwargs):
            raise AssertionError('Sampling attempted image encoding')
        model.encode = reject_image_encoding
        calls = [0]
        def count_call(module, inputs):
            assert inputs[0].shape[:2] == (32, count), 'CFG must retain the physical native count'
            calls[0] += 1
        handle = model.core.encoder.blocks[0].register_forward_pre_hook(count_call)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                images = model.generate(labels, noise).float().clamp(0, 1)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-start
            assert images.shape == (16, 3, 256, 256) and torch.isfinite(images).all()
            assert calls[0] > 0 and torch.equal(noise, original)
            values = np.lib.format.open_memmap(args.output/f'native{count}-fixture-features.npy',
                                               mode='w+', dtype=np.float32, shape=(16, 2048))
            logits = retained_features(values, images, inception, 0)
            values.flush()
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                expected, expected_logits = extract_inception_features(images, inception, batch_size=16)
            error = np.linalg.norm(values-expected.numpy())/max(np.linalg.norm(expected.numpy()), 1e-30)
            assert error <= 1e-5 and np.array_equal(logits, expected_logits.numpy())
            records[str(count)] = dict(microbatch=16, cfg_batch=32, physical_registers=count,
                shape=list(images.shape), finite=True, no_image_encoding=True, input_noise_unchanged=True,
                nfe=calls[0], sampling_decode_seconds=seconds, feature_relative_l2=float(error),
                feature_precision='FP32_TF32_OFF', peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        finally:
            handle.remove()
        del model, values, images, expected, expected_logits
        torch.cuda.empty_cache()
    assert sha(args.engineering_receipt) == args.engineering_receipt_sha
    native_proof = json.loads(args.engineering_receipt.read_text())
    engineering = native_proof['native']['64']['engineering_checkpoint']
    assert engineering['scientific_handoff'] is False
    train_path = ROOT/'campaign/a2/configs/r6-source-ddp-unused/NATIVE64_JOINT.yaml'
    try:
        load_ema(dict(checkpoint=engineering, training_config=dict(sha256=sha(train_path))),
                 configs[64], {}, torch.device('cpu'))
    except AssertionError as exc:
        assert 'Not a complete5000-update scientific checkpoint' in str(exc), 'Unexpected engineering-input rejection cause'
    else:
        raise AssertionError('Engineering snapshot was accepted as scientific stable64')
    save_json(args.output/'results.json', dict(schema='jat-a2-endpoint-engineering-check.v1',
        phase='INITIALIZED_NATIVE_ODE_AND_FEATURE_CHECK', phase_status='COMPLETED', engineering_only=True,
        scientific_updates=0, generated_fixture_images=32, scientific_generated_images=0,
        valid64_gate=False, trained_checkpoint=None, role='A2_ENDPOINT_ENGINEERING_CHECK',
        job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename, gpu=props.name, uuid=str(props.uuid),
        vram_bytes=props.total_memory, torch=torch.__version__, cuda=torch.version.cuda,
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        native=records, engineering_checkpoint_rejected=True,
        engineering_receipt_sha256=args.engineering_receipt_sha, limitation='Initialized ODE/feature smoke only. No learned-model quality, distributed/resume proof or stable64 claim.'))


if __name__ == '__main__':
    main()
