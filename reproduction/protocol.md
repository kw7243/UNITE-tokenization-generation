# Released UNITE checkpoint reproduction

## Numerical targets

Primary: paper v1 Table 1, ImageNet 256x256, released EMA checkpoints.

| Checkpoint | Training epochs in README | FID-50K | Inception Score |
| --- | ---: | ---: | ---: |
| UNITE-B | 240 | 2.12 | 294.1 |
| UNITE-L | 120 | 1.73 | 296.0 |

Keep Appendix Table 4 as a separate inconsistent published comparison: the nominal same 240-epoch Base checkpoint has balanced dopri5 FID 2.735 / IS 268.1 and balanced 50-step Heun FID 2.789 / IS 287.6.
Do not use that solver gap as an arbitrary tolerance on Table 1.
The paper gives no repeated-seed uncertainty.
Report each measured value and absolute difference, with no invented success threshold.
A value printed to two decimals has rounding resolution 0.01; that does not establish stochastic reproducibility.
Reconstruction Table 2 uses Base at 120 epochs, unlike released Base at 240 epochs.
Do not claim a Table 2 replication with a different training checkpoint.

## Locked release path

- Official source: 2389f2c65fedbd43ada851e69b2b810efd23af7d.
- Full resolved checkpoint SHA256 and tensor/configuration identity are required before GPU execution.
- Architecture from source/checkpoint: Base or Large encoder, Base decoder, 256 latent registers of dimension 32, 256x256 RGB, patch size 16, 1000 class labels.
- Both released checkpoints trained with 14 flow iterations per reconstruction step. This is training identity, not 14 inference steps.
- Source generation: UNITE.diffusion_generate, Sampler.sample_ode, torchdiffeq dopri5, atol=1e-6, rtol=1e-3; 50 requested output time points do not mean 50 adaptive solver evaluations.
- Noise schedule shift alpha=0.5, source training configuration.
- Classifier-free guidance (CFG): paper sweep 1.0 through 4.0 inclusive in increments of 0.2, 16 scales. Source normalized-before-guidance mode is norm_first, guidance interval [0.1, 1.0].
- Exactly 50,000 images per final generation evaluation: 50 per each of 1000 classes, labels tiled 0 through 999 as in engines/eval.py.
- ImageNet label map: read-only shared imagenet_class_index.json must agree with lexicographically sorted validation synsets; hash required.
- Primary FID feature source: released utils/fid.py, torch-fidelity InceptionV3, published weights-inception-2015-12-05-6726825d.pth and jit_in256_stats.npz; hashes required.
- Quantization: clamp generated [0,1] float values; multiply by 255 and cast to uint8, exactly as released extract_inception_features.
- FID: release uses FP64 feature moments/covariance and SciPy sqrtm against mu/sigma reference statistics. Do not replace it with clean-fid, a new resize path, another statistics set or sample count.
- Source IS: unbiased Inception logits, first 1000 columns, softmax in FP32, ten contiguous splits. This source detail must be disclosed beside the paper's torch-fidelity description; do not silently change it.
- Generation requires no image split loader; reference statistics represent ImageNet training images, stated by the paper as 1,281,167. Shared dataset documentation instead says 1,281,168.
- Reconstruction uses all 50,000 validation images and the released ADM/JiT center crop: repeated BOX downscale, shortest edge to 256 with BICUBIC, center crop, RGB ToTensor [0,1]. No horizontal flip at evaluation.
- Source training seed is zero; paper evaluation RNG state/seeds are not released. Use recorded explicit zero for independent replay and disclose that the authors' exact random draws cannot be recovered.
- Source evaluation uses FP32 weights outside the training BF16 autocast context, with the training entry's TF32 settings. Check the compute node before execution. Batch-size, world-size and seeds must be recorded because they affect adaptive integration and random draws.

## Unresolved before final execution

Paper Appendix B and Table 8 state ODE interval [0.1,1.0], while released Transport.check_interval(sde=False) returns [0,1].
Source CFG interval [0.1,1.0] is not the same setting.
Do not describe a source-path adapter as a literal paper-interval implementation.
Resolve and record the selected protocol and any bounded alternate before expensive evaluation.

The README promises standalone generation/reconstruction evaluation later.
Prefer a small entrypoint that loads the exact released checkpoint and calls the existing training evaluation functions.
Verify any adapter against the training path with shared fixed inputs before paper-scale execution.

## Storage and ownership

The sole submitter, terminal-result owner and canonical notebook writer is unite-reproduce-20261023.
Keep code and unique artifacts under /data/vision/torralba/u/kwen1/image-tokenization-20261023.
Use UNITE-specific scratch paths for the environment and performance stages.
Read shared ImageNet without changes.
Preserve failed attempts, every complete stage and every run record.
Every experimental command, including smoke and scoring, requires a fresh full timestamped stage with verified files, executable bits, symlinks and Git identity.

- Source trace confirms the evaluation hooks at engines/train.py:125-134 run outside the batch autocast context at line 63. Preserve FP32 evaluation with the training entry's TF32 settings.
- Source main_train.py:170-176 compiles the decoder in default mode for grad_accum_steps=2 before copying EMA. Restore compilation after loading normalized checkpoint keys; do not silently switch final evaluation to eager execution.
- At CFG scale 1.0, released diffusion_generate takes a separate branch and only applies encoder output normalization for cfg_first/both, unlike guided norm_first. Preserve and disclose that source behavior until a bounded correction is justified; the other fifteen requested scales use the guided branch.
