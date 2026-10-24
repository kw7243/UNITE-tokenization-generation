# Bounded GPU namespace counterfactual

The first error in job 2834097 is `transport/shm.cc:590`, CUDA 101, during shared-memory handle import.
NCCL 2.27.5 initialization completed on all ranks before this error.
Later peer-proxy and socket failures follow it.
The pinned [NCCL source](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/transport/shm.cc#L590) identifies the failing call as `cuMemImportFromShareableHandle`.
The software is Torch 2.10.0+cu128, Torch CUDA 12.8, NCCL 2.27.5 built for CUDA 12.9, driver 580.178.04.

The trigger is the first CUDA collective and its shared-memory transport setup.
The proposed masking condition is the separate one-device namespace created for each Slurm task.
Each failed process used local CUDA0 and saw one GPU; physical UUIDs were distinct.
No script changed CUDA_VISIBLE_DEVICES after CUDA initialization.
Explicit device_id did not fix the failure, so device inference alone is insufficient.
The source single-GPU path passes but does not test shared-memory communication.
No verified earlier CSAIL four-GPU result was found in the assigned project records.
The source train.sh and preserved reference launcher use torchrun, but their existence does not prove a successful four-GPU run.

The next counterfactual uses one Slurm task per node, both allocated GPUs visible to that task, and torchrun with two workers per node.
Torchrun assigns LOCAL_RANK 0/1 within that common namespace.
The minimal probe first allocates a CUDA tensor, then checks an all-reduced sum of 10 across ranks 0..3, then executes a barrier.
It records launch masks, rank values, process-visible UUIDs and unchanged visibility before/after CUDA initialization.
Only a passing probe permits direct denoiser-output and full four-rank model checks in the same allocation.
If it still fails, preserve the first transport error and return the contrary evidence before any further transport-specific change.
No NCCL transport is disabled and no library or driver is changed.

The launch follows [PyTorch 2.10 torchrun](https://docs.pytorch.org/docs/2.10/elastic/run.html).
[Slurm GRES documentation](https://slurm.schedmd.com/gres.html#GPU_Management) explains that GPU visibility is set per step and can be renumbered under cgroups.
[Slurm srun documentation](https://slurm.schedmd.com/srun.html) defines `--gpu-bind=none` and per-step GRES allocation.
These contracts support this counterfactual; they do not establish its result.

The resource request remains two nodes and four homogeneous RTX 3090 GPUs.
The request is one launcher task with eight CPUs per node: 16 CPUs total, eight per node.
Each launcher runs two workers with four CPU threads each.
Host memory is 32 GiB per node, 64 GiB total.
The starting request is 20 minutes, an unmeasured maximum of 1.3333 RTX 3090 device-hours.
It covers the minimal probe and required follow-on checks, not scientific training.
