"""Check source rank splitting and NCCL gathering with synthetic data, no paper metric."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--labels-only", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
proof = {}
for size in (1, 3, 4):
    indices = np.array_split(np.arange(50000), size)
    gathered = np.concatenate(indices)
    labels = np.tile(np.arange(1000), 50)[gathered]
    assert np.array_equal(gathered, np.arange(50000))
    assert np.array_equal(np.bincount(labels, minlength=1000), np.full(1000, 50))
    proof[str(size)] = {"counts": [len(x) for x in indices], "total": len(gathered),
                        "class_count_min": 50, "class_count_max": 50}
if args.labels_only:
    (args.output / "label-count-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from utils.distributed import setup_distributed, cleanup_distributed
from engines import eval as evaluation

assert (Path.cwd() / "STAGING_MANIFEST.json").is_file()
assert os.getenv("SLURM_JOB_ID")
rank, size, device = setup_distributed()
start = time.perf_counter()
record = {"run_id": args.output.name, "node_id": "DISTRIBUTED_SYNTHETIC_PREFLIGHT",
          "status": "PLANNED", "job_id": os.environ["SLURM_JOB_ID"],
          "rank": rank, "world_size": size, "stage": str(Path.cwd()),
          "paper_metric": False, "checkpoint_used": False, "label_50k_proof": proof}
try:
    assert torch.cuda.current_device() == int(os.environ["LOCAL_RANK"])
    assert torch.cuda.device_count() == size
    identity = {"rank": rank, "device": device.index,
                "uuid": str(torch.cuda.get_device_properties(device).uuid)}
    identities = [None] * size
    dist.all_gather_object(identities, identity)
    assert len({x["uuid"] for x in identities}) == size, "Ranks share a GPU"
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value)
    assert value.item() == size * (size + 1) / 2
    record["rank_gpu_bindings"] = identities
    count = 2000
    local_indices = np.array_split(np.arange(count), size)[rank]
    expected_labels = np.tile(np.arange(1000), 2)[local_indices]
    offset = [0]
    class SyntheticModel:
        def diffusion_generate(self, device, num_visuals, y_given, **kwargs):
            begin = offset[0]
            end = begin + num_visuals
            assert np.array_equal(y_given.cpu().numpy(), expected_labels[begin:end])
            index = torch.tensor(local_indices[begin:end], dtype=torch.float32, device=device)
            offset[0] = end
            return (index / count).reshape(-1, 1, 1, 1)
    def synthetic_features(images, inception, batch_size):
        index = (images[:, 0, 0, 0].cpu() * count).round()
        features = torch.zeros(len(index), 2048)
        logits = torch.zeros(len(index), 1008)
        features[:, 0] = index
        logits[:, 0] = index
        return features, logits
    assertions = []
    def check_features(features, ref):
        assert features.shape == (count, 2048)
        assert torch.equal(features[:, 0], torch.arange(count, dtype=torch.float32))
        assertions.append("features_exact_valid_rows")
        return 0.0
    def check_logits(logits):
        assert logits.shape == (count, 1008)
        assert torch.equal(logits[:, 0], torch.arange(count, dtype=torch.float32))
        assertions.append("logits_exact_valid_rows")
        return 0.0, 0.0
    evaluation.get_inception_model = lambda device: None
    evaluation.extract_inception_features = synthetic_features
    evaluation.compute_fid_from_features = check_features
    evaluation.compute_inception_score = check_logits
    evaluation.evaluate_fid(SyntheticModel(), device, global_step=0, num_images=count,
                           num_classes=1000, batch_size=256, cfg_scales=[2.0],
                           log_to_wandb=False, log_fid_best=False)
    assert offset[0] == len(local_indices)
    if rank == 0:
        assert assertions == ["features_exact_valid_rows", "logits_exact_valid_rows"]
    record.update(status="PASS", synthetic_total=count,
                  synthetic_rank_counts=[len(x) for x in np.array_split(np.arange(count), size)],
                  local_count=len(local_indices), source_engine_gather_assertions=assertions,
                  interpretation="Structural test only. Zero placeholder scores are not scientific metrics.")
except Exception as exc:
    record.update(status="BLOCKED_ENVIRONMENT", error=repr(exc))
    raise
finally:
    record["elapsed_seconds"] = time.perf_counter() - start
    target = args.output / f"rank-{rank:04d}"
    target.mkdir(exist_ok=True)
    payload = json.dumps(record, indent=2) + "\n"
    (target / "results.json").write_text(payload)
    report = f"""# {record['run_id']}: {record['node_id']}

## Objective
Verify actual rank-to-GPU binding, NCCL collectives and the original source evaluator's unequal-count gathering.

## Setup
Job {record['job_id']}; rank {rank}/{size}; full stage {record['stage']}.
Synthetic data only, no checkpoint, no model inference and no scientific metric.
The source engines.eval.evaluate_fid performs its real padding, gathering and valid-row slicing.

## Baseline
For 50,000 images and three ranks, source numpy.array_split gives [16667,16667,16666].
The native engine gathers counts, pads to the maximum, then removes padding using each rank's actual count.
The synthetic 2,000-row case gives [667,667,666], exercising unequal counts through that same source engine.

## Results
Status {record['status']}; elapsed {record['elapsed_seconds']} seconds; error {record.get('error')}.
Full 50K label proof: {proof}.
Actual rank GPU identities and source-gather assertions are in results.json.

## Interpretation
PASS proves structural checks for this allocation. Placeholder zero scores are not FID or IS results.

## Limitations
This check does not measure model quality, throughput or paper fidelity.

## Next action
Start the frozen full 50K checkpoint evaluation only if every rank passes.
"""
    (target / "report.md").write_text(report)
    if rank == 0:
        (args.output / "results.json").write_text(payload)
        (args.output / "report.md").write_text(report)
    cleanup_distributed()
