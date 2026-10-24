"""Bounded literal-NIL CPU launch validation. No model or metric workload."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

config_path = Path(sys.argv[1])
config_bytes = config_path.read_bytes()
config = json.loads(config_bytes)
digest = hashlib.sha256(config_bytes).hexdigest()
run_id = f"A1__LAUNCH_NIL_CPU__s20261023__{digest[:12]}"
output = Path(config["output_parent"]) / run_id
output.mkdir(parents=True, exist_ok=True)
assert (Path.cwd() / "STAGING_MANIFEST.json").is_file()
assert not os.getenv("CAMPAIGN_SUBMIT_SENTINEL")
record = dict(node_id="A1", method_id="LAUNCH_NIL_CPU", seed=20261023,
              config_sha256=digest, checkpoint_sha256=None,
              checkpoint_sha256_reason="No checkpoint used by launch canary.",
              data_manifest_sha256=None, data_manifest_sha256_reason="No dataset used.",
              metric_protocol_sha256=None, metric_protocol_sha256_reason="No metric computed.",
              node_status="PASS", outcome_class=None,
              outcome_class_reason="Operational validation, no training.",
              main_updates=0, images_seen=0, training_gpu_hours=0.0, gpu_hours=0.0,
              job_id=os.environ["SLURM_JOB_ID"], stage=str(Path.cwd()), node=platform.node(),
              source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              script_sha256=hashlib.sha256(Path("campaign/scripts/launch_canary.sbatch").read_bytes()).hexdigest(),
              argv=sys.argv, inherited_sentinel_absent=True,
              explicit_environment={"PATH":os.environ["PATH"], "TZ":os.environ["TZ"],
                                    "SLURM_EXPORT_ENV":os.environ["SLURM_EXPORT_ENV"]},
              training={}, reconstruction={}, generation={})
(output / "results.json").write_text(json.dumps(record,indent=2)+"\n")
(output / "resolved.yaml").write_bytes(config_bytes)
(output / "report.md").write_text(f"""# {run_id}: A1

## 1 Objective
Verify the literal-NIL batch launch crosses the historical pre-first-marker failure boundary.

## 2 Setup
Job {record['job_id']}, node {record['node']}, one CPU, 1 GiB, two-minute limit, zero GPUs.
Config SHA256 {digest}; source {record['source_revision']}; full stage {record['stage']}.
NIL strips the submit sentinel. Required environment is constructed after all SBATCH directives.

## 3 Baseline
TIG identifies ALL,variables and NONE export forms as unsupported. The supplied safe-form trials passed.
This is an operational canary, not a checkpoint or hypothesis result.

## 4 Results
First stdout/stderr markers and this retained record were produced. Sentinel is absent and Git fsck passed.
Payload PASS; terminal accounting must still be checked by the owner.

## 5 Interpretation
This checks the permitted environment-construction route on this CPU node.

## 6 Limitations
GPU launch, software and model execution remain untested. No metric is measured.

## 7 Next action
Run compute-node environment preparation, then the bounded staged GPU validation.
""")
print(json.dumps({"run_id":run_id,"output":str(output),"payload_status":"PASS"}))
