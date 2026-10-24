"""Create the isolated A1 environment and lock installed wheels on a CPU node."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

cfg_path = Path(sys.argv[1])
raw = cfg_path.read_bytes()
cfg = json.loads(raw)
digest = hashlib.sha256(raw).hexdigest()
run_id = f"A1__{cfg['method_id']}__s20261023__{digest[:12]}"
out = Path(cfg["output_parent"]) / run_id
out.mkdir(parents=True,exist_ok=True)
(out/"resolved.yaml").write_bytes(raw)
env_path = Path(cfg["environment"])
record={"node_id":"A1","method_id":cfg["method_id"],"seed":20261023,"config_sha256":digest,
        "checkpoint_sha256":cfg["checkpoint_sha256"],"data_manifest_sha256":None,
        "data_manifest_sha256_reason":"Manifest preparation has not completed.",
        "metric_protocol_sha256":None,"metric_protocol_sha256_reason":"Metric dependency preparation has not completed.",
        "node_status":"IN_PROGRESS","run_status":"STARTING","outcome_class":None,"outcome_class_reason":"Operational CPU setup, no training.",
        "main_updates":0,"images_seen":0,"training_gpu_hours":0.0,"gpu_hours":0.0,
        "job_id":os.environ["SLURM_JOB_ID"],"node":platform.node(),"stage":str(Path.cwd()),
        "environment":str(env_path),"training":{},"reconstruction":{},"generation":{}}
start=time.perf_counter()
try:
    assert (Path.cwd()/"STAGING_MANIFEST.json").is_file()
    assert not env_path.exists(), "Do not overwrite an existing environment"
    subprocess.run([cfg["bootstrap_python"],"-m","venv",str(env_path)],check=True)
    python=str(env_path/"bin/python")
    subprocess.run([python,"-m","pip","install","pip=="+cfg["bootstrap_pip_version"],
                    "--index-url","https://pypi.org/simple",
                    "--report",str(out/"pip-bootstrap-wheel.json")],check=True)
    pip_report=json.loads((out/"pip-bootstrap-wheel.json").read_text())
    assert pip_report["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"]==cfg["bootstrap_pip_wheel_sha256"]
    subprocess.run([python,"-m","pip","install","torch==2.10.0+cu128","torchvision==0.25.0+cu128",
                    "--index-url","https://pypi.org/simple",
                    "--extra-index-url","https://download.pytorch.org/whl/cu128",
                    "--report",str(out/"torch-wheels.json")],check=True)
    subprocess.run([python,"-m","pip","install","-r","campaign/requirements.txt",
                    "--report",str(out/"remaining-wheels.json")],check=True)
    subprocess.run([python,"-m","pip","check"],check=True)
    freeze=subprocess.check_output([python,"-m","pip","freeze"],text=True)
    (out/"packages.txt").write_text(freeze)
    (out/"pip-inspect.json").write_bytes(subprocess.check_output([python,"-m","pip","inspect"]))
    record["package_lock_sha256"]=hashlib.sha256(freeze.encode()).hexdigest()
    subprocess.run([python,"campaign/scripts/verify_artifacts.py",str(out)],check=True)
    artifacts=json.loads((out/"artifacts.json").read_text())
    record["data_manifest_sha256"]=artifacts["reconstruction_manifest_sha256"]
    record["metric_protocol_sha256"]=artifacts["metric_protocol_sha256"]
    del record["data_manifest_sha256_reason"],record["metric_protocol_sha256_reason"]
    record["run_status"]="COMPLETED"
except Exception as exc:
    record.update(node_status="BLOCKED_REFERENCE",run_status="FAILED_OPERATIONAL",scientific_negative=False,error=repr(exc))
    raise
finally:
    record["elapsed_seconds"]=time.perf_counter()-start
    (out/"results.json").write_text(json.dumps(record,indent=2)+"\n")
    (out/"report.md").write_text(f"""# {run_id}: A1

## 1 Objective
Validate and freeze the proposed environment and reference inputs before GPU use.

## 2 Setup
Job {record['job_id']}, node {record['node']}, stage {record['stage']}.
Config SHA256 {digest}. Environment {env_path}. Literal NIL with explicit environment after all directives.
Exact wheel URLs/hashes are in torch-wheels.json and remaining-wheels.json.

## 3 Baseline
Plan pins include Torch 2.10 CUDA12.8, torchvision0.25.0 and torch-fidelity0.3.0.
Legacy reference input bytes can be reused only after current hash/config checks.

## 4 Results
Operational payload status {record['run_status']}; elapsed {record['elapsed_seconds']} seconds; error {record.get('error')}.
CPU only, zero GPU-hours. Installed package lock SHA256 {record.get('package_lock_sha256')}.
Artifact/manifest validation details are separate retained files in this directory.

## 5 Interpretation
A completed operational payload establishes this CPU environment and artifact checks; the A1 node remains open. Actual GPU identity remains untested.

## 6 Limitations
No scientific score or adapter correctness claim follows from setup. Terminal accounting remains to be recorded by the owner.

## 7 Next action
Run the smallest useful plan-compatible GPU validation after setup passes.
""")
