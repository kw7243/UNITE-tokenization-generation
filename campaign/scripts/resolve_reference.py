"""Resolve A1 scientific and allocation inputs only after the CPU lock passes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PLAN="4cac088f79f6ad4c848e9fc91b56590711c0fb4596f184f3babe81f5dfee40c5"
PLAN_VERSION="1.1"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-run",type=Path,required=True)
    parser.add_argument("--workload",choices=["VALIDATION_AND_ESSENTIAL_CHECKS","FULL_REFERENCE"],required=True)
    parser.add_argument("--gpu",choices=["H100","H200","A10080","RTX4090"],required=True)
    parser.add_argument("--account",required=True)
    parser.add_argument("--partition",required=True)
    parser.add_argument("--qos",required=True)
    parser.add_argument("--gres",required=True)
    parser.add_argument("--time",required=True)
    parser.add_argument("--cpu-count",type=int,default=4)
    parser.add_argument("--memory-gib",type=int,default=32)
    parser.add_argument("--karl-evidence",type=Path,required=True)
    parser.add_argument("--extension-record",type=Path)
    args=parser.parse_args()
    envrun=args.environment_run.resolve()
    setup=json.loads((envrun/"results.json").read_text())
    assert setup["node_status"] in {"PASS","IN_PROGRESS","AWAITING_COORDINATOR_REVIEW"}
    assert setup.get("run_status","PASS") in {"PASS","COMPLETED"}
    assert "error" not in setup
    assert setup.get("terminal_accounting",{}).get("State")=="COMPLETED","Record terminal accounting before resolving GPU inputs"
    artifacts=json.loads((envrun/"artifacts.json").read_text())
    checkpoint=artifacts["files"]["UNITE-B.pt"]
    state=artifacts["strict_checkpoint_load"]
    assert state["source_training_M"]==14 and state["num_latent_tokens"]==256
    assert state["epoch"]==240 and state["saved_cfg"]==2.0
    assert not state["missing_keys"] and not state["unexpected_keys"]
    assert artifacts["runtime"]["packages"]["torch-fidelity"]=="0.3.0"
    assert sha(envrun/"packages.txt")==setup["package_lock_sha256"]
    assert sha(envrun/"reconstruction-manifest.json")==artifacts["reconstruction_manifest_sha256"]
    assert sha(envrun/"metric-protocol.json")==artifacts["metric_protocol_sha256"]
    karl=json.loads(args.karl_evidence.read_text())
    assert karl["coordinator_decision"]["coordinator_decision"]=="PASS"
    assert karl["coordinator_decision"]["plan_sha256"] in {PLAN,"71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b"}
    assert karl["revision_compatibility"]["current_plan_sha256"]==PLAN
    assert karl["revision_compatibility"]["accepted"]
    code_paths=["campaign/adapters/prefix.py","campaign/scripts/check_adapter.py",
                "campaign/scripts/evaluate_a1.py","campaign/scripts/reference.sbatch",
                "campaign/scripts/resolve_reference.py"]
    code_hashes={name:sha(ROOT/name) for name in code_paths}
    patch_sha=hashlib.sha256(json.dumps(code_hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    imports=json.loads((ROOT/"campaign/locks/source-imports.json").read_text())
    hardware={
        "H100":("H100",80_000_000_000),
        "H200":("H200",140_000_000_000),
        "A10080":("A100",80_000_000_000),
        "RTX4090":("RTX 4090",24_000_000_000),
    }[args.gpu]
    resolved_gen_tok=dict(state["gen_tok"])
    resolved_gen_tok.update(cfg_norm_order="norm_first",gradient_checkpointing_denoiser=True)
    lock={"plan_sha256":PLAN,"plan_version":PLAN_VERSION,"source_imports":imports,
          "source_import_manifest_sha256":sha(ROOT/"campaign/locks/source-imports.json"),
          "reference_inputs":artifacts,
          "cpu_setup":{"job_id":setup["job_id"],"stage":setup["stage"],
                       "execution_plan_sha256":"71728da427a6946146d12cc4bff4eddfbf308e6f79d15a86c39b993d64196f5b",
                       "compatible_current_plan_sha256":PLAN,
                       "compatibility_basis":"Section5.3 environment,6 reference protocol and5.11 checks unchanged; retain submitted stage/config identity.",
                       "results_path":str(envrun/"results.json"),"results_sha256":sha(envrun/"results.json")},
          "environment":{"path":setup["environment"],"packages":str(envrun/"packages.txt"),
                         "packages_sha256":sha(envrun/"packages.txt"),"package_versions":artifacts["runtime"]["packages"],
                         "pip_bootstrap_report_sha256":sha(envrun/"pip-bootstrap-wheel.json"),
                         "torch_wheel_report_sha256":sha(envrun/"torch-wheels.json"),
                         "remaining_wheel_report_sha256":sha(envrun/"remaining-wheels.json"),
                         "pip_inspect_sha256":sha(envrun/"pip-inspect.json")},
          "karl_accepted_read_only":karl,
          "adapter_patch_sha256":patch_sha,"code_sha256":code_hashes,
          "private_plan":"Only the verified digest is committed; the supplied plan remains in retained administrative outputs."}
    pilot_path=ROOT/"campaign/locks/pilot-data.json"
    if pilot_path.exists():
        lock["pilot_data_accepted"]={"path":str(pilot_path),"sha256":sha(pilot_path),
                                      "evidence":json.loads(pilot_path.read_text()),
                                      "A1_use":"Provenance only; pilot training rows are not consumed in A1."}
    lockpath=ROOT/"campaign/locks/artifacts.json"
    if lockpath.exists():
        previous=json.loads(lockpath.read_text())
        assert previous["reference_inputs"]==lock["reference_inputs"],"Do not change existing scientific input identities"
        assert previous["environment"]==lock["environment"],"Do not change the locked environment"
    lockpath.write_text(json.dumps(lock,indent=2)+"\n")
    cfg={
        "node_id":"A1",
        "method_id":"UNITE_B_VALIDATE_ESSENTIAL" if args.workload=="VALIDATION_AND_ESSENTIAL_CHECKS" else "UNITE_B_REFERENCE",
        "seed":20261023,"plan_sha256":PLAN,"plan_version":PLAN_VERSION,"workload":args.workload,
        "output_parent":str(ROOT/"outputs"),
        "source":{"official_commit":imports["unite_commit"],"third_party_untouched":True,
                  "source_import_manifest_sha256":lock["source_import_manifest_sha256"],
                  "patch_sha256":patch_sha,"code_sha256":code_hashes},
        "checkpoint":checkpoint,
        "model":{"num_latent_tokens":256,"num_classes":1000,"state":"EMA","training_M":14,
                 "gen_tok":resolved_gen_tok,
                 "transport":{"train_eps":.05,"sample_eps":.05,"use_cosine_loss":True,
                              "use_lognorm":True,"partial_ratio":1.0,"partitial_train":None,"shift_lg":False},
                 "encoding_class_dropout":1.0,"denoising_class_dropout":.1,
                 "latent_layer_norm":{"eps":1e-5,"elementwise_affine":True},
                 "decoder_compile_mode":"default"},
        "sampling":{"cfg":2.0,"cfg_interval":[.1,1.0],"cfg_norm_order":"norm_first",
                    "batch_size":16,"num_samples":50000,"num_classes":1000,"ode_method":"dopri5",
                    "output_grid_size":50,"time_interval":[0.0,1.0],"atol":1e-6,"rtol":1e-3,
                    "timestep_shift":.5,"precision":"FP32_TF32","compile_decoder":True},
        "reconstruction":{"sample_count":5000,"batch_size":64,"precision":"FP32_TF32",
                          "no_flip":True,"clean_latents":True,"output_clamp":[0.0,1.0],
                          "rfid_reference":"Exactly the same named5000 real validation images"},
        "data":{"reconstruction_manifest":str(envrun/"reconstruction-manifest.json"),
                "reconstruction_manifest_sha256":artifacts["reconstruction_manifest_sha256"],
                "generation_label_manifest":str(envrun/"generation-labels.json"),
                "generation_label_manifest_sha256":artifacts["generation_label_manifest_sha256"],
                "class_map":artifacts["class_map"]},
        "metrics":{"protocol_path":str(envrun/"metric-protocol.json"),
                   "protocol_sha256":artifacts["metric_protocol_sha256"],
                   "torch_fidelity":"0.3.0","external_lpips":"AlexNet v0.1 on2*x-1",
                   "inception":artifacts["files"]["weights-inception-2015-12-05-6726825d.pth"],
                   "statistics":artifacts["files"]["jit_in256_stats.npz"]},
        "environment":lock["environment"],
        "hardware":{"selection":args.gpu,"gpu_model_substring":hardware[0],
                    "minimum_vram_bytes":hardware[1],"gpu_count":1},
        "resources":{"account":args.account,"partition":args.partition,"qos":args.qos,"gres":args.gres,
                     "cpus":args.cpu_count,"memory_gib":args.memory_gib,"time_limit":args.time,
                     "export":"NIL","submission_mode":"attached sbatch --wait"},
        "essential_fixture":{"only_validation_workload":True,"candidate_registers":64,"counts":[4,16,64],
                             "training_M":3,"flow_chunk":3,"num_classes":1000,"precision":"FP32",
                             "tf32":False,"compile":False,"tolerance":1e-5,"optimizer_updates":0,
                             "note":"BaseEMA cropped positional fixture. It is not a trained pilot model."},
        "runtime_mount_paths":[str(ROOT),str(envrun),setup["environment"],
                               "/data/vision/torralba/datasets/imagenet_pytorch_new/val",checkpoint["path"]],
        "compute_guidance":{"a1_initial_h100_hours":24.0,"advisory_only":True,
                            "hard_cap":None,"fixed_reserve":None,
                            "estimate_exceedance":"Does not change status or trigger a stop/approval.",
                            "extension_protocol":"Section15.2 evidence, revised config/endpoints, affected controls, forecasts and actual costs.",
                            "actual_resource_constraints":[],
                            "device_accounting":"Actual reserved hours by device; measured throughput forecasts, no theoretical FLOP conversion."},
        "extensions":[],
        "gate":{"name":"CHECKPOINT_EFFECT_REPRODUCTION","gfid_50k_maximum":3.1,
                "strict_load":True,"required_reconstruction":True,"coordinator_acceptance_required":True},
        "training":{"main_updates":0,"optimizer":None,"learning_rate":None},
    }
    if args.extension_record:
        extension=json.loads(args.extension_record.read_text())
        assert extension["extension_reason"] and len(extension["original_config_sha256"])==64
        extension["revised_config_sha256"]=None
        extension["revised_config_sha256_reason"]="Runtime/result envelope fills this config's own SHA256 after serialization."
        cfg["extensions"]=[extension]
    name="validation" if args.workload=="VALIDATION_AND_ESSENTIAL_CHECKS" else "reference"
    payload=(json.dumps(cfg,indent=2)+"\n").encode()
    digest=hashlib.sha256(payload).hexdigest()
    dest=ROOT/"campaign/configs"/f"{name}-{digest[:12]}.yaml"
    assert not dest.exists(),"Resolved config already exists; do not replace or resubmit it"
    dest.write_bytes(payload)
    if cfg["extensions"]:
        extension_output=ROOT/"outputs"/f"A1__{cfg['method_id']}__s20261023__{digest[:12]}"
        extension_output.mkdir(parents=True,exist_ok=True)
        complete=[]
        for item in cfg["extensions"]:
            item=dict(item);item["revised_config_sha256"]=digest;item.pop("revised_config_sha256_reason",None);complete.append(item)
        (extension_output/"extension-plan.json").write_text(json.dumps(complete,indent=2)+"\n")
    print(json.dumps({"config":str(dest),"config_sha256":digest,
                      "run_id":f"A1__{cfg['method_id']}__s20261023__{digest[:12]}",
                      "output":str(ROOT/"outputs"/f"A1__{cfg['method_id']}__s20261023__{digest[:12]}"),
                      "artifact_lock_sha256":sha(lockpath)},indent=2))
if __name__=="__main__":main()
