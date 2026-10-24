"""A1 native Base reference and the bounded essential adapter checks."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/"third_party/unite"))
from models.unite import UNITE
from utils.crop import CenterCropTransform
from utils.fid import get_inception_model, extract_inception_features, compute_fid_from_features
from engines import eval as evaluation
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
sys.path.insert(0,str(root))
from campaign.scripts.check_adapter import check_essential

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config",type=Path,required=True)
args=parser.parse_args()
raw=args.config.read_bytes()
cfg=json.loads(raw)  # Resolved files use JSON, a YAML1.2 subset; retain numeric exponent types.
digest=hashlib.sha256(raw).hexdigest()
run_id=f"A1__{cfg['method_id']}__s{cfg['seed']}__{digest[:12]}"
out=Path(cfg["output_parent"])/run_id
out.mkdir(parents=True,exist_ok=True)
(out/"resolved.yaml").write_bytes(raw)
assert (Path.cwd()/"STAGING_MANIFEST.json").is_file()
assert os.environ.get("SLURM_JOB_ID")
start=time.perf_counter()
phase="PREFLIGHT"
record={"node_id":"A1","method_id":cfg["method_id"],"seed":cfg["seed"],
        "config_sha256":digest,"checkpoint_sha256":cfg["checkpoint"]["sha256"],
        "data_manifest_sha256":cfg["data"]["reconstruction_manifest_sha256"],
        "metric_protocol_sha256":cfg["metrics"]["protocol_sha256"],
        "main_updates":0,"images_seen":0,"training_gpu_hours":0.0,
        "images_seen_definition":"Reconstructed real images plus generated samples; correctness fixture counts real encoding inputs and generated samples separately.",
        "gpu_hours":None,"gpu_hours_reason":"Awaiting authoritative terminal allocation accounting.",
        "outcome_class":None,"outcome_class_reason":"Checkpoint evaluation/correctness checks, no training.",
        "node_status":"IN_PROGRESS","run_status":"STARTING","job_id":os.environ["SLURM_JOB_ID"],
        "stage":str(Path.cwd()),"node":platform.node(),
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "patch_sha256":cfg["source"]["patch_sha256"],
        "training":{},"reconstruction":{},"generation":{},
        "plan_sha256":cfg["plan_sha256"],"plan_version":cfg["plan_version"],"extensions":cfg["extensions"],"paper_headline_gfid":2.12,"paper_appendix_gfid":2.735,"paper_headline_is":294.1,"paper_appendix_is":268.1}

for extension in record["extensions"]:
    extension["revised_config_sha256"]=digest
    extension.pop("revised_config_sha256_reason",None)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(8<<20),b""):h.update(part)
    return h.hexdigest()

def seed_all():
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])

def save():
    (out/"results.json").write_text(json.dumps(record,indent=2)+"\n")
    sections=[
      ("1 Objective","A1 UNITE-B positive-reference effect and/or the three essential checks. Practical gate: gFID-50K<=3.1, strict load and valid reconstruction. Coordinator owns acceptance."),
      ("2 Setup",f"Method {cfg['method_id']}; seed {cfg['seed']}; config SHA256 {digest}; checkpoint SHA256 {record['checkpoint_sha256']}. Stage {record['stage']}; job {record['job_id']}; source {record['source_revision']}. Complete resolved.yaml, runtime.json and input locks are retained."),
      ("3 Baseline","Untouched source BaseEMA,256 registers,source M14 identity and native FP32/TF32 sampling. HeadlineFID2.12/IS294.1 and appendixFID2.735/IS268.1 remain separate. Reference validation smokes and M3 correctness fixtures cannot replace the full reference. Required A1 reference workloads are5000 fixed reconstructions and independent balanced50000 generation."),
      ("4 Results",json.dumps({k:record.get(k) for k in ("run_status","reconstruction","generation","validation","adapter_checks","gpu_hours","elapsed_seconds","error")},indent=2)),
      ("5 Interpretation","Run completion does not claim coordinator acceptance. Reference effect is distinct from exact headline reproduction; adapter correctness does not establish an allocation hypothesis."),
      ("6 Limitations","Authors' evaluation RNG states and full scientific config are absent. Paper/source ODE intervals differ; this uses native[0,1]. Reconstruction is the declared5K subset, with no matching published rFID target. Single-run metrics do not establish training-seed uncertainty. All failures and allocated hours remain in the ledger."),
      ("7 Next action",("Inspect all three correctness proofs and measured throughput, then prepare the complete5000/50000 reference request."
                         if cfg["workload"]=="VALIDATION_AND_ESSENTIAL_CHECKS" else
                         "Inspect terminal accounting and give complete A1 evidence and essential checks to the coordinator. Do not start A2."))]
    (out/"report.md").write_text("# "+run_id+": A1\n\n"+"\n\n".join("## "+name+"\n"+text for name,text in sections)+"\n")
save()

def save_grid(name,images):
    grid=make_grid(images.detach().cpu(),nrow=4,padding=2)
    pixels=(grid*255).clamp(0,255).to(torch.uint8).permute(1,2,0).numpy()
    Image.fromarray(pixels).save(out/name)

def image_batch(rows,device):
    transform=CenterCropTransform(256)
    convert=ToTensor()
    images=[]
    for row in rows:
        payload=Path(row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==row["sha256"],row["image_id"]
        with Image.open(io.BytesIO(payload)) as image:
            images.append(convert(transform(image.convert("RGB"))))
    return torch.stack(images).to(device)

try:
    for relative,expected in cfg["source"]["code_sha256"].items():
        assert sha(root/relative)==expected,relative
    imports_path=root/"campaign/locks/source-imports.json"
    assert sha(imports_path)==cfg["source"]["source_import_manifest_sha256"]
    imports=json.loads(imports_path.read_text())
    for project,files in imports["files"].items():
        for name,expected in files.items():
            assert sha(root/"third_party"/project/name)==expected,(project,name)
    assert sha(cfg["checkpoint"]["path"])==cfg["checkpoint"]["sha256"]
    assert sha(cfg["data"]["generation_label_manifest"])==cfg["data"]["generation_label_manifest_sha256"]
    assert sha(cfg["data"]["reconstruction_manifest"])==cfg["data"]["reconstruction_manifest_sha256"]
    assert sha(cfg["metrics"]["protocol_path"])==cfg["metrics"]["protocol_sha256"]
    assert sha(cfg["environment"]["packages"])==cfg["environment"]["packages_sha256"]
    assert cfg["seed"]==20261023
    assert cfg["sampling"]==dict(cfg=2.0,cfg_interval=[.1,1.0],cfg_norm_order="norm_first",
                                batch_size=16,num_samples=50000,num_classes=1000,
                                ode_method="dopri5",output_grid_size=50,time_interval=[0.0,1.0],
                                atol=1e-6,rtol=1e-3,timestep_shift=.5,precision="FP32_TF32",
                                compile_decoder=True)
    assert cfg["reconstruction"]["sample_count"]==5000
    assert cfg["reconstruction"]["batch_size"]==64
    for name,expected in cfg["environment"]["package_versions"].items():
        assert importlib.metadata.version(name)==expected,name
    assert importlib.metadata.version("torch-fidelity")=="0.3.0"
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    device=torch.device("cuda",0)
    torch.cuda.set_device(device)
    props=torch.cuda.get_device_properties(device)
    assert cfg["hardware"]["gpu_model_substring"] in props.name,props.name
    assert torch.cuda.get_device_capability(device)[0]>=8
    assert props.total_memory>=cfg["hardware"]["minimum_vram_bytes"],props.total_memory
    test=torch.tensor([1.0],device=device)
    assert (test+test).item()==2.0
    gpu_uuid=str(props.uuid)
    smi_uuid=gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-"+gpu_uuid
    nvidia=subprocess.check_output(["nvidia-smi","--id="+smi_uuid,"--query-gpu=name,uuid,memory.total,driver_version","--format=csv,noheader"],text=True).strip()
    memory_kib=int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
    mounts={path:subprocess.check_output(["df","-PT",path],text=True).splitlines()[-1]
            for path in cfg["runtime_mount_paths"]}
    runtime={"memory_total_kib":memory_kib,"mounts":mounts,
             "slurm_job_id":os.environ["SLURM_JOB_ID"],
             "slurm_cpus_per_task":os.getenv("SLURM_CPUS_PER_TASK"),
             "slurm_mem_per_node":os.getenv("SLURM_MEM_PER_NODE"),
             "python":sys.version,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
             "torch_fidelity":importlib.metadata.version("torch-fidelity"),
             "gpu":props.name,"uuid":gpu_uuid,"vram_bytes":props.total_memory,
             "capability":list(torch.cuda.get_device_capability(device)),
             "compiled_arches":torch.cuda.get_arch_list(),"nvidia_smi":nvidia,
             "visible_devices":os.getenv("CUDA_VISIBLE_DEVICES"),"cpus_allowed":Path("/proc/self/status").read_text().split("Cpus_allowed_list:")[1].splitlines()[0].strip(),
             "gcc":subprocess.check_output(["gcc","--version"],text=True).splitlines()[0],
             "g++":subprocess.check_output(["g++","--version"],text=True).splitlines()[0]}
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2)+"\n")
    record["runtime"]=runtime
    for path in cfg["runtime_mount_paths"]:
        assert Path(path).exists(),path
    seed_all()
    core=UNITE(cfg["model"]["gen_tok"],num_latent_tokens=256,num_classes=1000).eval()
    checkpoint=torch.load(cfg["checkpoint"]["path"],map_location="cpu",weights_only=True,mmap=True)
    assert checkpoint["epoch"]==240
    assert float(checkpoint["current_best_cfg_scale"])==cfg["sampling"]["cfg"]
    state=checkpoint["ema"]
    renamed={k.replace("._orig_mod.","."):v for k,v in state.items()}
    assert len(renamed)==len(state)
    key_renames=sum(k!=k.replace("._orig_mod.",".") for k in state)
    assert key_renames==152
    loaded=core.load_state_dict(renamed,strict=True)
    record["strict_load"]={"missing_keys":loaded.missing_keys,"unexpected_keys":loaded.unexpected_keys,
                           "compiled_key_renames":key_renames,"epoch":checkpoint["epoch"],
                           "step":checkpoint["step"],"saved_cfg":checkpoint["current_best_cfg_scale"]}
    del checkpoint,state,renamed
    core=core.to(device)
    record["resolved_ode_output_times"]=core.flow_sampler.sample_ode(num_steps=50,timestep_shift=.5).__self__.t.tolist()
    assert record["resolved_ode_output_times"][0]==0.0 and record["resolved_ode_output_times"][-1]==1.0
    # Consistent legacy-only TF32 setters avoid Torch2.10Inductor's mixed-API error.
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    core.decoder=torch.compile(core.decoder,mode="default")
    rows=json.loads(Path(cfg["data"]["reconstruction_manifest"]).read_text())["rows"]
    assert len(rows)==5000
    inception=get_inception_model(device)
    torch.cuda.reset_peak_memory_stats()
    nfe=[0]
    nfe_hook=core.encoder.blocks[0].register_forward_pre_hook(lambda module,inputs:nfe.__setitem__(0,nfe[0]+1))
    if cfg["workload"]=="VALIDATION_AND_ESSENTIAL_CHECKS":
        seed_all()
        times=[];nfes=[];smoke=[]
        labels=torch.linspace(0,999,16,device=device).long()
        with torch.no_grad():
            for _ in range(2):
                previous=nfe[0];torch.cuda.synchronize();begin=time.perf_counter()
                images=core.diffusion_generate(device=device,num_visuals=16,cfg_scale=2.0,cfg_interval=(.1,1),cfg_norm_order="norm_first",y_given=labels).clamp(0,1)
                torch.cuda.synchronize();times.append(time.perf_counter()-begin);nfes.append(nfe[0]-previous)
                assert images.shape==(16,3,256,256) and torch.isfinite(images).all()
                smoke.append(images.cpu().numpy())
                record["images_seen"]+=len(images)
            features,logits=extract_inception_features(images,inception,batch_size=16)
            assert features.shape==(16,2048) and torch.isfinite(features).all() and torch.isfinite(logits).all()
            real=image_batch(rows[:3],device)
            before_reconstruction=nfe[0]
            reconstruction=core.reconstruct_batch(real)
            reconstruction_nfe=nfe[0]-before_reconstruction
            assert reconstruction.shape==real.shape and torch.isfinite(reconstruction).all()
        np.savez(out/"validation-images.npz",generated=np.stack(smoke),labels=labels.cpu().numpy(),
                 real=real.cpu().numpy(),reconstruction=reconstruction.cpu().numpy())
        save_grid("validation-generation.png",torch.from_numpy(smoke[0]))
        save_grid("validation-reconstruction.png",torch.stack([real,reconstruction],dim=1).flatten(0,1))
        record["validation"]={"paper_metric":False,"generation_smoke_samples":32,"reconstruction_smoke_samples":3,
                              "per_rank_batch_size":16,"world_size":1,"sampling_seconds_per_call":times,
                              "nfe_per_call":nfes,"reconstruction_nfe":reconstruction_nfe,"native_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                              "native_peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        record["images_seen"]+=3
        nfe_hook.remove()
        del inception,features,logits,images,reconstruction
        core.decoder=core.decoder._orig_mod
        torch.cuda.empty_cache()
        phase="ESSENTIAL_ADAPTER_CHECKS"
        proof=check_essential(core,real,seed=cfg["seed"])
        record["adapter_checks"]=proof
        record["images_seen"]+=proof["real_image_encoding_inputs"]+proof["generated_samples"]
        record["reconstruction"]={"rfid_eval":None,"rfid_eval_reason":"Full5000 reference remains unrun; this is declared validation."}
        record["generation"]={"gfid_50k":None,"gfid_50k_reason":"Full50000 reference remains unrun; validation is not a shortened paper result."}
    elif cfg["workload"]=="FULL_REFERENCE":
        core.requires_grad_(False)
        import lpips
        alex=lpips.LPIPS(net="alex",version="0.1").to(device).eval().requires_grad_(False)
        seed_all()
        phase="RECONSTRUCTION_REFERENCE"
        begin=time.perf_counter()
        real_features=[];recon_features=[];l1_values=[];lpips_values=[];reconstruction_nfes=[]
        with torch.no_grad():
            for offset in range(0,len(rows),cfg["reconstruction"]["batch_size"]):
                batch_rows=rows[offset:offset+cfg["reconstruction"]["batch_size"]]
                real=image_batch(batch_rows,device)
                before_reconstruction=nfe[0]
                reconstruction=core.reconstruct_batch(real)
                reconstruction_nfes.append(nfe[0]-before_reconstruction)
                assert torch.isfinite(reconstruction).all()
                feat_real,_=extract_inception_features(real,inception,batch_size=len(real))
                feat_recon,_=extract_inception_features(reconstruction,inception,batch_size=len(real))
                real_features.append(feat_real);recon_features.append(feat_recon)
                l1_values.extend((reconstruction-real).abs().flatten(1).mean(1).cpu().tolist())
                lpips_values.extend(alex(real*2-1,reconstruction*2-1).flatten().cpu().tolist())
                if offset==0:
                    np.savez(out/"reconstruction-samples.npz",real=real[:16].cpu().numpy(),reconstruction=reconstruction[:16].cpu().numpy())
                    save_grid("reconstruction-samples.png",torch.stack([real[:16],reconstruction[:16]],dim=1).flatten(0,1))
                record["images_seen"]+=len(real)
                (out/"reconstruction-progress.json").write_text(json.dumps({"completed_images":record["images_seen"],"required":5000})+"\n")
        actual_real=torch.cat(real_features);actual_recon=torch.cat(recon_features)
        np.save(out/"real-features.npy",actual_real.numpy())
        np.save(out/"reconstruction-features.npy",actual_recon.numpy())
        real_np=actual_real.numpy().astype(np.float64)
        np.savez(out/"real-statistics.npz",mu=real_np.mean(0),sigma=np.cov(real_np,rowvar=False))
        rfid=compute_fid_from_features(actual_recon,str(out/"real-statistics.npz"))
        assert np.isfinite(rfid) and len(l1_values)==len(lpips_values)==5000
        (out/"reconstruction-per-image.json").write_text(json.dumps([dict(image_id=row["image_id"],recon_l1=a,recon_lpips=b) for row,a,b in zip(rows,l1_values,lpips_values)])+"\n")
        record["reconstruction"]={"sample_count":5000,"split":"official validation,first5filenames/class",
            "rfid_eval":rfid,"recon_l1":float(np.mean(l1_values)),"recon_lpips":float(np.mean(lpips_values)),
            "recon_l1_p90":float(np.percentile(l1_values,90)),"mean_tokens":256.0,"mean_tokens_sq":65536.0,
            "validity":"Complete finite native-source reconstruction and metrics; coordinator assesses intact quality.",
            "elapsed_seconds":time.perf_counter()-begin,"nfe":float(np.mean(reconstruction_nfes)),
            "nfe_total":sum(reconstruction_nfes),"nfe_definition":"Actual shared GE forward calls per reconstruction batch; no ODE in native reconstruction.",
            "encode_ms_per_image":None,"encode_ms_per_image_reason":"Encoding latency was not measured separately from reconstruction and metrics.",
            "real_features_sha256":sha(out/"real-features.npy"),
            "reconstruction_features_sha256":sha(out/"reconstruction-features.npy")}
        save()
        del alex,real_features,recon_features,actual_real,actual_recon,real_np,inception
        torch.cuda.empty_cache()
        # Source engine performs the actual50Kloop. Instrumentation retains outputs only.
        phase="GENERATION_REFERENCE"
        seed_all()
        features_file=np.lib.format.open_memmap(out/"generation-features.npy",mode="w+",dtype=np.float32,shape=(50000,2048))
        logits_file=np.lib.format.open_memmap(out/"generation-logits.npy",mode="w+",dtype=np.float32,shape=(50000,1008))
        expected_labels=np.array(json.loads(Path(cfg["data"]["generation_label_manifest"]).read_text())["labels"])
        assert np.array_equal(expected_labels,np.tile(np.arange(1000),50))
        captured=[0];batches=[]
        original_extract=evaluation.extract_inception_features
        original_fid=evaluation.compute_fid_from_features
        original_is=evaluation.compute_inception_score
        original_generate=core.diffusion_generate
        def retain_features(images,model,batch_size):
            features,logits=original_extract(images,model,batch_size)
            first=captured[0];last=first+len(images)
            assert last<=50000
            features_file[first:last]=features.numpy();logits_file[first:last]=logits.numpy();captured[0]=last
            record["images_seen"]=5000+last
            if last==50000 or len(batches)%20==0:
                features_file.flush();logits_file.flush()
                (out/"generation-progress.json").write_text(json.dumps({"completed_features":last,"required":50000})+"\n")
            return features,logits
        def retain_generate(*positional,**keywords):
            supplied=keywords["y_given"].cpu().numpy()
            assert np.array_equal(supplied,expected_labels[captured[0]:captured[0]+len(supplied)])
            previous=nfe[0];torch.cuda.synchronize();tick=time.perf_counter()
            images=original_generate(*positional,**keywords)
            torch.cuda.synchronize()
            batches.append({"samples":len(images),"seconds":time.perf_counter()-tick,"nfe":nfe[0]-previous})
            if len(batches)%20==0:
                (out/"generation-batches.partial.json").write_text(json.dumps(batches)+"\n")
            if len(batches)==1:
                np.savez(out/"generation-samples.npz",images=images.clamp(0,1).cpu().numpy(),labels=keywords["y_given"].cpu().numpy())
                save_grid("generation-samples.png",images.clamp(0,1))
            return images
        def retain_fid(features,reference):
            assert features.shape==(50000,2048) and captured[0]==50000
            value=original_fid(features,reference)
            record["generation"]["gfid_50k"]=value
            return value
        def retain_is(logits):
            assert logits.shape==(50000,1008)
            value,std=original_is(logits)
            record["generation"].update(inception_score=value,inception_score_split_std=std)
            return value,std
        evaluation.extract_inception_features=retain_features
        evaluation.compute_fid_from_features=retain_fid
        evaluation.compute_inception_score=retain_is
        core.diffusion_generate=retain_generate
        record["generation"]={}
        begin=time.perf_counter()
        evaluation.evaluate_fid(core,device,global_step=0,ema_decay=.9978,image_size=256,
            cfg_scales=[cfg["sampling"]["cfg"]],cfg_intervals=[(.1,1)],cfg_norm_orders=["norm_first"],
            num_images=50000,num_classes=1000,batch_size=cfg["sampling"]["batch_size"],
            log_to_wandb=False,log_fid_best=False,current_best_cfg_scale=cfg["sampling"]["cfg"])
        features_file.flush();logits_file.flush()
        assert captured[0]==50000
        (out/"generation-batches.json").write_text(json.dumps(batches,indent=2)+"\n")
        record["generation"].update(sample_count=50000,images_per_class=50,expected_mean_tokens=256.0,
            mean_tokens=256.0,mean_tokens_sq=65536.0,elapsed_seconds=time.perf_counter()-begin,
            sampling_decode_seconds=sum(x["seconds"] for x in batches),
            gen_ms_per_image=1000*sum(x["seconds"] for x in batches)/50000,
            gen_images_per_s=50000/sum(x["seconds"] for x in batches),
            timing_batch_size=16,timing_scope="Native quality sampling+decode at batch16; no batch64 timing comparison is claimed.",
            nfe_definition="ODE denoiser invocations; conditional/null CFG halves are evaluated in the same forward batch.",
            nfe=float(np.mean([x["nfe"] for x in batches])),
            nfe_total=sum(x["nfe"] for x in batches),nfe_mean_per_batch=float(np.mean([x["nfe"] for x in batches])),
            features_sha256=sha(out/"generation-features.npy"),logits_sha256=sha(out/"generation-logits.npy"),
            absolute_difference_headline=abs(record["generation"]["gfid_50k"]-2.12),
            absolute_difference_appendix=abs(record["generation"]["gfid_50k"]-2.735),
            absolute_is_difference_headline=abs(record["generation"]["inception_score"]-294.1),
            absolute_is_difference_appendix=abs(record["generation"]["inception_score"]-268.1))
        record["reference_gate_candidate"]={"gfid_le_3_1":record["generation"]["gfid_50k"]<=3.1,
            "strict_load":True,"reconstruction_complete_finite":True,"coordinator_acceptance_required":True}
        nfe_hook.remove()
    else:
        raise ValueError("Unknown declared workload")
    torch.cuda.synchronize()
    record.update(run_status="COMPLETED",node_status="AWAITING_COORDINATOR_REVIEW",
                  peak_memory_gib=torch.cuda.max_memory_allocated()/(1024**3),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as exc:
    failed_check=phase=="ESSENTIAL_ADAPTER_CHECKS" and isinstance(exc,AssertionError)
    record.update(run_status="FAILED_CORRECTNESS" if failed_check else "FAILED_OPERATIONAL",
                  node_status="BLOCKED_REFERENCE",failure_phase=phase,error=repr(exc))
    record["scientific_negative"]=False
    record["failure_interpretation"]=("Adapter correctness assertion failed; reference quality is not assessed."
                                     if failed_check else "Execution did not establish a scientific negative.")
    raise
finally:
    record["elapsed_seconds"]=time.perf_counter()-start
    save()
