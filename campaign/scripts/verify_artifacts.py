"""CPU artifact, source, metric and fixed-validation checks for A1."""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import numpy as np
import torch
import yaml

out=Path(sys.argv[1])
sys.path.insert(0,str(Path.cwd()/"third_party/unite"))
from models.unite import UNITE
from utils.fid import get_inception_model
from utils.crop import CenterCropTransform
from PIL import Image
from torchvision.transforms import ToTensor

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()

source=json.loads(Path("campaign/locks/source-imports.json").read_text())
for name,files in source["files"].items():
    for relative,expected in files.items():
        assert sha(Path("third_party")/name/relative)==expected,(name,relative)
inputs=Path("/data/vision/torralba/u/kwen1/image-tokenization-20261023/UNITE-released-20261023-inputs")
expected={
"UNITE-B.pt":"7b7f40dd37111196cad6e4401bf9737156268612e001c29cb0a760ae416bff4b",
"weights-inception-2015-12-05-6726825d.pth":"6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
"jit_in256_stats.npz":"412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c"}
files={}
for name,digest in expected.items():
    p=inputs/name
    actual=sha(p)
    assert actual==digest,(name,actual)
    files[name]={"path":str(p),"sha256":actual,"bytes":p.stat().st_size}
stats_bytes=(inputs/"jit_in256_stats.npz").read_bytes()
blob=hashlib.sha1(b"blob "+str(len(stats_bytes)).encode()+b"\0"+stats_bytes).hexdigest()
assert blob=="3c94ea961be94d06efa9e5bf4da71eff52deaebb",blob
stats=np.load(inputs/"jit_in256_stats.npz")
assert stats["mu"].shape==(2048,) and stats["sigma"].shape==(2048,2048)
files["jit_in256_stats.npz"].update(git_blob=blob,source_commit="cbc743a2ada5e9762697da2c83f8c4f8379e8c17")

data=Path("/data/vision/torralba/datasets/imagenet_pytorch_new")
mapping_path=data/"imagenet_class_index.json"
mapping=json.loads(mapping_path.read_text())
classes=sorted(p.name for p in (data/"val").iterdir() if p.is_dir())
assert len(classes)==1000
manifest=[]
for index,synset in enumerate(classes):
    assert mapping[str(index)][0]==synset
    images=sorted(p for p in (data/"val"/synset).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    assert len(images)==50,(synset,len(images))
    for p in images[:5]:
        manifest.append({"image_id":synset+"/"+p.name,"path":str(p),"class_index":index,"synset":synset,"sha256":sha(p)})
assert len(manifest)==5000
manifest_bytes=(json.dumps({"selection":"first five lexicographically sorted validation filenames per sorted synset","rows":manifest},indent=2)+"\n").encode()
(out/"reconstruction-manifest.json").write_bytes(manifest_bytes)
label_bytes=(json.dumps({"labels":np.tile(np.arange(1000),50).tolist(),"rule":"native source np.tile(0..999,50), 50 per class"},indent=2)+"\n").encode()
(out/"generation-labels.json").write_bytes(label_bytes)

cfg=yaml.safe_load(Path("third_party/unite/configs/imagenet.yaml").read_text())["gen_tok"]
cfg.update(encoder_model_size="Base",decoder_model_size="Base",flow_steps_per_recon=14)
checkpoint=torch.load(inputs/"UNITE-B.pt",map_location="cpu",weights_only=True,mmap=True)
assert checkpoint["epoch"]==240
assert float(checkpoint["current_best_cfg_scale"])==2.0
state=checkpoint["ema"]
renamed={k.replace("._orig_mod.","."):v for k,v in state.items()}
assert len(renamed)==len(state)
assert sum(k!=k.replace("._orig_mod.",".") for k in state)==152
model=UNITE(cfg,num_latent_tokens=256,num_classes=1000).eval().requires_grad_(False)
load=model.load_state_dict(renamed,strict=True)
assert not load.missing_keys and not load.unexpected_keys
strict={"epoch":checkpoint["epoch"],"step":checkpoint["step"],"saved_cfg":checkpoint["current_best_cfg_scale"],"compiled_key_renames":152,"missing_keys":load.missing_keys,"unexpected_keys":load.unexpected_keys,"gen_tok":cfg,"num_latent_tokens":256,"num_classes":1000,"source_training_M":14}
del model,state,renamed,checkpoint
gc.collect()
assert importlib.metadata.version("torch-fidelity")=="0.3.0"
inception=get_inception_model(torch.device("cpu"))
del inception
import lpips
external=lpips.LPIPS(net="alex",version="0.1").eval().requires_grad_(False)
del external
crop=CenterCropTransform(256)
sample=ToTensor()(crop(Image.open(manifest[0]["path"]).convert("RGB")))
assert sample.shape==(3,256,256) and sample.min()>=0 and sample.max()<=1
module_versions={name:importlib.metadata.version(name) for name in ("torch","torchvision","timm","torchdiffeq","transformers","omegaconf","numpy","scipy","einops","lpips","PyYAML","torch-fidelity")}
runtime={"python":sys.version,"platform":platform.platform(),"torch_cuda":torch.version.cuda,"packages":module_versions,"executable":sys.executable,"base_prefix":sys.base_prefix,"site_prefix":sys.prefix}
metric={"torch_fidelity":"0.3.0","extractor":"inception-v3-compat","features":["2048","logits_unbiased"],"inception_weights_sha256":expected["weights-inception-2015-12-05-6726825d.pth"],"statistics_sha256":expected["jit_in256_stats.npz"],"fid_implementation_sha256":sha("third_party/unite/utils/fid.py"),"statistics_precision":"float64, numpy.cov N-1, pinned scipy.sqrtm fallback only after nonfinite","pixel_conversion":"source (x*255).clamp(0,255).to(uint8), own extractor resize","lpips_reporting":"external AlexNet v0.1 on clamped [0,1] mapped to [-1,1]","is":"all50000 samples,first1000 class dimensions/image,10contiguous splits"}
metric_bytes=(json.dumps(metric,indent=2)+"\n").encode()
(out/"metric-protocol.json").write_bytes(metric_bytes)
artifacts={"files":files,"source_import_manifest_sha256":sha("campaign/locks/source-imports.json"),"strict_checkpoint_load":strict,"runtime":runtime,"metric_protocol_sha256":hashlib.sha256(metric_bytes).hexdigest(),"reconstruction_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),"generation_label_manifest_sha256":hashlib.sha256(label_bytes).hexdigest(),"class_map":{"path":str(mapping_path),"sha256":sha(mapping_path)},"archive_sha256":None,"archive_sha256_reason":"A1 consumes the permitted extracted validation images, not an exposed original ImageNet image archive. The consumed5000 image bytes are hashed individually.","model_cache_files":{str(p):sha(p) for p in sorted(Path(os.environ["TORCH_HOME"]).rglob("*")) if p.is_file()}}
(out/"artifacts.json").write_text(json.dumps(artifacts,indent=2)+"\n")
print(json.dumps({"strict_load":"PASS","metric_cpu_load":"PASS","manifest_count":len(manifest),"stats_blob":blob,"runtime":runtime}))
