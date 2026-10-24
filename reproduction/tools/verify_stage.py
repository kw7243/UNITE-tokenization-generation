"""Verify staged source bytes, links, execute permissions, and Git identity."""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

source, stage = map(Path, sys.argv[1:])
excluded = {"staging", "out", "outputs", "eval_outputs", "slurm_outputs", "wandb", ".cache", "__pycache__", ".venv", "venv"}
def manifest(root):
    records = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        rel_parent = Path(parent).relative_to(root)
        if rel_parent == Path("."):
            dirs[:] = [d for d in dirs if d not in excluded]
        for name in dirs[:] + files:
            path = Path(parent) / name
            rel = path.relative_to(root).as_posix()
            if rel == "REPRODUCIBILITY_METADATA.json" or rel == "STAGING_MANIFEST.json":
                continue
            if path.is_symlink():
                records[rel] = {"kind": "symlink", "target": os.readlink(path)}
                if name in dirs:
                    dirs.remove(name)
            elif path.is_file():
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                records[rel] = {"kind": "file", "bytes": path.stat().st_size, "sha256": h.hexdigest(),
                                "execute_bits": stat.S_IMODE(path.stat().st_mode) & 0o111}
    return records
expected, actual = manifest(source), manifest(stage)
assert expected == actual, {"missing": sorted(expected.keys() - actual.keys()),
                            "extra": sorted(actual.keys() - expected.keys()),
                            "changed": [k for k in expected.keys() & actual.keys() if expected[k] != actual[k]]}
commands = [["git", "rev-parse", "HEAD"], ["git", "branch", "--show-current"],
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":(exclude)REPRODUCIBILITY_METADATA.json", ":(exclude)STAGING_MANIFEST.json"], ["git", "ls-files", "--stage"]]
git_identity = []
for cmd in commands:
    left = subprocess.check_output(cmd, cwd=source, text=True)
    right = subprocess.check_output(cmd, cwd=stage, text=True)
    assert left == right, (cmd, left, right)
    git_identity.append({"command": cmd, "value": left})
receipt = {"source": str(source), "stage": str(stage), "file_count": len(expected), "files": expected,
           "git_identity": git_identity, "policy": "All included bytes, links and execute bits identical; directory/group/owner/setgid attributes are not copied; root generated-output exclusions follow installed helper.",
           "installed_helper_sha256": "44e5dc6f1a958b1f4b32e8dceeb49814885ad8dcca59716090ca87b1033731fa",
           "proven_compatibility_helper_sha256": "f69e0ffd7b8129302952d13d62d658645b8a83f26af9c10e68f8abeed024ae3b"}
(stage / "STAGING_MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(f"Verified {len(expected)} files/links, executable bits and Git identity", flush=True)
