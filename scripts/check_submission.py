"""Local source/provenance audit, supplemental to the untouched official validator."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch


def main():
    errors=[]
    allowed={"__future__","pathlib","numpy","torch","submission"}
    for path in Path("submission").rglob("*.py"):
        tree=ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            roots=[]
            if isinstance(node,ast.Import): roots=[a.name.split(".")[0] for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module: roots=[node.module.split(".")[0]]
            for root in roots:
                if root not in allowed: errors.append(f"Unreviewed inference import {root}: {path}")
    if list(Path(".").rglob("Dockerfile")): errors.append("Dockerfile present in submission tree")
    path=Path("submission/weights/best_model.pt")
    checkpoint={}
    if path.exists():
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        if checkpoint["trained_steps"]<=0 or checkpoint["updates"]<=0: errors.append("Untrained checkpoint")
        if checkpoint.get("selection_split")!="validation": errors.append("No validation selection provenance")
    else:
        errors.append("No trained checkpoint: ViZDoom engine IPC was blocked on this host")
    lfs=subprocess.check_output(["git","lfs","ls-files"],text=True)
    if "submission/weights/best_model.pt" not in lfs: errors.append("Selected weights missing from Git LFS index")
    report=dict(passed=not errors,errors=errors,trained_steps=checkpoint.get("trained_steps",0),
                updates=checkpoint.get("updates",0),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                lfs=lfs.strip(),note="Static import allowlist supplements manual source review; not a security proof")
    Path("benchmarks/submission-audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if errors: raise SystemExit(1)


if __name__=="__main__": main()
