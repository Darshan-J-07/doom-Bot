"""Create a clean local Git/LFS checkout of an already frozen campaign candidate."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {"env", "submission", "training", "tests", "scripts", "configs", "maps", "docs", "docker"}


def git(repo, *args, env=None):
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env).decode().strip()


def package(artifact, output):
    manifest = json.loads((artifact / "export.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    if output.resolve().is_relative_to(artifact.resolve()):
        raise ValueError("Package must be outside the export tree")
    filenames = git(ROOT, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    for name in sorted(set(filenames) - {""}):
        path = Path(name)
        if path.parts[0] in SOURCE_DIRS and path.parts[0] != "submission" or (
            len(path.parts) == 1 and name != "benchmark_model.py"
        ) or path.parts[:2] == ("benchmarks", "skill-campaign"):
            dst = output / path
            if (ROOT / path).is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, dst)
    shutil.copytree(artifact / "submission", output / "submission")
    shutil.copy2(artifact / "export.json", output / "export.json")
    git(output, "init", "-b", "candidate")
    git(output, "config", "core.autocrlf", "false")
    git(output, "lfs", "install", "--local")
    git(output, "add", ".")
    git(output, "-c", "user.name=DOOM NEXUS Local Candidate", "-c", "user.email=local-candidate@invalid.example",
        "commit", "-m", "Freeze independently evaluated campaign candidate")
    pointer = git(output, "show", "HEAD:submission/weights/best_model.pt")
    expected = manifest["files"]["submission/weights/best_model.pt"]
    if f"oid sha256:{expected}" not in pointer:
        raise ValueError("The candidate weight was not committed as the expected LFS object")
    git(output, "lfs", "fsck")
    check = output.with_name(output.name + "-checkout")
    environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "clone", "--no-hardlinks", str(output.resolve()), str(check)],
                   check=True, env=environment, capture_output=True)
    objects = check / ".git/lfs/objects"
    shutil.copytree(output / ".git/lfs/objects", objects, dirs_exist_ok=True)
    git(check, "lfs", "checkout")
    for name, identity in manifest["files"].items():
        if sha256((check / name).read_bytes()).hexdigest() != identity:
            raise ValueError(f"Reproducibility check failed for {name}")
    if git(check, "status", "--porcelain"):
        raise ValueError("Candidate checkout has working tree differences")
    report = dict(artifact=str(artifact.resolve()), commit=git(check, "rev-parse", "HEAD"),
                  checkout=str(check.resolve()), lfs_pointer=pointer, submission_files=manifest["files"],
                  packed_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()), clean=True)
    (output.parent / "package.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package(args.artifact, args.out)
