"""Hardware and package evidence without modifying any global Python installation."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import psutil
import torch


def main():
    packages = {}
    for name in ("torch","numpy","vizdoom","gymnasium","pytest","psutil","doom-env"):
        try: packages[name] = md.version(name)
        except md.PackageNotFoundError: packages[name] = None
    gpu = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"],capture_output=True,text=True).stdout.strip() if shutil.which("nvidia-smi") else None
    report = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  cpu=os.environ.get("PROCESSOR_IDENTIFIER"),logical_cpus=os.cpu_count(),
                  physical_cpus=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total,
                  available_ram_bytes=psutil.virtual_memory().available, gpu=gpu,
                  cuda_available=torch.cuda.is_available(),torch_cuda_runtime=torch.version.cuda,
                  packages=packages,docker_executable=shutil.which("docker"))
    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/hardware.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
