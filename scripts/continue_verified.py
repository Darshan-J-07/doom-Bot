"""Continue the existing project using the already-verified Linux Python environment."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def expected_missing_weights_only(path):
    """A pre-training report can fail only at the known missing-weight prerequisite.

    This does not change assertions or convert failures to passes; the full suite
    is rerun after genuine training and then must have exit code zero.
    """
    allowed={"test_checkpoint_load_once_and_readonly_act","test_memory_reset_and_independent_instances",
             "test_real_episode_without_invalid_actions","test_template_baseline_passes","test_cuda_agent"}
    failures=[]
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.find("error") is not None: return False
        node=case.find("failure")
        if node is None: continue
        body="".join(node.itertext())
        if case.get("name") not in allowed or "FileNotFoundError" not in body or "best_model.pt" not in body:
            return False
        failures.append(case.get("name"))
    return bool(failures)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds",type=int,default=1800,help="main GRU invocation budget; baseline/bench/eval are additional")
    args=ap.parse_args()
    if args.seconds<30: ap.error("Use at least 30 seconds for the main training budget")
    tag=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=Path("benchmarks/verified-environment")/tag; out.mkdir(parents=True,exist_ok=False)
    def run(name,*command,allow_failure=False):
        print(f"\n[{name}] {' '.join(str(c) for c in command)}",flush=True)
        with (out/f"{name}.txt").open("w",encoding="utf-8") as log:
            proc=subprocess.Popen([str(c) for c in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding="utf-8",errors="replace")
            for line in proc.stdout:
                print(line,end="",flush=True); log.write(line); log.flush()
            code=proc.wait()
        with (out/"exit-codes.jsonl").open("a",encoding="utf-8") as log:
            log.write(json.dumps(dict(stage=name,exit_code=code))+"\n")
        if code and not allow_failure:
            raise RuntimeError(f"Stage {name} failed (exit {code}); inspect {out/name}.txt")
        return code
    def python(name,*command,**kwargs): return run(name,sys.executable,*command,**kwargs)
    try:
        python("project-engine-smoke","-m","scripts.preflight","--engine","--require-cuda","--out",out/"preflight.json")
        run("packages",sys.executable,"-m","pip","freeze")
        before=out/"tests-before-training.xml"
        code=python("full-tests-before-training","-m","pytest","-q","-ra",f"--junitxml={before}",allow_failure=True)
        if code and not expected_missing_weights_only(before):
            raise RuntimeError("Pre-training suite has failures beyond missing genuine weights; inspect and fix them first")
        bench=Path("runs")/tag/"device-benchmark"
        python("cpu-vs-cuda-ppo","-m","scripts.benchmark_training","--out",bench)
        selected=json.loads((bench/"selection.json").read_text())["selected_device"]
        (out/"device-selection.json").write_text((bench/"selection.json").read_text(),encoding="utf-8")
        baseline=Path("runs")/tag/"mlp"
        python("baseline-training","-m","training.train","--model","mlp","--stage","movement","--device",selected,
               "--envs","1","--steps","8192","--seconds","300","--rollout","64","--sequence","16",
               "--epochs","2","--cooldown-ms","200","--out",baseline)
        python("baseline-selection","-m","training.evaluate","--checkpoint",baseline/"latest.pt","--split","validation",
               "--seeds","0","1","--ticks","2100","--device",selected,"--promote","--out",out/"baseline-validation.json")
        python("full-baseline-tests","-m","pytest","-q","-ra",f"--junitxml={out/'baseline-tests.xml'}")
        python("baseline-official-validator","validate.py","--device",selected)
        recurrent=Path("runs")/tag/"gru"
        python("curriculum-training","-m","training.train","--model","gru","--stage","auto",
               "--curriculum-clock","time","--device",selected,"--envs","1","--steps","1000000",
               "--seconds",str(args.seconds),"--rollout","128","--sequence","32","--epochs","3",
               "--cooldown-ms","200","--out",recurrent)
        # Verify real resume on the saved optimizer/RNG state and preserve an additional candidate.
        import torch
        saved=torch.load(recurrent/"latest.pt",map_location="cpu",weights_only=True)
        resume=Path("runs")/tag/"gru-resumed"
        python("resume-training","-m","training.train","--model","gru","--stage","mixed","--device",selected,
               "--envs","1","--steps",str(saved["trained_steps"]+2048),"--seconds","180",
               "--resume",recurrent/"latest.pt","--cooldown-ms","200","--out",resume)
        python("best-checkpoint-selection","-m","training.evaluate","--checkpoint",baseline/"latest.pt",
               "--checkpoint",recurrent/"latest.pt","--checkpoint",resume/"latest.pt","--split","validation",
               "--seeds","0","1","--ticks","2100","--device",selected,"--promote","--out",out/"validation.json")
        python("heldout-evaluation","-m","training.evaluate","--checkpoint","submission/weights/best_model.pt",
               "--split","test","--seeds","0","1","--ticks","2100","--device",selected,"--out",out/"heldout.json")
        python("full-final-tests","-m","pytest","-q","-ra",f"--junitxml={out/'final-tests.xml'}")
        for device in ("cpu","cuda"):
            python(f"inference-{device}","-m","scripts.benchmark","--device",device,"--steps","512")
            (out/f"performance-{device}.json").write_text(Path("benchmarks/performance.json").read_text(),encoding="utf-8")
        run("git-lfs", "git","lfs","install","--local")
        run("stage-genuine-weights","git","add","--",".gitattributes","submission/weights/best_model.pt")
        python("source-audit","-m","scripts.check_submission")
        for device in ("cpu","cuda"): python(f"official-validator-{device}","validate.py","--device",device)
        summary=dict(status="LOCAL BASELINE CHECKS PASSED; COMPETITION READINESS REQUIRES REVIEW",
                     selected_training_device=selected,evidence=str(out),checkpoint="submission/weights/best_model.pt",
                     limitations="Organizer base-image/offline sandbox certification, strong held-out skill and public event-valid delivery require review")
        (out/"completed.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        (out/"blocked.json").write_text(json.dumps(dict(status="BLOCKED",error=str(exc)),indent=2),encoding="utf-8")
        raise


if __name__=="__main__": main()
