"""UNTRAINED model-only CPU microbenchmark. Not trained-agent or gameplay evidence."""
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from doom_env import observation_space
from submission.model import SpatialPolicy, buttons, policy_stats
from submission.preprocessing import tensor_batch


@torch.inference_mode()
def main():
    torch.set_num_threads(1); torch.manual_seed(7)
    space=observation_space(); space.seed(7); observation=space.sample()
    observation["visible_entities"]["mask"][:]=0
    observation["visible_entities"]["entity_type"][:]=0
    observation["visible_entities"]["features"][:]=0
    results=[]
    for recurrent in (False,True):
        model=SpatialPolicy(recurrent=recurrent).eval()
        hidden=model.initial_state(); previous=torch.zeros(1,9); starts=torch.zeros(1)
        for _ in range(16): model(tensor_batch([observation]),hidden,previous,starts)
        timings=[]
        for _ in range(512):
            t=time.perf_counter()
            logits,_,hidden=model(tensor_batch([observation]),hidden,previous,starts)
            groups,_,_=policy_stats(logits)
            previous=buttons(groups).float()
            # Include the NumPy action transfer used by the real entry point.
            previous[0].numpy().astype(np.int8)
            timings.append((time.perf_counter()-t)*1000)
        memory=psutil.Process().memory_info()
        results.append(dict(model="GRU" if recurrent else "MLP",parameters=sum(p.numel() for p in model.parameters()),
                            samples=len(timings),mean_ms=float(np.mean(timings)),peak_ms=float(max(timings)),
                            p95_ms=float(np.percentile(timings,95)),process_rss_bytes=memory.rss,
                            windows_peak_working_set_bytes=getattr(memory,"peak_wset",None)))
    report=dict(scope="UNTRAINED synthetic-observation microbenchmark; not DoomAgent or official latency",device="cpu",
                cpu_threads=torch.get_num_threads(),trained_weights_saved=False,results=results)
    Path("benchmarks/model-microbenchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
