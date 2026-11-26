"""Produce a row for every previous failure, setup error and skip, preserving evidence."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def analyze(path="benchmarks/full-tests-utf8.xml"):
    rows=[]
    for case in ET.parse(path).getroot().iter("testcase"):
        node=next((case.find(tag) for tag in ("failure","error","skipped") if case.find(tag) is not None),None)
        if node is None: continue
        message="".join(node.itertext())
        if node.tag=="skipped":
            cause="CPU-only PyTorch; RTX 2050 hardware was detected, but CUDA agent execution was not tested"
            action="Install CUDA PyTorch in WSL 2; pass CUDA tensor preflight, train real weights, rerun this case"
            category="CUDA build"
        elif "MessageQueueException" in message:
            cause="Actual ViZDoom initialization raises MessageQueueException before the tested behavior is reached"
            action="Pass Linux engine smoke, then rerun this exact case without changing its assertions"
            category="Engine IPC"
        elif "best_model.pt" in message:
            cause="DoomAgent cannot initialize because genuine trained weights do not exist"
            action="Complete real PPO updates, validate and promote a checkpoint, then rerun this case"
            category="Missing weights"
        elif "sched_getaffinity" in message:
            cause="The upstream test calls os.sched_getaffinity, which native Windows does not provide"
            action="Run unchanged in Linux; verify real one-CPU affinity and peak RSS assertions"
            category="Linux affinity"
        elif "No module named 'resource'" in message:
            cause="The upstream test imports the Unix resource module, absent in native Windows"
            action="Run unchanged in Linux; preserve protocol failure and memory-growth checks"
            category="Unix resource"
        else:
            cause="Unclassified: inspect attached traceback before changing code"
            action="Review exact assertion and reproduce on the supported platform"
            category="Needs investigation"
        source=Path(case.get("classname").replace(".","/")+".py")
        function=case.get("name").split("[")[0]
        tree=ast.parse(source.read_text(encoding="utf-8"))
        lineno=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==function)
        rows.append(dict(nodeid=source.as_posix()+"::"+case.get("name"),previous_result=node.tag,
                         source_line=lineno,category=category,cause=cause,remediation=action,
                         evidence=node.get("message",message[:350]),rerun_status="NOT EXECUTED ON LINUX"))
    return rows


def main():
    rows=analyze(); out=Path("benchmarks/continuation"); out.mkdir(parents=True,exist_ok=True)
    (out/"individual-test-audit.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    doc=["# Individual investigation of previous failed, errored and skipped tests", "",
         "All 29 failed cases, 12 setup errors and the one CUDA skip are listed individually below.",
         "Classification comes from each JUnit traceback plus its test source. A classified cause is",
         "not a passed rerun. No organizer test or validator was changed.", ""]
    for i,row in enumerate(rows,1):
        doc += [f"## {i}. `{row['nodeid']}`", "",
                f"Previous result: **{row['previous_result']}**. Source line: {row['source_line']}.","",
                f"Cause: {row['cause']}.","",f"Action: {row['remediation']}.","",
                f"Linux rerun: **{row['rerun_status']}**.","",
                "```bash",f"python -m pytest '{row['nodeid']}' -q -ra","```",""]
    Path("docs/INDIVIDUAL_TEST_AUDIT.md").write_text("\n".join(doc),encoding="utf-8")
    counts={name:sum(r["previous_result"]==name for r in rows) for name in ("failure","error","skipped")}
    assert counts==dict(failure=29,error=12,skipped=1)
    assert not any(r["category"]=="Needs investigation" for r in rows)
    print(json.dumps(dict(individual_cases=len(rows),previous_results=counts),indent=2))


if __name__=="__main__": main()
