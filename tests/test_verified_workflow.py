"""Control-flow unit fixtures only; these are not measured PPO performance or model weights."""
from pathlib import Path
import pytest
from scripts.benchmark_training import choose_device
from scripts.continue_verified import expected_missing_weights_only
from training.train import curriculum_progress, model_cuda_rng
import torch


def test_device_choice_uses_complete_repeated_jobs():
    rows=[dict(device="cpu",completed=True,decisions_per_second=x) for x in (4,6)]
    rows += [dict(device="cuda",completed=True,decisions_per_second=x) for x in (8,10)]
    device,rates=choose_device(rows,2)
    assert device=="cuda" and rates==dict(cpu=5,cuda=9)


def test_partial_gpu_run_is_not_called_a_completed_comparison():
    rows=[dict(device="cpu",completed=True,decisions_per_second=4) for _ in range(2)]
    rows.append(dict(device="cuda",completed=False))
    device,rates=choose_device(rows,2)
    assert device=="cpu" and set(rates)=={"cpu"}


def test_no_device_selected_without_completed_real_jobs():
    with pytest.raises(RuntimeError): choose_device([],2)


def report(tmp_path,name,body,kind="failure"):
    path=tmp_path/"unit-only.xml"
    path.write_text(f'<testsuites><testsuite><testcase name="{name}"><{kind}>{body}</{kind}></testcase></testsuite></testsuites>')
    return path


def test_only_known_missing_weights_can_precede_training(tmp_path):
    path=report(tmp_path,"test_cuda_agent","FileNotFoundError: submission/weights/best_model.pt")
    assert expected_missing_weights_only(path)


@pytest.mark.parametrize("name,body,kind", [
    ("test_cuda_agent","AssertionError: numerical instability","failure"),
    ("test_unknown","FileNotFoundError: best_model.pt","failure"),
    ("test_cuda_agent","FileNotFoundError: best_model.pt","error")],
    ids=("numeric_failure","unknown_test","setup_error"))
def test_genuine_failures_stop_before_training(tmp_path,name,body,kind):
    assert not expected_missing_weights_only(report(tmp_path,name,body,kind))


def test_time_curriculum_reaches_late_stages_under_a_wall_budget():
    assert curriculum_progress(10,1000000,90,100,"time")==pytest.approx(.9)
    assert curriculum_progress(10,1000000,90,100,"steps")==pytest.approx(.00001)
    assert curriculum_progress(10,100,120,100,"time")==1


def test_cpu_checkpoint_rng_does_not_initialize_cuda(monkeypatch):
    def forbidden(): raise AssertionError("CPU checkpoint tried to initialize CUDA")
    monkeypatch.setattr(torch.cuda,"get_rng_state_all",forbidden)
    monkeypatch.setattr(torch.cuda,"is_available",lambda: True)
    assert model_cuda_rng(torch.nn.Linear(2,2))==[]
