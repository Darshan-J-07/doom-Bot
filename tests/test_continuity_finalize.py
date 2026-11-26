from pathlib import Path
import sys

import pytest
import torch

from scripts import continuity_finalize as finalize


def test_checkpoint_backed_learning_time_rejects_a_forged_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize, "TARGET", 3)
    folder = tmp_path / "training"
    folder.mkdir()
    checkpoint = folder / "step-25.pt"
    torch.save({"trained_steps": 25, "updates": 2, "args": {finalize.METADATA_KEY: {
        "meter_id": "pair-59", "active_rollout_ns": 2_500_000_000, "active_optimization_ns": 500_000_000,
        "checkpoint_steps": 25, "checkpoint_updates": 2, "start_steps": 10, "start_updates": 1,
    }}}, checkpoint)
    block = {"label": "pair-59", "meter_id": "pair-59", "phase": "validation test", "checkpoint": str(checkpoint),
             "active_training_ns": 3_000_000_000, "added_decisions": 15, "added_updates": 1}
    status = {"blocks": [block], "supervised_blocks": [], "active_training_seconds": 3.0}
    verified = finalize.verify_learning(tmp_path, status)
    assert verified["pair-59"]["decisions"] == 15
    with pytest.raises(ValueError, match="duplicate learning credit"):
        finalize.verify_learning(tmp_path, {**status, "active_training_seconds": 6.0,
                                          "blocks": [block, {**block, "label": "duplicate"}]})
    block["active_training_ns"] += 1
    with pytest.raises(ValueError, match="Checkpoint-backed active time"):
        finalize.verify_learning(tmp_path, status)


def test_reserved_scope_stays_sealed_without_both_authorizations(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("The reserved panel or freeze was invoked")

    monkeypatch.setattr(finalize.evaluation, "freeze", unexpected)
    monkeypatch.setattr(finalize, "command", unexpected)
    assert finalize.run_reserved(tmp_path, tmp_path, "candidate", None, True, lambda: False)["status"] == "sealed"
    assert finalize.run_reserved(tmp_path, tmp_path, "candidate", Path("gate.json"), False, lambda: False)["status"] == "sealed"


def test_final_jobs_stop_and_preserve_the_receipt_when_their_bound_expires(tmp_path):
    result = finalize.command(tmp_path, tmp_path, "test-wall", [sys.executable, "-c", "import time;time.sleep(60)"],
                              .02, lambda: False)
    assert result["stop_reason"] == "wall limit"
    assert not result["passed"]
    assert finalize.read(tmp_path / "test-wall.json")["return_code"] is not None
