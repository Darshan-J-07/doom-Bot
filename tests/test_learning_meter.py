from types import SimpleNamespace

import pytest

from training import learning_meter


def test_active_cycle_excludes_viewer_and_records_only_successful_update(monkeypatch):
    clock = SimpleNamespace(value=0)
    monkeypatch.setattr(learning_meter.time, "monotonic_ns", lambda: clock.value)
    cycle = learning_meter.ActiveCycle()
    cycle.begin()
    clock.value += 7

    def pause():
        clock.value += 80

    cycle.excluding(pause)
    clock.value += 11

    def optimize():
        clock.value += 12
        return "updated"

    assert cycle.optimize(optimize) == "updated"
    assert cycle.pending == (18, 12)
    clock.value += 25
    assert cycle.pending == (18, 12)
    cycle.pending = None
    cycle.begin()

    def failed_update():
        clock.value += 100
        raise RuntimeError("optimizer failed")

    with pytest.raises(RuntimeError, match="optimizer failed"):
        cycle.optimize(failed_update)
    assert cycle.pending is None


def test_meter_resume_keeps_exact_totals_only_for_matching_branch():
    state = learning_meter._initial_state("branch1", {"trained_steps": 12, "updates": 2})
    state.update(checkpoint_steps=48, checkpoint_updates=5, active_rollout_ns=311, active_optimization_ns=90)
    checkpoint = {
        "trained_steps": 48,
        "updates": 5,
        "args": {learning_meter.METADATA_KEY: state},
    }
    restored = learning_meter._initial_state("branch1", checkpoint)
    assert restored == state
    receipt = learning_meter._receipt(restored, None, 401, 3, "completed")
    assert receipt["invocation_active_training_seconds"] == 0
    assert receipt["active_training_ns"] == 401
    assert receipt["added_decisions"] == 36
    assert receipt["added_updates"] == 3
    assert learning_meter._active_ns(learning_meter._initial_state("branch2", checkpoint)) == 0
    with pytest.raises(ValueError, match="checkpoint counters"):
        learning_meter._initial_state("branch1", {**checkpoint, "updates": 6})
