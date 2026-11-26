"""Check stage ordering without substituting geometry for engine completion."""
import gzip
import json

from scripts.continuity_post_lock_eval import funnel


def run(tmp_path, *, spawn, tiles, exit_reached):
    trace = tmp_path / "episode.jsonl.gz"
    samples = {"area": True, "visible": True, "near": True, "facing": True}
    frames = [dict(x=64 * tile + 32, y=32, tic=4 * (i + 1), decision_tic_before=4 * i,
                   switch_before_action=samples, use_rising=True) for i, tile in enumerate(tiles)]
    with gzip.open(trace, "wt") as stream:
        for frame in frames:
            stream.write(json.dumps(frame) + "\n")
    spec = dict(id="synthetic", family="test", exit_type="switch", post_lock_spawn=spawn,
                tile=[64, 64], expected_start=[32 if spawn == "approach-side" else 160, 32],
                open_door_cell=[1, 0], exit_cell=[3, 0],
                drill={"cells": [[i, 0, "."] for i in range(4)]})
    episode = dict(trace=trace, exit=exit_reached, ticks=4 * len(tiles), engine_seed=0,
                   dead=False, invalid_actions=0, agent_error=None)
    return funnel(spec, episode)["stage_tics"]


def test_entry_side_switch_samples_wait_for_observed_crossing(tmp_path):
    assert run(tmp_path, spawn="approach-side", tiles=[1, 2, 3], exit_reached=True) == {
        "door_crossing": 8, "exit_area_discovery": 8, "interaction_range": 8,
        "alignment": 8, "fresh_use": 8, "confirmed_exit": 12,
    }


def test_far_side_start_does_not_imply_crossing_or_exit(tmp_path):
    assert run(tmp_path, spawn="far-side", tiles=[2, 3], exit_reached=False) == {
        "door_crossing": None, "exit_area_discovery": 0, "interaction_range": 0,
        "alignment": 0, "fresh_use": 0, "confirmed_exit": None,
    }
