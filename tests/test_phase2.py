import numpy as np
import pytest
import torch

from submission.model import SpatialPolicy
from training.ppo import _minibatch
from training.rewards import ProgressReward


@pytest.mark.parametrize("recurrent", [False, True])
def test_batched_ppo_matches_unpadded_sequences_across_episode_start(recurrent):
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(14)
    steps, envs = 4, 2
    def floats(*shape):
        return torch.randn(steps, envs, *shape, generator=rng)
    model = SpatialPolicy(recurrent=recurrent)
    data = dict(
        obs=dict(distance=floats(64), hit_type=torch.zeros(steps, envs, 64, dtype=torch.long),
                 vitals=floats(17), entities=floats(32, 5),
                 entity_type=torch.zeros(steps, envs, 32, dtype=torch.long), mask=floats(32).gt(0)),
        hidden=floats(128), previous_action=floats(9), starts=torch.zeros(steps, envs),
        values=floats(),
    )
    data["starts"][1, 0] = 1
    sequences = [(0, 0, 4), (1, 2, 4)]
    index, logits, values = _minibatch(model, data, sequences)
    assert len(logits) == 6
    for env, left, right in sequences:
        hidden = data["hidden"][left, env:env+1]
        for step in range(left, right):
            where = (index[0] == step) & (index[1] == env)
            obs = {key: value[step, env:env+1] for key, value in data["obs"].items()}
            expected_logits, expected_value, hidden = model(
                obs, hidden, data["previous_action"][step, env:env+1], data["starts"][step, env:env+1])
            torch.testing.assert_close(logits[where], expected_logits, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(values[where], expected_value, atol=1e-6, rtol=1e-5)


def test_progress_credit_is_once_per_key_door_and_terminal():
    initial = dict(x=0, y=0, keys=np.zeros(6), kills=0, damage_taken=0,
                   sector_floor=np.zeros(1), sector_ceiling=np.zeros(1), door_lock=np.ones(1),
                   health=100, dead=False, exit_reached=False, tic=0)
    key = np.array([1, 0, 0, 0, 0, 0])
    locked = ProgressReward()
    locked.reset(initial)
    opened = dict(initial, keys=key, sector_ceiling=np.array([100.0]))
    assert locked.step(opened) > 10
    assert locked.step(initial) < 0
    assert locked.step(opened) < 0
    complete = dict(opened, exit_reached=True)
    assert locked.step(complete) > 24
    assert locked.step(complete) == 0
    assert locked.summary()["keys"] == locked.summary()["locks"] == 1


def test_two_lock_generation_preserves_original_map_bytes(tmp_path):
    from doom_env.mapinfo import LOCK_BLUE, LOCK_RED, load_mapinfo, parse_udmf
    from doom_env.wad import map_lumps, read_wad
    from training.generate_maps import create_map
    from training.maps import load_verified_maps

    rows = load_verified_maps()
    for stage in ("movement", "combat", "exploration", "keys", "gauntlet"):
        original = next(row for row in rows if row["stage"] == stage)
        sample = create_map(tmp_path / f"{stage}.wad", stage, original["seed"])
        assert sample["sha256"] == original["sha256"]
    expanded = tmp_path / "expanded.wad"
    create_map(expanded, "gauntlet", 310000, two_locks=True)
    assert sorted(load_mapinfo(expanded, "MAP01").door_sectors.values()) == [LOCK_RED, LOCK_BLUE]
    things = parse_udmf(map_lumps(read_wad(expanded), "MAP01")["TEXTMAP"].decode())["thing"]
    assert {5, 13}.issubset(thing["type"] for thing in things)


@pytest.mark.parametrize("deterministic", [False, True])
def test_action_only_inference_matches_ppo_sampling(deterministic):
    from submission.model import GROUP_SIZES, action_groups, policy_stats

    logits = torch.arange(4 * sum(GROUP_SIZES), dtype=torch.float32).reshape(4, -1) / 17
    expected = torch.Generator().manual_seed(13)
    actual = torch.Generator().manual_seed(13)
    for _ in range(8):
        groups, _, _ = policy_stats(logits, deterministic=deterministic, generator=expected)
        assert torch.equal(action_groups(logits, deterministic, actual), groups)
