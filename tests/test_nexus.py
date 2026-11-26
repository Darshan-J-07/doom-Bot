"""Local acceptance tests; these are not the organizers' hidden gameplay tests."""
from __future__ import annotations
import inspect
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from doom_env import action_space, observation_space, make_env
from submission.model import SpatialPolicy, buttons, policy_stats
from submission.preprocessing import tensor_batch
from training.ppo import generalized_advantage, update
from training.rewards import EpisodeMetrics

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def official_observation(entities=0, hit=0, health=100, ammo=50):
    obs = observation_space().sample()
    obs["rays"]["hit_type"][:] = hit
    obs["rays"]["distance"][:] = .5
    obs["vitals"][:] = 0
    obs["vitals"][[0,2,3,4]] = health, 2, ammo, ammo
    obs["visible_entities"]["mask"][:] = 0
    obs["visible_entities"]["mask"][:entities] = 1
    obs["visible_entities"]["features"][:] = 0
    obs["visible_entities"]["features"][:entities,0] = .5
    obs["visible_entities"]["features"][:entities,2] = 1
    obs["visible_entities"]["entity_type"][:] = 0
    obs["visible_entities"]["entity_type"][:entities] = 2
    assert observation_space().contains(obs)
    return obs


@pytest.mark.parametrize("entities", [0,1,8,32])
@pytest.mark.parametrize("hit", range(6))
def test_valid_spatial_observations(entities, hit):
    model = SpatialPolicy()
    obs = tensor_batch([official_observation(entities, hit)])
    logits, value, hidden = model(obs, model.initial_state(), torch.zeros(1,9), torch.ones(1))
    assert all(torch.isfinite(t).all() for t in (logits,value,hidden))
    groups, lp, ent = policy_stats(logits)
    assert action_space().contains(buttons(groups)[0].numpy())
    assert torch.isfinite(lp).all() and (ent >= 0).all()


@pytest.mark.parametrize("health,ammo", [(0,0),(1,0),(200,0),(200,9999)])
def test_vitals_and_extremes(health,ammo):
    obs = official_observation(0, 1, health, ammo)
    obs["vitals"][8:14] = 1
    obs["vitals"][14:] = np.finfo(np.float32).max
    obs["visible_entities"]["features"][:] = np.finfo(np.float32).min
    assert all(torch.isfinite(v).all() for v in tensor_batch([obs]).values())


def test_entity_permutation_and_padding():
    model = SpatialPolicy().eval(); obs = official_observation(8)
    obs["visible_entities"]["features"][:8,0] = np.linspace(.1,.8,8)
    a = tensor_batch([obs]); permutation = np.random.default_rng(7).permutation(32)
    for key in obs["visible_entities"]:
        obs["visible_entities"][key] = obs["visible_entities"][key][permutation]
    b = tensor_batch([obs])
    inputs = model.initial_state(), torch.zeros(1,9), torch.ones(1)
    assert torch.allclose(model(a,*inputs)[0], model(b,*inputs)[0], atol=1e-6)


def test_gae_boundaries_and_time_limit_bootstrap():
    r = torch.tensor([[1.],[2.],[3.]])
    v = torch.zeros_like(r); nv = torch.tensor([[10.],[20.],[30.]])
    term = torch.tensor([[False],[True],[False]])
    trunc = torch.tensor([[True],[False],[False]])
    a, ret = generalized_advantage(r,v,nv,term,trunc,gamma=.9,lam=1)
    assert a.flatten().tolist() == pytest.approx([10.,2.,30.])
    assert torch.equal(a,ret)


@pytest.mark.parametrize("recurrent", [False,True])
def test_ppo_updates_parameters_with_finite_gradients(recurrent):
    model = SpatialPolicy(recurrent=recurrent)
    obs = tensor_batch([official_observation()]); t = 4
    data = dict(obs={k:v.unsqueeze(0).repeat((t,)+(1,)*v.ndim) for k,v in obs.items()},
                hidden=torch.zeros(t,1,128), previous_action=torch.zeros(t,1,9),
                starts=torch.ones(t,1), rewards=torch.tensor([[0.],[1.],[2.],[3.]]),
                terminated=torch.zeros(t,1,dtype=torch.bool), truncated=torch.zeros(t,1,dtype=torch.bool),
                next_values=torch.zeros(t,1))
    with torch.no_grad():
        logits, values, _ = model(obs, model.initial_state(), torch.zeros(1,9), torch.ones(1))
        acts, lp, _ = policy_stats(logits)
    data.update(actions=acts.unsqueeze(0).repeat(t,1,1), log_probs=lp.unsqueeze(0).repeat(t,1),
                values=values.unsqueeze(0).repeat(t,1))
    before = model.actor.weight.detach().clone()
    stats = update(model, torch.optim.Adam(model.parameters(),lr=1e-3), data, epochs=1,sequence_length=2)
    assert np.isfinite(list(stats.values())).all()
    assert not torch.equal(before, model.actor.weight)


def test_official_files_unchanged():
    # Fixed hashes captured from the inspected upstream commit also work in a
    # fresh template/snapshot clone that does not contain upstream Git objects.
    manifest=json.loads((ROOT/"docs/official-source-sha256.json").read_text())
    assert manifest["upstream_commit"]=="63b90c441ca92bcdf043de9ee3930ac6370a44bc"
    paths = ["validate.py", *[str(p.relative_to(ROOT)).replace("\\","/")
                             for p in (ROOT/"env/doom_env").glob("*.py")]]
    assert set(paths)==set(manifest["files"])
    for path in paths:
        raw=(ROOT/path).read_bytes().replace(b"\r\n",b"\n")
        assert hashlib.sha256(raw).hexdigest()==manifest["files"][path]


def test_map_splits_disjoint():
    rows = json.loads((ROOT/"training/maps/manifest.json").read_text())["maps"]
    groups = [{r["sha256"] for r in rows if r["split"]==split} for split in ("train","validation","test")]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])


def test_generated_curriculum_engine_loads():
    rows = json.loads((ROOT/"training/maps/manifest.json").read_text())["maps"]
    for stage in ("movement","combat","exploration","keys","gauntlet"):
        spec = next(r for r in rows if r["stage"]==stage and r["split"]=="train")
        env = make_env(spec["path"],episode_ticks=40,render_buffers=False)
        try:
            obs,_=env.reset(seed=0)
            assert env.observation_space.contains(obs)
            for _ in range(12):
                obs,_,term,trunc,_=env.step(np.zeros(9,np.int8))
                if term or trunc: break
            assert term or trunc
        finally: env.close()


def test_reward_novelty_and_door_rewards_are_once_only():
    state = dict(x=0,y=0,keys=np.zeros(6),kills=0,damage_taken=0,sector_floor=np.zeros(1),
                 sector_ceiling=np.zeros(1),door_lock=np.zeros(1),health=100,dead=False,
                 exit_reached=False,tic=0)
    tracker=EpisodeMetrics(); tracker.reset(state)
    moved=dict(state,x=150,sector_ceiling=np.array([100.]))
    first=tracker.step(moved); repeated=tracker.step(moved)
    assert first > 2 and repeated == pytest.approx(-.002)
    tracker.step(state)
    assert tracker.step(moved) == pytest.approx(-.002)


def trained_agent():
    from submission.agent import DoomAgent
    return DoomAgent(observation_space(),action_space(),"cpu")


def test_checkpoint_load_once_and_readonly_act(monkeypatch):
    calls=[]; original=torch.load
    def load(*a,**kw): calls.append(a[0]); return original(*a,**kw)
    monkeypatch.setattr(torch,"load",load)
    agent=trained_agent()
    def forbidden(*a,**kw): raise AssertionError("Inference accessed external I/O")
    import builtins, socket
    monkeypatch.setattr(builtins,"open",forbidden)
    monkeypatch.setattr(socket,"socket",forbidden)
    for _ in range(4): assert action_space().contains(agent.act(official_observation()))
    agent.reset(); agent.act(official_observation())
    assert len(calls)==1


def test_memory_reset_and_independent_instances():
    a,b=trained_agent(),trained_agent(); obs=official_observation()
    original=b.hidden.clone(); first=a.act(obs)
    assert torch.equal(original,b.hidden)
    if a.model.recurrent: assert not torch.equal(original,a.hidden)
    for _ in range(512): a.act(obs)
    assert torch.isfinite(a.hidden).all()
    a.reset()
    assert not a.hidden.any() and not a.previous_action.any()
    assert np.array_equal(a.act(obs),first)


def test_real_episode_without_invalid_actions():
    agent=trained_agent(); env=make_env(ROOT/"maps/two_rooms.wad",episode_ticks=100)
    try:
        obs,_=env.reset(seed=11)
        for _ in range(30):
            action=agent.act(obs); assert env.action_space.contains(action)
            obs,_,term,trunc,_=env.step(action)
            if term or trunc: break
        assert term or trunc
    finally: env.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA execution not verified: local PyTorch is the CPU build")
def test_cuda_agent():
    from submission.agent import DoomAgent
    agent=DoomAgent(observation_space(),action_space(),"cuda")
    assert action_space().contains(agent.act(official_observation()))
