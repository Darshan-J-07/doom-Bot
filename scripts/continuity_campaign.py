"""Resume a checkpoint-backed learning budget under a persistent Ubuntu user service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from training.learning_meter import write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "runs/continuity-campaign-20261127"
SERVICE = "doom-nexus-continuity.service"
TARGET = 10_800
PHASE_SECONDS = 2700
BLOCK_SECONDS = 450
PHASES = (
    ("Post-lock progression", ("post-lock-bootstrap", "post-lock-varied", "post-lock-gentle")),
    ("Skill retention and imitation", ("bc-retention", "retention-distill", "retention-low")),
    ("Spatial intelligence and exploration", ("memory-control", "gru", "occupancy")),
    ("Full-level mastery", ("mastery", "count", "rnd")),
)
FALLBACKS = (
    ("post-lock-gentle", "retention-safe"), ("retention-anchor", "retention-safe"),
    ("rnd", "retention-safe"), ("retention-anchor", "retention-safe"),
)
PYTHON_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def note(run, message):
    with (run / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp()} {message}\n")


def module(name, *options):
    return [sys.executable, "-m", name, *map(str, options)]


def checkpoint_root(run, block):
    path = Path(block["output"])
    if not path.resolve().is_relative_to(run / "training"):
        raise ValueError("A training block must live inside this campaign.")
    return path


def latest_row(path):
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 131072))
        rows = stream.read().splitlines()
    for line in reversed(rows):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def service_pid(unit):
    proc = subprocess.run(["systemctl", "--user", "show", unit, "--property=MainPID", "--value"],
                          capture_output=True, text=True, timeout=6)
    return int(proc.stdout.strip() or 0) if proc.returncode == 0 else 0


def child_cleanup(process, *, grace=24):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    # The ViZDoom processes are created inside this job's process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class Campaign:
    def __init__(self, run, *, visualize=False):
        self.run = run
        self.clock = read(run / "clock.json")
        if self.clock["active_training_target_seconds"] != TARGET:
            raise ValueError("This campaign requires 10,800 active training seconds.")
        self.status_path = run / "status.json"
        self.state = read(self.status_path)
        self.start_utc = datetime.fromisoformat(self.clock["start_utc"])
        initial = read(run / "provenance/initial-identities.json")
        self.warm = (ROOT / initial["starting_optimizer"]["path"]).resolve()
        self.state.setdefault("evaluation_queue", [])
        self.state.setdefault("gates", {})
        self.state.setdefault("evaluation_seconds", 0.)
        self.state.setdefault("best_validated_checkpoint", str(self.warm))
        self.state.setdefault("best_rank", None)
        self.state.setdefault("rejected_profiles", [])
        self.state.setdefault("supervised_blocks", [])
        self.state["controller_pid"] = os.getpid()
        self.learner = None
        self.evaluator = None
        self.pause = False
        self.visualize = visualize
        self.last_note = time.monotonic()
        self.min_ram = 1200
        self.last_health_check = 0.
        self.resource_problem = None

    @property
    def blocks(self):
        return self.state["blocks"]

    def active(self):
        return (sum(b.get("active_training_ns", 0) for b in self.blocks)
                + sum(b.get("active_training_ns", 0) for b in self.state["supervised_blocks"])) / 1e9

    def update_receipt(self, block):
        directory = checkpoint_root(self.run, block)
        file = directory / "training-progress.json"
        if not file.is_file():
            return
        receipt = read(file)
        if receipt["meter_id"] != block["meter_id"]:
            raise ValueError(f"Unexpected accounting identity for {block['label']}")
        ns = receipt["active_training_ns"]
        if ns < block.get("active_training_ns", 0):
            return
        if ns and not Path(receipt["checkpoint"]).is_file():
            raise ValueError(f"Completed training receipt lost its checkpoint for {block['label']}")
        if not isinstance(ns, int) or ns < 0 or (ns and not receipt["added_updates"]):
            raise ValueError(f"Invalid learning receipt for {block['label']}")
        for key in ("active_training_ns", "active_optimization_ns", "active_rollout_ns", "checkpoint",
                    "checkpoint_steps", "checkpoint_updates", "added_decisions", "added_updates"):
            block[key] = receipt[key]
        block["active_decisions_per_second"] = receipt["added_decisions"] * 1e9 / ns if ns else 0.
        telemetry = latest_row(directory / "learning.jsonl")
        block["latest_update_metrics"] = {key: telemetry[key] for key in
            ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm", "anchor_kl",
             "learning_rate", "replay_log_prob_max_error") if key in telemetry and telemetry[key] is not None}
        episodes = telemetry.get("episodes", [])
        if episodes:
            block["latest_rollout"] = dict(episodes=len(episodes), exits=sum(bool(ep.get("exit")) for ep in episodes),
                deaths=sum(bool(ep.get("dead")) for ep in episodes),
                shaped_reward=sum(ep.get("shaped_return", 0.) for ep in episodes) / len(episodes),
                mean_health=sum(ep.get("health", 0.) for ep in episodes) / len(episodes))

    def refresh(self, block=None, pid=None, *, mode=None):
        if block:
            self.update_receipt(block)
        active = self.active()
        elapsed = (datetime.now(timezone.utc) - self.start_utc).total_seconds()
        if block and block.get("checkpoint") and block.get("added_updates", 0):
            self.state["last_successful_checkpoint"] = block["checkpoint"]
        self.state.update(active_training_seconds=active, remaining_training_seconds=max(0., TARGET - active),
            training_decisions=sum(b.get("added_decisions", 0) for b in self.blocks),
            ppo_updates=sum(b.get("added_updates", 0) for b in self.blocks),
            supervised_updates=sum(b.get("supervised_updates", 0) for b in self.state["supervised_blocks"]),
            total_elapsed_wall_seconds=elapsed, nontraining_wall_seconds=max(0., elapsed - active),
            training_pid=pid, updated_utc=stamp(), state=mode or ("training" if pid else "between jobs"),
            current_experiment=block["label"] if block else None,
            phase=block["phase"] if block else self.state.get("phase"),
            current_checkpoint=block.get("numbered_checkpoint", block.get("checkpoint")) if block else None,
            decisions_per_second=block.get("active_decisions_per_second", 0.) if block else 0.,
            evaluation_pid=self.evaluator["process"].pid if self.evaluator else None,
            next_evaluation_milestone=block["active_limit_seconds"] if block else None,
            evaluation_age_seconds=(time.monotonic() - self.evaluator["start"]) if self.evaluator else 0.)
        write_json(self.status_path, self.state)

    def healthy(self):
        now = time.monotonic()
        if now - self.last_health_check < 8:
            return self.resource_problem is None
        self.last_health_check = now
        ram = psutil.virtual_memory().available / 2**20
        disk = shutil.disk_usage(self.run).free / 2**30
        temperatures = []
        for sensor in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                temp = float(sensor.read_text().strip()) / 1000
                if 0 < temp < 150:
                    temperatures.append(temp)
            except (OSError, ValueError):
                pass
        self.state["resources"] = dict(available_ram_mib=ram, disk_free_gib=disk,
                                       load_average=os.getloadavg(), thermal_celsius=temperatures)
        if disk < 1:
            self.resource_problem = "Less than 1 GiB free in the campaign filesystem."
        elif ram < self.min_ram:
            self.resource_problem = f"Available RAM fell to {ram:.0f} MiB."
        elif temperatures and max(temperatures) >= 94:
            self.resource_problem = "A readable system thermal sensor reached 94 °C."
        else:
            self.resource_problem = None
        return self.resource_problem is None

    def stopping(self):
        return self.pause or (self.run / "STOP").exists() or not self.healthy()

    def launch(self, command, logfile):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("ab") as log:
            log.write(("$ " + subprocess.list2cmdline(command) + "\n").encode())
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **PYTHON_ENV},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process

    def queue_evaluation(self, block):
        item = dict(label=block["label"], checkpoint=block["numbered_checkpoint"], pair=block["pair"],
                    seed=block["seed"], status="queued", attempts=0)
        queue = self.state["evaluation_queue"]
        if not any(old["label"] == item["label"] for old in queue):
            queue.append(item)

    def finish_block(self, block):
        self.update_receipt(block)
        directory = checkpoint_root(self.run, block)
        latest = directory / "latest.pt"
        if block.get("added_updates", 0) and latest.is_file():
            numbered = directory / f"step-{block['checkpoint_steps']}.pt"
            if not numbered.is_file():
                os.link(latest, numbered)
            block["numbered_checkpoint"] = str(numbered)
            self.state["last_successful_checkpoint"] = str(numbered)
            self.queue_evaluation(block)
        block["state"] = "complete" if block.get("active_training_ns", 0) >= 1e9 * block["active_limit_seconds"] else "partial"
        note(self.run, f"{block['label']} {block['state']} with {block.get('active_training_ns',0)/1e9:.1f} active seconds, "
                       f"{block.get('added_decisions',0):,} decisions and {block.get('added_updates',0)} PPO updates.")
        self.refresh()

    def bootstrap(self):
        if self.blocks:
            block = self.blocks[0]
        else:
            block = read(self.run / "bootstrap.json")
            block.update(pair="p1-post-lock-bootstrap", state="running", attempts=1, phase_index=0)
            self.blocks.append(block)
            self.refresh(block)
        if block.get("unit") is None or block["state"] != "running":
            return
        while (pid := service_pid(block["unit"])):
            if self.stopping():
                subprocess.run(["systemctl", "--user", "stop", block["unit"]], timeout=65, check=False)
                break
            self.refresh(block, pid)
            time.sleep(2)
        self.finish_block(block)

    def promote_development_pairs(self):
        from scripts.adaptive_campaign_eval import final_rank

        for pair in dict.fromkeys(b["pair"] for b in self.blocks):
            if pair in self.state["gates"]:
                continue
            members = [x for x in self.state["evaluation_queue"] if x["pair"] == pair]
            if len(members) < 2 or any(x["status"] not in {"complete", "rejected"} for x in members):
                continue
            labels = [x["label"] for x in members]
            if any(x["status"] == "rejected" for x in members):
                self.state["gates"][pair] = {"passed":False,"source":"evaluation runtime failure","labels":labels}
                continue
            from scripts.adaptive_campaign_eval import decide
            decide(self.run, labels, milestone=None, name=pair, late=True, publish_status=False)
            path = self.run / "evaluation" / f"gate-{pair}.json"
            data = read(path)
            self.state["gates"][pair] = dict(passed=data["passed"], path=str(path), labels=labels)
            profiles = {b["profile"] for b in self.blocks if b["pair"] == pair}
            if not data["passed"]:
                note(self.run, f"{pair} rejected for advancement. Training continues with the next supported profile.")
                severe = all(row["development"]["familiar"]["overall"]["exits"] < 20
                             or row["development"]["familiar"]["by_tier"]["gauntlet"]["deaths"] > 25
                             for row in data["results"].values())
                if severe:
                    self.state["rejected_profiles"] = sorted(set(self.state["rejected_profiles"]) | profiles)
                continue
            candidate = max((label for label, passed in data["qualified"].items() if passed),
                            key=lambda label: final_rank(data["results"][label]))
            rank = final_rank(data["results"][candidate])
            if self.state["best_rank"] is None or rank > tuple(self.state["best_rank"]):
                artifact = next(x for x in members if x["label"] == candidate)
                self.state.update(best_validated_checkpoint=artifact["checkpoint"], best_rank=list(rank),
                                  best_validated_label=candidate, best_gate=str(path))
                note(self.run, f"{candidate} is the strongest passing development checkpoint. Published weights stay frozen.")

    def evaluation_tick(self):
        if self.evaluator:
            job = self.evaluator
            process = job["process"]
            code = process.poll()
            if code is None and time.monotonic() - job["start"] < 840:
                return
            if code is None:
                child_cleanup(process)
                code = process.returncode
            elapsed = time.monotonic() - job["start"]
            self.state["evaluation_seconds"] += elapsed
            item = job["item"]
            item["wall_seconds"] = item.get("wall_seconds", 0.) + elapsed
            success = code == 0 and (self.run / "evaluation" / f"{item['label']}.json").is_file()
            item["status"] = "complete" if success else ("queued" if item["attempts"] < 2 else "rejected")
            if success:
                data = read(self.run / "evaluation" / f"{item['label']}.json")
                self.state["latest_development"] = dict(label=item["label"],
                    far_switch_exits=data["switch"]["overall"]["exits_from_outside_range"],
                    far_range_entries=data["switch"]["overall"]["entered_range_from_outside"],
                    familiar_exits=data["familiar"]["overall"]["exits"],
                    gauntlet_deaths=data["familiar"]["by_tier"]["gauntlet"]["deaths"])
            note(self.run, f"Development evaluation {item['label']} {item['status']} in {elapsed:.1f} s.")
            self.evaluator = None
            self.promote_development_pairs()
        if not self.stopping():
            next_item = next((item for item in self.state["evaluation_queue"] if item["status"] == "queued"), None)
            if next_item:
                next_item["attempts"] += 1
                next_item["status"] = "evaluating"
                name = next_item["label"]
                args = module("scripts.continuity_campaign", "evaluate", "--out", self.run, "--label", name,
                              "--checkpoint", next_item["checkpoint"])
                process = self.launch(args, self.run / "jobs" / f"evaluation-{name}.log")
                self.evaluator = dict(item=next_item, process=process, start=time.monotonic())

    def choose_block(self):
        if self.blocks and self.blocks[-1]["state"] == "running":
            block = self.blocks[-1]
            self.update_receipt(block)
            return block
        phase_index = min(3, int(self.active() // PHASE_SECONDS))
        phase, recipes = PHASES[phase_index]
        members = [b for b in self.blocks if b["phase_index"] == phase_index]
        count = len(members)
        seed = (59, 83)[count % 2]
        if seed == 83 and members:
            first = members[-1]
            profile, pair, parent = first["profile"], first["pair"], first["parent"]
            label = first["label"].removesuffix("-59") + "-83"
        else:
            index = count // 2
            fallback = FALLBACKS[phase_index]
            profile = recipes[index] if index < len(recipes) else fallback[(index - len(recipes)) % len(fallback)]
            if profile in self.state["rejected_profiles"]:
                profile = next((p for p in (*recipes, *fallback) if p not in self.state["rejected_profiles"]),
                               "retention-safe")
            pair = f"p{phase_index+1}-{index+1:02d}-{profile}"
            parent = self.state["best_validated_checkpoint"]
            label_of_parent = self.state.get("best_validated_label")
            if label_of_parent and profile != "mastery":
                cfg = read(self.run / "exports" / label_of_parent / "export.json")["model_config"]
                if cfg.get("recurrent"):
                    parent = str(self.warm)
            label = pair + "-59"
        duration = min(BLOCK_SECONDS, (phase_index+1)*PHASE_SECONDS - self.active(), TARGET - self.active())
        block = dict(label=label, pair=pair, phase=phase, phase_index=phase_index, profile=profile, seed=seed,
                     parent=parent, meter_id=label, output=str(self.run / "training" / label), attempts=0,
                     active_limit_seconds=max(.001, duration), state="running", created_utc=stamp())
        self.blocks.append(block)
        self.refresh(block)
        return block

    def training_command(self, block, source):
        prof = block["profile"]
        budget = block["active_limit_seconds"]
        visible = ["--visualize", "--speed", "0"] if self.visualize else []
        if prof.startswith("post-lock"):
            args = module("scripts.continuity_post_lock", "learner", "--out", block["output"], "--resume", source,
                          "--seed", block["seed"], "--active-seconds", budget, "--meter-id", block["meter_id"],
                          "--seconds", max(120, 1.9 * budget + 30), "--decisions", 500000, "--stop-file", self.run / "STOP")
            if prof != "post-lock-bootstrap":
                args += ["--map-root", str(self.run / "post-lock-maps")]
            if prof == "post-lock-gentle":
                args += ["--lr", "0.000135"]
            return args + visible
        settings = module("scripts.continuity_learning", "learner", "--out", self.run,
                          "--label", Path(block["output"]).name, "--source", source,
                          "--profile", "retention-low" if prof == "bc-retention" else prof,
                          "--seed", block["seed"], "--meter-id", block["meter_id"],
                          "--active-seconds", budget, *visible)
        if block["phase_index"] == 3 and (self.run / "mastery-training" / "manifest.json").is_file():
            settings += ["--map-root", str(self.run / "mastery-training")]
        return settings

    def imitation_start(self, block):
        label = block["label"] + "-bc"
        training = self.run / "training" / label
        progress = training / "training-progress.json"
        if not progress.is_file():
            cmd = module("scripts.continuity_learning", "clone", "--out", self.run, "--label", label,
                         "--source", block["parent"], "--seed", block["seed"], "--max-updates", 240)
            self.learner = self.launch(cmd, self.run / "jobs" / f"supervised-{label}.log")
            begin = time.monotonic()
            note(self.run, f"Started supervised imitation for {block['label']} (PID {self.learner.pid}).")
            while self.learner.poll() is None:
                self.evaluation_tick()
                self.refresh(block, self.learner.pid, mode="supervised learning")
                if self.stopping() or time.monotonic() - begin > 170:
                    child_cleanup(self.learner)
                    break
                time.sleep(1)
            self.learner = None
        if progress.is_file():
            receipt = read(progress)
            checkpoint = Path(receipt["checkpoint"])
            if checkpoint.is_file() and receipt["added_supervised_updates"] > 0:
                credit = dict(label=label, checkpoint=str(checkpoint), phase=block["phase"], command="clone",
                              active_training_ns=round(1e9 * receipt["active_training_seconds"]),
                              supervised_updates=receipt["added_supervised_updates"])
                if not any(row["label"] == label for row in self.state["supervised_blocks"]):
                    self.state["supervised_blocks"].append(credit)
                    self.refresh(block)
                return str(checkpoint)
        note(self.run, f"Supervised start for {label} did not save a usable checkpoint. The PPO arm restores the validated parent.")
        return block["parent"]

    def train_block(self, block):
        source = block.get("checkpoint") or block["parent"]
        if not block.get("added_updates") and block["profile"] == "bc-retention":
            source = self.imitation_start(block)
        if self.stopping():
            return
        if block["attempts"]:
            block["output"] = str(self.run / "training" / f"{block['label']}-try-{block['attempts']+1}")
        block["attempts"] += 1
        args = self.training_command(block, source)
        block.setdefault("commands", []).append(args)
        self.refresh(block)
        self.learner = self.launch(args, self.run / "jobs" / f"training-{Path(block['output']).name}.log")
        note(self.run, f"Started {block['label']} PID {self.learner.pid} from {source}.")
        previous_updates, changed = -1, time.monotonic()
        start = changed
        cap = max(120., 2.05 * block["active_limit_seconds"] + 90)
        reason = None
        while self.learner.poll() is None:
            self.evaluation_tick()
            self.refresh(block, self.learner.pid)
            recent = block.get("added_updates", 0)
            if recent != previous_updates:
                previous_updates, changed = recent, time.monotonic()
            now = time.monotonic()
            if self.stopping():
                reason = "operator pause" if not self.resource_problem else self.resource_problem
            elif now - changed >= 100:
                reason = "no completed PPO update for 100 seconds"
            elif now - start >= cap:
                reason = "per-child monotonic wall limit"
            elif any(not math.isfinite(v) for v in block.get("latest_update_metrics", {}).values()):
                reason = "nonfinite PPO metrics"
            if reason:
                child_cleanup(self.learner)
                break
            if now - self.last_note > 60:
                note(self.run, f"Active training {self.active():.1f}/{TARGET} seconds. {self.state['training_decisions']:,} decisions.")
                self.last_note = now
            time.sleep(2)
        block["return_code"] = self.learner.returncode
        self.learner = None
        block["last_stop_reason"] = reason
        self.update_receipt(block)
        if self.stopping():
            self.refresh(block, mode="paused")
            return
        if block.get("active_training_ns", 0) < 1e9*block["active_limit_seconds"] and block["attempts"] < 2 and (
            block["return_code"] == 0 or block.get("added_updates", 0)) and reason != "nonfinite PPO metrics":
            note(self.run, f"{block['label']} stopped short. One safe resume will use its durable checkpoint.")
            self.refresh(block)
            return
        self.finish_block(block)
        if not block.get("added_updates", 0):
            self.state["failed_zero_progress"] = self.state.get("failed_zero_progress", 0) + 1
            if self.state["failed_zero_progress"] >= 3:
                self.resource_problem = f"Three successive learners saved no update. Inspect {self.run / 'jobs'}."
                self.pause = True
        else:
            self.state["failed_zero_progress"] = 0

    def run_all(self):
        for item in self.state["evaluation_queue"]:
            if item["status"] == "evaluating":
                item["status"] = "queued"
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "pause", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "pause", True))
        self.bootstrap()
        try:
            while self.active() < TARGET and not self.stopping():
                self.evaluation_tick()
                self.train_block(self.choose_block())
            while not self.stopping() and (self.evaluator or any(
                item["status"] == "queued" for item in self.state["evaluation_queue"])):
                self.evaluation_tick()
                self.refresh(mode="final development evaluation")
                time.sleep(2)
            if not self.stopping():
                self.refresh(mode="final audit")
                subprocess.run(module("scripts.continuity_finalize", "--out", self.run), cwd=ROOT,
                               env={**os.environ, **PYTHON_ENV}, check=True)
                final = self.run / "finalized.json"
                if final.is_file():
                    self.state["finalization"] = read(final)
                self.refresh(mode="complete")
                note(self.run, f"Completed {self.active():.2f} active training seconds and finalized the report.")
        finally:
            child_cleanup(self.learner)
            if self.evaluator:
                job = self.evaluator
                child_cleanup(job["process"], grace=4)
                self.state["evaluation_seconds"] += time.monotonic() - job["start"]
                self.evaluator = None
                job["item"]["status"] = "queued"
            if self.stopping():
                self.state["pause_reason"] = self.resource_problem or "operator pause requested"
                self.refresh(mode="paused")
                note(self.run, f"Campaign paused. {self.state['pause_reason']}")


def evaluate(run, label, checkpoint):
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) > 4:
        os.sched_setaffinity(0, set(affinity[-2:]))
    repo = run / "exports" / label
    if not (repo / "export.json").is_file():
        if repo.exists():
            repo.rename(repo.with_name(f"{label}-incomplete-{time.time_ns()}"))
        subprocess.run(module("scripts.export_candidate", "--checkpoint", checkpoint, "--out", repo), check=True)
    for scope in ("switch", "familiar"):
        folder = run / "evaluation" / scope / label
        if folder.is_dir() and not (folder / "summary.json").is_file():
            folder.rename(folder.with_name(f"{label}-interrupted-{time.time_ns()}"))
    from scripts.adaptive_campaign_eval import run as paired_development

    paired_development(run, [(label, repo)], wait=0, scope=None, publish_status=False)
    maps = run / "post-lock-maps"
    target = run / "post-lock-diagnostics" / label
    if (maps / "qa.json").is_file() and not (target / "summary.json").is_file():
        if target.exists():
            target.rename(target.with_name(f"{label}-interrupted-{time.time_ns()}"))
        subprocess.run(module("scripts.continuity_post_lock_eval", "--repo", repo, "--out", target,
                              "--map-root", maps, "--seeds", 0, 1, 2), check=True)


def start_service(run, visualize):
    if service_pid(SERVICE):
        raise SystemExit("The campaign user service is already running.")
    command = ["systemd-run", "--user", f"--unit={SERVICE}", "--collect",
               f"--working-directory={ROOT}", "--property=Restart=on-failure", "--property=RestartSec=6s",
               "--property=KillMode=mixed", "--property=TimeoutStopSec=70s",
               f"--property=StandardOutput=append:{run/'controller.log'}",
               f"--property=StandardError=append:{run/'controller.log'}"]
    command += [f"--setenv={k}={v}" for k, v in PYTHON_ENV.items()]
    if visualize:
        command += ["--setenv=SDL_VIDEODRIVER=x11", f"--setenv=DISPLAY={os.environ.get('DISPLAY', ':0')}"]
    command += module("scripts.continuity_campaign", "run", "--out", run)
    if visualize:
        command += ["--visualize"]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "start", "stop", "resume", "status", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--label")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    run = args.out.resolve(strict=True)
    if args.action == "status":
        state = read(run / "status.json")
        fields = ("state", "phase", "current_experiment", "training_pid", "active_training_seconds",
                  "remaining_training_seconds", "total_elapsed_wall_seconds", "evaluation_seconds",
                  "nontraining_wall_seconds", "training_decisions", "ppo_updates", "decisions_per_second",
                  "current_checkpoint", "last_successful_checkpoint", "best_validated_checkpoint",
                  "next_evaluation_milestone", "latest_development", "pause_reason")
        print(json.dumps({k:state.get(k) for k in fields}, indent=2))
    elif args.action == "stop":
        (run / "STOP").touch()
    elif args.action in {"start", "resume"}:
        if args.action == "resume":
            (run / "STOP").unlink(missing_ok=True)
        start_service(run, args.visualize)
    elif args.action == "evaluate":
        evaluate(run, args.label, args.checkpoint)
    else:
        with (run / ".controller.lock").open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("This campaign already has a controller.") from None
            Campaign(run, visualize=args.visualize).run_all()


if __name__ == "__main__":
    main()
