"""Sample underperforming skills while retaining practice of mastered levels."""
from __future__ import annotations

from collections import defaultdict, deque


class AdaptiveCurriculum:
    def __init__(self, rows, *, threshold=.72, window=40, minimum=12, probe=.12):
        if not 0 < threshold <= 1 or not 0 <= probe < 1 or not 0 < minimum <= window:
            raise ValueError("Invalid adaptive curriculum thresholds")
        self.groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            self.groups[row["skill"]][row["level"]].append(row)
        if not self.groups or any(set(levels) != set(range(len(levels))) for levels in self.groups.values()):
            raise ValueError("Each curriculum skill needs contiguous levels starting at zero")
        self.threshold, self.minimum, self.probe = threshold, minimum, probe
        self.history = {skill: [deque(maxlen=window) for _ in levels] for skill, levels in self.groups.items()}
        self.frontier = dict.fromkeys(self.groups, 0)
        self.validation = {}

    def rates(self, skill, level):
        results = self.history[skill][level]
        if not results:
            return 0., 0.
        return tuple(sum(result[i] for result in results) / len(results) for i in range(2))

    def observe(self, skill, level, result):
        self.history[skill][level].append((bool(result["drill_success"]), float(result["coverage"])))
        self.advance(skill)

    def advance(self, skill):
        level = self.frontier[skill]
        outcomes = self.history[skill][level]
        observed = self.validation.get(skill, {}).get(str(level))
        supported = observed is None or observed["episodes"] < self.minimum or observed["success"] >= self.threshold
        if (level + 1 < len(self.groups[skill]) and len(outcomes) >= self.minimum and supported
                and self.rates(skill, level)[0] >= self.threshold):
            self.frontier[skill] += 1

    def set_validation(self, validation):
        self.validation = validation
        for skill in self.groups:
            self.advance(skill)

    def choose(self, rng):
        skills = list(self.groups)
        rates = [self.rates(skill, self.frontier[skill]) for skill in skills]
        weights = [.4 + (1 - success) + .25 * (1 - coverage) for success, coverage in rates]
        skill, = rng.choices(skills, weights=weights)
        limit = self.frontier[skill]
        if limit + 1 < len(self.groups[skill]) and rng.random() < self.probe:
            level = limit + 1
        else:
            eligible = range(limit + 1)
            weights = [.35 + (1 - self.rates(skill, level)[0]) for level in eligible]
            level, = rng.choices(eligible, weights=weights)
        return rng.choice(self.groups[skill][level])

    def snapshot(self):
        return {skill: {"frontier": self.frontier[skill], "levels": [
            {"episodes": len(outcomes), "success": self.rates(skill, level)[0],
             "coverage": self.rates(skill, level)[1]} for level, outcomes in enumerate(history)]}
                for skill, history in self.history.items()}

    def state_dict(self):
        return dict(frontier=self.frontier, validation=self.validation,
                    history={skill: [list(entries) for entries in levels] for skill, levels in self.history.items()})

    def restore(self, state):
        if set(state["frontier"]) != set(self.frontier) or any(
            len(state["history"][skill]) != len(levels) for skill, levels in self.history.items()
        ):
            raise ValueError("Curriculum differs from the resumed checkpoint")
        self.frontier.update(state["frontier"])
        self.validation = state["validation"]
        for skill, levels in self.history.items():
            for history, saved in zip(levels, state["history"][skill]):
                history.extend(saved)
