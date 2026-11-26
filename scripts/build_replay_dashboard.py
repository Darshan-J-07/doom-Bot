"""Build a self-contained offline dashboard from real synchronized engine recordings."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

# pygame-ce is already a ViZDoom dependency in the development environment.
import pygame


def export_episode(source, destination, label):
    video = np.load(source.with_suffix(".npy"), mmap_mode="r")
    if video.shape[1:] != (120, 160, 3) or video.dtype != np.uint8:
        raise ValueError(f"Unexpected ViZDoom frame data in {source}")
    with gzip.open(source.with_suffix(".jsonl.gz"), "rt") as file:
        rows = [json.loads(line) for line in file]
    if len(rows) != len(video):
        raise ValueError(f"Trajectory and frame counts differ in {source}")
    cols = 10
    chunks = 100
    sprites = []
    for start in range(0, len(video), chunks):
        batch = video[start:start + chunks]
        atlas = pygame.Surface((cols * 160, ((len(batch) + cols - 1) // cols) * 120))
        for i, frame in enumerate(batch):
            pixels = pygame.image.frombuffer(frame.tobytes(), (160, 120), "RGB")
            atlas.blit(pixels, ((i % cols) * 160, (i // cols) * 120))
        asset = f"{label}-{len(sprites)}.png"
        pygame.image.save(atlas, destination / asset)
        sprites.append(asset)
    return dict(sprites=sprites, rows=rows, columns=cols, frames_per_sprite=chunks)


def learning_series(path):
    series = []
    latest = []
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            latest.extend(e["shaped_return"] for e in row["episodes"])
            latest = latest[-30:]
            if latest:
                series.append([row["steps"], sum(latest) / len(latest)])
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="recording stem")
    parser.add_argument("--candidate", required=True, type=Path, help="recording stem")
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--learning", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline, candidate = [
        export_episode(source, args.out, label)
        for source, label in ((args.baseline, "baseline"), (args.candidate, "candidate"))
    ]
    report = json.loads(args.comparison.read_text())
    comparisons = [dict(path=r["checkpoint"], score=r["tier_weighted_proxy"], tiers=r["tiers"])
                   for r in report["candidates"]]
    data = dict(baseline=baseline, candidate=candidate, scenario=args.baseline.name,
                comparisons=comparisons, split=report["split"],
                learning=learning_series(args.learning) if args.learning else [])
    (args.out / "data.js").write_text("window.replayData = " + json.dumps(data, separators=(",", ":")) + ";\n")
    shutil.copyfile(Path(__file__).with_name("replay_dashboard.html"), args.out / "index.html")


if __name__ == "__main__":
    main()
