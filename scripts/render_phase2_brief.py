"""Render the two-page project brief from measured Phase 2 results (requires reportlab)."""
import argparse
import json
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def line(pdf, x, y, text, size=11, bold=False, color="#253142"):
    pdf.setFillColor(colors.HexColor(color))
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    pdf.drawString(x, y, text)


def prose(pdf, y, text):
    for part in wrap(text, 94, break_long_words=False):
        line(pdf, 44, y, part, 10.5)
        y -= 15
    return y - 8


def title(pdf, heading, subheading):
    pdf.setFillColor(colors.HexColor("#16263a"))
    pdf.rect(0, A4[1]-112, A4[0], 112, fill=1, stroke=0)
    line(pdf, 44, A4[1]-53, heading, 28, True, "#ffffff")
    line(pdf, 44, A4[1]-81, subheading, 12, False, "#a8ddcd")
    return A4[1]-143


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--release", required=True)
    args = ap.parse_args()
    report = json.loads(args.comparison.read_text())
    stats = json.loads(args.summary.read_text())
    baseline, challenger = report["candidates"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.out), pagesize=A4, pageCompression=1)
    pdf.setTitle("DOOM NEXUS - Phase 2 project brief")
    y = title(pdf, "DOOM NEXUS", "Yugaantar DevForge Track 1 | Phase 2")
    y = prose(pdf, y, "A learned ViZDoom agent uses local rays, visible entities and vitals to act under partial observability. The same policy code operates with either a compact MLP or a recurrent GRU.")
    line(pdf, 44, y, "Measured comparison", 15, True)
    y = prose(pdf, y-22, f"Split {report['split']}. Each score is the local, tier-weighted proxy. The challenger completed {stats['steps']:,} cumulative training decisions and {stats['updates']:,} PPO updates.")
    headers = ("Tier", "Baseline exits", "Challenger exits", "Keys / ep", "Unlocks / ep")
    columns = (44, 150, 247, 365, 458)
    for x, h in zip(columns, headers):
        line(pdf, x, y, h, 9, True)
    for stage in ("combat", "exploration", "keys", "gauntlet"):
        a, b = baseline["tiers"][stage], challenger["tiers"][stage]
        y -= 21
        values = (stage.title(), f"{a['exit']:.0%}", f"{b['exit']:.0%}",
                  f"{a['keys']:.2f} / {b['keys']:.2f}", f"{a['locks']:.2f} / {b['locks']:.2f}")
        for x, value in zip(columns, values):
            line(pdf, x, y, value, 9)
    y = prose(pdf, y-32, f"Weighted proxy: {baseline['tier_weighted_proxy']:.2f} baseline, {challenger['tier_weighted_proxy']:.2f} challenger. Release: {args.release}.")
    y = prose(pdf, y, "Original baseline failure replay: two exploration timeouts, four key scenario timeouts and four gauntlet deaths. The traces showed repeated visits, keys missed or acquired without a subsequent unlock, and combat damage.")
    prose(pdf, y, "The benchmark uses the project map generator. Organizer hidden scores and finals qualification have not been measured.")
    pdf.showPage()
    y = title(pdf, "From perception to action", "Implementation, evidence and live demonstration")
    for heading, description in (
        ("Spatial encoder", "Directional convolution reads adjacent rays and learned hit-type embeddings. A shared encoder pools valid visible entities and combines them with vitals and the previous action."),
        ("Recurrent memory", "A GRU carries a compact vector between decisions. In a matched validation run, zeroing that vector each decision reduced the weighted proxy from 29.62 to 13.66 and removed observed Tier 3 unlocks."),
        ("Learning and resource controls", "Custom PPO trains with engine rollouts. Batched updates raised short-run four-engine CPU throughput to 339 decisions/sec. Each completed update saves a resumable optimizer and model checkpoint."),
        ("Legal deployment", "The policy acts from official observations and learned weights. Engine state is restricted to reward shaping and diagnostics. Final inference uses the organizer interface with a one-thread limit."),
        ("Demonstration", "The offline dashboard synchronizes real ViZDoom replays. It visualizes health, ammunition, rays, entities, distribution probabilities and progression. Tier tables and reward curves come from recorded evaluation and training.")
    ):
        line(pdf, 44, y, heading, 14, True)
        y = prose(pdf, y-20, description) - 9
    pdf.save()


if __name__ == "__main__":
    main()
