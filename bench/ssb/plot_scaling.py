# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The scaling graphic: 13 official SSB queries vs data size (DuckDB, torbles CPU, torbles GPU) and vs CPU threads.
    python plot_scaling.py results/scaling/scaling.json out.png [--title-note "..."]
Needs matplotlib. Writes a PNG (and an SVG next to it)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

S = json.load(open(sys.argv[1])); out = sys.argv[2]
note = sys.argv[sys.argv.index("--title-note") + 1] if "--title-note" in sys.argv else ""
# palette: validated categorical slots 1-3 (light surface), text tokens, recessive grid
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
GPU, CPU, DUCK = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

def ms_fmt(v, _=None):
    if v >= 1000: return f"{v / 1000:g} s"
    return f"{v:g} ms"
def val(v):                                            # a measured total, for labels
    return f"{v / 1000:.2f} s" if v >= 1000 else (f"{v:.0f} ms" if v >= 20 else f"{v:.1f} ms")
def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.grid(True, which="major", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
def line(ax, x, y, color, lw=2.4, z=3, ms=7, dashed=False):
    ax.plot(x, y, color=color, lw=lw, zorder=z, solid_capstyle="round", ls=(0, (4, 3)) if dashed else "-")
    if not dashed: ax.plot(x, y, "o", ms=ms, color=color, mec=SURF, mew=1.8, zorder=z + 1)
def end_labels(ax, items, gap_pt=30):
    """direct labels at the line ends, pushed apart vertically where they would overlap: items = (x, y, text)"""
    fig = ax.figure; fig.canvas.draw(); dpi = fig.dpi
    pts = [(ax.transData.transform((x, y))[1] * 72 / dpi, x, y, t) for x, y, t in items]
    order = sorted(range(len(pts)), key=lambda i: pts[i][0]); placed = {}
    prev = ax.transAxes.transform((0, 0))[1] * 72 / dpi + gap_pt * 0.45 - gap_pt     # lowest label stays above the axis
    for i in order:                                    # bottom to top: keep at least gap_pt between labels
        want = pts[i][0]; at = max(want, prev + gap_pt); placed[i] = at; prev = at
    for i, (yp, x, y, t) in enumerate(pts):
        moved = abs(placed[i] - yp) > 6                # displaced labels get a thin leader to their point
        ax.annotate(t, (x, y), xytext=(12, placed[i] - yp), textcoords="offset points", va="center", ha="left",
                    fontsize=11.5, color=INK, fontweight="bold", linespacing=1.25, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=4, shrinkB=5) if moved else None)

fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=0.065, right=0.955, top=0.76, bottom=0.15, wspace=0.42)

# ---- left: data size (linear scales: the curves as they are)
sc = sorted(S["scale"], key=lambda r: r["rows"])
rows = [r["rows"] / 1e6 for r in sc]
duck, cpu, gpu = ([r[k] for r in sc] for k in ("duckdb_ms", "torbles_cpu_ms", "torbles_gpu_ms"))
ax = fig.add_subplot(gs[0]); style(ax)
line(ax, rows, duck, DUCK); line(ax, rows, cpu, CPU); line(ax, rows, gpu, GPU)
ax.set_xlim(0, rows[-1] * 1.42); ax.set_ylim(0, max(duck) * 1.08)
step = 50 if rows[-1] > 150 else (20 if rows[-1] > 60 else 10)
ax.xaxis.set_major_locator(FixedLocator([v for v in range(0, int(rows[-1]) + step, step) if v <= rows[-1] * 1.05]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:,.0f}M"))
ax.yaxis.set_major_formatter(FuncFormatter(ms_fmt))
ax.set_xlabel("fact rows   (" + ", ".join(f"SF{r['sf']}" for r in sc) + ")", labelpad=8)
ax.set_title("Time for all 13 queries as the data grows", loc="left", fontsize=14, fontweight="bold", color=INK, pad=12)
end_labels(ax, [(rows[-1], duck[-1], f"DuckDB, 24 threads\n{val(duck[-1])}"),
                (rows[-1], cpu[-1], f"torbles, 24 CPU threads\n{val(cpu[-1])}"),
                (rows[-1], gpu[-1], f"torbles, GPU\n{val(gpu[-1])}")], gap_pt=34)
# the speed-up at the largest scale: a bracket from DuckDB down to the GPU, left of the last points
xb = rows[-1] - (rows[-1] - (rows[-2] if len(rows) > 1 else 0)) * 0.18
ax.annotate("", xy=(xb, gpu[-1]), xytext=(xb, duck[-1]),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3, shrinkA=2, shrinkB=2), zorder=2)
ax.annotate(f"{duck[-1] / gpu[-1]:.0f}x\nfaster", (xb, (duck[-1] + cpu[-1]) / 2), xytext=(-10, 0), textcoords="offset points",
            ha="right", va="center", fontsize=20, fontweight="bold", color=INK, linespacing=1.0)

# ---- right: threads at SF10
T = S.get("threads")
ax2 = fig.add_subplot(gs[1]); style(ax2)
if T:
    th = [p["threads"] for p in T["points"]]
    d2, c2 = [p["duckdb_ms"] for p in T["points"]], [p["torbles_cpu_ms"] for p in T["points"]]
    line(ax2, th, d2, DUCK); line(ax2, th, c2, CPU)
    g10 = next((r for r in S["scale"] if r["sf"] == T["sf"]), None)
    items = [(th[-1], d2[-1], f"DuckDB\n{val(d2[-1])}"), (th[-1], c2[-1], f"torbles CPU\n{val(c2[-1])}")]
    if g10:
        line(ax2, [th[0], th[-1]], [g10["torbles_gpu_ms"]] * 2, GPU, dashed=True, z=2)
        items.append((th[-1], g10["torbles_gpu_ms"], f"torbles GPU\n{val(g10['torbles_gpu_ms'])}"))
    ax2.set_xlim(0, th[-1] * 1.55); ax2.set_ylim(0, max(d2) * 1.08)
    ax2.xaxis.set_major_locator(FixedLocator(th)); ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(ms_fmt))
    ax2.set_xlabel("CPU threads", labelpad=8)
    ax2.set_title(f"Same queries at SF{T['sf']} ({T['rows'] / 1e6:.0f}M rows), more cores", loc="left", fontsize=14,
                  fontweight="bold", color=INK, pad=12)
    end_labels(ax2, items, gap_pt=34)

# ---- headline
big = sc[-1]
fig.text(0.065, 0.925, "torbles 0.1.0 on the official Star Schema Benchmark", fontsize=24, fontweight="bold", color=INK)
fig.text(0.065, 0.868, f"All 13 queries on {big['rows'] / 1e6:,.0f} million rows in {val(big['torbles_gpu_ms'])} on a laptop GPU: "
         f"{big['duckdb_ms'] / big['torbles_gpu_ms']:.0f}x DuckDB on 24 threads. Every result verified against DuckDB.",
         fontsize=13.5, color=INK2)
inmem = all(r.get("duckdb_in_memory") for r in sc)
fig.text(0.065, 0.828, "End to end (plan, execute, return), best of 3; both engines answer from memory"
         + (" (DuckDB on an in-memory copy of its database)" if inmem else "") + ". " + note, fontsize=11, color=MUTED)
fig.text(0.065, 0.025, "Core Ultra 9 290HX (24 cores) · RTX 5090 Laptop GPU · WSL2 · DuckDB "
         + S.get("duckdb_version", "2.0") + " · official ssb-dbgen data and queries · thread panel: a separate sweep at SF10 · torbles is Apache-2.0",
         fontsize=10, color=MUTED)
fig.savefig(out, facecolor=SURF)
fig.savefig(out.rsplit(".", 1)[0] + ".svg", facecolor=SURF)
print("wrote", out)
