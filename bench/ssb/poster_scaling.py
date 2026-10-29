# Copyright 2026 Brian Richard RAMOS
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0).
"""The scaling poster: a hand-built SVG (ROYGBIV on a white-gradient ground) from results/scaling/scaling.json, rendered to PNG.
    python poster_scaling.py results/scaling/scaling.json docs/img/ssb_scaling        (writes .svg and .png)
Rendering the PNG needs Playwright with Chromium (pip install playwright; playwright install chromium) and the Inter
font for the intended look; the SVG alone needs neither."""
import json, sys, math, base64, os

S = json.load(open(sys.argv[1])); out = sys.argv[2]
sc = sorted(S["scale"], key=lambda r: r["rows"]); T = S.get("threads")
big = sc[-1]
W, H = 1600, 900
R, O, Y, G, B, I, V = "#ff3b5c", "#ff8a1f", "#ffc21a", "#22cc6a", "#1fa2ff", "#5b6cff", "#c04dff"
BG, CARD, LINE, GRID, INK, INK2, INK3 = "#ffffff", "#ffffff", "#e2ddf2", "#efecf8", "#15102e", "#4b4768", "#8b88a8"
FONT = "Inter, 'Inter Display', 'Helvetica Neue', Arial, sans-serif"

def fmt_ms(v):
    if v >= 1000: return f"{v / 1000:.2f} s".replace(".00 s", " s").replace("0 s", " s") if v % 1000 == 0 else f"{v / 1000:.2f} s"
    return f"{v:.0f} ms" if v >= 100 else f"{v:.1f} ms".replace(".0 ms", " ms")
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")
def text(x, y, s, size, color=INK, weight=400, anchor="start", family=FONT, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" {extra}>{esc(s)}</text>')
def nice_max(v):
    for e in range(-1, 7):
        for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8):
            c = m * 10 ** e
            if c >= v: return c
    return v

def ticks(vmax, n=5):
    """a round step giving about n intervals, and the axis top as a whole number of steps"""
    raw = vmax / n; e = 10 ** math.floor(math.log10(raw))
    step = next(m * e for m in (1, 2, 2.5, 5, 10) if m * e >= raw)
    top = math.ceil(vmax / step - 1e-9) * step
    return step, top

# the wordmark: bubble letters in Luckiest Guy (Apache-2.0, embedded), one ROYGBIV hue per letter
_font = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "luckiest-guy-latin-400-normal.woff2")
_b64 = base64.b64encode(open(_font, "rb").read()).decode()
WORD = "TORBLES"; HUES = [R, O, Y, G, B, I, V]
_shades = "".join(
    f'<linearGradient id="bub{i}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity="1"/>'
    f'<stop offset=".18" stop-color="{h}"/><stop offset=".85" stop-color="{h}"/><stop offset="1" stop-color="#000000" stop-opacity=".9"/></linearGradient>'
    for i, h in enumerate(HUES))
BUBBLE_DEFS = (f'<style>@font-face{{font-family:"Bubble";src:url(data:font/woff2;base64,{_b64}) format("woff2");}}</style>'
               + _shades.replace('stop-color="#000000" stop-opacity=".9"', 'stop-color="#2a0f4a"'))
def bubble_word(x, y, size):
    """the word as bubble letters: drop shadow, dark outline, white rim, glossy hue fill; each letter bounced and tilted"""
    tilt = [-7, 5, -4, 6, -6, 4, -5]; bounce = [0, -5, 3, -6, 2, -4, 3]
    def spans(fill):
        out, prev = [], 0
        for i, ch in enumerate(WORD):
            f = fill(i); dy = bounce[i] - prev; prev = bounce[i]
            out.append(f'<tspan dy="{dy}" rotate="{tilt[i]}" dx="{size * 0.24 if i else 0:.1f}" {f}>{ch}</tspan>')
        return "".join(out)
    base = f'x="{x}" y="{y}" font-family="Bubble, \'Luckiest Guy\', Impact, sans-serif" font-size="{size}"'
    g = []
    g.append(f'<text {base} transform="translate(5,8)" fill="#3a2a7a" stroke="#3a2a7a" stroke-width="18" stroke-linejoin="round" opacity=".22">{spans(lambda i: "")}</text>')
    g.append(f'<text {base} fill="#140a2e" stroke="#140a2e" stroke-width="17" stroke-linejoin="round">{spans(lambda i: "")}</text>')
    g.append(f'<text {base} fill="#ffffff" stroke="#ffffff" stroke-width="7" stroke-linejoin="round">{spans(lambda i: "")}</text>')
    hue = lambda i: 'fill="url(#bub%d)"' % i
    g.append(f'<text {base}>{spans(hue)}</text>')
    return "".join(g)

el = []
defs = f'''<defs>
  <linearGradient id="rainbow" x1="0" x2="1" y1="0" y2="0">
    <stop offset="0" stop-color="{R}"/><stop offset=".17" stop-color="{O}"/><stop offset=".34" stop-color="{Y}"/>
    <stop offset=".5" stop-color="{G}"/><stop offset=".67" stop-color="{B}"/><stop offset=".84" stop-color="{I}"/><stop offset="1" stop-color="{V}"/>
  </linearGradient>
  <linearGradient id="gDuck" x1="0" x2="1"><stop offset="0" stop-color="{O}"/><stop offset="1" stop-color="{R}"/></linearGradient>
  <linearGradient id="gCpu" x1="0" x2="1"><stop offset="0" stop-color="{Y}"/><stop offset="1" stop-color="{G}"/></linearGradient>
  <linearGradient id="gGpu" x1="0" x2="1"><stop offset="0" stop-color="{B}"/><stop offset=".5" stop-color="{I}"/><stop offset="1" stop-color="{V}"/></linearGradient>
  <linearGradient id="aDuck" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{R}" stop-opacity=".16"/><stop offset="1" stop-color="{R}" stop-opacity="0"/></linearGradient>
  <linearGradient id="aCpu" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{Y}" stop-opacity=".18"/><stop offset="1" stop-color="{G}" stop-opacity="0"/></linearGradient>
  <linearGradient id="aGpu" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="{V}" stop-opacity=".28"/><stop offset="1" stop-color="{B}" stop-opacity="0"/></linearGradient>
  <linearGradient id="ground" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f4f1ff"/></linearGradient>
  <radialGradient id="glow1" cx=".88" cy=".08" r=".55"><stop offset="0" stop-color="{V}" stop-opacity=".16"/><stop offset="1" stop-color="{V}" stop-opacity="0"/></radialGradient>
  <radialGradient id="glow2" cx=".02" cy="1" r=".6"><stop offset="0" stop-color="{B}" stop-opacity=".13"/><stop offset="1" stop-color="{B}" stop-opacity="0"/></radialGradient>
  <radialGradient id="glow3" cx=".45" cy="0" r=".45"><stop offset="0" stop-color="{Y}" stop-opacity=".14"/><stop offset="1" stop-color="{Y}" stop-opacity="0"/></radialGradient>
  <radialGradient id="glow4" cx="1" cy=".95" r=".45"><stop offset="0" stop-color="{R}" stop-opacity=".09"/><stop offset="1" stop-color="{R}" stop-opacity="0"/></radialGradient>
  <linearGradient id="cardg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#fbfaff"/></linearGradient>
  <filter id="lift" x="-5%" y="-5%" width="110%" height="115%"><feDropShadow dx="0" dy="10" stdDeviation="16" flood-color="#3a2a7a" flood-opacity=".10"/></filter>
  <filter id="neon" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <linearGradient id="gloss" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#ffffff" stop-opacity=".75"/><stop offset=".3" stop-color="#ffffff" stop-opacity="0"/><stop offset=".85" stop-color="#2a0f4a" stop-opacity="0"/><stop offset="1" stop-color="#2a0f4a" stop-opacity=".45"/></linearGradient>
  <filter id="soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="14"/></filter>
  {BUBBLE_DEFS}
</defs>'''
el.append(f'<rect width="{W}" height="{H}" fill="url(#ground)"/>')
el.append("".join(f'<rect width="{W}" height="{H}" fill="url(#glow{k})"/>' for k in (1, 2, 3, 4)))
el.append(f'<rect width="{W}" height="7" fill="url(#rainbow)"/>')

# ---------------------------------------------------------------- header
sp = big["duckdb_ms"] / big["torbles_gpu_ms"]
el.append(bubble_word(72, 102, 74))
el.append(text(548, 74, "v0.1.0  ·  RESEARCH PROTOTYPE", 15, INK2, 700, extra='letter-spacing="2.5"'))
el.append(text(548, 98, "OFFICIAL STAR SCHEMA BENCHMARK", 15, INK3, 600, extra='letter-spacing="2.5"'))
el.append(text(64, 168, f"13 queries. {big['rows'] / 1e6:,.0f} million rows.", 50, INK, 800, family="'Inter Display', " + FONT, extra='letter-spacing="-1"'))
el.append(f'<text x="64" y="228" font-family="\'Inter Display\', {FONT}" font-size="50" font-weight="800" letter-spacing="-1">'
          f'<tspan fill="url(#rainbow)">{fmt_ms(big["torbles_gpu_ms"])}</tspan><tspan fill="{INK}"> on a laptop GPU.</tspan></text>')
el.append(text(64, 264, "Every result, at every scale, verified against DuckDB  ·  end to end, best of 3  ·  both engines in memory",
               17, INK2, 400))
# the hero multiplier
_hb = f'x="1528" y="172" font-family="Bubble, \'Luckiest Guy\', Impact, sans-serif" font-size="138" text-anchor="end" letter-spacing="4" stroke-linejoin="round"'
el.append(f'<text {_hb} transform="translate(5,8)" fill="#3a2a7a" stroke="#3a2a7a" stroke-width="20" opacity=".22">{sp:.0f}×</text>')
el.append(f'<text {_hb} fill="#140a2e" stroke="#140a2e" stroke-width="18">{sp:.0f}×</text>')
el.append(f'<text {_hb} fill="#ffffff" stroke="#ffffff" stroke-width="8">{sp:.0f}×</text>')
el.append(f'<text {_hb} fill="url(#rainbow)">{sp:.0f}×</text>')
el.append(f'<text {_hb} fill="url(#gloss)">{sp:.0f}×</text>')
el.append(text(1536, 214, "faster than DuckDB on 24 threads", 21, INK, 700, "end"))
el.append(text(1536, 240, f"at {big['rows'] / 1e6:,.0f}M rows: {fmt_ms(big['torbles_gpu_ms'])} vs {fmt_ms(big['duckdb_ms'])}", 16, INK2, 400, "end"))

def card(x, y, w, h, title, sub=None):
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="url(#cardg)" stroke="{LINE}" stroke-width="1.2" filter="url(#lift)"/>')
    el.append(text(x + 26, y + 40, title, 19, INK, 700))
    if sub: el.append(text(x + 26, y + 63, sub, 14, INK3, 400))

def series_path(pts):
    return "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)

def end_labels(items, x, y_top, y_bottom, gap=40):
    """items: (y, label, value, color); pushed apart to `gap`, kept inside [y_top, y_bottom]"""
    items = sorted(items, key=lambda t: t[0]); placed = []
    for it in items:
        y = max(it[0], (placed[-1] + gap) if placed else y_top)
        placed.append(y)
    over = placed[-1] - y_bottom
    if over > 0: placed = [p - over for p in placed]
    for (y0, lab, val, col), y in zip(items, placed):
        if abs(y - y0) > 3: el.append(f'<path d="M {x - 14},{y0:.1f} L {x - 4},{y - 6:.1f}" stroke="{INK3}" stroke-width="1"/>')
        el.append(f'<circle cx="{x + 5}" cy="{y - 6:.1f}" r="5" fill="{col}"/>')
        el.append(text(x + 18, y - 1, lab, 15, INK2, 600))
        el.append(text(x + 18, y + 20, val, 19, INK, 800))

# ---------------------------------------------------------------- main chart: time vs data size
SIDE = "--side" in sys.argv          # the two side panels (lead per scale, thread sweep): off by default
cx, cy, cw, ch = 64, 296, (936 if SIDE else 1472), 544
card(cx, cy, cw, ch, "Time for all 13 queries as the data grows", "linear axes  ·  6M to 300M fact rows (scale factors 1 to 50)")
px0, px1, py0, py1 = cx + 92, cx + cw - 236, cy + ch - 62, cy + 98
xstep, xmax = ticks(big["rows"] / 1e6 - 0.5, 6); ystep, ymax = ticks(max(r["duckdb_ms"] for r in sc) * 1.04, 5)
X = lambda v: px0 + (px1 - px0) * v / xmax
Yp = lambda v: py0 - (py0 - py1) * v / ymax
for k in range(int(round(ymax / ystep)) + 1):
    v = ystep * k; yy = Yp(v)
    el.append(f'<line x1="{px0}" x2="{px1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
    el.append(text(px0 - 14, yy + 5, "0" if v == 0 else fmt_ms(v), 13, INK3, 500, "end"))
for v in range(0, int(xmax) + 1, int(xstep)):
    el.append(text(X(v), py0 + 28, "0" if v == 0 else f"{v}M", 13, INK3, 500, "middle"))
el.append(f'<line x1="{px0}" x2="{px1}" y1="{py0}" y2="{py0}" stroke="{LINE}" stroke-width="1.5"/>')
rows = [r["rows"] / 1e6 for r in sc]
for key, grad, area, glow in (("duckdb_ms", "gDuck", "aDuck", False), ("torbles_cpu_ms", "gCpu", "aCpu", False), ("torbles_gpu_ms", "gGpu", "aGpu", True)):
    pts = [(X(x), Yp(r[key])) for x, r in zip(rows, sc)]
    el.append(f'<path d="{series_path(pts)} L {pts[-1][0]:.1f},{py0} L {pts[0][0]:.1f},{py0} Z" fill="url(#{area})"/>')
    if glow: el.append(f'<path d="{series_path(pts)}" stroke="url(#{grad})" stroke-width="12" fill="none" opacity=".35" filter="url(#soft)"/>')
    el.append(f'<path d="{series_path(pts)}" stroke="url(#{grad})" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
    for i, (qx, qy) in enumerate(pts):
        col = {"gDuck": [O, O, R, R, R], "gCpu": [Y, Y, G, G, G], "gGpu": [B, B, I, V, V]}[grad][min(i, 4) if len(pts) == 5 else 0]
        el.append(f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="7" fill="{col}" stroke="{CARD}" stroke-width="3"/>')
end_labels([(Yp(big["duckdb_ms"]), "DuckDB, 24 threads", fmt_ms(big["duckdb_ms"]), R),
            (Yp(big["torbles_cpu_ms"]), "torbles, 24 CPU threads", fmt_ms(big["torbles_cpu_ms"]), G),
            (Yp(big["torbles_gpu_ms"]), "torbles, GPU", fmt_ms(big["torbles_gpu_ms"]), V)], px1 + 26, py1 + 10, py0 - 8, gap=52)

# ---------------------------------------------------------------- the lead, per scale factor
bx, by, bw, bh = 1032, 276, 504, 268
if SIDE: card(bx, by, bw, bh, "The lead grows with the data", "GPU vs DuckDB on 24 threads, same 13 queries")
if SIDE:
    cols5 = [R, O, Y, G, B, I, V]
    n = len(sc); slot = (bw - 52) / n; base = by + bh - 48; top = by + 102
    spm = max(r["duckdb_ms"] / r["torbles_gpu_ms"] for r in sc)
    for i, r in enumerate(sc):
        s_ = r["duckdb_ms"] / r["torbles_gpu_ms"]; h_ = (base - top) * s_ / (spm * 1.08)
        x_ = bx + 26 + slot * i + slot * 0.16; w_ = slot * 0.68; col = cols5[min(len(cols5) - 1, round(i * (len(cols5) - 1) / max(1, n - 1)))]
        el.append(f'<rect x="{x_:.1f}" y="{base - h_:.1f}" width="{w_:.1f}" height="{h_:.1f}" rx="7" fill="{col}"/>')
        el.append(text(x_ + w_ / 2, base - h_ - 10, f"{s_:.0f}×", 22, INK, 800, "middle"))
        el.append(text(x_ + w_ / 2, base + 22, f"{r['rows'] / 1e6:,.0f}M", 13, INK2, 600, "middle"))

    # ---------------------------------------------------------------- threads
    tx, ty, tw, th_ = 1032, 568, 504, 268
    if T:
        card(tx, ty, tw, th_, "More cores, 60M rows", "1 to 24 CPU threads (a separate sweep)")
        pts_ = T["points"]; tmax = max(p["threads"] for p in pts_)
        qx0, qx1, qy0, qy1 = tx + 74, tx + tw - 150, ty + th_ - 46, ty + 92
        tstep, ym = ticks(max(p["duckdb_ms"] for p in pts_) * 1.04, 4)
        XX = lambda v: qx0 + (qx1 - qx0) * v / tmax; YY = lambda v: qy0 - (qy0 - qy1) * v / ym
        for k in range(int(round(ym / tstep)) + 1):
            v = tstep * k; yy = YY(v)
            el.append(f'<line x1="{qx0}" x2="{qx1}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            el.append(text(qx0 - 10, yy + 4, "0" if v == 0 else fmt_ms(v), 12, INK3, 500, "end"))
        for p in pts_:
            if p["threads"] in (1, 4, 8, 12, 16, 24): el.append(text(XX(p["threads"]), qy0 + 22, str(p["threads"]), 12, INK3, 500, "middle"))
        g10 = next((r for r in sc if r["sf"] == T["sf"]), None)
        if g10:
            el.append(f'<line x1="{qx0}" x2="{qx1}" y1="{YY(g10["torbles_gpu_ms"]):.1f}" y2="{YY(g10["torbles_gpu_ms"]):.1f}" stroke="url(#gGpu)" stroke-width="4" stroke-dasharray="2 9" stroke-linecap="round"/>')
        for key, grad, cols in (("duckdb_ms", "gDuck", R), ("torbles_cpu_ms", "gCpu", G)):
            pts = [(XX(p["threads"]), YY(p[key])) for p in pts_]
            el.append(f'<path d="{series_path(pts)}" stroke="url(#{grad})" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')
            for qx, qy in pts: el.append(f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="4.5" fill="{cols}" stroke="{CARD}" stroke-width="2"/>')
        last = pts_[-1]
        items = [(YY(last["duckdb_ms"]), "DuckDB", fmt_ms(last["duckdb_ms"]), R), (YY(last["torbles_cpu_ms"]), "torbles CPU", fmt_ms(last["torbles_cpu_ms"]), G)]
        if g10: items.append((YY(g10["torbles_gpu_ms"]), "torbles GPU", fmt_ms(g10["torbles_gpu_ms"]), V))
        end_labels(items, qx1 + 22, qy1, qy0 + 6, gap=46)

# ---------------------------------------------------------------- footer
el.append(text(64, 872, f"Core Ultra 9 290HX (24 cores) · RTX 5090 Laptop GPU · WSL2 · DuckDB {S.get('duckdb_version', '')} on an in-memory copy of its tables · "
               "official ssb-dbgen data and queries", 13, INK3, 400))
el.append(text(1536, 872, "torbles · Apache-2.0", 13, INK2, 700, "end"))

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}{"".join(el)}</svg>'
open(out + ".svg", "w").write(svg)
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        pg.set_content(f'<html><body style="margin:0;background:#ffffff">{svg}</body></html>'); pg.wait_for_timeout(300)
        pg.screenshot(path=out + ".png", clip={"x": 0, "y": 0, "width": W, "height": H}); b.close()
    print("wrote", out + ".svg", out + ".png")
except ImportError:
    print("wrote", out + ".svg (install playwright to render the PNG)")
