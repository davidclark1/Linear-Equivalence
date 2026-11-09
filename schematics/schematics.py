#!/usr/bin/env python
"""Schematic figures for "Linear equivalence of nonlinear recurrent neural networks".

  linear_equivalence_schematic.pdf  (Fig. 1)  nonlinear network = linear network with the same J + independent noise
  labeling.pdf                      (Fig. 5)  two copies of a topology, as a sum over labelings

The network layout is rnn_layout.json.  Everything is drawn in
points on an axes that fills the figure, so sizes in this file are sizes on the printed page if the figure is included
at its natural width.  Colors are opaque blends instead of transparency, so the PDFs contain no alpha.

Traces in Fig. 1 are simulated (traces.npz, cached; delete to regenerate): (1 + d/dt) x = J tanh(x) at g = 2, then a
linear network with the same J and response S = gamma / (1 + i omega), gamma = <tanh'(x)>, driven by independent noise
with the spectrum of the residual phi - gamma x.
The covariance panels of Fig. 1 use cov_blocks.npz: C_ij(tau = 0) of the simulation (C) and of the ansatz (Cbar) for the first
40 sites of the SCS run pm2_a800, seed 0, N = 10000 (C_blk[0] and Cbar_dmft_blk[0] of its result file).

Run (needs a LaTeX installation):  python schematics.py [a] [b]     -> schematics/linear_equivalence_schematic.pdf, schematics/labeling.pdf
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

HERE = pathlib.Path(__file__).resolve().parent

BG, INK, TRACE, FAINT = "#ffffff", "#000000", "#2b2d33", "#767a8f"
# Chroma OKLCH C <= 0.18, to match the data figures (paperstyle.py).
CHAOS, CHAOS_SHADE = "#c74997", "#822660"
STABLE, STABLE_SHADE = "#2e5eba", "#183773"
RNN, RIM, NODE, NODE_SHADE = "#6756b9", "#6756b9", "#2b306a", "#13153a"

FONT = "Liberation Sans"
# All text through LaTeX: math in Computer Modern, words in Helvetica (helvet), at the sizes given here.
PREAMBLE = r"\usepackage{amsmath,bm}\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}"
RC_TEX = {"pdf.fonttype": 42, "text.usetex": True, "font.family": "sans-serif", "font.size": 8,
          "text.latex.preamble": PREAMBLE}
RC_BASE = {"pdf.fonttype": 42, "font.family": FONT, "font.size": 8, "path.simplify": False,
           "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
           "mathtext.bf": f"{FONT}:bold", "mathtext.bfit": f"{FONT}:italic:bold", "mathtext.sf": FONT, "mathtext.cal": "cmsy10"}


def blend(col, a, bg=BG):
    """col at opacity a over bg, as an opaque color."""
    c, b = np.array(matplotlib.colors.to_rgb(col)), np.array(matplotlib.colors.to_rgb(bg))
    return tuple(a * c + (1 - a) * b)


def canvas(w, h):
    """Figure of w x h points with an axes in points."""
    fig = plt.figure(figsize=(w / 72, h / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def bezier(a, c, b, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(a) + 2 * t * (1 - t) * np.array(c) + t ** 2 * np.array(b)


def ball(ax, x, y, r, off, col, shade, rim=0.0, z=5):
    """A flat neuron: offset darker shade under a disk with a thin white rim."""
    ax.add_patch(Circle((x + off, y - off), r, fc=shade, ec="none", zorder=z))
    ax.add_patch(Circle((x, y), r, fc=col, ec=BG if rim else "none", lw=rim, zorder=z + 0.1))


def rich(ax, x, y, parts, size=8, ha="center", lead=0.0):
    """One line of text in several colors, parts = [(string, color), ...], placed as a unit, after an empty space of
    width lead (for an icon); returns the left end of the line (where the icon goes)."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    ts = [ax.text(0, y, s, color=c, fontsize=size, ha="left", va="baseline") for s, c in parts]
    inv = ax.transData.inverted()
    # spaces at the ends of a string are not measured, so measure each part with its spaces protected
    ws = []
    for t in ts:
        s = t.get_text()
        t.set_text("." + s + ".")
        wd = t.get_window_extent(rend).width
        t.set_text("..")
        wd -= t.get_window_extent(rend).width
        t.set_text(s)
        ws.append(inv.transform((wd, 0))[0] - inv.transform((0, 0))[0])
    start = x - (sum(ws) + lead) / 2 if ha == "center" else x
    x0 = start + lead
    for t, w in zip(ts, ws):
        t.set_x(x0)
        x0 += w
    return start


def transfer_icon(ax, x0, yb, w, h, f, col):
    """A transfer-function icon: f(u) for u in [-3, 3] over a faint baseline;
    w x 2h points, baseline at yb."""
    u = np.linspace(-3, 3, 61)
    ax.plot([x0, x0 + w], [yb, yb], color=FAINT, lw=0.5, solid_capstyle="butt", zorder=6)
    ax.plot(x0 + (u + 3) / 6 * w, yb + h * f(u), color=col, lw=1.1, solid_capstyle="round", solid_joinstyle="round", zorder=6.1)


# ------------------------------------------------------------------------------------------- Figure A
def simulate():
    path = HERE / "traces.npz"
    if path.exists():
        return dict(np.load(path))
    rng = np.random.default_rng(17)
    N, g, dt, sub = 400, 2.0, 0.05, 2             # samples every h = 0.1
    h = dt * sub
    J = g * rng.standard_normal((N, N)) / np.sqrt(N)
    np.fill_diagonal(J, 0)
    x = rng.standard_normal(N)
    for _ in range(int(200 / dt)):
        x += dt * (-x + J @ np.tanh(x))
    T = 6000
    phi, xs = np.zeros((T, N)), np.zeros((T, N))
    for t in range(T):
        xs[t], phi[t] = x, np.tanh(x)
        for _ in range(sub):
            x += dt * (-x + J @ np.tanh(x))
    # Large N: x is Gaussian, so the linear response of phi to its local field is S(omega) = gamma / (1 + i omega),
    # with gamma = <phi'(x)>, and the residual is Delta_i = phi_i - gamma x_i.
    gamma = (1 / np.cosh(xs) ** 2).mean()
    print(f"g gamma = {g * gamma:.3f}")
    Delta = phi - gamma * xs
    # independent noise with the spectrum of the residual (phase randomized, independently per unit)
    P = (np.abs(np.fft.rfft(Delta, axis=0)) ** 2).mean(1)
    P = np.convolve(P, np.ones(21) / 21, mode="same")
    noise = np.fft.irfft(np.sqrt(P)[:, None] * np.exp(2j * np.pi * rng.random((P.size, N))), n=T, axis=0)
    noise *= Delta.std() / noise.std()
    # linear network with the same J: (1 + d/dt) x = J phi, phi = gamma x + noise
    # g gamma is close to 1 and finite-N outliers of J can pass 1 (slow modes); cap gamma so the display run is stable.
    # Only the traces drawn in the figure depend on this.
    gam = min(gamma, 0.95 / np.linalg.eigvals(J).real.max())
    print(f"gamma {gamma:.3f}, capped {gam:.3f}")
    lin = np.zeros((T, N))
    xl = np.zeros(N)
    for t in range(T):
        lin[t] = gam * xl + noise[t]
        xl = xl + h * (-xl + J @ lin[t])
    a, b = phi[1000:], lin[1000:]
    Ca, Cb = np.cov(a.T), np.cov(b.T)
    iu = np.triu_indices(N, 1)
    print(f"var nonlinear {np.diag(Ca).mean():.3f}, linear {np.diag(Cb).mean():.3f}; "
          f"off-diagonal correlation of the two equal-time covariance matrices {np.corrcoef(Ca[iu], Cb[iu])[0, 1]:.3f}")
    out = dict(phi=phi[-1500:, :24], lin=lin[-1500:, :24], noise=noise[-1500:, :24], h=np.array(h))
    np.savez_compressed(path, **out)
    return out


def network(ax, lay, cx, cy, R, node=(NODE, NODE_SHADE)):
    """Network diagram.  rnn_layout's y axis points down."""
    u = R / 121
    P = lambda q: (cx + q[0] * R, cy - q[1] * R)
    ax.add_patch(Circle((cx, cy), 119 * u, fc=blend(RNN, 0.025), ec="none", zorder=1))
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=RIM, lw=max(0.9, 2.6 * u), zorder=2.6))
    for s in lay["stubs"]:          # edges that fade toward the rim
        pts = bezier(*[P(q) for q in s["pts"]])
        f = np.linspace(0, 1, len(pts) - 1)
        op = np.interp(f, [0, 0.75, 1], [s["o0"], s["o1"], 0])
        ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:] + 0.02 * (pts[1:] - pts[:-1])], 1),
                                         colors=[blend(RNN, o) for o in op], lw=max(0.3, s["w"] * R), capstyle="butt", zorder=1.5))
    for e in sorted(lay["edges"], key=lambda e: e["opacity"]):
        pts = bezier(*[P(q) for q in e["pts"]])
        ax.plot(*pts.T, color=blend(RNN, e["opacity"]), lw=max(0.45, e["w"] * R), solid_capstyle="round", zorder=2)
    nodes = []
    for b in lay["balls"]:
        x, y = P((b["x"], b["y"]))
        r = b["r"] * R
        ball(ax, x, y, r, max(0.5, b["off"] * R), node[0], node[1], rim=max(0.35, 1.2 * u) if b.get("rim") else 0)
        nodes.append((x, y, r))
    return nodes


def squiggle(ax, x, y, r0, L, th, v, amp, lw, col=CHAOS):
    """A short noise trace, in magenta, along angle th from the unit at (x, y): strongest where it enters
    the unit (distance r0), fading out at its far end.  v[-1] is the sample at the unit."""
    u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    # keep the wiggle on one reading direction: the normal points up for traces on either side
    if n[1] < 0:
        n = -n
    f = np.linspace(1, 0, len(v))[:, None]
    pts = np.array([x, y]) + (r0 + f * L) * u + amp * v[:, None] * n
    a = np.interp(np.linspace(0, 1, len(pts) - 1), [0, 0.45, 1], [0.08, 0.9, 1])
    ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:] + 0.02 * (pts[1:] - pts[:-1])], 1),
                                     colors=[blend(col, q) for q in a], lw=lw, capstyle="round", zorder=3))


def stack(ax, x0, x1, y0, y1, traces, sign_label, lw=0.65, col=TRACE):
    n = len(traces)
    rh = (y1 - y0) / n
    for r, v in enumerate(traces):
        yc = y1 - rh * (r + 0.5)
        ax.plot(np.linspace(x0, x1, len(v)), yc + v * rh * 0.39, color=col, lw=lw, solid_joinstyle="round")


# covariance panels: block of the first NBLOCK sites; standard diverging map for signed covariances (blue < 0 < red)
COV_CMAP = matplotlib.colormaps["RdBu_r"]
NBLOCK, COV_VMAX, DIAG_GRAY = 24, 8.0, TRACE   # vmax ~ 2.3 s.d. of the off-diagonal values

# direction of each unit's noise trace, chosen to keep it clear of the other units and inside the disk
NOISE_ANGLE = {0: 192, 9: 168}   # all others 180 (from the left)


def figure_a():
    plt.rcParams.update(RC_TEX)
    sim = simulate()
    lay = json.load(open(HERE / "rnn_layout.json"))
    W, H = 453.6, 216.0      # 6.3 in x 3 in
    fig, ax = canvas(W, H)
    R = 44.0
    cy = H - 25 - R
    tw, gap = 80.0, 10.0     # width of the trace stacks, and their gap to the disk
    # each half: the network, then its activity phi_i(t) on the right; the halves are parallel
    cxL = 11.0 + R           # wider left margin: the title group is wider than the disk
    xtL = cxL + R + gap
    xtR = W - 4 - tw
    cxR = xtR - gap - R
    # the networks: identical wiring, the same drawing twice
    colored = "indigo" not in sys.argv   # "indigo": one neuron color in both networks
    network(ax, lay, cxL, cy, R, node=(CHAOS, CHAOS_SHADE) if colored else (NODE, NODE_SHADE))
    nodes = network(ax, lay, cxR, cy, R, node=(STABLE, STABLE_SHADE) if colored else (NODE, NODE_SHADE))

    # activity traces phi_i(t), one stack on each side: simulated, the same units in both networks
    n0, dn = 1000, 400       # samples; h = 0.1 so 40 time units
    ids = [2, 5, 9, 14, 21]
    y0, y1 = cy - 0.84 * R, cy + 0.84 * R

    def shown(a):
        v = a[n0:n0 + dn][:, ids]
        v = v - v.mean(0)
        return list((v / np.abs(v).max(0)).T)   # each row scaled to fill its height

    stack(ax, xtL, xtL + tw, y0, y1, shown(sim["phi"]), "")
    stack(ax, xtR, xtR + tw, y0, y1, shown(sim["lin"]), "")
    for xt in (xtL, xtR):
        ax.text(xt + tw / 2, y1 + 3.5, r"$\phi_i(t)$", ha="center", va="bottom", fontsize=8)
        ax.annotate("", xy=(xt + tw, y0 - 5), xytext=(xt + tw - 18, y0 - 5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.11,head_length=0.28", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
        ax.text(xt + tw - 20, y0 - 5, "time", ha="right", va="center", fontsize=7)

    # independent noise entering each unit of the linear network, along the direction NOISE_ANGLE (degrees) from the unit
    # from the left; units near the rim get a shorter trace, so that every trace stays inside the disk
    L0, amp = 0.29 * R, 0.055 * R
    for i, (x, y, r) in enumerate(nodes):
        th = np.deg2rad(NOISE_ANGLE.get(i, 180))
        d = np.array([np.cos(th), np.sin(th)])
        p = np.array([x - cxR, y - cy])
        to_rim = -p @ d + np.sqrt((p @ d) ** 2 - p @ p + (R - 3.5) ** 2)   # distance from the unit to the rim, minus 3.5
        L = float(np.clip(to_rim - r - 0.7, 0.12 * R, L0))
        n = int(round(100 * L / L0))          # the same time scale on every trace: shorter traces show less time
        v = sim["noise"][300 + 37 * i:300 + 37 * i + n, i] / sim["noise"].std()
        v = np.convolve(v, np.ones(5) / 5, mode="same")[2:-2]   # display only: lines stay clean at print size
        v = v - v[-1]
        v = v * min(1.0, 1.6 / np.abs(v).max())
        squiggle(ax, x, y, r + 0.7, L, th, v, amp, 0.8)

    # the covariance matrices: a block of sites, sqrt(N) C_ij at lag 0, diagonal masked (gray); simulation data (cov_blocks.npz, see the module docstring)
    cb = np.load(HERE / "cov_blocks.npz")
    nb, M, vmax = NBLOCK, 80.0, COV_VMAX
    sN = np.sqrt(float(cb["N"]))
    cmap = COV_CMAP
    ym1 = 3.0                 # bottom of the matrices
    ym0 = ym1 + M
    # each matrix under its half, with the "=" between them
    hcL = (cxL - R + xtL + tw) / 2
    hcR = (cxR - R + xtR + tw) / 2
    for hc, key, lab in [(hcL, "C", r"$\sqrt{N}\,C^\phi_{ij}(\tau = 0)$"), (hcR, "Cbar", r"$\sqrt{N}\,\bar{C}^\phi_{ij}(\tau = 0)$")]:
        A = sN * cb[key][:nb, :nb].astype(float)
        A = np.ma.masked_where(np.eye(nb, dtype=bool), A)
        xe = np.linspace(hc - M / 2, hc + M / 2, nb + 1)
        ye = np.linspace(ym0, ym1, nb + 1)          # row i = 1 at the top
        cm = cmap.copy()
        cm.set_bad(DIAG_GRAY)
        ax.pcolormesh(xe, ye, A, cmap=cm, vmin=-vmax, vmax=vmax, shading="flat", edgecolors="face", linewidth=0.05, zorder=4)
        ax.add_patch(matplotlib.patches.Rectangle((hc - M / 2, ym1), M, M, fc="none", ec=INK, lw=0.4, zorder=4.5))
        ax.text(hc, ym0 + 4, lab, ha="center", va="bottom", fontsize=8)
    # the equality, between the matrices, and the shared color scale
    xm = (hcL + hcR) / 2
    ymid = (ym0 + ym1) / 2
    ax.text(xm, ymid + 6, "$=$", ha="center", va="center", fontsize=30)
    ax.text(xm, ymid - 14, "at large $N$", ha="center", va="center", fontsize=8)
    cw, ch, cyb = 46.0, 3.6, ym1 + 6
    xs = np.linspace(xm - cw / 2, xm + cw / 2, 129)
    ax.pcolormesh(xs, [cyb, cyb + ch], np.linspace(-vmax, vmax, 128)[None, :], cmap=cmap, vmin=-vmax, vmax=vmax,
                  shading="flat", edgecolors="face", linewidth=0.05, zorder=4)
    ax.add_patch(matplotlib.patches.Rectangle((xm - cw / 2, cyb), cw, ch, fc="none", ec=INK, lw=0.4, zorder=4.5))
    for t in (-vmax, 0, vmax):
        xt = xm + t / vmax * cw / 2
        ax.plot([xt, xt], [cyb, cyb - 1.6], color=INK, lw=0.4)
        ax.text(xt, cyb - 2.6, f"${t:g}$".replace("-", "-"), ha="center", va="top", fontsize=7)

    # labels
    yt1, yt2 = H - 8, H - 18.5
    # title line: the unit's transfer function (icon), then the name, both in the network's color
    # (icon + name) centered over the disk; the second line centered under the name
    iw, ih, ig = 18.0, 5.0, 3.5
    yi = yt1 + 3.1            # icon baseline at mid x-height of the title
    xi = rich(ax, cxL, yt1, [("nonlinear network", CHAOS)], size=8.5, lead=iw + ig)
    transfer_icon(ax, xi, yi, iw, ih, np.tanh, CHAOS)
    xi = rich(ax, cxR, yt1, [("linear network", STABLE)], size=8.5, lead=iw + ig)
    transfer_icon(ax, xi, yi, iw, ih, lambda u: u / 3, STABLE)
    sh = (iw + ig) / 2
    ax.text(cxL + sh, yt2, r"couplings $\bm{J}$", ha="center", va="baseline", fontsize=8)
    rich(ax, cxR + sh, yt2, [(r"same $\bm{J}$, ", INK), ("independent noise", CHAOS)], size=8)
    fig.savefig(HERE / "linear_equivalence_schematic_2col.pdf")
    plt.close(fig)


EQ_FS, EQ_DY, EQ_TXT, CB_H = 20, 3.5, 7.0, 26.0


def figure_a1(out="linear_equivalence_schematic.pdf", R=36.0, M=68.0):
    """Fig. 1 at one column (246 pt): one row per network (network, its activity phi_i(t), its covariance block), the
    nonlinear network on top; the equality between the two covariance blocks, on the right."""
    plt.rcParams.update(RC_TEX)
    sim = simulate()
    lay = json.load(open(HERE / "rnn_layout.json"))
    W = 246.0
    # vertical layout: two title lines over each disk
    tl1, tl2, tdisk = 8.0, 18.0, 24.5          # baselines of the title lines and top of the disk, below the row's top
    rowgap = 6.0                               # between the bottom of the upper row's disk and the top of the lower row
    padtop = 3.5                               # headroom above the top row (icon and block label reach above the title line)
    H = 2 * (tdisk + 2 * R) + rowgap + 2.5 + padtop
    cy1 = H - padtop - tdisk - R
    # (the extra 1 pt at the bottom: room for the lower block's label)
    cy2 = cy1 - 2 * R - rowgap - tdisk
    fig, ax = canvas(W, H)
    cx = 1.2 + R
    gap, tw = 6.0, 61.0
    xt = cx + R + gap
    xm0 = W - 20.0 - M                         # left edge of the covariance blocks (color scale to their right)
    colored = "indigo" not in sys.argv
    network(ax, lay, cx, cy1, R, node=(CHAOS, CHAOS_SHADE) if colored else (NODE, NODE_SHADE))
    nodes = network(ax, lay, cx, cy2, R, node=(STABLE, STABLE_SHADE) if colored else (NODE, NODE_SHADE))

    n0, dn = 1000, 400
    ids = [2, 5, 9, 14, 21]

    def shown(a):
        v = a[n0:n0 + dn][:, ids]
        v = v - v.mean(0)
        return list((v / np.abs(v).max(0)).T)

    for cy, key in ((cy1, "phi"), (cy2, "lin")):
        y0, y1 = cy - 0.8 * R, cy + 0.8 * R
        stack(ax, xt, xt + tw, y0, y1, shown(sim[key]), "", lw=0.6)
        ax.text(xt + tw / 2, y1 + 3.0, r"$\phi_i(t)$", ha="center", va="bottom", fontsize=8)
        ax.annotate("", xy=(xt + tw, y0 - 5), xytext=(xt + tw - 16, y0 - 5),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.11,head_length=0.28", lw=0.5, color=INK, shrinkA=0, shrinkB=0))
        ax.text(xt + tw - 18, y0 - 5, "time", ha="right", va="center", fontsize=7)

    L0, amp = 0.29 * R, 0.055 * R
    for i, (x, y, r) in enumerate(nodes):
        th = np.deg2rad(NOISE_ANGLE.get(i, 180))
        d = np.array([np.cos(th), np.sin(th)])
        p = np.array([x - cx, y - cy2])
        to_rim = -p @ d + np.sqrt((p @ d) ** 2 - p @ p + (R - 3.0) ** 2)
        L = float(np.clip(to_rim - r - 0.6, 0.12 * R, L0))
        n = int(round(100 * L / L0))
        v = sim["noise"][300 + 37 * i:300 + 37 * i + n, i] / sim["noise"].std()
        v = np.convolve(v, np.ones(5) / 5, mode="same")[2:-2]
        v = v - v[-1]
        v = v * min(1.0, 1.6 / np.abs(v).max())
        squiggle(ax, x, y, r + 0.6, L, th, v, amp, 0.75)

    # covariance blocks, one per row: the upper one's label above it, the lower one's below it, the equality between them
    cb = np.load(HERE / "cov_blocks.npz")
    nb, vmax = NBLOCK, COV_VMAX
    sN = np.sqrt(float(cb["N"]))
    hc = xm0 + M / 2
    top1 = cy1 + R + tdisk - 13.0               # top of the upper block (its label above, inside the figure)
    bot2 = 14.5                                 # bottom of the lower block (its label below)
    for key, lab, ym0 in (("C", r"$\sqrt{N}\,C^\phi_{ij}(\tau = 0)$", top1), ("Cbar", r"$\sqrt{N}\,\bar{C}^\phi_{ij}(\tau = 0)$", bot2 + M)):
        A = sN * cb[key][:nb, :nb].astype(float)
        A = np.ma.masked_where(np.eye(nb, dtype=bool), A)
        ym1 = ym0 - M
        cm = COV_CMAP.copy()
        cm.set_bad(DIAG_GRAY)
        ax.pcolormesh(np.linspace(xm0, xm0 + M, nb + 1), np.linspace(ym0, ym1, nb + 1), A, cmap=cm, vmin=-vmax, vmax=vmax,
                      shading="flat", edgecolors="face", linewidth=0.05, zorder=4)
        ax.add_patch(matplotlib.patches.Rectangle((xm0, ym1), M, M, fc="none", ec=INK, lw=0.4, zorder=4.5))
        if key == "C":
            ax.text(hc, ym0 + 3.0, lab, ha="center", va="bottom", fontsize=8)
        else:
            ax.text(hc, ym1 - 3.0, lab, ha="center", va="top", fontsize=8)
    ymid = (top1 - M + bot2 + M) / 2
    ax.text(hc, ymid + EQ_DY, "$=$", ha="center", va="center", fontsize=EQ_FS)
    ax.text(hc, ymid - EQ_TXT, "at large $N$", ha="center", va="center", fontsize=7)
    # the shared color scale, at the right edge, level with the equality
    cw, ch = 3.2, CB_H
    xcb = xm0 + M + 4.0
    ys = np.linspace(ymid - ch / 2, ymid + ch / 2, 129)
    ax.pcolormesh([xcb, xcb + cw], ys, np.linspace(-vmax, vmax, 128)[:, None], cmap=COV_CMAP, vmin=-vmax, vmax=vmax,
                  shading="flat", edgecolors="face", linewidth=0.05, zorder=4)
    ax.add_patch(matplotlib.patches.Rectangle((xcb, ymid - ch / 2), cw, ch, fc="none", ec=INK, lw=0.4, zorder=4.5))
    for t in (-vmax, 0, vmax):
        yy = ymid + t / vmax * ch / 2
        ax.plot([xcb + cw, xcb + cw + 1.6], [yy, yy], color=INK, lw=0.4)
        ax.text(xcb + cw + 2.4, yy, f"${t:g}$", ha="left", va="center", fontsize=7)

    # titles, left-aligned over each row: transfer-function icon and name in the network's color; second line in black
    iw, ih, ig = 16.0, 4.5, 3.0
    for cy, name, col, f, line2 in ((cy1, "nonlinear network", CHAOS, np.tanh, [(r"couplings $\bm{J}$", INK)]),
                                    (cy2, "linear network", STABLE, lambda u: u / 3, [(r"same $\bm{J}$, ", INK), ("independent noise", CHAOS)])):
        top = cy + R + tdisk
        x0 = 1.2
        rich(ax, x0, top - tl1, [(name, col)], size=8.5, ha="left", lead=iw + ig)
        transfer_icon(ax, x0, top - tl1 + 3.1, iw, ih, f, col)
        rich(ax, x0, top - tl2, line2, size=8, ha="left")
    fig.savefig(HERE / out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------- Figure B
# Positions in pixels of an 800 dpi drawing, normalized below to the disk (center 758, 641; radius 488).  'kind': e external, v in V_1, n not in V_1.
NODES_B = {
    "E1": (123, 552, 56, "e"), "E2": (123, 708, 56, "e"),
    "A": (555, 348, 43, "v"), "B": (841, 244, 43, "v"), "C": (1017, 375, 38, "v"), "D": (893, 507, 62, "v"),
    "F": (585, 590, 58, "v"), "G": (1163, 687, 44, "v"), "H": (826, 814, 58, "n"), "I": (983, 912, 46, "v"),
    "K": (533, 933, 46, "v"), "L": (758, 1062, 40, "n"),
}
# (source, target, bend): bend = control-point offset, as a fraction of the chord, to the left of the direction of travel.
EDGES_B = [
    ("E1", "A", -0.09), ("F", "E1", -0.10), ("K", "E1", 0.12), ("K", "E2", 0.05),
    ("B", "A", 0.09), ("C", "B", 0.0), ("D", "B", 0.0), ("D", "C", 0.04),
    ("D", "F", 0.17), ("D", "F", -0.10),
    ("A", "F", -0.40), ("F", "A", -0.04), ("F", "A", -0.50),
    ("D", "G", -0.05), ("G", "I", -0.10), ("I", "D", 0.0),
    ("F", "H", -0.20), ("F", "H", 0.16),
    ("H", "I", -0.50), ("H", "I", 0.50),
    ("K", "F", -0.06),
    ("L", "K", -0.42), ("L", "K", -0.04), ("L", "K", 0.34),
]
# Flat style, three gray levels: open (external), mid (in V_1), dark (not in V_1).
STYLE_B = {"e": dict(col=BG, shade=None), "v": dict(col=CHAOS, shade=CHAOS_SHADE), "n": dict(col=NODE, shade=NODE_SHADE)}


def node_b(ax, x, y, r, kind, z=5):
    s = STYLE_B[kind]
    if kind == "e":
        ax.add_patch(Circle((x, y), r, fc=BG, ec=NODE, lw=0.9, zorder=z))
    else:
        ball(ax, x, y, r, max(0.45, 0.14 * r), s["col"], s["shade"], rim=0.3, z=z)


def arrow_edge(ax, a, c, b, ra, rb, lw=0.5, col=NODE, head=(2.9, 1.15), gap=0.35):
    """Quadratic curve from a to c to b, trimmed at the node rims, with a notched arrowhead on the target rim."""
    pts = bezier(a, c, b, 200)
    da, db = np.hypot(*(pts - a).T), np.hypot(*(pts - b).T)
    i0 = np.argmax(da >= ra + gap * 0.5)
    i1 = len(pts) - 1 - np.argmax(db[::-1] >= rb + gap)
    tip = pts[i1]
    j = i1 - np.argmax(np.hypot(*(pts[:i1 + 1][::-1] - tip).T) >= head[0])
    d = tip - pts[j]
    d = d / np.hypot(*d)
    n = np.array([-d[1], d[0]])
    base = tip - head[0] * d
    notch = tip - 0.72 * head[0] * d
    # the line stops inside the notch, so it never pokes past the tip
    k = np.argmax(np.hypot(*(pts[:i1 + 1] - tip).T) <= 0.62 * head[0])
    ax.plot(*pts[i0:k + 1].T, color=col, lw=lw, solid_capstyle="butt", zorder=3)
    ax.add_patch(Polygon([tip, base + head[1] * n, notch, base - head[1] * n], closed=True, fc=col, ec="none", zorder=3.2))


def graph_b(ax, cx, cy, R):
    u = R / 488
    P = {k: (cx + (v[0] - 758) * u, cy - (v[1] - 641) * u, v[2] * u, v[3]) for k, v in NODES_B.items()}
    ax.add_patch(Circle((cx, cy), R - 0.2, fc=blend(RNN, 0.025), ec="none", zorder=1))
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=RIM, lw=0.9, zorder=1.1))
    for s, t, bend in EDGES_B:
        a, b = np.array(P[s][:2]), np.array(P[t][:2])
        chord = b - a
        c = (a + b) / 2 + bend * np.array([-chord[1], chord[0]])
        arrow_edge(ax, a, c, b, P[s][2], P[t][2])
    for x, y, r, kind in P.values():
        node_b(ax, x, y, r, kind)


def figure_b():
    plt.rcParams.update(RC_TEX)
    W, H = 216.0, 96.0        # 3 in x 1.33 in
    fig, ax = canvas(W, H)
    R, cy = 28.0, 53.0
    # sum, its subscript, brackets
    xs = 24.0
    ax.text(xs, cy + 1.5, r"$\displaystyle\sum$", ha="center", va="center", fontsize=19)
    ax.text(xs, cy - 21, r"$\Lambda(\theta),\Lambda'(\theta)$", ha="center", va="center", fontsize=8.5)
    x_open, x_close = 48.5, 209.0
    half, dx = 38.0, 8.5
    br = dict(color=INK, lw=0.9, solid_capstyle="round", solid_joinstyle="round")
    ax.plot([x_open + dx, x_open, x_open + dx], [cy + half, cy, cy - half], **br)
    ax.plot([x_close - dx, x_close, x_close - dx], [cy + half, cy, cy - half], **br)
    ax.text(x_close + 0.5, cy - half + 0.5, r"$\bm{J}$", ha="center", va="center", fontsize=8.5)
    # the two copies
    ext = (758 - 123) / 488 * R + 56 / 488 * R
    gap = 5.0
    c1 = x_open + 6.0 + ext
    c2 = c1 + R + gap + ext
    graph_b(ax, c1, cy, R)
    graph_b(ax, c2, cy, R)
    ytop = cy + R + 7.0
    ax.text(c1, ytop, r"$\Lambda(\theta)$", ha="center", va="center", fontsize=9)
    ax.text(c2, ytop, r"$\Lambda'(\theta)$", ha="center", va="center", fontsize=9)
    # key
    yk = 8.0
    r = 3.0
    for x, kind, s in [(66, "e", r"$\in$ external"), (124, "v", r"$\in V_1$"), (162, "n", r"$\notin V_1$")]:
        node_b(ax, x, yk, r, kind)
        ax.text(x + r + 2.5, yk - 0.2, s, ha="left", va="center", fontsize=8.5)
    fig.savefig(HERE / "labeling.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = [q for q in sys.argv[1:] if q in ("a", "a2", "b")] or ["a", "b"]
    if "a" in which:
        figure_a1()
    if "a2" in which:
        figure_a()
    if "b" in which:
        figure_b()
