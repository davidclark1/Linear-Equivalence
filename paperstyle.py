"""One plotting style and one exponent fit for every figure of the paper.

Exponent: ordinary least-squares slope of log(median over realizations) against log N, over N >= FIT_NMIN (the medians are the
plotted lines). Its standard error is the s.d. of the slope over B = 1000 bootstrap resamples of the realizations, drawn with
replacement within each N (number of realizations per N kept fixed; medians recomputed), with a fixed RNG seed. Medians are used
because a few near-critical realizations at N <= 1000 would dominate a fit to the mean (Appendix I of the paper).
"""
import logging
import numpy as np
import matplotlib
import matplotlib.ticker as mticker
from matplotlib import font_manager

B_BOOT = 1000
BOOT_SEED = 0
FIT_NMIN = 464      # every exponent: N >= 464 (nonzero-mean grid, 500-8000: all sizes); smaller networks are plotted, never fitted


# print sizes (revtex aps two-column, 10 pt): every figure is made at the width at which the paper prints it, so fonts print at true size
TEXT_W = 510 / 72.27      # \\textwidth (in)
COL_W = 246 / 72.27       # \\columnwidth (in)

# Colours have one meaning across the figures. Sampling ratios alpha = 200, 800, 3200 (SCS model, cavity tests) keep their colours; no other
# quantity uses them, except that single-series data of the nonzero-mean models (all at alpha = 200) use the alpha = 200 colour.
ALPHA_COLORS = {200: '#264653', 800: '#e76f51', 3200: '#7b2d8e'}
C_MEASURED, C_THEORY, C_SIGNAL, C_SINGLE = '#2a9d8f', '#e9a03b', '#8c8c8c', ALPHA_COLORS[200]      # nonzero-mean: measured, single-site theory, signal, single series
C_TAP = '#b5446e'      # nonzero-mean: single-site theory at hbar from the solved TAP-like equation (TAP-solved)
LAG_COLORS = {0.0: '#111111', 2.0: '#1f78b4', 5.0: '#33a02c', 10.0: '#b8a100'}                     # time lags (at alpha = 800)
PAIR_COLORS = ['#0072B2', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']                              # pairs of sites (Okabe-Ito)
REF_LABELS = {-0.5: r'$N^{-1/2}$', -1.0: r'$N^{-1}$', -1.5: r'$N^{-3/2}$'}
REF_GRAY, REF_TEXT = '#9a9a9a', '#6e6e6e'     # reference power laws: dotted line, label
KEY_FS = 7                # every key and annotation (>= 7 pt at print size)
PANEL_FS = 8              # bold panel labels "(a)"
KEY_GRAY = '#666666'      # handles of line-vs-marker keys
LW_DATA, MS, BAND_ALPHA = 0.7, 2.8, 0.13      # medians: markers joined by a thin line; interquartile band
FLOOR_KW = dict(lw=0.5, alpha=0.6)            # sampling floor: thin line, no markers
LW_REF, LW_THEORY = 0.7, 0.8
OPEN = dict(marker='o', ls='', mfc='none', mew=0.45, ms=2.6)   # simulation in theory-vs-simulation comparisons: open circles
SIM_LW, SIM_TINT = 2.4, 0.4     # simulated curves under theory curves (Fig. cij_tau): thick line in the curve's colour at this opacity, as an opaque tint


def tint(color, a=SIM_TINT):
    """color at opacity a over white, as an opaque color (no transparency in the PDF)."""
    c = np.array(matplotlib.colors.to_rgb(color)); return tuple(a * c + (1 - a))


def _font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'):
        if name in names: return name
    return 'sans-serif'


FONT = _font()      # same rule as schematics/schematics.py


def use():
    matplotlib.rcParams.update({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': KEY_FS, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'lines.linewidth': LW_DATA, 'lines.markersize': MS,
        'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
        'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0, 'axes.labelpad': 2.0, 'axes.titlepad': 3.0,
        'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.labelspacing': 0.25, 'lines.markeredgewidth': 0,
        # LaTeX: text in Helvetica (helvet, scaled), math in Computer Modern (no sfmath); needs a LaTeX installation
        'text.usetex': True, 'font.family': 'sans-serif', 'font.sans-serif': ['computer modern sans serif'],
        'text.latex.preamble': r'\usepackage{amsmath,bm}\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}',
        'pdf.fonttype': 42, 'savefig.bbox': 'standard', 'figure.constrained_layout.use': False})
    logging.getLogger('fontTools').setLevel(logging.ERROR)


def pooled_fit(points, Nmin=FIT_NMIN, Nmax=None):
    """Median fit. points: dict {N: sequence of positive values, one per realization}.
    Returns dict(slope, se, intercept, n_per_N, N_range)."""
    Ns = sorted(n for n in points if (Nmin is None or n >= Nmin) and (Nmax is None or n <= Nmax) and len(points[n]) > 0)
    vals = {n: np.asarray(points[n], float) for n in Ns}
    if len(Ns) < 2 or any(np.any(~np.isfinite(v)) or np.any(v <= 0) for v in vals.values()):
        return dict(slope=float('nan'), se=float('nan'), intercept=float('nan'), n_per_N={int(n): int(len(vals[n])) for n in Ns}, N_range=[0, 0])
    def fit(vs):     # OLS of log(median) against log N
        x = np.log(np.array(Ns, float)); y = np.log(np.array([np.median(vs[n]) for n in Ns]))
        return np.polyfit(x, y, 1)
    s, c = fit(vals)
    rng = np.random.default_rng(BOOT_SEED)
    bs = [fit({n: vals[n][rng.integers(0, len(vals[n]), len(vals[n]))] for n in Ns})[0] for _ in range(B_BOOT)]
    return dict(slope=float(s), se=float(np.std(bs, ddof=1)), intercept=float(c), n_per_N={int(n): int(len(vals[n])) for n in Ns}, N_range=[int(Ns[0]), int(Ns[-1])])


def subtract_floor(points, floors):
    """Per realization, sqrt(value^2 - floor^2) (0 where the floor exceeds the value): removes the sampling contribution in quadrature."""
    return {n: [float(np.sqrt(max(v * v - f * f, 0.))) for v, f in zip(points[n], floors[n])] for n in points}


def fmt(fit, digits=2):
    """'slope +- s.e.' in LaTeX math."""
    if not np.isfinite(fit['se']): return 'n/a'
    return f'${fit["slope"]:.{digits}f}\\pm{fit["se"]:.{digits}f}$'


def alpha_lab(al):
    return fr'$\alpha\!=\!{al}$'



def quartiles(points):
    Ns = sorted(points)
    return (np.array(Ns), np.array([np.median(points[n]) for n in Ns]), np.array([np.percentile(points[n], 25) for n in Ns]),
            np.array([np.percentile(points[n], 75) for n in Ns]))


def band(ax, points, color, label=None, ls='-', marker='o', fill=True, **kw):
    """Median over realizations (markers, joined by a thin line) with a light interquartile band."""
    N, med, lo, hi = quartiles(points)
    if fill: ax.fill_between(N, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    kw.setdefault('lw', LW_DATA); kw.setdefault('ms', MS); kw.setdefault('zorder', 3)
    ax.plot(N, med, ls, color=color, marker=marker, label=label, **kw)


def floor_line(ax, points, color, label=None):
    """Sampling floor (from two halves of the trajectories): thin line, no markers."""
    N, med, _, _ = quartiles(points)
    ax.plot(N, med, '-', color=color, label=label, zorder=2, **FLOOR_KW)


def ref_law(ax, p, N0, y0, span=(400, 1e4), label=None):
    """Reference power law y0 (N/N0)^p over N in span: dotted gray, behind the data.
    label = (N, dx, dy, ha, va): the law's name at the point of the line at N, offset by (dx, dy) points; 'right': just beyond the
    right end of the line; None: no label."""
    N = np.geomspace(*span, 50)
    ax.plot(N, y0 * (N / N0) ** p, ':', color=REF_GRAY, lw=LW_REF, zorder=0, gid='ref')
    if label == 'right': label = (span[1], 1.5, 0, 'left', 'center')
    if label is not None:
        Nl, dx, dy, ha, va = label
        ax.annotate(REF_LABELS[p], (Nl, y0 * (Nl / N0) ** p), xytext=(dx, dy), textcoords='offset points', color=REF_TEXT, fontsize=KEY_FS, ha=ha, va=va)


def color_key(ax, items, x=0.97, y=0.97, ha='right', va='top', fs=KEY_FS, spacing=1.25):
    """Colour key as coloured text, stacked in the given order: items = [(text, colour), ...]; (x, y) in axes coordinates is the top
    (va='top') or bottom (va='bottom') of the stack; line spacing in points."""
    n = len(items)
    for k, (t, c) in enumerate(items):
        off = -k * spacing * fs if va == 'top' else (n - 1 - k) * spacing * fs
        ax.annotate(t, (x, y), xycoords='axes fraction', xytext=(0, off), textcoords='offset points', color=c, fontsize=fs, ha=ha, va=va)


def text_label(ax, x, y, text, color, dx=0, dy=0, ha='left', va='center'):
    """Direct label of a curve: text at data point (x, y), offset by (dx, dy) points, in the curve's colour."""
    ax.annotate(text, (x, y), xycoords='data', xytext=(dx, dy), textcoords='offset points', color=color, fontsize=KEY_FS, ha=ha, va=va)


def handle_key(ax, handles, **kw):
    """Line-vs-marker key with gray handles (no frame)."""
    kw = {**dict(fontsize=KEY_FS, borderaxespad=0.3, borderpad=0.1, handlelength=1.5, handletextpad=0.5, labelspacing=0.3), **kw}
    leg = ax.legend(handles=handles, **kw); leg.set_zorder(5)
    return leg


def floor_handle(label='sampling floor'):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=KEY_GRAY, label=label, **FLOOR_KW)


def data_ylim(ax):
    """Smallest and largest positive y of the data drawn in ax (lines and bands; reference laws excluded)."""
    ys = [np.asarray(l.get_ydata(), float) for l in ax.lines if l.get_gid() != 'ref']
    for col in ax.collections:
        for pth in col.get_paths(): ys.append(pth.vertices[:, 1])
    y = np.concatenate(ys); y = y[np.isfinite(y) & (y > 0)]
    return y.min(), y.max()


def two_decades(lo, hi, side=None):
    """y-limits widened, if needed, so that at least two decade ticks show: the side that needs less (or side='lo'/'hi') is extended
    just past the next decade."""
    if np.floor(np.log10(hi)) - np.ceil(np.log10(lo)) >= 1: return lo, hi
    down, up = np.log10(lo) - np.floor(np.log10(lo)), np.ceil(np.log10(hi)) - np.log10(hi)
    if side == 'lo' or (side is None and down <= up): return 10 ** np.floor(np.log10(lo)) / 10 ** 0.03, hi
    return lo, 10 ** np.ceil(np.log10(hi)) * 10 ** 0.03


def loglog(ax, xlabel=r'$N$', ylim=None, pad=0.08, xlim=(70, 1.45e4)):
    """Log-log axes with decade ticks. y-limits: ylim if given, else the data range widened by pad decades on each side."""
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(xlabel); ax.set_xlim(*xlim)
    if ylim is None:
        lo, hi = data_ylim(ax); ylim = (lo / 10 ** pad, hi * 10 ** pad)
    ax.set_ylim(*two_decades(*ylim))
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=9)); ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12)); ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def finish(fig, path, letters=(), dy=1.0, **savekw):
    """Freeze the (constrained) layout, put bold lowercase panel labels "(a)" [(ax, 'a'), ...] at the left edge of each axes' labels, with their
    tops level with the highest top of the axes or their titles (+ dy points) among the axes at the same height, and save at the
    figure size (= printed width)."""
    fig.canvas.draw(); fig.set_layout_engine('none'); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, s in letters:
        row = [a for a, _ in letters if abs(a.get_position().y1 - ax.get_position().y1) < 1e-3]
        titled = [a for a in row if a.get_title().strip()]
        x0 = inv.transform((ax.get_tightbbox(r).x0, 0))[0]
        if titled: y, va = max(a.title.get_window_extent(r).y1 for a in titled) + dy * fig.dpi / 72, 'top'      # level with the titles
        else: y, va = max(a.get_window_extent(r).y1 for a in row) + dy * fig.dpi / 72, 'bottom'               # just above the axes
        fig.text(x0, inv.transform((0, y))[1], rf'\textbf{{({s})}}', fontsize=PANEL_FS, ha='left', va=va)
    savekw.setdefault('dpi', 600); fig.savefig(path, **savekw)
