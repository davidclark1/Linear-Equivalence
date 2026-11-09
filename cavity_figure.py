"""Cavity-variable tests (Fig. 6 and Appendix C 7 of the paper) -> figs/appendix_figures.pdf
Reads data/cavity.pkl (extract_cache.py). Exponents: least squares on medians, N >= 464, bootstrap s.e. (paperstyle.pooled_fit), drawn on the panels -> data/exponents_cavity.json
Row A: standardized joint cumulants kappa_{p,q} of the cavity fields (left) and of the reservoir local fields (right) vs N.
Row B: Price's theorem at the three largest N (one realization, largest alpha).   Row C: F-kernel decoupling by number of shared indices."""
import os, pickle, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import paperstyle as ps
from matplotlib.lines import Line2D
ps.use()
SEED_PRICE = 0
PQ_LIST = [(1, 1), (1, 3), (2, 2)]
COLORS = ps.ALPHA_COLORS
DATA = pickle.load(open('data/cavity.pkl', 'rb'))
Ns = sorted({k[0] for k in DATA}); alphas = sorted({k[1] for k in DATA}); seeds = sorted({k[2] for k in DATA})
print(f'{len(DATA)} runs: N={Ns}, alpha={alphas}, seeds={seeds}')
HAS_FLOOR = all('floor' in r['cav'] for r in DATA.values())    # sampling floors of the runs in results/cavity_halves
print('sampling floors:', 'drawn' if HAS_FLOOR else 'not in the data')


def points(getter, alpha):
    d = {}
    for (N, a, s), r in DATA.items():
        if a == alpha: d.setdefault(N, []).append(getter(r))
    return d


REF_SPAN = (464, 2154)      # reference laws: N = 464 to 2154, above the data and below the exponent key (the floors lie below the data)


def scaling_panel(ax, getter, fac, labels=False, floor=None):
    """Median and IQR over realizations vs N for every alpha; reference power laws fanning from fac x the median of the largest alpha
    at N = 464, labelled just beyond their right ends if labels. With floors (runs of results/cavity_halves): every series is plotted and
    fitted floor-subtracted in quadrature per realization (paperstyle.subtract_floor; the floor reaches 0.3-0.96 of the value at N = 10000
    for most series at alpha = 200), and the median floor is drawn as a thin line in the series colour. Returns {('cav', alpha): fit}."""
    out = {}; sub = floor is not None and HAS_FLOOR; ZEROS.setdefault('n', 0)
    for alpha in alphas:
        pts = points(getter, alpha)
        if sub:
            F = points(floor, alpha); pts = ps.subtract_floor(pts, F); ZEROS['n'] += sum(1 for n in pts for x in pts[n] if x == 0)
        ps.band(ax, pts, COLORS[alpha], fill=(alpha == 800)); out[('cav', alpha)] = ps.pooled_fit(pts)
        if sub: ps.floor_line(ax, F, COLORS[alpha])
    pts0 = points(getter, max(alphas)); pts0 = ps.subtract_floor(pts0, points(floor, max(alphas))) if sub else pts0
    N, med, _, _ = ps.quartiles(pts0); i0 = list(N).index(REF_SPAN[0])
    for p, dy in ((-0.5, 3.5), (-1.0, 0.0), (-1.5, -3.5)):      # labels spread vertically: the three ends lie close together
        ps.ref_law(ax, p, N[i0], fac * med[i0], span=REF_SPAN, label=(REF_SPAN[1], 1.5, dy, 'left', 'center') if labels else None)
    ps.color_key(ax, [(ps.fmt(out[('cav', alpha)]), COLORS[alpha]) for alpha in alphas], x=1.0, y=0.99)      # fitted exponents
    return out


W, H = ps.TEXT_W, 6.1
fig = plt.figure(figsize=(W, H))
outer = gridspec.GridSpec(3, 1, hspace=0.55, height_ratios=[1, 0.85, 1], left=0.088, right=0.995, top=0.925, bottom=0.065)


def pair_axes(spec):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.16)
    pairs = []
    for g in range(3):
        pr = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[g], wspace=0.1)
        pairs.append((fig.add_subplot(pr[0]), fig.add_subplot(pr[1])))
    return pairs


def group_title(pair, text, dy=0.03):
    bl, br = pair[0].get_position(), pair[1].get_position()
    fig.text(0.5 * (bl.x0 + br.x1), max(bl.y1, br.y1) + dy, text, ha='center', va='bottom', fontsize=8)


def scaling_row(row, pad=0.08, padtop=1.0, refs=False):
    """Common log-log limits for a row of scaling panels (data range, with the reference laws if refs, + pad decades), tick labels on
    the first panel only."""
    lims = [ps.data_ylim(a) for pr in row for a in pr if a.axison]
    if refs: lims += [(l.get_ydata().min(), l.get_ydata().max()) for pr in row for a in pr for l in a.lines if l.get_gid() == 'ref']
    ylim = (min(l[0] for l in lims) / 10 ** pad, max(l[1] for l in lims) * 10 ** padtop)
    for g, pr in enumerate(row):
        for si, a in enumerate(pr):
            ps.loglog(a, ylim=ylim, xlim=(70, 1.45e4)); a.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
            if g or si: a.tick_params(labelleft=False)


SLOPES = {}
ZEROS = {}

# ------------------------------------------------------------------ Row A
FAC_A = {(pq, side): 6.0 for pq in ((1, 1), (1, 3), (2, 2)) for side in ('cav', 'res')}     # reference laws below (< 1) or above (> 1) the data
axA = pair_axes(outer[0])
for g, pq in enumerate(PQ_LIST):
    for si, side in enumerate(('cav', 'res')):
        ax = axA[g][si]
        st = scaling_panel(ax, lambda d, pq=pq, side=side: d[side]['kappa_rms'][pq], FAC_A[pq, side], labels=(g == 1 and si == 0),      # labels where the data fall away below the fan
                           floor=lambda d, pq=pq, side=side: d[side]['floor']['kappa_rms'][pq])
        for (_, alpha), f in st.items(): SLOPES[f'A/kappa{pq[0]}{pq[1]}/{side}/a{alpha}'] = f
        ax.set_title(side, fontsize=8, pad=2)
scaling_row(axA, pad=0.9, padtop=1.6, refs=True)
ps.color_key(axA[0][0], [(ps.alpha_lab(a), COLORS[a]) for a in alphas], x=0.03, y=0.02, ha='left', va='bottom')
if HAS_FLOOR: ps.handle_key(axA[0][1], [ps.floor_handle()], loc='lower left')      # placement to be finalized with the figure
axA[0][0].set_ylabel('Off-diagonal RMS of $\\kappa_{p,q}$' + (',\nfloor subtracted' if HAS_FLOOR else ''))
for g, pq in enumerate(PQ_LIST):
    group_title(axA[g], r'$\kappa_{%d,%d}$' % pq)

# ------------------------------------------------------------------ Row B (Price's theorem)
axB = pair_axes(outer[1])
alpha_price = max(alphas)
for g, N in enumerate(Ns[-3:]):
    d = DATA[(N, alpha_price, SEED_PRICE)]; sN = np.sqrt(N)
    arrs = {sd: (sN * d[sd]['price']['empirical'], sN * d[sd]['price']['predicted'], d[sd]['price']['slope']) for sd in ('cav', 'res')}
    lim = 1.05 * max(np.abs(np.concatenate([v for sd in arrs for v in arrs[sd][:2]])).max(), 1e-12)
    for si, side in enumerate(('cav', 'res')):
        ax = axB[g][si]; xs, ys, sl = arrs[side]
        ax.plot([-lim, lim], [-lim, lim], '--', color='#777777', lw=0.6, zorder=1)
        ax.axhline(0, color='#bbbbbb', lw=0.5, zorder=0); ax.axvline(0, color='#bbbbbb', lw=0.5, zorder=0)
        ax.scatter(xs, ys, s=1.2, color=COLORS[alpha_price], lw=0, rasterized=True, zorder=2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        txt = f'slope {sl:.3f}'
        if HAS_FLOOR: txt += '\n' + r'$\pm$ %.4f' % d[side]['floor']['price_slope']      # sampling floor of the slope (two IC halves)
        ax.text(0.96, 0.04, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=ps.KEY_FS, linespacing=1.1)      # empty lower-right corner, off the diagonal
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, symmetric=True)); ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, symmetric=True))
        ax.set_xlabel('empirical'); ax.set_title(side, fontsize=8, pad=2)
        if si == 1 or g > 0: ax.tick_params(labelleft=False)
        SLOPES[f'B/price_slope/{side}/N{N}'] = float(sl)
    group_title(axB[g], fr'$N={N}$')
axB[0][0].set_ylabel("Price prediction")

# ------------------------------------------------------------------ Row C (F-kernel decoupling; cavity analogues)
axC = pair_axes(outer[2])
for g, c in enumerate((0, 1, 2)):
    for si, (key, lab) in enumerate((('prop2_off', 'off-diagonal'), ('prop2_diag', 'diagonal'))):
        ax = axC[g][si]
        if c not in next(iter(DATA.values()))['cav'][key]:
            ax.axis('off'); continue
        st = scaling_panel(ax, lambda d, key=key, c=c: d["cav"][key][c], 10.0, labels=(g == 0 and si == 0),      # reference laws above the data, labels clear of it
                           floor=lambda d, key=key, c=c: d['cav']['floor'][key][c])
        for (_, alpha), f in st.items(): SLOPES[f'C/overlap{c}/{key}/a{alpha}'] = f
        ax.set_title(lab, fontsize=8, pad=2)
scaling_row(axC, padtop=1.4, refs=True)
axC[0][0].set_ylabel('RMS of $\\kappa_t(\\hat{F}, \\hat\\eta\\hat\\eta)$' + (',\nfloor subtracted' if HAS_FLOOR else ''))
for g, c in enumerate((0, 1, 2)): group_title(axC[g], f'overlap = {c}')

for lab, row in zip('ABC', (axA, axB, axC)):
    b = row[0][0].get_position(); fig.text(0.004, b.y1 + 0.032, rf'\textbf{{({lab.lower()})}}', fontsize=ps.PANEL_FS, ha='left', va='bottom')
os.makedirs('figs', exist_ok=True)
fig.savefig('figs/appendix_figures.pdf', dpi=600); print('saved figs/appendix_figures.pdf')
print('realizations with floor >= value (subtracted value 0):', ZEROS.get('n'))
json.dump(SLOPES, open('data/exponents_cavity.json', 'w'), indent=1)
for k, v in SLOPES.items():
    if not isinstance(v, dict): print(f'{k:40s} {v:.3f}')
