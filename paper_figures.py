"""The data figures of the paper (Figs. 2-4 and 7-10), in one style (paperstyle.py), from the small files under data/ (extract_cache.py).

    python paper_figures.py [scaling] [nzm] [cij] [autocov] [lags] [moments] [noise]       (default: all)

scaling -> Fig. 3, nzm -> Fig. 4, cij -> Fig. 2, autocov -> Fig. 9, lags -> Fig. 8, moments -> Fig. 7, noise -> Fig. 10.
Writes figs/{scaling_grid,nzm_scaling,cij_tau_vs_N,autocorrelation_diagnostic,lags_diagonal,moments_scaling,noise_drive}.pdf.
The cavity figure (Fig. 6) is cavity_figure.py; the schematics (Figs. 1 and 5) are schematics/schematics.py.
Exponents (least squares on medians, N >= 464, bootstrap s.e.) are written to data/exponents_figs.json (Table I: make_tables.py)."""
import json, pickle, sys, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle as ps

ps.use()
WHAT = [a for a in sys.argv[1:] if not a.startswith('-')] or ['scaling', 'nzm', 'cij', 'autocov', 'lags', 'moments', 'noise']
OUT = 'figs'
os.makedirs(OUT, exist_ok=True)
EXP = {}


def group(rows, key):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: d[r['case']][r['N']].append(key(r))
    return d


# ----------------------------------------------------------------------------------------------- Fig. 3, scaling_grid (g = 2.5)
SCS_ALPHAS = (200, 800, 3200)

def fig_scaling():
    A = json.load(open('data/scalars_moments.json'))
    err = group(A, lambda r: r['rms_err_dmft']); rel = group(A, lambda r: r['rms_err_dmft'] / r['rms_C'])
    res = group(A, lambda r: r['rms_CD_star']); resi = group(A, lambda r: r['rms_CD'])
    sig = group(A, lambda r: r['rms_C'])
    fl = group(A, lambda r: r['rms_noise']); flrel = group(A, lambda r: r['rms_noise'] / r['rms_C']); flD = group(A, lambda r: r['rms_noise_D_star']); fli = group(A, lambda r: r['rms_noise_D'])
    resF = {c: ps.subtract_floor(res[c], flD[c]) for c in res}      # floor subtracted in quadrature, per realization: plotted and fitted
    fig, ax = plt.subplots(1, 3, figsize=(ps.TEXT_W, 2.45), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)
    # per panel: quantity, floor, y-label, reference fan (from N0, y0 to N = 5000), in the empty space above the data and below the key
    panels = [(ax[0], err, fl, 'Off-diagonal RMS of\n$C^\\phi_{ij}(\\tau{=}0) - \\bar{C}^\\phi_{ij}(\\tau{=}0)$', 'err', (110, 0.032), (-0.5, -1.0)),
              (ax[1], rel, flrel, 'Relative RMS error', 'rel', (1500, 1.1), (-0.5, -1.0)),
              (ax[2], resF, flD, 'Off-diagonal RMS of\n$C^\\Delta_{ij}(\\tau{=}0)$,\nfloor subtracted', 'resid', (1000, 0.011), (-0.5, -1.0, -1.5))]
    for k, (a_, Q, F, yl, name, (N0, y0), laws) in enumerate(panels):
        key = []
        for al in SCS_ALPHAS:
            c = f'pm_a{al}'; f = ps.pooled_fit(Q[c]); EXP[f'{c}/{name}'] = f
            key.append((f'{ps.alpha_lab(al)}: {ps.fmt(f)}', ps.ALPHA_COLORS[al]))
            ps.band(a_, Q[c], ps.ALPHA_COLORS[al], fill=(al == 800)); ps.floor_line(a_, F[c], ps.ALPHA_COLORS[al])
        if k == 0:      # the off-diagonal RMS of C_ij(0) itself, for scale (as in Fig. 4(b)); nearly independent of alpha, so alpha = 800 only
            ps.band(a_, sig['pm_a800'], ps.C_SIGNAL, marker='s', fill=False, zorder=2); key.append((r'$C^\phi_{ij}(\tau{=}0)$', ps.C_SIGNAL))
        for p in laws: ps.ref_law(a_, p, N0, y0, span=(N0, 400 if k == 0 else 5000), label='right')      # (a): fan in the gap between the data and the floors at small N
        ps.loglog(a_); a_.set_ylabel(yl); a_.set_title(' ', fontsize=ps.PANEL_FS)
        ps.color_key(a_, key)
        if k == 0: ps.handle_key(a_, [ps.floor_handle()], loc='lower left')
    for al in SCS_ALPHAS:      # measured site gains: numbers only (Appendix I 5)
        c = f'pm_a{al}'; EXP[f'{c}/resid_site'] = ps.pooled_fit(ps.subtract_floor(resi[c], fli[c]))
    ps.finish(fig, f'{OUT}/scaling_grid.pdf', letters=list(zip(ax, 'abc'))); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 4, nzm_scaling
NZM_ROWS = [('sig', 'Sigmoid'), ('sigbias', 'Sigmoid, biases'), ('ca_km2', 'Hebbian, $k=-2$'), ('ca_kp2', 'Hebbian, $k=+2$'), ('glvsat', 'Lotka--Volterra')]      # order of Sec. VII

def fig_nzm():
    P = json.load(open('data/scalars_prod_tap.json'))      # one run set for the whole figure (results/prod_tap)
    Q = {k: group(P, f) for k, f in dict(meaneq=lambda r: r['meaneq'], resid=lambda r: r['rms_CD'], floorD=lambda r: r['rms_noise_D'],
         err_emp=lambda r: r['rms_err_emp'], err_th=lambda r: r['rms_err_th'], err_tap=lambda r: r['rms_err_tap'], floorM=lambda r: r['floor_meaneq'], signal=lambda r: r['rms_C'], floor=lambda r: r['rms_noise'],
         rel_emp=lambda r: r['rms_err_emp'] / r['rms_C'], rel_th=lambda r: r['rms_err_th'] / r['rms_C'], rel_tap=lambda r: r['rms_err_tap'] / r['rms_C'], rel_floor=lambda r: r['rms_noise'] / r['rms_C']).items()}
    fig, ax = plt.subplots(len(NZM_ROWS), 4, figsize=(0.87 * ps.TEXT_W, 7.4), sharex=True, sharey='col', layout='constrained')      # printed at 0.87 textwidth
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.07, hspace=0.06)
    span = (450, 9000); XL = (380, 1.2e4)
    def note(a_, fits, y=0.03):      # fitted exponents as coloured text, lower left
        ps.color_key(a_, [(ps.fmt(f), col) for f, col in fits], x=0.04, y=y, ha='left', va='bottom')
    def ref(a_, Q_, law, fac):      # predicted power law, a factor fac from the median at the smallest N
        N, med, _, _ = ps.quartiles(Q_); ps.ref_law(a_, law, N[0], med[0] * fac, span=span); return N[0], med[0] * fac
    for i, (case, title) in enumerate(NZM_ROWS):
        fits = {k: ps.pooled_fit(Q[k][case]) for k in ('meaneq', 'err_emp', 'err_th', 'err_tap', 'rel_emp', 'rel_th', 'rel_tap', 'signal')}
        fits['resid'] = ps.pooled_fit(ps.subtract_floor(Q['resid'][case], Q['floorD'][case]))     # residual: floor subtracted, as in Fig. 3
        for k, f in fits.items(): EXP[f'{case}/{k}'] = f
        a_ = ax[i, 0]; ps.band(a_, Q['meaneq'][case], ps.C_SINGLE); ps.floor_line(a_, Q['floorM'][case], ps.C_SINGLE); rA = ref(a_, Q['meaneq'][case], -0.5, 1.6); note(a_, [(fits['meaneq'], ps.C_SINGLE)])
        RF = ps.subtract_floor(Q['resid'][case], Q['floorD'][case])      # floor subtracted per realization, as in Fig. 3(c) and the fit
        a_ = ax[i, 3]; ps.band(a_, RF, ps.C_SINGLE); ps.floor_line(a_, Q['floorD'][case], ps.C_SINGLE)
        rD = ref(a_, RF, -1.0, 1.6); note(a_, [(fits['resid'], ps.C_SINGLE)])
        a_ = ax[i, 1]; ps.band(a_, Q['signal'][case], ps.C_SIGNAL, marker='s', fill=False); ps.band(a_, Q['err_emp'][case], ps.C_MEASURED); ps.band(a_, Q['err_th'][case], ps.C_THEORY, marker='^', fill=False); ps.band(a_, Q['err_tap'][case], ps.C_TAP, marker='D', ms=2.2, fill=False)
        ps.floor_line(a_, Q['floor'][case], ps.C_SINGLE); rB = ref(a_, Q['err_emp'][case], -1.0, 0.5); note(a_, [(fits['err_emp'], ps.C_MEASURED), (fits['err_th'], ps.C_THEORY), (fits['err_tap'], ps.C_TAP)])
        a_ = ax[i, 2]; ps.band(a_, Q['rel_emp'][case], ps.C_MEASURED); ps.band(a_, Q['rel_th'][case], ps.C_THEORY, marker='^', fill=False); ps.band(a_, Q['rel_tap'][case], ps.C_TAP, marker='D', ms=2.2, fill=False); ps.floor_line(a_, Q['rel_floor'][case], ps.C_SINGLE)
        rC = ref(a_, Q['rel_emp'][case], -0.5, 1.5); note(a_, [(fits['rel_emp'], ps.C_MEASURED), (fits['rel_th'], ps.C_THEORY), (fits['rel_tap'], ps.C_TAP)], y=0.23)      # above the floor
        if i == 0:      # keys and labels in the first row, each in the first panel it applies to
            ps.text_label(ax[0, 0], 3000, rA[1] * (3000 / rA[0]) ** -0.5, ps.REF_LABELS[-0.5], ps.REF_TEXT, dy=5, ha='center', va='bottom')
            ps.text_label(ax[0, 3], 2500, rD[1] * (2500 / rD[0]) ** -1.0, ps.REF_LABELS[-1.0], ps.REF_TEXT, dx=2, dy=2, ha='left', va='bottom')
            ps.text_label(ax[0, 1], 1100, rB[1] * (1100 / rB[0]) ** -1.0, ps.REF_LABELS[-1.0], ps.REF_TEXT, dx=-3, dy=-0.5, ha='right', va='top')      # (b) and (c) have their own laws, so each is labeled
            ps.text_label(ax[0, 2], 2500, rC[1] * (2500 / rC[0]) ** -0.5, ps.REF_LABELS[-0.5], ps.REF_TEXT, dx=2, dy=2, ha='left', va='bottom')
            ps.color_key(ax[0, 1], [('measured', ps.C_MEASURED), ('single-site', ps.C_THEORY), ('TAP-solved', ps.C_TAP)], x=0.98, y=1.0)
            Ns, ms, _, _ = ps.quartiles(Q['signal'][case]); ps.text_label(ax[0, 1], Ns[0], ms[0], r'$C^{\breve{\phi}}_{ij}(\tau{=}0)$', ps.C_SIGNAL, dy=3, ha='left', va='bottom')      # direct label above the curve's left end
            Nf, mf, _, _ = ps.quartiles(Q['floorM'][case]); ps.text_label(ax[0, 0], Nf[-1], mf[-1], 'sampling\nfloor', ps.KEY_GRAY, dy=-3, ha='right', va='top')      # direct label, as 'signal'
        ax[i, 0].set_ylabel(title)
    pads = [(0.75, 0.4), (1.05, 0.85), (0.08, 0.08), (0.3, 0.08)]      # (bottom, top) margins in decades per column, room for the keys
    for j in range(4):
        lims = [ps.data_ylim(ax[i, j]) for i in range(len(NZM_ROWS))]
        ylim = ps.two_decades(min(l[0] for l in lims) / 10 ** pads[j][0], max(l[1] for l in lims) * 10 ** pads[j][1])
        for i in range(len(NZM_ROWS)): ps.loglog(ax[i, j], xlabel=r'$N$' if i == len(NZM_ROWS) - 1 else '', xlim=XL, ylim=ylim)
    for j, t in enumerate(['RMS over sites of\n$\\bar{\\phi}_i - \\mathcal{M}_{\\star}(\\bar{h}_i)$', 'Off-diagonal RMS of\n$C^{\\breve{\\phi}}_{ij}(\\tau{=}0) - \\bar{C}^{\\breve{\\phi}}_{ij}(\\tau{=}0)$',
                          'Relative RMS error', 'Off-diagonal RMS of\n$C^{\\breve{\\Delta}}_{ij}(\\tau{=}0)$,\nfloor subtracted']):
        ax[0, j].set_title(t)
    ps.finish(fig, f'{OUT}/nzm_scaling.pdf', letters=[(ax[0, j], 'abcd'[j]) for j in range(4)]); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 2, cij_tau_vs_N, and Fig. 9, autocorrelation_diagnostic
NS_SHOWN = (464, 1000, 2154, 4642, 10000)
CIJ_W, CIJ_TICK = 0.92, 4      # Fig. 2: printed width (textwidth units; the includegraphics width in the paper), y-tick spacing

def fig_cij():
    Cv = pickle.load(open('data/fig2_curves.pkl', 'rb'))      # fig2_pairs.py: pairs at evenly spaced quantiles of the simulated C_ij(0)
    fig, ax = plt.subplots(1, len(NS_SHOWN), figsize=(CIJ_W * ps.TEXT_W, 1.6), sharey=True, layout='constrained')      # printed at CIJ_W textwidth
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03)
    ymax = 0.
    for a_, N in zip(ax, NS_SHOWN):
        c = Cv[N]; lags = c['lags']; tau = np.concatenate([-lags[:0:-1], lags])
        for k, p in enumerate(c['pairs']):      # colour k = quantile k, the same at every N
            y_sim = np.concatenate([c['sim'][p][1][:0:-1], c['sim'][p][0]]); y_th = np.concatenate([c['ansatz'][p][1][:0:-1], c['ansatz'][p][0]])
            a_.plot(tau, np.sqrt(N) * y_sim, '-', color=ps.tint(ps.PAIR_COLORS[k]), lw=ps.SIM_LW, solid_capstyle='round', zorder=2)     # simulation: thick tint
            a_.plot(tau, np.sqrt(N) * y_th, '-', color=ps.PAIR_COLORS[k], lw=ps.LW_THEORY, zorder=3)                                       # theory: thin line on top
            ymax = max(ymax, np.abs(np.sqrt(N) * y_sim).max(), np.abs(np.sqrt(N) * y_th).max())
        a_.axvline(0, color='#bbbbbb', lw=0.5, zorder=0); a_.axhline(0, color='#bbbbbb', lw=0.5, zorder=0)
        a_.set_title(f'$N={N}$'); a_.set_xlabel(r'$\tau$'); a_.set_xticks([-10, 0, 10]); a_.set_xlim(-10.8, 10.8)
    ax[0].set_ylabel(r'$\sqrt{N}\,C^\phi_{ij}(\tau)$'); ax[0].set_ylim(-1.08 * ymax, 1.08 * ymax); ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(CIJ_TICK))
    fig.legend(handles=[Line2D([], [], color=ps.KEY_GRAY, lw=ps.LW_THEORY, label=r'theory,' + '\n' + r'$\bar{C}^\phi_{ij}(\tau)$'),
                        Line2D([], [], color=ps.tint(ps.KEY_GRAY), lw=ps.SIM_LW, solid_capstyle='round', label='simulation')],
               loc='outside left center', fontsize=ps.KEY_FS, handlelength=1.2, handletextpad=0.4, labelspacing=0.8, frameon=False)      # key left of the panels
    ps.finish(fig, f'{OUT}/cij_tau_vs_N.pdf'); plt.close(fig)

def fig_autocov():
    Cv = pickle.load(open('data/moments_curves.pkl', 'rb'))
    fig, ax = plt.subplots(1, len(NS_SHOWN), figsize=(ps.TEXT_W, 1.55), sharey=True, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.04)
    for a_, N in zip(ax, NS_SHOWN):
        c = Cv[N]; col = ps.ALPHA_COLORS[800]; t = np.asarray(c['tau']); every = int(round(0.5 / (t[1] - t[0])))     # markers every 0.5 in tau
        a_.fill_between(t, c['ac_mean'] - c['ac_std'], c['ac_mean'] + c['ac_std'], color=col, alpha=0.18, lw=0, zorder=1)
        if c['C_star'] is not None: a_.plot(t, c['C_star'], '-', color='k', lw=ps.LW_THEORY, zorder=2)
        a_.plot(t[::every], np.asarray(c['ac_mean'])[::every], color=col, zorder=3, **ps.OPEN)
        a_.axhline(0, color='#bbbbbb', lw=0.5, zorder=0)
        a_.set_title(f'$N={N}$'); a_.set_xlabel(r'$\tau$'); a_.set_xlim(-0.3, 10.3); a_.set_xticks([0, 5, 10])
    ax[0].set_ylabel(r'$C^\phi_{ii}(\tau)$'); ax[0].set_ylim(-0.06, 0.72); ax[0].set_yticks([0, 0.2, 0.4, 0.6])
    ps.handle_key(ax[0], [Line2D([], [], color=ps.KEY_GRAY, lw=ps.LW_THEORY, label=r'theory, $C^\phi_\star(\tau)$'),
                          Line2D([], [], color=ps.KEY_GRAY, label='simulation', **ps.OPEN)], loc='upper right', handlelength=1.2, handletextpad=0.4)
    ps.finish(fig, f'{OUT}/autocorrelation_diagnostic.pdf'); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 8, lags_diagonal
def fig_lags(alpha=800):
    R = [r for r in json.load(open('data/lags_moments.json')) if r['case'] == f'pm_a{alpha}']
    def pts(f):
        d = collections.defaultdict(list)
        for r in R: d[r['N']].append(f(r))
        return d
    fig, ax = plt.subplots(1, 3, figsize=(ps.TEXT_W, 2.45), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.06)
    key = {0: [], 1: [], 2: []}
    for tau, col in ps.LAG_COLORS.items():      # interquartile band at tau = 0 only (those at the other lags are similar)
        l = R[0]['lags'].index(tau)
        for k, key_, name in ((0, 'err_lags', 'err'), (2, 'diag_err_lags', 'diag')):
            P = pts(lambda r: r[key_][l]); f = ps.pooled_fit(P); EXP[f'pm_a{alpha}/{name}_tau{tau:g}'] = f
            ps.band(ax[k], P, col, fill=(tau == 0)); key[k].append((fr'$\tau\!=\!{tau:g}$: {ps.fmt(f)}', col))
        P = pts(lambda r: r['err_lags'][l] / r['C_lags'][l]); f = ps.pooled_fit(P); EXP[f'pm_a{alpha}/rel_tau{tau:g}'] = f
        ps.band(ax[1], P, col, fill=(tau == 0)); key[1].append((fr'$\tau\!=\!{tau:g}$: {ps.fmt(f)}', col))
    for tau, col in ps.LAG_COLORS.items():      # sampling floors at every lag (two-half accumulators of pm2_a800), in the lag's colour
        l = R[0]['lags'].index(tau)
        ps.floor_line(ax[0], pts(lambda r: r['floor_lags'][l]), col); ps.floor_line(ax[1], pts(lambda r: r['floor_lags'][l] / r['C_lags'][l]), col)
        ps.floor_line(ax[2], pts(lambda r: r['diag_floor_lags'][l]), col)
    P = pts(lambda r: r['diag_spread']); ps.band(ax[2], P, ps.C_SIGNAL, marker='s', fill=False, zorder=2)      # site-to-site s.d. of C_ii(0), for scale
    EXP[f'pm_a{alpha}/diag_spread'] = ps.pooled_fit(P)
    # per panel: y-label, reference fan (from N0, y0 to N = 5000) in empty space: above the data in (a), (b), between data and floor in (c)
    for k, (a_, yl, (N0, y0)) in enumerate(((ax[0], 'Off-diagonal RMS of\n$C^\\phi_{ij}(\\tau) - \\bar{C}^\\phi_{ij}(\\tau)$', (2000, 0.09)),
                                            (ax[1], 'Relative RMS error', (2000, 0.75)),
                                            (ax[2], 'RMS over sites of\n$C^\\phi_{ii}(\\tau) - \\bar{C}^\\phi_{ii}(\\tau)$', (1000, 0.0068)))):
        for p in (-0.5, -1.0): ps.ref_law(a_, p, N0, y0, span=(N0, 5000), label='right')
        ps.loglog(a_); a_.set_ylabel(yl); a_.set_title(' ', fontsize=ps.PANEL_FS)
        ps.color_key(a_, key[k])
        if k == 0: ps.handle_key(a_, [ps.floor_handle()], loc='lower left')
        if k == 2: ps.handle_key(a_, [Line2D([], [], color=ps.KEY_GRAY, marker='s', label='standard deviation\nof $C^\\phi_{ii}(\\tau{=}0)$')], loc='lower left')
    ps.finish(fig, f'{OUT}/lags_diagonal.pdf', letters=list(zip(ax, 'abc'))); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 7, moments_scaling (two-site moments of odd powers)
def fig_moments():
    if not os.path.exists('data/scalars_moments.json'): print('no data/scalars_moments.json'); return
    A = json.load(open('data/scalars_moments.json')); alphas = sorted({r['alpha_nominal'] for r in A})
    fig, ax = plt.subplots(1, 2, figsize=(0.55 * ps.TEXT_W, 2.7), layout='constrained', gridspec_kw=dict(width_ratios=(1.2, 1)))      # printed at 0.55 textwidth (one-column appendix)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.05)
    for key, mk in (('C31', 'o'), ('C33', '^')):
        for al in alphas:
            c = f'pm_a{al}'; col = ps.ALPHA_COLORS[al]
            E = group(A, lambda r: r[f'rms_err_{key}'])[c]; Rl = group(A, lambda r: r[f'rms_err_{key}'] / r[f'rms_{key}'])[c]
            fe, fr = ps.pooled_fit(E), ps.pooled_fit(Rl); EXP[f'{c}/err_{key}'] = fe; EXP[f'{c}/rel_{key}'] = fr; EXP[f'{c}/signal_{key}'] = ps.pooled_fit(group(A, lambda r: r[f'rms_{key}'])[c])
            band = key == 'C31' and al == 800      # interquartile band for (p, q) = (3, 1) at alpha = 800 only (the others are similar)
            ps.band(ax[0], E, col, marker=mk, fill=band); ps.band(ax[1], Rl, col, marker=mk, fill=band)
            ps.floor_line(ax[0], group(A, lambda r: r[f'rms_noise_{key}'])[c], col); ps.floor_line(ax[1], group(A, lambda r: r[f'rms_noise_{key}'] / r[f'rms_{key}'])[c], col)
    for a_, yl, y0 in ((ax[0], 'Off-diagonal RMS of\n$C^{(p,q)}_{ij}(\\tau{=}0) - \\bar{C}^{(p,q)}_{ij}(\\tau{=}0)$', 0.09), (ax[1], 'Relative RMS error', 1.2)):      # reference fan above the data, below the key
        for p in (-0.5, -1.0): ps.ref_law(a_, p, 1500, y0, span=(1500, 4000), label='right')
        ps.loglog(a_); a_.set_ylabel(yl); a_.set_title(' ', fontsize=ps.PANEL_FS)
    lo, hi = ax[0].get_ylim(); ax[0].set_ylim(lo / 10 ** 0.45, hi * 10 ** 0.15)      # room for the keys
    lo, hi = ax[1].get_ylim(); ax[1].set_ylim(lo / 10 ** 0.1, hi * 10 ** 0.6)
    for a_, nm in ((ax[0], 'err'), (ax[1], 'rel')):      # exponents of (p, q) = (3, 1); those of (3, 3) are within 0.01
        ps.color_key(a_, [(f'{ps.alpha_lab(al)}: {ps.fmt(EXP[f"pm_a{al}/{nm}_C31"])}', ps.ALPHA_COLORS[al]) for al in alphas])
    leg = ps.handle_key(ax[0], [Line2D([], [], color=ps.KEY_GRAY, marker='o', label=r'$(p,q)=(3,1)$'), Line2D([], [], color=ps.KEY_GRAY, marker='^', label=r'$(3,3)$'), ps.floor_handle()],
                        loc='lower left', handlelength=1.3, handletextpad=0.4)
    ps.finish(fig, f'{OUT}/moments_scaling.pdf', letters=list(zip(ax, 'ab'))); plt.close(fig)


# ----------------------------------------------------------------------------------------------- Fig. 10, noise_drive (SCS with white-noise drive)
GEFF_COLORS = {0.0: ps.ALPHA_COLORS[3200], 1.5: '#08519c', 2.0: '#3182bd', 2.5: '#5fa8dc'}      # undriven = pm_a3200 (its Fig. 3 colour); driven: blues, lighter = further from marginality

def fig_noise():
    import noise_numbers as nn
    R = nn.load()
    fig, ax = plt.subplots(1, 3, figsize=(ps.TEXT_W, 2.45), layout='constrained', gridspec_kw=dict(width_ratios=(1, 1, 0.8)))
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.05)
    panels = [(ax[0], 'err', 'floor', 'Off-diagonal RMS of\n$C^\\phi_{ij}(\\tau{=}0) - \\bar{C}^\\phi_{ij}(\\tau{=}0)$,\nfloor subtracted', (1500, 0.08), (-0.5, -1.0, -1.5), 6.0),
              (ax[1], 'rel', 'relfloor', 'Relative RMS error,\nfloor subtracted', (1500, 0.9), (-0.5, -1.0), 10.0)]
    fits = {}
    for k, (a_, name, fname, yl, (N0, y0), laws, top) in enumerate(panels):      # top: y-limit with room for the key above the data
        key = []
        for case, sx in nn.CASES:
            G = {q: nn.grp(R, case, nn.Q[q]) for q in ('err', 'floor', 'signal', 'resid', 'floorD')}
            if name == 'err': P, F = ps.subtract_floor(G['err'], G['floor']), G['floor']
            elif name == 'rel':
                E = ps.subtract_floor(G['err'], G['floor']); P = {n: [e / c for e, c in zip(E[n], G['signal'][n])] for n in E}
                F = {n: [f / c for f, c in zip(G['floor'][n], G['signal'][n])] for n in E}
            else: P, F = ps.subtract_floor(G['resid'], G['floorD']), G['floorD']
            f = ps.pooled_fit(P); fits[name, sx] = f; EXP[f'noise_s{sx:g}/{name}_sub'] = f
            ge = np.median([r['g_eff_star'] for r in R if r['case'] == case]); col = GEFF_COLORS[sx]
            key.append((fr'${ge:.2f}$: {ps.fmt(f)}', col))
            ps.band(a_, P, col, fill=(sx == 2.0)); ps.floor_line(a_, F, col)
        for p in laws: ps.ref_law(a_, p, N0, y0, span=(N0, 5000), label='right')
        lo, _ = ps.data_ylim(a_); ps.loglog(a_, ylim=(lo / 10 ** 0.08, top)); a_.set_ylabel(yl); a_.set_title(' ', fontsize=ps.PANEL_FS)
        ps.color_key(a_, ([(r'$g_{\mathrm{eff}}$', ps.KEY_GRAY)] if k == 0 else []) + key)
        if k == 0: ps.handle_key(a_, [ps.floor_handle()], loc='lower left')
    a_ = ax[2]; ge = {sx: np.median([r['g_eff_star'] for r in R if r['case'] == c]) for c, sx in nn.CASES}
    for name, mk, pred in (('err', 'o', -1.0), ('rel', 's', -0.5)):
        a_.axhline(pred, ls='--', color=ps.REF_GRAY, lw=ps.LW_REF, zorder=0)
        a_.plot([ge[s] for _, s in nn.CASES], [fits[name, s]['slope'] for _, s in nn.CASES], '-', color=ps.KEY_GRAY, lw=0.5, zorder=1)
        for _, sx in nn.CASES:
            a_.errorbar(ge[sx], fits[name, sx]['slope'], yerr=fits[name, sx]['se'], fmt=mk, color=GEFF_COLORS[sx], ms=ps.MS, elinewidth=0.6, capsize=1.5, zorder=3)
    a_.text(0.775, -1.01, '$-1$', color=ps.REF_TEXT, fontsize=ps.KEY_FS, ha='left', va='top'); a_.text(0.775, -0.51, '$-1/2$', color=ps.REF_TEXT, fontsize=ps.KEY_FS, ha='left', va='top')
    a_.set_xlim(0.77, 0.98); a_.set_ylim(-1.15, -0.25); a_.set_xticks([0.8, 0.9]); a_.set_xlabel(r'$g_{\mathrm{eff}}$'); a_.set_ylabel('Fitted exponent'); a_.set_title(' ', fontsize=ps.PANEL_FS)
    ps.handle_key(a_, [Line2D([], [], color=ps.KEY_GRAY, marker='o', ls='', label='error'), Line2D([], [], color=ps.KEY_GRAY, marker='s', ls='', label='relative error')], loc='center right', handletextpad=0.3)
    ps.finish(fig, f'{OUT}/noise_drive.pdf', letters=list(zip(ax, 'abc'))); plt.close(fig)


if __name__ == '__main__':
    for name, fn in (('scaling', fig_scaling), ('nzm', fig_nzm), ('cij', fig_cij), ('autocov', fig_autocov), ('lags', fig_lags), ('moments', fig_moments), ('noise', fig_noise)):
        if name in WHAT: fn(); print('made', name, flush=True)
    if EXP:
        old = json.load(open('data/exponents_figs.json')) if os.path.exists('data/exponents_figs.json') else {}
        old.update(EXP); json.dump(old, open('data/exponents_figs.json', 'w'), indent=1)
