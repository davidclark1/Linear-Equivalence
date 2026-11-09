"""Noise-driven SCS model (data/scalars_noise.json) next to the undriven pm_a3200: exponents against g_eff (Appendix I and Fig. 10).
load() is used by paper_figures.py (Fig. 10).  Usage: python noise_numbers.py   (prints the numbers quoted in the paper)"""
import json, collections, numpy as np, paperstyle as ps
CASES = [('pm_a3200', 0.0), ('scsn15a3200', 1.5), ('scsn2a3200', 2.0), ('scsn25a3200', 2.5)]
def load():
    R = [r for r in json.load(open('data/scalars_moments.json')) if r['case'] == 'pm_a3200'] + json.load(open('data/scalars_noise.json'))
    for r in R:
        r.setdefault('sig_xi', 0.0)
        if 'g_eff_star' not in r: r['g_eff_star'] = r['g'] * r['beta_star']
    return R
def grp(R, case, f):
    d = collections.defaultdict(list)
    for r in R:
        if r['case'] == case: d[r['N']].append(f(r))
    return d
Q = dict(err=lambda r: r['rms_err_dmft'], rel=lambda r: r['rms_err_dmft'] / r['rms_C'], signal=lambda r: r['rms_C'], floor=lambda r: r['rms_noise'],
         relfloor=lambda r: r['rms_noise'] / r['rms_C'], resid=lambda r: r['rms_CD_star'], floorD=lambda r: r['rms_noise_D_star'], lyap=lambda r: r['lyap'])
if __name__ == '__main__':
    R = load(); L = []; w = L.append; f = lambda x: f'{x["slope"]:.2f} +- {x["se"]:.2f}'
    w('Noise-driven SCS model (cases scsn{15,2,25}a3200, 5 realizations, alpha = 3200; undriven = pm_a3200)')
    for case, sx in CASES:
        G = {k: grp(R, case, q) for k, q in Q.items()}; ge = np.median([r['g_eff_star'] for r in R if r['case'] == case])
        sub = lambda a, b: ps.subtract_floor(G[a], G[b])
        fe, fr, fs = ps.pooled_fit(G['err']), ps.pooled_fit(G['rel']), ps.pooled_fit(G['signal'])
        fes = ps.pooled_fit(sub('err', 'floor')); frs = ps.pooled_fit({n: [e / c for e, c in zip(sub('err', 'floor')[n], G['signal'][n])] for n in G['err']})
        RS = sub('resid', 'floorD'); fd = ps.pooled_fit(RS); nz = sum(1 for n in RS for x in RS[n] if x == 0)
        w(f'- sig_xi = {sx:g}, g_eff = g beta_star = {ge:.3f}: error {f(fe)}; relative {f(fr)}; signal {f(fs)}; residual (beta_star, floor-subtracted) ' + (f(fd) if np.isfinite(fd['slope']) else 'n/a') + f' ({nz} realizations with floor >= value)')
        w(f'  - diagnostic, error floor-subtracted in quadrature: {f(fes)}; relative {f(frs)}')
        md = lambda k, n: np.median(G[k][n])
        w('  - floor/error: ' + ', '.join(f'{n}: {md("floor", n) / md("err", n):.2f}' for n in sorted(G['err'])) + '; residual floor/raw at N = 10000: ' + f'{md("floorD", 10000) / md("resid", 10000):.2f}')
        w('  - median Lyapunov exponent: ' + ', '.join(f'{n}: {md("lyap", n):.3f}' for n in sorted(G['err'])))
        w('  - relative error at N = 10000: ' + f'{md("rel", 10000):.3f}; local slopes of error: ' + ', '.join(f'{a}-{b}: {np.log(md("err", b) / md("err", a)) / np.log(b / a):.2f}' for a, b in zip(sorted(G['err'])[2:-1], sorted(G['err'])[3:])))
    w('- g_eff from the runs (g_eff_star, DMFT closure inside each run): ' + ', '.join(f'{sx:g}: {min(r["g_eff_star"] for r in R if r["case"] == c):.4f}-{max(r["g_eff_star"] for r in R if r["case"] == c):.4f}' for c, sx in CASES) + '.')
    print('\n'.join(L))
