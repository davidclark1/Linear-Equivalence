"""Reduce the result pickles (several GB) to the small files under data/ from which every figure, fit and table is made.
Run it where the results are (it needs a few tens of GB of memory); everything downstream runs on a laptop.

  data/scalars_moments.json  SCS model, per run (cases pm_a200, pm2_a800, pm_a3200; the alpha = 800 rows are labeled pm_a800, with
                             run_case = pm2_a800): all scalar entries, plus the per-lag sampling floors at alpha = 800
  data/lags_moments.json     SCS model at alpha = 800, per run: off-diagonal RMS error and signal at every stored lag, RMS error of
                             the diagonal elements, sampling floors, site spread of C_ii(0)
  data/moments_curves.pkl    SCS model at alpha = 800, seed 0, every N: site autocovariances (mean and s.d. over sites) and C_star
  data/scalars_noise.json    noise-driven SCS model (cases scsn{15,2,25}a3200), per run: all scalar entries
  data/scalars_prod_tap.json models with nonzero mean activities (cases *_tap), per run: all scalar entries + meaneq = RMS over sites
                             of the error of the TAP-like equation; case = model name without '_tap'
  data/cavity.pkl            cavity tests (cavity_stats.py), per run: the statistics and sampling floors of Fig. 6; the Price scatter
                             arrays are kept only for seed 0 at the largest alpha

Input directories: results/moments (pm_a200, pm_a3200), results/moments2 (pm2_a800), results/noise, results/prod_tap,
results/cavity_halves.
Usage: python extract_cache.py [moments] [pmlags] [noise] [prodtap] [cavity]     (default: all)"""
import glob, json, os, pickle, re, sys
import numpy as np

os.makedirs('data', exist_ok=True)
TARGETS = ('moments', 'pmlags', 'noise', 'prodtap', 'cavity')
WHAT = [a for a in sys.argv[1:] if a in TARGETS] or list(TARGETS)
CAV_DIR = 'results/cavity_halves'


def pm_files(alpha=None):
    """[(file, alpha, seed, N, run_case)] of the SCS run set: alpha = 200 and 3200 from results/moments, alpha = 800 from results/moments2."""
    out = []
    for f in sorted(glob.glob('results/moments/pm_a*_s*_N*.pkl')):
        m = re.match(r'results/moments/pm_a(\d+)_s(\d+)_N(\d+)\.pkl', f); a = int(m.group(1))
        if a != 800: out.append((f, a, int(m.group(2)), int(m.group(3)), f'pm_a{a}'))
    for f in sorted(glob.glob('results/moments2/pm2_a800_s*_N*.pkl')):
        m = re.match(r'results/moments2/pm2_a800_s(\d+)_N(\d+)\.pkl', f); out.append((f, 800, int(m.group(1)), int(m.group(2)), 'pm2_a800'))
    assert any(o[4] == 'pm2_a800' for o in out), 'no results/moments2/pm2_a800_* files'
    return [o for o in out if alpha is None or o[1] == alpha]


def lag_floors(r, d):
    """Per-lag sampling floors of a pm2_a800 run (off-diagonal RMS over B x B; diagonal RMS over the stored block's sites d); {} otherwise."""
    if 'rms_noise_lags' not in r: return {}
    nb = r['noise_blk_lags']
    return dict(floor_lags=[float(x) for x in r['rms_noise_lags']], diag_floor_lags=[float(np.sqrt(np.mean(nb[l][d, d].astype(np.float64) ** 2))) for l in range(nb.shape[0])])


def scalars(r):
    out = {}
    for k, v in r.items():
        if isinstance(v, (bool, int, float, str, np.integer, np.floating)):
            out[k] = v.item() if isinstance(v, np.generic) else v
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            out[k] = v.item()
    return out


if 'cavity' in WHAT:
    D = {}
    for f in sorted(glob.glob(f'{CAV_DIR}/cav_N*_a*_s*.pkl')):
        d = pickle.load(open(f, 'rb')); D[(d['N'], d['alpha'], d['seed'])] = d
    amax = max(k[1] for k in D)
    for (N, a, sd), d in D.items():
        if not (a == amax and sd == 0):
            for side in ('cav', 'res'):
                d[side]['price'] = dict(slope=d[side]['price']['slope'])
        for side in ('cav', 'res'):
            for k in ('raw_halves', 'raw', 'halves', 'kappa_std'): d[side].pop(k, None)     # raw accumulators and per-half values: not used by the figure
    pickle.dump(D, open('data/cavity.pkl', 'wb'))
if 'moments' in WHAT:
    rows = []
    for f, a, sd, N, rc in pm_files():
        r = pickle.load(open(f, 'rb')); s = scalars(r); s.update(case=f'pm_a{a}', alpha_nominal=a, seed=sd, N=N); rows.append(s)
        if rc != s['case']: s['run_case'] = rc
        if 'rms_noise_lags' in r: s['rms_noise_lags'] = [float(x) for x in r['rms_noise_lags']]
        print(f, flush=True)
    json.dump(rows, open('data/scalars_moments.json', 'w'), indent=0)
if 'pmlags' in WHAT:
    rows, curves = [], {}
    rng = np.random.default_rng(0)
    for f, _, sd, N, rc in pm_files(800):
        r = pickle.load(open(f, 'rb')); C = r['C_blk']; Cb = r['Cbar_dmft_blk']; d = np.arange(C.shape[1])
        ac = r['ac_phi_site']; nbs = min(ac.shape[0], 1000)
        rows.append(dict(case='pm_a800', seed=sd, N=N, lags=[float(x) for x in r['lags']],
                         err_lags=[float(x) for x in r['rms_err_dmft_lags']], C_lags=[float(x) for x in r['rms_C_lags']],
                         diag_err_lags=[float(np.sqrt(np.mean((Cb[l, d, d] - C[l, d, d]) ** 2))) for l in range(C.shape[0])],
                         diag_floor=float(np.sqrt(np.mean(r['noise_blk'][d, d] ** 2))), diag_mean=float(C[0, d, d].mean()), n_diag=int(len(d)),
                         diag_spread=float(ac[:nbs, 0].std()), diag_spread_blk=float(C[0, d, d].std()), diag_ansatz_spread_blk=float(Cb[0, d, d].std()),
                         diag_err_star=float(np.sqrt(np.mean((np.asarray(r['C_star'])[0] - C[0, d, d]) ** 2))), run_case=rc, **lag_floors(r, d)))
        if sd == 0:
            nb = C.shape[1]; pairs = [(int(i), int(j)) for i, j in zip(rng.integers(0, nb, 40), rng.integers(0, nb, 40)) if i != j][:8]
            dt_r = r['dt_r']; L = int(round(10 / dt_r)) + 1
            curves[N] = dict(lags=np.asarray(r['lags']), pairs=pairs,
                             sim={p: (C[:, p[0], p[1]], C[:, p[1], p[0]]) for p in pairs}, ansatz={p: (Cb[:, p[0], p[1]], Cb[:, p[1], p[0]]) for p in pairs},
                             tau=np.arange(L) * dt_r, ac_mean=ac[:nbs, :L].mean(0), ac_std=ac[:nbs, :L].std(0), C_star=np.asarray(r['C_star'])[:L])
        print(f, flush=True)
    json.dump(rows, open('data/lags_moments.json', 'w'), indent=0); pickle.dump(curves, open('data/moments_curves.pkl', 'wb'))
if 'noise' in WHAT:
    rows = []
    for f in sorted(glob.glob('results/noise/scsn*a3200_s*_N*.pkl')):
        m = re.match(r'results/noise/(scsn(\d+)a3200)_s(\d+)_N(\d+)\.pkl', f)
        r = pickle.load(open(f, 'rb')); s = scalars(r); s.update(case=m.group(1), seed=int(m.group(3)), N=int(m.group(4))); rows.append(s)
        print(f, flush=True)
    json.dump(rows, open('data/scalars_noise.json', 'w'), indent=0)
if 'prodtap' in WHAT:
    rows = []
    for f in sorted(glob.glob('results/prod_tap/*_tap_s*_N*.pkl')):
        m = re.match(r'results/prod_tap/(\w+)_tap_s(\d+)_N(\d+)\.pkl', f)
        r = pickle.load(open(f, 'rb')); s = scalars(r)
        s.update(case=m.group(1), run_set='prod_tap', seed=int(m.group(2)), N=int(m.group(3)))
        s['meaneq'] = float(np.sqrt(np.mean((np.asarray(r['phibar']) - np.asarray(r['m_pred'])) ** 2)))
        if s.get('J_path'): s['J_path'] = os.path.basename(s['J_path'])     # file name only
        rows.append(s); print(f, flush=True)
    json.dump(rows, open('data/scalars_prod_tap.json', 'w'), indent=0)
print('done')
