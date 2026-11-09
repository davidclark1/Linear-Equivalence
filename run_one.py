"""One (case, seed, N) run -> $NZM_RUNS/<case>_s<seed>_N<N>.pkl (default results/runs).  Skips if the file exists, so an
interrupted or requeued job only redoes what is missing.
Usage: python run_one.py <case> <seed> <N> [json overrides]   (overrides write to results/runs_test/ instead)"""
import sys, os, json, pickle, time, torch
CASES = {
    # models with nonzero mean activities (Sec. VII of the paper): g = 20, alpha = 200, no external drive
    'sig':          dict(driver='model', model='sig', g=20., alpha=200., sI=0., dt=0.05, dt_r=0.05, tau_max=20.),
    'sigbias':      dict(driver='model', model='sig', g=20., alpha=200., sI=5., dt=0.05, dt_r=0.05, tau_max=20.),
    'glvsat':       dict(driver='model', model='glvsat', g=20., alpha=200., lam=0.01, dt=0.01, dt_r=0.01, tau_max=20., resid_kernel='empirical', hgrid_n=129),
    'ca_km2':       dict(driver='ca', model='ca', g=20., k=-2., p=2., alpha=200., dt=0.05, tau_max=20., hgrid_n=129),
    'ca_kp2':       dict(driver='ca', model='ca', g=20., k=2., p=2., alpha=200., dt=0.05, tau_max=20., hgrid_n=129),
    # SCS model (Secs. V-VI): erf units, g = 2.5, no drive, dt = 0.025, recorded every 0.1, lags 0..10 by 0.5; also the residual
    # with beta_star and the two-site moments <phi_i^3 phi_j>, <phi_i^3 phi_j^3>
    **{f'pm_a{a}': dict(driver='model', model='erf', g=2.5, alpha=float(a), sI=0., dt=0.025, dt_r=0.1, tau_max=40., lags=tuple(0.5 * i for i in range(21)), powers=True)
       for a in (200, 800, 3200)},
    # alpha = 800 with the lagged covariances also accumulated over the two halves of the initial conditions (sampling floors at
    # every lag, Fig. 8); otherwise identical to pm_a800.  The paper's alpha = 800 results use this case.
    'pm2_a800': dict(driver='model', model='erf', g=2.5, alpha=800., sI=0., dt=0.025, dt_r=0.1, tau_max=40., lags=tuple(0.5 * i for i in range(21)), powers=True, lag_halves=True),
    # SCS model driven by i.i.d. white noise (sig_xi sqrt(dt) per Euler step on x), alpha = 3200 (Appendix I, Fig. 10):
    # DMFT g_eff = g beta_star = 0.92 (sig_xi = 1.5), 0.86 (2), 0.79 (2.5), vs 0.96 undriven
    **{f'scsn{tag}a3200': dict(driver='model', model='erf', g=2.5, alpha=3200., sI=0., sig_xi=s, dt=0.025, dt_r=0.1, tau_max=40., lags=tuple(0.5 * i for i in range(21)), star=True)
       for tag, s in (('15', 1.5), ('2', 2.), ('25', 2.5))},
}
# the nonzero-mean cases with all three ways of evaluating the ansatz (Fig. 4): tap=True adds the ansatz with the temporal mean
# inputs from the solved TAP-like equation (tap.py); J is saved to $NZM_JDIR if that directory exists.  These are the runs used in the paper.
CASES.update({f'{c}_tap': dict(CASES[c], tap=True) for c in ('sig', 'sigbias', 'glvsat', 'ca_km2', 'ca_kp2')})
if __name__ == '__main__':
    case, seed, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}          # optional parameter overrides -> results/runs_test/
    RUNS = os.environ.get('NZM_RUNS', 'results/runs')                    # output directory (env NZM_RUNS)
    os.makedirs(RUNS, exist_ok=True); os.makedirs('results/runs_test', exist_ok=True)
    out = f'{RUNS}/{case}_s{seed}_N{N}.pkl' if not over else f'results/runs_test/{case}_s{seed}_N{N}_' + '_'.join(f'{k}{v}' for k, v in over.items()) + '.pkl'
    if os.path.exists(out):
        print('exists:', out); sys.exit(0)
    cfg = dict(CASES[case]); cfg.update(over); driver = cfg.pop('driver')
    t0 = time.time()
    if driver == 'model':
        import run_model; r = run_model.run(N=N, seed=seed, **cfg)
    else:
        import run_ca; r = run_ca.run(N=N, seed=seed, **cfg)
    r['case'] = case
    pickle.dump(r, open(out + '.tmp', 'wb')); os.replace(out + '.tmp', out)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if isinstance(v, (int, float, str, bool))}), flush=True)
    print(f'done {out} in {time.time() - t0:.0f}s', flush=True)
