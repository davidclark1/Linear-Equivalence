import sys, json, math, time, pickle, numpy as np, torch
from nzm import *

def run(model, N, g, seed, k, p, alpha, sI=0., sig_xi=0., dt=0.05, tau_max=20., T1=1500., n_ics1=8, T2=None, n_ics2=None,
        hgrid_n=33, M=512, P=None, device='cuda', lags=(0., 1., 2., 5., 10.), tap=False, th_Tpath=None):
    """tap=True: also the TAP-solved ansatz, with hbar from the solved TAP-like equation (tap.py; keys *_tap); saves J to $NZM_JDIR.
    th_Tpath: minimum length of the sampled single-site paths when P is not given (default T_PATH_MIN; smaller only for CPU tests)."""
    t0 = time.time(); P = P or 2 ** int(math.ceil(math.log2((th_Tpath or T_PATH_MIN) / dt)))     # sampled-theory paths of at least T_PATH_MIN time units
    J = make_J(N, g, seed, device); b = make_bias(N, sI, seed, device)
    p1 = pass1_ca(model, N, g, seed, k, p, n_ics=n_ics1, T=T1, dt=dt, sI=sI, sig_xi=sig_xi, tau_max=tau_max, J=J, b=b)
    print(f'[{model} N={N}] pass1 {p1["t_pass1"]:.0f}s lyap={p1["lyap"]:.3f} var_phi={p1["var_phi"]:.4f} q={p1["q"]:.3f} kurt={p1["eta_kurt"]:.3f}', flush=True)
    L = int(round(tau_max / dt)); C1 = p1['C_pop'].cpu().numpy(); hb1 = p1['hbar'].cpu().numpy()
    hg = np.linspace(hb1.min() - 0.3, hb1.max() + 0.3, hgrid_n)
    tt = time.time()
    th = theory_sampled(model, C1, g, hg, dt, L, k=k, p=p, q=p1['q'], sig_xi=sig_xi, M=M, P=P, device=device)
    print(f'[{model} N={N}] theory {time.time() - tt:.0f}s', flush=True)
    S_th = torch.as_tensor(th['S'], device=device, dtype=torch.float64)
    def interp_sites(A, hb):        # [n_h, ...] -> [N, ...] linear interpolation over hgrid
        hbt = torch.as_tensor(hb, device=device, dtype=torch.float64); hgt = torch.as_tensor(hg, device=device)
        idx = torch.clamp(torch.searchsorted(hgt, hbt) - 1, 0, len(hg) - 2); frac = ((hbt - hgt[idx]) / (hgt[idx + 1] - hgt[idx]))[:, None]
        return (1 - frac) * A[idx] + frac * A[idx + 1]
    S_sites = p1['S_site'].double()                                     # [N, L+1] measured per-site kernels (two-time tangent along the trajectory)
    S_sites_th = interp_sites(S_th, hb1)                                # theory kernels at the sites' hbar (diagnostic comparison)
    geff = g * (S_sites_th.sum(1) * dt).pow(2).mean().sqrt().item(); geff_meas = g * (S_sites.sum(1) * dt).pow(2).mean().sqrt().item()
    act = (p1['C_pop'][0] * 0 + 1) > 0
    print(f'[{model} N={N}] g_eff = g*rms S(w=0): theory {geff:.3f}, measured {geff_meas:.3f}; median ratio measured/theory of int S over sites {((S_sites.sum(1)) / (S_sites_th.sum(1))).median().item():.3f}', flush=True)
    if T2 is None: T2 = 2000. if N <= 2000 else (4000. if N <= 4000 else 8000.)   # longer, fewer trajectories at large N (ring memory ~ L n_ics N)
    n_ics2 = max(8, 2 * int(round(alpha * N / T2 / 2))) if n_ics2 is None else n_ics2
    p2 = pass2_ca(model, N, g, seed, k, p, S_sites, n_ics=n_ics2, T=T2, dt=dt, sI=sI, sig_xi=sig_xi, tau_max=tau_max, J=J, b=b, lags_block=lags[1:])
    print(f'[{model} N={N}] pass2 {p2["t_pass2"]:.0f}s alpha={p2["alpha"]:.0f}', flush=True)
    B = p2['B']
    omega_max = math.pi / dt; n_omega = int(round(omega_max / 0.05)) + 1
    omega = torch.linspace(0, omega_max, n_omega, device=device, dtype=torch.float64)
    hbar = p2['hbar']; hb = hbar.cpu().numpy(); C_pop = p2['C_pop'].cpu().numpy()
    # theory recomputed with the pass-2 population statistics (consistent Cc, S and C_pop in the D's)
    hg = np.linspace(hb.min() - 0.3, hb.max() + 0.3, hgrid_n)
    th = theory_sampled(model, C_pop, g, hg, dt, L, k=k, p=p, q=p2['q'], sig_xi=sig_xi, M=M, P=P, device=device, seed=1)
    S_th = torch.as_tensor(th['S'], device=device, dtype=torch.float64)
    Cc_w = spectrum_even(torch.as_tensor(th['Cc'], device=device), dt, omega)
    Cpop_w = spectrum_even(torch.as_tensor(C_pop, device=device), dt, omega)
    S_w = spectrum_causal(S_th, dt, omega)                                                    # [n_h, n_w]
    DDh = Cc_w - g ** 2 * S_w.abs() ** 2 * Cpop_w[None, :]
    DD_th = interp_sites(DDh, hb).clamp_min(0); DS_th = interp_sites(S_w, hb)
    Cbar_th = ansatz(J, DS_th, DD_th, omega, list(lags), B)
    if tap:   # TAP-solved: hbar from the solved TAP-like equation, same single-site functions (same C_pop, q, paths, grid) as the single-site ansatz
        import tap as tapm
        t3 = time.time()
        tfn = tapm.extend_sampled(lambda h_: theory_sampled(model, C_pop, g, h_, dt, L, k=k, p=p, q=p2['q'], sig_xi=sig_xi, M=M, P=P, device=device, seed=1))
        sol, hg3, th3, info = tapm.tap_on_grid(J, b, hg, th, tfn, p2['phibar'])
        hb3 = sol['hbar']
        S3_w = spectrum_causal(torch.as_tensor(th3['S'], device=device, dtype=torch.float64), dt, omega)
        DDh3 = spectrum_even(torch.as_tensor(th3['Cc'], device=device), dt, omega) - g ** 2 * S3_w.abs() ** 2 * Cpop_w[None, :]
        DD_tap = tapm.interp_sites(DDh3, hg3, hb3).clamp_min(0); DS_tap = tapm.interp_sites(S3_w, hg3, hb3)
        Cbar_tap = ansatz(J, DS_tap, DD_tap, omega, list(lags), B)
        floors = tapm.mean_floors(J, b, hg, th['m'], p2['phibar_ic_halves'])   # single-site grid and M_star, as for meaneq
        Jpath, Jstat = tapm.save_J(J, N, g, seed)
        tapres = dict(info, **floors, phibar_tap=sol['phibar'].float().cpu().numpy(), hbar_tap=hb3.float().cpu().numpy(), th_hgrid_tap=np.asarray(hg3),
                      th_m_tap=np.asarray(th3['m']), th_Cc_tap=np.asarray(th3['Cc'], dtype=np.float32), th_S_tap=np.asarray(th3['S'], dtype=np.float32),
                      tap_rms_dh=(hb3 - hbar.double()).pow(2).mean().sqrt().item(), J_path=Jpath, J_save=Jstat, **tapm.J_checksum(J), t_tap=time.time() - t3)
        print(f'[{model} N={N}] TAP: converged={info["tap_converged"]} ({info["tap_method"]}, {info["tap_iters"]} it, max|F|={info["tap_resid"]:.1e}), '
              f'grid +{info["tap_grid_ext"]} pts, rms dphibar={info["tap_rms_dphi"]:.4f}, rms dhbar={tapres["tap_rms_dh"]:.4f}, '
              f'start check {info["tap_rms_start_diff"]:.1e}; J {Jstat}; {tapres["t_tap"]:.0f}s', flush=True)
    DD_emp = spectrum_even(p2['ac_D'], dt, omega).clamp_min(0); DS_emp = spectrum_causal(S_sites, dt, omega)
    Cbar_emp = ansatz(J, DS_emp, DD_emp, omega, list(lags), B)
    Clag = torch.cat([p2['Cphi'][None], p2['Cphi_lag']], 0)
    mask = ~torch.eye(B, dtype=bool, device=device); rms = lambda A: math.sqrt((A[mask] ** 2).mean().item())
    noise = (p2['Cphi_halves'][0] - p2['Cphi_halves'][1]) / 2; noise_D = (p2['CD_halves'][0] - p2['CD_halves'][1]) / 2
    act = torch.diag(Clag[0]) > 0.1 * torch.diag(Clag[0]).mean()
    drel = lambda Cb: ((torch.diag(Cb[0]) - torch.diag(Clag[0])).abs() / torch.diag(Clag[0]))[act].median().item()
    m_pred = np.interp(hb, hg, th['m'])
    # naive: fixed point of the unit with the static self-coupling
    wsum = float((ca_kernel(k, p, dt, int(round(8 * p / dt))).numpy() * (p2['q'] + np.interp(np.arange(int(round(8 * p / dt)) + 1) * dt, np.arange(L + 1) * dt, C_pop))).sum())
    f = (lambda x: np.tanh(x)) if model == 'ca_tanh' else (lambda x: 0.5 * (1 + np.tanh(x)))
    naive = f(hb)
    for _ in range(200): naive = f(hb + wsum * naive)
    # empirical site response proxy at lag dt: <phi_i(t+dt) eta_i(t)> / (g^2 C_pop(dt))  (FN, ignoring O(N^-1/2) feedback)
    XC = p1['XC_site'].cpu().numpy(); Lp = XC.shape[1] // 2
    # empirical vs theory: <phi_i(t+dt) eta_i(t)> (Furutsu-Novikov prediction dt sum_{l>=1} S_l g^2 C_pop((l-1) dt))
    S_site_dt = XC[:, Lp + 1]
    S0 = dt * (S_sites[:, 1:].cpu().numpy() * (g ** 2 * C1[None, :L])).sum(1)
    nb = min(B, 200)
    res = dict(model=model, N=N, g=g, seed=seed, k=k, p=p, sI=sI, lam=0., mu=0., b0=0., sig_xi=sig_xi, dt=dt, dt_r=dt, omega_max=omega_max, n_omega=n_omega,
               alpha=p2['alpha'], B=B, lyap=p1['lyap'], eta_kurt=p1['eta_kurt'], var_phi=p1['var_phi'], smin=float('inf'), lags=list(lags), q=p2['q'], g_eff=geff,
               phibar=p2['phibar'].cpu().numpy(), hbar=hb, b=b.cpu().numpy(), phibar_halves=p1['phibar_halves'].cpu().numpy(), C_pop=C_pop,
               C_eta_pop=p1['C_eta_pop'].cpu().numpy(), CD_pop=p2['CD_pop'].cpu().numpy(), C_pop1=C1, m_pred=m_pred, naive=naive, th_hgrid=hg, th_m=th['m'],
               th_S0=th['S'][:, 1], S0_pred=S0, S_site0=S_site_dt, Sint_pred=np.interp(hb, hg, th['S'].sum(1) * dt), S_site_int=(S_sites.sum(1) * dt).cpu().numpy(), S_site_int_th=(S_sites_th.sum(1) * dt).cpu().numpy(), g_eff_meas=geff_meas,
               rms_C=rms(Clag[0]), rms_CD=rms(p2['CD']), rms_noise=rms(noise), rms_noise_D=rms(noise_D),
               rms_err_emp=rms(Cbar_emp[0] - Clag[0]), rms_err_th=rms(Cbar_th[0] - Clag[0]), diag_relerr_emp=drel(Cbar_emp), diag_relerr_th=drel(Cbar_th),
               rms_err_emp_lags=[rms(Cbar_emp[i] - Clag[i]) for i in range(len(lags))], rms_err_th_lags=[rms(Cbar_th[i] - Clag[i]) for i in range(len(lags))],
               rms_C_lags=[rms(Clag[i]) for i in range(len(lags))], C_blk=Clag[:, :nb, :nb].cpu().numpy(), Cbar_emp_blk=Cbar_emp[:, :nb, :nb].cpu().numpy(),
               Cbar_th_blk=Cbar_th[:, :nb, :nb].cpu().numpy(), CD_blk=p2['CD'][:nb, :nb].cpu().numpy(), noise_blk=noise[:nb, :nb].cpu().numpy(),
               ac_D_mean=p2['ac_D'].mean(0).cpu().numpy(), t_total=time.time() - t0, resid_kernel='theory')
    if tap:
        from run_model import _tap_metrics
        res.update(tapres, **_tap_metrics(Cbar_tap, Clag, mask, rms, p2['phibar'], tapres['phibar_tap'], B, diag='median_active', act=act))
        res['Cbar_tap_blk'] = Cbar_tap[:, :nb, :nb].cpu().numpy(); res['t_total'] = time.time() - t0
    return res

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); out = sys.argv[2]; Ns = cfg.pop('Ns'); allres = []
    for N in Ns:
        r = run(N=N, **cfg); allres.append(r)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not hasattr(v, 'shape') and not isinstance(v, list)}), flush=True)
        pickle.dump(allres, open(out, 'wb')); torch.cuda.empty_cache()
