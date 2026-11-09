import sys, json, math, time, pickle, numpy as np, torch
from nzm import *

def _tap_metrics(Cbar_tap, Clag, mask, rms, phibar_meas, phibar_tap, B, diag='mean', act=None):
    """Metrics of the TAP-solved ansatz with the estimator, block and lags of the single-site ansatz; plus the disconnected part and the full covariance at lag 0."""
    out = dict(rms_err_tap=rms(Cbar_tap[0] - Clag[0]), rms_err_tap_lags=[rms(Cbar_tap[q] - Clag[q]) for q in range(Clag.shape[0])])
    dr = (Cbar_tap[0].diag() - Clag[0].diag()).abs() / Clag[0].diag()
    out['diag_relerr_tap'] = dr.mean().item() if diag == 'mean' else dr[act].median().item()
    pm = phibar_meas[:B].double(); pt = torch.as_tensor(phibar_tap[:B], device=pm.device).double()
    disc_m = pm[:, None] * pm[None, :]; disc_t = pt[:, None] * pt[None, :]
    out.update(rms_disc=rms(disc_m), rms_disc_err_tap=rms(disc_t - disc_m), rms_full_err_tap=rms(disc_t + Cbar_tap[0] - disc_m - Clag[0]))
    return out

def run(model, N, g, seed, alpha, sI=0., lam=0., mu=0., b0=0., resid_kernel='empirical', sig_xi=0., dt=0.05, dt_r=0.1, tau_max=20., T1=1500., n_ics1=8,
        T2=None, n_ics2=None, omega_max=12., n_omega=241, hgrid_n=33, device='cuda', theory=True, lags=(0., 1., 2., 5., 10.), powers=False, star=False,
        tap=False, th_M=1024, th_Tpath=None, lag_halves=False):
    """star=True (erf only): also the residual with the DMFT population gain, Delta_i = phi_i - beta_star y_i (pass2 CD_star), without
    the odd-power moments (powers=True does both; it assumes no drive).  With sig_xi > 0 the closure includes the white-noise drive.
    tap=True: also the TAP-solved ansatz, with hbar from the solved TAP-like equation (tap.py; keys *_tap); saves J to $NZM_JDIR.
    th_M, th_Tpath: number and minimum length of the sampled single-site paths (defaults = production values; smaller only for CPU tests)."""
    t0 = time.time(); th_Tpath = th_Tpath or T_PATH_MIN
    if model in LINEAR:         # linear-filter units: spectra on the record grid up to its Nyquist frequency (dw = 0.05); the Euler
        omega_max = math.pi / dt_r; n_omega = int(round(omega_max / 0.05)) + 1   # filter Kd is always evaluated with the true step dt
    else:                       # gLV-type: band up to min(Nyquist, 60) with dw = 0.1 (fast saturated sites carry negligible residual power)
        omega_max = min(math.pi / dt_r, 60.); n_omega = int(round(omega_max / 0.1)) + 1
    J = make_J(N, g, seed, device, mu=mu); b = make_bias(N, sI, seed, device) + b0
    p1 = pass1(model, N, g, seed, n_ics=n_ics1, T=T1, dt=dt, sI=sI, lam=lam, dt_r=dt_r, tau_max=tau_max, J=J, b=b, sig_xi=sig_xi)
    print(f'[{model} N={N}] pass1 {p1["t_pass1"]:.0f}s lyap={p1["lyap"]:.3f} var_phi={p1["var_phi"]:.4f} kurt={p1["eta_kurt"]:.3f} smin={p1["smin"]:.2e}', flush=True)
    kern = dict(beta=p1['beta']) if model in LINEAR else dict(S=p1['S_site'])
    th1 = None
    if model not in LINEAR and resid_kernel == 'theory':
        # residual kernels from the single-site theory evaluated at each site's hbar (pass-1 C_pop), robust to the heavy-tailed empirical estimator
        hb1 = p1['hbar'].cpu().numpy(); hg1 = np.linspace(hb1.min() - 0.3, hb1.max() + 0.3, hgrid_n)
        L1 = int(round(tau_max / dt_r))
        th1 = (theory_sampled(model, p1['C_pop'].cpu().numpy(), g, hg1, dt, L1, lam=lam, sig_xi=sig_xi, M=th_M, P=2 ** int(math.ceil(math.log2(th_Tpath / dt))), device=device) if dt_r == dt
               else theory_sampled_coarse(model, p1['C_pop'].cpu().numpy(), g, hg1, dt, dt_r, L1, lam=lam, sig_xi=sig_xi, M=th_M, P=2 ** int(math.ceil(math.log2(th_Tpath / dt))), device=device))
        S_th = torch.as_tensor(th1['S'], device=device, dtype=torch.float64)      # [n_h, L+1]
        hbt = torch.as_tensor(hb1, device=device, dtype=torch.float64); hgt = torch.as_tensor(hg1, device=device)
        idx = torch.clamp(torch.searchsorted(hgt, hbt) - 1, 0, len(hg1) - 2); frac = ((hbt - hgt[idx]) / (hgt[idx + 1] - hgt[idx]))[:, None]
        kern = dict(S=(1 - frac) * S_th[idx] + frac * S_th[idx + 1])
        print(f'[{model} N={N}] residual kernels from theory: sigma*rms S(w=0) = {g * (kern["S"].sum(1) * dt_r).pow(2).mean().sqrt().item():.3f} (empirical estimator: {g * (p1["S_site"].sum(1) * dt_r).pow(2).mean().sqrt().item():.3f})', flush=True)
    if T2 is None:
        T2 = 2000. if N <= 2000 else (4000. if N <= 4000 else 8000.)   # longer, fewer trajectories at large N (ring memory ~ L n_ics N)
    if n_ics2 is None:
        n_ics2 = min(400, max(8, 2 * int(round(alpha * N / T2 / 2))))            # at most 400 parallel trajectories (ring memory)
        T2 = max(T2, alpha * N / n_ics2)
    extra = {}
    if powers or star:   # erf only: residual with the population gain beta_star next to the site gains (+ two-site moments of odd powers if powers)
        assert model == 'erf' and not (powers and sig_xi > 0), 'moments.power_theory has no drive term'
        _, beta_star0, _ = closure_odd(p1['C_pop'].cpu().numpy(), g, dt_r, dt_filter=dt, sig_xi=sig_xi)
        extra = dict(beta_star=beta_star0, powers=powers)
    p2 = pass2(model, N, g, seed, kern, n_ics=n_ics2, T=T2, dt=dt, sI=sI, lam=lam, dt_r=dt_r, tau_max=tau_max, J=J, b=b, lags_block=lags[1:], sig_xi=sig_xi, lag_halves=lag_halves, **extra)
    print(f'[{model} N={N}] pass2 {p2["t_pass2"]:.0f}s alpha={p2["alpha"]:.0f} n_ics={n_ics2}', flush=True)
    B = p2['B']; L = int(round(tau_max / dt_r))
    omega = torch.linspace(0, omega_max, n_omega, device=device, dtype=torch.float64)
    # ---- semi-empirical D's (per-site empirical kernels and residual spectra)
    DD_emp = spectrum_even(p2['ac_D'], dt_r, omega).clamp_min(0)                     # [N, n_w]
    if model in LINEAR:
        DS_emp = p1['beta'].double()[:, None] * Kd(omega, dt).to(device)[None, :]
    else:
        DS_emp = spectrum_causal(kern['S'] if resid_kernel == 'theory' else p1['S_site'], dt_r, omega)
    Cbar_emp = ansatz(J, DS_emp, DD_emp, omega, list(lags), B)
    # ---- theory (single-site conditional quantities from the population C_pop of pass2)
    hbar = p2['hbar']; C_pop = p2['C_pop'].cpu().numpy()
    hg = (np.arange(hbar.min().item() - 0.3, hbar.max().item() + 0.3, 0.25) if model in LINEAR else np.linspace(hbar.min().item() - 0.3, hbar.max().item() + 0.3, hgrid_n))   # sig: spacing 0.25 << input std (interpolation of m, beta, D over hbar must be exact)
    res = dict(model=model, N=N, g=g, seed=seed, sI=sI, lam=lam, mu=mu, b0=b0, resid_kernel=resid_kernel, sig_xi=sig_xi, dt=dt, dt_r=dt_r, omega_max=omega_max, n_omega=n_omega, alpha=p2['alpha'], B=B,
               lyap=p1['lyap'], eta_kurt=p1['eta_kurt'], var_phi=p1['var_phi'], smin=p1['smin'], lags=list(lags),
               phibar=p2['phibar'].cpu().numpy(), hbar=hbar.cpu().numpy(), b=b.cpu().numpy(),
               phibar_halves=p1['phibar_halves'].cpu().numpy(), C_pop=C_pop, C_eta_pop=p1['C_eta_pop'].cpu().numpy(),
               CD_pop=p2['CD_pop'].cpu().numpy(), C_pop1=p1['C_pop'].cpu().numpy())
    if model in LINEAR:
        res['beta'] = p1['beta'].cpu().numpy()
    else:
        res['S_site0'] = p1['S_site'][:, 0].cpu().numpy(); res['S_site_int'] = (p1['S_site'].sum(1) * dt_r).cpu().numpy()
    if theory:
        tt = time.time()
        if model in LINEAR:
            th = theory_sig(C_pop, g, hg, dt_r, sig_xi=sig_xi, model=model, dt_filter=dt)
            m_pred = np.interp(res['hbar'], hg, th['m']); beta_pred = np.interp(res['hbar'], hg, th['beta'])
            res.update(m_pred=m_pred, beta_pred=beta_pred, naive=f_np(model)[0](res['hbar']), th_hgrid=hg, th_m=th['m'], th_beta=th['beta'])
            # theory D's
            Cc = torch.as_tensor(th['Cc'], device=device)                      # [n_h, L+1]
            Cc_w = spectrum_even(Cc, dt_r, omega)                                # [n_h, n_w]
            Cpop_w = spectrum_even(torch.as_tensor(C_pop, device=device), dt_r, omega)  # [n_w]
            beta_h = torch.as_tensor(th['beta'], device=device)
            K2 = (Kd(omega, dt).abs() ** 2).to(device)
            DDh = Cc_w - g ** 2 * beta_h[:, None] ** 2 * K2[None, :] * Cpop_w[None, :]
            # interpolate over hbar for each site
            hb = hbar.double(); hgt = torch.as_tensor(hg, device=device)
            idx = torch.clamp(torch.searchsorted(hgt, hb) - 1, 0, len(hg) - 2); frac = ((hb - hgt[idx]) / (hgt[idx + 1] - hgt[idx]))[:, None]
            DD_th = ((1 - frac) * DDh[idx] + frac * DDh[idx + 1]).clamp_min(0)
            DS_th = torch.as_tensor(beta_pred, device=device, dtype=torch.float64)[:, None] * Kd(omega, dt).to(device)[None, :]
        else:
            L_ = int(round(tau_max / dt_r))
            if dt_r == dt:   # sampled theory on the integration grid 
                th = theory_sampled(model, C_pop, g, hg, dt, L_, lam=lam, sig_xi=sig_xi, M=th_M, P=2 ** int(math.ceil(math.log2(th_Tpath / dt))), device=device)
            else:            # integration step dt, recording step dt_r: theory on the dt grid, subsampled
                th = theory_sampled_coarse(model, C_pop, g, hg, dt, dt_r, L_, lam=lam, sig_xi=sig_xi, M=th_M, P=2 ** int(math.ceil(math.log2(th_Tpath / dt))), device=device)
            m_pred = np.interp(res['hbar'], hg, th['m'])
            res.update(m_pred=m_pred, th_hgrid=hg, th_m=th['m'], th_S0=th['S'][:, 0], S0_pred=np.interp(res['hbar'], hg, th['S'][:, 0]),
                       Sint_pred=np.interp(res['hbar'], hg, th['S'].sum(1) * dt_r))
            # naive: fixed point N = 1 - h (+lam correction) clipped at 0
            hb_np = res['hbar']; Nfp = 0.5 * ((1 - hb_np) + np.sqrt((1 - hb_np) ** 2 + 4 * lam)); res['naive'] = Nfp / (1 + Nfp) if model == 'glvsat' else Nfp
            Cc = torch.as_tensor(th['Cc'], device=device); Cc_w = spectrum_even(Cc, dt_r, omega)
            Cpop_w = spectrum_even(torch.as_tensor(C_pop, device=device), dt_r, omega)
            S_w = spectrum_causal(torch.as_tensor(th['S'], device=device), dt_r, omega)   # [n_h, n_w]
            DDh = Cc_w - g ** 2 * S_w.abs() ** 2 * Cpop_w[None, :]
            hb = hbar.double(); hgt = torch.as_tensor(hg, device=device)
            idx = torch.clamp(torch.searchsorted(hgt, hb) - 1, 0, len(hg) - 2); frac = ((hb - hgt[idx]) / (hgt[idx + 1] - hgt[idx]))[:, None]
            DD_th = ((1 - frac) * DDh[idx] + frac * DDh[idx + 1]).clamp_min(0)
            DS_th = (1 - frac) * S_w[idx] + frac * S_w[idx + 1]
        Cbar_th = ansatz(J, DS_th, DD_th, omega, list(lags), B)
        if tap:   # TAP-solved: hbar from the solved TAP-like equation, same single-site functions (same C_pop, grid, interpolation) as the single-site ansatz
            import tap as tapm
            t3 = time.time()
            if model in LINEAR:   # theory_sig is recomputed on the whole extended grid (its quadrature grid depends on the hbar range)
                tfn = lambda hg_new, n_lo, n_hi, th_old: theory_sig(C_pop, g, hg_new, dt_r, sig_xi=sig_xi, model=model, dt_filter=dt)
            else:                 # sampled theory: only the new points (same seed -> same paths), concatenated
                Pth = 2 ** int(math.ceil(math.log2(th_Tpath / dt)))
                tfn = tapm.extend_sampled((lambda h_: theory_sampled(model, C_pop, g, h_, dt, L_, lam=lam, sig_xi=sig_xi, M=th_M, P=Pth, device=device)) if dt_r == dt else
                                          (lambda h_: theory_sampled_coarse(model, C_pop, g, h_, dt, dt_r, L_, lam=lam, sig_xi=sig_xi, M=th_M, P=Pth, device=device)))
            sol, hg3, th3, info = tapm.tap_on_grid(J, b, hg, th, tfn, p2['phibar'])
            hb3 = sol['hbar']
            Cc3_w = spectrum_even(torch.as_tensor(th3['Cc'], device=device), dt_r, omega)
            if model in LINEAR:
                DDh3 = Cc3_w - g ** 2 * torch.as_tensor(th3['beta'], device=device)[:, None] ** 2 * K2[None, :] * Cpop_w[None, :]
                DS_tap = torch.as_tensor(np.interp(hb3.cpu().numpy(), hg3, th3['beta']), device=device, dtype=torch.float64)[:, None] * Kd(omega, dt).to(device)[None, :]
                res.update(th_beta_tap=np.asarray(th3['beta']))
            else:
                S3_w = spectrum_causal(torch.as_tensor(th3['S'], device=device), dt_r, omega)
                DDh3 = Cc3_w - g ** 2 * S3_w.abs() ** 2 * Cpop_w[None, :]
                DS_tap = tapm.interp_sites(S3_w, hg3, hb3)
                res.update(th_S_tap=np.asarray(th3['S'], dtype=np.float32))
            DD_tap = tapm.interp_sites(DDh3, hg3, hb3).clamp_min(0)
            Cbar_tap = ansatz(J, DS_tap, DD_tap, omega, list(lags), B)
            res.update(tapm.mean_floors(J, b, hg, th['m'], p2['phibar_ic_halves']))   # single-site grid and M_star, as for meaneq
            Jpath, Jstat = tapm.save_J(J, N, g, seed)
            res.update(info, phibar_tap=sol['phibar'].float().cpu().numpy(), hbar_tap=hb3.float().cpu().numpy(), th_hgrid_tap=np.asarray(hg3),
                       th_m_tap=np.asarray(th3['m']), th_Cc_tap=np.asarray(th3['Cc'], dtype=np.float32),
                       tap_rms_dh=(hb3 - hbar.double()).pow(2).mean().sqrt().item(), J_path=Jpath, J_save=Jstat, **tapm.J_checksum(J), t_tap=time.time() - t3)
            print(f'[{model} N={N}] TAP: converged={info["tap_converged"]} ({info["tap_method"]}, {info["tap_iters"]} it, max|F|={info["tap_resid"]:.1e}), '
                  f'grid +{info["tap_grid_ext"]} pts, rms dphibar={info["tap_rms_dphi"]:.4f}, rms dhbar={res["tap_rms_dh"]:.4f}, '
                  f'start check {info["tap_rms_start_diff"]:.1e}; J {Jstat}; {res["t_tap"]:.0f}s', flush=True)
        if model == 'erf':   # the paper's fully theoretical version: DMFT closure C_star, beta_star (same for all sites, hbar = 0)
            C_star, beta_star, hist = closure_odd(C_pop, g, dt_r, dt_filter=dt, sig_xi=sig_xi)
            Cstar_w = spectrum_even(torch.as_tensor(C_star, device=device), dt_r, omega)
            DD_dmft = ((1 - g ** 2 * beta_star ** 2 * K2) * Cstar_w).clamp_min(0)[None, :].expand(N, -1).contiguous()
            DS_dmft = beta_star * Kd(omega, dt).to(device)[None, :].expand(N, -1).contiguous()
            Cbar_dmft = ansatz(J, DS_dmft, DD_dmft, omega, list(lags), B)
            res.update(C_star=C_star, beta_star=beta_star, closure_hist=hist)
            if star: res.update(beta_star_pass2=beta_star0, g_eff_star=g * beta_star)
            print(f'[{model} N={N}] DMFT closure: C_star(0)={C_star[0]:.4f} (measured C_pop(0)={C_pop[0]:.4f}), beta_star={beta_star:.4f} (measured rms beta={float(p1["beta"].pow(2).mean().sqrt()):.4f}), converged to {hist[-1]:.1e} in {len(hist)} it', flush=True)
            if powers:
                import moments
                pred_pq, th_pq = moments.predict(J, C_star, beta_star, g, dt, dt_r, omega, B)
                res.update(beta_pow={int(a): v for a, v in th_pq['beta'].items()}, C_pow_star={f'{a}{b_}': v for (a, b_), v in th_pq['C'].items()}, beta_star_pass2=beta_star0)
        res['t_theory'] = time.time() - tt
    # ---- metrics at lag 0
    C = p2['Cphi'][None]  # [1,B,B] at lag 0
    Clag = torch.cat([p2['Cphi'][None], p2['Cphi_lag']], 0)        # [n_lags, B, B] (lags 0, then lags_block)
    mask = ~torch.eye(B, dtype=bool, device=device)
    rms = lambda A: math.sqrt((A[mask] ** 2).mean().item())
    noise = (p2['Cphi_halves'][0] - p2['Cphi_halves'][1]) / 2
    noise_D = (p2['CD_halves'][0] - p2['CD_halves'][1]) / 2
    res.update(rms_C=rms(Clag[0]), rms_CD=rms(p2['CD']), rms_noise=rms(noise), rms_noise_D=rms(noise_D),
               rms_err_emp=rms(Cbar_emp[0] - Clag[0]), diag_relerr_emp=((Cbar_emp[0].diag() - Clag[0].diag()).abs() / Clag[0].diag()).mean().item(),
               rms_err_emp_lags=[rms(Cbar_emp[q] - Clag[q]) for q in range(len(lags))], rms_C_lags=[rms(Clag[q]) for q in range(len(lags))])
    if theory:
        res.update(rms_err_th=rms(Cbar_th[0] - Clag[0]), diag_relerr_th=((Cbar_th[0].diag() - Clag[0].diag()).abs() / Clag[0].diag()).mean().item(),
                   rms_err_th_lags=[rms(Cbar_th[q] - Clag[q]) for q in range(len(lags))])
        if tap: res.update(_tap_metrics(Cbar_tap, Clag, mask, rms, p2['phibar'], res['phibar_tap'], B, diag='mean'))
        if model == 'erf':
            res.update(rms_err_dmft=rms(Cbar_dmft[0] - Clag[0]), rms_err_dmft_lags=[rms(Cbar_dmft[q] - Clag[q]) for q in range(len(lags))],
                       diag_relerr_dmft=((Cbar_dmft[0].diag() - Clag[0].diag()).abs() / Clag[0].diag()).mean().item())
    nb = min(B, 200)
    if 'CD_star' in p2:
        res.update(rms_CD_star=rms(p2['CD_star']), rms_noise_D_star=rms((p2['CD_star_halves'][0] - p2['CD_star_halves'][1]) / 2))
    if star: res['CD_star_blk'] = p2['CD_star'][:nb, :nb].cpu().numpy()
    if powers:
        for (pp, qq), key in (((3, 1), 'C31'), ((3, 3), 'C33')):
            res.update({f'rms_{key}': rms(p2[key]), f'rms_noise_{key}': rms((p2[key + '_halves'][0] - p2[key + '_halves'][1]) / 2)})
            if theory: res[f'rms_err_{key}'] = rms(pred_pq[(pp, qq)] - p2[key]); res[f'Cbar{key[1:]}_blk'] = pred_pq[(pp, qq)][:nb, :nb].cpu().numpy()
            res[f'{key}_blk'] = p2[key][:nb, :nb].cpu().numpy()
        if theory: res['rms_check11'] = rms(pred_pq[(1, 1)] - Cbar_dmft[0])      # (p, q) = (1, 1) must reproduce the covariance ansatz
    # store small blocks for plots: first 200 x 200 at all lags
    if lag_halves:   # sampling floor at every stored lag (lag 0 = rms_noise): half the difference of the two IC halves, off-diagonal RMS over B x B
        noise_lags = torch.cat([noise[None], (p2['Cphi_lag_halves'][0] - p2['Cphi_lag_halves'][1]) / 2], 0)    # [n_lags, B, B]
        res['rms_noise_lags'] = [rms(noise_lags[q]) for q in range(noise_lags.shape[0])]
        res['noise_blk_lags'] = noise_lags[:, :nb, :nb].float().cpu().numpy()                                    # diagonal floors (first nb sites)
    res['C_blk'] = Clag[:, :nb, :nb].cpu().numpy(); res['Cbar_emp_blk'] = Cbar_emp[:, :nb, :nb].cpu().numpy()
    res['CD_blk'] = p2['CD'][:nb, :nb].cpu().numpy(); res['noise_blk'] = noise[:nb, :nb].cpu().numpy()
    if theory:
        res['Cbar_th_blk'] = Cbar_th[:, :nb, :nb].cpu().numpy()
        if model == 'erf': res['Cbar_dmft_blk'] = Cbar_dmft[:, :nb, :nb].cpu().numpy()
        if tap: res['Cbar_tap_blk'] = Cbar_tap[:, :nb, :nb].cpu().numpy()
    res['ac_D_mean'] = p2['ac_D'].mean(0).cpu().numpy(); res['ac_D_site'] = p2['ac_D'].cpu().numpy().astype('float32'); res['ac_phi_site'] = p2['ac_phi'].cpu().numpy().astype('float32'); res['beta_site'] = p1['beta'].cpu().numpy() if model in LINEAR else None; res['t_total'] = time.time() - t0
    return res

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); out = sys.argv[2]
    Ns = cfg.pop('Ns'); allres = []
    for N in Ns:
        r = run(N=N, **cfg)
        allres.append(r)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if not hasattr(v, 'shape') and not isinstance(v, list)}), flush=True)
        pickle.dump(allres, open(out, 'wb'))
        torch.cuda.empty_cache()
