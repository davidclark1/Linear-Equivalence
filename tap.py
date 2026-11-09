"""TAP-solved evaluation of the nonzero-mean ansatz (Sec. VII of the paper): the temporal mean inputs come from solving the
TAP-like equation  phibar_i = M_star(b_i + sum_j J_ij phibar_j)  (Eq. (63) of the paper) instead of from the measured phibar.
M_star is the same single-site function as in the single-site ansatz (theory on an hbar grid with the MEASURED centered autocovariance C_pop and,
for the plasticity model, the measured q), interpolated linearly over the grid exactly as there (np.interp semantics: constant
beyond the ends).  The TAP solution hbar_TAP then replaces the measured hbar in the site-specific responses and residual spectra.

solve_tap: damped Newton (line search on |F|_2) on F(x) = x - M(b + J x) with the exact Jacobian of the piecewise-linear interpolant,
I - diag(M'(h)) J; fallback to damped fixed-point iteration if Newton does not reach tol.  Start: x0 = M(b) (no measured input).
tap_on_grid: solve, and if hbar_TAP leaves the theory grid, extend the grid (same spacing, margin 0.3) by calling theory_fn, re-solve.
save_J: full J as float32 to $NZM_JDIR (only if that directory exists and is writable), one file per (g, seed, N); an existing file is
checked against the checksum instead of being rewritten."""
import os, math, numpy as np, torch


def _M_slope(h, hg, m):
    """Piecewise-linear interpolant of m over hg at h (np.interp semantics) and its slope (0 beyond the ends)."""
    n = len(hg)
    hc = h.clamp(hg[0], hg[-1])
    idx = torch.clamp(torch.searchsorted(hg, hc) - 1, 0, n - 2)
    sl = (m[idx + 1] - m[idx]) / (hg[idx + 1] - hg[idx])
    val = m[idx] + sl * (hc - hg[idx])
    sl = torch.where((h < hg[0]) | (h > hg[-1]), torch.zeros_like(sl), sl)
    return val, sl


def solve_tap(J, b, hg, m, x0=None, tol=1e-10, max_newton=60, max_picard=20000, damp=0.5):
    """Returns dict(phibar, hbar, converged, iters, method, resid (max |F|), hist (max |F| per iteration))."""
    dev = J.device
    J64 = J.double(); b64 = torch.as_tensor(b, device=dev).double()
    hg = torch.as_tensor(np.asarray(hg), device=dev, dtype=torch.float64); m = torch.as_tensor(np.asarray(m), device=dev, dtype=torch.float64)
    Mf = lambda x: _M_slope(b64 + J64 @ x, hg, m)
    x = Mf(torch.zeros_like(b64))[0] if x0 is None else torch.as_tensor(x0, device=dev).double().clone()
    I = torch.eye(len(b64), device=dev, dtype=torch.float64)
    hist = []; method = 'newton'; conv = False; it = 0        # it = number of Newton (+ fixed-point) steps taken
    Mv, sl = Mf(x); F = x - Mv; x_best, r_best = x, F.abs().max().item()
    for _ in range(max_newton):
        r = F.abs().max().item(); hist.append(r)
        if r < r_best: x_best, r_best = x, r
        if r < tol: conv = True; break
        it += 1
        dx = torch.linalg.solve(I - sl[:, None] * J64, -F)
        if not torch.isfinite(dx).all(): break                     # singular Newton matrix: go to the fixed-point fallback
        n0 = F.norm().item(); t = 1.0
        while True:
            xn = x + t * dx; Mn, sln = Mf(xn); Fn = xn - Mn
            if Fn.norm().item() < (1 - 1e-4 * t) * n0 or t < 1e-3: break
            t *= 0.5
        x, Mv, sl, F = xn, Mn, sln, Fn
    if not conv:
        r = F.abs().max().item(); hist.append(r); conv = r < tol
        if r < r_best: x_best, r_best = x, r
    if not conv:          # fallback: damped fixed point from the best Newton iterate (M is bounded, so the iterates stay finite)
        method = 'newton+picard'; k = -1; x = x_best
        for k in range(max_picard):
            x = (1 - damp) * x + damp * Mf(x)[0]
            if k % 10 == 9:
                r = (x - Mf(x)[0]).abs().max().item(); hist.append(r)
                if r < tol: conv = True; break
            if not math.isfinite(x.abs().max().item()): break
        it += k + 1
    F = x - Mf(x)[0]
    return dict(phibar=x, hbar=b64 + J64 @ x, converged=bool(conv), iters=it, method=method, resid=F.abs().max().item(), hist=hist)


def tap_on_grid(J, b, hg, th, theory_fn, phibar_meas, margin=0.3, max_ext=4, **kw):
    """Solve the TAP-like equation with M = th['m'] on hg; extend the grid while hbar_TAP leaves it.
    theory_fn(hg_new, n_lo, n_hi, th) -> theory dict on hg_new (n_lo / n_hi new points below / above the old grid).
    Also solves from the measured phibar as a uniqueness check.  Returns (sol, hg, th, info)."""
    hg = np.asarray(hg, dtype=np.float64); dh = float(hg[1] - hg[0]); n_ext = 0
    for ext in range(max_ext + 1):
        sol = solve_tap(J, b, hg, th['m'], **kw)
        lo, hi = sol['hbar'].min().item(), sol['hbar'].max().item()
        if (lo >= hg[0] and hi <= hg[-1]) or ext == max_ext or not sol['converged']: break
        n_lo = int(math.ceil((hg[0] - (lo - margin)) / dh)) if lo < hg[0] else 0
        n_hi = int(math.ceil((hi + margin - hg[-1]) / dh)) if hi > hg[-1] else 0
        hg_new = np.concatenate([hg[0] - dh * np.arange(n_lo, 0, -1), hg, hg[-1] + dh * np.arange(1, n_hi + 1)])
        th = theory_fn(hg_new, n_lo, n_hi, th); hg = hg_new; n_ext += n_lo + n_hi
    sol2 = solve_tap(J, b, hg, th['m'], x0=phibar_meas, **kw)
    pm = torch.as_tensor(phibar_meas, device=J.device).double()
    hgt = torch.as_tensor(hg, device=J.device, dtype=torch.float64); mt = torch.as_tensor(np.asarray(th['m']), device=J.device, dtype=torch.float64)
    sl_tap = _M_slope(sol['hbar'], hgt, mt)[1]; sl_meas = _M_slope(torch.as_tensor(b, device=J.device).double() + J.double() @ pm, hgt, mt)[1]
    # disk radius of diag(M') J (= g rms_i M'(hbar_i); M' = S_star(omega = 0 | hbar)): the TAP Jacobian is stable only if < 1
    g_est = J.double().pow(2).sum().sqrt().item() / math.sqrt(J.shape[0])
    info = dict(tap_converged=sol['converged'], tap_iters=sol['iters'], tap_method=sol['method'], tap_resid=sol['resid'], tap_hist=sol['hist'],
                tap_grid_ext=n_ext, tap_n_out_of_grid=int(((sol['hbar'] < hg[0]) | (sol['hbar'] > hg[-1])).sum().item()),
                tap_converged_measstart=sol2['converged'], tap_iters_measstart=sol2['iters'],
                tap_rms_start_diff=(sol['phibar'] - sol2['phibar']).pow(2).mean().sqrt().item(),
                tap_rms_dphi=(sol['phibar'] - pm).pow(2).mean().sqrt().item(),
                tap_radius=g_est * sl_tap.pow(2).mean().sqrt().item(), tap_radius_meas=g_est * sl_meas.pow(2).mean().sqrt().item())
    return sol, hg, th, info


def extend_sampled(theory_call):
    """theory_fn for the sampled single-site theory: compute only the new points (same seed -> same Gaussian paths) and concatenate."""
    def fn(hg_new, n_lo, n_hi, th):
        parts_lo = theory_call(hg_new[:n_lo]) if n_lo else None
        parts_hi = theory_call(hg_new[len(hg_new) - n_hi:]) if n_hi else None
        out = dict(hgrid=hg_new)
        for k in ('m', 'Cc', 'S'):
            out[k] = np.concatenate([a[k] for a in (parts_lo, th, parts_hi) if a is not None], 0)
        return out
    return fn


def interp_sites(A, hg, hb):
    """[n_h, ...] tensor on grid hg -> [N, ...] linear interpolation at hb (the same rule as the single-site ansatz; clamped at the ends)."""
    dev = A.device
    hbt = torch.as_tensor(hb, device=dev, dtype=torch.float64); hgt = torch.as_tensor(np.asarray(hg), device=dev, dtype=torch.float64)
    idx = torch.clamp(torch.searchsorted(hgt, hbt) - 1, 0, len(hg) - 2); frac = ((hbt - hgt[idx]) / (hgt[idx + 1] - hgt[idx]))[:, None]
    return (1 - frac) * A[idx] + frac * A[idx + 1]


def J_checksum(J):
    J64 = J.double()
    return dict(J_sum=J64.sum().item(), J_sumsq=(J64 ** 2).sum().item(), J_row0=J[0, :4].double().cpu().numpy())


def save_J(J, N, g, seed):
    """Full J as float32 .npy in $NZM_JDIR if it exists and is writable.  Returns the path or None (skipped).  If the file exists,
    its checksum is compared and a mismatch is reported (the existing file is left as it is)."""
    d = os.environ.get('NZM_JDIR', '')
    if not d or not os.path.isdir(d) or not os.access(d, os.W_OK):
        return None, 'skipped (NZM_JDIR unset, missing or not writable)'
    path = f'{d}/J_g{g:g}_s{seed}_N{N}.npy'
    Jn = J.float().cpu().numpy()
    if os.path.exists(path):
        old = np.load(path, mmap_mode='r')
        same = old.shape == Jn.shape and float(np.abs(old[:64] - Jn[:64]).max()) == 0.0 and abs(float(old.sum(dtype=np.float64)) - float(Jn.sum(dtype=np.float64))) < 1e-6
        return path, 'exists, identical' if same else 'exists, DIFFERENT J (not overwritten)'
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'wb') as f: np.save(f, Jn)
    os.replace(tmp, path)
    return path, 'written'


def mean_floors(J, b, hg, m, phibar_halves):
    """Sampling floors for the TAP-like equation from the two disjoint halves of the initial conditions (pass 2):
    floor of phibar = RMS_i (phibar_i^(1) - phibar_i^(2)) / 2, and floor of the TAP residual r_i = phibar_i - M_star(hbar_i) with
    hbar_i^(h) = b_i + sum_j J_ij phibar_j^(h): RMS_i (r_i^(1) - r_i^(2)) / 2.  Also the residual RMS of each half."""
    dev = J.device; J64 = J.double(); b64 = torch.as_tensor(b, device=dev).double()
    ph = torch.as_tensor(phibar_halves, device=dev).double()                     # [2, N]
    hh = b64[None, :] + ph @ J64.T                                               # [2, N]
    hgt = torch.as_tensor(np.asarray(hg), device=dev, dtype=torch.float64); mt = torch.as_tensor(np.asarray(m), device=dev, dtype=torch.float64)
    r = ph - torch.stack([_M_slope(hh[k], hgt, mt)[0] for k in range(2)])
    rms = lambda x: x.pow(2).mean().sqrt().item()
    return dict(phibar_ic_halves=ph.float().cpu().numpy(), hbar_ic_halves=hh.float().cpu().numpy(),
                floor_phibar=rms((ph[0] - ph[1]) / 2), floor_meaneq=rms((r[0] - r[1]) / 2), meaneq_half1=rms(r[0]), meaneq_half2=rms(r[1]))
