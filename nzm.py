"""
Core module: network simulation, single-site theory, and the linear-equivalent ansatz.
Framework: phi_i(t) = T[h_i](t),  h_i = sum_j J_ij phi_j + b_i (+ external drive),  J_ij ~ N(0, g^2/N) iid.
Models (argument model):
  'erf':     (1+d/dt) x = h,  phi = erf(sqrt(pi) x / 2)        (the SCS model of the paper, f'(0) = 1)
  'sig':     (1+d/dt) x = h,  phi = (1+tanh x)/2               (recurrent network with sigmoid units)
  'glvsat':  dn/dt = n (1 - n - h) + lam,  phi = n/(1+n)       (saturating Lotka-Volterra model, immigration rate lam)
  'ca':      sigmoid units with Hebbian plasticity, absorbed into a self-coupling (pass1_ca, pass2_ca)
Two passes: pass1 runs a few trajectories and measures the temporal means, the gains or response kernels, the population
autocovariance and the largest Lyapunov exponent; pass2 streams long runs over many parallel initial conditions, accumulating
B x B block covariances of the centered activities and of the centered residuals  Delta_i = phi_i - S_i * eta_i.
"""
import math, time
import numpy as np
import torch

# ----------------------------------------------------------------------------- helpers
def make_J(N, g, seed, device='cuda', mu=0.0):
    gen = torch.Generator(device=device).manual_seed(seed)
    J = g / math.sqrt(N) * torch.randn(N, N, generator=gen, device=device) + mu / N
    J.fill_diagonal_(0.0)
    return J

def make_bias(N, sI, seed, device='cuda'):
    gen = torch.Generator(device=device).manual_seed(seed + 12345)
    return sI * torch.randn(N, generator=gen, device=device) if sI > 0 else torch.zeros(N, device=device)

LINEAR = ('sig', 'erf')      # units whose state is a linear filter of the input: x' = -x + h, phi = f(x)
SQRT_PI_2 = math.sqrt(math.pi) / 2

def phi_of(model, s):
    if model == 'sig': return 0.5 * (1 + torch.tanh(s))
    if model == 'erf': return torch.erf(SQRT_PI_2 * s)
    if model == 'glvsat': return s / (1 + s)
    return s

def fprime(model, s):
    if model == 'erf': return torch.exp(-math.pi * s ** 2 / 4)
    return 0.5 * (1 - torch.tanh(s) ** 2)

def f_np(model):
    """(f, f') as numpy functions for the single-site Gaussian integrals."""
    if model == 'erf':
        from scipy.special import erf
        return (lambda x: erf(SQRT_PI_2 * x)), (lambda x: np.exp(-np.pi * x ** 2 / 4))
    return (lambda x: 0.5 * (1 + np.tanh(x))), (lambda x: 0.5 * (1 - np.tanh(x) ** 2))

def step(model, s, h, dt, lam, noise=None):
    if model in LINEAR:
        return s + dt * (-s + h) + (noise if noise is not None else 0.)
    if noise is None:
        sn = s + dt * (s * (1 - s - h) + lam)
    else:   # with multiplicative input noise: log-space (exponential) integrator, stable and positivity-preserving
        sn = s * torch.exp(dt * (1 - s - h) - noise) + dt * lam
    return sn.clamp(max=20.0) if model == 'glv' else sn   # 'glv' (not used in the paper): guard against rare unbounded growth

def init_state(model, n_ics, N, device):
    if model in LINEAR:
        return torch.randn(n_ics, N, device=device)
    return 0.1 + torch.rand(n_ics, N, device=device)

# ----------------------------------------------------------------------------- pass 1 (streaming: means, C_pop, per-site kernels)
def pass1(model, N, g, seed, n_ics=8, T=2000., T_burn=300., dt=0.05, sI=0., lam=0., dt_r=0.1, tau_max=20.,
          lyap=True, device='cuda', J=None, b=None, sig_xi=0.):
    """Short pass with n_ics trajectories: time-averaged activities phibar_i, population autocovariance C_pop(tau) (lags 0..L on the
    dt_r grid), site gains beta_i = <f'(x_i)> (sigmoid) or per-site response kernels S_i(tau) by two-time tangent integration of the
    unit's own dynamics along the network trajectory (Lotka-Volterra), Lyapunov exponent, and Gaussianity check of the local field.
    Everything is accumulated on the fly (no stored snapshots); memory ~ (L+1) * n_ics * N floats."""
    torch.manual_seed(seed)
    J = make_J(N, g, seed, device) if J is None else J
    b = make_bias(N, sI, seed, device) if b is None else b
    k = int(round(dt_r / dt)); n_burn = int(T_burn / dt); n_rec = int(T / dt); n_save = n_rec // k
    L = int(round(tau_max / dt_r)); glvlike = model in ('glv', 'glvsat'); nE = min(N, 256)
    s = init_state(model, n_ics, N, device)
    s2 = s[:1] + 1e-3 * torch.randn(1, N, device=device); lyap_acc, lyap_n, eps = 0., 0, 1e-3
    smin = float('inf'); t0 = time.time()
    # accumulators
    sum_phi = torch.zeros(2, N, device=device, dtype=torch.float64)                  # two halves of the record (stationarity check)
    ring = torch.zeros(N, L + 1, n_ics, device=device); ringE = torch.zeros(nE, L + 1, n_ics, device=device)
    ac = torch.zeros(N, L + 1, device=device, dtype=torch.float64); acE = torch.zeros(nE, L + 1, device=device, dtype=torch.float64)
    sumE = torch.zeros(4, nE, device=device, dtype=torch.float64)                    # per-site moments 1..4 of the local field (nE sites)
    if model in LINEAR:
        sum_x = torch.zeros(N, device=device, dtype=torch.float64); sum_x2 = torch.zeros(N, device=device, dtype=torch.float64); sum_fp = torch.zeros(N, device=device, dtype=torch.float64)
    else:   # per-site two-time tangent kernels: unit impulse (dh = 1 for one step) every `stride` record steps, L/stride slots
        stride = max(1, L // 40)
        while L % stride: stride -= 1
        Ls = L // stride; ds = torch.zeros(Ls, n_ics, N, device=device)
        acc = torch.zeros(L + 2, N, device=device, dtype=torch.float64); acc32 = torch.zeros(L + 2, N, device=device)
        cnt = torch.zeros(L + 2, device=device, dtype=torch.float64); kslot = torch.full((Ls,), -10 ** 9, device=device, dtype=torch.long)
        ones_s = torch.ones(Ls, device=device, dtype=torch.float64); slot_ptr = 0
    ar = torch.arange(L + 1, device=device); ptr = 0; n_ac = 0
    for it in range(n_burn + n_rec):
        phi = phi_of(model, s); h = phi @ J.T + b
        noise = sig_xi * math.sqrt(dt) * torch.randn(n_ics, N, device=device) if sig_xi > 0 else None
        rec_i = (it - n_burn) // k if it >= n_burn and (it - n_burn) % k == 0 else None      # record index on the dt_r grid
        if rec_i is not None:
            eta = (h - b)[:, :nE]
            sum_phi[rec_i * 2 // n_save if rec_i < n_save else 1] += phi.sum(0).double()
            ring[:, ptr, :] = phi.T; ringE[:, ptr, :] = eta.T
            if rec_i >= L:
                n_ac += 1; idx = (ptr - ar) % (L + 1)
                ac += torch.bmm(ring, phi.T[:, :, None])[:, idx, 0].double()
                acE += torch.bmm(ringE, eta.T[:, :, None])[:, idx, 0].double()
            e = eta.double(); sumE[0] += e.sum(0); sumE[1] += (e ** 2).sum(0); sumE[2] += (e ** 3).sum(0); sumE[3] += (e ** 4).sum(0)
            ptr = (ptr + 1) % (L + 1)
            if model in LINEAR:
                sum_x += s.sum(0).double(); sum_x2 += (s.double() ** 2).sum(0); sum_fp += fprime(model, s).sum(0).double()
            if glvlike and rec_i % 100 == 0: smin = min(smin, s.min().item())
        if glvlike:
            dh = 0.
            if rec_i is not None:
                dphi = (ds / (1 + s) ** 2) if model == 'glvsat' else ds                            # [Ls, n_ics, N] = dt * G
                lag = rec_i - kslot; lag_i = torch.where((lag >= 1) & (lag <= L), lag, torch.full_like(lag, L + 1))
                acc32.index_add_(0, lag_i, dphi.sum(1)); cnt.index_add_(0, lag_i, ones_s)
                if rec_i % 256 == 255: acc += acc32.double(); acc32.zero_()
                if rec_i % stride == 0:
                    ds[slot_ptr] = 0; dh = torch.zeros(Ls, 1, 1, device=device); dh[slot_ptr] = 1.0
                    kslot[slot_ptr] = rec_i; slot_ptr = (slot_ptr + 1) % Ls
            if noise is None:
                ds = ds * (1 + dt * (1 - 2 * s - h))[None] - dt * s[None] * dh
            else:
                fac = torch.exp(dt * (1 - s - h) - noise)
                ds = ds * (fac * (1 - dt * s))[None] - dt * (s * fac)[None] * dh
        s = step(model, s, h, dt, lam, noise)
        if lyap:
            s2 = step(model, s2, phi_of(model, s2) @ J.T + b, dt, lam, noise[:1] if noise is not None else None)
            if it % 20 == 19:
                d = torch.sqrt(((s2 - s[:1]) ** 2).sum())
                if it >= n_burn // 2:
                    lyap_acc += math.log(d.item() / eps); lyap_n += 20
                s2 = s[:1] + (s2 - s[:1]) * (eps / d)
    n_tot = n_save * n_ics
    phibar = (sum_phi.sum(0) / n_tot).float()
    out = dict(model=model, N=N, g=g, seed=seed, sI=sI, lam=lam, sig_xi=sig_xi, dt=dt, dt_r=dt_r, tau_max=tau_max, T=T,
               n_ics=n_ics, lyap=lyap_acc / (lyap_n * dt) if lyap_n else float('nan'), smin=smin, t_pass1=time.time() - t0,
               phibar=phibar, b=b, J=J, hbar=J @ phibar + b,
               phibar_halves=torch.stack([sum_phi[0] / (n_tot // 2), sum_phi[1] / (n_tot - n_tot // 2)]).float())
    ac_site = ac / (n_ac * n_ics) - phibar.double()[:, None] ** 2                     # per-site centered autocovariance
    out['C_pop'] = ac_site.mean(0); out['var_phi'] = out['C_pop'][0].item()
    if model in LINEAR:
        out['xbar'] = (sum_x / n_tot).float(); out['beta'] = (sum_fp / n_tot).float()
        out['var_x'] = (sum_x2 / n_tot - (sum_x / n_tot) ** 2).mean().item()
    else:
        acc += acc32.double()
        out['S_site'] = (acc[:L + 1] / cnt[:L + 1].clamp_min(1)[:, None] / n_ics / dt).T.contiguous()   # [N, L+1] measured per-site kernels
        out['S_site_stride'] = stride
    # local-field Gaussianity check on nE sites: kurtosis of eta-breve and its autocovariance (to compare with g^2 C_pop)
    m1, m2, m3, m4 = sumE / n_tot; mE = m1
    c2 = m2 - m1 ** 2; c4 = m4 - 4 * m3 * m1 + 6 * m2 * m1 ** 2 - 3 * m1 ** 4                 # per-site central moments
    out['eta_kurt'] = (c4.mean() / c2.mean() ** 2 - 3).item()
    out['C_eta_pop'] = (acE / (n_ac * n_ics) - mE[:, None] ** 2).mean(0)
    del ring, ringE; torch.cuda.empty_cache()
    return out

def autocorr_pop(Pc, L):
    """Pc: [n_ics, N, T] centered. Returns population-averaged autocov at lags 0..L (unbiased in T)."""
    n_ics, N, T = Pc.shape
    acc = torch.zeros(L + 1, device=Pc.device, dtype=torch.float64)
    for c in range(n_ics):
        for s in range(0, N, 512):
            Y = Pc[c, s:s + 512]
            F = torch.fft.rfft(Y, n=2 * T)
            acc += torch.fft.irfft(F.abs() ** 2, n=2 * T)[:, :L + 1].sum(0).double()
    norm = torch.arange(T, T - L - 1, -1, device=Pc.device, dtype=torch.float64)
    return acc / norm / (N * n_ics)

# ----------------------------------------------------------------------------- pass 2 (streaming)
def pass2(model, N, g, seed, kernels, n_ics=400, T=2000., T_burn=300., dt=0.05, sI=0., lam=0., dt_r=0.1,
          tau_max=20., B=None, lags_block=(1., 2., 5., 10.), device='cuda', J=None, b=None, sig_xi=0., ac_every=5, tau_res=10.,
          beta_star=None, powers=False, lag_halves=False):
    """kernels: {'beta': [N]} for sig (Delta = phi - beta x), {'S': [N, L+1]} for glv (ring-buffer convolution).
    LINEAR units only: beta_star (float) also accumulates the residual with the population gain, Delta_i = phi_i - beta_star x_i (CD_star);
    powers=True also accumulates the equal-time two-site moments of odd powers <phi_i^3 phi_j> (C31[i, j]) and <phi_i^3 phi_j^3> (C33).
    lag_halves=True (case pm2_a800): the lagged block covariances are also accumulated separately for the two halves of the
    initial conditions (Cphi_lag_halves [2, n_lags, B, B], each centered with its own half mean, as Cphi_halves at lag 0), for a sampling
    floor at every stored lag.  The pooled Cphi_lag is the same estimator as without the flag (only the summation order changes)."""
    torch.manual_seed(seed + 7)
    J = make_J(N, g, seed, device) if J is None else J
    b = make_bias(N, sI, seed, device) if b is None else b
    B = min(N, 1000) if B is None else B
    k = int(round(dt_r / dt)); n_burn = int(T_burn / dt); n_rec = int(T / dt)
    L = int(round(tau_max / dt_r)); lag_idx = [int(round(l / dt_r)) for l in lags_block]
    s = init_state(model, n_ics, N, device)
    y = s.clone() if model in LINEAR else None   # noise-free filtered local field (1+d/dt) y = h
    half = n_ics // 2
    # accumulators (two halves of the ICs for a noise estimate)
    sum_phi = torch.zeros(2, N, device=device, dtype=torch.float64); sum_D = torch.zeros(2, N, device=device, dtype=torch.float64)
    PP = torch.zeros(2, B, B, device=device, dtype=torch.float64); DD = torch.zeros(2, B, B, device=device, dtype=torch.float64)
    PD = torch.zeros(2, B, B, device=device, dtype=torch.float64)
    if beta_star is not None: DD2 = torch.zeros(2, B, B, device=device, dtype=torch.float64); sum_D2 = torch.zeros(2, N, device=device, dtype=torch.float64)
    if powers: P3P = torch.zeros(2, B, B, device=device, dtype=torch.float64); P3P3 = torch.zeros(2, B, B, device=device, dtype=torch.float64); sum_p3 = torch.zeros(2, N, device=device, dtype=torch.float64)
    PPlag = torch.zeros(len(lag_idx), B, B, device=device, dtype=torch.float64)
    if lag_halves: PPlag_h = torch.zeros(2, len(lag_idx), B, B, device=device, dtype=torch.float64)
    ac_phi = torch.zeros(N, L + 1, device=device, dtype=torch.float64); ac_D = torch.zeros(N, L + 1, device=device, dtype=torch.float64)
    # rings in layout [N, slot, n_ics] so that every contraction is a contiguous batched matmul (no transposed copies per step)
    ring_phi = torch.zeros(N, L + 1, n_ics, device=device); ring_D = torch.zeros(N, L + 1, n_ics, device=device)
    glvlike = model in ('glv', 'glvsat')
    if glvlike:
        Lr = min(L, int(round(tau_res / dt_r)))                        # residual convolution length (kernel negligible beyond tau_res)
        ring_eta = torch.zeros(N, Lr + 1, n_ics, device=device); arr = torch.arange(Lr + 1, device=device); ptr_e = 0
        S_k = kernels['S'][:, :Lr + 1].float() * dt_r                   # [N, Lr+1]
    beta = kernels.get('beta', None)
    n_acc = 0; n_ac = 0; ptr = 0; t0 = time.time(); ar = torch.arange(L + 1, device=device)
    for it in range(n_burn + n_rec):
        phi = phi_of(model, s); h = phi @ J.T + b
        rec = it >= n_burn and (it - n_burn) % k == 0
        if rec:
            eta = h - b
            if glvlike:
                ring_eta[:, ptr_e, :] = eta.T
                # Delta_c,n = sum_j S_k[n, (ptr_e - j) mod (Lr+1)] ring_eta[n, j, c]
                D = phi - torch.bmm(S_k[:, (ptr_e - arr) % (Lr + 1)][:, None, :], ring_eta)[:, 0, :].T
                ptr_e = (ptr_e + 1) % (Lr + 1)
            else:
                D = phi - beta[None, :] * y
            ring_phi[:, ptr, :] = phi.T; ring_D[:, ptr, :] = D.T
            if (it - n_burn) // k >= L:       # rings full
                n_acc += 1
                for hh, sl in enumerate((slice(0, half), slice(half, n_ics))):
                    p, d = phi[sl], D[sl]
                    sum_phi[hh] += p.sum(0).double(); sum_D[hh] += d.sum(0).double()
                    PP[hh] += (p[:, :B].T @ p[:, :B]).double(); DD[hh] += (d[:, :B].T @ d[:, :B]).double()
                    PD[hh] += (p[:, :B].T @ d[:, :B]).double()
                    if beta_star is not None:
                        d2 = p - beta_star * y[sl]; sum_D2[hh] += d2.sum(0).double(); DD2[hh] += (d2[:, :B].T @ d2[:, :B]).double()
                    if powers:
                        p3 = p ** 3; sum_p3[hh] += p3.sum(0).double()
                        P3P[hh] += (p3[:, :B].T @ p[:, :B]).double(); P3P3[hh] += (p3[:, :B].T @ p3[:, :B]).double()
                if n_acc % ac_every == 0:     # per-site autocorrelations from every ac_every-th reference point (correlation time >> ac_every dt_r)
                    n_ac += 1
                    idx = (ptr - ar) % (L + 1)                           # slot holding lag l
                    ac_phi += torch.bmm(ring_phi, phi.T[:, :, None])[:, idx, 0].double()
                    ac_D += torch.bmm(ring_D, D.T[:, :, None])[:, idx, 0].double()
                for q, li in enumerate(lag_idx):
                    pl = ring_phi[:B, (ptr - li) % (L + 1), :]           # [B, n_ics]
                    if lag_halves:                                       # same products, summed per half of the ICs
                        for hh, sl in enumerate((slice(0, half), slice(half, n_ics))):
                            PPlag_h[hh, q] += (phi[sl, :B].T @ pl[:, sl].T).double()
                    else:
                        PPlag[q] += (phi[:, :B].T @ pl.T).double()
            ptr = (ptr + 1) % (L + 1)
        noise = sig_xi * math.sqrt(dt) * torch.randn(n_ics, N, device=device) if sig_xi > 0 else None
        if model in LINEAR: y = y + dt * (-y + h)
        s = step(model, s, h, dt, lam, noise)
    # finalize: centered block covariances (per half and pooled)
    nh = n_acc * half   # n_ics is forced even by the driver
    mphi = sum_phi / nh; mD = sum_D / nh
    def cov(Sxy, mx, my):
        return Sxy / nh - mx[:B, None] * my[None, :B]
    Cphi_h = torch.stack([cov(PP[i], mphi[i], mphi[i]) for i in range(2)])
    CD_h = torch.stack([cov(DD[i], mD[i], mD[i]) for i in range(2)])
    CPD_h = torch.stack([cov(PD[i], mphi[i], mD[i]) for i in range(2)])
    phibar = (sum_phi.sum(0) / (n_acc * 2 * half)); Dbar = sum_D.sum(0) / (n_acc * 2 * half)
    Cphi = Cphi_h.mean(0); CD = CD_h.mean(0); CPD = CPD_h.mean(0)
    if lag_halves: PPlag = PPlag_h.sum(0)
    Cphi_lag = PPlag / (n_acc * n_ics) - phibar[:B, None] * phibar[None, :B]
    ac_phi = ac_phi / (n_ac * n_ics) - phibar[:, None] ** 2
    ac_D = ac_D / (n_ac * n_ics) - Dbar[:, None] ** 2
    out = dict(model=model, N=N, g=g, seed=seed, sI=sI, lam=lam, dt=dt, dt_r=dt_r, tau_max=tau_max, B=B, n_ics=n_ics,
               T=T, T_tot=n_acc * dt_r * n_ics, alpha=n_acc * dt_r * n_ics / N, t_pass2=time.time() - t0,
               phibar=phibar.float(), Dbar=Dbar.float(), hbar=(J @ phibar.float() + b), b=b,
               Cphi=Cphi, CD=CD, CPD=CPD, Cphi_halves=Cphi_h, CD_halves=CD_h, Cphi_lag=Cphi_lag, lags_block=list(lags_block),
               ac_phi=ac_phi, ac_D=ac_D, C_pop=ac_phi.mean(0), CD_pop=ac_D.mean(0),
               phibar_ic_halves=mphi.float())      # [2, N] phibar from the two disjoint halves of the initial conditions
    if lag_halves:
        out['Cphi_lag_halves'] = torch.stack([PPlag_h[i] / nh - mphi[i][None, :B, None] * mphi[i][None, None, :B] for i in range(2)])
    if beta_star is not None:
        mD2 = sum_D2 / nh; out['CD_star_halves'] = torch.stack([cov(DD2[i], mD2[i], mD2[i]) for i in range(2)]); out['CD_star'] = out['CD_star_halves'].mean(0)
    if powers:
        mp3 = sum_p3 / nh
        out['C31_halves'] = torch.stack([cov(P3P[i], mp3[i], mphi[i]) for i in range(2)]); out['C31'] = out['C31_halves'].mean(0)
        out['C33_halves'] = torch.stack([cov(P3P3[i], mp3[i], mp3[i]) for i in range(2)]); out['C33'] = out['C33_halves'].mean(0)
    return out

# ----------------------------------------------------------------------------- ansatz
def spectrum_even(c, dt_r, omega, taper=True):
    """Two-sided FT of an even autocov given at lags 0..L: c(omega) = dt_r [c0 + 2 sum_l w_l c_l cos(omega tau_l)].
    c: [..., L+1] (float64), omega: [n_w]. Returns [..., n_w] real."""
    L = c.shape[-1] - 1
    tau = torch.arange(L + 1, device=c.device, dtype=torch.float64) * dt_r
    w = torch.ones(L + 1, device=c.device, dtype=torch.float64)
    if taper:
        i0 = 3 * L // 4
        w[i0:] = 0.5 * (1 + torch.cos(math.pi * (tau[i0:] - tau[i0]) / (tau[-1] - tau[i0])))
    w[0] = 0.5
    cosm = torch.cos(omega[None, :] * tau[:, None])            # [L+1, n_w]
    return 2 * dt_r * ((c * w) @ cosm)

def spectrum_causal(S, dt_r, omega, taper=True):
    """One-sided FT of a causal kernel on lags 0..L (rectangle rule, consistent with the discrete convolution)."""
    L = S.shape[-1] - 1
    tau = torch.arange(L + 1, device=S.device, dtype=torch.float64) * dt_r
    w = torch.ones(L + 1, device=S.device, dtype=torch.float64)
    if taper:
        i0 = 3 * L // 4
        w[i0:] = 0.5 * (1 + torch.cos(math.pi * (tau[i0:] - tau[i0]) / (tau[-1] - tau[i0])))
    e = torch.exp(-1j * omega[None, :] * tau[:, None])          # [L+1, n_w]
    return dt_r * ((S * w).to(torch.complex128) @ e)

def ansatz(J, DS, DD, omega, lags, B, chunk=16):
    """Cbar(tau)[:B,:B] = (1/2pi) int dw e^{i w tau} M(w) diag(DD(w)) M(w)^dagger,  M = (I - diag(DS(w)) J)^{-1}.
    DS: [N, n_w] complex, DD: [N, n_w] real (>=0), omega: [n_w] >= 0 uniform grid (symmetry used).
    Returns [n_lags, B, B] real (float64)."""
    N = J.shape[0]; dev = J.device
    chunk = max(1, min(chunk, int(5e9 // (N * N * 16))))     # keep the batched complex128 inversion within ~20 GB
    dw = (omega[1] - omega[0]).item()
    Jc = J.to(torch.complex128)
    I = torch.eye(N, device=dev, dtype=torch.complex128)
    out = torch.zeros(len(lags), B, B, device=dev, dtype=torch.complex128)
    lags_t = torch.tensor(lags, device=dev, dtype=torch.float64)
    for s in range(0, len(omega), chunk):
        w = omega[s:s + chunk]; nw = len(w)
        A = I[None] - DS[:, s:s + chunk].T.to(torch.complex128)[:, :, None] * Jc[None]      # [nw, N, N]
        M = torch.linalg.inv(A)[:, :B, :]                                                  # [nw, B, N]
        C = torch.einsum('wbn,wn,wcn->wbc', M, DD[:, s:s + chunk].T.to(torch.complex128), M.conj())  # [nw, B, B]
        wt = torch.full((nw,), dw / math.pi, device=dev, dtype=torch.float64)
        wt[w == 0] = dw / (2 * math.pi)
        ph = torch.exp(1j * lags_t[:, None] * w[None, :].to(torch.float64)) * wt[None, :]           # [n_lags, nw]
        out += torch.einsum('lw,wbc->lbc', ph, C)
    return out.real

def Kd(omega, dt):
    """Discrete-time transfer function of the Euler filter x_{n+1} = (1-dt) x_n + dt h_n, sampled every step:
    Kd = dt e^{-i w dt} / (1 - (1-dt) e^{-i w dt}); Kd -> 1/(1+iw) for w dt -> 0."""
    om = torch.as_tensor(omega, dtype=torch.float64)
    e = torch.exp(-1j * om * dt)
    return dt * e / (1 - (1 - dt) * e)

def filt_x_discrete(C_pop, g, dt, sig_xi=0., n_w=4001, dt_filter=None):
    """C^x(tau_l) (lag spacing dt = record step) for x = Kd*(eta) + noise path, eta ~ GP(0, g^2 C_pop), noise sig_xi sqrt(dt) eps_n:
    C^x(w) = |Kd|^2 g^2 C_pop(w) + |Kd/dt|^2 sig_xi^2 dt^2, inverted over the band [-pi/dt, pi/dt] of the record grid.
    dt_filter: Euler step of the leaky integrator when it differs from the record step (Kd is evaluated with it).
    Noise with dt_filter < dt (record grid coarser than the Euler step, e.g. erf dt 0.025 / record 0.1): the noise spectrum
    K2 sig_xi^2 reaches up to pi/dt_filter, so the in-band integral would miss ~2% of its variance; the noise part is then the
    exact autocovariance of the Euler AR(1) path x_{n+1} = (1 - dtf) x_n + xi_n, var xi = sig_xi^2 dtf, sampled at the record lags:
    sig_xi^2 (1 - dtf)^(tau/dtf) / (2 - dtf).  With dt_filter = dt (or None) the in-band formula is used (it is exact there)."""
    C_pop = np.asarray(C_pop, dtype=np.float64); L = len(C_pop) - 1; dtf = dt_filter or dt
    exact_noise = sig_xi > 0 and abs(dtf - dt) > 1e-12
    om = torch.linspace(0, math.pi / dt, n_w, dtype=torch.float64)
    Cw = spectrum_even(torch.as_tensor(C_pop), dt, om)
    K2 = Kd(om, dtf).abs() ** 2
    Cx_w = K2 * g ** 2 * Cw + (0. if exact_noise else K2 * sig_xi ** 2)
    tau = torch.arange(L + 1, dtype=torch.float64) * dt
    dw = om[1] - om[0]
    wt = torch.full_like(om, dw / math.pi); wt[0] = dw / (2 * math.pi)
    Cx = (torch.cos(tau[:, None] * om[None, :]) * (Cx_w * wt)[None, :]).sum(1).numpy()
    if exact_noise:
        Cx = Cx + sig_xi ** 2 * (1 - dtf) ** np.round(tau.numpy() / dtf) / (2 - dtf)
    return Cx

# ----------------------------------------------------------------------------- single-site theory
def filt_x(C_pop, g, dt_r):
    """C^x(tau) for (1+d/dt)x = eta, eta ~ GP(0, g^2 C_pop):  C^x = g^2 C_pop * k,  k(tau) = e^{-|tau|}/2, on lags 0..L."""
    L = len(C_pop) - 1
    tau = np.arange(-L, L + 1) * dt_r
    c2 = np.concatenate([C_pop[:0:-1], C_pop])                # symmetric, lags -L..L
    kern = 0.5 * np.exp(-np.abs(tau))
    full = g ** 2 * np.convolve(c2, kern, mode="full") * dt_r   # lags -2L..2L
    return full[2 * L: 3 * L + 1]

def theory_sig(C_pop, g, hgrid, dt_r, nq=None, sig_xi=0., discrete=True, dx=0.05, model='sig', dt_filter=None):
    """Single-site theory for the sig model on a fine uniform grid (exact to grid resolution; Gauss-Hermite fails
    when sqrt(v0) >> width of f').  x = hbar + x-breve, x-breve Gaussian with autocov Cx(tau).
    m(h) = E f(x), beta(h) = E f'(x), Cc(tau|h) = E f(x1) f(x2) - m^2 with corr(x1, x2) = rho_tau = Cx(tau)/Cx(0).
    Uses E[f(x2)|x1] = (f * N(0, s_tau^2))(h + rho (x1 - h)),  s_tau^2 = v0 (1 - rho^2)."""
    C_pop = np.asarray(C_pop, dtype=np.float64)
    if discrete:
        Cx = filt_x_discrete(C_pop, g, dt_r, sig_xi, dt_filter=dt_filter)
    else:
        Cx = filt_x(C_pop, g, dt_r) + 0.5 * sig_xi ** 2 * np.exp(-np.arange(len(C_pop)) * dt_r)
    v0 = max(Cx[0], 1e-10); sd = math.sqrt(v0)
    f, fp = f_np(model)
    hgrid = np.asarray(hgrid, dtype=np.float64)
    W = np.abs(hgrid).max() + 9 * sd
    X = np.arange(-W, W + dx, dx)                               # global fine grid
    fX = f(X)
    # per-lag Gaussian smoothing of f (FFT convolution); rho clipped away from |rho| = 1
    rho = np.clip(Cx / v0, -0.999999, 0.999999); rho[0] = 1.0
    nX = len(X); nfft = 1 << int(np.ceil(np.log2(2 * nX)))
    FX = np.fft.rfft(fX, nfft)
    k = np.fft.rfftfreq(nfft, d=dx) * 2 * np.pi
    zloc = np.arange(-8 * sd, 8 * sd + dx, dx)                  # local grid for x1 - h
    pdf = np.exp(-zloc ** 2 / (2 * v0)) / math.sqrt(2 * math.pi * v0) * dx
    m = np.array([np.sum(pdf * f(h + zloc)) for h in hgrid])
    beta = np.array([np.sum(pdf * fp(h + zloc)) for h in hgrid])
    Cc = np.zeros((len(hgrid), len(Cx)))
    for l, r_ in enumerate(rho):
        s2 = v0 * max(1 - r_ ** 2, 0.0)
        if s2 < 1e-12:
            gX = fX
        else:
            gX = np.fft.irfft(FX * np.exp(-0.5 * s2 * k ** 2), nfft)[:nX]
        for a, h in enumerate(hgrid):
            x1 = h + zloc
            g2 = np.interp(h + r_ * (x1 - h), X, gX)
            Cc[a, l] = np.sum(pdf * f(x1) * g2) - m[a] ** 2
    return dict(hgrid=hgrid, m=m, beta=beta, Cc=Cc, Cx=Cx, v0=v0)

def closure_sig(C_pop0, q0, g, sI, dt_r, n_iter=40, damp=0.5, nq=None, dh=0.25, verbose=False, sig_xi=0., discrete=True):
    """Self-consistent (C_pop, q) for the sig model: hbar ~ N(0, g^2 q + sI^2), averaged on a fine uniform hbar grid
    (a Gauss-Hermite rule over the wide hbar distribution under-resolves the narrow active window)."""
    C = np.asarray(C_pop0, dtype=np.float64).copy(); q = float(q0); hist = []
    for k in range(n_iter):
        sd = math.sqrt(g ** 2 * q + sI ** 2)
        hg = np.arange(-8 * sd, 8 * sd + dh, dh)
        wh = np.exp(-hg ** 2 / (2 * sd ** 2)); wh /= wh.sum()
        th = theory_sig(C, g, hg, dt_r, sig_xi=sig_xi, discrete=discrete)
        qn = float(np.sum(wh * th['m'] ** 2)); Cn = wh @ th['Cc']
        hist.append((k, qn, Cn[0], np.abs(Cn - C).max()))
        if verbose: print(f'  closure it {k}: q={qn:.5f} C0={Cn[0]:.5f} dC={np.abs(Cn - C).max():.2e}', flush=True)
        C = (1 - damp) * C + damp * Cn; q = (1 - damp) * q + damp * qn
    return C, q, hist

def closure_odd(C0, g, dt_r, n_iter=400, damp=0.6, model='erf', dt_filter=None, tol=1e-8, verbose=False, sig_xi=0.):
    """Self-consistent C_star(tau) for an odd unit with zero mean input (the paper's DMFT):  C <- Cc(tau | hbar = 0; input g^2 C).
    sig_xi > 0: i.i.d. white-noise drive (sig_xi sqrt(dt) per Euler step, as in pass1/pass2) added to the state x; C^x then
    includes the noise path (filt_x_discrete).  C0: starting autocovariance on lags 0..L (e.g. the measured C_pop).
    Returns (C_star, beta_star, history)."""
    C = np.asarray(C0, dtype=np.float64).copy(); hist = []
    for it in range(n_iter):
        th = theory_sig(C, g, np.array([0.]), dt_r, model=model, dt_filter=dt_filter, sig_xi=sig_xi)
        Cn = th['Cc'][0]; d = float(np.abs(Cn - C).max()); hist.append(d)
        if verbose: print(f'  closure_odd it {it}: C0={Cn[0]:.5f} beta={th["beta"][0]:.5f} dC={d:.2e}', flush=True)
        C = (1 - damp) * C + damp * Cn
        if d < tol: break
    th = theory_sig(C, g, np.array([0.]), dt_r, model=model, dt_filter=dt_filter, sig_xi=sig_xi)
    return C, float(th['beta'][0]), hist

def gp_samples(C, M, P, device='cuda'):
    """Stationary Gaussian paths (M x P, periodic) with autocov C[0..L] on the same grid as P (C tapered)."""
    C = torch.as_tensor(np.asarray(C), dtype=torch.float64, device=device)
    L = len(C) - 1; assert P >= 2 * L
    w = torch.ones(L + 1, device=device, dtype=torch.float64); i0 = 3 * L // 4
    w[i0:] = 0.5 * (1 + torch.cos(math.pi * (torch.arange(L + 1, device=device)[i0:] - i0) / (L - i0)))
    c = torch.zeros(P, dtype=torch.float64, device=device); Ct = C * w
    c[:L + 1] = Ct; c[P - L:] = Ct[1:].flip(0)
    S = torch.fft.fft(c).real.clamp_min(0)
    Z = torch.randn(M, P, device=device, dtype=torch.float64)
    Y = torch.fft.ifft(torch.fft.fft(Z, dim=1) * torch.sqrt(S / P)[None, :], dim=1).real * math.sqrt(P)
    return Y.float()

def ca_phi(model, x):
    return torch.tanh(x) if model == 'ca_tanh' else 0.5 * (1 + torch.tanh(x))

def ca_kernel(k, p, dt, Lk):
    l = torch.arange(Lk + 1, dtype=torch.float64) * dt
    return (k * torch.exp(-l / p) / p * dt)            # w_l (without the C^phi factor), rectangle rule

def pass1_ca(model, N, g, seed, k, p, n_ics=8, T=1500., T_burn=300., dt=0.05, sI=0., sig_xi=0., tau_max=20.,
             device='cuda', J=None, b=None, Lk_time=None):
    """Stored-snapshot pass for the network with Hebbian plasticity (dt_r = dt). Returns means, hbar, C_pop (centered), q, C_phi_unc, Lyapunov,
    and per-site cross-correlations <phi_i(t+tau) eta_i(t)> for tau in [-L, L] (diagnostic response estimator)."""
    torch.manual_seed(seed)
    J = make_J(N, g, seed, device) if J is None else J
    b = make_bias(N, sI, seed, device) if b is None else b
    Lk = int(round((Lk_time if Lk_time else 8 * p) / dt)); L = int(round(tau_max / dt))
    n_burn, n_rec = int(T_burn / dt), int(T / dt)
    x = torch.randn(n_ics, N, device=device)
    ring = torch.zeros(Lk + 1, n_ics, N, device=device); ptr = 0
    wk = ca_kernel(k, p, dt, Lk).float().to(device)                          # [Lk+1]
    Phi = torch.empty(n_ics, N, n_rec, device=device); Eta = torch.empty(n_ics, N, n_rec, device=device)
    x2 = x[:1] + 1e-3 * torch.randn(1, N, device=device); ring2 = torch.zeros(Lk + 1, 1, N, device=device)
    lyap_acc, lyap_n, eps = 0., 0, 1e-3
    idx_all = None
    def selfcoup(phi, ring, ptr):
        idx = (ptr - torch.arange(Lk + 1, device=device)) % (Lk + 1)      # slot l -> phi(t - l dt)
        R = ring[idx]                                                          # [Lk+1, c, N]
        Q = torch.einsum('cn,lcn->cl', phi, R) / N                             # [c, Lk+1]
        return torch.einsum('cl,l,lcn->cn', Q, wk, R)                          # z_i = sum_l w_l Q_l phi_i(t - l dt)
    t0 = time.time()
    fp = (lambda v: 1 - torch.tanh(v) ** 2) if model == 'ca_tanh' else (lambda v: 0.5 * (1 - torch.tanh(v) ** 2))
    stride = max(1, L // 40)
    while L % stride: stride -= 1
    Ls = L // stride; ds = torch.zeros(Ls, n_ics, N, device=device); dring = torch.zeros(Lk + 1, Ls, n_ics, N, device=device)
    acc = torch.zeros(L + 2, N, device=device, dtype=torch.float64); acc32 = torch.zeros(L + 2, N, device=device)
    cnt = torch.zeros(L + 2, device=device, dtype=torch.float64); kslot = torch.full((Ls,), -10 ** 9, device=device, dtype=torch.long)
    ones_s = torch.ones(Ls, device=device, dtype=torch.float64); slot_ptr = 0; ark = torch.arange(Lk + 1, device=device)
    for it in range(n_burn + n_rec):
        phi = ca_phi(model, x); ring[ptr] = phi
        eta = phi @ J.T
        wsh = wk[(ptr - ark) % (Lk + 1)]                                        # shifted weights: contract the raw rings, no gather
        Qr = torch.einsum('cn,lcn->cl', phi, ring) / N; z = torch.einsum('cl,l,lcn->cn', Qr, wsh, ring)
        noise = sig_xi * math.sqrt(dt) * torch.randn(n_ics, N, device=device) if sig_xi > 0 else 0.
        phi2 = ca_phi(model, x2); ring2[ptr] = phi2
        z2 = selfcoup(phi2, ring2, ptr)
        x2 = x2 + dt * (-x2 + phi2 @ J.T + b + z2) + (noise[:1] if sig_xi > 0 else 0.)
        # tangent slots of the unit functional (self-coupling with the network's own Q taken as given): d(dx)/dt = -dx + dh + dz
        dphi = fp(x)[None] * ds; dring[ptr] = dphi
        n = it - n_burn
        if n >= 0:
            Phi[:, :, n] = phi; Eta[:, :, n] = eta
            lag = n - kslot; lag_i = torch.where((lag >= 1) & (lag <= L), lag, torch.full_like(lag, L + 1))
            acc32.index_add_(0, lag_i, dphi.sum(1)); cnt.index_add_(0, lag_i, ones_s)
            if n % 256 == 255: acc += acc32.double(); acc32.zero_()
        if it % stride == 0:
            ds[slot_ptr] = 0; dring[:, slot_ptr] = 0; dh = torch.zeros(Ls, 1, 1, device=device); dh[slot_ptr] = 1.0
            kslot[slot_ptr] = n; slot_ptr = (slot_ptr + 1) % Ls
        else: dh = 0.
        dz = torch.einsum('cl,l,lscn->scn', Qr, wsh, dring)
        ds = ds + dt * (-ds + dh + dz)
        x = x + dt * (-x + eta + b + z) + noise
        if it % 20 == 19:
            d = torch.sqrt(((x2 - x[:1]) ** 2).sum())
            if it >= n_burn // 2: lyap_acc += math.log(d.item() / eps); lyap_n += 20
            x2 = x[:1] + (x2 - x[:1]) * (eps / d); ring2 = ring[:, :1] + (ring2 - ring[:, :1]) * (eps / d)
        ptr = (ptr + 1) % (Lk + 1)
    phibar = Phi.mean((0, 2)); etabar = Eta.mean((0, 2))
    Pc = Phi - phibar[None, :, None]; Ec = Eta - etabar[None, :, None]
    C_pop = autocorr_pop(Pc, L); q = (phibar ** 2).mean().item()
    # cross-correlation <phi_i(t+tau) eta_i(t)>, tau = -L..L, site-resolved (for the empirical response diagnostic)
    Tn = Pc.shape[2]; n = 2 * Tn
    XC = torch.zeros(N, 2 * L + 1, device=device, dtype=torch.float64)
    for c in range(n_ics):
        for s in range(0, N, 256):
            Fp = torch.fft.rfft(Pc[c, s:s + 256], n=n); Fe = torch.fft.rfft(Ec[c, s:s + 256], n=n)
            xc = torch.fft.irfft(Fp * Fe.conj(), n=n)            # index tau >= 0: <phi(t+tau) eta(t)>; negative tau wrap at end
            XC[s:s + 256, L:] += xc[:, :L + 1].double(); XC[s:s + 256, :L] += xc[:, n - L:].double()
    norm = torch.cat([torch.arange(Tn - L, Tn, device=device), torch.arange(Tn, Tn - L - 1, -1, device=device)]).double()
    XC = XC / norm[None, :] / n_ics
    hv = torch.stack([Phi[:, :, :Tn // 2].mean((0, 2)), Phi[:, :, Tn // 2:].mean((0, 2))])
    var_phi = (Pc ** 2).mean().item()
    E256 = torch.einsum('ij,cjt->cit', J[:256], Pc); kurt = ((E256 ** 4).mean() / (E256 ** 2).mean() ** 2 - 3).item()
    acc += acc32.double()
    S_site = (acc[:L + 1] / cnt[:L + 1].clamp_min(1)[:, None] / n_ics / dt).T.contiguous()   # [N, L+1] measured per-site kernels
    out = dict(model=model, N=N, g=g, seed=seed, k=k, p=p, sI=sI, sig_xi=sig_xi, dt=dt, dt_r=dt, tau_max=tau_max, T=T, n_ics=n_ics,
               lyap=lyap_acc / (lyap_n * dt), phibar=phibar, hbar=J @ phibar + b, b=b, J=J, C_pop=C_pop, q=q, var_phi=var_phi,
               phibar_halves=hv, XC_site=XC, eta_kurt=kurt, C_eta_pop=autocorr_pop(E256, L), t_pass1=time.time() - t0, smin=float('inf'),
               S_site=S_site, S_site_stride=stride)
    del Phi, Eta, Pc, Ec; torch.cuda.empty_cache()
    return out

def pass2_ca(model, N, g, seed, k, p, S_kern, n_ics=400, T=2000., T_burn=300., dt=0.05, sI=0., sig_xi=0., tau_max=20., B=None,
             lags_block=(1., 2., 5., 10.), device='cuda', J=None, b=None, Lk_time=None, ac_every=5):
    """Streaming pass for the network with Hebbian plasticity. Residual Delta_i = phi_i - sum_l S_kern[i, l] eta_i(t - l dt) dt (S_kern: [N, L+1])."""
    torch.manual_seed(seed + 7)
    J = make_J(N, g, seed, device) if J is None else J
    b = make_bias(N, sI, seed, device) if b is None else b
    B = min(N, 1000) if B is None else B
    Lk = int(round((Lk_time if Lk_time else 8 * p) / dt)); L = int(round(tau_max / dt)); lag_idx = [int(round(l / dt)) for l in lags_block]
    n_burn, n_rec = int(T_burn / dt), int(T / dt)
    x = torch.randn(n_ics, N, device=device); half = n_ics // 2
    ringk = torch.zeros(Lk + 1, n_ics, N, device=device); pk = 0
    wk = ca_kernel(k, p, dt, Lk).float().to(device)
    S_k = S_kern.float().to(device) * dt                                   # [N, L+1]
    sum_phi = torch.zeros(2, N, device=device, dtype=torch.float64); sum_D = torch.zeros(2, N, device=device, dtype=torch.float64)
    PP = torch.zeros(2, B, B, device=device, dtype=torch.float64); DD = torch.zeros(2, B, B, device=device, dtype=torch.float64)
    PD = torch.zeros(2, B, B, device=device, dtype=torch.float64); PPlag = torch.zeros(len(lag_idx), B, B, device=device, dtype=torch.float64)
    ac_phi = torch.zeros(N, L + 1, device=device, dtype=torch.float64); ac_D = torch.zeros(N, L + 1, device=device, dtype=torch.float64)
    ring_phi = torch.zeros(L + 1, n_ics, N, device=device); ring_D = torch.zeros(L + 1, n_ics, N, device=device); ring_eta = torch.zeros(L + 1, n_ics, N, device=device)
    n_acc = 0; n_ac = 0; ptr = 0; t0 = time.time()
    ar = torch.arange(L + 1, device=device); ark = torch.arange(Lk + 1, device=device)
    for it in range(n_burn + n_rec):
        phi = ca_phi(model, x); ringk[pk] = phi
        eta = phi @ J.T
        wsh = wk[(pk - ark) % (Lk + 1)]                                   # w_shift[j] = wk[(pk - j) mod (Lk+1)]: contract the raw ring, no gather
        Q = torch.einsum('cn,lcn->cl', phi, ringk) / N; z = torch.einsum('cl,l,lcn->cn', Q, wsh, ringk)
        if it >= n_burn:
            ring_eta[ptr] = eta
            D = phi - torch.einsum('nj,jcn->cn', S_k[:, (ptr - ar) % (L + 1)], ring_eta)
            ring_phi[ptr] = phi; ring_D[ptr] = D
            if it - n_burn >= L:
                n_acc += 1
                for hh, sl in enumerate((slice(0, half), slice(half, n_ics))):
                    pp, dd = phi[sl], D[sl]
                    sum_phi[hh] += pp.sum(0).double(); sum_D[hh] += dd.sum(0).double()
                    PP[hh] += (pp[:, :B].T @ pp[:, :B]).double(); DD[hh] += (dd[:, :B].T @ dd[:, :B]).double(); PD[hh] += (pp[:, :B].T @ dd[:, :B]).double()
                if n_acc % ac_every == 0:
                    n_ac += 1; idx = (ptr - ar) % (L + 1)
                    ac_phi += torch.einsum('cn,lcn->nl', phi, ring_phi[idx]).double(); ac_D += torch.einsum('cn,lcn->nl', D, ring_D[idx]).double()
                for qq, li in enumerate(lag_idx):
                    PPlag[qq] += (phi[:, :B].T @ ring_phi[(ptr - li) % (L + 1)][:, :B]).double()
            ptr = (ptr + 1) % (L + 1)
        noise = sig_xi * math.sqrt(dt) * torch.randn(n_ics, N, device=device) if sig_xi > 0 else 0.
        x = x + dt * (-x + eta + b + z) + noise
        pk = (pk + 1) % (Lk + 1)
    nh = n_acc * half
    mphi = sum_phi / nh; mD = sum_D / nh
    cov = lambda Sxy, mx, my: Sxy / nh - mx[:B, None] * my[None, :B]
    Cphi_h = torch.stack([cov(PP[i], mphi[i], mphi[i]) for i in range(2)]); CD_h = torch.stack([cov(DD[i], mD[i], mD[i]) for i in range(2)])
    CPD_h = torch.stack([cov(PD[i], mphi[i], mD[i]) for i in range(2)])
    phibar = sum_phi.sum(0) / (n_acc * n_ics); Dbar = sum_D.sum(0) / (n_acc * n_ics)
    Cphi_lag = PPlag / (n_acc * n_ics) - phibar[:B, None] * phibar[None, :B]
    ac_phi = ac_phi / (n_ac * n_ics) - phibar[:, None] ** 2; ac_D = ac_D / (n_ac * n_ics) - Dbar[:, None] ** 2
    return dict(model=model, N=N, g=g, seed=seed, k=k, p=p, sI=sI, sig_xi=sig_xi, dt=dt, dt_r=dt, tau_max=tau_max, B=B, n_ics=n_ics, T=T,
                T_tot=n_acc * dt * n_ics, alpha=n_acc * dt * n_ics / N, t_pass2=time.time() - t0, phibar=phibar.float(), Dbar=Dbar.float(),
                hbar=(J @ phibar.float() + b), b=b, Cphi=Cphi_h.mean(0), CD=CD_h.mean(0), CPD=CPD_h.mean(0), Cphi_halves=Cphi_h, CD_halves=CD_h,
                Cphi_lag=Cphi_lag, lags_block=list(lags_block), ac_phi=ac_phi, ac_D=ac_D, C_pop=ac_phi.mean(0), CD_pop=ac_D.mean(0), q=(phibar ** 2).mean().item(),
                phibar_ic_halves=mphi.float())     # [2, N] phibar from the two disjoint halves of the initial conditions

def theory_sampled_tt(model, C_pop, g, hgrid, dt, L, k=0., p=1., q=0., lam=0., sig_xi=0., M=512, P=2 ** 15, n_burn=4000, device='cuda',
                      seed=0, Lk_time=None, stride=None):
    """Sampled single-site theory with the two-time tangent estimator of the response.  Same inputs/outputs as theory_sampled.
    Every `stride` steps a unit input impulse (dh = 1 for one step, i.e. an input increment dt) is injected into a fresh tangent slot;
    L/stride slots are propagated with the linearized dynamics and dphi(n) = dt * G(n, k_slot) is accumulated by lag.  S_l = <G(t + l dt, t)>_t
    with no sampling noise beyond the path-to-path variation."""
    torch.manual_seed(seed)
    C_pop = np.asarray(C_pop, dtype=np.float64)
    E = gp_samples(g ** 2 * C_pop, M, P, device); E = E - E.mean(1, keepdim=True)
    hg = torch.as_tensor(np.asarray(hgrid), dtype=torch.float32, device=device); V = len(hg)
    is_glv = model in ('glv', 'glvsat'); is_ca = model in ('ca', 'ca_tanh')
    stride = stride or max(1, L // 40)
    while L % stride: stride -= 1
    Ls = L // stride
    s = torch.full((M, V), 0.5, device=device) if is_glv else torch.zeros(M, V, device=device)
    ds = torch.zeros(Ls, M, V, device=device)
    if is_ca:
        Lk = int(round((Lk_time if Lk_time else 8 * p) / dt)); ring = torch.zeros(Lk + 1, M, V, device=device); dring = torch.zeros(Lk + 1, Ls, M, V, device=device); pk = 0
        Cunc = torch.as_tensor(q + np.interp(np.arange(Lk + 1) * dt, np.arange(len(C_pop)) * dt, C_pop), dtype=torch.float32, device=device)
        wk = ca_kernel(k, p, dt, Lk).float().to(device) * Cunc
        ark = torch.arange(Lk + 1, device=device)
    Ph = torch.empty(M, V, P, device=device)
    fp = (lambda x: 1 - torch.tanh(x) ** 2) if model == 'ca_tanh' else (lambda x: 0.5 * (1 - torch.tanh(x) ** 2))
    acc = torch.zeros(L + 2, V, device=device, dtype=torch.float64); acc32 = torch.zeros(L + 2, V, device=device); cnt = torch.zeros(L + 2, device=device, dtype=torch.float64)
    kslot = torch.full((Ls,), -10 ** 9, device=device, dtype=torch.long); ones = torch.ones(Ls, device=device, dtype=torch.float64); slot_ptr = 0
    for it in range(n_burn + P):
        j = it % P; n = it - n_burn
        h = hg[None, :] + E[:, j:j + 1]
        noise = sig_xi * math.sqrt(dt) * torch.randn(M, V, device=device) if sig_xi > 0 else None
        if is_glv:
            phi = phi_of(model, s); dphi = (ds / (1 + s) ** 2) if model == 'glvsat' else ds
        elif is_ca:
            phi = ca_phi(model, s); dphi = fp(s)[None] * ds
        else:
            phi = phi_of(model, s); dphi = fp(s)[None] * ds
        if n >= 0:
            Ph[:, :, n] = phi
            lag = n - kslot; lag_i = torch.where((lag >= 1) & (lag <= L), lag, torch.full_like(lag, L + 1))
            acc32.index_add_(0, lag_i, dphi.sum(1)); cnt.index_add_(0, lag_i, ones)
            if n % 256 == 255 or n == P - 1: acc += acc32.double(); acc32.zero_()
        if it % stride == 0:                                                # new impulse in the recycled slot
            ds[slot_ptr] = 0
            if is_ca: dring[:, slot_ptr] = 0
            dh = torch.zeros(Ls, 1, 1, device=device); dh[slot_ptr] = 1.0
            kslot[slot_ptr] = n; slot_ptr = (slot_ptr + 1) % Ls
        else:
            dh = 0.
        if is_glv:
            if noise is None:
                ds = ds * (1 + dt * (1 - 2 * s - h))[None] - dt * s[None] * dh
                s = s + dt * (s * (1 - s - h) + lam)
            else:
                fac = torch.exp(dt * (1 - s - h) - noise)
                ds = ds * (fac * (1 - dt * s))[None] - dt * (s * fac)[None] * dh
                s = s * fac + dt * lam
        elif is_ca:
            ring[pk] = phi; dring[pk] = dphi
            wsh = wk[(pk - ark) % (Lk + 1)]
            z = torch.einsum('l,lmv->mv', wsh, ring); dz = torch.einsum('l,lsmv->smv', wsh, dring)
            s = s + dt * (-s + h + z) + (noise if noise is not None else 0.)
            ds = ds + dt * (-ds + dh + dz); pk = (pk + 1) % (Lk + 1)
        else:
            s = s + dt * (-s + h) + (noise if noise is not None else 0.); ds = ds + dt * (-ds + dh)
    m = Ph.mean((0, 2)); Pc = Ph - m[None, :, None]
    Cc = torch.stack([autocorr_pop(Pc[:, v:v + 1, :], L) for v in range(V)])
    S = (acc[:L + 1] / cnt[:L + 1].clamp_min(1)[:, None] / M / dt).T          # [V, L+1]
    out = dict(hgrid=np.asarray(hgrid), m=m.cpu().numpy(), Cc=Cc.cpu().numpy(), S=S.cpu().numpy(), stride=stride)
    del Ph, Pc, E; torch.cuda.empty_cache()
    return out

T_PATH_MIN = 2600.   # minimum length (time units) of the sampled single-site paths: the residual spectrum at low frequency is a small
                     # difference of large terms and is biased low for shorter paths (Lotka-Volterra at dt = 0.01: 4x longer paths moved
                     # the theory-kernel ansatz error from 0.23 to the measured-kernel value 0.20; 8x longer changed nothing further)

def theory_sampled(model, C_pop, g, hgrid, dt, L, V_batch=None, **kw):
    """Sampled single-site theory (two-time tangent estimator), the hbar grid processed in batches of V_batch values
    (memory ~ M * V_batch * P floats; V_batch chosen for ~16 GB per batch unless given).  Returns m, Cc, S on the grid."""
    hgrid = np.asarray(hgrid); parts = []
    if V_batch is None: V_batch = int(max(1, min(24, 4e9 // (kw.get('M', 512) * kw.get('P', 2 ** 15)))))   # ~16 GB of stored paths per batch
    for s0 in range(0, len(hgrid), V_batch):
        parts.append(theory_sampled_tt(model, C_pop, g, hgrid[s0:s0 + V_batch], dt, L, **kw)); torch.cuda.empty_cache()
    out = dict(hgrid=hgrid)
    for k in ('m', 'Cc', 'S'): out[k] = np.concatenate([p[k] for p in parts], 0)
    return out

def theory_sampled_coarse(model, C_pop_r, g, hgrid, dt, dt_r, L_r, **kw):
    """Sampled theory for a model integrated with step dt but recorded on the coarser dt_r grid: C_pop_r (lags 0..L_r on dt_r) is
    interpolated to dt, the theory is run on the dt grid, and m, Cc, S are returned subsampled on the dt_r grid."""
    kk = int(round(dt_r / dt)); L = L_r * kk
    tau_r = np.arange(L_r + 1) * dt_r; tau_f = np.arange(L + 1) * dt
    Cf = np.interp(tau_f, tau_r, np.asarray(C_pop_r, dtype=np.float64))
    th = theory_sampled(model, Cf, g, hgrid, dt, L, **kw)
    return dict(hgrid=th['hgrid'], m=th['m'], Cc=th['Cc'][:, ::kk], S=th['S'][:, ::kk])
