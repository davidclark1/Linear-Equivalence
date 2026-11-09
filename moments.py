"""Two-site moments of odd powers, C^(p,q)_ij = <phi_i^p phi_j^q>_t (Appendix F of the paper, Eq. (F11)), for LINEAR units with
sign-flip symmetry (erf model, hbar = 0, no drive).  Prediction (off-diagonal, equal time):

    Cbar^(p,q)(w) = C^D(w) S^(p) J M M^+ J^T S^(q)* + C^{D1 Dq}(w) S^(p) J M + C^{Dp D1}(w) M^+ J^T S^(q)* + C^{Dp Dq}(w) I,
    M = (I - S J)^-1,  S(w) = beta_star K(w),  S^(a)(w) = beta_a K(w),  beta_a = <d f^a/dx> = a <f^(a-1) f'>,
    C^{Da Db}(w) = C^(a,b)(w) - g^2 beta_a beta_b |K(w)|^2 C_star(w),   C^(a,b)(tau) = <f(x(t+tau))^a f(x(t))^b>  (single site).

With J M = (M - I)/S this is, off the diagonal,  a(w) (M-I)(M-I)^+ + b(w) (M-I) + c(w) (M-I)^+  with the real coefficients
    a = C^D beta_p beta_q / beta_star^2,   b = C^{D1 Dq} beta_p / beta_star,   c = C^{Dp D1} beta_q / beta_star.
For p = q = 1 it is the covariance ansatz C^D M M^+."""
import math
import numpy as np
import torch
from nzm import f_np, filt_x_discrete, spectrum_even, Kd


def power_theory(C_star, g, dt_r, omega, powers=(1, 3), model='erf', dt_filter=None, dx=0.02):
    """Single-site quantities for x ~ GP(0, C^x), C^x from eta ~ GP(0, g^2 C_star) through the Euler filter.
    Returns dict(beta={a: float}, C={(a, b): [L+1]}, CDD={(a, b): [n_w] torch float64}) for a, b in powers."""
    C_star = np.asarray(C_star, dtype=np.float64)
    Cx = filt_x_discrete(C_star, g, dt_r, dt_filter=dt_filter); v0 = float(Cx[0]); sd = math.sqrt(v0)
    f, fp = f_np(model)
    X = np.arange(-12 * sd, 12 * sd + dx, dx); nX = len(X); fX = f(X); fpX = fp(X)
    pdf = np.exp(-X ** 2 / (2 * v0)) / math.sqrt(2 * math.pi * v0) * dx
    beta = {a: float(np.sum(pdf * a * fX ** (a - 1) * fpX)) for a in powers}
    nfft = 1 << int(np.ceil(np.log2(2 * nX))); k = np.fft.rfftfreq(nfft, d=dx) * 2 * np.pi
    FT = {b: np.fft.rfft(fX ** b, nfft) for b in powers}
    rho = np.clip(Cx / v0, -0.999999, 0.999999); rho[0] = 1.0
    C = {(a, b): np.zeros(len(Cx)) for a in powers for b in powers}
    for l, r_ in enumerate(rho):
        s2 = v0 * max(1 - r_ ** 2, 0.0)
        for b in powers:
            gX = fX ** b if s2 < 1e-12 else np.fft.irfft(FT[b] * np.exp(-0.5 * s2 * k ** 2), nfft)[:nX]    # E[f(x2)^b | x1] at x1 = X / rho
            g2 = np.interp(r_ * X, X, gX)
            for a in powers: C[(a, b)][l] = np.sum(pdf * fX ** a * g2)
    om = torch.as_tensor(omega, dtype=torch.float64).cpu()
    K2 = Kd(om, dt_filter or dt_r).abs() ** 2; Cs_w = spectrum_even(torch.as_tensor(C_star), dt_r, om)
    CDD = {ab: spectrum_even(torch.as_tensor(C[ab]), dt_r, om) - g ** 2 * beta[ab[0]] * beta[ab[1]] * K2 * Cs_w for ab in C}
    return dict(beta=beta, C=C, CDD=CDD, Cx=Cx)


def ansatz_pq(J, S_w, a_w, b_w, c_w, omega, B, chunk=16):
    """Equal-time [B, B] block of (1/2pi) int dw [a (M-I)(M-I)^+ + b (M-I) + c (M-I)^+],  M = (I - S(w) J)^-1.
    S_w: [n_w] complex (same for all sites); a_w, b_w, c_w: [n_w] real; omega: uniform grid >= 0 (the w -> -w half gives the conjugate)."""
    N = J.shape[0]; dev = J.device
    chunk = max(1, min(chunk, int(5e9 // (N * N * 16))))
    dw = (omega[1] - omega[0]).item()
    Jc = J.to(torch.complex128); I = torch.eye(N, device=dev, dtype=torch.complex128)
    S_w = S_w.to(dev).to(torch.complex128); a_w, b_w, c_w = (t.to(dev).to(torch.complex128) for t in (a_w, b_w, c_w))
    out = torch.zeros(B, B, device=dev, dtype=torch.complex128)
    for s in range(0, len(omega), chunk):
        w = omega[s:s + chunk]; sl = slice(s, s + chunk)
        Mm = torch.linalg.inv(I[None] - S_w[sl, None, None] * Jc[None])[:, :B, :] - I[None, :B, :]      # (M - I)[:B, :]   [nw, B, N]
        X = a_w[sl, None, None] * torch.einsum('wbn,wcn->wbc', Mm, Mm.conj()) + b_w[sl, None, None] * Mm[:, :, :B] \
            + c_w[sl, None, None] * Mm[:, :, :B].conj().transpose(1, 2)
        wt = torch.full((len(w),), dw / math.pi, device=dev, dtype=torch.float64); wt[w == 0] = dw / (2 * math.pi)
        out += torch.einsum('w,wbc->bc', wt.to(torch.complex128), X)
    return out.real


def predict(J, C_star, beta_star, g, dt, dt_r, omega, B, pairs=((1, 1), (3, 1), (3, 3))):
    """Off-diagonal predictions Cbar^(p,q)[:B, :B] at equal time. Returns ({(p, q): tensor}, theory dict)."""
    powers = tuple(sorted({a for pq in pairs for a in pq}))
    th = power_theory(C_star, g, dt_r, omega, powers=powers, dt_filter=dt)
    S_w = beta_star * Kd(omega, dt); out = {}
    for p, q in pairs:
        a = th['CDD'][(1, 1)] * th['beta'][p] * th['beta'][q] / beta_star ** 2
        b = th['CDD'][(1, q)] * th['beta'][p] / beta_star
        c = th['CDD'][(p, 1)] * th['beta'][q] / beta_star
        out[(p, q)] = ansatz_pq(J, S_w, a, b, c, omega, B)
    return out, th
