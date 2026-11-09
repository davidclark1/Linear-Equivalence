"""Cavity-variable statistics (Appendix C 7 and Fig. 6 of the paper) for one (N, alpha, seed).

    python cavity_stats.py N alpha seed [T_collect_override]

Model (the SCS model of the paper): (1+d/dt) x_i = eta_i = sum_j J_ij phi_j, phi = f(x),
f(x) = erf(sqrt(pi) x/2), f'(x) = exp(-pi x^2/4), J_ij ~ N(0, g^2/N) iid, J_ii = 0 (nzm.make_J), g = 20,
no external drive. Euler dt = 0.025, burn-in 500 time units, then record every T_SAVE = 0.5 time units.
n_ics independent initial conditions are run in parallel; sampling ratio alpha = n_ics * T_collect / N.

Cavity analogues (Appendix C 7): K = 64 random projections W_in[mu, i], W_out[nu, i] ~ N(0, g^2/N) iid,
independent of J.
    eta_hat_mu(t) = sum_i W_in[mu, i] phi_i(t)              (1+d/dt) x_hat_mu = eta_hat_mu,  phi_hat = f(x_hat)
    F_hat[mu, nu](t) = sqrt(N) sum_i W_in[mu, i] W_out[nu, i] f'(x_i(t))
Reservoir analogues: the same quantities on the first K reservoir sites, with W_in -> J[:K, :], W_out[nu, i] -> J[i, nu]:
    eta_i = sum_j J_ij phi_j,  x_i,  phi_i,  F[mu, nu](t) = sqrt(N) sum_i J[mu, i] J[i, nu] f'(x_i(t)).

All statistics are time averages at fixed J at tau = 0, pooled over the n_ics trajectories:
  A. joint moments <eta_mu^a eta_nu^b> for a + b <= 4 -> joint cumulants kappa_{p,q}(eta_mu (p copies), eta_nu (q copies))
     via the set-partition (moment -> cumulant) formula, standardized by std(eta_mu)^p std(eta_nu)^q; RMS over mu != nu.
  B. Price's theorem: kappa(phi_mu, phi_nu) vs <f'(x_mu)>_t <f'(x_nu)>_t kappa(x_mu, x_nu), mu < nu.
  C. Pearson correlation between F[mu, nu](t) and eta_a(t) eta_b(t), binned by the number of shared indices
     between (mu, nu) and (a, b); off-diagonal (mu != nu, a != b) and diagonal (mu = nu) kernel components.
Sampling floors: all accumulators are kept separately for the two disjoint
halves of the initial conditions (ICs 0..n_ics//2-1 and the rest).  Each half's statistics are computed from that half's data alone
(cumulants of every order, standardization, Price covariances and slopes, F-kernel correlations), and the floor of a plotted RMS is
the RMS, over the same index set, of w (X_half0 - X_half1), w = sqrt(n0 n1)/(n0 + n1) (= 1/2, half the difference as in nzm.py,
for even n_ics).  -> out[side]['floor'] (same keys as the statistics), out[side]['halves'] (per-half values), out[side]['raw_halves'].

Output: $NZM_CAVITY/cav_N{N}_a{alpha}_s{seed}.pkl, default results/cavity_halves (skipped if it exists).
"""
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from nzm import make_J

G = 20.0
K = 64
DT = 0.025
T_BURN = 500.0
T_SAVE = 0.5
T_COLLECT_TARGET = 5000.0   # per-trajectory recording length aimed at when choosing n_ics
N_ICS_MAX = 400
F_CHUNK = 32                # ICs per chunk when forming the (c, K, K) F-kernel tensors
PQ_LIST = [(1, 1), (1, 3), (2, 2)]
MOMENT_PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2)]   # (a, b), a <= b, a + b <= 4
P_MAX = 4
OUT_DIR = os.environ.get('NZM_CAVITY', 'results/cavity_halves')       # output directory
DEVICE = os.environ.get('NZM_DEVICE', 'cuda')                   # 'cpu' only for tests

SQRT_PI_2 = math.sqrt(math.pi) / 2
PI_4 = math.pi / 4


def f(x):
    return torch.erf(SQRT_PI_2 * x)


def fprime(x):
    return torch.exp(-PI_4 * x * x)


# ---------------------------------------------------------------- moments -> cumulants
def _set_partitions(items):
    items = list(items)
    if len(items) == 1:
        yield [tuple(items)]
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for i, block in enumerate(part):
            yield part[:i] + [block + (first,)] + part[i + 1:]
        yield [(first,)] + part


_FORMULA = {}


def cumulant_formula(p, q):
    """kappa_{p,q} = sum over set partitions pi of the p+q slots of (-1)^(|pi|-1) (|pi|-1)! prod_B E[X^a_B Y^b_B].
    Returns [(coefficient, sorted tuple of (a_B, b_B))] grouped by block-size signature."""
    if (p, q) not in _FORMULA:
        types = ['X'] * p + ['Y'] * q
        terms = defaultdict(int)
        for part in _set_partitions(range(p + q)):
            m = len(part)
            sig = tuple(sorted((sum(types[i] == 'X' for i in b), sum(types[i] == 'Y' for i in b)) for b in part))
            terms[sig] += (-1) ** (m - 1) * math.factorial(m - 1)
        _FORMULA[(p, q)] = [(c, s) for s, c in terms.items() if c != 0]
    return _FORMULA[(p, q)]


def joint_cumulant(p, q, means, joint):
    """kappa_{p,q}[i, j] with X = eta_i (p copies), Y = eta_j (q copies).
    means[k]: (K,) = <eta^k>;  joint[(a, b)]: (K, K) = <eta_i^a eta_j^b> for a <= b (M_{b,a} = M_{a,b}^T)."""
    def get(a, b):
        if a == 0 and b == 0:
            return 1.0
        if a == 0:
            return means[b][None, :]
        if b == 0:
            return means[a][:, None]
        return joint[(a, b)] if (a, b) in joint else joint[(b, a)].T
    dim = means[1].shape[0]
    out = np.zeros((dim, dim))
    for coeff, sig in cumulant_formula(p, q):
        prod = 1.0
        for (a, b) in sig:
            prod = prod * get(a, b)
        out += coeff * prod
    return out


# ---------------------------------------------------------------- side accumulators
class Side:
    """Streaming accumulators for one side (cavity or reservoir): K field variables e, preactivations xs,
    activities ps, gains fp, and a (c, K, K) kernel tensor F per record."""

    def __init__(self, K, device):
        f64 = dict(device=device, dtype=torch.float64)
        self.means = {k: torch.zeros(K, **f64) for k in range(1, P_MAX + 1)}
        self.M = {ab: torch.zeros(K, K, **f64) for ab in MOMENT_PAIRS}
        self.x1 = torch.zeros(K, **f64); self.xx = torch.zeros(K, K, **f64)
        self.p1 = torch.zeros(K, **f64); self.pp = torch.zeros(K, K, **f64)
        self.fp1 = torch.zeros(K, **f64)
        self.F1 = torch.zeros(K, K, **f64); self.F2 = torch.zeros(K, K, **f64)
        self.Fe = torch.zeros(K * K, K, **f64)          # <F[mu,nu] eta_a>
        self.Fee = torch.zeros(K * K, K * K, **f64)     # <F[mu,nu] eta_a eta_b>
        self.K = K

    def add(self, e, xs, ps, fp, F):
        """e, xs, ps, fp: (c, K) float64; F: (c, K, K) float64 (all for the same record)."""
        K = self.K
        pw = [None, e]
        for k in range(2, P_MAX + 1):
            pw.append(pw[-1] * e)
        for k in range(1, P_MAX + 1):
            self.means[k] += pw[k].sum(0)
        for (a, b) in MOMENT_PAIRS:
            self.M[(a, b)].addmm_(pw[a].T, pw[b])
        self.x1 += xs.sum(0); self.xx.addmm_(xs.T, xs)
        self.p1 += ps.sum(0); self.pp.addmm_(ps.T, ps)
        self.fp1 += fp.sum(0)
        self.F1 += F.sum(0); self.F2 += (F * F).sum(0)
        Ff = F.reshape(-1, K * K)
        self.Fe.addmm_(Ff.T, e)
        ee = (e[:, :, None] * e[:, None, :]).reshape(-1, K * K)
        self.Fee.addmm_(Ff.T, ee)

    def finalize(self, n):
        K = self.K
        cpu = lambda t: (t / n).cpu().numpy()
        return dict(means={k: cpu(v) for k, v in self.means.items()}, M={ab: cpu(v) for ab, v in self.M.items()},
                    x1=cpu(self.x1), xx=cpu(self.xx), p1=cpu(self.p1), pp=cpu(self.pp), fp1=cpu(self.fp1),
                    F1=cpu(self.F1), F2=cpu(self.F2), Fe=cpu(self.Fe).reshape(K, K, K),
                    Fee=cpu(self.Fee).reshape(K, K, K, K))


def kernel_tensor(A_in, A_out, D, sqrtN):
    """F[c, mu, nu] = sqrtN * sum_i A_in[mu, i] D[c, i] A_out[nu, i];  A_in, A_out: (K, N), D: (c, N)  (float64)."""
    return sqrtN * ((A_in[None, :, :] * D[:, None, :]) @ A_out.T)


# ---------------------------------------------------------------- post-processing (one side)
def side_arrays(raw, K):
    """The per-index arrays behind every plotted statistic of one side (from one set of accumulated averages)."""
    means, M = raw['means'], raw['M']
    k11 = joint_cumulant(1, 1, means, M)
    sig = np.sqrt(np.clip(np.diag(k11), 1e-30, None))
    kappa = {(p, q): joint_cumulant(p, q, means, M) / (sig[:, None] ** p * sig[None, :] ** q) for (p, q) in PQ_LIST}
    iu, ju = np.triu_indices(K, k=1)
    cov_x = raw['xx'] - np.outer(raw['x1'], raw['x1'])
    cov_p = raw['pp'] - np.outer(raw['p1'], raw['p1'])
    fp = raw['fp1']
    var_x = np.clip(np.diag(cov_x), 1e-30, None)
    gain_gauss = 1 / np.sqrt(1 + np.pi * var_x / 2)      # <f'(x)> for x ~ N(0, var_x)
    price = dict(empirical=cov_p[iu, ju], predicted=(np.outer(fp, fp) * cov_x)[iu, ju],
                 predicted_gauss=(np.outer(gain_gauss, gain_gauss) * cov_x)[iu, ju],
                 fprime_mean=fp, gain_gauss=gain_gauss, var_x=var_x)
    slope = lambda xs, ys: float((xs * ys).sum() / (xs * xs).sum())
    price['slope'] = slope(price['empirical'], price['predicted'])              # y = pred on x = empirical, through origin
    price['slope_gauss'] = slope(price['empirical'], price['predicted_gauss'])
    F1, F2, Fee = raw['F1'], raw['F2'], raw['Fee']
    M11, M22 = M[(1, 1)], M[(2, 2)]
    sigF = np.sqrt(np.clip(F2 - F1 * F1, 1e-30, None))
    sig_ab = np.sqrt(np.clip(M22 - M11 ** 2, 1e-30, None))          # std of eta_a eta_b (a = b: std of eta_a^2)
    corr = (Fee - F1[:, :, None, None] * M11[None, None, :, :]) / (sigF[:, :, None, None] * sig_ab[None, None, :, :])
    iK = np.arange(K)
    corr_d = corr[iK, iK]                                              # (K, K, K) [mu, a, b], F[mu, mu]
    Fe = raw['Fe']
    corr1 = (Fe - F1[:, :, None] * means[1][None, None, :]) / (sigF[:, :, None] * sig[None, None, :])
    return dict(sig=sig, k11=k11, kappa=kappa, price=price, corr=corr, corr_d=corr_d, corr1=corr1, sigF=sigF)


def index_sets(K):
    """Index sets of the RMS statistics: off-diagonal (mu != nu) pairs; F-kernel overlap classes."""
    iK = np.arange(K)
    ne = iK[:, None] != iK[None, :]
    mu = iK[:, None, None, None]; nu = iK[None, :, None, None]; aa = iK[None, None, :, None]; bb = iK[None, None, None, :]
    overlap = ((mu == aa) | (mu == bb)).astype(int) + ((nu == aa) | (nu == bb)).astype(int)
    valid = ne[:, :, None, None] & ne[None, None, :, :]                # mu != nu and a != b
    ovl_d = (iK[:, None, None] == iK[None, :, None]).astype(int) + (iK[:, None, None] == iK[None, None, :]).astype(int)
    off = {c: (overlap == c) & valid for c in (0, 1, 2)}
    diag = {c: (ovl_d == c) & ne[None, :, :] for c in (0, 1)}
    diag[2] = (iK[:, None, None] == iK[None, :, None]) & (iK[:, None, None] == iK[None, None, :])   # a = b = mu
    return dict(pairs=ne, off=off, diag=diag)


def analyse_side(raw, K, A=None):
    A = side_arrays(raw, K) if A is None else A
    I = index_sets(K)
    means, M = raw['means'], raw['M']
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    out = {}
    # A. standardized joint cumulants, RMS over mu != nu
    sig = A['sig']
    out['sigma_eta'] = sig
    out['kappa_std'] = {pq: A['kappa'][pq] for pq in PQ_LIST}
    out['kappa_rms'] = {pq: rms(A['kappa'][pq][I['pairs']]) for pq in PQ_LIST}
    # B. Price's theorem (mu < nu)
    out['price'] = A['price']
    # C. Pearson correlation between F[mu,nu](t) and eta_a(t) eta_b(t), by overlap class
    out['prop2_off'] = {c: rms(A['corr'][I['off'][c]]) for c in (0, 1, 2)}
    out['prop2_diag'] = {c: rms(A['corr_d'][I['diag'][c]]) for c in (0, 1, 2)}
    # kappa(F[mu,nu], eta_a) standardized (the F-kernel cumulant bound of Appendix C with p+q = 1), RMS over all (mu, nu, a) and by overlap
    out['prop2_Fe_rms'] = rms(A['corr1'])
    # diagnostics
    dg = np.eye(K, dtype=bool); sigF = A['sigF']
    out['F_mean_abs'] = float(np.abs(raw['F1']).mean())
    out['F_std_diag'] = float(np.sqrt(np.mean(sigF[dg] ** 2)))
    out['F_std_off'] = float(np.sqrt(np.mean(sigF[~dg] ** 2)))
    out['eta_std_mean'] = float(sig.mean())
    out['eta_mean_abs'] = float(np.abs(means[1]).mean())
    out['eta_kurtosis_excess'] = float(np.mean(np.diag(joint_cumulant(2, 2, means, M)) / sig ** 4))
    return out


def side_floors(A0, A1, n0, n1, K):
    """Sampling floors from the two disjoint halves of the initial conditions (n0, n1 trajectories), each statistic estimated
    from its half's own data (cumulants, standardization and correlations included).  Floor of an array = RMS over the SAME index
    set as the plotted RMS of w (X_0 - X_1), w = sqrt(n0 n1) / (n0 + n1): the standard error of the pooled estimate; w = 1/2
    (half the difference, as in nzm.py) when n0 = n1 (n_ics even).  Scalars (Price slopes): w |X_0 - X_1|."""
    I = index_sets(K); w = math.sqrt(n0 * n1) / (n0 + n1)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    P0, P1 = A0['price'], A1['price']
    fl = dict(w=w, n_ics_halves=(n0, n1),
              kappa_rms={pq: rms(w * (A0['kappa'][pq] - A1['kappa'][pq])[I['pairs']]) for pq in PQ_LIST},
              prop2_off={c: rms(w * (A0['corr'] - A1['corr'])[I['off'][c]]) for c in (0, 1, 2)},
              prop2_diag={c: rms(w * (A0['corr_d'] - A1['corr_d'])[I['diag'][c]]) for c in (0, 1, 2)},
              prop2_Fe_rms=rms(w * (A0['corr1'] - A1['corr1'])),
              # Price's theorem, mu < nu: floors of the empirical covariance, the prediction and their difference; slopes
              price_emp_rms=rms(w * (P0['empirical'] - P1['empirical'])),
              price_pred_rms=rms(w * (P0['predicted'] - P1['predicted'])),
              price_diff_rms=rms(w * ((P0['predicted'] - P0['empirical']) - (P1['predicted'] - P1['empirical']))),
              price_slope=w * abs(P0['slope'] - P1['slope']), price_slope_gauss=w * abs(P0['slope_gauss'] - P1['slope_gauss']))
    halves = [dict(kappa_rms={pq: rms(A['kappa'][pq][I['pairs']]) for pq in PQ_LIST},
                   prop2_off={c: rms(A['corr'][I['off'][c]]) for c in (0, 1, 2)},
                   prop2_diag={c: rms(A['corr_d'][I['diag'][c]]) for c in (0, 1, 2)},
                   price_slope=A['price']['slope']) for A in (A0, A1)]
    return fl, halves


def pool_raw(r0, r1, n0, n1):
    """Sample-weighted mean of two finalized accumulator dicts (= the pooled average over all trajectories)."""
    if isinstance(r0, dict):
        return {k: pool_raw(r0[k], r1[k], n0, n1) for k in r0}
    return (n0 * r0 + n1 * r1) / (n0 + n1)


# ---------------------------------------------------------------- main simulation
def run(N, alpha, seed, T_collect_override=None, device='cuda', out_path=None):
    T_tot = alpha * N
    if T_collect_override is None:
        n_ics = int(min(N_ICS_MAX, max(1, round(T_tot / T_COLLECT_TARGET))))
        n_save = int(round(T_tot / (n_ics * T_SAVE)))
    else:
        n_ics = int(min(N_ICS_MAX, max(1, round(T_tot / T_COLLECT_TARGET))))
        n_save = int(round(T_collect_override / T_SAVE))
    T_collect = n_save * T_SAVE
    alpha_eff = n_ics * T_collect / N
    n_burn = round(T_BURN / DT)
    save_iter = round(T_SAVE / DT)
    n_steps = n_burn + (n_save - 1) * save_iter + 1
    sqrtN = math.sqrt(N)
    print(f'N={N} alpha={alpha} seed={seed}: n_ics={n_ics}, T_collect={T_collect:.1f}, n_save={n_save}, '
          f'alpha_eff={alpha_eff:.1f}, Euler steps={n_steps}', flush=True)

    assert n_ics >= 2, 'two halves of the initial conditions need n_ics >= 2'
    halves = ((0, n_ics // 2), (n_ics // 2, n_ics))                   # disjoint IC halves (sizes differ by 1 if n_ics is odd)
    cuda = str(device).startswith('cuda')
    if cuda: torch.cuda.reset_peak_memory_stats()
    J = make_J(N, G, seed, device)                                  # float32, J_ii = 0
    gW1 = torch.Generator(device=device).manual_seed(seed + 1_000_003)
    gW2 = torch.Generator(device=device).manual_seed(seed + 2_000_003)
    gx = torch.Generator(device=device).manual_seed(seed + 3_000_003)
    W_in = torch.randn(K, N, generator=gW1, device=device) * (G / sqrtN)
    W_out = torch.randn(K, N, generator=gW2, device=device) * (G / sqrtN)
    x = torch.randn(n_ics, N, generator=gx, device=device) * G
    x_hat = torch.zeros(n_ics, K, device=device)
    eta = torch.empty(n_ics, N, device=device)
    JT = J.t().contiguous()
    W_inT = W_in.t().contiguous()
    # float64 projection matrices for the kernel tensors
    Win64, Wout64 = W_in.double(), W_out.double()
    Jrow64 = J[:K, :].double().contiguous()            # "W_in" of the reservoir: J[mu, i]
    Jcol64 = J[:, :K].t().double().contiguous()        # "W_out" of the reservoir: J[i, nu] indexed [nu, i]

    cav = [Side(K, device), Side(K, device)]; res = [Side(K, device), Side(K, device)]    # one accumulator set per IC half
    count = 0
    t0 = time.time()
    with torch.no_grad():
        for step in range(n_steps):
            phi = f(x)
            torch.mm(phi, JT, out=eta)                                  # eta_i = sum_j J_ij phi_j
            eta_hat = phi @ W_inT                                       # eta_hat_mu = sum_i W_in[mu,i] phi_i
            if step >= n_burn and (step - n_burn) % save_iter == 0:
                D = fprime(x).double()                                  # (n_ics, N)
                for hh, (lo, hi) in enumerate(halves):                # chunks never straddle the two halves
                    for c0 in range(lo, hi, F_CHUNK):
                        sl = slice(c0, min(c0 + F_CHUNK, hi))
                        xh = x_hat[sl].double()
                        cav[hh].add(eta_hat[sl].double(), xh, f(xh), fprime(xh), kernel_tensor(Win64, Wout64, D[sl], sqrtN))
                        xs = x[sl, :K].double()
                        res[hh].add(eta[sl, :K].double(), xs, phi[sl, :K].double(), D[sl, :K],
                                    kernel_tensor(Jrow64, Jcol64, D[sl], sqrtN))
                del D
                count += 1
                if count in (1, 10) or count % max(1, n_save // 10) == 0:
                    el = time.time() - t0
                    print(f'  record {count}/{n_save}  {el:.0f}s  ETA {el / step * (n_steps - step):.0f}s', flush=True)
            x.mul_(1 - DT).add_(eta, alpha=DT)
            x_hat.mul_(1 - DT).add_(eta_hat, alpha=DT)
            del phi, eta_hat
    if cuda: torch.cuda.synchronize()
    sim_time = time.time() - t0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    n_samp = float(count * n_ics)
    nh = [float(count * (hi - lo)) for lo, hi in halves]
    print(f'simulation: {count} records x {n_ics} ICs = {int(n_samp)} samples, {sim_time:.1f}s, peak GPU mem {mem_gb:.2f} GB',
          flush=True)

    raw_h = [dict(cav=cav[i].finalize(nh[i]), res=res[i].finalize(nh[i])) for i in range(2)]
    raw = pool_raw(raw_h[0], raw_h[1], nh[0], nh[1])                  # pooled averages over all trajectories
    out = dict(N=N, alpha=alpha, seed=seed, g=G, K=K, dt=DT, T_burn=T_BURN, T_save=T_SAVE, n_ics=n_ics,
               n_save=count, T_collect=T_collect, alpha_eff=alpha_eff, n_samples=n_samp, sim_time=sim_time,
               peak_mem_gb=mem_gb, device=torch.cuda.get_device_name(0) if cuda else 'cpu', n_ics_halves=[hi - lo for lo, hi in halves])
    keep = ('means', 'M', 'x1', 'xx', 'p1', 'pp', 'fp1', 'F1', 'F2')
    for side in ('cav', 'res'):
        out[side] = analyse_side(raw[side], K)
        r = raw[side]
        out[side]['raw'] = dict(means=r['means'], M=r['M'], x1=r['x1'], xx=r['xx'], p1=r['p1'], pp=r['pp'],
                                fp1=r['fp1'], F1=r['F1'], F2=r['F2'], Fe=r['Fe'])   # Fee (K^4) not stored
        # sampling floors from the two IC halves (each half's statistics from its own data; see side_floors)
        A0, A1 = side_arrays(raw_h[0][side], K), side_arrays(raw_h[1][side], K)
        out[side]['floor'], out[side]['halves'] = side_floors(A0, A1, *out['n_ics_halves'], K)
        out[side]['raw_halves'] = [{k: raw_h[i][side][k] for k in keep} for i in range(2)]   # small accumulators of each half
        del A0, A1
    for side in ('cav', 'res'):
        a = out[side]
        fmt = lambda d: ' '.join('%.3e' % d[c] for c in (0, 1, 2))
        print('%s: kappa RMS %s | Price slope %.3f (gauss %.3f) | prop2 off [%s] diag [%s] | std eta %.3f, std F diag/off %.3f/%.3f'
              % (side, ', '.join('%s=%.3e' % (pq, a['kappa_rms'][pq]) for pq in PQ_LIST), a['price']['slope'],
                 a['price']['slope_gauss'], fmt(a['prop2_off']), fmt(a['prop2_diag']), a['eta_std_mean'],
                 a['F_std_diag'], a['F_std_off']), flush=True)
        fl = a['floor']
        print('%s floors: kappa %s | Price slope %.1e, emp %.1e, pred-emp %.1e | prop2 off [%s] diag [%s]'
              % (side, ', '.join('%s=%.3e' % (pq, fl['kappa_rms'][pq]) for pq in PQ_LIST), fl['price_slope'], fl['price_emp_rms'],
                 fl['price_diff_rms'], fmt(fl['prop2_off']), fmt(fl['prop2_diag'])), flush=True)
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path + '.tmp', 'wb') as fh:        # atomic: a requeued task never sees a partial file
            pickle.dump(out, fh)
        os.replace(out_path + '.tmp', out_path)
        print(f'saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)', flush=True)
    return out


if __name__ == '__main__':
    N, alpha, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    if len(sys.argv) > 4:        # quick test: fixed per-trajectory recording length, output to results/cavity_test
        T_over = float(sys.argv[4])
        path = os.path.join('results/cavity_test', f'cav_N{N}_a{alpha}_s{seed}_T{int(T_over)}.pkl')
        run(N, alpha, seed, T_collect_override=T_over, out_path=path, device=DEVICE)
    else:
        path = os.path.join(OUT_DIR, f'cav_N{N}_a{alpha}_s{seed}.pkl')
        if os.path.exists(path):
            print(f'{path} exists, skipping')
            sys.exit(0)
        run(N, alpha, seed, out_path=path, device=DEVICE)
