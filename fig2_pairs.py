"""Pairs for Fig. 2 of the paper -> data/fig2_curves.pkl.
Runs: results/moments2/pm2_a800_s0_N*.pkl (SCS model, alpha = 800, seed 0; the same runs as data/moments_curves.pkl).
Rule (fixed in advance, independent of the agreement with theory): over the off-diagonal pairs i < j of the recorded 200-site block, the five pairs at evenly
spaced quantiles 0.1, 0.3, 0.5, 0.7, 0.9 of the simulated C_ij(tau = 0). Curves: simulation C_blk and ansatz Cbar_dmft_blk, both lag signs."""
import pickle
import numpy as np

QS = (0.1, 0.3, 0.5, 0.7, 0.9)
out = {}
for N in (464, 1000, 2154, 4642, 10000):
    r = pickle.load(open(f'results/moments2/pm2_a800_s0_N{N}.pkl', 'rb'))
    C, Cb = r['C_blk'], r['Cbar_dmft_blk']; iu = np.triu_indices(C.shape[1], 1); order = np.argsort(C[0][iu], kind='stable')
    ks = [order[int(round(q * (len(order) - 1)))] for q in QS]
    pairs = [(int(iu[0][k]), int(iu[1][k])) for k in ks]
    out[N] = dict(lags=np.asarray(r['lags']), pairs=pairs, quantiles=QS,
                  sim={p: (C[:, p[0], p[1]], C[:, p[1], p[0]]) for p in pairs}, ansatz={p: (Cb[:, p[0], p[1]], Cb[:, p[1], p[0]]) for p in pairs})
    print(N, pairs, np.round([np.sqrt(N) * C[0, i, j] for i, j in pairs], 2))
pickle.dump(out, open('data/fig2_curves.pkl', 'wb'))
