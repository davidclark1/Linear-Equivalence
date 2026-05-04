# Linear equivalence of nonlinear recurrent neural networks

David G. Clark

Kempner Institute for the Study of Natural and Artificial Intelligence,
Harvard University, Cambridge, MA, USA

**Paper:** [arXiv:2604.23489](https://arxiv.org/abs/2604.23489)

This repository contains the simulation and analysis code for the paper. The two-site cavity method is used to derive an ansatz for the covariance matrix of activity in a nonlinear random recurrent network at large $N$: it takes the same form as that of a linear network with the same couplings, driven by independent effective noise, with mean-field order parameters setting the effective transfer function and the noise spectrum.

## Repository structure

```
sims/
  rnn.py                         # Core RNN simulation + theory prediction
  single_site_dmft.py            # Sompolinsky-Crisanti-Sommers DMFT solver (erf nonlinearity)
  run_combined_sweep.py          # Batch experiment: covariance prediction + residual scaling
  run_combined_sweep.sh          # SLURM submission script
  run_residual_highalpha.py      # High-alpha residual-only sweep (alpha=3200, 12800)
  run_residual_highalpha.sh      # SLURM submission script
  analysis.ipynb                 # Analysis notebook (generates all paper figures)
  test_dmft.py                   # DMFT solver tests
  test_rnn.py                    # Streaming-vs-reference simulation tests
  test_spectral.py               # Spectral resampling and Parseval tests

figs/                            # Generated figures (PDFs)
```

## Reproducing the figures

The `sims/analysis.ipynb` notebook generates the three main numerical figures of the paper:

1. **Time-domain comparison** — `cij_tau_vs_N.pdf`. Cross-covariance $C^\phi_{ij}(\tau)$, theory vs. simulation, across network sizes.
2. **Scaling grid** — `scaling_grid.pdf`. Three side-by-side panels: absolute and relative prediction error of the matrix equation vs. $N$, plus residual cross-covariance scaling vs. $N$.
3. **Single-site autocorrelation** — `autocorrelation_diagnostic.pdf` (appendix). DMFT vs. empirical $C^\phi_{ii}(\tau)$ across network sizes.

### 1. Run the simulations

The two SLURM scripts launch the experiments. **Edit them first** to set your cluster account, partition, environment activation, and output directory. The scripts are self-documented at the top.

```bash
# Edit sims/run_combined_sweep.sh and sims/run_residual_highalpha.sh, then:
sbatch sims/run_combined_sweep.sh         # 42 array tasks × 10 seeds = 420 runs
sbatch sims/run_residual_highalpha.sh     # 70 tasks (1 seed each)
```

By default the scripts write to `./results/combined_sweep/` and `./results/residual_highalpha/`. The combined sweep uses ~10 GB GPU memory at $N=10000$; the high-alpha sweep is comparable.

### 2. Run the analysis notebook

Open `sims/analysis.ipynb` and run all cells. By default the notebook reads from the `./results/` paths above; override via environment variables `CAVITY_COMBINED_DIR` and `CAVITY_HIGHALPHA_DIR` if results are elsewhere.

Figures are written to `./figs/`.

### 3. Tests (optional)

```bash
pytest sims/
```

The test suite covers the DMFT solver (Gaussian averages vs. Monte Carlo), the streaming covariance computation (vs. a full-timeseries reference), and the spectral resampling pipeline (Parseval, convergence).

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for the simulations)
- NumPy, SciPy, Matplotlib
- Jupyter
- pytest (for tests)

## License

MIT License — see [LICENSE](LICENSE).
