# Linear equivalence of nonlinear recurrent neural networks

Code for the paper

> D. G. Clark, *Linear equivalence of nonlinear recurrent neural networks*, [arXiv:2604.23489](https://arxiv.org/abs/2604.23489).

The paper studies recurrent networks with activities φ_i(t) = T[h_i](t), inputs h_i = Σ_j J_ij φ_j + b_i + ξ_i, i.i.d.
Gaussian couplings J_ij ~ N(0, g²/N), and a general single-unit functional T. At large N, for a typical realization of
J, the covariance matrix of the activities matches C̄(ω) = C^Δ(ω) M(ω) M(ω)†, with M(ω) = (I − S(ω) J)⁻¹. This is the
covariance of a linear network with the same couplings, driven by independent noise, with the single-unit response S and
the residual power spectrum C^Δ given by a single-site theory. With nonzero mean activities, the same holds for the
connected covariance, with site-specific S and C^Δ.

This repository contains the code for the simulations, the single-site theory, and the linear-equivalent ansatz, together
with the processed data and the scripts that make every figure and table of the paper from them.

## Requirements

- Python 3.11 with the packages in `requirements.txt` (NumPy, SciPy, PyTorch, Matplotlib).
- An NVIDIA GPU with CUDA for the simulations. The paper's runs used GPUs with 80 GB of memory (A100 or H100).
- A LaTeX installation, used for the figure labels.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the figures and tables

The processed data in `data/` are sufficient to make every figure and table of the paper in a few minutes without a GPU.
From the repository root, run

```bash
python paper_figures.py        # Figs. 2-4 and 7-10 -> figs/, fitted exponents -> data/exponents_figs.json
python cavity_figure.py        # Fig. 6 -> figs/appendix_figures.pdf
python make_tables.py          # Table I -> figs/tab_exponents.tex
cd schematics && python schematics.py && cd ..   # Figs. 1 and 5 -> schematics/
```

`python check_figures.py` does all of this and checks that every figure and Table I match the committed versions pixel by
pixel (it also needs `pdftoppm` from poppler-utils).

| Paper | File | Script | Data |
|---|---|---|---|
| Fig. 1 | `schematics/linear_equivalence_schematic.pdf` | `schematics/schematics.py` | `schematics/traces.npz`, `schematics/cov_blocks.npz` |
| Fig. 2 | `figs/cij_tau_vs_N.pdf` | `paper_figures.py cij` | `data/fig2_curves.pkl` |
| Fig. 3 | `figs/scaling_grid.pdf` | `paper_figures.py scaling` | `data/scalars_moments.json` |
| Fig. 4 | `figs/nzm_scaling.pdf` | `paper_figures.py nzm` | `data/scalars_prod_tap.json` |
| Fig. 5 | `schematics/labeling.pdf` | `schematics/schematics.py` | |
| Fig. 6 | `figs/appendix_figures.pdf` | `cavity_figure.py` | `data/cavity.pkl` |
| Fig. 7 | `figs/moments_scaling.pdf` | `paper_figures.py moments` | `data/scalars_moments.json` |
| Fig. 8 | `figs/lags_diagonal.pdf` | `paper_figures.py lags` | `data/lags_moments.json` |
| Fig. 9 | `figs/autocorrelation_diagnostic.pdf` | `paper_figures.py autocov` | `data/moments_curves.pkl` |
| Fig. 10 | `figs/noise_drive.pdf` | `paper_figures.py noise` | `data/scalars_noise.json`, `data/scalars_moments.json` |
| Table I | `figs/tab_exponents.tex` | `make_tables.py` | `data/exponents_figs.json` |

The figures in `figs/` and `schematics/` are the ones in the paper. All exponents are least-squares slopes of the logarithm
of the median over realizations against log N, with standard errors from a bootstrap over realizations (`paperstyle.py`).

## Raw simulation results

The result files of every run used in the paper (about 15 GB) are archived at
[doi:10.5281/zenodo.24091317](https://doi.org/10.5281/zenodo.24091317). To rebuild the processed data from them, download
and unpack the archives at the repository root, then run the reduction and the figure check:

```bash
for f in results_scs_alpha200_3200 results_scs_alpha800 results_nonzero_mean results_noise_driven results_cavity_tests; do
  curl -L -o $f.tar "https://zenodo.org/records/24091317/files/$f.tar?download=1" && tar -xf $f.tar
done
python extract_cache.py && python fig2_pairs.py && python check_figures.py
```

The record also holds SHA-256 checksums of the archives and of every result file (`MANIFEST_archives.sha256`,
`MANIFEST_files.sha256`).

## Rerunning the simulations

Each run simulates one realization of the network at one size and writes one result file. `run_one.py` runs one
case of the paper (`python run_one.py <case> <seed> <N>`), and `cavity_stats.py` runs one cavity test
(`python cavity_stats.py <N> <alpha> <seed>`). The task files in `slurm/` list every run used in the paper, one per line.

| Run set | Task file | Output directory | Runs | GPU-hours |
|---|---|---|---|---|
| SCS model, alpha = 200 and 3200 | `slurm/tasks_scs.txt` | `results/moments` | 105 | 37 |
| SCS model, alpha = 800 | `slurm/tasks_scs_a800.txt` | `results/moments2` | 70 | 29 |
| Models with nonzero mean activities | `slurm/tasks_nonzero_mean.txt` | `results/prod_tap` | 200 | 96 |
| SCS model with white-noise drive | `slurm/tasks_noise.txt` | `results/noise` | 105 | 153 |
| Cavity tests | `slurm/tasks_cavity.txt` | `results/cavity_halves` | 63 | 24 |

The GPU-hours are those of the paper's runs on A100 and H100 GPUs, about 340 in total.

`slurm/run_tasks.sh` runs one line of a task file per array task. Edit its `#SBATCH` lines and environment setup for your
cluster, then submit from the repository root, for example

```bash
NZM_TASKS=slurm/tasks_scs.txt          NZM_RUNS=results/moments  sbatch --array=0-104 slurm/run_tasks.sh
NZM_TASKS=slurm/tasks_scs_a800.txt     NZM_RUNS=results/moments2 sbatch --array=0-69  slurm/run_tasks.sh
NZM_TASKS=slurm/tasks_nonzero_mean.txt NZM_RUNS=results/prod_tap sbatch --array=0-199 slurm/run_tasks.sh
NZM_TASKS=slurm/tasks_noise.txt        NZM_RUNS=results/noise    sbatch --array=0-104 slurm/run_tasks.sh
NZM_TASKS=slurm/tasks_cavity.txt                                 sbatch --array=0-62  slurm/run_tasks.sh
```

`run_one.py` and `cavity_stats.py` skip runs whose result files already exist, so an interrupted array can simply be
resubmitted. The longest runs take up to about 10 hours each on an A100 or H100 GPU, and the result files occupy about
15 GB in total. Once the runs are done, reduce the result files to the processed data, and then remake the figures as above.

```bash
python extract_cache.py        # results/ -> data/ (may need tens of GB of memory)
python fig2_pairs.py           # results/moments2 -> data/fig2_curves.pkl
```

Setting `NZM_JDIR` to an existing directory also saves the coupling matrix of each run with nonzero mean activities.

## Processed data

| File | Contents |
|---|---|
| `data/scalars_moments.json` | SCS model, one entry per run: all scalar results, including the per-lag sampling floors at alpha = 800 |
| `data/lags_moments.json` | SCS model at alpha = 800, one entry per run: errors, signal and sampling floors at every stored lag, and the diagonal elements |
| `data/moments_curves.pkl` | SCS model at alpha = 800, seed 0: site autocovariances (mean and s.d. over sites), the DMFT autocovariance, and lagged covariances of eight random pairs of sites |
| `data/fig2_curves.pkl` | SCS model at alpha = 800, seed 0: lagged covariances of five pairs of sites, simulation and ansatz |
| `data/scalars_prod_tap.json` | models with nonzero mean activities, one entry per run: all scalar results, including the error of the TAP-like equation (`meaneq`) |
| `data/scalars_noise.json` | SCS model with white-noise drive, one entry per run: all scalar results |
| `data/cavity.pkl` | cavity tests, one entry per run: the statistics and sampling floors of Fig. 6 |

The meaning of the keys used by the figures is in `AGENTS.md`.

## Cases

| Case | Model | Paper |
|---|---|---|
| `pm_a200`, `pm2_a800`, `pm_a3200` | SCS model (error-function units, g = 2.5, no drive), alpha = 200, 800, 3200 | Secs. V and VI, Figs. 2, 3, 7-9 |
| `sig_tap`, `sigbias_tap` | recurrent network with sigmoid units, without and with random biases | Sec. VII, Fig. 4 |
| `ca_km2_tap`, `ca_kp2_tap` | recurrent network with Hebbian plasticity, k = -2 and +2 | Sec. VII, Fig. 4 |
| `glvsat_tap` | saturating Lotka-Volterra model | Sec. VII, Fig. 4 |
| `scsn15a3200`, `scsn2a3200`, `scsn25a3200` | SCS model driven by white noise, alpha = 3200 | Appendix I, Fig. 10 |

The parameters of every case are in `run_one.py`. The case `pm2_a800` is `pm_a800` with the lagged covariances also
accumulated separately over the two halves of the initial conditions, which gives sampling floors at every lag. The
paper's results at alpha = 800 use `pm2_a800`.

## Code

| File | Contents |
|---|---|
| `nzm.py` | Network simulation, single-site theory (DMFT), and the linear-equivalent ansatz |
| `run_model.py`, `run_ca.py` | One run of a model without and with Hebbian plasticity: simulation, theory, ansatz, error statistics |
| `tap.py` | Solution of the TAP-like equation for the temporal means |
| `moments.py` | Two-site moments of odd powers (Appendix F) |
| `run_one.py` | The cases of the paper, and one run of a case |
| `cavity_stats.py` | Tests of the cavity properties (Appendix C) |
| `extract_cache.py`, `fig2_pairs.py` | Result files to processed data |
| `paper_figures.py`, `cavity_figure.py`, `make_tables.py`, `noise_numbers.py`, `paperstyle.py` | Figures, Table I, and the shared plotting style and exponent fit |
| `check_figures.py` | Remakes the figures and Table I and compares them with the committed versions |
| `schematics/` | Figs. 1 and 5 |
| `slurm/` | Job script and task files |

## For AI agents

Read `AGENTS.md` (identical to `CLAUDE.md`) first. It describes the pipeline, maps the paper's equations to the code,
explains the result keys, and lists the conventions to keep when changing the code.

## Citation

```bibtex
@misc{clark2026linear,
  title         = {Linear equivalence of nonlinear recurrent neural networks},
  author        = {Clark, David G.},
  year          = {2026},
  eprint        = {2604.23489},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.dis-nn},
  doi           = {10.48550/arXiv.2604.23489}
}
```

## License

MIT (see `LICENSE`).
