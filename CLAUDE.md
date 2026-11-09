# Guide for AI agents (Claude Code, Codex, and others)

This repository reproduces every numerical result of the paper *Linear equivalence of nonlinear recurrent neural
networks* (D. G. Clark, [arXiv:2604.23489](https://arxiv.org/abs/2604.23489)). Read `README.md` first. Equation, section,
figure and table numbers below refer to arXiv version 4.

## Orientation

```bash
pip install -r requirements.txt                 # NumPy, SciPy, PyTorch, Matplotlib; LaTeX is also needed for figures
python paper_figures.py && python cavity_figure.py && python make_tables.py   # all data figures and Table I from data/
(cd schematics && python schematics.py)        # Figs. 1 and 5
python check_figures.py                        # all of the above, checked pixel by pixel against the committed files
python run_one.py pm_a200 0 100                # smallest SCS run (needs a CUDA GPU, ~2 min) -> results/runs/
NZM_DEVICE=cpu python cavity_stats.py 100 200 0 20   # short cavity test on CPU -> results/cavity_test/
```

The figure scripts need no GPU and no PyTorch. They read only `data/` and reproduce the paper's figures exactly.

## Pipeline

```
run_one.py <case> <seed> <N>         ->  $NZM_RUNS/<case>_s<seed>_N<N>.pkl      (one network realization per file)
cavity_stats.py <N> <alpha> <seed>   ->  $NZM_CAVITY/cav_N<N>_a<alpha>_s<seed>.pkl
        |
extract_cache.py, fig2_pairs.py      ->  data/*.json, data/*.pkl                 (small; committed)
        |
paper_figures.py, cavity_figure.py   ->  figs/*.pdf, data/exponents_*.json
make_tables.py                       ->  figs/tab_exponents.tex                  (Table I)
```

The result files of every run used in the paper are archived on Zenodo (doi:10.5281/zenodo.24091317; download commands
in `README.md`), so `data/` can be rebuilt without rerunning the simulations.

`run_one.py` holds the parameters of every case (`CASES`) and dispatches to `run_model.run` (models without
plasticity) or `run_ca.run` (Hebbian plasticity). One call simulates the network, evaluates the single-site theory and the
ansatz, and stores the error statistics. The task files in `slurm/` list every run of the paper; `slurm/run_tasks.sh`
runs them as a Slurm array. Which run set feeds which figure is in `README.md`.

## Paper → code

| Paper | What | Code |
|---|---|---|
| Eqs. (1), (2), (58) | dynamics φ_i = T[h_i], h_i = Σ_j J_ij φ_j + b_i + ξ_i | `nzm.step`, `nzm.phi_of`, `nzm.make_J`, `nzm.make_bias` |
| Eq. (3) | SCS model, φ = erf(√π x/2), (1 + d/dt) x = h (model `'erf'`) | `nzm.phi_of`, `nzm.step` |
| Sec. VII, model list | sigmoid units `'sig'`, saturating Lotka–Volterra `'glvsat'`, Hebbian plasticity `'ca'` | `nzm.phi_of`, `nzm.step`; `nzm.pass1_ca`, `nzm.pass2_ca` |
| Eqs. (12), (13) | single-site DMFT for the SCS model (C⋆, S⋆, β⋆) | `nzm.closure_odd` |
| Sec. VII, App. H | single-site theory at a temporal mean input h̄ (Gaussian integrals for `'sig'`, `'erf'`; sampled Gaussian-process inputs for `'glvsat'`, `'ca'`) | `nzm.theory_sig`, `nzm.theory_sampled` |
| Eqs. (19)–(22) | linear-equivalent ansatz C̄ = C^Δ M M†, M = (I − S J)⁻¹ (site-specific S and C^Δ allowed) | `nzm.ansatz` |
| Eq. (34) | residual Δ_i = φ_i − S⋆ ∗ η_i and its cross covariances | accumulated in `nzm.pass2` (`nzm.pass2_ca`) |
| Eq. (23), App. I | off-diagonal RMS error of the ansatz, sampling floors | `nzm.pass2` (two halves of the initial conditions), `run_model.run` |
| Eq. (63) | TAP-like equation φ̄_i = M⋆(h̄_i), solved by Newton's method | `tap.solve_tap`, `tap.tap_on_grid` |
| Eq. (64) | ansatz for the connected covariance (measured, single-site, TAP-solved) | `run_model.run`, `run_ca.run` (keys `*_emp`, `*_th`, `*_tap`) |
| Eq. (F11) | two-site moments of odd powers ⟨φ_i^p φ_j^q⟩ | `moments.predict`, `moments.power_theory`, `moments.ansatz_pq` |
| App. C 7, Fig. 6 | cavity-variable tests (joint cumulants, Price's theorem, F-kernel decoupling) | `cavity_stats.run` |
| App. I 3 | exponent fit (medians, N ≥ 464, bootstrap s.e.) and floor subtraction in quadrature | `paperstyle.pooled_fit`, `paperstyle.subtract_floor` |
| App. I 9, Table II | simulation parameters | `CASES` in `run_one.py`; constants at the top of `cavity_stats.py` |

## Result keys used by the figures

All are time-averaged statistics at fixed J for one realization, at τ = 0 unless the key ends in `_lags`.

| Key | Meaning |
|---|---|
| `rms_C` | off-diagonal RMS of the simulated covariance C^φ_ij |
| `rms_err_dmft` | off-diagonal RMS error of the ansatz with DMFT inputs (SCS model) |
| `rms_err_emp`, `rms_err_th`, `rms_err_tap` | the same for the nonzero-mean models, ansatz evaluated in the three ways of Sec. VII (measured, single-site, TAP-solved) |
| `rms_CD`, `rms_CD_star` | off-diagonal RMS of the residual cross covariance, with measured site gains or responses, and with β⋆ |
| `rms_noise`, `rms_noise_D`, `rms_noise_D_star` | sampling floors of the corresponding quantities (half the difference of the two halves of the initial conditions) |
| `meaneq`, `floor_meaneq` | RMS over sites of the error of the TAP-like equation, and its sampling floor |
| `beta_star`, `g_eff_star` | β⋆ from the DMFT inside the run, and g_eff = g β⋆ |
| `err_lags`, `C_lags`, `floor_lags` | `rms_err_dmft`, `rms_C`, `rms_noise` at every stored lag |

## Conventions (keep them)

- **Units.** Time is in units of the single-unit time constant. `dt` is the Euler step and `dt_r` the recording interval.
  The sampling ratio is α = (number of initial conditions × recording time) / N.
- **Precision.** Network states are float32; every accumulator, spectrum and theory quantity is float64 (complex128).
- **Seeds.** J and the biases are drawn on the GPU by `nzm.make_J` and `nzm.make_bias` from (N, g, seed). A rerun can
  reproduce a stored run exactly (on an A100, the smallest runs of `pm2_a800`, `scsn15a3200` and the cavity tests do,
  bit for bit), but PyTorch's GPU random numbers for a given seed can differ between GPU models, and a rerun on other
  hardware is then a different realization with the same statistics. Compare such reruns with stored results
  statistically. For the same reason, in `data/scalars_prod_tap.json` the five models with nonzero mean activities share one saved
  coupling matrix per (g, seed, N), and `J_save` reads "exists, DIFFERENT J (not overwritten)" when a run's matrix was
  drawn on a different GPU model than the saved one; the saved file is left as it is.
- **One run, one file.** Result files are written atomically and skipped if they exist. Never overwrite a result to
  "update" it; write a new run set to a new directory.
- **Floors.** A sampling floor is half the difference between the estimates from the two disjoint halves of the initial
  conditions. Plotted and fitted quantities subtract it in quadrature, per realization (`paperstyle.subtract_floor`).
- **Exponents.** Every exponent is `paperstyle.pooled_fit`: least squares of log(median over realizations) against log N,
  N ≥ 464, standard error from 1000 bootstrap resamples of the realizations within each N.

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `NZM_RUNS` | output directory of `run_one.py` | `results/runs` |
| `NZM_CAVITY` | output directory of `cavity_stats.py` | `results/cavity_halves` |
| `NZM_TASKS` | task file for `slurm/run_tasks.sh` | none |
| `NZM_JDIR` | if set to an existing directory, the coupling matrix of each nonzero-mean run is saved there | unset |
| `NZM_DEVICE` | device for `cavity_stats.py` (`cpu` only for tests) | `cuda` |

## GPUs

- `run_one.py` needs a CUDA GPU; its simulation passes run on `cuda`. `cavity_stats.py` also runs on a CPU for small tests.
- The paper's runs used A100 or H100 GPUs with 80 GB of memory. The largest runs (N = 8000–10000) take up to about 10 hours.
- Run one process per GPU (`CUDA_VISIBLE_DEVICES`).

## Changing numerics

Do not change a numerical path that the paper uses without checking that the paper's numbers survive. At minimum:

1. Rerun a few affected runs at small N and check that their scalar results fall within the spread of the stored
   realizations.
2. Rebuild `data/` with `extract_cache.py`, then run `check_figures.py` to compare the figures and Table I with the committed
   versions.

Add new behavior as new options whose defaults keep the old behavior, or as new cases in `run_one.py`.

## Do not

- **Do not hand-edit `data/`.** It is produced only by `extract_cache.py` and `fig2_pairs.py` from result files.
- **Do not commit `results/`.** Result files are large (about 15 GB for the paper) and git-ignored.
- **Do not rename cases or keys** that `extract_cache.py` and the figure scripts read; the paper's run sets use them.
