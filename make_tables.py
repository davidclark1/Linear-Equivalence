"""Table I of the paper (fitted exponents) from data/exponents_figs.json (written by paper_figures.py) -> figs/tab_exponents.tex.
Entries: least-squares exponent of the medians over realizations +- bootstrap standard error (paperstyle.pooled_fit)."""
import json
E = json.load(open('data/exponents_figs.json'))
def c(key):
    f = E.get(key); return '--' if f is None or f['slope'] != f['slope'] else f'${f["slope"]:.2f} \\pm {f["se"]:.2f}$'
L = [r'\begin{table*}[t]', r'\centering',
     r'\caption{Fitted exponents, with standard errors. Top, SCS model at $g = 2.5$ (Fig.~\ref{fig:scaling}), by sampling ratio $\alpha$. Bottom, models with nonzero mean activities (Fig.~\ref{fig:nzm_scaling}) at $g = 20$ and $\alpha = 200$, with the ansatz evaluated in three ways. The first way uses the site-specific responses and residual power spectra measured in the network (measured), the second computes them from the single-site theory at the measured temporal mean inputs (single-site), and the third computes them from the single-site theory at the temporal mean inputs of the solved TAP-like equation (TAP-solved). For $C^\Delta_{ij}(\tau = 0)$ and $C^{\breve{\Delta}}_{ij}(\tau = 0)$, the sampling floor is subtracted in quadrature.}',
     r'\label{tab:exponents}', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
     r'$\alpha$ & error of the ansatz, $C^\phi_{ij}(\tau = 0) - \bar{C}^\phi_{ij}(\tau = 0)$ & relative error & $C^\Delta_{ij}(\tau = 0)$ with $\beta_\star$ \\',
     r'predicted & $-1$ & $-1/2$ & $\leq -1$ \\', r'\midrule']
for a in (200, 800, 3200):
    L.append(f'${a}$ & {c(f"pm_a{a}/err")} & {c(f"pm_a{a}/rel")} & {c(f"pm_a{a}/resid")} \\\\')
L += [r'\bottomrule', r'\end{tabular}', r'', r'\vspace{1em}', r'', r'\medmuskip=1mu\begin{tabular}{@{}l*{8}{@{\hspace{6.5pt}}c}@{}}', r'\toprule',
      r' & $\bar{\phi}_i - \mathcal{M}_\star(\bar{h}_i)$ & \multicolumn{3}{c}{$C^{\breve{\phi}}_{ij}(\tau = 0) - \bar{C}^{\breve{\phi}}_{ij}(\tau = 0)$} & \multicolumn{3}{c}{relative error} & $C^{\breve{\Delta}}_{ij}(\tau = 0)$ \\',
      r'\cmidrule(lr){3-5} \cmidrule(lr){6-8}', r'Model &  & measured & single-site & TAP-solved & measured & single-site & TAP-solved &  \\',
      r'predicted & $-1/2$ & $-1$ & $-1$ & $-1$ & $-1/2$ & $-1/2$ & $-1/2$ & $\leq -1$ \\', r'\midrule']
for case, name in (('sig', 'Sigmoid'), ('sigbias', 'Sigmoid, biases'), ('ca_km2', 'Hebbian, $k = -2$'), ('ca_kp2', 'Hebbian, $k = +2$'), ('glvsat', 'Lotka--Volterra')):
    L.append(f'{name} & ' + ' & '.join(c(f'{case}/{k}') for k in ('meaneq', 'err_emp', 'err_th', 'err_tap', 'rel_emp', 'rel_th', 'rel_tap', 'resid')) + r' \\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
open('figs/tab_exponents.tex', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
