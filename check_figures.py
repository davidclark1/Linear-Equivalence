"""Remake every figure and Table I from data/ and check them against the committed versions.

    python check_figures.py        (needs LaTeX, and pdftoppm from poppler-utils)

The committed files are copied to a temporary directory first. The figure scripts then overwrite figs/ and schematics/,
and each new PDF is compared pixel by pixel with its committed version (both rendered at 100 dpi); Table I is compared as
text. If anything differs, `git checkout figs schematics` restores the committed versions.  Exit status 1 if any differ."""
import os, shutil, subprocess, sys, tempfile
import numpy as np
import matplotlib.image as mpimg

FIGS = ['figs/' + f + '.pdf' for f in ('cij_tau_vs_N', 'scaling_grid', 'nzm_scaling', 'appendix_figures', 'moments_scaling',
                                     'lags_diagonal', 'autocorrelation_diagnostic', 'noise_drive')] + \
       ['schematics/linear_equivalence_schematic.pdf', 'schematics/labeling.pdf']
TABLE = 'figs/tab_exponents.tex'


def render(pdf, out):
    subprocess.run(['pdftoppm', '-r', '100', '-png', '-singlefile', pdf, out], check=True)
    return mpimg.imread(out + '.png')


ref = tempfile.mkdtemp()
for f in FIGS + [TABLE]:
    os.makedirs(os.path.join(ref, os.path.dirname(f)), exist_ok=True); shutil.copy(f, os.path.join(ref, f))
for cmd, cwd in ((['paper_figures.py'], '.'), (['cavity_figure.py'], '.'), (['make_tables.py'], '.'), (['schematics.py'], 'schematics')):
    print('running', cmd[0], flush=True)
    subprocess.run([sys.executable] + cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)
bad = 0
for f in FIGS:
    a = render(f, os.path.join(ref, 'new')); b = render(os.path.join(ref, f), os.path.join(ref, 'old'))
    same = a.shape == b.shape and np.array_equal(a, b)
    bad += not same; print(f'{f:48s}', 'identical' if same else 'DIFFERS')
same = open(TABLE).read() == open(os.path.join(ref, TABLE)).read()
bad += not same; print(f'{TABLE:48s}', 'identical' if same else 'DIFFERS')
shutil.rmtree(ref)
print('all figures and Table I match the committed versions' if not bad else f'{bad} file(s) differ')
sys.exit(1 if bad else 0)
