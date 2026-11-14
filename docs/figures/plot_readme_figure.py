#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure of the README: QNM Analyser applied to the example spectra of examples/schwarzschild-l1/.

The three files hold every finite root of a spectral discretisation of the massless scalar field with
l = 1 around a Schwarzschild black hole, at the resolutions N = 180, 190 and 200. The files are read by
parse_upload and analysed by compute_converged and classify_converged, taken unchanged from app.py
(without importing Dash), with the default tolerance 1e-4, as the application does. From this folder,
    python3 plot_readme_figure.py
writes qnm-analyser-example.png. The style is the stamped copy manuscript_style.py beside it.
"""
import ast
import base64
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXAMPLE = ROOT / "examples" / "schwarzschild-l1"
sys.path.insert(0, str(HERE))
import manuscript_style as hs  # noqa: E402

# The three functions of the application, executed from its own source.
tool = {"np": np, "_re": re, "base64": base64, "MAX_ROWS_PER_FILE": 100_000}
for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8")).body:
    if isinstance(node, ast.FunctionDef) and node.name in (
            "parse_upload", "compute_converged", "classify_converged"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), tool)

TOL = 1e-4
eigs = {}
for path in sorted(EXAMPLE.glob("eigs_*.dat")):
    url = "data:text/plain;base64," + base64.b64encode(path.read_bytes()).decode()
    re_v, im_v, n = tool["parse_upload"](url, path.name)
    eigs[n] = np.column_stack([re_v, im_v])
resolutions = sorted(eigs)
highest, others = resolutions[-1], resolutions[:-1]
trees = {n: cKDTree(eigs[n]) for n in others}
conv_re, conv_im = tool["compute_converged"](eigs[highest], trees, others, TOL)
keep = conv_re >= -TOL
conv_re, conv_im = conv_re[keep], conv_im[keep]
general, pure_imag, pure_real = tool["classify_converged"](conv_re, conv_im, TOL)

RE_MAX, IM_MIN = 0.33, -1.95
hs.configure(family="latin-modern")
plt.rcParams["savefig.dpi"] = 265
plt.rcParams["figure.facecolor"] = hs.PAPER
plt.rcParams["savefig.facecolor"] = hs.PAPER

fig, (left, right) = plt.subplots(1, 2, figsize=(hs.WIDTH, 2.9), constrained_layout=True,
                                  width_ratios=[1.0, 1.15])
shades = {180: hs.LILAC, 190: hs.TEAL, 200: hs.OCHRE}
shapes = {180: "s", 190: "D", 200: "^"}
for n in resolutions:
    z = eigs[n]
    w = (z[:, 0] >= -0.01) & (z[:, 0] <= RE_MAX) & (z[:, 1] >= IM_MIN)
    left.plot(z[w, 0], z[w, 1], shapes[n], ms=3.0, markerfacecolor="white",
              markeredgecolor=shades[n], markeredgewidth=0.7, linestyle="none")
inside = (conv_re <= RE_MAX) & (conv_im >= IM_MIN)
left.plot(conv_re[inside], conv_im[inside], "o", ms=7.5, markerfacecolor="none",
          markeredgecolor=hs.NAVY, markeredgewidth=1.1, linestyle="none")
left.set_xlim(-0.02, RE_MAX)
left.set_ylim(IM_MIN, 0.05)
left.set_xlabel(r"$\mathrm{Re}\,\Omega$")
left.set_ylabel(r"$\mathrm{Im}\,\Omega$")
hs.panel(left, "(a) roots of the three resolutions")

# The test itself: distance from each root of N = 200 to the nearest root of every lower resolution.
z200 = eigs[highest]
w = (z200[:, 0] >= -TOL) & (z200[:, 0] <= RE_MAX) & (z200[:, 1] >= IM_MIN)
distance = np.max([trees[n].query(z200[w])[0] for n in others], axis=0)
distance = np.maximum(distance, 1e-16)   # exact agreement in double precision drawn at 1e-16
passed = distance <= TOL
on_axis = np.abs(z200[w, 0]) < TOL
right.axhline(TOL, color=hs.SLATE, lw=0.8, linestyle=hs.DASHES[1])
right.semilogy(z200[w, 1][~passed & on_axis], distance[~passed & on_axis], "^", ms=3.0,
               markerfacecolor="white", markeredgecolor=hs.OCHRE, markeredgewidth=0.7, linestyle="none")
right.semilogy(z200[w, 1][passed], distance[passed], "o", ms=4.5, markerfacecolor="white",
               markeredgecolor=hs.NAVY, markeredgewidth=1.0, linestyle="none")
right.set_xlim(IM_MIN, 0.05)
right.set_ylim(4e-17, 2.5)
right.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1.0])
right.set_xlabel(r"$\mathrm{Im}\,\Omega$ of the root at $N = 200$")
right.set_ylabel("distance to the other resolutions")
hs.panel(right, "(b) the test against the tolerance")

handles = [Line2D([], [], marker=shapes[n], color=shades[n], markerfacecolor="white", ms=3.5,
                  linestyle="none") for n in resolutions]
labels = [rf"roots at $N = {n}$" for n in resolutions]
handles += [Line2D([], [], marker="o", color=hs.NAVY, markerfacecolor="white", ms=5.5, linestyle="none"),
            Line2D([], [], color=hs.SLATE, lw=0.8, linestyle=hs.DASHES[1])]
labels += ["converged", r"tolerance $10^{-4}$"]
hs.legend_outside(fig, where="lower", ncol=5, handles=handles, labels=labels)
hs.save(fig, HERE / "qnm-analyser-example.png")

print("written", HERE / "qnm-analyser-example.png")
print("resolutions", resolutions, "converged", len(conv_re), "complex", len(general),
      "imaginary axis", len(pure_imag), "real axis", len(pure_real))
for zr, zi in sorted(general.tolist(), key=lambda p: -p[1]):
    print(f"  {zr:.6f} {zi:+.6f} i")
print("in the window: roots of N = 200", int(w.sum()), "passed", int(passed.sum()))
