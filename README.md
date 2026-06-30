# qudpy

# qudpy

**Current version: 1.1.0**

`qudpy` is a small Python toolkit, built on top of [QuTiP](https://qutip.org/), for simulating quantum dynamics and computing linear and 2D (nonlinear) spectra of model systems via double-sided Feynman diagrams. It bundles:

- **`qudpy.classes.System`** — defines the physical model (Hamiltonian, density matrix, collapse/dipole operators) and runs the simulations: pulse-sequence evolution, 2D coherence scans, linear spectra, and population-time studies.
- **`qudpy.plot_functions`** — a collection of plotting utilities for visualizing 1D/2D spectra and time-domain signals, from quick diagnostic plots to publication-quality contour figures.

This repository tracks the **current/updated version** of qudpy. This README also documents the main differences from the previous version, for anyone migrating existing analysis scripts.

## Installation

```bash
git clone [<repo-url>](https://github.com/SilvaScience/QuDPy.git)
cd qudpy
pip install -e .
```

or just install the dependencies and import the modules directly:

```bash
pip install -r requirements.txt
```

## Quick start

```python
from qudpy import System
from qudpy import plot_functions as pf

# define a simple harmonic system (default parameters)
sys = System(n=3)

# compute a linear spectrum
dipole, t_list, spec, freq = sys.linear_spec(scan_time=200)

# compute a 2D coherence map for a chosen diagram and plot it
# (see docstrings in qudpy/classes.py for the diagram format)
```

See the docstrings in `qudpy/classes.py` and `qudpy/plot_functions.py` for full parameter documentation of each function.

## What's new in this version

This release focuses on simulation performance/cleanup in `classes.py` and a major expansion of plotting/analysis capability in `plot_functions.py`. The core physics model is unchanged — these are implementation and tooling improvements, not changes to the underlying simulation method.

### `classes.py` (`System` class)

- **Sequential-only `coherence2d`.** The previous version supported `parallel=True` (using QuTiP's `parallel_map`/`parfor` to evolve batches of states across CPU cores during a scan). This has been removed in favor of a simpler, fully sequential implementation. If you relied on multi-core scaling for large 2D scans, note that this capability is no longer wired up in `coherence2d`.
- **New optimized method: `coherence2d_s`.** A rewritten version of `coherence2d` that reduces redundant `mesolve` calls, pre-computes all time-delay arrays once, and flattens the original's nested loops into list comprehensions over trajectories. Same physics/inputs/outputs as `coherence2d`, intended as a faster implementation of the same calculation.
- **FFT convention change in `spectra()`.** Switched from `np.fft.fft2` to `np.fft.ifft2` when transforming dipole response into 2D spectra. This changes the sign/orientation convention of the resulting frequency axes — please verify this matches your expected rephasing/non-rephasing pathway convention before comparing results against older data.

### `plot_functions.py`

Grew from 4 functions to 10, shifting from quick diagnostic plots toward flexible, publication-ready figures:

- **`silva_plot` reworked to take full coordinate arrays (`x_val`, `y_val`)** instead of a 4-number `scan_range`. This enables non-uniform axis handling and a new `plot_quadrant='Zoom'` mode for zooming into an arbitrary rectangular region (paired with the new `Zoom_coor` parameter).
- **New helper `coor(data_x, data_y, z)`** — converts a coordinate window `[xmin, xmax, ymin, ymax]` into array index bounds, used by the new zoom functionality across several plotting functions.
- **Three new contour-based plotting functions**, in increasing order of polish:
  - `plot_contourf_multi_spectra` — `contourf`-based version of `silva_plot`.
  - `plot_contourf_multi_spectra_norm` — adds independent per-panel normalization so datasets of different magnitude are visually comparable.
  - `plot_contourf_multi_spectra_norm_b` — adds separate signed/diverging colormap for real & imaginary parts vs. one-sided colormap for the absolute value, shared per-row colorbars, consistent tick styling, and an optional `save_file=True` argument to export figures directly to PDF.
- **New `antidiagonal_cut` function** — extracts a 1D line-cut along the (anti-)diagonal of a 2D spectrum around a chosen center point, fits it to a **Lorentzian or Gaussian** lineshape via `scipy.optimize.curve_fit`, and returns the fitted linewidth. Useful for quantitatively extracting peak linewidths/dephasing rates from 2D spectra — there was no equivalent tool in the previous version.
- New dependencies introduced: `matplotlib.colors` (`TwoSlopeNorm`, `Normalize`), `matplotlib.cm`, `scipy.optimize.curve_fit`.
- `multiplot`, `plot`, `log_scale`, and `pop_plot` are unchanged from the previous version.

### Summary

| Area | Old version | New version |
|---|---|---|
| 2D coherence scan | Sequential or parallel (`parallel=True`) | Sequential only, plus separate optimized `coherence2d_s` |
| 2D spectra FFT | `np.fft.fft2` | `np.fft.ifft2` (convention flipped — verify before comparing to old results) |
| Plotting backend | `imshow`-based | `imshow` (legacy) **+** new `contourf`-based publication plots |
| Axis specification | Single `[xmin,xmax,ymin,ymax]` range | Full coordinate arrays (`x_val`, `y_val`), enabling arbitrary zoom regions |
| Linewidth extraction | Not available | `antidiagonal_cut` (Lorentzian/Gaussian fitting) |
| Figure export | Not available | PDF export via `plot_contourf_multi_spectra_norm_b(save_file=True)` |

## Repository structure

```
qudpy/
├── qudpy/
│   ├── __init__.py
│   ├── classes.py          # System class: model, simulation, diagrams
│   └── plot_functions.py   # plotting & spectral analysis utilities
├── requirements.txt
├── setup.py
└── README.md
```

## Notes / known caveats

- The `parallel` argument is still present in `coherence2d`'s signature for backward compatibility but is currently a no-op — it does not parallelize the computation in this version.
- The `ifft2` vs `fft2` change in `spectra()` is a deliberate convention change but should be double-checked against your expected sign convention for rephasing/non-rephasing pathways before trusting absolute peak positions in 2D spectra.
- `coherence2d_s` is an alternate, performance-oriented implementation kept alongside the original `coherence2d`; the two should produce equivalent results, but `coherence2d_s` has not been as extensively used/tested.
