#!/usr/bin/env python3
"""
Compare specific angular momentum distributions from two MCMC chains.

Overplots two angular_momentum_*_samples.npz files in a single corner plot.
Edit the configuration section below, then run: python compare-j.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner

mpl.rcParams['font.family'] = 'serif'

# ============================================================
# Configuration
# ============================================================

# MCMC
# FILE_1 = 'angular_momentum_blue_samples.npz'
# FILE_2 = 'angular_momentum_red_samples.npz'

# NS
FILE_1 = 'results/angular_momentum_ns_north_0825.npz'
# FILE_2 = 'angular_momentum_ns_no_pressure_south.npz'
FILE_2 = "results/angular_momentum_ns_south_0825.npz"

LABEL_1 = 'North'
LABEL_2 = 'South'
LABEL_NET = 'Net'

# COLOR_1 = 'steelblue'
# COLOR_2 = 'lightsalmon'
# COLOR_NET = 'darkseagreen'

# 10 rmb color
COLOR_1 = '#8BCFDC'
COLOR_2 = '#415882'
COLOR_NET = '#C5C1D8'

# 100 rmb. color 
# COLOR_1 = "#EE9C83"
# COLOR_2 = '#DD5670'
# # COLOR_NET = '#851321'
# COLOR_NET = "#E8ACB3"

SMOOTH_1D = True
SHOW_BEST_FIT = True
BINS = 30
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 15
DPI = 150

TRUTH_LW = 1.0
QUANTILE_LW = 1.0

RESCALE_HIST = True     # independently scale 1D KDE heights
# MCMC
# HIST_SCALE = [
#     (4.0, 3, 1),   # |j|
#     (20, 7, 1),   # θ_j
#     (20, 4, 1.0),   # φ_j
# ]

# NS
HIST_SCALE = [
    (0.3, 1, 1),   # |j|
    (1, 1, 1),   # θ_j
    (0.7, 1, 1),   # φ_j
]

PLOT_NET = False  # if True, compute & overplot j_blue + j_red via independent draws

# PLOT_RANGE = None
## No Net
# PLOT_RANGE = [
#             (2000,11000),
#             (56,86),
#             (90,170)
# ]

# With Net
if PLOT_NET:
    PLOT_RANGE = [
                (0,13000),
                (50,100),
                (100,118)
    ]
else:
    PLOT_RANGE = [
            (2000,11000),
            (56,86),
            (90,170)
]

PLOT_RANGE = None


# ============================================================
# Load data
# ============================================================

d1 = np.load(FILE_1)
d2 = np.load(FILE_2)

samples_1 = np.column_stack([d1['j_mag'], d1['theta_j'], d1['phi_j']])
samples_2 = np.column_stack([d2['j_mag'], d2['theta_j'], d2['phi_j']])

bf_1 = np.percentile(samples_1, 50, axis=0)
bf_2 = np.percentile(samples_2, 50, axis=0)

# ============================================================
# Net angular momentum: j_total = j_blue + j_red (independent)
# ============================================================
# Since blue and red samples are independent, draw random pairs
# from each posterior to build the distribution of j_total.

if PLOT_NET:
    N_NET = 20000
    rng = np.random.default_rng(42)

    def _to_cartesian(samples):
        from streamer_ic import get_axis_unit_vector
        mag = samples[:, 0]
        t = np.radians(samples[:, 1])
        p = np.radians(samples[:, 2])
        x = -mag * np.sin(t) * np.sin(p)
        y = mag * np.cos(t)
        z = mag * np.sin(t) * np.cos(p)
        return np.column_stack([x, y, z])

    def _to_sph(cart):
        mag = np.linalg.norm(cart, axis=1)
        theta = np.degrees(np.arccos(np.clip(cart[:, 1] / mag, -1, 1)))
        phi = np.degrees(np.arctan2(-cart[:, 0], cart[:, 2]))
        phi = np.where(phi < 0, phi + 360, phi)
        return np.column_stack([mag, theta, phi])

    cart_1 = _to_cartesian(samples_1)
    cart_2 = _to_cartesian(samples_2)

    idx_1 = rng.choice(len(cart_1), size=N_NET)
    idx_2 = rng.choice(len(cart_2), size=N_NET)
    cart_net = cart_1[idx_1] + cart_2[idx_2]
    samples_net = _to_sph(cart_net)

    bf_net = np.percentile(samples_net, 50, axis=0)

    for i, name in enumerate(['|j| [AU·km/s]', 'θ_j [deg]', 'φ_j [deg]']):
        q = np.percentile(samples_net[:, i], [16, 50, 84])
        print(f'  {name}: {q[1]:.4g}  (-{q[1]-q[0]:.4g} / +{q[2]-q[1]:.4g})')

for i, name in enumerate(['|j| [AU·km/s]', 'θ_j [deg]', 'φ_j [deg]']):
    q = np.percentile(samples_1[:, i], [16, 50, 84])
    print(f'  {name}: {q[1]:.4g}  (-{q[1]-q[0]:.4g} / +{q[2]-q[1]:.4g})')

for i, name in enumerate(['|j| [AU·km/s]', 'θ_j [deg]', 'φ_j [deg]']):
    q = np.percentile(samples_2[:, i], [16, 50, 84])
    print(f'  {name}: {q[1]:.4g}  (-{q[1]-q[0]:.4g} / +{q[2]-q[1]:.4g})')

# ============================================================
# Auto range from data if PLOT_RANGE not set
# ============================================================

if PLOT_RANGE is None:
    all_data = [samples_1, samples_2]
    if PLOT_NET:
        all_data.append(samples_net)
    all_data = np.vstack(all_data)
    PLOT_RANGE = []
    for i in range(3):
        lo, hi = np.percentile(all_data[:, i], [0.5, 99.5])
        pad = (hi - lo) * 0.05
        PLOT_RANGE.append((lo - pad, hi + pad))

# ============================================================
# Corner plot
# ============================================================

labels = [
    r'$|j|$ [AU$\cdot$km/s]',
    r'$\theta_j$ [$^\circ$]',
    r'$\phi_j$ [$^\circ$]',
]

ndim = 3
figsize = (min(ndim * 2.5, 14), min(ndim * 2.5, 14))

# Plot LABEL_1 first
fig = corner.corner(
    samples_1,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=False,
    title_fmt='.2f',
    title_kwargs={'fontsize': TITLE_FONT_SIZE},
    label_kwargs={'fontsize': LABEL_FONT_SIZE, 'weight': 'bold'},
    smooth=SMOOTH_1D,
    smooth1d=SMOOTH_1D,
    bins=BINS,
    color=COLOR_1,
    hist_kwargs={'linewidth': 3, 'alpha': 0.6},
    fig=plt.figure(figsize=figsize),
    truths=bf_1 if SHOW_BEST_FIT else None,
    truth_color=COLOR_1,
    range=PLOT_RANGE,
    plot_datapoints=False
)

# Overplot LABEL_2
corner.corner(
    samples_2,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=False,
    title_fmt='.2f',
    title_kwargs={'fontsize': TITLE_FONT_SIZE},
    label_kwargs={'fontsize': LABEL_FONT_SIZE, 'weight': 'bold'},
    smooth=SMOOTH_1D,
    smooth1d=SMOOTH_1D,
    bins=BINS,
    color=COLOR_2,
    hist_kwargs={'linewidth': 3, 'alpha': 0.6},
    fig=fig,
    truths=bf_2 if SHOW_BEST_FIT else None,
    truth_color=COLOR_2,
    range=PLOT_RANGE,
    plot_datapoints=False
)

if PLOT_NET:
    corner.corner(
        samples_net,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        title_fmt='.2f',
        title_kwargs={'fontsize': TITLE_FONT_SIZE},
        label_kwargs={'fontsize': LABEL_FONT_SIZE, 'weight': 'bold'},
        smooth=SMOOTH_1D,
        smooth1d=SMOOTH_1D,
        bins=BINS,
        color=COLOR_NET,
        hist_kwargs={'linewidth': 3, 'alpha': 0.6},
        fig=fig,
        truths=bf_net if SHOW_BEST_FIT else None,
        truth_color=COLOR_NET,
        range=PLOT_RANGE,
        plot_datapoints=False
    )

# --- Adjust truth & quantile line widths ---
for ax in fig.axes:
    for line in ax.get_lines():
        npts = len(line.get_xdata())
        if npts == 2:
            if line.get_linestyle() == 'dashed':
                line.set_linewidth(QUANTILE_LW)
            else:
                line.set_linewidth(TRUTH_LW)

# --- Auto-detect 1D histogram axes ---
# 1D axes = those with KDE/hist lines but no collections (2D axes have contours)
hist_axes = [ax for ax in fig.axes
             if [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
             and not ax.collections]

if hist_axes:
    hist_axes[0].set_ylabel('Scaled PDF', weight="bold", fontsize=12)

# --- Independent 1D histogram height scaling ---
if RESCALE_HIST:
    for i, ax in enumerate(hist_axes):
        kde_lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
        scale_factors = HIST_SCALE[i]
        for j, ln in enumerate(kde_lines):
            if j < len(scale_factors):
                ln.set_ydata(ln.get_ydata() * scale_factors[j])
        ax.relim()
        ax.autoscale_view()

# --- Compute median +/- 1-sigma for text annotation ---
q1 = np.percentile(samples_1, [16, 50, 84], axis=0)
q2 = np.percentile(samples_2, [16, 50, 84], axis=0)
if PLOT_NET:
    qn = np.percentile(samples_net, [16, 50, 84], axis=0)

param_names = [r'$|j|$', r'$\theta_j$', r'$\phi_j$']
param_units = [r'AU$\cdot$km/s', 'deg', 'deg']

lines_1 = [f'{LABEL_1}']
lines_2 = [f'{LABEL_2}']
for i in range(3):
    lines_1.append(
        f'{param_names[i]}: ${q1[1,i]:.4g}^{{+{q1[2,i]-q1[1,i]:.2f}}}_{{-{q1[1,i]-q1[0,i]:.2f}}}$ {param_units[i]}')
    lines_2.append(
        f'{param_names[i]}: ${q2[1,i]:.4g}^{{+{q2[2,i]-q2[1,i]:.2f}}}_{{-{q2[1,i]-q2[0,i]:.2f}}}$ {param_units[i]}')

text_1 = '\n'.join(lines_1)
text_2 = '\n'.join(lines_2)

if PLOT_NET:
    lines_net = [f'{LABEL_NET}']
    for i in range(3):
        lines_net.append(
            f'{param_names[i]}: ${qn[1,i]:.5g}^{{+{qn[2,i]-qn[1,i]:.2f}}}_{{-{qn[1,i]-qn[0,i]:.2f}}}$ {param_units[i]}')
    text_net = '\n'.join(lines_net)

# --- Legend on right side of top-left histogram axis ---
ax_topleft = fig.axes[0]
legend_elements = [
    mpl.lines.Line2D([0], [0], color=COLOR_1, lw=2, label=LABEL_1),
    mpl.lines.Line2D([0], [0], color=COLOR_2, lw=2, label=LABEL_2),
]
if PLOT_NET:
    legend_elements.append(
        mpl.lines.Line2D([0], [0], color=COLOR_NET, lw=2, label=LABEL_NET))
ax_topleft.legend(handles=legend_elements, loc='upper left',
                  bbox_to_anchor=(1.02, 1.0), fontsize=11,
                  frameon=True, fancybox=True, framealpha=0.9)

# --- Median +/- 1-sigma text in upper-right empty area ---
fig.text(0.42, 0.82, text_1, fontsize=10, color=COLOR_1, va='top',
         family='monospace', weight="bold",
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=COLOR_1))
fig.text(0.70, 0.82, text_2, fontsize=10, color=COLOR_2, va='top',
         family='monospace', weight="bold",
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=COLOR_2))
if PLOT_NET:
    fig.text(0.70, 0.62, text_net, fontsize=10, color=COLOR_NET, va='center',
             family='monospace', weight="bold",
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=COLOR_NET))


OUTPUT = 'angular_momentum_comparison_blue_red_ns.png'

fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight')
print(f'\nComparison plot saved to {OUTPUT}')
plt.close(fig)
