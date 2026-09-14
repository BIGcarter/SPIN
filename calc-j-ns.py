#!/usr/bin/env python3
"""
Compute specific angular momentum distribution from nested sampling results.

For each posterior sample, computes the specific angular momentum vector
j = r x v at the initial condition, then derives its magnitude |j|
and direction (theta_j, phi_j) in the model's spherical convention.

Outputs:
  - angular_momentum_ns_<suffix>.npz
  - angular_momentum_ns_<suffix>_corner.png
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner

from streamer_ic import get_streamer_initial_state, cartesian_to_spherical

mpl.rcParams['font.family'] = 'serif'

# ============================================================
# Configuration
# ============================================================

# SAMPLES_FILE = 'ns_samples_no_pressure_north.npz'
# SAMPLES_FILE = 'ns_samples_no_pressure_south.npz'
# SAMPLES_FILE = 'ns_samples_no_pressure_M_26south_codex_v2.npz'
SAMPLES_FILE = 'results/ns_samples_south_0825_v3.npz'
# SAMPLES_FILE = 'ns_samples_no_pressure_M_26_north.npz'
SAVE_SUFFIX = '_south_0825'

N_SUBSAMPLE = 0           # 0 = use all; >0 = random subsample (for speed)

SMOOTH_1D = True
SHOW_BEST_FIT = True
BINS = 30
COLOR = 'lightsalmon'        # north: steelblue  red: lightsalmon
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 15
DPI = 150

OUTPUT_PREFIX = 'angular_momentum_ns'

# Constants matching fit_streamer_ns.py PARAM_CONFIG defaults
X_FIXED = -440.0
Y_FIXED = -1000.0
M_STAR = 26.0              # not used for j (purely kinematic)

# Free parameter names (order must match the samples columns)
# Current: z, v_r, omega, theta_axis, phi_axis
# Note: omega is LINEAR (round/yr), unlike MCMC version which uses log_omega
FREE_NAMES = ['z', 'v_r', 'omega', 'theta_axis', 'phi_axis']

PLOT_RANGE = None           # None = auto;

## North
# PLOT_RANGE = [
#     (3400, 4400),
#     (90, 100),
#     (135, 162),
# ]

## South
# PLOT_RANGE = [
#     (0,3000),
#     (30,75),
#     (-40,100)
# ]


# ============================================================
# Load samples
# ============================================================

data = np.load(SAMPLES_FILE)
equal_samples = data['equal_samples']   # (N, ndim) equal-weight posterior
ndim = equal_samples.shape[1]

if N_SUBSAMPLE > 0 and N_SUBSAMPLE < len(equal_samples):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(equal_samples), size=N_SUBSAMPLE, replace=False)
    equal_samples = equal_samples[idx]

# Best-fit = posterior median
best_theta = np.median(equal_samples, axis=0)

print(f'Loaded {equal_samples.shape[0]} samples from {SAMPLES_FILE}')
print(f'  ndim = {ndim}')
print(f'  Parameters: {FREE_NAMES}')

# ============================================================
# Compute specific angular momentum for each sample
# ============================================================

j_mags = []
theta_js = []
phi_js = []

for theta in equal_samples:
    params = dict(zip(FREE_NAMES, theta))
    z = params['z']

    r0, theta_part, phi_part = cartesian_to_spherical(X_FIXED, Y_FIXED, z)

    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=params['v_r'],
        omega_round_yr=params['omega'],           # already linear in NS samples
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)

    r_vec = np.array([x0, y0, z0])
    v_vec = np.array([vx0, vy0, vz0])

    j_vec = np.cross(r_vec, v_vec)
    j_mag = np.linalg.norm(j_vec)

    _, theta_j, phi_j = cartesian_to_spherical(j_vec[0], j_vec[1], j_vec[2])

    j_mags.append(j_mag)
    theta_js.append(theta_j)
    phi_js.append(phi_j)

j_mags = np.array(j_mags)
theta_js = np.array(theta_js)
phi_js = np.array(phi_js)

# ============================================================
# Statistics
# ============================================================

samples_3d = np.column_stack([j_mags, theta_js, phi_js])
q = np.percentile(samples_3d, [16, 50, 84], axis=0)

# Best-fit j (from posterior median params)
bf_p = dict(zip(FREE_NAMES, best_theta))
bf_r0, bf_tp, bf_pp = cartesian_to_spherical(X_FIXED, Y_FIXED, bf_p['z'])
bf_x0, bf_y0, bf_z0, bf_vx0, bf_vy0, bf_vz0 = get_streamer_initial_state('mendoza',
    r0=bf_r0, theta_part_deg=bf_tp, phi_part_deg=bf_pp,
    v_r=bf_p['v_r'], omega_round_yr=bf_p['omega'],
    theta_axis_deg=bf_p['theta_axis'], phi_axis_deg=bf_p['phi_axis'])
bf_j_vec = np.cross([bf_x0, bf_y0, bf_z0], [bf_vx0, bf_vy0, bf_vz0])
bf_j_mag = np.linalg.norm(bf_j_vec)
_, bf_theta_j, bf_phi_j = cartesian_to_spherical(bf_j_vec[0], bf_j_vec[1], bf_j_vec[2])
bf_j = np.array([bf_j_mag, bf_theta_j, bf_phi_j])

print('\nSpecific angular momentum (posterior median):')
for name, val in [('|j| [AU·km/s]', bf_j[0]),
                   ('theta_j [deg]', bf_j[1]),
                   ('phi_j [deg]', bf_j[2])]:
    print(f'  {name}: {val:.4g}')

print('\nSpecific angular momentum distribution:')
for i, name in enumerate(['|j| [AU·km/s]', 'theta_j [deg]', 'phi_j [deg]']):
    print(f'  {name}: {q[1,i]:.4g}  (-{q[1,i] - q[0,i]:.4g} / +{q[2,i] - q[1,i]:.4g})')

# ============================================================
# Save npz
# ============================================================

np.savez(
    f'{OUTPUT_PREFIX}{SAVE_SUFFIX}.npz',
    j_mag=j_mags,
    theta_j=theta_js,
    phi_j=phi_js,
    equal_samples=equal_samples,
    free_names=FREE_NAMES,
    x_fixed=X_FIXED,
    y_fixed=Y_FIXED,
    best_fit=bf_j,
)
print(f'\nSaved to {OUTPUT_PREFIX}{SAVE_SUFFIX}.npz')

# ============================================================
# Corner plot
# ============================================================

labels = [
    r'$|j|$ [AU$\cdot$km/s]',
    r'$\theta_j$ [$^\circ$]',
    r'$\phi_j$ [$^\circ$]',
]

ndim_j = 3
figsize = (min(ndim_j * 2.5, 14), min(ndim_j * 2.5, 14))

fig = corner.corner(
    samples_3d,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.2f',
    title_kwargs={'fontsize': TITLE_FONT_SIZE},
    label_kwargs={'fontsize': LABEL_FONT_SIZE, 'weight': 'bold'},
    smooth=SMOOTH_1D,
    smooth1d=SMOOTH_1D,
    bins=BINS,
    color=COLOR,
    hist_kwargs={'linewidth': 5, 'alpha': 0.6},
    fig=plt.figure(figsize=figsize),
    truths=q[1] if SHOW_BEST_FIT else None,
    truth_color=COLOR,
    range=PLOT_RANGE,
    plot_datapoints=False
)

outfile = f'{OUTPUT_PREFIX}{SAVE_SUFFIX}_corner.png'
fig.savefig(outfile, dpi=DPI, bbox_inches='tight')
print(f'Corner plot saved to {outfile}')
plt.close(fig)
