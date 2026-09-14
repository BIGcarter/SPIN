import sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import multiprocessing as mp
import copy

import dynesty
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot

from streamer_ic import get_streamer_initial_state, cartesian_to_spherical
from streamer_model import gm_from_mstar, integrate_trajectory, linear_drag, stopping_sphere, azimuth_cutoff_idx
from visual_orbit import plot_orbit

# ============================================================
# Configuration
# ============================================================

PARAM_CONFIG = {
    'z':           {'is_constant': False, 'prior_range': [-1000, 0],       'label': r'$z$ [AU]'},  # 0, 1500 north
    'v_r':         {'is_constant': False, 'prior_range': [-10, 1],        'label': r'$v_r$ [km/s]'},
    'omega':       {'is_constant': False, 'prior_range': [-6, -3], 'log_uniform': True, 'label': r'$\log_{10}(\omega)$ [round/yr]'},
    'theta_axis':  {'is_constant': False, 'prior_range': [0, 90],         'label': r'$\theta_{axis}$ [deg]'},
    'phi_axis':    {'is_constant': False, 'prior_range': [20, 180],        'label': r'$\phi_{axis}$ [deg]'},
    'M':           {'is_constant': True,  'value': 26,                  'label': r'$M$ [$M_\odot$]'},
    # 'alpha':       {'is_constant': True,  'value': 500,                 'label': r'$\alpha$'},
    'alpha':       {'is_constant': True,  'value': 1e5,                 'label': r'$\alpha$'},
    'x':           {'is_constant': True,  'value': -50,                'label': r'$x$ [AU]'},
    'y':           {'is_constant': True,  'value': -750,                'label': r'$y$ [AU]'},   # north -50, 1000 south -50 -750
}

NLIVE_INIT = 2500
NLIVE_BATCH = 1000
N_EFFECTIVE = 100000
DLOGZ_INIT = 0.02
N_CPUS = 10
SAMPLER_SEED = 32
RESAMPLE_SEED = 33
CHECKPOINT_EVERY = 600

T_SPAN = (0, 3000)
T_EVAL = np.linspace(T_SPAN[0], T_SPAN[1], 1200)
STOPPING_R = 150.0
AZIMUTH_MAX_DELTA_DEG = 70.0

# OBS_DATA = '../data/CH3OH/CH3OH-CH3CN-south-masked-2600pc-center76-75-vsys10-au-relative-vsys.npz'
OBS_DATA = '../data/HNCO/HNCO-south-masked-2600pc-center76-75-vsys10-au-relative-vsys.npz'
SAVE_SUFFIX = '_HNCO_south_0908_v3'
RESULTS_DIR = Path(__file__).resolve().parent / 'results'
SIGMA_XY = 60.0
SIGMA_V = 1.331

# Flux weighting for the data -> model Chamfer term.  The weights are
# normalized to mean 1, so enabling them does not arbitrarily rescale the
# overall likelihood.  Set USE_FLUX_WEIGHTS=False (or gamma=0) to recover
# equal weighting.  gamma=0.5 gives a tempered sqrt(flux) dependence.
USE_FLUX_WEIGHTS = True
FLUX_WEIGHT_GAMMA = 1
FLUX_CLIP_PERCENTILES = (5.0, 95.0)

# --- Parse PARAM_CONFIG into free / constant ---
_free_names = []
_constant_values = {}
_prior_bounds = []
_prior_log_uniform = []

for _name, _cfg in PARAM_CONFIG.items():
    if _cfg['is_constant']:
        _constant_values[_name] = _cfg['value']
    else:
        _free_names.append(_name)
        _prior_bounds.append(_cfg['prior_range'])
        _prior_log_uniform.append(_cfg.get('log_uniform', False))

N_FREE = len(_free_names)
_prior_bounds = np.array(_prior_bounds)
_param_labels = [PARAM_CONFIG[n]['label'] for n in _free_names]

print(f'Free parameters ({N_FREE}): {_free_names}')
print(f'Constants: {list(_constant_values.keys())}')

# ============================================================
# Static data & KDTree (built once)
# ============================================================

_data = np.load(OBS_DATA)
_data_x = _data['x']
_data_y = _data['y']
_data_v = _data['v']
_data_flux = _data['flux']


def _build_flux_weights(flux):
    """Return mean-one Chamfer weights derived from observed flux."""
    flux = np.asarray(flux, dtype=float)
    if not USE_FLUX_WEIGHTS or FLUX_WEIGHT_GAMMA == 0:
        return np.ones_like(flux)
    if FLUX_WEIGHT_GAMMA < 0:
        raise ValueError('FLUX_WEIGHT_GAMMA must be non-negative')
    if not np.all(np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError('Flux weighting requires finite, positive flux values')

    clip_lo, clip_hi = np.percentile(flux, FLUX_CLIP_PERCENTILES)
    clipped_flux = np.clip(flux, clip_lo, clip_hi)
    weights = clipped_flux ** FLUX_WEIGHT_GAMMA
    return weights / np.mean(weights)


DATA_WEIGHTS = _build_flux_weights(_data_flux)
print(
    f'Flux weighting: enabled={USE_FLUX_WEIGHTS}, '
    f'gamma={FLUX_WEIGHT_GAMMA:g}, '
    f'weight range=[{DATA_WEIGHTS.min():.3f}, {DATA_WEIGHTS.max():.3f}]'
)

_data_ppv = np.column_stack([
    _data_x / SIGMA_XY,
    _data_y / SIGMA_XY,
    _data_v / SIGMA_V,
])
DATA_KD_TREE = KDTree(_data_ppv)
DATA_XY_XRANGE = np.ptp(_data_x)
DATA_XY_YRANGE = np.ptp(_data_y)

# ============================================================
# Trajectory wrapper
# ============================================================

def _theta_to_params(theta):
    params = dict(_constant_values)
    for i, name in enumerate(_free_names):
        params[name] = theta[i]
    return params


def compute_trajectory(params):
    z = params['z']
    x = params.get('x', _constant_values.get('x'))
    y = params.get('y', _constant_values.get('y'))

    r0, theta_part, phi_part = cartesian_to_spherical(x, y, z)

    ic_params = dict(
        r0=r0,
        theta_part_deg=theta_part,
        phi_part_deg=phi_part,
        v_r=params['v_r'],
        omega_round_yr=params['omega'],
        theta_axis_deg=params['theta_axis'],
        phi_axis_deg=params['phi_axis'],
    )

    x0, y0, z0, vx0, vy0, vz0 = get_streamer_initial_state('mendoza', **ic_params)
    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    M_val = params.get('M', _constant_values.get('M'))
    alpha_val = params.get('alpha', _constant_values.get('alpha'))
    GM = gm_from_mstar(M_val)
    drag_func = linear_drag(alpha=alpha_val)

    r0 = np.sqrt(x0**2 + y0**2 + z0**2)
    E0 = 0.5 * (vx0**2 + vy0**2 + vz0**2) - GM / r0
    if E0 >= 0:
        return None, None, None, None

    try:
        sol = integrate_trajectory(initial_state, T_SPAN, T_EVAL, GM, drag_func, events=stopping_sphere(STOPPING_R))
        if not sol.success:
            return None, None, None, None
        x_arr = sol.y[0].copy()
        y_arr = sol.y[1].copy()
        z_arr = sol.y[2].copy()
        v_arr = sol.y[5].copy()

        cut = azimuth_cutoff_idx(x_arr, y_arr, max_delta_deg=AZIMUTH_MAX_DELTA_DEG)
        return x_arr[:cut+1], y_arr[:cut+1], z_arr[:cut+1], v_arr[:cut+1]
    except Exception:
        return None, None, None, None


# ============================================================
# Chamfer loss in error-normalized PPV space
# ============================================================

def chamfer_loss(traj_x, traj_y, traj_v, n_sample=200):
    traj_ppv = np.column_stack([
        traj_x / SIGMA_XY, traj_y / SIGMA_XY, traj_v / SIGMA_V,
    ])

    # data -> model
    kd_traj = KDTree(traj_ppv)
    d_dm, _ = kd_traj.query(DATA_KD_TREE.data)
    loss_dm = np.sum(DATA_WEIGHTS * d_dm ** 2)

    # model -> data
    # n_pts = traj_ppv.shape[0]
    # if n_pts <= n_sample:
    #     sample_ppv = traj_ppv
    # else:
    #     idx = np.linspace(0, n_pts - 1, n_sample, dtype=int)
    #     sample_ppv = traj_ppv[idx]

    # d_md, _ = DATA_KD_TREE.query(sample_ppv)
    # loss_md = np.sum(d_md ** 2)
    # loss_md = np.mean(d_md**2)

    # lamb = 0.2

    return loss_dm
    # return loss_dm + loss_md * len(DATA_KD_TREE.data) * lamb
    # return loss_dm + loss_md


# ============================================================
# Prior transform (unit cube -> parameter space)
# ============================================================

def prior_transform(u):
    theta = np.empty(N_FREE)
    for i in range(N_FREE):
        lo, hi = _prior_bounds[i]
        val = lo + u[i] * (hi - lo)
        if _prior_log_uniform[i]:
            theta[i] = 10.0 ** val
        else:
            theta[i] = val
    return theta


# ============================================================
# Log-likelihood
# ============================================================

def log_likelihood(theta):
    params = _theta_to_params(theta)

    try:
        traj_x, traj_y, traj_z, traj_v = compute_trajectory(params)
    except Exception:
        return -np.inf

    if traj_x is None:
        return -np.inf

    x_range = np.ptp(traj_x)
    y_range = np.ptp(traj_y)
    if x_range < 0.3 * DATA_XY_XRANGE and y_range < 0.3 * DATA_XY_YRANGE:
        return -np.inf

    loss = chamfer_loss(traj_x, traj_y, traj_v)
    return -0.5 * loss


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pool = mp.Pool(N_CPUS) if N_CPUS > 1 else None
    sampler = DynamicNestedSampler(
        log_likelihood,
        prior_transform,
        ndim=N_FREE,
        nlive=NLIVE_INIT,
        bound='multi',
        sample='rwalk',
        walks=25,
        pool=pool,
        queue_size=N_CPUS,
        use_pool={'prior_transform': False},
        rstate=np.random.default_rng(SAMPLER_SEED),
        update_interval=1.5,
    )

    sampler.run_nested(
        dlogz_init=DLOGZ_INIT,
        nlive_batch=NLIVE_BATCH,
        n_effective=N_EFFECTIVE,
        save_bounds=False,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_file=str(RESULTS_DIR / f'ns_checkpoint{SAVE_SUFFIX}.h5'),
        )
    print('Done. Wake up.')

    if pool is not None:
        pool.close()
        pool.join()

    # --- Post-processing ---
    results = sampler.results
    samples = results.samples
    logvol = results.logvol
    logl = results.logl
    logz = results.logz[-1]
    logzerr = results.logzerr[-1]

    print(f'log Z = {logz:.3f} +/- {logzerr:.3f}')

    weights = results.importance_weights()
    equal_samples = results.samples_equal(
        rstate=np.random.default_rng(RESAMPLE_SEED),
    )

    print(f'Equal-weight posterior samples: {equal_samples.shape[0]}')

    np.savez(RESULTS_DIR / f'ns_samples{SAVE_SUFFIX}.npz',
             samples=samples,
             equal_samples=equal_samples,
             logvol=logvol,
             logl=logl,
             logwt=results.logwt,
             weights=weights,
             logz=logz,
             logzerr=logzerr,
             sampler_seed=SAMPLER_SEED,
             resample_seed=RESAMPLE_SEED,
             use_flux_weights=USE_FLUX_WEIGHTS,
             flux_weight_gamma=FLUX_WEIGHT_GAMMA,
             flux_clip_percentiles=FLUX_CLIP_PERCENTILES,
             data_weights=DATA_WEIGHTS,
    )

    # --- Build visualization results with omega in log10 space ---
    omega_idx = _free_names.index('omega')
    results_viz = copy.deepcopy(results)
    results_viz.samples[:, omega_idx] = np.log10(results_viz.samples[:, omega_idx])

    # --- Summary plot (logZ, dlogZ, nlive vs ncall) ---
    fig_summary, axes_summary = dyplot.runplot(results)
    fig_summary.savefig(RESULTS_DIR / f'ns_summary{SAVE_SUFFIX}.png', dpi=150)
    # print('Summary plot saved to ns_summary.png')

    # --- Trace plots ---
    param_labels = _param_labels
    fig_trace, axes_trace = dyplot.traceplot(results_viz, labels=param_labels)
    fig_trace.savefig(RESULTS_DIR / f'ns_trace{SAVE_SUFFIX}.png', dpi=150)
    # print('Trace plots saved to ns_trace.png')

    # --- Corner plot ---
    fig_corner, axes_corner = dyplot.cornerplot(results_viz, labels=param_labels,
                                                 quantiles=[0.16, 0.5, 0.84],
                                                 show_titles=True,
                                                 title_kwargs={'fontsize': 10})
    fig_corner.savefig(RESULTS_DIR / f'ns_corner{SAVE_SUFFIX}.png', dpi=150)
    # print('Corner plot saved to ns_corner.png')

    # --- Bounding distribution plot (pairwise grid) ---
    # try:
    #     import matplotlib.pyplot as plt
    #     nd = N_FREE
    #     fig_bound, axes = plt.subplots(nd, nd, figsize=(3 * nd, 3 * nd))
    #     for i in range(nd):
    #         for j in range(nd):
    #             ax = axes[i, j] if nd > 1 else axes
    #             if i == j:
    #                 ax.text(0.5, 0.5, _free_names[i], ha='center', va='center',
    #                         transform=ax.transAxes, fontsize=10)
    #                 ax.set_xticks([]); ax.set_yticks([])
    #             elif j < i:
    #                 dyplot.boundplot(results_viz, dims=(j, i), it=-1, ax=ax,
    #                                 show_titles=False)
    #             else:
    #                 ax.axis('off')
    #     fig_bound.tight_layout()
    #     fig_bound.savefig('ns_bound.png', dpi=150)
    # except Exception:
    #     print("bound plot errors.")

    # --- Best-fit: posterior median (more robust than max-likelihood) ---
    best_theta = np.median(equal_samples, axis=0)
    params_bf = _theta_to_params(best_theta)

    print(f'\nBest-fit params (posterior median):')
    for name in _free_names:
        print(f'  {name} = {params_bf[name]:.4g}')
    print(f'  lnZ = {logz:.3f} +/- {logzerr:.3f}')

    traj_x, traj_y, traj_z, traj_v = compute_trajectory(params_bf)
    if traj_x is None:
        print('WARNING: Best-fit trajectory computation failed!')
        sys.exit(1)

    # --- Posterior trajectory samples ---
    N_LINES = 100
    multi_trajs = []
    rng = np.random.default_rng(42)
    idx_samples = rng.choice(len(equal_samples),
                             size=min(N_LINES, len(equal_samples)),
                             replace=False)
    for idx in idx_samples:
        p = equal_samples[idx]
        pm = _theta_to_params(p)
        tx, ty, tz, tv = compute_trajectory(pm)
        if tx is not None:
            multi_trajs.append((tx, ty, tz, tv))
    print(f'Posterior trajectories: {len(multi_trajs)}/{min(N_LINES, len(equal_samples))} computed')

    # --- PPP + PPV Plotly figure ---
    plot_orbit(
        traj_x, traj_y, traj_z, traj_v,
        _data_x, _data_y, _data_v,
        params_bf['x'], params_bf['y'], params_bf['z'],
        params_bf['theta_axis'], params_bf['phi_axis'],
        multi_trajs=multi_trajs,
        sigma_xy=SIGMA_XY, sigma_v=SIGMA_V,
        v_range=(-6, 6),
        output_html=str(RESULTS_DIR / f'ns_bestfit{SAVE_SUFFIX}.html'),
    )
    # print('Best-fit trajectory saved to ns_bestfit.html')

    # --- Summary percentiles ---
    q = np.percentile(equal_samples, [16, 50, 84], axis=0)
    print(f'\nParameter ranges (16%, 50%, 84%):')
    for i, name in enumerate(_free_names):
        print(f'  {name}: {q[1, i]:.4g}  (-{q[1, i] - q[0, i]:.3g} / +{q[2, i] - q[1, i]:.3g})')

    trajectory_path = RESULTS_DIR / f'ns_trajectory{SAVE_SUFFIX}.npz'
    np.savez(trajectory_path,
             x=traj_x, y=traj_y, z=traj_z, v_los=traj_v,
             params=q,
    )
    print(f'Trajectory data saved to {trajectory_path}')
