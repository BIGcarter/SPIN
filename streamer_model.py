import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import solve_ivp

from streamer_ic import get_axis_unit_vector


def gm_from_mstar(m_star):
    """Compute gravitational parameter GM in km²/s²·AU."""
    gm_physical = const.G * (m_star * u.M_sun)
    return gm_physical.to(u.km**2 / u.s**2 * u.au).value


def linear_drag(alpha=550.0):
    """
    Return a drag function for constant linear damping.

    Parameters
    ----------
    alpha : float
        Damping timescale. Larger alpha = weaker drag.

    Returns
    -------
    drag_func : callable
        Signature: drag_func(t, x, y, z, vx, vy, vz) -> (ax, ay, az)
    """
    def drag(t, x, y, z, vx, vy, vz):
        ax = -vx / alpha
        ay = -vy / alpha
        az = -vz / alpha
        return ax, ay, az
    return drag


def stopping_sphere(r_min):
    """
    Return an event function that terminates integration when the particle
    enters a sphere of radius r_min around the origin.

    Parameters
    ----------
    r_min : float
        Stopping radius (AU). Must be > 0.

    Returns
    -------
    event : callable
        Signature: event(t, Y, GM, drag_func) -> float.
        Has .terminal = True and .direction = -1 (trigger when crossing
        from above, i.e. r decreasing through r_min).
    """
    def event(t, Y, GM, drag_func):
        x, y, z = Y[0], Y[1], Y[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        return r - r_min

    event.terminal = True
    event.direction = -1
    return event


def azimuth_cutoff_idx(x, y, max_delta_deg=180.0):
    """
    Return the index of the last trajectory point before cumulative azimuth
    change (in the x-y plane) exceeds ``max_delta_deg``.

    Azimuth is measured from +y clockwise: ``atan2(x, y)``. The function
    unwraps the angle and finds the first point where the absolute
    difference from the initial azimuth exceeds the threshold.

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates of the trajectory.
    max_delta_deg : float
        Maximum allowed azimuthal change in degrees.

    Returns
    -------
    cutoff : int
        Index of the last point that satisfies |Δazim| ≤ max_delta_deg.
        If all points satisfy the condition, returns ``len(x) - 1``.
    """
    azim = np.arctan2(x, y)  # range [-π, π], 0 at +y, clockwise increasing
    azim_unwrapped = np.unwrap(azim)
    delta = np.abs(azim_unwrapped - azim_unwrapped[0])
    max_delta_rad = np.deg2rad(max_delta_deg)
    exceed = np.where(delta > max_delta_rad)[0]
    if len(exceed) == 0:
        return len(x) - 1
    return exceed[0] - 1


def eq_streamer(t, Y, GM, drag_func=None):
    """
    ODE right-hand side: gravity + optional drag.

    Parameters
    ----------
    t : float
        Time (independent variable).
    Y : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz].
    GM : float
        Gravitational parameter in km2/s2 AU.
    drag_func : callable or None
        drag_func(t, x, y, z, vx, vy, vz) -> (ax_drag, ay_drag, az_drag).
        If None, no drag is applied (pressureless).

    Returns
    -------
    dYdt : list of float, length 6
    """
    x, y, z, vx, vy, vz = Y
    r = np.sqrt(x**2 + y**2 + z**2)

    if r < 1e-3:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    ax_grav = -GM * x / r**3
    ay_grav = -GM * y / r**3
    az_grav = -GM * z / r**3

    if drag_func is not None:
        ax_drag, ay_drag, az_drag = drag_func(t, x, y, z, vx, vy, vz)
    else:
        ax_drag = ay_drag = az_drag = 0.0

    return [vx, vy, vz,
            ax_grav + ax_drag,
            ay_grav + ay_drag,
            az_grav + az_drag]


def integrate_trajectory(initial_state, t_span, t_eval, GM, drag_func=None, events=None):
    """
    Integrate streamer ODE.

    Parameters
    ----------
    initial_state : sequence of 6 floats
        [x0, y0, z0, vx0, vy0, vz0].
    t_span : tuple (t_start, t_end)
    t_eval : np.ndarray
        Evaluation time points.
    GM : float
        Gravitational parameter in km²/s²·AU.
    drag_func : callable or None
        drag_func(t, x, y, z, vx, vy, vz) -> (ax_drag, ay_drag, az_drag).
    events : callable or list of callable or None
        Event function(s) for solve_ivp. Use stopping_sphere(r_min) to
        terminate when the particle reaches radius r_min.

    Returns
    -------
    solution : OdeResult
        solution.y has shape (6, n_points).
        If an event triggered, solution.t_events contains the event times.
    """
    return solve_ivp(eq_streamer, t_span, initial_state, args=(GM, drag_func),
                     t_eval=t_eval, method='RK45', events=events)


def sample_sphere_trajectories(
    x_center, y_center, z_center,
    sphere_radius,
    n_particles,
    v_r,
    log10_omega,
    theta_axis_deg,
    phi_axis_deg,
    M=15.0,
    alpha=500.0,
    t_span=(0, 3000),
    t_eval=None,
    stopping_r=100.0,
    azimuth_max_delta_deg=200.0,
    rng_seed=None,
):
    """
    Generate n_particles trajectories from uniformly sampled positions within
    a sphere of radius sphere_radius centered at (x_center, y_center, z_center).

    All particles share the same physical parameters (v_r, rotation, axis).
    Velocity is recomputed per particle from its perturbed position.

    Parameters
    ----------
    x_center, y_center, z_center : float
        Center of the sampling sphere (AU).
    sphere_radius : float
        Radius of the sampling sphere (AU).
    n_particles : int
        Number of test particles.
    v_r : float
        Radial velocity magnitude (km/s, negative for infall).
    log10_omega : float
        log10 of rotation angular speed (round/yr). E.g. -4.3 means 10^-4.3.
    theta_axis_deg, phi_axis_deg : float
        Rotation axis direction (degrees).
    M : float
        Central stellar mass (Msun).
    alpha : float
        Drag damping timescale.
    t_span : tuple
        Integration time range.
    t_eval : np.ndarray or None
        Evaluation time points. If None, 1200 points in t_span.
    stopping_r : float
        Stopping sphere radius (AU).
    azimuth_max_delta_deg : float
        Max azimuthal change before truncation (degrees).
    rng_seed : int or None
        Seed for reproducible sampling.

    Returns
    -------
    list of tuple
        Each element is (x, y, z, v_los) for a successful trajectory.
        The **first** element is always the center (unperturbed) trajectory.
        Failed trajectories are skipped.
    """
    rng = np.random.default_rng(rng_seed)

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1200)

    # --- Uniform volume sampling in sphere ---
    # Direction: uniform on unit sphere
    u_angle = rng.uniform(0, 1, n_particles)
    v_angle = rng.uniform(0, 1, n_particles)
    theta_sphere = 2 * np.pi * u_angle
    phi_sphere = np.arccos(2 * v_angle - 1)

    sin_phi = np.sin(phi_sphere)
    dx = sphere_radius * sin_phi * np.cos(theta_sphere)
    dy = sphere_radius * sin_phi * np.sin(theta_sphere)
    dz = sphere_radius * np.cos(phi_sphere)

    # Radius: PDF ∝ r^2 → r = R * u^(1/3)
    r_frac = rng.uniform(0, 1, n_particles) ** (1 / 3)

    positions = np.column_stack([
        x_center + r_frac * dx,
        y_center + r_frac * dy,
        z_center + r_frac * dz,
    ])

    # --- Physical constants for velocity ---
    omega_round_yr = 10 ** log10_omega
    omega_rad_s = (omega_round_yr * u.cycle / u.yr).to(u.rad / u.s).value
    au_s_to_kms = (1.0 * u.au / u.s).to(u.km / u.s).value
    n_axis = get_axis_unit_vector(theta_axis_deg, phi_axis_deg)

    GM = gm_from_mstar(M)
    drag_func = linear_drag(alpha=alpha)
    events = stopping_sphere(stopping_r)

    def _integrate_one(px, py, pz, vx, vy, vz):
        initial_state = [px, py, pz, vx, vy, vz]
        sol = integrate_trajectory(
            initial_state, t_span, t_eval, GM, drag_func, events=events,
        )
        if not sol.success:
            return None
        x_arr = sol.y[0].copy()
        y_arr = sol.y[1].copy()
        z_arr = sol.y[2].copy()
        v_arr = sol.y[5].copy()
        cut = azimuth_cutoff_idx(x_arr, y_arr, max_delta_deg=azimuth_max_delta_deg)
        return (x_arr[:cut + 1], y_arr[:cut + 1], z_arr[:cut + 1], v_arr[:cut + 1])

    # --- Center particle velocity (shared by all tube particles) ---
    r_c = np.sqrt(x_center**2 + y_center**2 + z_center**2)
    v_radial_c = v_r * np.array([x_center, y_center, z_center]) / r_c
    v_rot_c_au_s = np.cross(omega_rad_s * n_axis, np.array([x_center, y_center, z_center]))
    v_rot_c = v_rot_c_au_s * au_s_to_kms
    v_center = v_radial_c + v_rot_c

    trajectories = []

    # --- Center trajectory (reference) ---
    center = _integrate_one(x_center, y_center, z_center, v_center[0], v_center[1], v_center[2])
    if center is not None:
        trajectories.append(center)

    # --- Tube particles (same velocity as center) ---
    for i in range(n_particles):
        px, py, pz = positions[i]
        try:
            traj = _integrate_one(px, py, pz, v_center[0], v_center[1], v_center[2])
            if traj is not None:
                trajectories.append(traj)
        except Exception:
            continue

    return trajectories
