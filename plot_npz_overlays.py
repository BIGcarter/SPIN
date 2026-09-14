"""IDE-configured PPP/PPV visualization from saved NPZ files and a FITS cube.

Edit the configuration block below, then run this file directly.  Observation
NPZ files contain x/y/v arrays.  Trajectory NPZ files contain x/y/z/v_los
arrays; combined files with blue_x/... and red_x/... are also supported.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from astropy.io import fits
from astropy.constants import c
from matplotlib.cm import RdBu_r
from matplotlib.colors import Normalize
from plotly.subplots import make_subplots


# =============================================================================
# Configuration -- edit these values in the IDE
# =============================================================================

OBSERVATION_NPZ_FILES = [
    "../blue-ppvf.npz",
    "../red-ppvf.npz",
]

TRAJECTORY_NPZ_FILES = [
    "ns_trajectory_north_final.npz",
    "ns_trajectory_south_final_v3.npz",
]

# Number of points removed from the end of each loaded trajectory, in load
# order.  A combined NPZ may expand into multiple loaded trajectories, so give
# one value for each expanded trajectory.  For example, 10 plots x[:-10],
# y[:-10], etc.  Use 0 to draw through the final saved point.
TRAJECTORY_TRIM_END_POINTS = [
    0,  # traj 1
    30,  # traj 2
]

PPV_CUBE_FITS = "../cut_restfreq.fits"
OUTPUT_HTML = "npz_ppp_ppv_cube.html"

# The NPZ positions in channel-peak.ipynb were constructed from pixel offsets
# with these values.  Keep the same convention so the cube and NPZ overlap.
CUBE_CENTER_PIXEL_XY = (195.0, 193.0)  # zero-based (x, y) pixel coordinates
PIXEL_SCALE_ARCSEC = 0.01
DISTANCE_PC = 2600.0

# The cube is cropped to the observation region before constructing isosurfaces.
CUBE_RMS = 8.8e-4              # Jy/beam
CUBE_SIGMA_THRESHOLD = 5.0
CUBE_SURFACE_COUNT = 4
CUBE_OPACITY = 0.28

V_RANGE = (-8.0, 8.0)          # km/s; shared display and cube selection range

# Focus both PPV panels on the bounding box of all observation points.  The
# padding prevents points near the boundary from touching the scene frame.
PPV_FOCUS_PADDING_AU = 300.0

# Relative widths of PPP, PPV-points and PPV-cube panels.  The cube is wider
# because its translucent voxel structure benefits from more screen space.
SUBPLOT_COLUMN_WIDTHS = (0.27, 0.31, 0.42)
PPV_VELOCITY_ASPECT = 0.8


LINE_COLORS = (
    "#2166ac", "#b2182b", "#1b9e77", "#984ea3",
    "#ff7f00", "#4d4d4d", "#a65628", "#f781bf",
)
DATA_SYMBOLS = ("circle", "diamond", "square", "cross", "x")


def _validate_arrays(source, label, arrays):
    arrays = tuple(np.asarray(a).ravel() for a in arrays)
    sizes = {a.size for a in arrays}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError(f"{source}: {label} arrays must be non-empty and equal length")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"{source}: {label} arrays contain non-finite values")
    return arrays


def _matching_prefixes(keys, suffixes):
    """Return prefixes for which every requested ``_<suffix>`` key exists."""
    keys = set(keys)
    prefixes = []
    first_suffix = "_" + suffixes[0]
    for key in sorted(keys):
        if not key.endswith(first_suffix):
            continue
        prefix = key[:-len(first_suffix)]
        if prefix and all(f"{prefix}_{suffix}" in keys for suffix in suffixes):
            prefixes.append(prefix)
    return prefixes


def load_observations(paths):
    observations = []
    for path_string in paths:
        path = Path(path_string)
        with np.load(path) as data:
            if all(key in data for key in ("x", "y", "v")):
                arrays = _validate_arrays(path, "observation", (data["x"], data["y"], data["v"]))
                observations.append((path.stem, *arrays))
                continue

            prefixes = _matching_prefixes(data.files, ("x", "y", "v"))
            if not prefixes:
                raise KeyError(f"{path}: expected x/y/v arrays (optionally with a common prefix)")
            for prefix in prefixes:
                arrays = _validate_arrays(
                    path, f"observation '{prefix}'",
                    (data[f"{prefix}_x"], data[f"{prefix}_y"], data[f"{prefix}_v"]),
                )
                observations.append((f"{path.stem}:{prefix}", *arrays))
    return observations


def load_trajectories(paths):
    trajectories = []
    for path_string in paths:
        path = Path(path_string)
        with np.load(path) as data:
            if all(key in data for key in ("x", "y", "z", "v_los")):
                arrays = _validate_arrays(
                    path, "trajectory", (data["x"], data["y"], data["z"], data["v_los"]),
                )
                trajectories.append((path.stem, *arrays))
                continue

            prefixes = _matching_prefixes(data.files, ("x", "y", "z", "v_los"))
            if not prefixes:
                raise KeyError(
                    f"{path}: expected x/y/z/v_los arrays "
                    "(optionally with a common prefix)"
                )
            for prefix in prefixes:
                arrays = _validate_arrays(
                    path, f"trajectory '{prefix}'",
                    (data[f"{prefix}_x"], data[f"{prefix}_y"],
                     data[f"{prefix}_z"], data[f"{prefix}_v_los"]),
                )
                trajectories.append((f"{path.stem}:{prefix}", *arrays))
    return trajectories


def trim_trajectory_ends(trajectories, trim_end_points):
    """Synchronously remove a configurable number of final trajectory points."""
    if np.isscalar(trim_end_points):
        trim_end_points = [trim_end_points] * len(trajectories)
    if len(trim_end_points) != len(trajectories):
        raise ValueError(
            "TRAJECTORY_TRIM_END_POINTS must contain one value per loaded "
            f"trajectory ({len(trajectories)} required, {len(trim_end_points)} given)"
        )

    trimmed = []
    for (label, x, y, z, v_los), npoint in zip(trajectories, trim_end_points):
        if not isinstance(npoint, (int, np.integer)) or npoint < 0:
            raise ValueError(f"trajectory '{label}': trim count must be a non-negative integer")
        if npoint >= x.size:
            raise ValueError(
                f"trajectory '{label}': cannot trim {npoint} points from "
                f"a trajectory containing {x.size} points"
            )
        end = -npoint if npoint else None
        trimmed.append((label, x[:end], y[:end], z[:end], v_los[:end]))
    return trimmed


def load_cube_grid(
    path_string,
    center_pixel_xy,
    pixel_scale_arcsec,
    distance_pc,
    rms,
    sigma_threshold,
    v_range,
    x_range,
    y_range,
):
    """Load a focused, regular x/y/v/intensity grid for isosurface rendering."""
    path = Path(path_string)
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        cube = np.asarray(hdul[0].data).squeeze()

    if cube.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D cube after squeezing, got {cube.shape}")
    if not str(header.get("CTYPE3", "")).upper().startswith("FREQ"):
        raise ValueError(f"{path}: spectral axis CTYPE3 must be FREQ")

    rest_frequency = header.get("RESTFRQ", header.get("RESTFREQ"))
    if rest_frequency is None:
        raise KeyError(f"{path}: missing RESTFRQ/RESTFREQ")

    channel = np.arange(cube.shape[0], dtype=float)
    frequency_hz = (
        header["CRVAL3"]
        + ((channel + 1.0) - header["CRPIX3"]) * header["CDELT3"]
    )
    velocity_kms = c.to_value("km/s") * (1.0 - frequency_hz / rest_frequency)

    velocity_mask = (velocity_kms >= v_range[0]) & (velocity_kms <= v_range[1])
    au_per_pixel = pixel_scale_arcsec * distance_pc
    x_axis = (np.arange(cube.shape[2]) - center_pixel_xy[0]) * au_per_pixel
    y_axis = (np.arange(cube.shape[1]) - center_pixel_xy[1]) * au_per_pixel
    x_mask = (x_axis >= x_range[0]) & (x_axis <= x_range[1])
    y_mask = (y_axis >= y_range[0]) & (y_axis <= y_range[1])
    if not velocity_mask.any() or not x_mask.any() or not y_mask.any():
        raise ValueError(f"{path}: configured PPV focus region does not intersect the cube")

    focused_cube = cube[np.ix_(velocity_mask, y_mask, x_mask)]
    focused_cube = np.nan_to_num(focused_cube, nan=0.0)
    focused_v = velocity_kms[velocity_mask]
    focused_y = y_axis[y_mask]
    focused_x = x_axis[x_mask]
    grid_v, grid_y, grid_x = np.meshgrid(
        focused_v, focused_y, focused_x, indexing="ij"
    )

    threshold = sigma_threshold * rms
    if focused_cube.max() < threshold:
        raise ValueError(
            f"{path}: focused cube has no emission above {sigma_threshold:g} sigma"
        )
    return (
        grid_x.ravel(), grid_y.ravel(), grid_v.ravel(), focused_cube.ravel(),
        focused_cube.shape,
    )


def plot_overlays(
    observations,
    trajectories,
    cube_voxels,
    output_html,
    v_range=(-8.0, 8.0),
    x_range=None,
    y_range=None,
):
    """Create a three-panel PPP/PPV/PPV-cube Plotly view."""
    v_norm = Normalize(vmin=v_range[0], vmax=v_range[1])

    def velocity_colors(values):
        rgba = RdBu_r(v_norm(values))
        return [f"rgb({c[0] * 255:.0f},{c[1] * 255:.0f},{c[2] * 255:.0f})" for c in rgba]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "PPP trajectories",
            "PPV observations + trajectories",
            "PPV cube + trajectories",
        ),
        horizontal_spacing=0.04,
        column_widths=SUBPLOT_COLUMN_WIDTHS,
    )

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", name="central source",
        marker=dict(size=7, color="black", symbol="diamond"),
        legendgroup="central-source",
    ), row=1, col=1)

    radius = 250.0  
    center_x, center_y, center_z = 0, 0, 0  

    phi = np.linspace(0, np.pi, 60)       
    theta = np.linspace(0, 2 * np.pi, 60) 
    phi, theta = np.meshgrid(phi, theta)

    x_sphere = center_x + radius * np.sin(phi) * np.cos(theta)
    y_sphere = center_y + radius * np.sin(phi) * np.sin(theta)
    z_sphere = center_z + radius * np.cos(phi)

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        name="influence sphere",
        opacity=0.3,                  
        showscale=False,              
        colorscale=[[0, 'blue'], [1, 'blue']], 
        legendgroup="central-source", 
        showlegend=True               
    ), row=1, col=1)

    radius_2 = 400.0  
    x_sphere_2 = center_x + radius_2 * np.sin(phi) * np.cos(theta)
    y_sphere_2 = center_y + radius_2 * np.sin(phi) * np.sin(theta)
    z_sphere_2 = center_z + radius_2 * np.cos(phi)

    fig.add_trace(go.Surface(
        x=x_sphere_2, y=y_sphere_2, z=z_sphere_2,
        name="influence sphere",
        opacity=0.3,                  
        showscale=False,              
        colorscale=[[0, 'red'], [1, 'red']], 
        legendgroup="central-source", 
        showlegend=True               
    ), row=1, col=1)

    # Add the translucent cube first so trajectories remain visually prominent.
    cube_x, cube_y, cube_v, cube_intensity, _cube_shape = cube_voxels
    fig.add_trace(go.Isosurface(
        x=cube_x, y=cube_y, z=cube_v, value=cube_intensity,
        name="PPV cube",
        isomin=CUBE_SIGMA_THRESHOLD * CUBE_RMS,
        isomax=float(np.max(cube_intensity)),
        surface=dict(count=CUBE_SURFACE_COUNT),
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale="Magma",
        opacity=CUBE_OPACITY,
        colorbar=dict(title="Jy/beam", len=0.55, x=1.01),
        hovertemplate=(
            "X=%{x:.0f} AU<br>Y=%{y:.0f} AU<br>"
            "V=%{z:.2f} km/s<br>I=%{value:.4g} Jy/beam<extra></extra>"
        ),
        legendgroup="cube",
    ), row=1, col=3)

    for index, (label, x, y, z, v_los) in enumerate(trajectories):
        line_color = LINE_COLORS[index % len(LINE_COLORS)]
        marker_colors = velocity_colors(v_los)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines+markers", name=f"trajectory: {label}",
            line=dict(color=line_color, width=4),
            marker=dict(size=2.5, color=marker_colors),
            legendgroup=f"trajectory-{index}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=v_los, mode="lines+markers", name=f"trajectory: {label}",
            line=dict(color=line_color, width=4),
            marker=dict(size=2.5, color=marker_colors),
            legendgroup=f"trajectory-{index}", showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=v_los, mode="lines+markers", name=f"trajectory: {label}",
            line=dict(color=line_color, width=5),
            marker=dict(size=3, color=marker_colors),
            legendgroup=f"trajectory-{index}", showlegend=False,
        ), row=1, col=3)

    for index, (label, x, y, v) in enumerate(observations):
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=v, mode="markers", name=f"data: {label}",
            marker=dict(
                size=5, color=velocity_colors(v),
                symbol=DATA_SYMBOLS[index % len(DATA_SYMBOLS)],
                line=dict(color="black", width=0.5),
            ),
            legendgroup=f"data-{index}",
        ), row=1, col=2)

    if x_range is None:
        x_range = (min(cube_x), max(cube_x))
    if y_range is None:
        y_range = (min(cube_y), max(cube_y))
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    spatial_span = max(x_span, y_span)
    x_scale = x_span / spatial_span if spatial_span else 1.0
    y_scale = y_span / spatial_span if spatial_span else 1.0

    camera = dict(
        up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=-2.5),
    )
    fig.update_layout(
        template="plotly_white", legend=dict(itemsizing="constant"),
        scene1=dict(
            xaxis=dict(title="X [AU]", autorange="reversed"),
            yaxis=dict(title="Y [AU]"), zaxis=dict(title="Z [AU]"),
            aspectmode="data", camera=camera,
        ),
        scene2=dict(
            xaxis=dict(title="X [AU]", range=list(x_range)),
            yaxis=dict(title="Y [AU]", range=list(y_range)),
            zaxis=dict(title="V_los [km/s]", range=list(v_range)), aspectmode="manual",
            aspectratio=dict(x=x_scale, y=y_scale, z=PPV_VELOCITY_ASPECT), camera=camera,
        ),
        scene3=dict(
            xaxis=dict(title="X [AU]", range=list(x_range)),
            yaxis=dict(title="Y [AU]", range=list(y_range)),
            zaxis=dict(title="V_los [km/s]", range=list(v_range)), aspectmode="manual",
            aspectratio=dict(x=x_scale, y=y_scale, z=PPV_VELOCITY_ASPECT), camera=camera,
        ),
    )
    fig.write_html(output_html)
    return fig


def main():
    observations = load_observations(OBSERVATION_NPZ_FILES)
    trajectories = load_trajectories(TRAJECTORY_NPZ_FILES)
    trajectories = trim_trajectory_ends(trajectories, TRAJECTORY_TRIM_END_POINTS)

    observation_x = np.concatenate([item[1] for item in observations])
    observation_y = np.concatenate([item[2] for item in observations])
    x_range = (
        float(observation_x.min() - PPV_FOCUS_PADDING_AU),
        float(observation_x.max() + PPV_FOCUS_PADDING_AU),
    )
    y_range = (
        float(observation_y.min() - PPV_FOCUS_PADDING_AU),
        float(observation_y.max() + PPV_FOCUS_PADDING_AU),
    )

    cube_voxels = load_cube_grid(
        PPV_CUBE_FITS,
        center_pixel_xy=CUBE_CENTER_PIXEL_XY,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
        distance_pc=DISTANCE_PC,
        rms=CUBE_RMS,
        sigma_threshold=CUBE_SIGMA_THRESHOLD,
        v_range=V_RANGE,
        x_range=x_range,
        y_range=y_range,
    )
    plot_overlays(
        observations,
        trajectories,
        cube_voxels,
        OUTPUT_HTML,
        v_range=V_RANGE,
        x_range=x_range,
        y_range=y_range,
    )
    print(f"Loaded {len(observations)} observation set(s) and {len(trajectories)} trajectory set(s)")
    n_above = np.count_nonzero(cube_voxels[3] >= CUBE_SIGMA_THRESHOLD * CUBE_RMS)
    print(
        f"Loaded cube grid {cube_voxels[4]} with {n_above} voxels "
        f"above {CUBE_SIGMA_THRESHOLD:g} sigma"
    )
    print(f"PPV focus: X={x_range} AU, Y={y_range} AU")
    print(f"Saved {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
