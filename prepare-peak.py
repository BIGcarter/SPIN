"""Prepare channel-map dendrogram peaks for the PPV fitting scripts.

Run without arguments to use the configuration block, or pass CLI arguments
for agent/automation use. CLI values override the corresponding configuration.
"""

import argparse
from pathlib import Path
from typing import Sequence

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astrodendro import Dendrogram
from astrodendro.analysis import PPStatistic
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from spectral_cube import SpectralCube


# =============================================================================
# Configuration (all user-editable parameters live here)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Input cube and the FITS header used for WCS/beam metadata.  Set
# HEADER_FITS_PATH = None to use the cube header itself.
CUBE_FITS_PATH = SCRIPT_DIR / "cut_restfreq.fits"
CUBE_HDU = 0
HEADER_FITS_PATH = None
HEADER_HDU = 0

# RMS is expressed in the cube's BUNIT (Jy/beam for the notebook data).
RMS = 8.8e-4
REST_FREQUENCY = 220.071219 * u.GHz
MOLNAME = "CH3OH"
VELOCITY_CONVENTION = "radio"

# Moment maps use pixels above MOMENT_MASK_SIGMA * RMS.
MOMENT_MASK_SIGMA = 5.0

# Inclusive channel range passed to astrodendro.  Use None for the first/last
# cube channel, respectively.
CHANNEL_START = 24
CHANNEL_END = 36
DENDRO_MIN_VALUE_SIGMA = 5.0
DENDRO_MIN_DELTA_SIGMA = 1.0

# None computes one beam area in pixels from BMAJ/BMIN and the celestial WCS.
# Set an integer (e.g. 24, as in the notebook) to override it.
DENDRO_MIN_NPIX = None
DENDRO_MIN_BEAM_FRACTION = 1.0

# Pixel-to-AU conversion used by the fitting code.  At a distance D pc,
# 1 arcsec corresponds to D AU.  The signs follow the notebook's pixel-axis
# convention; change them here if a sky-coordinate convention is preferred.
REFERENCE_PIXEL_XY = (195.0, 193.0)
PIXEL_SCALE_ARCSEC = 0.01
DISTANCE_PC = 2600.0
X_AXIS_SIGN = 1.0
Y_AXIS_SIGN = 1.0

# Optional selection applied only to the fitting NPZ.  The defaults reproduce
# the notebook's peak_y < 300 cut.  Each bound is (minimum, maximum), and None
# disables that side of the bound.
FIT_X_PIXEL_BOUNDS = (None, None)
FIT_Y_PIXEL_BOUNDS = (None, 300.0)

# Outputs are grouped by molecule under data/<MOLNAME>/, and every filename
# contains MOLNAME.  The directory is reused when it already exists.
OUTPUT_ROOT = SCRIPT_DIR / "data"
OUTPUT_DIR = OUTPUT_ROOT / MOLNAME
OUTPUT_PEAK_NPZ = OUTPUT_DIR / f"{MOLNAME}-prepared-peaks.npz"
OUTPUT_MOM0_FITS = OUTPUT_DIR / f"{MOLNAME}-prepared-mom0.fits"
OUTPUT_MOM1_FITS = OUTPUT_DIR / f"{MOLNAME}-prepared-mom1.fits"
OUTPUT_FIGURE = OUTPUT_DIR / f"{MOLNAME}-prepared-peaks-on-mom01.png"
OUTPUT_SPECTRUM_FIGURE = OUTPUT_DIR / f"{MOLNAME}-aoi-average-spectrum.png"
SAVE_MOMENT_FITS = True
OVERWRITE_OUTPUTS = False

# Plot appearance.  PLOT_CENTER_XY=None displays the full maps.
FIGURE_SIZE = (8.0, 6.0)
SPECTRUM_FIGURE_SIZE = (8.0, 4.5)
PLOT_CENTER_XY = (190.0, 210.0)
PLOT_SIZE_XY = (100.0, 140.0)
MOM0_CMAP = "magma"
VELOCITY_CMAP = "RdBu_r"
MARKER_SIZE_RANGE = (15.0, 60.0)
MARKER_FLUX_PERCENTILES = (5.0, 95.0)
FIGURE_DPI = 200


def _refresh_output_paths() -> None:
    """Rebuild molecule-dependent paths after configuration/CLI overrides."""
    global OUTPUT_DIR, OUTPUT_PEAK_NPZ, OUTPUT_MOM0_FITS, OUTPUT_MOM1_FITS
    global OUTPUT_FIGURE, OUTPUT_SPECTRUM_FIGURE

    OUTPUT_DIR = Path(OUTPUT_ROOT) / MOLNAME
    OUTPUT_PEAK_NPZ = OUTPUT_DIR / f"{MOLNAME}-prepared-peaks.npz"
    OUTPUT_MOM0_FITS = OUTPUT_DIR / f"{MOLNAME}-prepared-mom0.fits"
    OUTPUT_MOM1_FITS = OUTPUT_DIR / f"{MOLNAME}-prepared-mom1.fits"
    OUTPUT_FIGURE = OUTPUT_DIR / f"{MOLNAME}-prepared-peaks-on-mom01.png"
    OUTPUT_SPECTRUM_FIGURE = OUTPUT_DIR / f"{MOLNAME}-aoi-average-spectrum.png"


def _read_header(cube_path: Path) -> fits.Header:
    """Read the configured metadata header without retaining an open HDU."""
    path = cube_path if HEADER_FITS_PATH is None else Path(HEADER_FITS_PATH)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    with fits.open(path, memmap=True) as hdul:
        return hdul[HEADER_HDU if HEADER_FITS_PATH is not None else CUBE_HDU].header.copy()


def _beam_and_pixel_metadata(header: fits.Header, data_unit: u.UnitBase):
    """Build PPStatistic metadata and return the beam size in pixels."""
    missing = [key for key in ("BMAJ", "BMIN") if key not in header]
    if missing:
        raise KeyError(f"Header is missing required beam keyword(s): {', '.join(missing)}")

    celestial_wcs = WCS(header).celestial
    pixel_area_deg2 = abs(float(proj_plane_pixel_area(celestial_wcs)))
    if not np.isfinite(pixel_area_deg2) or pixel_area_deg2 <= 0:
        raise ValueError("Could not determine a positive celestial pixel area from the header")

    beam_major_deg = float(header["BMAJ"])
    beam_minor_deg = float(header["BMIN"])
    beam_area_deg2 = np.pi * beam_major_deg * beam_minor_deg / (4.0 * np.log(2.0))
    beam_npix = beam_area_deg2 / pixel_area_deg2

    metadata = {
        "data_unit": data_unit,
        "spatial_scale": np.sqrt(pixel_area_deg2) * u.deg,
        "beam_major": beam_major_deg * u.deg,
        "beam_minor": beam_minor_deg * u.deg,
    }
    return celestial_wcs, metadata, beam_npix


def _channel_limits(n_channels: int) -> tuple[int, int]:
    start = 0 if CHANNEL_START is None else int(CHANNEL_START)
    end = n_channels - 1 if CHANNEL_END is None else int(CHANNEL_END)
    if start < 0 or end >= n_channels or start > end:
        raise ValueError(
            f"Invalid inclusive channel range [{start}, {end}] for {n_channels} channels"
        )
    return start, end


def _bounded(values: np.ndarray, bounds) -> np.ndarray:
    lower, upper = bounds
    keep = np.ones(values.shape, dtype=bool)
    if lower is not None:
        keep &= values >= lower
    if upper is not None:
        keep &= values < upper
    return keep


def _marker_sizes(flux: np.ndarray) -> np.ndarray:
    if flux.size == 0:
        return np.empty(0, dtype=float)
    low, high = np.percentile(flux, MARKER_FLUX_PERCENTILES)
    size_min, size_max = MARKER_SIZE_RANGE
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(flux.shape, 0.5 * (size_min + size_max))
    scaled = np.clip((flux - low) / (high - low), 0.0, 1.0)
    return size_min + (size_max - size_min) * scaled


def _safe_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(
            f"Refusing to overwrite {path}. Change the output path or set "
            "OVERWRITE_OUTPUTS=True."
        )


def _preflight_outputs() -> None:
    """Fail before cube processing if a configured output is protected."""
    if not MOLNAME.strip() or Path(MOLNAME).name != MOLNAME:
        raise ValueError("MOLNAME must be a non-empty directory/file name without slashes")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT_PEAK_NPZ, OUTPUT_FIGURE, OUTPUT_SPECTRUM_FIGURE]
    if SAVE_MOMENT_FITS:
        paths.extend([OUTPUT_MOM0_FITS, OUTPUT_MOM1_FITS])
    for path in paths:
        _safe_output(path)


def _validate_configuration() -> None:
    """Reject invalid top-level or CLI-overridden scientific parameters."""
    if not np.isfinite(RMS) or RMS <= 0:
        raise ValueError("RMS must be finite and positive")
    if not REST_FREQUENCY.unit.is_equivalent(u.Hz) or REST_FREQUENCY <= 0 * u.Hz:
        raise ValueError("REST_FREQUENCY must be a positive frequency")
    for name, value in (
        ("MOMENT_MASK_SIGMA", MOMENT_MASK_SIGMA),
        ("DENDRO_MIN_VALUE_SIGMA", DENDRO_MIN_VALUE_SIGMA),
        ("DENDRO_MIN_DELTA_SIGMA", DENDRO_MIN_DELTA_SIGMA),
        ("DENDRO_MIN_BEAM_FRACTION", DENDRO_MIN_BEAM_FRACTION),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if DENDRO_MIN_NPIX is not None and DENDRO_MIN_NPIX <= 0:
        raise ValueError("DENDRO_MIN_NPIX must be positive or None")
    if not np.isfinite(PIXEL_SCALE_ARCSEC) or PIXEL_SCALE_ARCSEC <= 0:
        raise ValueError("PIXEL_SCALE_ARCSEC must be finite and positive")
    if not np.isfinite(DISTANCE_PC) or DISTANCE_PC <= 0:
        raise ValueError("DISTANCE_PC must be finite and positive")


def extract_peaks(cube, velocities, metadata, min_npix):
    """Run a 2-D dendrogram independently on every configured channel map."""
    channel_start, channel_end = _channel_limits(cube.shape[0])
    rows = []
    flux_unit = None

    for channel in range(channel_start, channel_end + 1):
        channel_map = np.asarray(cube.unmasked_data[channel].value, dtype=float)
        dendrogram = Dendrogram.compute(
            channel_map,
            min_value=DENDRO_MIN_VALUE_SIGMA * RMS,
            min_delta=DENDRO_MIN_DELTA_SIGMA * RMS,
            min_npix=min_npix,
            verbose=False,
        )
        velocity = float(velocities[channel].to_value(u.km / u.s))
        print(
            f"channel {channel:4d}: v={velocity:9.3f} km/s, "
            f"leaves={len(dendrogram.leaves)}"
        )

        for leaf in dendrogram.leaves:
            statistic = PPStatistic(leaf, metadata=metadata)
            flux_unit = statistic.flux.unit
            rows.append(
                (
                    float(statistic.x_cen.value),
                    float(statistic.y_cen.value),
                    velocity,
                    float(statistic.flux.value),
                    channel,
                )
            )

    if not rows:
        raise RuntimeError(
            "No dendrogram leaves were found. Check RMS, channel range, and "
            "dendrogram thresholds."
        )
    return np.asarray(rows, dtype=float), str(flux_unit)


def save_peak_table(
    rows: np.ndarray,
    celestial_wcs: WCS,
    flux_unit: str,
    cube_path: Path,
) -> int:
    """Convert selected centroids to fitting coordinates and save an NPZ."""
    x_pix, y_pix, velocity, flux, channel_float = rows.T
    selection = _bounded(x_pix, FIT_X_PIXEL_BOUNDS)
    selection &= _bounded(y_pix, FIT_Y_PIXEL_BOUNDS)

    x_sel = x_pix[selection]
    y_sel = y_pix[selection]
    au_per_pixel = PIXEL_SCALE_ARCSEC * DISTANCE_PC
    x_au = X_AXIS_SIGN * (x_sel - REFERENCE_PIXEL_XY[0]) * au_per_pixel
    y_au = Y_AXIS_SIGN * (y_sel - REFERENCE_PIXEL_XY[1]) * au_per_pixel
    ra_deg, dec_deg = celestial_wcs.pixel_to_world_values(x_sel, y_sel)

    _safe_output(OUTPUT_PEAK_NPZ)
    np.savez_compressed(
        OUTPUT_PEAK_NPZ,
        # Keys consumed by the existing fitting scripts:
        x=x_au,
        y=y_au,
        v=velocity[selection],
        flux=flux[selection],
        # Traceability fields:
        x_pix=x_sel,
        y_pix=y_sel,
        channel=channel_float[selection].astype(int),
        ra_deg=np.asarray(ra_deg),
        dec_deg=np.asarray(dec_deg),
        rms=np.asarray(RMS),
        reference_pixel_xy=np.asarray(REFERENCE_PIXEL_XY),
        pixel_scale_arcsec=np.asarray(PIXEL_SCALE_ARCSEC),
        distance_pc=np.asarray(DISTANCE_PC),
        molecule_name=np.asarray(MOLNAME),
        velocity_unit=np.asarray("km/s"),
        flux_unit=np.asarray(flux_unit),
        cube_path=np.asarray(str(cube_path)),
    )
    return int(selection.sum())


def plot_moments(
    moment0,
    moment1,
    rows: np.ndarray,
    celestial_wcs: WCS,
    velocity_limits: tuple[float, float],
) -> None:
    """Plot every extracted peak over the moment-0 and moment-1 maps."""
    x_pix, y_pix, velocity, flux, _ = rows.T
    marker_sizes = _marker_sizes(flux)
    velocity_norm = colors.Normalize(vmin=velocity_limits[0], vmax=velocity_limits[1])
    moment0_norm = ImageNormalize(
        moment0.value,
        interval=PercentileInterval(99.5),
        clip=True,
    )

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection=celestial_wcs)
    ax1 = fig.add_subplot(1, 2, 2, projection=celestial_wcs)

    im0 = ax0.imshow(moment0.value, origin="lower", cmap=MOM0_CMAP, norm=moment0_norm)
    ax0.scatter(
        x_pix,
        y_pix,
        c="white",
        s=marker_sizes,
        edgecolors="black",
        linewidths=0.7,
    )
    im1 = ax1.imshow(
        moment1.value,
        origin="lower",
        cmap=VELOCITY_CMAP,
        norm=velocity_norm,
    )
    ax1.scatter(
        x_pix,
        y_pix,
        c=velocity,
        s=marker_sizes,
        cmap=VELOCITY_CMAP,
        norm=velocity_norm,
        edgecolors="black",
        linewidths=0.7,
    )

    for axis in (ax0, ax1):
        axis.set_xlabel("RA")
        axis.set_facecolor("whitesmoke")
        axis.coords[0].set_ticks(number=3)
        axis.coords[0].set_ticklabel(size=9)
        axis.coords[1].set_ticklabel(size=9)
    ax0.set_ylabel("Dec")
    ax1.set_ylabel("")
    ax1.coords[1].set_ticklabel_visible(False)

    if PLOT_CENTER_XY is not None:
        half_x = PLOT_SIZE_XY[0] / 2.0
        half_y = PLOT_SIZE_XY[1] / 2.0
        for axis in (ax0, ax1):
            axis.set_xlim(PLOT_CENTER_XY[0] - half_x, PLOT_CENTER_XY[0] + half_x)
            axis.set_ylim(PLOT_CENTER_XY[1] - half_y, PLOT_CENTER_XY[1] + half_y)

    cbar0 = fig.colorbar(im0, ax=ax0, orientation="horizontal", pad=0.08, shrink=0.75)
    cbar0.locator = MaxNLocator(nbins=4)
    cbar0.update_ticks()
    cbar0.set_label(str(moment0.unit))
    cbar1 = fig.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.08, shrink=0.75)
    cbar1.locator = MaxNLocator(nbins=5)
    cbar1.update_ticks()
    cbar1.set_label("km/s")

    _safe_output(OUTPUT_FIGURE)
    fig.savefig(OUTPUT_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_aoi_average_spectrum(cube) -> None:
    """Plot the spatially averaged spectrum inside the configured AOI."""
    n_channels, ny, nx = cube.shape
    if PLOT_CENTER_XY is None:
        x_min, x_max = 0, nx
        y_min, y_max = 0, ny
        aoi_description = f"full map ({nx} x {ny} pixels)"
    else:
        if PLOT_SIZE_XY[0] <= 0 or PLOT_SIZE_XY[1] <= 0:
            raise ValueError("PLOT_SIZE_XY values must both be positive")
        half_x = PLOT_SIZE_XY[0] / 2.0
        half_y = PLOT_SIZE_XY[1] / 2.0
        x_min = max(0, int(np.floor(PLOT_CENTER_XY[0] - half_x)))
        x_max = min(nx, int(np.ceil(PLOT_CENTER_XY[0] + half_x)))
        y_min = max(0, int(np.floor(PLOT_CENTER_XY[1] - half_y)))
        y_max = min(ny, int(np.ceil(PLOT_CENTER_XY[1] + half_y)))
        aoi_description = (
            f"center=({PLOT_CENTER_XY[0]:g}, {PLOT_CENTER_XY[1]:g}), "
            f"size=({PLOT_SIZE_XY[0]:g}, {PLOT_SIZE_XY[1]:g}) pixels"
        )

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("The Area of Interest does not overlap the cube")

    aoi_data = np.asarray(
        cube.unmasked_data[:, y_min:y_max, x_min:x_max].value,
        dtype=float,
    )
    finite = np.isfinite(aoi_data)
    valid_count = finite.sum(axis=(1, 2))
    spectrum_sum = np.where(finite, aoi_data, 0.0).sum(axis=(1, 2))
    mean_spectrum = np.divide(
        spectrum_sum,
        valid_count,
        out=np.full(n_channels, np.nan, dtype=float),
        where=valid_count > 0,
    )
    velocities = cube.spectral_axis.to_value(u.km / u.s)
    channel_start, channel_end = _channel_limits(n_channels)

    # Use channel-bin edges for the highlighted interval, including both the
    # configured start and end channels even when the spectral axis descends.
    if n_channels == 1:
        selected_edges = (velocities[0] - 0.5, velocities[0] + 0.5)
    else:
        start_edge = (
            velocities[channel_start]
            - 0.5 * (velocities[channel_start + 1] - velocities[channel_start])
            if channel_start == 0
            else 0.5 * (velocities[channel_start - 1] + velocities[channel_start])
        )
        end_edge = (
            velocities[channel_end]
            + 0.5 * (velocities[channel_end] - velocities[channel_end - 1])
            if channel_end == n_channels - 1
            else 0.5 * (velocities[channel_end] + velocities[channel_end + 1])
        )
        selected_edges = tuple(sorted((start_edge, end_edge)))

    selected = slice(channel_start, channel_end + 1)
    fig, ax = plt.subplots(figsize=SPECTRUM_FIGURE_SIZE, constrained_layout=True)
    ax.plot(velocities, mean_spectrum, color="black", linewidth=1.2, label="AOI mean")
    ax.axvspan(
        selected_edges[0],
        selected_edges[1],
        color="tab:red",
        alpha=0.18,
        label=f"selected channels {channel_start}-{channel_end}",
    )
    ax.scatter(
        velocities[selected],
        mean_spectrum[selected],
        color="tab:red",
        s=22,
        zorder=3,
    )
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LSR velocity (km/s)")
    ax.set_ylabel(f"Mean intensity ({cube.unit})")
    ax.set_title(f"{MOLNAME} AOI average spectrum\n{aoi_description}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    _safe_output(OUTPUT_SPECTRUM_FIGURE)
    fig.savefig(OUTPUT_SPECTRUM_FIGURE, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _optional_float(value: str) -> float | None:
    """Parse a float or the word 'none' for optional pixel bounds."""
    if value.lower() == "none":
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute masked moment maps and channel-by-channel astrodendro "
            "peaks for PPV fitting. Supplied CLI values override the top-level "
            "configuration."
        ),
        epilog=(
            "Example: %(prog)s cube.fits --molname HNCO --restfreq-ghz "
            "219.798274 --rms 8.8e-4 --all-channels\n\n"
            "Run without arguments to use only the configuration block."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_cube",
        type=Path,
        nargs="?",
        help="input FITS cube (default: configured CUBE_FITS_PATH)",
    )
    parser.add_argument("--molname", help="molecule name used for directory and filenames")
    parser.add_argument("--restfreq-ghz", type=float, metavar="GHZ")
    parser.add_argument(
        "--rms",
        type=float,
        help="RMS in the cube BUNIT",
    )
    parser.add_argument(
        "--velocity-convention",
        choices=("radio", "optical", "relativistic"),
    )
    parser.add_argument("--cube-hdu", type=int)
    header_group = parser.add_mutually_exclusive_group()
    header_group.add_argument("--header-fits", type=Path)
    header_group.add_argument(
        "--use-cube-header",
        action="store_true",
        help="use the input cube header for WCS/beam metadata",
    )
    parser.add_argument("--header-hdu", type=int)
    parser.add_argument("--moment-mask-sigma", type=float)

    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument(
        "--channel-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="inclusive channel range",
    )
    channel_group.add_argument(
        "--all-channels",
        action="store_true",
        help="process every channel in the input cube",
    )
    parser.add_argument("--dendro-min-value-sigma", type=float)
    parser.add_argument("--dendro-min-delta-sigma", type=float)
    min_npix_group = parser.add_mutually_exclusive_group()
    min_npix_group.add_argument("--dendro-min-npix", type=int)
    min_npix_group.add_argument(
        "--auto-min-npix",
        action="store_true",
        help="derive min_npix from the FITS beam and pixel area",
    )
    parser.add_argument("--dendro-min-beam-fraction", type=float)

    parser.add_argument(
        "--reference-pixel",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
    )
    parser.add_argument("--pixel-scale-arcsec", type=float)
    parser.add_argument("--distance-pc", type=float)
    parser.add_argument(
        "--axis-signs",
        type=float,
        nargs=2,
        metavar=("X_SIGN", "Y_SIGN"),
    )
    parser.add_argument(
        "--fit-x-pixel-bounds",
        type=_optional_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="use 'none' for an open bound",
    )
    parser.add_argument(
        "--fit-y-pixel-bounds",
        type=_optional_float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="use 'none' for an open bound",
    )

    aoi_group = parser.add_mutually_exclusive_group()
    aoi_group.add_argument(
        "--plot-center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="AOI center in pixels",
    )
    aoi_group.add_argument(
        "--full-map",
        action="store_true",
        help="display and average the full spatial map",
    )
    parser.add_argument(
        "--plot-size",
        type=float,
        nargs=2,
        metavar=("X_SIZE", "Y_SIZE"),
        help="AOI size in pixels",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="parent output directory; MOLNAME is created below it",
    )
    parser.add_argument(
        "--save-moment-fits",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace existing outputs",
    )
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global CUBE_FITS_PATH, CUBE_HDU, HEADER_FITS_PATH, HEADER_HDU
    global RMS, REST_FREQUENCY, MOLNAME, VELOCITY_CONVENTION
    global MOMENT_MASK_SIGMA, CHANNEL_START, CHANNEL_END
    global DENDRO_MIN_VALUE_SIGMA, DENDRO_MIN_DELTA_SIGMA
    global DENDRO_MIN_NPIX, DENDRO_MIN_BEAM_FRACTION
    global REFERENCE_PIXEL_XY, PIXEL_SCALE_ARCSEC, DISTANCE_PC
    global X_AXIS_SIGN, Y_AXIS_SIGN, FIT_X_PIXEL_BOUNDS, FIT_Y_PIXEL_BOUNDS
    global PLOT_CENTER_XY, PLOT_SIZE_XY, OUTPUT_ROOT
    global SAVE_MOMENT_FITS, OVERWRITE_OUTPUTS

    if args.input_cube is not None:
        CUBE_FITS_PATH = args.input_cube.expanduser().resolve()
    if args.cube_hdu is not None:
        CUBE_HDU = args.cube_hdu
    if args.header_fits is not None:
        HEADER_FITS_PATH = args.header_fits.expanduser().resolve()
    elif args.use_cube_header:
        HEADER_FITS_PATH = None
    if args.header_hdu is not None:
        HEADER_HDU = args.header_hdu
    if args.rms is not None:
        RMS = args.rms
    if args.restfreq_ghz is not None:
        REST_FREQUENCY = args.restfreq_ghz * u.GHz
    if args.molname is not None:
        MOLNAME = args.molname
    if args.velocity_convention is not None:
        VELOCITY_CONVENTION = args.velocity_convention
    if args.moment_mask_sigma is not None:
        MOMENT_MASK_SIGMA = args.moment_mask_sigma

    if args.channel_range is not None:
        CHANNEL_START, CHANNEL_END = args.channel_range
    elif args.all_channels:
        CHANNEL_START, CHANNEL_END = None, None
    if args.dendro_min_value_sigma is not None:
        DENDRO_MIN_VALUE_SIGMA = args.dendro_min_value_sigma
    if args.dendro_min_delta_sigma is not None:
        DENDRO_MIN_DELTA_SIGMA = args.dendro_min_delta_sigma
    if args.dendro_min_npix is not None:
        DENDRO_MIN_NPIX = args.dendro_min_npix
    elif args.auto_min_npix:
        DENDRO_MIN_NPIX = None
    if args.dendro_min_beam_fraction is not None:
        DENDRO_MIN_BEAM_FRACTION = args.dendro_min_beam_fraction

    if args.reference_pixel is not None:
        REFERENCE_PIXEL_XY = tuple(args.reference_pixel)
    if args.pixel_scale_arcsec is not None:
        PIXEL_SCALE_ARCSEC = args.pixel_scale_arcsec
    if args.distance_pc is not None:
        DISTANCE_PC = args.distance_pc
    if args.axis_signs is not None:
        X_AXIS_SIGN, Y_AXIS_SIGN = args.axis_signs
    if args.fit_x_pixel_bounds is not None:
        FIT_X_PIXEL_BOUNDS = tuple(args.fit_x_pixel_bounds)
    if args.fit_y_pixel_bounds is not None:
        FIT_Y_PIXEL_BOUNDS = tuple(args.fit_y_pixel_bounds)

    if args.plot_center is not None:
        PLOT_CENTER_XY = tuple(args.plot_center)
    elif args.full_map:
        PLOT_CENTER_XY = None
    if args.plot_size is not None:
        PLOT_SIZE_XY = tuple(args.plot_size)
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root.expanduser().resolve()
    if args.save_moment_fits is not None:
        SAVE_MOMENT_FITS = args.save_moment_fits
    if args.overwrite is not None:
        OVERWRITE_OUTPUTS = args.overwrite

    _refresh_output_paths()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_cli_overrides(args)
    _validate_configuration()
    _preflight_outputs()
    cube_path = Path(CUBE_FITS_PATH)
    if not cube_path.is_absolute():
        cube_path = SCRIPT_DIR / cube_path
    cube_path = cube_path.resolve()
    header = _read_header(cube_path)
    cube = SpectralCube.read(cube_path, hdu=CUBE_HDU)
    cube = cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention=VELOCITY_CONVENTION,
        rest_value=REST_FREQUENCY,
    )
    if cube.unit == u.dimensionless_unscaled:
        raise ValueError("The cube header must define BUNIT so RMS has a physical unit")

    celestial_wcs, metadata, beam_npix = _beam_and_pixel_metadata(header, cube.unit)
    min_npix = (
        int(DENDRO_MIN_NPIX)
        if DENDRO_MIN_NPIX is not None
        else max(1, int(np.ceil(DENDRO_MIN_BEAM_FRACTION * beam_npix)))
    )
    print(f"Using min_npix={min_npix} (header beam area: {beam_npix:.2f} pixels)")

    rms_quantity = RMS * cube.unit
    masked_cube = cube.with_mask(cube > MOMENT_MASK_SIGMA * rms_quantity)
    moment0 = masked_cube.moment(order=0)
    moment1 = masked_cube.moment(order=1)

    rows, flux_unit = extract_peaks(cube, cube.spectral_axis, metadata, min_npix)
    selected_count = save_peak_table(rows, celestial_wcs, flux_unit, cube_path)

    if SAVE_MOMENT_FITS:
        _safe_output(OUTPUT_MOM0_FITS)
        _safe_output(OUTPUT_MOM1_FITS)
        moment0.write(OUTPUT_MOM0_FITS, overwrite=OVERWRITE_OUTPUTS)
        moment1.write(OUTPUT_MOM1_FITS, overwrite=OVERWRITE_OUTPUTS)
    channel_start, channel_end = _channel_limits(cube.shape[0])
    velocity_limits = tuple(
        sorted(
            (
                float(cube.spectral_axis[channel_start].to_value(u.km / u.s)),
                float(cube.spectral_axis[channel_end].to_value(u.km / u.s)),
            )
        )
    )
    plot_moments(moment0, moment1, rows, celestial_wcs, velocity_limits)
    plot_aoi_average_spectrum(cube)

    print(f"Extracted {len(rows)} peaks; saved {selected_count} fitting points")
    print(f"Peak table: {OUTPUT_PEAK_NPZ}")
    print(f"Overlay plot: {OUTPUT_FIGURE}")
    print(f"AOI spectrum: {OUTPUT_SPECTRUM_FIGURE}")


if __name__ == "__main__":
    main()
