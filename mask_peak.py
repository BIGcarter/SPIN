from __future__ import annotations

"""Optionally select prepared peak points inside DS9 regions and plot them.

Run without arguments to use the configuration block, or pass CLI arguments
for agent/automation use. CLI values override the corresponding configuration.
When no mask-region path is configured, every input peak is retained.
"""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Affine2D
from regions import PixCoord, PixelRegion, Regions


# =============================================================================
# Human-operated configuration mode
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PEAK_NPZ_PATH = SCRIPT_DIR / "data/HNCO/HNCO-prepared-peaks.npz"
MOM0_FITS_PATH = SCRIPT_DIR / "data/HNCO/HNCO-prepared-mom0.fits"
# Set to None to keep every input peak (no DS9-region filtering).
MASK_REGION_PATH = None
MOM0_HDU = 0
REGION_FORMAT = "ds9"

# None derives output names beside PEAK_NPZ_PATH using the region filename.
OUTPUT_NPZ_PATH = None
OUTPUT_FIGURE_PATH = None
VSYS = None  # km/s; set to save an additional velocity-relative NPZ.
OUTPUT_RELATIVE_NPZ_PATH = None
# Set both to save an additional NPZ whose x/y coordinates are recomputed in AU.
DISTANCE_PC = None
CENTER_XY = None
PIXEL_SCALE_ARCSEC = None  # Optional override; otherwise derive it from the WCS.
OUTPUT_AU_NPZ_PATH = None
OUTPUT_AU_RELATIVE_NPZ_PATH = None
OVERWRITE_OUTPUTS = False

FIGURE_SIZE = (7.0, 6.0)
FIGURE_DPI = 200
MOM0_CMAP = "magma"
VELOCITY_CMAP = "RdBu_r"
MARKER_SIZE = 48.0
SELECTED_EDGE_COLOR = "black"
UNSELECTED_EDGE_COLOR = "0.65"
MASK_EDGE_COLOR = "cyan"
MASK_FACE_COLOR = "cyan"
MASK_FILL_ALPHA = 0.10


POINT_KEYS = {
    "x",
    "y",
    "v",
    "flux",
    "x_pix",
    "y_pix",
    "channel",
    "ra_deg",
    "dec_deg",
}


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _derived_output_paths(
    peak_path: Path,
    region_path: Path | None,
) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    stem = (
        f"{peak_path.stem}-{region_path.stem}-masked"
        if region_path is not None
        else f"{peak_path.stem}-unmasked"
    )
    output_npz = (
        peak_path.with_name(f"{stem}.npz")
        if OUTPUT_NPZ_PATH is None
        else _resolve_path(OUTPUT_NPZ_PATH)
    )
    output_figure = (
        peak_path.with_name(f"{stem}-overlay.png")
        if OUTPUT_FIGURE_PATH is None
        else _resolve_path(OUTPUT_FIGURE_PATH)
    )
    if VSYS is None:
        output_relative_npz = None
    else:
        output_relative_npz = (
            output_npz.with_name(f"{output_npz.stem}-relative-vsys.npz")
            if OUTPUT_RELATIVE_NPZ_PATH is None
            else _resolve_path(OUTPUT_RELATIVE_NPZ_PATH)
        )
    if DISTANCE_PC is None or CENTER_XY is None:
        output_au_npz = None
    else:
        output_au_npz = (
            output_npz.with_name(f"{output_npz.stem}-au.npz")
            if OUTPUT_AU_NPZ_PATH is None
            else _resolve_path(OUTPUT_AU_NPZ_PATH)
        )
    if output_au_npz is None or VSYS is None:
        output_au_relative_npz = None
    else:
        output_au_relative_npz = (
            output_au_npz.with_name(f"{output_au_npz.stem}-relative-vsys.npz")
            if OUTPUT_AU_RELATIVE_NPZ_PATH is None
            else _resolve_path(OUTPUT_AU_RELATIVE_NPZ_PATH)
        )
    return (
        output_npz,
        output_figure,
        output_relative_npz,
        output_au_npz,
        output_au_relative_npz,
    )


def _safe_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(
            f"Refusing to overwrite {path}. Choose another output path or "
            "pass --overwrite."
        )


def _load_mom0(path: Path) -> tuple[np.ndarray, WCS, str]:
    with fits.open(path, memmap=True) as hdul:
        data = np.squeeze(np.asarray(hdul[MOM0_HDU].data, dtype=float))
        header = hdul[MOM0_HDU].header.copy()
    if data.ndim != 2:
        raise ValueError(f"Moment-0 data must be 2-D after squeeze; got shape {data.shape}")
    return data, WCS(header).celestial, str(header.get("BUNIT", ""))


def _load_peaks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        peaks = {key: np.array(archive[key]) for key in archive.files}
    missing = [key for key in ("x_pix", "y_pix", "v") if key not in peaks]
    if missing:
        raise KeyError(f"Peak NPZ is missing required key(s): {', '.join(missing)}")
    n_peaks = len(peaks["x_pix"])
    for key in ("y_pix", "v"):
        if peaks[key].ndim != 1 or len(peaks[key]) != n_peaks:
            raise ValueError(f"Peak field {key!r} must be a length-{n_peaks} 1-D array")
    if not all(np.all(np.isfinite(peaks[key])) for key in ("x_pix", "y_pix", "v")):
        raise ValueError("x_pix, y_pix, and v must contain only finite values")
    return peaks


def _pixel_regions(region_path: Path, celestial_wcs: WCS) -> list[PixelRegion]:
    regions = Regions.read(region_path, format=REGION_FORMAT)
    if not regions:
        raise ValueError(f"No regions were found in {region_path}")
    pixel_regions = []
    for region in regions:
        if isinstance(region, PixelRegion):
            pixel_regions.append(region)
        elif hasattr(region, "to_pixel"):
            pixel_regions.append(region.to_pixel(celestial_wcs))
        else:
            raise TypeError(f"Unsupported region type: {type(region).__name__}")
    return pixel_regions


def _selection_mask(
    pixel_regions: list[PixelRegion],
    x_pix: np.ndarray,
    y_pix: np.ndarray,
) -> np.ndarray:
    coordinates = PixCoord(x=x_pix, y=y_pix)
    included = [region for region in pixel_regions if region.meta.get("include", 1)]
    excluded = [region for region in pixel_regions if not region.meta.get("include", 1)]

    if included:
        selected = np.zeros(x_pix.shape, dtype=bool)
        for region in included:
            selected |= np.asarray(region.contains(coordinates), dtype=bool)
    else:
        selected = np.ones(x_pix.shape, dtype=bool)
    for region in excluded:
        selected &= ~np.asarray(region.contains(coordinates), dtype=bool)
    return selected


def _save_selected_peaks(
    peaks: dict[str, np.ndarray],
    selected: np.ndarray,
    region_path: Path | None,
    output_path: Path,
    velocity_offset: float | None = None,
    au_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    center_xy: tuple[float, float] | None = None,
    pixel_scale_arcsec_xy: tuple[float, float] | None = None,
    distance_pc: float | None = None,
) -> None:
    n_peaks = len(selected)
    output = {}
    for key, value in peaks.items():
        if key in POINT_KEYS:
            if value.ndim != 1 or len(value) != n_peaks:
                raise ValueError(f"Point field {key!r} is not a length-{n_peaks} array")
            output[key] = value[selected]
        else:
            output[key] = value
    output["source_peak_npz"] = np.asarray(str(_resolve_path(PEAK_NPZ_PATH)))
    output["selection_mode"] = np.asarray(
        "region-mask" if region_path is not None else "all-peaks-unmasked"
    )
    if region_path is not None:
        output["mask_region_path"] = np.asarray(str(region_path))
    output["source_peak_index"] = np.flatnonzero(selected)
    output["source_peak_count"] = np.asarray(n_peaks)
    if au_coordinates is not None:
        if center_xy is None or pixel_scale_arcsec_xy is None or distance_pc is None:
            raise ValueError("AU coordinates require center, pixel scale, and distance")
        if "x" in output:
            output["x_input"] = np.asarray(output["x"])
        if "y" in output:
            output["y_input"] = np.asarray(output["y"])
        for key in ("reference_pixel_xy", "pixel_scale_arcsec", "distance_pc"):
            if key in output:
                output[f"input_{key}"] = np.asarray(output[key])
        output["x"] = np.asarray(au_coordinates[0])[selected]
        output["y"] = np.asarray(au_coordinates[1])[selected]
        output["reference_pixel_xy"] = np.asarray(center_xy, dtype=float)
        output["center_pixel_xy"] = np.asarray(center_xy, dtype=float)
        output["pixel_scale_arcsec_xy"] = np.asarray(
            pixel_scale_arcsec_xy,
            dtype=float,
        )
        output["pixel_scale_arcsec"] = np.asarray(
            pixel_scale_arcsec_xy[0]
            if np.isclose(pixel_scale_arcsec_xy[0], pixel_scale_arcsec_xy[1])
            else pixel_scale_arcsec_xy
        )
        output["distance_pc"] = np.asarray(distance_pc)
        output["spatial_unit"] = np.asarray("AU")
        output["au_coordinate_definition"] = np.asarray(
            "x/y = (x_pix/y_pix - center_x/center_y) * "
            "pixel_scale_arcsec_x/y * distance_pc"
        )
    if velocity_offset is not None:
        output["v_absolute"] = np.asarray(output["v"])
        output["v"] = output["v_absolute"] - velocity_offset
        output["vsys"] = np.asarray(velocity_offset)
        output["velocity_reference"] = np.asarray("v_relative = v_absolute - vsys")

    _safe_output(output_path)
    np.savez_compressed(output_path, **output)


def _pixel_coordinates_to_au(
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    celestial_wcs: WCS,
    pixel_scale_arcsec_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Convert pixel offsets from CENTER_XY to AU in pixel-axis directions."""
    if pixel_scale_arcsec_override is None:
        pixel_scale_arcsec = np.asarray(
            proj_plane_pixel_scales(celestial_wcs),
            dtype=float,
        ) * 3600.0
    else:
        pixel_scale_arcsec = np.full(2, pixel_scale_arcsec_override, dtype=float)
    if pixel_scale_arcsec.shape != (2,) or not np.all(np.isfinite(pixel_scale_arcsec)):
        raise ValueError("Could not derive two finite celestial pixel scales from WCS")
    scale_xy = (float(pixel_scale_arcsec[0]), float(pixel_scale_arcsec[1]))
    x_au = (x_pix - CENTER_XY[0]) * scale_xy[0] * DISTANCE_PC
    y_au = (y_pix - CENTER_XY[1]) * scale_xy[1] * DISTANCE_PC
    return x_au, y_au, scale_xy


def _velocity_norm(velocity: np.ndarray) -> colors.Normalize:
    vmin = float(np.min(velocity))
    vmax = float(np.max(velocity))
    if vmax <= vmin:
        padding = max(1.0, abs(vmin) * 0.01)
        vmin -= padding
        vmax += padding
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _plot_overlay(
    mom0: np.ndarray,
    mom0_wcs: WCS,
    mom0_unit: str,
    peaks: dict[str, np.ndarray],
    selected: np.ndarray,
    pixel_regions: list[PixelRegion],
    output_path: Path,
    au_coordinates: tuple[np.ndarray, np.ndarray] | None = None,
    center_xy: tuple[float, float] | None = None,
    pixel_scale_arcsec_xy: tuple[float, float] | None = None,
    distance_pc: float | None = None,
    mask_applied: bool = True,
) -> None:
    x_pix = peaks["x_pix"]
    y_pix = peaks["y_pix"]
    velocity = peaks["v"]
    unselected = ~selected
    velocity_norm = _velocity_norm(velocity)
    mom0_norm = ImageNormalize(mom0, interval=PercentileInterval(99.5), clip=True)

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    physical_coordinates = au_coordinates is not None
    if physical_coordinates:
        if center_xy is None or pixel_scale_arcsec_xy is None or distance_pc is None:
            raise ValueError("Physical-coordinate plotting requires center, scale, and distance")
        x_plot, y_plot = au_coordinates
        x_scale = pixel_scale_arcsec_xy[0] * distance_pc
        y_scale = pixel_scale_arcsec_xy[1] * distance_pc
        height, width = mom0.shape
        extent = (
            (-0.5 - center_xy[0]) * x_scale,
            (width - 0.5 - center_xy[0]) * x_scale,
            (-0.5 - center_xy[1]) * y_scale,
            (height - 0.5 - center_xy[1]) * y_scale,
        )
        ax = fig.add_subplot(1, 1, 1)
        image = ax.imshow(
            mom0,
            origin="lower",
            cmap=MOM0_CMAP,
            norm=mom0_norm,
            extent=extent,
            aspect="equal",
        )
        pixel_to_au = (
            Affine2D()
            .scale(x_scale, y_scale)
            .translate(-center_xy[0] * x_scale, -center_xy[1] * y_scale)
        )
    else:
        x_plot, y_plot = x_pix, y_pix
        ax = fig.add_subplot(1, 1, 1, projection=mom0_wcs)
        image = ax.imshow(mom0, origin="lower", cmap=MOM0_CMAP, norm=mom0_norm)

    ax.scatter(
        x_plot[unselected],
        y_plot[unselected],
        c=velocity[unselected],
        cmap=VELOCITY_CMAP,
        norm=velocity_norm,
        s=MARKER_SIZE,
        edgecolors=UNSELECTED_EDGE_COLOR,
        linewidths=1.2,
        zorder=3,
    )
    velocity_scatter = ax.scatter(
        x_plot[selected],
        y_plot[selected],
        c=velocity[selected],
        cmap=VELOCITY_CMAP,
        norm=velocity_norm,
        s=MARKER_SIZE,
        edgecolors=SELECTED_EDGE_COLOR,
        linewidths=2.0,
        zorder=4,
    )

    for region in pixel_regions:
        artist = region.as_artist(
            origin=(0, 0),
            facecolor=colors.to_rgba(MASK_FACE_COLOR, MASK_FILL_ALPHA),
            edgecolor=MASK_EDGE_COLOR,
            linewidth=2.0,
        )
        if physical_coordinates:
            artist.set_transform(pixel_to_au + ax.transData)
        artist.set_zorder(2)
        ax.add_artist(artist)

    if physical_coordinates:
        ax.scatter(
            0.0,
            0.0,
            marker="*",
            s=260.0,
            c="gold",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        ax.set_xlabel("X offset (AU)")
        ax.set_ylabel("Y offset (AU)")
    else:
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    selection_label = "mask-selected" if mask_applied else "unmasked"
    ax.set_title(
        f"Moment 0 + {selection_label} peaks ({int(selected.sum())}/{len(selected)})"
    )
    ax.set_facecolor("whitesmoke")

    mom0_cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.08, shrink=0.72)
    mom0_cbar.set_ticks(np.linspace(float(mom0_norm.vmin), float(mom0_norm.vmax), 4))
    mom0_cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.3g"))
    mom0_cbar.set_label(mom0_unit or "Moment 0")
    velocity_cbar = fig.colorbar(
        velocity_scatter,
        ax=ax,
        orientation="vertical",
        pad=0.03,
        shrink=0.72,
    )
    velocity_cbar.set_label("Velocity (km/s)")
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=SELECTED_EDGE_COLOR,
            markeredgewidth=2.0,
            label="inside mask",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=UNSELECTED_EDGE_COLOR,
            markeredgewidth=1.2,
            label="outside mask",
        ),
        Line2D([], [], color=MASK_EDGE_COLOR, linewidth=2.0, label="mask region"),
    ]
    if physical_coordinates:
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                markersize=12,
                markerfacecolor="gold",
                markeredgecolor="black",
                label="reference center (0, 0)",
            )
        )
    ax.legend(handles=legend_handles, loc="best", frameon=True)

    _safe_output(output_path)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optionally select prepared channel peaks inside DS9 regions and "
            "plot all peaks over a moment-0 map. Omitting the mask keeps all "
            "input peaks."
        ),
        epilog=(
            "Example: %(prog)s peaks.npz mom0.fits mask.reg\n\n"
            "Run without arguments to use the configuration block."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("peak_npz", type=Path, nargs="?")
    parser.add_argument("mom0_fits", type=Path, nargs="?")
    parser.add_argument("mask_region", type=Path, nargs="?")
    parser.add_argument("--mom0-hdu", type=int)
    parser.add_argument("--region-format", default=None, help="default: configured ds9")
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument("--output-figure", type=Path)
    parser.add_argument(
        "--vsys",
        type=float,
        help="systemic velocity in km/s; also save a relative-velocity NPZ",
    )
    parser.add_argument(
        "--output-relative-npz",
        type=Path,
        help="path for the NPZ with v relative to VSYS",
    )
    parser.add_argument(
        "-d",
        "--distance-pc",
        "--d",
        dest="d",
        type=float,
        metavar="PC",
        help="distance in pc; requires --center-xy and saves an AU-coordinate NPZ",
    )
    parser.add_argument(
        "--center-xy",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="reference center in pixels; requires --d",
    )
    parser.add_argument(
        "--pixel-scale-arcsec",
        type=float,
        help="optional AU-conversion pixel scale; default derives it from the WCS",
    )
    parser.add_argument(
        "--output-au-npz",
        type=Path,
        help="path for the additional NPZ with x/y recomputed in AU",
    )
    parser.add_argument(
        "--output-au-relative-npz",
        type=Path,
        help="path for the NPZ with AU x/y and v relative to VSYS",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global PEAK_NPZ_PATH, MOM0_FITS_PATH, MASK_REGION_PATH, MOM0_HDU
    global REGION_FORMAT, OUTPUT_NPZ_PATH, OUTPUT_FIGURE_PATH
    global VSYS, OUTPUT_RELATIVE_NPZ_PATH, OVERWRITE_OUTPUTS
    global DISTANCE_PC, CENTER_XY, PIXEL_SCALE_ARCSEC, OUTPUT_AU_NPZ_PATH, OUTPUT_AU_RELATIVE_NPZ_PATH

    if args.peak_npz is not None:
        PEAK_NPZ_PATH = args.peak_npz.expanduser().resolve()
    if args.mom0_fits is not None:
        MOM0_FITS_PATH = args.mom0_fits.expanduser().resolve()
    if args.mask_region is not None:
        MASK_REGION_PATH = args.mask_region.expanduser().resolve()
    if args.mom0_hdu is not None:
        MOM0_HDU = args.mom0_hdu
    if args.region_format is not None:
        REGION_FORMAT = args.region_format
    if args.output_npz is not None:
        OUTPUT_NPZ_PATH = args.output_npz.expanduser().resolve()
    if args.output_figure is not None:
        OUTPUT_FIGURE_PATH = args.output_figure.expanduser().resolve()
    if args.vsys is not None:
        if not np.isfinite(args.vsys):
            raise ValueError("--vsys must be finite")
        VSYS = args.vsys
    if args.output_relative_npz is not None:
        OUTPUT_RELATIVE_NPZ_PATH = args.output_relative_npz.expanduser().resolve()
    if args.d is not None:
        if not np.isfinite(args.d) or args.d <= 0:
            raise ValueError("--d must be finite and greater than zero")
        DISTANCE_PC = args.d
    if args.center_xy is not None:
        if not np.all(np.isfinite(args.center_xy)):
            raise ValueError("--center-xy values must be finite")
        CENTER_XY = tuple(args.center_xy)
    if args.pixel_scale_arcsec is not None:
        if not np.isfinite(args.pixel_scale_arcsec) or args.pixel_scale_arcsec <= 0:
            raise ValueError("--pixel-scale-arcsec must be finite and greater than zero")
        PIXEL_SCALE_ARCSEC = args.pixel_scale_arcsec
    if args.output_au_npz is not None:
        OUTPUT_AU_NPZ_PATH = args.output_au_npz.expanduser().resolve()
    if args.output_au_relative_npz is not None:
        OUTPUT_AU_RELATIVE_NPZ_PATH = args.output_au_relative_npz.expanduser().resolve()
    if args.overwrite is not None:
        OVERWRITE_OUTPUTS = args.overwrite


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _apply_cli_overrides(args)

    peak_path = _resolve_path(PEAK_NPZ_PATH)
    mom0_path = _resolve_path(MOM0_FITS_PATH)
    region_path = (
        None if MASK_REGION_PATH is None else _resolve_path(MASK_REGION_PATH)
    )
    required_inputs = [
        ("Peak NPZ", peak_path),
        ("Moment-0 FITS", mom0_path),
    ]
    if region_path is not None:
        required_inputs.append(("Mask region", region_path))
    for label, path in required_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if OUTPUT_RELATIVE_NPZ_PATH is not None and VSYS is None:
        raise ValueError("OUTPUT_RELATIVE_NPZ_PATH requires VSYS/--vsys")
    if (DISTANCE_PC is None) != (CENTER_XY is None):
        raise ValueError("--d and --center-xy must be provided together")
    if OUTPUT_AU_NPZ_PATH is not None and DISTANCE_PC is None:
        raise ValueError("OUTPUT_AU_NPZ_PATH requires --d and --center-xy")
    if OUTPUT_AU_RELATIVE_NPZ_PATH is not None and (DISTANCE_PC is None or VSYS is None):
        raise ValueError("OUTPUT_AU_RELATIVE_NPZ_PATH requires --d, --center-xy, and --vsys")
    (
        output_npz,
        output_figure,
        output_relative_npz,
        output_au_npz,
        output_au_relative_npz,
    ) = _derived_output_paths(
        peak_path,
        region_path,
    )
    _safe_output(output_npz)
    _safe_output(output_figure)
    if output_relative_npz is not None:
        _safe_output(output_relative_npz)
    if output_au_npz is not None:
        _safe_output(output_au_npz)
    if output_au_relative_npz is not None:
        _safe_output(output_au_relative_npz)

    mom0, mom0_wcs, mom0_unit = _load_mom0(mom0_path)
    peaks = _load_peaks(peak_path)
    pixel_regions = (
        _pixel_regions(region_path, mom0_wcs) if region_path is not None else []
    )
    selected = _selection_mask(
        pixel_regions,
        peaks["x_pix"],
        peaks["y_pix"],
    )
    _save_selected_peaks(peaks, selected, region_path, output_npz)
    if output_relative_npz is not None:
        _save_selected_peaks(
            peaks,
            selected,
            region_path,
            output_relative_npz,
            velocity_offset=VSYS,
        )
    au_pixel_scale = None
    if output_au_npz is not None:
        x_au, y_au, au_pixel_scale = _pixel_coordinates_to_au(
            peaks["x_pix"],
            peaks["y_pix"],
            mom0_wcs,
            pixel_scale_arcsec_override=PIXEL_SCALE_ARCSEC,
        )
        _save_selected_peaks(
            peaks,
            selected,
            region_path,
            output_au_npz,
            au_coordinates=(x_au, y_au),
            center_xy=CENTER_XY,
            pixel_scale_arcsec_xy=au_pixel_scale,
            distance_pc=DISTANCE_PC,
        )
    if output_au_relative_npz is not None:
        _save_selected_peaks(
            peaks,
            selected,
            region_path,
            output_au_relative_npz,
            velocity_offset=VSYS,
            au_coordinates=(x_au, y_au),
            center_xy=CENTER_XY,
            pixel_scale_arcsec_xy=au_pixel_scale,
            distance_pc=DISTANCE_PC,
        )
    _plot_overlay(
        mom0,
        mom0_wcs,
        mom0_unit,
        peaks,
        selected,
        pixel_regions,
        output_figure,
        au_coordinates=(x_au, y_au) if output_au_npz is not None else None,
        center_xy=CENTER_XY,
        pixel_scale_arcsec_xy=au_pixel_scale,
        distance_pc=DISTANCE_PC,
        mask_applied=region_path is not None,
    )

    print(f"Input peaks:    {len(selected)}")
    print(f"Selected peaks: {int(selected.sum())}")
    print(f"Selection mode: {'region mask' if region_path is not None else 'all peaks (unmasked)'}")
    print(f"Output NPZ:     {output_npz}")
    if output_relative_npz is not None:
        print(f"Relative NPZ:   {output_relative_npz} (vsys={VSYS:g} km/s)")
    if output_au_npz is not None:
        print(
            f"AU-coordinate NPZ: {output_au_npz} "
            f"(d={DISTANCE_PC:g} pc, center={CENTER_XY}, "
            f"scale={au_pixel_scale} arcsec/pixel)"
        )
    if output_au_relative_npz is not None:
        print(
            f"AU-relative NPZ: {output_au_relative_npz} "
            f"(vsys={VSYS:g} km/s)"
        )
    print(f"Output figure:  {output_figure}")


if __name__ == "__main__":
    main()
