"""Cut a velocity-range subcube from a spectral FITS cube.

Run without arguments to use the configuration block, or pass CLI arguments
for agent/automation use.  ``cut_subcube`` is also available as a Python API.
"""

import argparse
from pathlib import Path
from typing import Sequence

import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube


# =============================================================================
# Human-operated configuration mode
# =============================================================================
# Running ``python cut-subcube.py`` without arguments uses these values.
# CLI arguments override the corresponding values for agent/automation use.

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CUBE_PATH = SCRIPT_DIR / "09018-spw1-cut.fits"
OUTPUT_CUBE_PATH = SCRIPT_DIR / "HNCO-m6-p25.fits"
REST_FREQUENCY = 219.798274 * u.GHz
VELOCITY_RANGE = (-6.0, 25.0) * u.km / u.s
VELOCITY_CONVENTION = "radio"
INPUT_HDU = 0
OVERWRITE_OUTPUT = False


def _resolve_path(path: Path | str) -> Path:
    """Resolve paths relative to the caller when they are not absolute."""
    return Path(path).expanduser().resolve()


def _validate_spectral_inputs(
    rest_frequency: u.Quantity,
    velocity_range: u.Quantity,
) -> tuple[u.Quantity, u.Quantity]:
    if not rest_frequency.unit.is_equivalent(u.Hz):
        raise u.UnitConversionError("rest_frequency must have frequency units")
    if rest_frequency <= 0 * u.Hz:
        raise ValueError("rest_frequency must be positive")

    velocity_range = u.Quantity(velocity_range)
    if velocity_range.shape != (2,):
        raise ValueError("velocity_range must contain exactly two velocities")
    if not velocity_range.unit.is_equivalent(u.km / u.s):
        raise u.UnitConversionError("velocity_range must have velocity units")
    if not np.all(np.isfinite(velocity_range.to_value(u.km / u.s))):
        raise ValueError("velocity_range values must be finite")

    velocity_range = velocity_range.to(u.km / u.s)
    velocity_min = min(velocity_range[0], velocity_range[1])
    velocity_max = max(velocity_range[0], velocity_range[1])
    if velocity_min == velocity_max:
        raise ValueError("velocity_range must have non-zero width")
    return velocity_min, velocity_max


def cut_subcube(
    input_cube_path: Path | str,
    output_cube_path: Path | str,
    rest_frequency: u.Quantity,
    velocity_range: u.Quantity,
    *,
    velocity_convention: str = "radio",
    input_hdu: int = 0,
    overwrite: bool = False,
) -> Path:
    """Cut and write a subcube whose channel centers lie in velocity_range."""
    velocity_min, velocity_max = _validate_spectral_inputs(
        rest_frequency,
        velocity_range,
    )
    input_path = _resolve_path(input_cube_path)
    output_path = _resolve_path(output_cube_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input cube does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output cube paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}. Choose another output path "
            "or pass --overwrite."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cube = SpectralCube.read(input_path, hdu=input_hdu)
    velocity_cube = cube.with_spectral_unit(
        u.km / u.s,
        velocity_convention=velocity_convention,
        rest_value=rest_frequency,
    )
    velocity_axis = velocity_cube.spectral_axis.to(u.km / u.s)
    selected_channels = np.flatnonzero(
        (velocity_axis >= velocity_min) & (velocity_axis <= velocity_max)
    )
    if selected_channels.size == 0:
        raise RuntimeError(
            f"No spectral channels fall inside {velocity_min} to {velocity_max}"
        )
    if selected_channels.size > 1 and not np.all(np.diff(selected_channels) == 1):
        raise RuntimeError("Selected velocity channels are unexpectedly non-contiguous")
    first_channel = int(selected_channels[0])
    last_channel = int(selected_channels[-1])
    subcube = velocity_cube[first_channel : last_channel + 1, :, :]

    # Materializing the HDU lets us record the exact rest frequency and cut in
    # the output header while retaining the subcube WCS generated above.
    output_hdu = subcube.hdu
    output_hdu.header["RESTFRQ"] = (
        rest_frequency.to_value(u.Hz),
        "Rest frequency [Hz]",
    )
    output_hdu.header.add_history(
        f"Spectral subcube requested over {velocity_min.value:g} to "
        f"{velocity_max.value:g} km/s ({velocity_convention} convention)."
    )
    output_hdu.header.add_history(
        f"Retained input spectral channels {first_channel} through {last_channel}."
    )
    output_hdu.writeto(output_path, overwrite=overwrite)

    actual_velocity = subcube.spectral_axis.to(u.km / u.s)
    actual_min = np.nanmin(actual_velocity.value)
    actual_max = np.nanmax(actual_velocity.value)
    print(f"Input cube:  {input_path}")
    print(f"Input shape: {cube.shape}")
    print(
        f"Requested:   {velocity_min.value:g} to {velocity_max.value:g} km/s "
        f"({velocity_convention})"
    )
    print(f"Actual:      {actual_min:.6g} to {actual_max:.6g} km/s")
    print(f"Channels:    {first_channel} to {last_channel} (inclusive)")
    print(f"Output shape:{subcube.shape}")
    print(f"Output cube: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cut a FITS spectral cube by radio/optical/relativistic velocity. "
            "Only channels whose centers lie inside the inclusive requested "
            "range are retained."
        ),
        epilog=(
            "Example: %(prog)s input.fits output.fits --restfreq-ghz "
            "220.071219 --velocity-range-kms -10 10\n\n"
            "With no arguments, values from the configuration block at the "
            "top of the script are used. Any supplied argument overrides its "
            "configured value."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_cube",
        type=Path,
        nargs="?",
        help="input FITS spectral cube (default: configured INPUT_CUBE_PATH)",
    )
    parser.add_argument(
        "output_cube",
        type=Path,
        nargs="?",
        help="output FITS subcube (default: configured OUTPUT_CUBE_PATH)",
    )
    parser.add_argument(
        "--restfreq-ghz",
        type=float,
        default=None,
        metavar="GHZ",
        help="target-line rest frequency in GHz",
    )
    parser.add_argument(
        "--velocity-range-kms",
        type=float,
        nargs=2,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="inclusive velocity range in km/s; reversed bounds are accepted",
    )
    parser.add_argument(
        "--velocity-convention",
        choices=("radio", "optical", "relativistic"),
        default=None,
        help="Doppler convention (default: configured VELOCITY_CONVENTION)",
    )
    parser.add_argument(
        "--hdu",
        type=int,
        default=None,
        help="zero-based FITS HDU (default: configured INPUT_HDU)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace an existing output (default: configured OVERWRITE_OUTPUT)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    input_cube = INPUT_CUBE_PATH if args.input_cube is None else args.input_cube
    output_cube = OUTPUT_CUBE_PATH if args.output_cube is None else args.output_cube
    rest_frequency = (
        REST_FREQUENCY if args.restfreq_ghz is None else args.restfreq_ghz * u.GHz
    )
    velocity_range = (
        VELOCITY_RANGE
        if args.velocity_range_kms is None
        else np.asarray(args.velocity_range_kms) * u.km / u.s
    )
    velocity_convention = (
        VELOCITY_CONVENTION
        if args.velocity_convention is None
        else args.velocity_convention
    )
    input_hdu = INPUT_HDU if args.hdu is None else args.hdu
    overwrite = OVERWRITE_OUTPUT if args.overwrite is None else args.overwrite

    return cut_subcube(
        input_cube_path=input_cube,
        output_cube_path=output_cube,
        rest_frequency=rest_frequency,
        velocity_range=velocity_range,
        velocity_convention=velocity_convention,
        input_hdu=input_hdu,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
