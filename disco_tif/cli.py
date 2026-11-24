"""Command line: ``disco-tif hspca | fit-basis | apply-basis | hillshade``."""

import argparse
import logging
from pathlib import Path

from disco_tif import __version__
from disco_tif.hspca import OUTPUTS, apply_basis, fit_basis, hillshade, hillshade_pca
from disco_tif.io import Z_UNITS_TO_M
from disco_tif.pca import DEFAULT_ALTITUDES, DEFAULT_AZIMUTHS, Basis


def main(argv=None):
    """Run the disco-tif command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments after the program name; None reads them from ``sys.argv``.

    Returns
    -------
    int
        Exit status, 0 on success.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.run(args)
    return 0


def _parser():
    """Build the argument parser with one subcommand per workflow."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--z-units", choices=sorted(Z_UNITS_TO_M), help="elevation unit (default: the projected CRS unit; required for a geographic CRS)")
    common.add_argument("--block-size", type=int, default=1024, help="block side [cells] (default: %(default)s)")

    fit = argparse.ArgumentParser(add_help=False)
    fit.add_argument("--azimuths", type=float, nargs="+", default=DEFAULT_AZIMUTHS, help="sun azimuths [degrees clockwise from north]")
    fit.add_argument("--altitudes", type=float, nargs="+", default=DEFAULT_ALTITUDES, help="sun altitudes [degrees]")
    fit.add_argument("--sample-size", type=int, default=1_000_000, help="cells sampled for the stretch")
    fit.add_argument("--stretch", type=float, nargs=2, default=(1, 99), metavar=("LOW", "HIGH"), help="stretch percentiles")
    fit.add_argument("--seed", type=int, default=0, help="seed of the stretch sample")

    write = argparse.ArgumentParser(add_help=False)
    write.add_argument("--out-dir", type=Path, help="output folder (default: next to each DEM)")
    write.add_argument("--outputs", nargs="+", choices=OUTPUTS, default=OUTPUTS, help="products to write")
    write.add_argument("--dtype", choices=("uint8", "float32"), default="uint8", help="component data type")

    parser = argparse.ArgumentParser(prog="disco-tif", description="Principal-component hillshades of DEMs.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(required=True, metavar="command")

    p = sub.add_parser("hspca", parents=[common, fit, write], help="fit and write the hillshade PCA of one DEM")
    p.add_argument("dem", type=Path)
    p.set_defaults(run=_hspca)

    p = sub.add_parser("fit-basis", parents=[common, fit], help="fit one basis to one or more DEMs, saved as JSON")
    p.add_argument("dems", type=Path, nargs="+")
    p.add_argument("-o", "--output", type=Path, required=True, help="basis JSON to write")
    p.add_argument("--allow-mixed-resolution", action="store_true", help="pool DEMs whose cell sizes differ by more than 2x")
    p.set_defaults(run=_fit_basis)

    p = sub.add_parser("apply-basis", parents=[common, write], help="write the hillshade PCA of DEMs under a saved basis")
    p.add_argument("basis", type=Path)
    p.add_argument("dems", type=Path, nargs="+")
    p.set_defaults(run=_apply_basis)

    p = sub.add_parser("hillshade", parents=[common], help="write a single hillshade")
    p.add_argument("dem", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--azimuth", type=float, default=315, help="sun azimuth [degrees clockwise from north]")
    p.add_argument("--altitude", type=float, default=45, help="sun altitude [degrees]")
    p.add_argument("--dtype", choices=("uint8", "float32"), default="uint8")
    p.set_defaults(run=_hillshade)
    return parser


def _fit_options(args):
    """Keyword arguments shared by fit_basis and hillshade_pca."""
    return dict(azimuths=args.azimuths, altitudes=args.altitudes, z_units=args.z_units, block_size=args.block_size,
                sample_size=args.sample_size, stretch=tuple(args.stretch), seed=args.seed)


def _hspca(args):
    hillshade_pca(args.dem, args.out_dir or args.dem.parent, outputs=args.outputs, dtype=args.dtype,
                  **_fit_options(args))


def _fit_basis(args):
    basis = fit_basis(args.dems, allow_mixed_resolution=args.allow_mixed_resolution, **_fit_options(args))
    logging.getLogger(__name__).info("wrote %s", basis.save(args.output))


def _apply_basis(args):
    basis = Basis.load(args.basis)
    for dem in args.dems:
        apply_basis(dem, basis, args.out_dir or dem.parent, outputs=args.outputs, dtype=args.dtype,
                    z_units=args.z_units, block_size=args.block_size)


def _hillshade(args):
    hillshade(args.dem, args.output, azimuth=args.azimuth, altitude=args.altitude, z_units=args.z_units,
              block_size=args.block_size, dtype=args.dtype)
