#!/usr/bin/env python

############################################################################
#
# MODULE:       test_r_flexure
# AUTHOR:       Andrew Wickert
# PURPOSE:      Tests for r.flexure (gridded flexural isostasy)
# COPYRIGHT:    (C) 2026 by Andrew Wickert and the GRASS Development Team
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
############################################################################

"""
Tests for r.flexure.

Exercises FD, FFT, and SAS solution methods with a synthetic 10×10 load
raster (100 m resolution). Skips automatically when gFlex is not installed.

Run inside a GRASS session (e.g., with --tmp-location XY):
    python -m grass.gunittest.main
"""

import unittest

from grass.gunittest.case import TestCase
from grass.gunittest.main import test

# Synthetic 10×10 load raster (100 m resolution, 1 km × 1 km).
# A single central cell carries a 1e9 Pa load (roughly 100 m of dense rock).
LOAD_ASCII = """\
north: 1000
south: 0
east: 1000
west: 0
rows: 10
cols: 10
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 1e9 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0
"""

# Spatially variable Te raster at 100 m resolution (uniform 10 km).
TE_ASCII = """\
north: 1000
south: 0
east: 1000
west: 0
rows: 10
cols: 10
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
10000 10000 10000 10000 10000 10000 10000 10000 10000 10000
"""

# Fixtures for the -p padding test: 30×30 at 5000 m (150 km × 150 km).
# Te = 5000 m → α ≈ 23 km, flexural wavelength ≈ 144 km,
# recommended pad ≈ 29 cells/side → padded domain is 88×88 (manageable).
# The 10×10 at 100 m fixture above would produce ~1300 cells of padding
# and a 2600×2600 FD matrix that crashes the direct solver.
_PAD_ROW_ZERO = "0 " * 30
_PAD_ROW_LOAD = "0 " * 14 + "1e9 " + "0 " * 15
_PAD_TE_ROW = "5000 " * 30
LOAD_PAD_ASCII = (
    "north: 150000\nsouth: 0\neast: 150000\nwest: 0\nrows: 30\ncols: 30\n"
    + (_PAD_ROW_ZERO + "\n") * 14
    + _PAD_ROW_LOAD
    + "\n"
    + (_PAD_ROW_ZERO + "\n") * 15
)
TE_PAD_ASCII = (
    "north: 150000\nsouth: 0\neast: 150000\nwest: 0\nrows: 30\ncols: 30\n"
    + (_PAD_TE_ROW + "\n") * 30
)


def _gflex_ok():
    """Return True if gFlex is importable."""
    try:
        import gflex  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_gflex_ok(), "gFlex not available")
class TestRFlexure(TestCase):
    """Test r.flexure with synthetic raster data (no NC dataset required)."""

    load = "test_rflex_load"
    te_rast = "test_rflex_te"

    @classmethod
    def setUpClass(cls):
        cls.use_temp_region()
        cls.runModule("r.in.ascii", input="-", stdin_=LOAD_ASCII, output=cls.load)
        cls.runModule("r.in.ascii", input="-", stdin_=TE_ASCII, output=cls.te_rast)
        cls.runModule("g.region", raster=cls.load)

    @classmethod
    def tearDownClass(cls):
        cls.del_temp_region()
        cls.runModule(
            "g.remove",
            flags="f",
            type="raster",
            name=",".join([cls.load, cls.te_rast]),
            quiet=True,
        )

    def _run_and_check(self, output, **kwargs):
        """Run r.flexure, assert success, assert 100 non-null output cells."""
        try:
            self.assertModule("r.flexure", output=output, **kwargs)
            self.assertRasterExists(output)
            self.assertRasterFitsUnivar(
                raster=output, reference={"n": 100}, precision=0
            )
        finally:
            self.runModule(
                "g.remove", flags="f", type="raster", name=output, quiet=True
            )

    def test_fd_scalar_te(self):
        """FD method with scalar Te [m]."""
        self._run_and_check(
            "test_rflex_fd",
            method="FD",
            input=self.load,
            te="10000",
            te_units="m",
            northbc="0Moment0Shear",
            southbc="0Moment0Shear",
            eastbc="0Moment0Shear",
            westbc="0Moment0Shear",
        )

    def test_fft_scalar_te(self):
        """FFT method with scalar Te [m]."""
        self._run_and_check(
            "test_rflex_fft",
            method="FFT",
            input=self.load,
            te="10000",
            te_units="m",
        )

    def test_sas_scalar_te(self):
        """SAS method with scalar Te [m]."""
        self._run_and_check(
            "test_rflex_sas",
            method="SAS",
            input=self.load,
            te="10000",
            te_units="m",
        )

    def test_fd_raster_te(self):
        """FD method with spatially variable (raster) Te."""
        self._run_and_check(
            "test_rflex_fd_rte",
            method="FD",
            input=self.load,
            te=self.te_rast,
            te_units="m",
            northbc="0Moment0Shear",
            southbc="0Moment0Shear",
            eastbc="0Moment0Shear",
            westbc="0Moment0Shear",
        )

    def test_fd_sigma_stresses(self):
        """FD method with non-zero in-plane stresses."""
        self._run_and_check(
            "test_rflex_fd_sigma",
            method="FD",
            input=self.load,
            te="10000",
            te_units="m",
            sigma_xx="1e6",
            sigma_yy="1e6",
            sigma_xy="0",
            northbc="0Moment0Shear",
            southbc="0Moment0Shear",
            eastbc="0Moment0Shear",
            westbc="0Moment0Shear",
        )

    def test_te_km_units(self):
        """Te specified in km should match the same value in m (SAS, smoke test)."""
        self._run_and_check(
            "test_rflex_km",
            method="SAS",
            input=self.load,
            te="10",
            te_units="km",
        )

    def test_fft_periodic_bc(self):
        """FFT method with all-periodic boundary conditions (exact FFT path)."""
        self._run_and_check(
            "test_rflex_fft_per",
            method="FFT",
            input=self.load,
            te="10000",
            te_units="m",
            northbc="Periodic",
            southbc="Periodic",
            eastbc="Periodic",
            westbc="Periodic",
        )

    def test_fd_mirror_bc(self):
        """FD method with Mirror boundary conditions."""
        self._run_and_check(
            "test_rflex_fd_mirror",
            method="FD",
            input=self.load,
            te="10000",
            te_units="m",
            northbc="Mirror",
            southbc="Mirror",
            eastbc="Mirror",
            westbc="Mirror",
        )


@unittest.skipUnless(_gflex_ok(), "gFlex not available")
class TestRFlexurePadded(TestCase):
    """Test -p domain padding with a domain appropriately sized for the flexural wavelength.

    The main TestRFlexure fixture (10×10 at 100 m, Te=10 km) would produce
    ~1300 cells of padding per side, creating a 2600×2600 FD problem that
    crashes the direct solver.  Here we use 30×30 at 5000 m with Te=5000 m
    so padding is ~29 cells/side (88×88 padded domain).
    """

    load_pad = "test_rflex_pad_load"
    te_pad = "test_rflex_pad_te"

    @classmethod
    def setUpClass(cls):
        cls.use_temp_region()
        cls.runModule(
            "r.in.ascii", input="-", stdin_=LOAD_PAD_ASCII, output=cls.load_pad
        )
        cls.runModule(
            "r.in.ascii", input="-", stdin_=TE_PAD_ASCII, output=cls.te_pad
        )
        cls.runModule("g.region", raster=cls.load_pad)

    @classmethod
    def tearDownClass(cls):
        cls.del_temp_region()
        cls.runModule(
            "g.remove",
            flags="f",
            type="raster",
            name=",".join([cls.load_pad, cls.te_pad]),
            quiet=True,
        )

    def test_fd_raster_te_padded(self):
        """FD with raster Te and -p domain-padding flag; output trimmed to original region."""
        output = "test_rflex_pad_out"
        try:
            self.assertModule(
                "r.flexure",
                flags="p",
                method="FD",
                input=self.load_pad,
                te=self.te_pad,
                te_units="m",
                output=output,
                northbc="0Moment0Shear",
                southbc="0Moment0Shear",
                eastbc="0Moment0Shear",
                westbc="0Moment0Shear",
            )
            self.assertRasterExists(output)
            # Output must be trimmed back to the original 30×30 region
            self.assertRasterFitsUnivar(
                raster=output, reference={"n": 900}, precision=0
            )
        finally:
            self.runModule(
                "g.remove", flags="f", type="raster", name=output, quiet=True
            )


if __name__ == "__main__":
    test()
