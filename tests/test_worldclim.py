import os

import numpy as np
import pytest

from sensorsio import mgrs
from sensorsio.worldclim import (WorldClimBio, WorldClimData,
                                 WorldClimQuantity, WorldClimVar,
                                 WorldClimVarAll)


def get_worldclim_folder() -> str:
    """
    Retrieve SRTM folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'worldclim')


@pytest.mark.parametrize(
    "var, month, fail",
    [
        (WorldClimQuantity.TAVG, 10, False),
        (WorldClimQuantity.TAVG, 0, True),
        (WorldClimBio.PREC_SEASONALITY, None, False),
        (WorldClimBio.MEAN_DIURNAL_TEMP_RANGE, 10, True),
    ],
)
def test_wc_var(var, month, fail):
    try:
        WorldClimVar(var, month)
    except ValueError:
        if not fail:
            assert False, f"Failed with {var}, {month}"


@pytest.mark.parametrize(
    "var, month, str_repr",
    [
        (WorldClimQuantity.TAVG, 10, "CLIM_TAVG_10"),
        (WorldClimBio.PREC_SEASONALITY, None, "Prec_Seasonality"),
    ],
)
def test_wc_var_str(var, month, str_repr):
    assert str(WorldClimVar(var, month)) == str_repr.upper()


@pytest.mark.requires_test_data
def test_instantiate_worldclim_data():
    wcd = WorldClimData(wcdir=get_worldclim_folder())


@pytest.mark.requires_test_data
def test_climfiles_exist():
    wcd = WorldClimData(wcdir=get_worldclim_folder())
    for climfile in wcd.climfiles:
        assert os.path.isfile(climfile)


@pytest.mark.requires_test_data
def test_biofiles_exist():
    wcd = WorldClimData(wcdir=get_worldclim_folder())
    for biofile in wcd.biofiles:
        assert os.path.isfile(biofile)


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "wc_vars, suffix",
    [
        (WorldClimVarAll, "all"),
        ([WorldClimVar(WorldClimQuantity.PREC, m) for m in range(1, 6)], "prec"),
        ([WorldClimVar(wcb) for wcb in WorldClimBio], "bio"),
    ],
)
def test_wc_read_as_numpy(wc_vars, suffix):
    """
    Test the read_as_numpy method
    """
    TILE = "35NKA"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    resolution = 200.0
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    wcd = WorldClimData(wcdir=get_worldclim_folder())
    (dst_wc, xcoords, ycoords, dst_crs) = wcd.read_as_numpy(wc_vars=wc_vars,
                                                            crs=crs,
                                                            resolution=resolution,
                                                            bounds=bbox)
    expected_bands = len(wcd.climfiles + wcd.biofiles) if wc_vars is None else len(wc_vars)
    assert dst_wc.shape == (expected_bands, 549, 549)
    assert xcoords.shape == (549, )
    assert ycoords.shape == (549, )
    assert dst_crs == crs
    assert dst_wc.dtype == np.dtype('float32')


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "wc_vars, suffix",
    [(WorldClimVarAll, "all"),
     ([WorldClimVar(WorldClimQuantity.WIND, m) for m in range(1, 6)], "wind")],
)
def test_wc_read_as_xarray(wc_vars, suffix):
    """
    Test the read_as_xarray method
    """
    TILE = "35NKA"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    resolution = 200.0
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    wcd = WorldClimData(wcdir=get_worldclim_folder())
    wc_arr = wcd.read_as_xarray(wc_vars=wc_vars, crs=crs, resolution=resolution, bounds=bbox)

    # Check all variables are there
    for var in wc_vars:
        var_name = 'wc_' + wcd.get_var_name(var)
        assert var_name in wc_arr.variables
        assert wc_arr[var_name].shape == (549, 549)

    assert wc_arr.x.shape == (549, )
    assert wc_arr.y.shape == (549, )

    for field in ['nodata', 'crs', 'resolution']:
        assert field in wc_arr.attrs
