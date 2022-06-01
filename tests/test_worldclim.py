import os

import pytest
import rasterio as rio
from rasterio.coords import BoundingBox
from sensorsio import mgrs, utils
from sensorsio.worldclim import (WorldClimBio, WorldClimData,
                                 WorldClimQuantity, WorldClimVar)


@pytest.mark.parametrize(
    "var, month, fail",
    [
        (WorldClimQuantity.TAVG, 10, False),
        (WorldClimQuantity.TAVG, 0, True),
        (WorldClimBio.BIO15, None, False),
        (WorldClimBio.BIO02, 10, True),
    ],
)
def test_wc_var(var, month, fail):
    try:
        WorldClimVar(var, month)
    except ValueError:
        if not fail:
            assert False, f"Failed with {var}, {month}"


def test_instantiate_worldclim_data():
    wcd = WorldClimData()


def test_climfiles_exist():
    wcd = WorldClimData()
    for climfile in wcd.climfiles:
        assert os.path.isfile(climfile)


def test_biofiles_exist():
    wcd = WorldClimData()
    for biofile in wcd.biofiles:
        assert os.path.isfile(biofile)


def test_crop_to_bbox():
    wcd = WorldClimData()
    bbox = BoundingBox(
        left=1.7458519129811987,
        bottom=42.35763630809999,
        right=3.1204336461000004,
        top=43.35279198479999,
    )
    wc_data = wcd.crop_to_bbox(wcd.climfiles[0], bbox)
    assert wc_data.shape == (1, 119, 164)


@pytest.mark.parametrize(
    "vars",
    [
        [WorldClimVar(WorldClimQuantity.TAVG, 1)],
        [
            WorldClimVar(WorldClimQuantity.TAVG, 1),
            WorldClimVar(WorldClimBio.BIO03),
        ],
        None,
    ],
)
def test_get_wc_for_bbox(vars):
    wcd = WorldClimData()
    bbox = BoundingBox(
        left=1.7458519129811987,
        bottom=42.35763630809999,
        right=3.1204336461000004,
        top=43.35279198479999,
    )
    wc, transform = wcd.get_wc_for_bbox(bbox, vars=vars)
    nb_vars = 103
    if vars is not None:
        nb_vars = len(vars)
    assert wc.shape == (nb_vars, 119, 164)


@pytest.mark.parametrize(
    "vars, suffix",
    [
        (None, "all"),
        ([WorldClimVar(WorldClimQuantity.PREC, m)
          for m in range(1, 6)], "prec"),
        ([WorldClimVar(wcb) for wcb in WorldClimBio], "bio"),
    ],
)
def test_wc_read_as_numpy(vars, suffix):
    TILE = "35NKA"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    resolution = 200.0
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    wcd = WorldClimData()
    (dst_wc, xcoords, ycoords, crs,
     dst_wc_transform) = wcd.read_as_numpy(vars=vars,
                                           crs=crs,
                                           resolution=resolution,
                                           bounds=bbox)
    expected_bands = len(wcd.climfiles +
                         wcd.biofiles) if vars is None else len(vars)
    assert dst_wc.shape[0] == expected_bands
    # Write just 3 channels for simplicity
    dst_wc = dst_wc[:3, :, :]
    with rio.open(
            f"/work/scratch/{os.environ['USER']}/wc_test_{suffix}.tif",
            "w",
            driver="GTiff",
            height=dst_wc.shape[1],
            width=dst_wc.shape[2],
            count=dst_wc.shape[0],
            nodata=-32768.0,
            dtype=dst_wc.dtype,
            compress="lzw",
            crs=crs,
            transform=dst_wc_transform,
    ) as ds:
        ds.write(dst_wc)
