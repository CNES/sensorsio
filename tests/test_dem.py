#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test for the SRTM driver
"""
import os
import tempfile

import pytest
import rasterio as rio
from pyproj import CRS
from sensorsio import mgrs, srtm


def test_srtm_id_to_name():
    """
    Test conversion between srtm tile id and file name
    """
    assert srtm.SRTMTileId(1, -2).name() == "S02E001"
    assert srtm.SRTMTileId(-1, -2).name() == "S02W001"
    assert srtm.SRTMTileId(-1, 12).name() == "N12W001"


def test_crs_from_mgrs():
    """
    Test conversion between mgrs tile id and crs
    """
    assert mgrs.get_crs_mgrs_tile("31TDH").to_authority() == ("EPSG", "32631")


def test_mgrs_transform():
    """
    Test the conversion between mgrs tile id and geotransform
    """
    assert mgrs.get_transform_mgrs_tile("31TDH") == rio.Affine(10.0, 0.0, 399960.0, 0.0, -10.0,
                                                               4800000.0)


def test_get_bbox_mgrs_tile():
    """
    Check that get_bbox_mrgs_tile returns the correct geometry
    """
    TILE = "31TCJ"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    assert int((bbox[2] - bbox[0]) / 10.) == 10980
    assert int((bbox[3] - bbox[1]) / 10.) == 10980

    geotransform = mgrs.get_transform_mgrs_tile(TILE)

    assert geotransform[2] == bbox[0]
    assert geotransform[5] == bbox[3]


def test_srtm_tiles_from_mgrs_tile():
    def build_tile_list(tile):
        return [tid.name() for tid in srtm.get_srtm_tiles_for_mgrs_tile(tile)]

    assert build_tile_list("31TCJ") == [
        "N43E000",
        "N43E001",
        "N44E000",
        "N44E001",
    ]

    assert build_tile_list("36TTM") == [
        "N41E029",
        "N41E030",
        "N42E029",
        "N42E030",
    ]

    assert build_tile_list("35MMQ") == [
        "S06E026",
        "S06E027",
        "S05E026",
        "S05E027",
    ]

    assert build_tile_list("19GEP") == [
        "S43W070",
        "S43W069",
        "S43W068",
        "S42W070",
        "S42W069",
        "S42W068",
    ]


def get_srtm_folder() -> str:
    """
    Retrieve SRTM folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'srtm')


@pytest.mark.requires_test_data
def test_dem_on_mgrs_tile():
    """
    Test the dem_on_mgrs_tile helper function as well as the write_dem fonction
    """
    TILE = "31TDH"
    s2_dem = srtm.get_dem_mgrs_tile(TILE, get_srtm_folder())

    assert s2_dem.elevation.shape == (10980, 10980)
    assert s2_dem.aspect.shape == (10980, 10980)
    assert s2_dem.slope.shape == (10980, 10980)
    assert s2_dem.crs == CRS.from_string('EPSG:32631')
    assert s2_dem.transform == rio.Affine(10., 0, 399960.0, 0, -10., 4800000)

    stack = s2_dem.as_stack()

    assert stack.shape == (3, 10980, 10980)

    # Test the write_dem method
    with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file:
        srtm.write_dem(s2_dem, temporary_file.name)
        with rio.open(temporary_file.name) as readback_file:
            print(readback_file)
            assert readback_file.crs == 'EPSG:32631'
            assert readback_file.transform == rio.Affine(10., 0, 399960.0, 0, -10., 4800000)
            assert readback_file.count == 3
            assert readback_file.height == 10980
            assert readback_file.width == 10980


@pytest.mark.requires_test_data
def test_dem_read_as_numpy():
    TILE = "31TCJ"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    resolution = 100.0
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    assert int((bbox[2] - bbox[0]) / 10.) == 10980

    dem_handler = srtm.SRTM(get_srtm_folder())

    (
        elevation,
        slope,
        aspect,
        xcoords,
        ycoords,
        dem_crs,
        dem_transform,
    ) = dem_handler.read_as_numpy(crs, resolution, bbox)

    for arr in (elevation, slope, aspect):
        assert arr.shape == (1098, 1098)

    for arr in (xcoords, ycoords):
        assert arr.shape == (1098, )

    assert dem_crs == 'EPSG:32631'
    assert dem_transform == rio.Affine(100., 0, bbox[0], 0, -100., bbox[3])


@pytest.mark.requires_test_data
def test_dem_read_as_xarray():
    TILE = "31TDH"
    crs = mgrs.get_crs_mgrs_tile(TILE)
    resolution = 100.0
    bbox = mgrs.get_bbox_mgrs_tile(TILE, latlon=False)
    print("Bounds ", srtm.compute_latlon_bbox_from_region(bbox, crs))
    dem_handler = srtm.SRTM(get_srtm_folder())
    xarr_dem = dem_handler.read_as_xarray(crs, resolution, bbox)

    for var in ('height', 'aspect', 'slope'):
        assert var in xarr_dem

    assert xarr_dem.attrs['crs'] == 'EPSG:32631'

    assert xarr_dem.x.shape == (1098, )
    assert xarr_dem.y.shape == (1098, )
