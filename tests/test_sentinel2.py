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
This module contains tests for the Sentinel2 driver
"""
import datetime
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from affine import Affine
import pytest
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.windows import Window
from sensorsio import mgrs, sentinel2


def get_sentinel2_l2a_theia_folder() -> str:
    """
    Retrieve Sentinel2 folder from env var
    """
    return os.path.join(
        os.environ["SENSORSIO_TEST_DATA_PATH"],
        "sentinel2",
        "l2a_maja",
        "SENTINEL2B_20230219-105857-687_L2A_T31TCJ_C_V3-1",
    )


def test_get_theia_tiles():
    """
    Test the get theia tiles function
    """
    theia_tiles = sentinel2.get_theia_tiles()

    assert len(theia_tiles) == 1076


def test_find_tile_orbit_pairs():
    """
    Test the find tile orbit pairs function
    """
    tile_id = "31TCJ"
    tile_bounds = mgrs.get_bbox_mgrs_tile(tile_id, latlon=False)
    tile_crs = mgrs.get_crs_mgrs_tile(tile_id)

    tiles_orbits_df = sentinel2.find_tile_orbit_pairs(tile_bounds, tile_crs)
    assert len(tiles_orbits_df) == 15

    most_covered = tiles_orbits_df[tiles_orbits_df.tile_coverage > 0.9].copy()

    assert len(most_covered) == 2

    assert set(most_covered.tile_id) == {"31TCJ"}
    assert set(most_covered.relative_orbit_number) == {8, 51}


@pytest.mark.requires_test_data
def test_sentinel2_instantiate_l2a_theia():
    """
    Test sentinel2 class instantiation
    """
    s2 = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    assert s2.product_dir == get_sentinel2_l2a_theia_folder()
    assert s2.product_name == "SENTINEL2B_20230219-105857-687_L2A_T31TCJ_C_V3-1"
    assert s2.date == datetime.date(2023, 2, 19)
    assert s2.time == datetime.time(10, 58, 57)
    assert s2.year == 2023
    assert s2.day_of_year == 50
    assert s2.tile == "31TCJ"
    assert s2.cloud_cover == 11
    assert s2.satellite == sentinel2.Sentinel2.Satellite.S2B
    assert s2.crs == "epsg:32631"
    assert s2.bounds == mgrs.get_bbox_mgrs_tile(s2.tile, latlon=False)
    assert s2.transform == mgrs.get_transform_mgrs_tile(s2.tile)
    assert s2.relative_orbit_number == 51


def test_sentinel2_psf():
    """
    Test the PSF method
    """
    psf = sentinel2.Sentinel2.generate_psf_kernel(
        sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M,
        half_kernel_width=5,
    )
    assert psf.shape == (10, 11, 11)


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """

    bands: List[sentinel2.Sentinel2.Band] = field(
        default_factory=lambda: sentinel2.Sentinel2.GROUP_10M
    )
    band_type: sentinel2.Sentinel2.BandType = sentinel2.Sentinel2.FRE
    masks: List[sentinel2.Sentinel2.Mask] = field(
        default_factory=lambda: sentinel2.Sentinel2.ALL_MASKS
    )
    read_atmos: bool = False
    res: sentinel2.Sentinel2.Res = sentinel2.Sentinel2.R1
    scale: float = 10000
    crs: Optional[str] = None
    resolution: float = 10
    no_data_value: float = np.nan
    bounds: rio.coords.BoundingBox = None
    algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic
    dtype: np.dtype = np.dtype("float32")

    def expected_shape(self) -> Tuple[int, int]:
        """
        return expected shape
        """
        if self.bounds is not None:
            return (
                int((self.bounds[3] - self.bounds[1]) / self.resolution),
                int((self.bounds[2] - self.bounds[0]) / self.resolution),
            )

        return (int(10980 * 10 / self.resolution), int(10980 * 10 / self.resolution))


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "parameters",
    [
        ReadAsNumpyParams(
            bounds=rio.coords.BoundingBox(300000.0, 4790220.0, 301000, 4792220)
        ),
        # Use bounds to set output region, with 20m bands
        ReadAsNumpyParams(
            bands=sentinel2.Sentinel2.GROUP_20M,
            bounds=rio.coords.BoundingBox(300000.0, 4790220.0, 301000, 4792220),
        ),
        # Use bounds to set output region, with 20m bands and 10 bands
        ReadAsNumpyParams(
            bands=sentinel2.Sentinel2.GROUP_10M + sentinel2.Sentinel2.GROUP_20M,
            bounds=rio.coords.BoundingBox(300000.0, 4790220.0, 301000, 4792220),
        ),
        # Set a different target crs
        ReadAsNumpyParams(
            bounds=rio.coords.BoundingBox(499830.0, 6240795.0, 500830.0, 6242795.0),
            crs="EPSG:2154",
        ),
    ],
)
def test_read_as_numpy_xarray(parameters: ReadAsNumpyParams):
    """
    Test the read_as_numpy method
    """
    s2_dataset = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    # Read as numpy part
    bands_arr, mask_arr, atm_arr, xcoords, ycoords, crs = s2_dataset.read_as_numpy(
        **parameters.__dict__
    )

    assert bands_arr.shape == (len(parameters.bands), *parameters.expected_shape())
    assert mask_arr is not None and mask_arr.shape == (
        len(parameters.masks),
        *parameters.expected_shape(),
    )
    assert (~np.isnan(bands_arr)).sum() > 0

    if parameters.read_atmos:
        assert atm_arr is not None and atm_arr.shape == (
            2,
            *parameters.expected_shape(),
        )
    else:
        assert atm_arr is None

    assert ycoords.shape == (parameters.expected_shape()[0],)
    assert xcoords.shape == (parameters.expected_shape()[1],)

    if parameters.crs is not None:
        assert crs == parameters.crs
    else:
        assert crs == mgrs.get_crs_mgrs_tile(s2_dataset.tile)

    # Test read as xarray part
    s2_xr = s2_dataset.read_as_xarray(**parameters.__dict__)

    for c in ["t", "x", "y"]:
        assert c in s2_xr.coords

    assert s2_xr["t"].shape == (1,)
    assert s2_xr["x"].shape == (parameters.expected_shape()[1],)
    assert s2_xr["y"].shape == (parameters.expected_shape()[0],)

    for band in parameters.bands:
        assert band.value in s2_xr.variables
        assert s2_xr[band.value].shape == (1, *parameters.expected_shape())

    if parameters.read_atmos:
        for atm_band in ["WCV", "AOT"]:
            assert atm_band in s2_xr.variables
            assert s2_xr[atm_band].shape == (1, parameters.expected_shape())

    assert s2_xr.attrs["tile"] == "31TCJ"
    assert s2_xr.attrs["type"] == parameters.band_type.value
    if parameters.crs is not None:
        assert s2_xr.attrs["crs"] == parameters.crs
    else:
        assert s2_xr.attrs["crs"] == s2_dataset.crs


@pytest.mark.requires_test_data
def test_read_incidence_angle_as_numpy():
    """
    Test the function that reads incidence angles
    """
    s2_dataset = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    even_zenith_angle, odd_zenith_angle, even_azimuth_angle, odd_azimuth_angle = (
        s2_dataset.read_incidence_angles_as_numpy(res=sentinel2.Sentinel2.R2)
    )

    for arr in [
        even_zenith_angle,
        odd_zenith_angle,
        even_azimuth_angle,
        odd_azimuth_angle,
    ]:
        assert arr.shape == (5490, 5490)

    for arr in [even_zenith_angle, odd_zenith_angle]:
        assert np.nanmax(arr) < 12
        assert np.nanmin(arr) >= 0


@pytest.mark.requires_test_data
def test_read_solar_angle_as_numpy():
    """
    Test the function that reads solar angles
    """
    s2_dataset = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    zenith_angle, azimuth_angle = s2_dataset.read_solar_angles_as_numpy(
        res=sentinel2.Sentinel2.R2
    )

    for arr in [zenith_angle, azimuth_angle]:
        assert arr.shape == (5490, 5490)

    assert azimuth_angle.max() < 180
    assert azimuth_angle.min() > 0
    assert zenith_angle.max() < 60
    assert zenith_angle.min() > 0


@pytest.mark.requires_test_data
def test_read_solar_angle_with_bounding_box_as_numpy():
    """
    Test the function that reads solar angles with bounding box
    """
    # Open data
    s2_dataset = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    # Define Bounding Box
    col_off, row_off, width, height = 248, 2256, 2567, 2735
    window_10m = Window(col_off, row_off, width, height)
    bounding_box = BoundingBox(
        *rio.windows.bounds(window_10m, s2_dataset.transform),
    )
    transform = Affine(
        s2_dataset.transform[0] * 2,
        s2_dataset.transform[1],
        s2_dataset.transform[2],
        s2_dataset.transform[3],
        s2_dataset.transform[4] * 2,
        s2_dataset.transform[5],
    )
    window_20m = rio.windows.from_bounds(*bounding_box, transform).round()

    # Check on solar_angles
    for band, res, win in zip(
        [sentinel2.Sentinel2.B2, sentinel2.Sentinel2.B5],
        [sentinel2.Sentinel2.R1, sentinel2.Sentinel2.R2],
        [window_10m, window_20m],
    ):
        # Get slice for comparison
        slice_row = slice(win.row_off, win.row_off + win.height)
        slice_col = slice(win.col_off, win.col_off + win.width)

        # Read full data
        sun_zen, sun_az = s2_dataset.read_solar_angles_as_numpy(res=res)

        # Read bb area
        sun_zen_box, sun_az_box = s2_dataset.read_solar_angles_as_numpy(
            res=res, bounds=bounding_box
        )

        assert np.allclose(sun_zen[slice_row, slice_col], sun_zen_box, equal_nan=True)

        del sun_zen, sun_az, sun_zen_box, sun_az_box


@pytest.mark.requires_test_data
def test_read_incidence_angle_with_bounding_box_as_numpy():
    """
    Test the function that reads incidence angles with bounding box
    """
    # Open data
    s2_dataset = sentinel2.Sentinel2(get_sentinel2_l2a_theia_folder())

    # Define Bounding Box
    col_off, row_off, width, height = 248, 2256, 2567, 2735
    window_10m = Window(col_off, row_off, width, height)
    bounding_box = BoundingBox(
        *rio.windows.bounds(window_10m, s2_dataset.transform),
    )
    transform = Affine(
        s2_dataset.transform[0] * 2,
        s2_dataset.transform[1],
        s2_dataset.transform[2],
        s2_dataset.transform[3],
        s2_dataset.transform[4] * 2,
        s2_dataset.transform[5],
    )
    window_20m = rio.windows.from_bounds(*bounding_box, transform).round()

    # Check on incidence angles
    for band, res, win in zip(
        [sentinel2.Sentinel2.B2, sentinel2.Sentinel2.B5],
        [sentinel2.Sentinel2.R1, sentinel2.Sentinel2.R2],
        [window_10m, window_20m],
    ):
        # Get slice for comparison
        slice_row = slice(win.row_off, win.row_off + win.height)
        slice_col = slice(win.col_off, win.col_off + win.width)

        # Read full data
        even_zen, odd_zen, even_az, odd_az = s2_dataset.read_incidence_angles_as_numpy(
            band, res
        )

        # Read bb data
        even_zen_box, odd_zen_box, even_az_box, odd_az_box = (
            s2_dataset.read_incidence_angles_as_numpy(band, res, bounds=bounding_box)
        )

        for full, box in zip(
            [even_zen, odd_zen, even_az, odd_az],
            [even_zen_box, odd_zen_box, even_az_box, odd_az_box],
        ):
            assert np.allclose(full[slice_row, slice_col], box, equal_nan=True)

        del (
            even_zen,
            odd_zen,
            even_az,
            odd_az,
            even_zen_box,
            odd_zen_box,
            even_az_box,
            odd_az_box,
        )


@pytest.mark.requires_test_data
def test_read_incidence_angles_with_zip_file():
    """
    Test that angles for unzip/zip files are the same
    """
    s2_dataset_path = get_sentinel2_l2a_theia_folder()
    s2_dataset = sentinel2.Sentinel2(s2_dataset_path)
    s2_dataset_path_zip = s2_dataset_path + ".zip"
    s2_dataset_zip = sentinel2.Sentinel2(s2_dataset_path_zip)

    # Read angle for regular file
    angles = s2_dataset.read_incidence_angles_as_numpy()

    # Read angle for zip file
    angles_zip = s2_dataset_zip.read_incidence_angles_as_numpy()

    for angle, angle_zip in zip(angles, angles_zip):
        assert np.allclose(angle, angle_zip, equal_nan=True)

