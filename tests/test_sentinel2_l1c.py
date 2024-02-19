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
import pytest
import rasterio as rio
from sensorsio import mgrs, sentinel2_l1c


def get_sentinel2_l1c_folder() -> str:
    """
    Retrieve Sentinel2 folder from env var
    """
    return os.path.join(
        os.environ["SENSORSIO_TEST_DATA_PATH"],
        "sentinel2",
        "l1c",
        "S2B_MSIL1C_20231126T105309_N0509_R051_T31TCJ_20231126T113937.SAFE",
    )


@pytest.mark.requires_test_data
def test_sentinel2_instantiate_l1c():
    """
    Test sentinel2 class instantiation
    """
    s2 = sentinel2_l1c.Sentinel2L1C(get_sentinel2_l1c_folder())

    assert s2.product_dir == get_sentinel2_l1c_folder()
    assert (
        s2.product_name
        == "S2B_MSIL1C_20231126T105309_N0509_R051_T31TCJ_20231126T113937"
    )
    assert s2.date == datetime.datetime(2023, 11, 26, 10, 53, 9)
    assert s2.year == 2023
    assert s2.day_of_year == 330
    assert s2.tile == "31TCJ"
    assert s2.cloud_cover == 7
    assert s2.satellite == sentinel2_l1c.Sentinel2L1C.Satellite.S2B
    assert s2.crs == "epsg:32631"
    assert s2.bounds == mgrs.get_bbox_mgrs_tile(s2.tile, latlon=False)
    assert s2.transform == mgrs.get_transform_mgrs_tile(s2.tile)
    assert s2.relative_orbit_number == 25


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """

    bands: List[sentinel2_l1c.Sentinel2L1C.Band] = field(
        default_factory=lambda: sentinel2_l1c.Sentinel2L1C.GROUP_10M
    )
    masks: List[sentinel2_l1c.Sentinel2L1C.Mask] = field(
        default_factory=lambda: sentinel2_l1c.Sentinel2L1C.ALL_MASKS
    )
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
            bands=sentinel2_l1c.Sentinel2L1C.GROUP_20M,
            bounds=rio.coords.BoundingBox(300000.0, 4790220.0, 301000, 4792220),
        ),
        # Use bounds to set output region, with 20m bands and 10 bands
        ReadAsNumpyParams(
            bands=sentinel2_l1c.Sentinel2L1C.GROUP_10M
            + sentinel2_l1c.Sentinel2L1C.GROUP_20M,
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
    s2_dataset = sentinel2_l1c.Sentinel2L1C(get_sentinel2_l1c_folder())

    # Read as numpy part
    bands_arr, mask_arr, xcoords, ycoords, crs = s2_dataset.read_as_numpy(
        **parameters.__dict__
    )

    assert bands_arr.shape == (len(parameters.bands), *parameters.expected_shape())
    assert mask_arr is not None and mask_arr.shape == (
        len(parameters.masks),
        *parameters.expected_shape(),
    )
    assert (~np.isnan(bands_arr)).sum() > 0

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

    assert s2_xr.attrs["tile"] == "31TCJ"
    if parameters.crs is not None:
        assert s2_xr.attrs["crs"] == parameters.crs
    else:
        assert s2_xr.attrs["crs"] == s2_dataset.crs
