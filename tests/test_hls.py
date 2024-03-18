#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
#
# Licensed under the Lesser GNU LESSER GENERAL PUBLIC
# LICENSE, Version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.gnu.org/licenses/lgpl-3.0.txt

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains tests for the HLS driver
"""
import datetime
import os
from dataclasses import dataclass, field
from typing import Optional

import affine
import numpy as np
import pytest
import rasterio as rio
from pyproj import CRS

from sensorsio import hls

HLSL_PRODUCT_NAME = 'HLS.L30.T28PCA.2023086T112728.v2.0'
HLSS_PRODUCT_NAME = 'HLS.S30.T28PCA.2023092T113319.v2.0'


def get_hlsl_folder() -> str:
    """
    Retrieve HLSL folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'hls', HLSL_PRODUCT_NAME)


def get_hlss_folder() -> str:
    """
    Retrieve HLSS folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'hls', HLSS_PRODUCT_NAME)


@pytest.mark.requires_test_data
def test_hlsl_instantiate():
    """
    Test HLSL class instantiation 
    """
    hlsl_folder = get_hlsl_folder()
    hlsl_ds = hls.HLSLandsat(hlsl_folder)

    assert hlsl_ds.product_dir == hlsl_folder
    assert hlsl_ds.product_name == HLSL_PRODUCT_NAME
    assert hlsl_ds.tile == '28PCA'
    assert hlsl_ds.date == datetime.date(2023, 3, 27)
    assert hlsl_ds.time == datetime.time(11, 27, 28)
    assert hlsl_ds.year == 2023
    assert hlsl_ds.day_of_year == 86
    assert hlsl_ds.version == "v2.0"
    assert hlsl_ds.bounds == rio.coords.BoundingBox(left=300000.0,
                                                    bottom=1490220.0,
                                                    right=409800.0,
                                                    top=1600020.0)
    assert hlsl_ds.transform == affine.Affine(30.0, 0.0, 300000.0, 0.0, -30.0, 1600020.0)
    assert hlsl_ds.crs == CRS.from_epsg(32628)


@pytest.mark.requires_test_data
def test_hlss_instantiate():
    """
    Test HLSS class instantiation 
    """
    hlss_folder = get_hlss_folder()
    hlss_ds = hls.HLSSentinel2(hlss_folder)

    assert hlss_ds.product_dir == hlss_folder
    assert hlss_ds.product_name == HLSS_PRODUCT_NAME
    assert hlss_ds.tile == '28PCA'
    assert hlss_ds.date == datetime.date(2023, 4, 2)
    assert hlss_ds.time == datetime.time(11, 33, 19)
    assert hlss_ds.year == 2023
    assert hlss_ds.day_of_year == 92
    assert hlss_ds.version == "v2.0"
    assert hlss_ds.bounds == rio.coords.BoundingBox(left=300000.0,
                                                    bottom=1490220.0,
                                                    right=409800.0,
                                                    top=1600020.0)
    assert hlss_ds.transform == affine.Affine(30.0, 0.0, 300000.0, 0.0, -30.0, 1600020.0)
    assert hlss_ds.crs == CRS.from_epsg(32628)


@dataclass(frozen=True)
class HLSLReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters for HLSL
    """
    bands: list[hls.HLSLandsat.Band] = field(default_factory=lambda: hls.HLSLandsat.GROUP_ALL)
    masks: list[hls.HLSLandsat.Mask] = field(default_factory=lambda: hls.HLSLandsat.ALL_MASKS)
    crs: Optional[str] = None
    resolution: float = 30
    no_data_value: float = np.nan
    bounds: rio.coords.BoundingBox = None
    algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic
    dtype: np.dtype = np.dtype('float32')

    def expected_shape(self) -> tuple[int, int]:
        """
        return expected shape
        """
        if self.bounds is not None:
            return (int((self.bounds[3] - self.bounds[1]) / self.resolution),
                    int((self.bounds[2] - self.bounds[0]) / self.resolution))

        return (int(10980 * 10 / self.resolution), int(10980 * 10 / self.resolution))


@dataclass(frozen=True)
class HLSSReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters for HLSL
    """
    bands: list[hls.HLSSentinel2.Band] = field(default_factory=lambda: hls.HLSSentinel2.GROUP_ALL)
    masks: list[hls.HLSSentinel2.Mask] = field(default_factory=lambda: hls.HLSSentinel2.ALL_MASKS)
    crs: Optional[str] = None
    resolution: float = 30
    no_data_value: float = np.nan
    bounds: rio.coords.BoundingBox = None
    algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic
    dtype: np.dtype = np.dtype('float32')

    def expected_shape(self) -> tuple[int, int]:
        """
        return expected shape
        """
        if self.bounds is not None:
            return (int((self.bounds[3] - self.bounds[1]) / self.resolution),
                    int((self.bounds[2] - self.bounds[0]) / self.resolution))

        return (int(10980 * 10 / self.resolution), int(10980 * 10 / self.resolution))


@pytest.mark.parametrize(
    "parameters",
    [
        HLSLReadAsNumpyParams(),
        # SR bands
        HLSLReadAsNumpyParams(bands=hls.HLSLandsat.GROUP_SR),
        # ST bands
        HLSLReadAsNumpyParams(bands=hls.HLSLandsat.GROUP_ST),
        # Set a bounding bounding
        HLSLReadAsNumpyParams(
            bounds=rio.coords.BoundingBox(400000.0, 1590000.0, 407000.0, 1597000.0)),
        # Set a different target crs
        HLSLReadAsNumpyParams(bounds=rio.coords.BoundingBox(-16.2, 13.0, -15.2, 14.0),
                              crs='EPSG:4326',
                              resolution=0.01)
    ])
def test_hlsl_read_as_numpy_and_xarray(parameters: HLSLReadAsNumpyParams):
    """
    Test the read_as_numpy method for HLS Landsat
    """
    hls_ds = hls.HLSLandsat(get_hlsl_folder())

    # Read as numpy part
    bands_arr, mask_arr, xcoords, ycoords, crs = hls_ds.read_as_numpy(**parameters.__dict__)

    assert bands_arr.shape == (len(parameters.bands), *parameters.expected_shape())
    assert mask_arr is not None and mask_arr.shape == (len(
        parameters.masks), *parameters.expected_shape())
    assert (~np.isnan(bands_arr)).sum() > 0

    assert ycoords.shape == (parameters.expected_shape()[0], )
    assert xcoords.shape == (parameters.expected_shape()[1], )

    if parameters.crs is not None:
        assert crs == parameters.crs

    # Test read as xarray part
    hls_xr = hls_ds.read_as_xarray(**parameters.__dict__)

    for c in ['t', 'x', 'y']:
        assert c in hls_xr.coords

    assert hls_xr['t'].shape == (1, )
    assert hls_xr['x'].shape == (parameters.expected_shape()[1], )
    assert hls_xr['y'].shape == (parameters.expected_shape()[0], )

    for band in parameters.bands:
        assert band.value in hls_xr.variables
        assert hls_xr[band.value].shape == (1, *parameters.expected_shape())

    assert hls_xr.attrs['tile'] == '28PCA'
    if parameters.crs is not None:
        assert hls_xr.attrs['crs'] == parameters.crs


@pytest.mark.parametrize(
    "parameters",
    [
        HLSSReadAsNumpyParams(),
        # SR bands
        HLSSReadAsNumpyParams(bands=hls.HLSSentinel2.GROUP_SR),
        # Set a bounding bounding
        HLSSReadAsNumpyParams(
            bounds=rio.coords.BoundingBox(400000.0, 1590000.0, 407000.0, 1597000.0)),
        # Set a different target crs
        HLSSReadAsNumpyParams(bounds=rio.coords.BoundingBox(-16.2, 13.0, -15.2, 14.0),
                              crs='EPSG:4326',
                              resolution=0.01)
    ])
def test_hlss_read_as_numpy_and_xarray(parameters: HLSLReadAsNumpyParams):
    """
    Test the read_as_numpy method for HLS Sentinel2
    """
    hls_ds = hls.HLSSentinel2(get_hlss_folder())

    # Read as numpy part
    bands_arr, mask_arr, xcoords, ycoords, crs = hls_ds.read_as_numpy(**parameters.__dict__)

    assert bands_arr.shape == (len(parameters.bands), *parameters.expected_shape())
    assert mask_arr is not None and mask_arr.shape == (len(
        parameters.masks), *parameters.expected_shape())
    assert (~np.isnan(bands_arr)).sum() > 0

    assert ycoords.shape == (parameters.expected_shape()[0], )
    assert xcoords.shape == (parameters.expected_shape()[1], )

    if parameters.crs is not None:
        assert crs == parameters.crs

    # Test read as xarray part
    hls_xr = hls_ds.read_as_xarray(**parameters.__dict__)

    for c in ['t', 'x', 'y']:
        assert c in hls_xr.coords

    assert hls_xr['t'].shape == (1, )
    assert hls_xr['x'].shape == (parameters.expected_shape()[1], )
    assert hls_xr['y'].shape == (parameters.expected_shape()[0], )

    for band in parameters.bands:
        assert band.value in hls_xr.variables
        assert hls_xr[band.value].shape == (1, *parameters.expected_shape())

    assert hls_xr.attrs['tile'] == '28PCA'
    if parameters.crs is not None:
        assert hls_xr.attrs['crs'] == parameters.crs
