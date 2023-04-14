#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests for the Landsat-8 driver
"""
import datetime
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import rasterio as rio
from sensorsio import landsat


def get_landsat_c2L2_folder() -> str:
    """
    Retrieve SRTM folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'landsat8', 'c2l2',
                        'LC08_L2SP_118051_20200206_20200823_02_T2')


@pytest.mark.requires_test_data
def test_venus_instantiate_c2l2():
    """
    Test sentinel2 class instantiation 
    """
    ls8 = landsat.Landsat(get_landsat_c2L2_folder())

    assert ls8.product_dir == get_landsat_c2L2_folder()
    assert ls8.product_name == 'LC08_L2SP_118051_20200206_20200823_02_T2'
    assert ls8.date == datetime.datetime(2020, 2, 6)
    assert ls8.year == 2020
    assert ls8.day_of_year == 37
    assert ls8.crs == 'epsg:32650'
    assert ls8.bounds == rio.coords.BoundingBox(left=437085.0,
                                                bottom=1322085.0,
                                                right=664215.0,
                                                top=1554315.0)


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """
    bands: List[landsat.Landsat.Band] = field(
        default_factory=lambda: landsat.Landsat.GROUP_SR + landsat.Landsat.GROUP_ST)
    masks: List[landsat.Landsat.Mask] = field(default_factory=lambda: landsat.Landsat.ALL_MASKS)
    crs: Optional[str] = None
    resolution: float = 30
    region: Union[Tuple[int, int, int, int], rio.coords.BoundingBox] = None
    no_data_value: float = np.nan
    bounds: rio.coords.BoundingBox = None
    algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic
    dtype: np.dtype = np.dtype('float32')

    def expected_shape(self) -> Tuple[int, int]:
        """
        return expected shape
        """
        if self.region is not None:
            if isinstance(self.region, rio.coords.BoundingBox):
                return (int((self.region[3] - self.region[1]) / self.resolution),
                        int((self.region[2] - self.region[0]) / self.resolution))
            return (int((self.region[3] - self.region[1])), int((self.region[2] - self.region[0])))
        if self.bounds is not None:
            return (int(np.ceil((self.bounds[3] - self.bounds[1]) / self.resolution)),
                    int(np.ceil((self.bounds[2] - self.bounds[0]) / self.resolution)))

        raise NotImplementedError


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "parameters",
    [
        # # Use region to restrict source reading with bounding box
        ReadAsNumpyParams(
            region=rio.coords.BoundingBox(left=534000, bottom=1451500., right=534200, top=1451700.)
        ),
        # Use region to restrict source read with pixel coords
        ReadAsNumpyParams(region=[3500, 3500, 3600, 3700]),
        # Use bounds to set output region
        ReadAsNumpyParams(
            bounds=rio.coords.BoundingBox(left=534000, bottom=1451500., right=534200, top=1451700.)
        ),
        # Set a different target crs
        ReadAsNumpyParams(bounds=rio.coords.BoundingBox(left=533975.4843748295,
                                                        bottom=1451497.6588132062,
                                                        right=534175.4845015162,
                                                        top=1451697.6589390924),
                          crs='EPSG:32450')
    ])
def test_read_as_numpy_xarray(parameters: ReadAsNumpyParams):
    """
    Test the read_as_numpy method
    """
    ls8_dataset = landsat.Landsat(get_landsat_c2L2_folder())

    # Read as numpy part
    bands_arr, mask_arr, xcoords, ycoords, crs = ls8_dataset.read_as_numpy(**parameters.__dict__)

    assert bands_arr is not None and bands_arr.shape == (len(
        parameters.bands), *parameters.expected_shape())
    assert mask_arr is not None and mask_arr.shape == (len(
        parameters.masks), *parameters.expected_shape())
    assert (~np.isnan(bands_arr)).sum() > 0

    assert ycoords is not None and ycoords.shape == (parameters.expected_shape()[0], )
    assert xcoords is not None and xcoords.shape == (parameters.expected_shape()[1], )

    if parameters.crs is not None:
        assert crs == parameters.crs
    else:
        assert crs == 'epsg:32650'

    # Test read as xarray part
    ls8_xr = ls8_dataset.read_as_xarray(**parameters.__dict__)

    assert ls8_xr

    for c in ['t', 'x', 'y']:
        assert c in ls8_xr.coords

    assert ls8_xr['t'].shape == (1, )
    assert ls8_xr['x'].shape == (parameters.expected_shape()[1], )
    assert ls8_xr['y'].shape == (parameters.expected_shape()[0], )

    for band in parameters.bands:
        assert band.value in ls8_xr.variables
        assert ls8_xr[band.value].shape == (1, *parameters.expected_shape())

    if parameters.crs is not None:
        assert ls8_xr.attrs['crs'] == parameters.crs
    else:
        assert ls8_xr.attrs['crs'] == ls8_dataset.crs
