#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests for the Sentinel2 driver
"""
import datetime
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import rasterio as rio
from sensorsio import mgrs, venus


def get_venus_l2a_theia_folder() -> str:
    """
    Retrieve SRTM folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'venus', 'l2a',
                        'VENUS-XS_20200923-105325-000_L2A_SUDOUE-1_C_V3-1')


def test_get_theia_tiles():
    """
    Test the get theia tiles function
    """
    theia_tiles = venus.get_theia_sites()

    assert len(theia_tiles) == 127


@pytest.mark.requires_test_data
def test_venus_instantiate_l2a_theia():
    """
    Test sentinel2 class instantiation 
    """
    vns = venus.Venus(get_venus_l2a_theia_folder())

    assert vns.product_dir == get_venus_l2a_theia_folder()
    assert vns.product_name == 'VENUS-XS_20200923-105325-000_L2A_SUDOUE-1_C_V3-1'
    assert vns.date == datetime.datetime(2020, 9, 23)
    assert vns.year == 2020
    assert vns.day_of_year == 267
    assert vns.site == 'SUDOUE-1'
    assert vns.satellite == venus.Venus.VN
    assert vns.crs == 'epsg:32631'
    assert vns.bounds == rio.coords.BoundingBox(left=336145.0,
                                                bottom=4806680.0,
                                                right=383945.0,
                                                top=4853555.0)


def test_venus_psf():
    """
    Test the PSF method
    """
    psf = venus.Venus.PSF(venus.Venus.GROUP_5M, half_kernel_width=5)
    assert psf.shape == (12, 11, 11)


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """
    bands: List[venus.Venus.Band] = field(default_factory=lambda: venus.Venus.GROUP_5M)
    band_type: venus.Venus.BandType = venus.Venus.FRE
    masks: List[venus.Venus.Mask] = field(default_factory=lambda: venus.Venus.ALL_MASKS)
    mask_res: venus.Venus.MaskRes = venus.Venus.MaskRes.XS
    scale: float = 1000
    crs: Optional[str] = None
    resolution: float = 10
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
            return (int((self.bounds[3] - self.bounds[1]) / self.resolution),
                    int((self.bounds[2] - self.bounds[0]) / self.resolution))

        return (int(10980 * 10 / self.resolution), int(10980 * 10 / self.resolution))


@pytest.mark.requires_test_data
@pytest.mark.parametrize(
    "parameters",
    [
        # Use region to restrict source reading with bounding box
        ReadAsNumpyParams(region=rio.coords.BoundingBox(
            left=354650., bottom=4828620., right=355650., top=4829620.)),
        # Use region to restrict source reading with bounding box
        ReadAsNumpyParams(region=rio.coords.BoundingBox(
            left=354650., bottom=4828620., right=355650., top=4829620.),
                          mask_res=venus.Venus.MaskRes.XS),
        # Use region to restrict source read with pixel coords
        ReadAsNumpyParams(region=[0, 0, 100, 200]),
        # Use bounds to set output region
        ReadAsNumpyParams(bounds=rio.coords.BoundingBox(
            left=354650., bottom=4828620., right=355650., top=4829620.)),
        # Set a different target crs
        ReadAsNumpyParams(bounds=rio.coords.BoundingBox(
            left=554540.0, bottom=6279120.0, right=555540.0, top=6280120.0),
                          crs='EPSG:2154')
    ])
def test_read_as_numpy_xarray(parameters: ReadAsNumpyParams):
    """
    Test the read_as_numpy method
    """
    vns_dataset = venus.Venus(get_venus_l2a_theia_folder())

    # Read as numpy part
    bands_arr, mask_arr, xcoords, ycoords, crs = vns_dataset.read_as_numpy(**parameters.__dict__)

    assert bands_arr.shape == (len(parameters.bands), *parameters.expected_shape())
    assert mask_arr.shape == (len(parameters.masks), *parameters.expected_shape())
    assert (~np.isnan(bands_arr)).sum() > 0

    assert ycoords.shape == (parameters.expected_shape()[0], )
    assert xcoords.shape == (parameters.expected_shape()[1], )

    if parameters.crs is not None:
        assert crs == parameters.crs
    else:
        assert crs == 'epsg:32631'

    # Test read as xarray part
    s2_xr = vns_dataset.read_as_xarray(**parameters.__dict__)

    for c in ['t', 'x', 'y']:
        assert c in s2_xr.coords

    assert s2_xr['t'].shape == (1, )
    assert s2_xr['x'].shape == (parameters.expected_shape()[1], )
    assert s2_xr['y'].shape == (parameters.expected_shape()[0], )

    for band in parameters.bands:
        assert band.value in s2_xr.variables
        assert s2_xr[band.value].shape == (1, *parameters.expected_shape())

    assert s2_xr.attrs['site'] == 'SUDOUE-1'
    assert s2_xr.attrs['type'] == parameters.band_type.value
    if parameters.crs is not None:
        assert s2_xr.attrs['crs'] == parameters.crs
    else:
        assert s2_xr.attrs['crs'] == vns_dataset.crs
