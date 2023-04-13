#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests for the Ecostress driver
"""
import datetime
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import rasterio as rio
from sensorsio import master


def get_master_files() -> str:
    """
    Retrieve Master files from env var
    """
    master_l2_dir = os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'master',
                                 'MASTERL2_2193800_03_20210330_1832_1846_V01')
    master_l1b_file = os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'master',
                                   'MASTERL1B_2193800_03_20210330_1832_1846_V01.hdf')

    return master_l1b_file, master_l2_dir


@pytest.mark.requires_test_data
def test_master_instantiate():
    """
    Test Ecostress class instantiation 
    """
    master_files = get_master_files()
    master_ds = master.Master(*master_files)

    assert master_ds.l1b_file == master_files[0]
    assert master_ds.l2a_dir == master_files[1]
    assert master_ds.acquisition_date.to_pydatetime() == datetime.datetime(2021, 3, 30)

    assert master_ds.bounds == rio.coords.BoundingBox(-120.88484977370473, 37.89715853677833,
                                                      -119.2885910789107, 39.177174765508546)
    assert master_ds.crs == '+proj=latlon'


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """
    crs: str = None
    resolution: float = 30
    region: Tuple[int, int, int, int] = (0, 0, 100, 100)
    bounds: rio.coords.BoundingBox = None
    no_data_value: float = np.nan
    nprocs: int = 4
    strip_size: int = 375000
    dtype: np.dtype = np.dtype('float32')

    def expected_shape(self) -> Tuple[int, int]:
        """
        return expected shape
        """
        if self.region is not None:
            if isinstance(self.region, rio.coords.BoundingBox):
                return (int((self.region[3] - self.region[1]) / self.resolution),
                        int((self.region[2] - self.region[0]) / self.resolution))
            return (228, 215)

        if self.bounds is not None:
            return (int(np.ceil((self.bounds[3] - self.bounds[1]) / self.resolution)),
                    int(np.ceil((self.bounds[2] - self.bounds[0]) / self.resolution)))


@pytest.mark.parametrize(
    "parameters",
    [
        # Use region to restrict source reading with bounding box
        ReadAsNumpyParams(),
        ReadAsNumpyParams(region=None,
                          crs='epsg:32611',
                          bounds=rio.coords.BoundingBox(
                              left=158380.0, bottom=4196874., right=302311., top=4343632.))
    ])
def test_read_as_numpy_xarray(parameters: ReadAsNumpyParams):
    """
    Test the read_as_numpy method
    """
    master_dataset = master.Master(*get_master_files())

    lst, emis, angles, xcoords, ycoords, crs = master_dataset.read_as_numpy(**parameters.__dict__)
    master_xr = master_dataset.read_as_xarray(**parameters.__dict__)

    assert xcoords.shape == (parameters.expected_shape()[1], )
    assert ycoords.shape == (parameters.expected_shape()[0], )
    assert lst.shape == (*parameters.expected_shape(), 1)
    assert emis.shape == (*parameters.expected_shape(), 5)
    assert angles.shape == (*parameters.expected_shape(), 4)
    assert crs == 'epsg:32611'

    assert 'LST' in master_xr.variables
    for i in range(5):
        assert f'Emis{i+1}' in master_xr.variables
    for v in ['Solar_Azimuth', 'Solar_Zenith', 'View_Azimuth', 'View_Zenith']:
        assert v in master_xr.variables
