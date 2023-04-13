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
from sensorsio import ecostress


def get_ecostress_files() -> str:
    """
    Retrieve ecostress files from env var
    """
    ecostress_dir = os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'ecostress',
                                 'ECOSTRESS__10025_003_20200412T102039_0601_01')
    return (
        os.path.join(ecostress_dir, 'ECOSTRESS_L2_LSTE_10025_003_20200412T102039_0601_01.h5'),
        os.path.join(ecostress_dir, 'ECOSTRESS_L1B_GEO_10025_003_20200412T102039_0601_01.h5'),
        os.path.join(ecostress_dir, 'ECOSTRESS_L2_CLOUD_10025_003_20200412T102039_0601_01.h5'),
        os.path.join(ecostress_dir, 'ECOSTRESS_L1B_RAD_10025_003_20200412T102039_0601_01.h5'),
    )


@pytest.mark.requires_test_data
def test_ecostress_instantiate():
    """
    Test Ecostress class instantiation 
    """
    ecostress_files = get_ecostress_files()
    eco_ds = ecostress.Ecostress(*ecostress_files)

    assert eco_ds.lst_file == ecostress_files[0]
    assert eco_ds.geom_file == ecostress_files[1]
    assert eco_ds.cloud_file == ecostress_files[2]
    assert eco_ds.rad_file == ecostress_files[3]

    assert eco_ds.start_time == datetime.datetime(2020, 4, 12, 10, 20, 39, 807890)
    assert eco_ds.end_time == datetime.datetime(2020, 4, 12, 10, 21, 31, 777690)

    assert eco_ds.bounds == rio.coords.BoundingBox(-2.125451875064012, 38.87543991023487,
                                                   4.498900548635159, 43.80483839431817)
    assert eco_ds.crs == '+proj=latlon'

    eco_ds = ecostress.Ecostress(*ecostress_files[:-1])

    assert eco_ds.lst_file == ecostress_files[0]
    assert eco_ds.geom_file == ecostress_files[1]
    assert eco_ds.cloud_file == ecostress_files[2]
    assert eco_ds.rad_file is None

    eco_ds = ecostress.Ecostress(*ecostress_files[:-2])

    assert eco_ds.lst_file == ecostress_files[0]
    assert eco_ds.geom_file == ecostress_files[1]
    assert eco_ds.cloud_file is None
    assert eco_ds.rad_file is None


@dataclass(frozen=True)
class ReadAsNumpyParams:
    """
    Class to store read_as_numpy parameters
    """
    crs: Optional[str] = None
    resolution: float = 70
    region: Tuple[int, int, int, int] = (0, 0, 100, 100)
    no_data_value: float = np.nan
    read_lst: bool = True
    read_angles: bool = True
    read_emissivities: bool = True
    bounds: rio.coords.BoundingBox = None
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
            return (183, 179)

        if self.bounds is not None:
            return (int(np.ceil((self.bounds[3] - self.bounds[1]) / self.resolution)),
                    int(np.ceil((self.bounds[2] - self.bounds[0]) / self.resolution)))


@pytest.mark.parametrize(
    "parameters",
    [
        # Use region to restrict source reading with bounding box
        ReadAsNumpyParams(),
        ReadAsNumpyParams(read_lst=False),
        ReadAsNumpyParams(read_angles=False),
        ReadAsNumpyParams(read_emissivities=False),
        ReadAsNumpyParams(read_lst=False, read_angles=False, read_emissivities=False),
        ReadAsNumpyParams(bounds=rio.coords.BoundingBox(300000., 4790220., 301000, 4792220),
                          region=None),
    ])
def test_read_as_numpy_xarray(parameters: ReadAsNumpyParams):
    """
    Test the read_as_numpy method
    """
    ecostress_dataset = ecostress.Ecostress(*get_ecostress_files())

    lst, emis, radiances, angles, qc, masks, xcoords, ycoords, crs = ecostress_dataset.read_as_numpy(
        **parameters.__dict__)
    eco_xr = ecostress_dataset.read_as_xarray(**parameters.__dict__)

    assert crs == 'epsg:32631'
    assert eco_xr.attrs['crs'] == 'epsg:32631'

    assert radiances is not None
    assert radiances.shape == (*parameters.expected_shape(), 5)

    if parameters.read_lst:
        assert lst is not None
        assert lst.shape == (*parameters.expected_shape(), 2)
        assert 'LST' in eco_xr.variables
        assert 'LST_Err' in eco_xr.variables

    else:
        assert lst is None
        assert 'LST' not in eco_xr.variables
        assert 'LST_Err' not in eco_xr.variables

    if parameters.read_angles:
        assert angles is not None
        assert angles.shape == (*parameters.expected_shape(), 4)
        for v in ['Solar_Azimuth', 'Solar_Zenith', 'View_Azimuth', 'View_Zenith']:
            assert v in eco_xr.variables
    else:
        assert angles is None
        for v in ['Solar_Azimuth', 'Solar_Zenith', 'View_Azimuth', 'View_Zenith']:
            assert v not in eco_xr.variables

    if parameters.read_emissivities:
        assert emis is not None
        assert emis.shape == (*parameters.expected_shape(), 10)
        for i in range(5):
            assert f'Emis{i+1}' in eco_xr.variables
            assert f'Emis{i+1}_Err' in eco_xr.variables
    else:
        assert emis is None
        for i in range(5):
            assert f'Emis{i+1}' not in eco_xr.variables
            assert f'Emis{i+1}_Err' not in eco_xr.variables

    for i in range(5):
        assert f'Rad{i+1}' in eco_xr.variables

    for v in ['QC', 'Cloud_Mask', 'Land_Mask', 'Sea_Mask']:
        assert v in eco_xr.variables

    assert xcoords.shape == (parameters.expected_shape()[1], )
    assert ycoords.shape == (parameters.expected_shape()[0], )
    assert qc.shape == parameters.expected_shape()