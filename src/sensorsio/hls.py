#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
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
Module for HLS products
"""
import datetime
import glob
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from dateutil.parser import isoparse as parse_date

from sensorsio import regulargrid


class HLS(ABC):
    """
    Abstract class for HLS product reading
    """
    @abstractmethod
    def __init__(self, product_dir: str):
        """
        Constructor

        :param product_dir: Path to product directory
        """

        if not os.path.isdir(product_dir):
            raise FileNotFoundError(f"Error to access product directory {product_dir}")
        if product_dir.endswith(os.sep):
            product_dir = product_dir[:-1]
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        elts = self.product_name.split(".")
        self.tile = elts[2][1:]
        dt = parse_date(elts[3])
        self.date = datetime.date(dt.year, dt.month, dt.day)
        self.time = dt.time()
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday
        self.version = ".".join(elts[4:])

        with rio.open(self.build_band_path(HLS.B1)) as ds:
            # Get bounds
            self.bounds = ds.bounds
            self.transform = ds.transform
            # Get crs
            self.crs = ds.crs

    class Band(Enum):
        """
        Band list
        """
        B1 = 'B01'
        B2 = 'B02'
        B3 = 'B03'
        B4 = 'B04'
        B5 = 'B05'
        B6 = 'B06'
        B7 = 'B07'
        B8 = 'B08'
        B8A = 'B8A'
        B9 = 'B09'
        B10 = 'B10'
        B11 = 'B11'
        B12 = 'B12'

    class Mask(Enum):
        """
        Mask list
        """
        QA = 'Fmask'

    # Aliases
    B1 = Band.B1
    QA = Mask.QA
    ALL_MASKS = [QA]
    FACTORS = {B1: 1.0}
    SHIFTS = {B1: 0.0}
    NO_DATA_FLAGS = {B1: -9999}
    GROUP_ALL = [B1]

    @abstractmethod
    def __repr__(self):
        return f'HLS {self.date} {self.crs}'

    def build_band_path(self, band: Union[Band, Mask]) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        
        :return: The path to the band file
        """
        if not band.name in [b.name for b in self.GROUP_ALL + self.ALL_MASKS ]:
            raise ValueError(f"Band {band} not in {self.__class__} product")
        p = glob.glob(f"{self.product_dir}/*{band.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(
                f"Could not find band {band.value} in product directory {self.product_dir}")
        return p[0]

    def read_as_numpy(
        self,
        bands: List[Band],
        masks: List[Mask] = ALL_MASKS,
        crs: Optional[str] = None,
        resolution: float = 30,
        no_data_value: float = np.nan,
        bounds: Optional[rio.coords.BoundingBox] = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype('float32'),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[str]]:
        """
        Read bands from HLS products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height],
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        # Readn bands
        np_arr = None
        xcoords = None
        ycoords = None
        out_crs = crs
        if len(bands):
            img_files = [self.build_band_path(b) for b in bands]
            np_arr, xcoords, ycoords, out_crs = regulargrid.read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                output_no_data_value=no_data_value,
                bounds=bounds,
                algorithm=algorithm,
                separate=True,
                dtype=dtype)

            factors = np.array([self.FACTORS[b] for b in bands])
            shifts = np.array([self.SHIFTS[b] for b in bands])

            # Skip first dimension
            np_arr = np_arr[0, ...]

            np_arr_rescaled = (factors[:, None, None] * np_arr) + shifts[:, None, None]

            for i, b in enumerate(bands):
                np_arr_rescaled[i, ...][np_arr[i, ...] == self.NO_DATA_FLAGS[b]] = no_data_value
            np_arr = np_arr_rescaled

        np_arr_msk = None
        if len(masks):
            img_files = [self.build_band_path(m) for m in masks]
            np_arr_msk, xcoords, ycoords, crs = regulargrid.read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                output_no_data_value=no_data_value,
                bounds=bounds,
                algorithm=rio.enums.Resampling.nearest,
                separate=True,
                dtype=np.uint16,
                scale=None)
            # Drop first dimension
            np_arr_msk = np_arr_msk[0, ...]
        return np_arr, np_arr_msk, xcoords, ycoords, out_crs

    def read_as_xarray(
        self,
        bands: List[Band],
        masks: List[Mask] = ALL_MASKS,
        crs: Optional[str] = None,
        resolution: float = 30,
        no_data_value: float = np.nan,
        bounds: Optional[rio.coords.BoundingBox] = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype('float32')
    ) -> Optional[xr.Dataset]:
        """
        Read bands from HLS products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return:
        """
        np_arr, np_arr_msk, xcoords, ycoords, crs = self.read_as_numpy(
            bands, masks, crs, resolution, no_data_value, bounds, algorithm, dtype)

        vars = {}
        if np_arr is not None:
            for i in range(len(bands)):
                vars[bands[i].value] = (["t", "y", "x"], np_arr[None, i, ...])
        if np_arr_msk is not None:
            for i in range(len(masks)):
                vars[masks[i].value] = (["t", "y", "x"], np_arr_msk[None, i, ...])

        if xcoords is not None and ycoords is not None:
            xarr = xr.Dataset(vars,
                              coords={
                                  't': [self.date],
                                  'x': xcoords,
                                  'y': ycoords
                              },
                              attrs={
                                  'crs': crs,
                                  'tile': self.tile
                              })
        else:
            xarr = None
        return xarr


class HLSLandsat(HLS):
    """
    Class for HLS Landsat  product reading
    """
    def __init__(self, product_dir: str):
        """
        Constructor

        :param product_dir: Path to product directory
        """
        super().__init__(product_dir)

    def __repr__(self):
        return f'HLS Landsat {self.date} {self.crs}'

    # Aliases for HLS Landsat bands
    B1 = HLS.Band.B1
    B2 = HLS.Band.B2
    B3 = HLS.Band.B3
    B4 = HLS.Band.B4
    B5 = HLS.Band.B5
    B6 = HLS.Band.B6
    B7 = HLS.Band.B7
    B9 = HLS.Band.B9
    B10 = HLS.Band.B10
    B11 = HLS.Band.B11

    GROUP_SR = [B1, B2, B3, B4, B5, B6, B7, B9]
    GROUP_ST = [B10, B11]
    GROUP_ALL = [B1, B2, B3, B4, B5, B6, B7, B9, B10, B11]

    # From https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
    FACTORS = {
        B1: 0.0001,
        B2: 0.0001,
        B3: 0.0001,
        B4: 0.0001,
        B5: 0.0001,
        B6: 0.0001,
        B7: 0.0001,
        B9: 0.0001,
        B10: 0.01,
        B11: 0.01,
    }

    SHIFTS = {
        B1: 0.0,
        B2: 0.0,
        B3: 0.0,
        B4: 0.0,
        B5: 0.0,
        B6: 0.0,
        B7: 0.0,
        B9: 0.0,
        B10: 273.15,  # Celsius to Kelvin
        B11: 273.15,  # Celsius to Kelvin
    }

    NO_DATA_FLAGS = {
        B1: -9999,
        B2: -9999,
        B3: -9999,
        B4: -9999,
        B5: -9999,
        B6: -9999,
        B7: -9999,
        B9: -9999,
        B10: -9999,
        B11: -9999,
    }


class HLSSentinel2(HLS):
    """
    Class for HLS Sentinel2  product reading
    """
    def __init__(self, product_dir: str):
        """
        Constructor

        :param product_dir: Path to product directory
        """
        super().__init__(product_dir)

    def __repr__(self):
        return f'HLS Sentinel2 {self.date} {self.crs}'

    # Aliases for HLS Sentinel2 bands
    B1 = HLS.Band.B1
    B2 = HLS.Band.B2
    B3 = HLS.Band.B3
    B4 = HLS.Band.B4
    B5 = HLS.Band.B5
    B6 = HLS.Band.B6
    B7 = HLS.Band.B7
    B8 = HLS.Band.B8
    B8A = HLS.Band.B8A
    B9 = HLS.Band.B9
    B10 = HLS.Band.B10
    B11 = HLS.Band.B11
    B12 = HLS.Band.B12

    GROUP_SR = [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
    GROUP_ALL = GROUP_SR

    # From https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf
    FACTORS = {
        B1: 0.0001,
        B2: 0.0001,
        B3: 0.0001,
        B4: 0.0001,
        B5: 0.0001,
        B6: 0.0001,
        B7: 0.0001,
        B8: 0.0001,
        B8A: 0.0001,
        B9: 0.0001,
        B10: 0.0001,
        B11: 0.0001,
        B12: 0.0001,
    }

    SHIFTS = {
        B1: 0.0,
        B2: 0.0,
        B3: 0.0,
        B4: 0.0,
        B5: 0.0,
        B6: 0.0,
        B7: 0.0,
        B8: 0.0,
        B8A: 0.0,
        B9: 0.0,
        B10: 0.0,
        B11: 0.0,
        B12: 0.0,
    }

    NO_DATA_FLAGS = {
        B1: -9999,
        B2: -9999,
        B3: -9999,
        B4: -9999,
        B5: -9999,
        B6: -9999,
        B7: -9999,
        B8: -9999,
        B8A: -9999,
        B9: -9999,
        B10: -9999,
        B11: -9999,
        B12: -9999,
    }
