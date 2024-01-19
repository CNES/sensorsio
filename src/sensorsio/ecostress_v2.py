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

import datetime
import glob
import os
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from dateutil.parser import parse as parse_date

from sensorsio import regulargrid


class EcostressV2:
    """
    Class for ECOSTRESS product reading (Collection 2)
    """
    def __init__(self, product_dir: str):
        """
        Constructor

        :param product_dir: Path to product directory
        """
        if not os.path.isdir(product_dir):
            raise Exception(f"Error to access product directory {product_dir}")
        if product_dir.endswith(os.sep):
            product_dir = product_dir[:-1]
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        elts = self.product_name.split("_")
        self.tile = elts[5]
        dt = parse_date(elts[6])
        self.date = datetime.date(dt.year, dt.month, dt.day)
        self.time = dt.time()
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday

        with rio.open(self.build_band_path(EcostressV2.LST)) as ds:
            # Get bounds
            self.bounds = ds.bounds
            self.transform = ds.transform
            # Get crs
            self.crs = ds.crs

    def __repr__(self):
        return f'ECOSTRESS (Collection 2) {self.date} {self.tile} {self.crs}'

    # Enum class for ECOSTRESS bands
    class Band(Enum):
        LST = 'LST'
        EMIS = 'EmisWB'
        LST_ERR = 'LST_err'

    # Aliases
    LST = Band.LST
    EMIS = Band.EMIS
    LST_ERR = Band.LST_ERR

    GROUP_ALL = [LST, EMIS, LST_ERR]

    class Mask(Enum):
        CLOUDS = 'cloud'
        QUALITY = 'QC'
        WATER = 'water'

    # Aliases
    QUALITY = Mask.QUALITY
    CLOUDS = Mask.CLOUDS
    WATER = Mask.WATER

    ALL_MASKS = [QUALITY, CLOUDS, WATER]

    # From https://ecostress.jpl.nasa.gov/downloads/psd/ECOSTRESS_SDS_PSD_L2_ver1-1.pdf
    FACTORS = {LST: 1.0, EMIS: 1.0, LST_ERR: 1.0}

    SHIFTS = {LST: 0.0, EMIS: 0.0, LST_ERR: 0.0}

    NO_DATA_FLAGS = {LST: 0, EMIS: 0, LST_ERR: 0}

    def build_band_path(self, band: Union[Band, Mask]) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        
        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}{os.sep}*{band.value}.tif")
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
        resolution: float = 70,
        no_data_value: float = np.nan,
        bounds: Optional[rio.coords.BoundingBox] = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype('float32'),
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[str]]:
        """
        Read bands from ECOSTRESS products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
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
        Read bands from Sentinel2 products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
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
