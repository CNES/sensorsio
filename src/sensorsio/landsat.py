#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import os, glob
from typing import List, Union, Tuple
from enum import Enum
import dateutil
import rasterio as rio
import numpy as np
import xarray as xr
from sensorsio import utils


class Landsat:
    """
    Class for Landsat L2  product reading
    """
    def __init__(self, product_dir: str):
        """
        Constructor

        :param product_dir: Path to product directory
        """
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        self.date = dateutil.parser.parse(self.product_name[17:25])
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday

        with rio.open(self.build_band_path(Landsat.B1)) as ds:
            # Get bounds
            self.bounds = ds.bounds
            self.transform = ds.transform
            # Get crs
            self.crs = ds.crs

    def __repr__(self):
        return f'Landsat {self.date} {self.crs}'

    # Enum class for Sentinel2 bands
    class Band(Enum):
        B1 = 'SR_B1'
        B2 = 'SR_B2'
        B3 = 'SR_B3'
        B4 = 'SR_B4'
        B5 = 'SR_B5'
        B6 = 'SR_B6'
        B7 = 'SR_B7'
        B10 = 'ST_B10'
        ST_QA = 'ST_QA'
        ST_TRAD = 'ST_TRAD'
        ST_URAD = 'ST_URAD'
        ST_DRAD = 'ST_DRAD'
        ST_ATRAN = 'ST_ATRAN'
        ST_EMIS = 'ST_EMIS'
        ST_EMISD = 'ST_EMISD'
        ST_CDIST = 'ST_CDIST'

    # Aliases
    B1 = Band.B2
    B2 = Band.B2
    B3 = Band.B3
    B4 = Band.B4
    B5 = Band.B5
    B6 = Band.B6
    B7 = Band.B7
    B10 = Band.B10
    ST_QA = Band.ST_QA
    ST_TRAD = Band.ST_TRAD
    ST_URAD = Band.ST_URAD
    ST_DRAD = Band.ST_DRAD
    ST_ATRAN = Band.ST_ATRAN
    ST_EMIS = Band.ST_EMIS
    ST_EMISD = Band.ST_EMISD
    ST_CDIST = Band.ST_CDIST

    GROUP_SR = [B1, B2, B3, B4, B5, B6, B7]
    GROUP_ST = [B10]

    class Mask(Enum):
        CLOUDS = 'QA_PIXEL'
        AEROSOLS = 'QA_AEROSOL'
        SATURATIONS = 'QA_RADSAT'

    # Aliases
    CLOUDS = Mask.CLOUDS
    AEROSOLS = Mask.AEROSOLS
    SATURATIONS = Mask.SATURATIONS

    ALL_MASKS = [CLOUDS, AEROSOLS, SATURATIONS]

    # From https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1619_Landsat8-C2-L2-ScienceProductGuide-v2.pdf
    FACTORS = {
        B1: 0.0000275,
        B2: 0.0000275,
        B3: 0.0000275,
        B4: 0.0000275,
        B5: 0.0000275,
        B6: 0.0000275,
        B7: 0.0000275,
        B10: 0.00341802,
        ST_QA: 0.01,
        ST_TRAD: 0.001,
        ST_URAD: 0.001,
        ST_DRAD: 0.001,
        ST_ATRAN: 0.0001,
        ST_EMIS: 0.0001,
        ST_EMISD: 0.0001,
        ST_CDIST: 0.01
    }

    SHIFTS = {
        B1: -0.2,
        B2: -0.2,
        B3: -0.2,
        B4: -0.2,
        B5: -0.2,
        B6: -0.2,
        B7: -0.2,
        B10: 149,
        ST_QA: 0,
        ST_TRAD: 0,
        ST_URAD: 0,
        ST_DRAD: 0,
        ST_ATRAN: 0,
        ST_EMIS: 0,
        ST_EMISD: 0,
        ST_CDIST: 0
    }

    NO_DATA_FLAGS = {
        B1: 0,
        B2: 0,
        B3: 0,
        B4: 0,
        B5: 0,
        B6: 0,
        B7: 0,
        B10: 0,
        ST_QA: -9999,
        ST_TRAD: -9999,
        ST_URAD: -9999,
        ST_DRAD: -9999,
        ST_ATRAN: -9999,
        ST_EMIS: -9999,
        ST_EMISD: -9999,
        ST_CDIST: -9999
    }

    def build_band_path(self, band: Union[Band, Mask]) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        
        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}/*{band.value}.TIF")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(
                f"Could not find band {band.value} in product directory {self.product_dir}"
            )
        return p[0]

    def read_as_numpy(
        self,
        bands: List[Band],
        masks: List[Mask] = ALL_MASKS,
        crs: str = None,
        resolution: float = 30,
        region: Union[Tuple[int, int, int, int],
                      rio.coords.BoundingBox] = None,
        no_data_value: float = np.nan,
        bounds: rio.coords.BoundingBox = None,
        algorithm=rio.enums.Resampling.cubic,
        dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
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
        :return: The image pixels as a np.ndarray of shape [bands, width, height],
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        np_arr = None
        np_arr_msk = None
        xcoords = None
        ycoords = None
        crs = None

        # Readn bands
        if len(bands):
            img_files = [self.build_band_path(b) for b in bands]
            np_arr, xcoords, ycoords, crs = utils.read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                region=region,
                output_no_data_value=no_data_value,
                bounds=bounds,
                algorithm=algorithm,
                separate=True,
                dtype=dtype)

            factors = np.array([self.FACTORS[b] for b in bands])
            shifts = np.array([self.SHIFTS[b] for b in bands])

            # Skip first dimension
            np_arr = np_arr[0, ...]

            np_arr_rescaled = (factors[:, None, None] *
                               np_arr) + shifts[:, None, None]

            for i, b in enumerate(bands):
                np_arr_rescaled[i, ...][np_arr[i, ...] ==
                                        self.NO_DATA_FLAGS[b]] = no_data_value
            np_arr = np_arr_rescaled

        if len(masks):
            img_files = [self.build_band_path(m) for m in masks]
            np_arr_msk, xcoords, ycoords, crs = utils.read_as_numpy(
                img_files,
                crs=crs,
                resolution=resolution,
                region=region,
                output_no_data_value=no_data_value,
                bounds=bounds,
                algorithm=rio.enums.Resampling.nearest,
                separate=True,
                dtype=np.uint16,
                scale=None)
            # Drop first dimension
            np_arr_msk = np_arr_msk[0, ...]
        return np_arr, np_arr_msk, xcoords, ycoords, crs

    def read_as_xarray(self,
                       bands: List[Band],
                       masks: List[Mask] = ALL_MASKS,
                       crs: str = None,
                       resolution: float = 30,
                       region: Union[Tuple[int, int, int, int],
                                     rio.coords.BoundingBox] = None,
                       no_data_value: float = np.nan,
                       bounds: rio.coords.BoundingBox = None,
                       algorithm=rio.enums.Resampling.cubic,
                       dtype: np.dtype = np.float32) -> xr.Dataset:
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
            bands, masks, crs, resolution, region, no_data_value, bounds,
            algorithm, dtype)

        vars = {}
        for i in range(len(bands)):
            vars[bands[i].value] = (["t", "y", "x"], np_arr[None, i, ...])
            for i in range(len(masks)):
                vars[masks[i].value] = (["t", "y", "x"], np_arr_msk[None, i,
                                                                    ...])

        xarr = xr.Dataset(vars,
                          coords={
                              't': [self.date],
                              'x': xcoords,
                              'y': ycoords
                          },
                          attrs={'crs': crs})
        return xarr