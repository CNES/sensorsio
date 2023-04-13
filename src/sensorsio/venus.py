#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import glob
import os
from enum import Enum
from typing import List, Tuple, Union

import dateutil
import geopandas as gpd
import numpy as np
import rasterio as rio
import xarray as xr

from sensorsio import utils
"""
This module contains Venus (L2A MAJA) related functions
"""


def get_theia_sites():
    """
    Return a dataframe with tiles produced by Theia
    """
    return gpd.read_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data/venus/theia_venus.gpkg')).set_index('Name')


class Venus:
    """
    Class for Venus L2A (MAJA format) product reading
    """
    def __init__(self, product_dir, offsets: Tuple[float] = None):
        """
        Constructor
        """
        # Store product DIR
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        # Store offsets
        self.offsets = offsets

        # Get
        self.satellite = Venus.Satellite(self.product_name[0:5])

        # Get site
        self.site = self.product_name[33:41]

        # Get acquisition date
        self.date = dateutil.parser.parse(self.product_name[9:17])
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday

        with rio.open(self.build_band_path(Venus.B2)) as ds:
            # Get bounds
            self.bounds = ds.bounds
            # Get crs
            self.crs = ds.crs

        # TODO: decode cloud cover as well

    def __repr__(self):
        return f'{self.satellite.value}, {self.date}, {self.site}'

    # Enum class for sensor
    class Satellite(Enum):
        VN = 'VENUS'

    # Aliases
    VN = Satellite.VN

    # Enum class for Sentinel2 bands
    class Band(Enum):
        B1 = 'B1'
        B2 = 'B2'
        B3 = 'B3'
        B4 = 'B4'
        B5 = 'B5'
        B6 = 'B6'
        B7 = 'B7'
        B8 = 'B8'
        B9 = 'B9'
        B10 = 'B10'
        B11 = 'B11'
        B12 = 'B12'

    # Aliases
    B1 = Band.B1
    B2 = Band.B2
    B3 = Band.B3
    B4 = Band.B4
    B5 = Band.B5
    B6 = Band.B6
    B7 = Band.B7
    B8 = Band.B8
    B9 = Band.B9
    B10 = Band.B10
    B11 = Band.B11
    B12 = Band.B12

    # Enum class for Sentinel2 L2A masks
    class Mask(Enum):
        SAT = 'SAT'
        CLM = 'CLM'
        EDG = 'EDG'
        MG2 = 'MG2'

    # Aliases
    SAT = Mask.SAT
    CLM = Mask.CLM
    EDG = Mask.EDG
    MG2 = Mask.MG2

    # Enum class for mask resolutions
    class MaskRes(Enum):
        XS = 'XS'

    # Aliases for resolution
    XS = MaskRes.XS
    #R2 = MaskRes.R2

    # Band groups
    GROUP_5M = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12]
    ALL_MASKS = [SAT, CLM, EDG, MG2]

    # Enum for BandType
    class BandType(Enum):
        FRE = 'FRE'
        SRE = 'SRE'

    # Aliases for band type
    FRE = BandType.FRE
    SRE = BandType.SRE

    # Use of the MTF Values of Sentinel2A-2B except for B1 and B10 set in 0.2
    # MTF
    MTF = {
        B1: 0.2,
        B2: 0.304,
        B3: 0.276,
        B4: 0.233,
        B5: 0.343,
        B6: 0.336,
        B7: 0.338,
        B8: 0.222,
        B9: 0.39,
        B10: 0.2,
        B11: 0.21,
        B12: 0.19
    }

    # Resolution
    RES = {B1: 5, B2: 5, B3: 5, B4: 5, B5: 5, B6: 5, B7: 5, B8: 5, B9: 5, B10: 5, B11: 5, B12: 5}

    def PSF(bands: List[Band], resolution: float = 0.5, half_kernel_width: int = None):
        """
        Generate PSF kernels from list of bands

        :param bands: A list of VENUS Band Enum to generate PSF kernel for
        :param resolution: Resolution at which to sample the kernel
        :param half_kernel_width: The half size of the kernel
                                  (determined automatically if None)

        :return: The kernels as a Tensor of shape
                 [len(bands),2*half_kernel_width+1, 2*half_kernel_width+1]
        """
        return np.stack([(utils.generate_psf_kernel(resolution, Venus.RES[b], Venus.MTF[b],
                                                    half_kernel_width)) for b in bands])

    def build_xml_path(self) -> str:
        """
        Return path to root xml file
        """
        p = glob.glob(f"{self.product_dir}/*MTD_ALL.xml")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(
                f"Could not find root XML file in product directory {self.product_dir}")
        return p[0]

    def build_band_path(self, band: Band, band_type: BandType = FRE) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}/*{band_type.value}_{band.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(
                f"Could not find band {band.value} of type {band_type.value} in product directory {self.product_dir}"
            )
        return p[0]

    def build_mask_path(self, mask: Mask, resolution: MaskRes = XS) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}/MASKS/*{mask.value}_{resolution.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(
                f"Could not find mask {mask.value} of resolution {resolution.value} in product directory {self.product_dir}"
            )
        return p[0]

    def read_as_numpy(
            self,
            bands: List[Band],
            band_type: BandType = FRE,
            masks: List[Mask] = ALL_MASKS,
            mask_res: MaskRes = MaskRes.XS,
            scale: float = 1000,
            crs: str = None,
            resolution: float = 10,
            region: Union[Tuple[int, int, int, int], rio.coords.BoundingBox] = None,
            no_data_value: float = np.nan,
            bounds: rio.coords.BoundingBox = None,
            algorithm=rio.enums.Resampling.cubic,
            dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Read bands from Venus products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param band_type: The band type (FRE or SRE)
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height],
                 The masks pixels as a np.ndarray of shape [masks, width, height],
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        img_files = [self.build_band_path(b, band_type) for b in bands]
        np_arr, xcoords, ycoords, crs = utils.read_as_numpy(img_files,
                                                            crs=crs,
                                                            resolution=resolution,
                                                            offsets=self.offsets,
                                                            region=region,
                                                            output_no_data_value=no_data_value,
                                                            input_no_data_value=-1000,
                                                            bounds=bounds,
                                                            algorithm=algorithm,
                                                            separate=True,
                                                            dtype=dtype,
                                                            scale=scale)

        # Skip first dimension
        np_arr = np_arr[0, ...]

        # Read masks if needed
        np_arr_msk = None
        if len(masks) != 0:
            mask_files = [self.build_mask_path(m, mask_res) for m in masks]
            np_arr_msk, _, _, _ = utils.read_as_numpy(mask_files,
                                                      crs=crs,
                                                      resolution=resolution,
                                                      offsets=self.offsets,
                                                      region=region,
                                                      output_no_data_value=no_data_value,
                                                      input_no_data_value=-1000,
                                                      bounds=bounds,
                                                      algorithm=rio.enums.Resampling.nearest,
                                                      separate=True,
                                                      dtype=np.uint8,
                                                      scale=None)
            # Skip first dimension
            np_arr_msk = np_arr_msk[0, ...]

        # Return plain numpy array
        return np_arr, np_arr_msk, xcoords, ycoords, crs

    def read_as_xarray(self,
                       bands: List[Band],
                       band_type: BandType = FRE,
                       masks: List[Mask] = ALL_MASKS,
                       mask_res: MaskRes = MaskRes.XS,
                       scale: float = 1000,
                       crs: str = None,
                       resolution: float = 10,
                       region: Union[Tuple[int, int, int, int], rio.coords.BoundingBox] = None,
                       no_data_value: float = np.nan,
                       bounds: rio.coords.BoundingBox = None,
                       algorithm=rio.enums.Resampling.cubic,
                       dtype: np.dtype = np.float32) -> xr.Dataset:
        """
        Read bands from Venus products as a xarray

        ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param band_type: The band type (FRE or SRE)
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height]
        """
        np_arr, np_arr_msk, xcoords, ycoords, crs = self.read_as_numpy(
            bands, band_type, masks, mask_res, scale, crs, resolution, region, no_data_value,
            bounds, algorithm, dtype)

        vars = {}
        for i in range(len(bands)):
            vars[bands[i].value] = (["t", "y", "x"], np_arr[None, i, ...])
            for i in range(len(masks)):
                vars[masks[i].value] = (["t", "y", "x"], np_arr_msk[None, i, ...])

        xarr = xr.Dataset(vars,
                          coords={
                              't': [self.date],
                              'x': xcoords,
                              'y': ycoords
                          },
                          attrs={
                              'site': self.site,
                              'type': band_type.value,
                              'crs': crs
                          })
        return xarr
