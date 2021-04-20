#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

from enum import Enum
import dateutil
from typing import List, Tuple, Union
import glob
import os
import numpy as np
import xarray as xr
import rasterio as rio
from sensorsio import utils

"""
This module contains Pleiades related functions
"""


class Pleiades:
    """
    Class for Pleiades product reading
    """
    def __init__(self, product_file:str, offsets:Tuple[float, float]=None):
        """
        Constructor

        :param product_dir: Path to product file
        :param offsets: Shifts applied to image orgin (as computed by StackReg for instance)
        """
        # Store product DIR
        self.product_file = os.path.normpath(product_file)
        self.product_name = os.path.basename(self.product_file)

        # Store offsets
        self.offsets = offsets

        # Get
        self.satellite = Pleiades.Satellite(self.product_name[0:5])

        # Get acquisition date
        self.date = dateutil.parser.parse(self.product_name[9:17])
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday
        
        with rio.open(self.product_file) as ds:
        # Get bounds
            self.bounds  = ds.bounds
        # Get crs
            self.crs = ds.crs

    def __repr__(self):
        return f'{self.satellite.value}, {self.date}'


    # Enum class for sensor
    class Satellite(Enum):
        PHR1A = 'PHR1A'
        PHR1B = 'PHR1B'

    # Aliases
    PHR1A = Satellite.PHR1A
    PHR1B = Satellite.PHR1B

    # Enum class for Pleiades bands
    class Band(Enum):
        PAN = 'PAN'
        B0 = 'B0'
        B1 = 'B1'
        B2 = 'B2'
        B3 = 'B3'

    # Aliases
    PAN = Band.PAN
    B0 = Band.B0
    B1 = Band.B1
    B2 = Band.B2
    B3 = Band.B3

    # Band group
    GROUP_XS = [B0, B1, B2, B3]

    # MTFs
    MTF = {PAN: 0.15, B0: 0.35, B1: 0.35, B2: 0.33, B3: 0.33}

    # Resolutions
    RES = {PAN: 0.7, B0: 2.8, B1: 2.8, B2: 2.8, B3: 2.8}

    def PSF(bands: List[Band], resolution: float = 0.5,
            half_kernel_width: int = None):
        """
        Generate PSF kernels from list of bands

        :param bands: A list of Pleiades Band Enum to generate PSF kernel for
        :param resolution: Resolution at which to sample the kernel
        :param half_kernel_width: The half size of the kernel
                                  (determined automatically if None)

        :return: The kernels as a Tensor of shape
                 [len(bands),2*half_kernel_width+1, 2*half_kernel_width+1]
        """
        return np.stack([utils.generate_psf_kernel(resolution,
                                                Pleiades.RES[b],
                                                Pleiades.MTF[b],
                                                half_kernel_width) for b in bands])

    def read_as_numpy(self,
                      bands:List[Band],
                      scale:float=1000,
                      crs: str=None,
                      resolution:float = 2,
                      region:Union[Tuple[int,int,int,int],rio.coords.BoundingBox]=None,
                      no_data_value:float=np.nan,
                      bounds:rio.coords.BoundingBox=None,
                      algorithm=rio.enums.Resampling.cubic,
                      dtype:np.dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Read bands from Pléiades XS product as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.
        
        :param bands: The list of bands to read
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
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
        np_arr, xcoords, ycoords, crs =  utils.read_as_numpy([self.product_file],
                                                            crs=crs,
                                                            resolution=resolution,
                                                            offsets=self.offsets,
                                                            region=region,
                                                            output_no_data_value = no_data_value,
                                                            input_no_data_value = -10000,
                                                            bounds = bounds,
                                                            algorithm = algorithm,
                                                            separate=False,
                                                            dtype = dtype,
                                                            scale = scale)

        # Skip first dimension
        np_arr = np_arr[0,...]

        
        # Return plain numpy array
        return np_arr, xcoords, ycoords, crs


    def read_as_xarray(self,
                      bands:List[Band],
                      scale:float=1000,
                      crs: str=None,
                      resolution:float = 2,
                      region:Union[Tuple[int,int,int,int],rio.coords.BoundingBox]=None,
                      no_data_value:float=np.nan,
                      bounds:rio.coords.BoundingBox=None,
                      algorithm=rio.enums.Resampling.cubic,
                      dtype:np.dtype=np.float32):
        """
        Read bands from Pléiades XS product as xarray Dataset. Depending on the parameters, an internal WarpedVRT
        dataset might be used.
        
        :param bands: The list of bands to read
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a xarray.Dataset
        """
        np_arr, xcoords, ycoords, crs = self.read_as_numpy(bands,
                                                           scale, crs,
                                                           resolution, region,
                                                           no_data_value, bounds,
                                                           algorithm, dtype)    
        
        vars = {}
        for i in range(len(bands)):
            vars[bands[i].value]=(["t", "y", "x"] , np_arr[None,i,...])
            
        xarr = xr.Dataset(vars,
                          coords={'t' : [self.date], 'x' : xcoords, 'y':ycoords},
                          attrs= {'crs' : crs})
        return xarr

