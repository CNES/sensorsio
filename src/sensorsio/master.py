#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

from typing import Tuple
import glob, os
import pandas as pd
import rasterio as rio
import numpy as np
from pyhdf.SD import *
import utm
import pyproj
from sensorsio import utils
import xarray as xr


class Master():
    """
    Master dataset
    """
    def __init__(self, l1b_file: str, l2a_dir: str):
        """

        """
        self.l1b_file = l1b_file
        self.l2a_dir = l2a_dir

        # Look for appropriate files in l2a_dir
        self.l2a_emis_file = glob.glob(
            os.path.join(self.l2a_dir, '*-emissivity_tes.dat'))[0]
        self.l2a_lst_file = glob.glob(
            os.path.join(self.l2a_dir, '*-surface_temp.dat'))[0]
        self.acquisition_date = pd.to_datetime(self.l1b_file[-26:-18])

        self.crs = '+proj=latlon'

        l1b_ds = SD(self.l1b_file)
        master_lat = l1b_ds.select('PixelLatitude').get()
        master_lon = l1b_ds.select('PixelLongitude').get()
        self.bounds = rio.coords.BoundingBox(master_lon.min(),
                                             master_lat.min(),
                                             master_lon.max(),
                                             master_lat.max())

    def __repr__(self):
        return f'{self.acquisition_date}'

    def read_as_numpy(self,
                      crs: str = None,
                      resolution: float = 30,
                      region: Tuple[int, int, int, int] = None,
                      bounds: rio.coords.BoundingBox = None,
                      no_data_value: float = np.nan,
                      nprocs: int = 4,
                      dtype: np.dtype = np.float32):

        # Read master LST
        with rio.open(self.l2a_lst_file) as ds:
            master_lst = ds.read()[0]

        # Read master emissivity
        with rio.open(self.l2a_emis_file) as ds:
            master_emissivities = ds.read()

        # Read L1B file
        l1b_dataset = SD(self.l1b_file)
        # Read location and angles
        master_lat = l1b_dataset.select('PixelLatitude').get()
        master_lon = l1b_dataset.select('PixelLongitude').get()
        master_zenith = l1b_dataset.select('SensorZenithAngle').get()
        master_azimuth = l1b_dataset.select('SensorAzimuthAngle').get()
        master_sun_zenith = l1b_dataset.select('SolarZenithAngle').get()
        master_sun_azimuth = l1b_dataset.select('SolarAzimuthAngle').get()

        # Handle region
        if region is None:
            region = [0, 0, master_lat.shape[0], master_lat.shape[1]]

            master_lat = master_lat[region[0]:region[2], region[1]:region[3]]
            master_lon = master_lon[region[0]:region[2], region[1]:region[3]]

        # handle CRS if not available
        if crs is None:
            mean_master_lat = np.mean(master_lat)
            mean_master_lon = np.mean(master_lon)

            _, _, zone, zl = utm.from_latlon(mean_master_lat, mean_master_lon)

            south = zl < 'N'
            crs = pyproj.CRS.from_dict({
                'proj': 'utm',
                'zone': zone,
                'south': south
            })

        # Handle bounds if not available
        if bounds is None:
            min_master_lat = np.min(master_lat)
            max_master_lat = np.max(master_lat)
            min_master_lon = np.min(master_lon)
            max_master_lon = np.max(master_lon)
            transformer = pyproj.Transformer.from_crs('+proj=latlon', crs)
            (left, bottom, right, top) = transformer.transform_bounds(
                min_master_lon, min_master_lat, max_master_lon, max_master_lat)
            bounds = rio.coords.BoundingBox(left, bottom, right, top)

        vois = np.stack([
            master_lst, *(master_emissivities[i, ...]
                          for i in range(5)), master_zenith, master_azimuth,
            master_sun_zenith, master_sun_azimuth
        ],
                        axis=-1)

        # If resolution is less than 69 (the largest pixel size in both directions)
        # , use 30 to determine sigma. Else use target resolution.
        # Master actual resolution depend on carrier
        sigma = (max(resolution, 30.) / np.pi) * np.sqrt(-2 * np.log(0.1))

        _, result, xcoords, ycoords = utils.swath_resample(
            master_lat,
            master_lon,
            crs,
            bounds,
            resolution,
            sigma,
            continuous_variables=vois,
            fill_value=no_data_value,
            nthreads=nprocs)

        lst = result[:, :, :1]
        emis = result[:, :, 1:6]
        angles = result[:, :, 6:]

        return lst, emis, angles, xcoords, ycoords, crs

    def read_as_xarray(self,
                       crs: str = None,
                       resolution: float = 30,
                       region: Tuple[int, int, int, int] = None,
                       bounds: rio.coords.BoundingBox = None,
                       no_data_value: float = np.nan,
                       nprocs: int = 4,
                       dtype: np.dtype = np.float32):

        lst, emis, angles, xcoords, ycoords, crs = self.read_as_numpy(
            crs, resolution, region, bounds, no_data_value, nprocs, dtype)

        # Build variables for xarray
        vars = {}
        vars['LST'] = (['y', 'x'], lst[:, :, 0])

        for e in range(5):
            vars[f'Emis{e+1}'] = (['y', 'x'], emis[:, :, e])

        vars['Solar_Azimuth'] = (['y', 'x'], angles[:, :, 3])
        vars['Solar_Zenith'] = (['y', 'x'], angles[:, :, 2])
        vars['View_Azimuth'] = (['y', 'x'], angles[:, :, 1])
        vars['View_Zenith'] = (['y', 'x'], angles[:, :, 0])

        xarr = xr.Dataset(vars,
                          coords={
                              'x': xcoords,
                              'y': ycoords
                          },
                          attrs={
                              'crs': crs,
                              'aquisition_date': self.acquisition_date
                          })

        return xarr
