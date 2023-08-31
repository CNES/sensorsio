#!/usr/bin/env python
# coding: utf8

# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
#
# Licensed under the Lesser GNU LESSER GENERAL PUBLIC
# LICENSE, Version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.gnu.org/licenses/lgpl-3.0.txt

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import utm  # type: ignore
import xarray as xr
from pyhdf.SD import SD  # type: ignore

from sensorsio import irregulargrid


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
        self.l2a_emis_file = glob.glob(os.path.join(self.l2a_dir, '*-emissivity_tes.dat'))[0]
        self.l2a_lst_file = glob.glob(os.path.join(self.l2a_dir, '*-surface_temp.dat'))[0]
        self.acquisition_date = pd.to_datetime(self.l1b_file[-26:-18])

        self.crs = '+proj=latlon'

        l1b_ds = SD(self.l1b_file)
        master_lat = l1b_ds.select('PixelLatitude').get()
        master_lon = l1b_ds.select('PixelLongitude').get()
        self.bounds = rio.coords.BoundingBox(master_lon.min(), master_lat.min(), master_lon.max(),
                                             master_lat.max())

    def __repr__(self):
        return f'{self.acquisition_date}'

    def read_as_numpy(self,
                      crs: Optional[str] = None,
                      resolution: float = 30,
                      region: Optional[Tuple[int, int, int, int]] = None,
                      bounds: Optional[rio.coords.BoundingBox] = None,
                      no_data_value: float = np.nan,
                      nprocs: int = 4,
                      strip_size: int = 375000,
                      dtype: np.dtype = np.dtype('float32')):

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

        # Stick to convention adopted for landsat-8: North-Up, positive to the east, negative to the west
        master_azimuth = master_azimuth - 180.
        master_sun_azimuth = 180 + master_sun_azimuth

        # Handle region
        if region is None:
            region = (0, 0, master_lat.shape[0], master_lat.shape[1])

        master_lat = master_lat[region[0]:region[2], region[1]:region[3]]
        master_lon = master_lon[region[0]:region[2], region[1]:region[3]]

        # handle CRS if not available
        if crs is None:
            mean_master_lat = np.mean(master_lat)
            mean_master_lon = np.mean(master_lon)

            _, _, zone, zl = utm.from_latlon(mean_master_lat, mean_master_lon)

            south = zl < 'N'
            crs = pyproj.CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south}).to_string()

        # Handle bounds if not available
        if bounds is None:
            min_master_lat = np.min(master_lat)
            max_master_lat = np.max(master_lat)
            min_master_lon = np.min(master_lon)
            max_master_lon = np.max(master_lon)
            transformer = pyproj.Transformer.from_crs('+proj=latlon', crs)
            (left, bottom, right,
             top) = transformer.transform_bounds(min_master_lon, min_master_lat, max_master_lon,
                                                 max_master_lat)
            bounds = rio.coords.BoundingBox(left, bottom, right, top)

        vois = np.stack([
            master_lst, *(master_emissivities[i, ...] for i in range(5)), master_zenith,
            master_azimuth, master_sun_zenith, master_sun_azimuth
        ],
                        axis=-1)
        vois = vois[region[0]:region[2], region[1]:region[3], :]
        invalid_mask = ~(master_lst > 0)[region[0]:region[2], region[1]:region[3]]

        vois = np.ma.masked_array(vois,
                                  np.stack([invalid_mask for i in range(vois.shape[-1])], axis=-1))

        # If resolution is less than 69 (the largest pixel size in both directions)
        # , use 30 to determine sigma. Else use target resolution.
        # Master actual resolution depend on carrier
        sigma = (max(resolution, 30.) / np.pi) * np.sqrt(-2 * np.log(0.1))
        max_neighbours = max(4, int(np.ceil(resolution / 30.))**2)

        _, result, xcoords, ycoords = irregulargrid.swath_resample(master_lat,
                                                                   master_lon,
                                                                   crs,
                                                                   bounds,
                                                                   resolution,
                                                                   sigma,
                                                                   cutoff_sigma_mult=3.,
                                                                   continuous_variables=vois,
                                                                   fill_value=no_data_value,
                                                                   nthreads=nprocs,
                                                                   strip_size=strip_size,
                                                                   max_neighbours=max_neighbours)
        assert result is not None
        lst = result[:, :, :1]
        emis = result[:, :, 1:6]
        angles = result[:, :, 6:]

        return lst, emis, angles, xcoords, ycoords, crs

    def read_as_xarray(self,
                       crs: Optional[str] = None,
                       resolution: float = 30,
                       region: Optional[Tuple[int, int, int, int]] = None,
                       bounds: Optional[rio.coords.BoundingBox] = None,
                       no_data_value: float = np.nan,
                       nprocs: int = 4,
                       strip_size: int = 375000,
                       dtype: np.dtype = np.dtype('float32')):

        lst, emis, angles, xcoords, ycoords, crs = self.read_as_numpy(crs, resolution, region,
                                                                      bounds, no_data_value, nprocs,
                                                                      strip_size, dtype)

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
