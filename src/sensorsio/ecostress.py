#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import h5py
import numpy as np
import pyproj
import rasterio as rio
from typing import Tuple, Union
import pyresample
import utm
import dateutil
import xarray as xr


class Ecostress():
    """
    ECostress dataset
    """
    def __init__(self,
                 lst_file: str,
                 geom_file: str,
                 cloud_file: str = None,
                 rad_file: str = None):
        """

        """
        self.lst_file = lst_file
        self.geom_file = geom_file
        self.cloud_file = cloud_file
        self.rad_file = rad_file

        with h5py.File(self.geom_file) as ds:
            # Parse acquisition times
            start_date = str(ds['StandardMetadata/RangeBeginningDate'][()])
            start_time = str(ds['StandardMetadata/RangeBeginningTime'][()])
            end_date = str(ds['StandardMetadata/RangeEndingDate'][()])
            end_time = str(ds['StandardMetadata/RangeEndingTime'][()])

            self.start_time = dateutil.parser.parse(start_date[1:-1] + "T" +
                                                    start_time[1:-2])

            self.end_time = dateutil.parser.parse(end_date[1:-1] + "T" +
                                                  end_time[1:-2])

            # Parse bounds
            min_lon = ds['StandardMetadata/WestBoundingCoordinate'][()]
            max_lon = ds['StandardMetadata/EastBoundingCoordinate'][()]
            min_lat = ds['StandardMetadata/SouthBoundingCoordinate'][()]
            max_lat = ds['StandardMetadata/NorthBoundingCoordinate'][()]

            self.bounds = rio.coords.BoundingBox(min_lon, min_lat, max_lon,
                                                 max_lat)
            self.crs = '+proj=latlon'

    def __repr__(self):
        return f'{self.start_time} - {self.end_time}'

    def read_as_numpy(
        self,
        crs: str = None,
        resolution: float = 70,
        region: Union[Tuple[int, int, int, int],
                      rio.coords.BoundingBox] = None,
        no_data_value: float = np.nan,
        read_lst: bool = True,
        read_angles: bool = True,
        read_emissivities: bool = True,
        bounds: rio.coords.BoundingBox = None,
        nprocs: int = 4,
        dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param nprocs: Number of processors used for reading
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height],
                 The masks pixels as a np.ndarray of shape [masks, width, height],
                 The WVC band
                 The AOT band
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        # Variables of interest
        vois = []

        # Read geolocation grids
        with h5py.File(self.geom_file) as geomDS:
            latitude = np.array(geomDS['Geolocation/latitude'].astype(
                np.double))
            longitude = np.array(geomDS['Geolocation/longitude'].astype(
                np.double))

            # Handle region
            if region is None:
                region = [0, 0, latitude.shape[0], latitude.shape[1]]

            latitude = latitude[region[0]:region[2], region[1]:region[3]]
            longitude = longitude[region[0]:region[2], region[1]:region[3]]

            # handle CRS if not available
            if crs is None:
                mean_latitude = np.mean(latitude)
                mean_longitude = np.mean(longitude)

                _, _, zone, zl = utm.from_latlon(mean_latitude, mean_longitude)

                south = zl < 'N'
                crs = pyproj.CRS.from_dict({
                    'proj': 'utm',
                    'zone': zone,
                    'south': south
                })

            # Handle bounds if not available
            if bounds is None:
                min_latitude = np.min(latitude)
                max_latitude = np.max(latitude)
                min_longitude = np.min(longitude)
                max_longitude = np.max(longitude)
                transformer = pyproj.Transformer.from_crs('+proj=latlon', crs)
                (left, bottom, right, top) = transformer.transform_bounds(
                    min_longitude, min_latitude, max_longitude, max_latitude)
                bounds = rio.coords.BoundingBox(left, bottom, right, top)

            # Read angles
            if read_angles:
                for angle in [
                        'solar_azimuth', 'solar_zenith', 'view_azimuth',
                        'view_zenith'
                ]:
                    angle_array = np.array(geomDS[f'Geolocation/{angle}']
                                           [region[0]:region[2],
                                            region[1]:region[3]].astype(dtype))
                    vois.append(angle_array)

        # Open LST file
        with h5py.File(self.lst_file) as lstDS:

            # Read quality control
            qc = np.array(lstDS['SDS/QC'][region[0]:region[2],
                                          region[1]:region[3]].astype(dtype))

            if read_lst:
                # Read LST
                lst = np.array(lstDS['SDS/LST'][region[0]:region[2],
                                                region[1]:region[3]])
                invalid_mask = lst == 0
                lst = (0.02 * lst).astype(dtype)
                vois.append(lst)

                lst_err = np.array(lstDS['SDS/LST_err'][region[0]:region[2],
                                                        region[1]:region[3]])
                invalid_mask = np.logical_or(invalid_mask, lst_err == 0)
                lst_err = (0.04 * lst_err).astype(dtype)
                vois.append(lst_err)

            # Read emissivities
            if read_emissivities:
                for em in [f'Emis{b}' for b in range(1, 6)]:
                    emis = np.array(lstDS[f'SDS/{em}'][region[0]:region[2],
                                                       region[1]:region[3]])
                    invalid_mask = np.logical_or(invalid_mask, emis == 0)
                    emis = (0.49 + 0.002 * emis).astype(dtype)
                    vois.append(emis)

                    em_err = np.array(
                        lstDS[f'SDS/{em}_err'][region[0]:region[2],
                                               region[1]:region[3]])
                    invalid_mask = np.logical_or(invalid_mask, em_err == 0)
                    em_err = (0.0001 * em_err).astype(dtype)
                    vois.append(em_err)

        # Read cloud mask if available
        vois_discretes = [qc]
        if self.cloud_file:
            with h5py.File(self.cloud_file) as cloudDS:
                cld = np.array(cloudDS['SDS/CloudMask'][
                    region[0]:region[2], region[1]:region[3]].astype(dtype))
                # CAUTION: we can resample cloud mask with other
                # variables as long as we do nearest neighbor
                # interpolation
                vois_discretes.append(cld)

        if self.rad_file:
            with h5py.File(self.rad_file) as radDS:
                for rad in [f'radiance_{b}' for b in range(1, 6)]:
                    rad_arr = np.array(
                        radDS[f'Radiance/{rad}'][region[0]:region[2],
                                                 region[1]:region[3]])
                    invalid_mask = np.logical_or(
                        invalid_mask,
                        np.logical_or(
                            rad_arr == -9997,
                            np.logical_or(rad_arr == -9998, rad_arr == -9999)))
                    rad_arr = rad_arr.astype(dtype)
                    vois.append(rad_arr)

        # Stack variables of intereset into a single array
        vois = np.stack(vois, axis=-1)
        vois = np.ma.masked_array(
            vois,
            np.stack([invalid_mask for i in range(vois.shape[-1])], axis=-1))
        vois_discretes = np.stack(vois_discretes, axis=-1)

        nb_rows = int(np.floor((bounds[3] - bounds[1]) / resolution))
        nb_cols = int(np.floor((bounds[2] - bounds[0]) / resolution))

        #print(nb_rows, nb_cols)

        area_def = pyresample.geometry.AreaDefinition('test', 'test', crs, crs,
                                                      nb_cols, nb_rows, bounds)
        swath_def = pyresample.geometry.SwathDefinition(lons=longitude,
                                                        lats=latitude)

        # If resolution is less than 69 (the largest pixel size in both directions), use 69 to determine sigma. Else use target resolution.
        sigma = (max(resolution, 69.) / np.pi) * np.sqrt(-2 * np.log(0.1))
        radius = 2 * sigma

        result_discretes = pyresample.kd_tree.resample_nearest(
            swath_def,
            vois_discretes,
            area_def,
            radius_of_influence=radius,
            fill_value=no_data_value,
            nprocs=nprocs)

        result = pyresample.kd_tree.resample_gauss(
            swath_def,
            vois,
            area_def,
            radius_of_influence=radius,
            sigmas=[sigma for i in range(vois.shape[-1])],
            fill_value=no_data_value,
            nprocs=nprocs)

        angles_end = 4 if read_angles else 0
        lst_end = angles_end + (2 if read_lst else 0)
        emis_end = lst_end + (10 if read_emissivities else 0)

        angles = result[:, :, :angles_end] if read_angles else None
        lst = result[:, :, angles_end:lst_end] if read_lst else None
        emissivities = result[:, :,
                              lst_end:emis_end] if read_emissivities else None
        radiances = result[:, :, emis_end:] if self.rad_file else None
        qc = result_discretes[:, :, 0].astype(np.uint8)
        clouds = result_discretes[:, :, 1].astype(
            np.uint8) if self.cloud_file else None

        # Unpack cloud mask
        masks = None

        if self.cloud_file:
            valid_mask = np.bitwise_and(clouds, 0b00000001) > 0
            cloud_mask = np.logical_or(
                np.logical_or(
                    np.bitwise_and(clouds, 0b00000010) > 0,
                    np.bitwise_and(clouds, 0b00000100) > 0),
                np.bitwise_and(clouds, 0b00001000) > 0)
            land_mask = (np.bitwise_and(clouds, 0b00100000) > 0)
            sea_mask = np.logical_not(land_mask)
            cloud_mask[~valid_mask] = False
            land_mask[~valid_mask] = False
            sea_mask[~valid_mask] = False

            masks = np.stack((cloud_mask, land_mask, sea_mask), axis=-1)

        xcoords = np.arange(bounds[0], bounds[0] + nb_cols * resolution,
                            resolution)
        ycoords = np.arange(bounds[3], bounds[3] - nb_rows * resolution,
                            -resolution)

        return lst, emissivities, radiances, angles, qc, masks, xcoords, ycoords, crs

    def read_as_xarray(self,
                       crs: str = None,
                       resolution: float = 70,
                       region: Union[Tuple[int, int, int, int],
                                     rio.coords.BoundingBox] = None,
                       no_data_value: float = np.nan,
                       read_lst: bool = True,
                       read_angles: bool = True,
                       read_emissivities: bool = True,
                       bounds: rio.coords.BoundingBox = None,
                       nprocs: int = 4,
                       dtype: np.dtype = np.float32):
        """
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param nprocs: Number of processors used for reading
        :param dtype: dtype of the output Tensor
        """

        lst, emissivities, radiances, angles, qc, masks, xcoords, ycoords, crs = self.read_as_numpy(
            crs, resolution, region, no_data_value, read_lst, read_angles,
            read_emissivities, bounds, nprocs, dtype)

        # Build variables for xarray
        vars = {}

        if lst is not None:
            vars['LST'] = (['y', 'x'], lst[:, :, 0])
            vars['LST_Err'] = (['y', 'x'], lst[:, :, 1])

        if emissivities is not None:
            for i in range(0, 5):
                vars[f'Emis{i+1}'] = (['y', 'x'], emissivities[:, :, 2 * i])
                vars[f'Emis{i+1}_Err'] = (['y', 'x'], emissivities[:, :,
                                                                   2 * i + 1])

        if radiances is not None:
            for i in range(0, 5):
                vars[f'Rad{i+1}'] = (['y', 'x'], radiances[:, :, i])

        if angles is not None:
            vars['Solar_Azimuth'] = (['y', 'x'], angles[:, :, 0])
            vars['Solar_Zenith'] = (['y', 'x'], angles[:, :, 1])
            vars['View_Azimuth'] = (['y', 'x'], angles[:, :, 2])
            vars['View_Zenith'] = (['y', 'x'], angles[:, :, 3])

        if masks is not None:
            vars['QC'] = (['y', 'x'], qc[:, :])
            vars['Cloud_Mask'] = (['y', 'x'], masks[:, :, 0])
            vars['Land_Mask'] = (['y', 'x'], masks[:, :, 1])
            vars['Sea_Mask'] = (['y', 'x'], masks[:, :, 2])

        xarr = xr.Dataset(vars,
                          coords={
                              'x': xcoords,
                              'y': ycoords
                          },
                          attrs={
                              'start_time': self.start_time,
                              'end_time': self.end_time,
                              'crs': crs
                          })

        return xarr
