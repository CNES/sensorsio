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
""" Modeling and access tools for WorldClim 2.0 data """

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.coords import BoundingBox

from sensorsio import regulargrid


class WorldClimQuantity(Enum):
    """ The physical quantities available in the WorldClim data base"""
    PREC = "prec"
    SRAD = "srad"
    TAVG = "tavg"
    TMAX = "tmax"
    TMIN = "tmin"
    VAPR = "vapr"
    WIND = "wind"


WorldClimQuantityAll: List[WorldClimQuantity] = list(WorldClimQuantity)


class WorldClimBio(Enum):
    """ The BIO variables available in the WorldClim data base"""
    ANNUAL_MEAN_TEMP = "Annual_Mean_Temp"
    MEAN_DIURNAL_TEMP_RANGE = "Mean_Diurnal_Temp_Range"  # (Mean of monthly (max temp - min temp))
    ISOTHERMALITY = "Isothermality"  # Isothermality (BIO2/BIO7) (* 100)
    TEMP_SEASONALITY = "Temp_Seasonality"  # (standard deviation *100)
    MAX_TEMP_WARMEST_MONTH = "Max_Temp_Warmest_Month"
    MIN_TEMP_COLDEST_MONTH = "Min_Temp_Coldest_Month"
    TEMP_ANNUAL_RANGE = "Temp_Annual_Range"  # (BIO05-BIO06)
    MEAN_TEMP_WETTEST_QUART = "Mean_Temp_Wettest_Quart"
    MEAN_TEMP_DRIEST_QUART = "Mean_Temp_Driest_Quart"
    MEAN_TEMP_WARMEST_QUART = "Mean_Temp_Warmest_Quart"
    MEAN_TEMP_COLDEST_QUART = "Mean_Temp_Coldest_Quart"
    ANNUAL_PREC = "Annual_Prec"
    PREC_WETTEST_MONTH = "Prec_Wettest_Month"
    PREC_DRIEST_MONTH = "Prec_Driest_Month"
    PREC_SEASONALITY = "Prec_Seasonality"  # (Coefficient of Variation)
    PREC_WETTEST_QUART = "Prec_Wettest_Quart"
    PREC_DRIEST_QUART = "Prec_Driest_Quart"
    PREC_WARMEST_QUART = "Prec_Warmest_Quart"
    PREC_COLDEST_QUART = "Prec_Coldest_Quart"


WorldClimBioAll: List[WorldClimBio] = list(WorldClimBio)


class WorldClimVar:
    """ WorldClim variable (either climatic quantity or bio)"""
    def __init__(self, var: Union[WorldClimQuantity, WorldClimBio], month: Optional[int] = None):
        self.value = var.value
        if (month is None) and isinstance(var, WorldClimBio):
            self.typ = 'bio'
        elif month is not None and (1 <= month <= 12) and isinstance(var, WorldClimQuantity):
            self.typ = 'clim'
            self.month = month
        else:
            raise ValueError("Quantity needs month. Bio does not use month. "
                             "Received {var.value} {month}")

    def __str__(self):
        if self.typ == 'bio':
            return f"{self.value.upper()}"
        return f"CLIM_{self.value.upper()}_{self.month:02}"


WorldClimQuantityVarAll: List[WorldClimVar] = [
    WorldClimVar(v, m) for v in WorldClimQuantityAll for m in range(1, 13)
]

WorldClimBioVarAll: List[WorldClimVar] = [WorldClimVar(wcb) for wcb in WorldClimBio]

WorldClimVarAll: List[WorldClimVar] = WorldClimQuantityVarAll + WorldClimBioVarAll


class WorldClimData:
    """ WorldClim data model and reading"""
    def __init__(
        self,
        wcdir: str = "/datalake/static_aux/worldclim-2.0",
    ) -> None:
        self.wcdir = wcdir
        self.__wcprefix = "wc2.0"
        self.__crs = "+proj=latlong"
        self.__wcres = "30s"
        self.__resolution = 30 / 60 / 60  # convert to degrees

        months = range(1, 13)
        self.climfiles = [
            self.get_file_path(WorldClimVar(cv, m)) for m in months for cv in WorldClimQuantityAll
        ]
        self.biofiles = [self.get_file_path(WorldClimVar(b)) for b in WorldClimBioAll]

    def get_var_name(self, var):
        """ Return the variable name as 30s_tavg_08"""
        if var.typ == 'bio':
            bio_names = {wcb.value: f"{idx:02}" for idx, wcb in enumerate(WorldClimBio, start=1)}
            return f"bio_{self.__wcres}_{bio_names[var.value]}"
        return f"{self.__wcres}_{var.value}_{var.month:02}"

    def get_file_path(self, var: WorldClimVar) -> str:
        """ Return the file path for a variable"""
        var_name = self.get_var_name(var)
        return f"{self.wcdir}/{self.__wcprefix}_{var_name}.tif"

    def read_as_numpy(
        self,
        wc_vars: List[WorldClimVar] = WorldClimVarAll,
        crs: Optional[str] = None,
        resolution: float = 1000,
        bounds: Optional[BoundingBox] = None,
        no_data_value: float = np.nan,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Read the data corresponding to a bounding box and return it
        as a numpy array
        """
        # Safeguard
        assert wc_vars is not None and len(wc_vars) > 0

        wc_files = [self.get_file_path(wc_var) for wc_var in wc_vars]

        if crs is None:
            crs = self.__crs

        np_arr, xcoords, ycoords, crs = regulargrid.read_as_numpy(
            wc_files,
            crs=crs,
            resolution=resolution,
            bounds=bounds,
            output_no_data_value=no_data_value,
            algorithm=algorithm,
            separate=True,
            dtype=np.dtype("float32"))

        # skip first dimension
        np_arr = np_arr[0, ...]

        return np_arr, xcoords, ycoords, crs

    def read_as_xarray(self,
                       wc_vars: List[WorldClimVar] = WorldClimVarAll,
                       crs: Optional[str] = None,
                       resolution: float = 100,
                       bounds: Optional[BoundingBox] = None,
                       no_data_value: float = np.nan,
                       algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic) -> xr.Dataset:
        """Read the data corresponding to a bounding box and return it
        as an xarray"""
        (
            np_wc,
            xcoords,
            ycoords,
            crs,
        ) = self.read_as_numpy(wc_vars, crs, resolution, bounds, no_data_value, algorithm=algorithm)
        xr_vars: Dict[str, Tuple[List[str], np.ndarray]] = {
            f"wc_{self.get_var_name(var)}": (["y", "x"], np_wc[idx, :, :])
            for idx, var in enumerate(wc_vars)
        }
        xarr = xr.Dataset(
            xr_vars,
            coords={
                "x": xcoords,
                "y": ycoords
            },
            attrs={
                "crs": str(crs),
                "resolution": resolution,
                "nodata": no_data_value
            },
        )
        return xarr
