#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Modeling and access tools for WorldClim 2.0 data """

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.coords import BoundingBox
from rasterio.warp import reproject

from .utils import compute_latlon_bbox_from_region


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
    BIO01 = "Annual_Mean_Temp"
    BIO02 = "Mean_Diurnal_Temp_Range"  # (Mean of monthly (max temp - min temp))
    BIO03 = "Isothermality"  # Isothermality (BIO2/BIO7) (* 100)
    BIO04 = "Temp_Seasonality"  # (standard deviation *100)
    BIO05 = "Max_Temp_Warmest_Month"
    BIO06 = "Min_Temp_Coldest_Month"
    BIO07 = "Temp_Annual_Range"  # (BIO05-BIO06)
    BIO08 = "Mean_Temp_Wettest_Quart"
    BIO09 = "Mean_Temp_Driest_Quart"
    BIO10 = "Mean_Temp_Warmest_Quart"
    BIO11 = "Mean_Temp_Coldest_Quart"
    BIO12 = "Annual_Prec"
    BIO13 = "Prec_Wettest_Month"
    BIO14 = "Prec_Driest_Month"
    BIO15 = "Prec_Seasonality"  # (Coefficient of Variation)
    BIO16 = "Prec_Wettest_Quart"
    BIO17 = "Prec_Driest_Quart"
    BIO18 = "Prec_Warmest_Quart"
    BIO19 = "Prec_Coldest_Quart"


WorldClimBioAll: List[WorldClimBio] = list(WorldClimBio)


class WorldClimVar:
    """ WorldClim variable (either climatic quantity or bio)"""

    def __init__(self,
                 var: Union[WorldClimQuantity, WorldClimBio],
                 month: Optional[int] = None):
        self.value = var.value
        if (month is None) and isinstance(var, WorldClimBio):
            self.typ = 'bio'
        elif month is not None and (1 <= month <= 12) and isinstance(
                var, WorldClimQuantity):
            self.typ = 'clim'
            self.month = month
        else:
            raise ValueError("Quantity needs month. Bio does not use month. "
                             "Received {var.value} {month}")

    def __str__(self):
        if self.typ == 'bio':
            return f"BIO_{self.value}"
        return f"CLIM_{self.value.upper()}_{self.month:02}"


WorldClimQuantityVarAll: List[WorldClimVar] = [
    WorldClimVar(v, m) for v in WorldClimQuantityAll for m in range(1, 13)
]

WorldClimBioVarAll: List[WorldClimVar] = [
    WorldClimVar(wcb) for wcb in WorldClimBio
]

WorldClimVarAll: List[
    WorldClimVar] = WorldClimQuantityVarAll + WorldClimBioVarAll


class WorldClimData:
    """ WorldClim data model and reading"""

    def __init__(
        self,
        wcdir: str = "/datalake/static_aux/worldclim-2.0",
        wcprefix: str = "wc2.0",
        crs: str = "+proj=latlong",
    ) -> None:
        self.wcdir = wcdir
        self.wcprefix = wcprefix
        self.crs = crs
        self.wcres = "30s"
        self.resolution = 30 / 60 / 60  # convert to degrees

        months = range(1, 13)
        self.climfiles = [
            self.get_file_path(WorldClimVar(cv, m)) for m in months
            for cv in WorldClimQuantityAll
        ]
        self.biofiles = [
            self.get_file_path(WorldClimVar(b)) for b in WorldClimBioAll
        ]

        with rio.open(self.climfiles[0]) as clim_ds:
            self.transform = rio.Affine(
                clim_ds.transform.a, clim_ds.transform.b,
                clim_ds.transform.c - clim_ds.transform.a / 2,
                clim_ds.transform.d, clim_ds.transform.e,
                clim_ds.transform.f - clim_ds.transform.e / 2)

    def crop_to_bbox(self, imfile, bbox):
        "Crop a geotif file using the bbox"
        (top, bottom), (left,
                        right) = rio.transform.rowcol(self.transform,
                                                      [bbox.left, bbox.right],
                                                      [bbox.top, bbox.bottom])
        (left, right) = (min(left, right), max(left, right))
        (bottom, top) = (max(bottom, top), min(bottom,
                                               top))  # top is upper left

        with rio.open(imfile) as data_source:
            image = data_source.read(window=((top, bottom), (left, right)))

        return image

    def get_var_name(self, var):
        """ Return the variable name as 30s_tavg_08"""
        if var.typ == 'bio':
            return f"bio_{self.wcres}_{var.value}"
        return f"{self.wcres}_{var.value}_{var.month:02}"

    def get_file_path(self, var: WorldClimVar) -> str:
        """ Return the file path for a variable"""

        var_name = self.get_var_name(var)
        return f"{self.wcdir}/{self.wcprefix}_{var_name}.tif"

    def get_wc_for_bbox(
        self,
        bbox,
        wc_vars: Optional[List[WorldClimVar]] = None
    ) -> Tuple[np.ndarray, rio.Affine]:
        "Get a stack with all the WC vars croped to contain the bbox"
        if wc_vars is None:
            wc_vars = WorldClimVarAll
        files = [self.get_file_path(v) for v in wc_vars]
        wcvars: List[np.ndarray] = [
            self.crop_to_bbox(wc_file, bbox)[0, :, :] for wc_file in files
        ]
        transform = rio.Affine(self.transform.a, self.transform.b, bbox.left,
                               self.transform.d, self.transform.e, bbox.top)
        return np.stack(wcvars, axis=0), transform

    def read_as_numpy(
        self,
        wc_vars: Optional[List[WorldClimVar]] = None,
        crs: str = None,
        resolution: float = 1000,
        bounds: BoundingBox = None,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
        dtype: np.dtype = np.dtype("float32"),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, rio.Affine]:
        """Read the data corresponding to a bounding box and return it
        as a numpy array"""
        assert bounds is not None
        if crs is None:
            crs = self.crs
        if wc_vars is None:
            wc_vars = WorldClimVarAll
        dst_transform = rio.Affine(resolution, 0.0, bounds.left, 0.0,
                                   -resolution, bounds.top)
        dst_size_x = int(np.ceil((bounds.right - bounds.left) / resolution))
        dst_size_y = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        bbox = compute_latlon_bbox_from_region(bounds, crs)
        pad_factor = 0.05  # enlarge the bbox so that large resolution
        # factors do not produce bboxes which are
        # inside the needed region
        bbox = BoundingBox(bbox.left * (1 - pad_factor),
                           bbox.bottom * (1 - pad_factor),
                           bbox.right * (1 + pad_factor),
                           bbox.top * (1 + pad_factor))
        wc_bbox, src_transform = self.get_wc_for_bbox(bbox, wc_vars)
        dst_wc = np.zeros((wc_bbox.shape[0], dst_size_y, dst_size_x))
        dst_wc, dst_wc_transform = reproject(
            wc_bbox,
            destination=dst_wc,
            src_transform=src_transform,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=algorithm,
        )
        dst_wc = dst_wc.astype(dtype)
        xcoords = np.arange(bounds.left, bounds.right, resolution)
        ycoords = np.arange(bounds.top, bounds.bottom, -resolution)
        return (dst_wc, xcoords, ycoords, crs, dst_wc_transform)

    def read_as_xarray(
            self,
            wc_vars: Optional[List[WorldClimVar]] = None,
            crs: str = None,
            resolution: float = 100,
            bounds: BoundingBox = None,
            algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
            dtype: np.dtype = np.dtype("float32"),
    ) -> xr.Dataset:
        """Read the data corresponding to a bounding box and return it
        as an xarray"""
        if wc_vars is None:
            wc_vars = WorldClimVarAll
        (
            np_wc,
            xcoords,
            ycoords,
            crs,
            transform,
        ) = self.read_as_numpy(wc_vars, crs, resolution, bounds, algorithm,
                               dtype)
        xr_vars: Dict[str, Tuple[List[str], np.ndarray]] = {
            self.get_var_name(var): (["y", "x"], np_wc[idx, :, :])
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
                "transform": str(transform),
            },
        )
        return xarr
