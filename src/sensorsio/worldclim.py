#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Modeling and access tools for WorldClim 2.0 data """

from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
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


WorldClimQuantityAll: List[WorldClimQuantity] = [wcq
                                                 for wcq in WorldClimQuantity]


class WorldClimBio(Enum):
    """ The BIO variables available in the WorldClim data base"""
    BIO01 = "01"  # Annual Mean Temperature
    BIO02 = "02"  # Mean Diurnal Range (Mean of monthly (max temp - min temp))
    BIO03 = "03"  # Isothermality (BIO2/BIO7) (* 100)
    BIO04 = "04"  # Temperature Seasonality (standard deviation *100)
    BIO05 = "05"  # Max Temperature of Warmest Month
    BIO06 = "06"  # Min Temperature of Coldest Month
    BIO07 = "07"  # Temperature Annual Range (BIO05-BIO06)
    BIO08 = "08"  # Mean Temperature of Wettest Quarter
    BIO09 = "09"  # Mean Temperature of Driest Quarter
    BIO10 = "10"  # Mean Temperature of Warmest Quarter
    BIO11 = "11"  # Mean Temperature of Coldest Quarter
    BIO12 = "12"  # Annual Precipitation
    BIO13 = "13"  # Precipitation of Wettest Month
    BIO14 = "14"  # Precipitation of Driest Month
    BIO15 = "15"  # Precipitation Seasonality (Coefficient of Variation)
    BIO16 = "16"  # Precipitation of Wettest Quarter
    BIO17 = "17"  # Precipitation of Driest Quarter
    BIO18 = "18"  # Precipitation of Warmest Quarter
    BIO19 = "19"  # Precipitation of Coldest Quarter


WorldClimBioAll: List[WorldClimBio] = [wcb for wcb in WorldClimBio]


class WorldClimVar:
    """ WorldClim variable (either climatic quantity or bio)"""

    def __init__(self,
                 var: Union[WorldClimQuantity, WorldClimBio],
                 month: Optional[int] = None):
        self.value = var.value
        if (month is None) and isinstance(var, WorldClimBio):
            self.typ = 'bio'
        elif (1 <= month <= 12) and isinstance(var, WorldClimQuantity):
            self.typ = 'clim'
            self.month = month
        else:
            raise ValueError("Quantity needs month. Bio does not use month. "
                             "Received {var.value} {month}")


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

        with rio.open(self.climfiles[0]) as ds:
            self.transform = rio.Affine(ds.transform.a, ds.transform.b,
                                        ds.transform.c - ds.transform.a / 2,
                                        ds.transform.d, ds.transform.e,
                                        ds.transform.f - ds.transform.e / 2)

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

    def get_file_path(self, var: WorldClimVar):
        """ Return the file path for a variable"""
        if var.typ == 'bio':
            fname = f"bio_{self.wcres}_{var.value}"
        else:
            fname = f"{self.wcres}_{var.value}_{var.month:02}"
        return f"{self.wcdir}/{self.wcprefix}_{fname}.tif"

    def get_wc_for_bbox(self,
                        bbox,
                        vars: Optional[List[WorldClimVar]] = None) -> np.array:
        "Get a stack with all the WC vars croped to contain the bbox"
        if vars is None:
            files = self.climfiles + self.biofiles
        else:
            files = [self.get_file_path(v) for v in vars]
        wcvars: List[np.array] = [
            self.crop_to_bbox(wc_file, bbox)[0, :, :] for wc_file in files
        ]
        transform = rio.Affine(self.transform.a, self.transform.b, bbox.left,
                               self.transform.d, self.transform.e, bbox.top)
        return np.stack(wcvars, axis=0), transform

    def read_as_numpy(
        self,
        vars: Optional[List[WorldClimVar]] = None,
        crs: str = None,
        resolution: float = 100,
        bounds: BoundingBox = None,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               str]:
        """Read the data corresponding to a bounding box and return it
        as a numpy array"""
        assert bounds is not None
        dst_transform = rio.Affine(resolution, 0.0, bounds.left, 0.0,
                                   -resolution, bounds.top)
        dst_size_x = int(np.ceil((bounds.right - bounds.left) / resolution))
        dst_size_y = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        bbox = compute_latlon_bbox_from_region(bounds, crs)
        wc_bbox, src_transform = self.get_wc_for_bbox(bbox, vars)
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
        xcoords = np.linspace(bounds.left, bounds.right, dst_size_x)
        ycoords = np.linspace(bounds.top, bounds.bottom, dst_size_y)
        return (dst_wc, xcoords, ycoords, crs, dst_wc_transform)
