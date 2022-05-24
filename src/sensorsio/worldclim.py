#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
""" Modeling and access tools for WorldClim 2.0 data """

from typing import List, Tuple

import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.warp import reproject

from .utils import compute_latlon_bbox_from_region


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
        self.climvars = [
            "prec",
            "tavg",
            "tmin",
            "wind",
            "srad",
            "tmax",
            "vapr",
        ]
        self.climfiles = [
            f"{wcdir}/{wcprefix}_{self.wcres}_{cv}_{m:02}.tif" for m in months
            for cv in self.climvars
        ]
        biovars = range(1, 20)
        self.biofiles = [
            f"{wcdir}/{wcprefix}_bio_{self.wcres}_{b:02}.tif" for b in biovars
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

    def get_wc_for_bbox(self, bbox) -> np.array:
        "Get a stack with all the WC vars croped to contain the bbox"
        wcvars: List[np.array] = [
            self.crop_to_bbox(v, bbox)[0, :, :]
            for v in self.climfiles + self.biofiles
        ]
        transform = rio.Affine(self.transform.a, self.transform.b, bbox.left,
                               self.transform.d, self.transform.e, bbox.top)
        return np.stack(wcvars, axis=0), transform

    def read_as_numpy(
        self,
        crs: str,
        resolution: float,
        bounds: BoundingBox,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               str]:
        """ Read the data corresponding to a bounding box and return it as a numpy array"""
        assert bounds is not None
        dst_transform = rio.Affine(resolution, 0.0, bounds.left, 0.0,
                                   -resolution, bounds.top)
        dst_size_x = int(np.ceil((bounds.right - bounds.left) / resolution))
        dst_size_y = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        bbox = compute_latlon_bbox_from_region(bounds, crs)
        wc_bbox, src_transform = self.get_wc_for_bbox(bbox)
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
