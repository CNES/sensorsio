#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

from typing import List, Tuple

import numpy as np
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.warp import reproject

from utils import compute_latlon_bbox_from_region


class WorldClimData:
    def __init__(
        self,
        wcdir: str = "/datalake/static_aux/worldclim-2.0",
        wcprefix: str = "wc2.0",
        wcres: str = "30s",
        crs: str = "+proj=latlong",
    ) -> None:
        self.wcdir = wcdir
        self.wcprefix = wcprefix
        self.crs = crs
        self.wcres = wcres

        self.transform = rio.Affine(
            resolution, 0.0, -180.0, 0.0, resolution, -90.0
        )  # TODO: compute resolution from wcres
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
            f"{wcdir}/{wcprefix}_{wcres}_{cv}_{m:02}.tif"
            for m in months
            for cv in self.climvars
        ]
        biovars = range(1, 20)
        self.biofiles = [
            f"{wcdir}/{wcprefix}_bio_{wcres}_{b:02}.tif" for b in biovars
        ]

    def crop_to_bbox(self, imfile, bbox):
        "Crop a geotif file using the bbox"
        (left, right), (top, bottom) = rio.transform.rowcol(
            self.transform, [bbox.left, bbox.right], [bbox.top, bbox.bottom]
        )
        with rio.open(imfile) as ds:
            image = ds.read(window=((top, bottom), (left, right)))
        return image

    def get_wc_for_bbox(self, bbox):
        "Get a stack with all the WC vars croped to contain the bbox"
        wcvars: List[np.array] = [
            self.crop_to_bbox(v, bbox) for v in self.climfiles + self.biofiles
        ]
        return np.stack(wcvars, axis=0)

    def read_as_numpy(
        self,
        crs: str,
        resolution: float,
        bounds: BoundingBox,
        no_data_value: float = np.nan,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
        dtype: np.dtype = np.float32,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str
    ]:
        assert bounds is not None
        dst_transform = rio.Affine(
            resolution, 0.0, bounds.left, 0.0, resolution, bounds.bottom
        )
        dst_size_x = int((bounds.right - bounds.left) / resolution)
        dst_size_y = int((bounds.top - bounds.bottom) / resolution)
        dst_wc = np.zeros((3, dst_size_x, dst_size_y))
        bbox = compute_latlon_bbox_from_region(bounds, crs)
        wc_bbox = self.get_wc_for_bbox(BoundingBox(bbox))
        dst_wc, dst_wc_transform = reproject(
            wc_bbox,
            destination=dst_wc,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=algorithm,
        )
        dst_wc = dst_wc.astype(dtype)
        xcoords = np.linspace(bounds.left, bounds.right, dst_size_x)
        ycoords = np.linspace(bounds.top, bounds.bottom, dst_size_y)
        return (dst_wc, xcoords, ycoords, crs, dst_wc_transform)
