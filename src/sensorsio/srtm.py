#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
import geopandas as gpd
import os
import rasterio as rio
from rasterio.merge import merge


@dataclass
class BoundingBox:
    lonmin: float
    latmin: float
    lonmax: float
    latmax: float


@dataclass
class SRTMTileId:
    lon: int
    lat: int


def srtm_id_to_name(id: SRTMTileId) -> str:
    northing = 'S' if id.lat < 0 else 'N'
    easting = 'W' if id.lon < 0 else 'E'
    lat = str(np.abs(id.lat)).zfill(2)
    lon = str(np.abs(id.lon)).zfill(3)
    return f"{northing}{lat}{easting}{lon}"


def srtm_tiles_from_bbox(bbox: BoundingBox) -> List[SRTMTileId]:
    latmin = int(np.floor(bbox.latmin))
    latmax = int(np.floor(bbox.latmax))
    lonmin = int(np.floor(bbox.lonmin))
    lonmax = int(np.floor(bbox.lonmax))
    return [
        SRTMTileId(lon, lat) for lat in range(latmin, latmax + 1)
        for lon in range(lonmin, lonmax + 1)
    ]


def srtm_tiles_from_mgrs_tile(tile: str) -> List[SRTMTileId]:
    assert tile[0] != 'T'
    mgrs_df = gpd.read_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data/sentinel2/mgrs_tiles.shp'))
    bbox = BoundingBox(*mgrs_df[mgrs_df.Name == tile].iloc[0].geometry.bounds)
    return srtm_tiles_from_bbox(bbox)


@dataclass(frozen=True)
class DEM:
    elevation: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    transform: rio.transform.Affine


@dataclass(frozen=True)
class SRTM:
    """
    Class for SRTM DEM (hgt format) product reading
    """
    base_dir: str = "/datalake/static_aux/MNT/SRTM_30_hgt"

    def get_dem_from_tiles(self, tiles: List[SRTMTileId]) -> DEM:
        elevation, transform = self.build_hgt(tiles)
        x, y = np.gradient(elevation)
        slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        return DEM(elevation, slope, aspect, transform)

    def get_dem_for_mgrs_tile(self, tile: str) -> DEM:
        srtm_tiles = srtm_tiles_from_mgrs_tile(tile)
        return self.get_dem_from_tiles(srtm_tiles)

    def get_dem_for_bbox(self, bbox: BoundingBox) -> DEM:
        srtm_tiles = srtm_tiles_from_bbox(bbox)
        return self.get_dem_from_tiles(srtm_tiles)

    def build_hgt(
            self, tiles: List[SRTMTileId]
    ) -> Tuple[np.ndarray, rio.transform.Affine]:
        file_names = [f"{self.base_dir}/{t}.hgt" for t in tiles]
        return rio.merge(file_names)  # type: ignore
