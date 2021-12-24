#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import os
from dataclasses import dataclass
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.merge import merge


class BoundingBox:
    def __init__(self, lonmin: float, latmin: float, lonmax: float,
                 latmax: float):
        self.lonmin = min(lonmax, lonmin)
        self.latmin = min(latmax, latmin)
        self.lonmax = max(lonmax, lonmin)
        self.latmax = max(latmax, latmin)


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
    crs: Optional[str]
    transform: Optional[rio.transform.Affine]

    def as_stack(self):
        return np.stack([self.elevation, self.slope, self.aspect], axis=0)


@dataclass(frozen=True)
class SRTM:
    """
    Class for SRTM DEM (hgt format) product reading
    """
    base_dir: str = "/datalake/static_aux/MNT/SRTM_30_hgt"

    def get_dem_from_tiles(self, tiles: List[SRTMTileId]) -> DEM:
        """Build a DEM (elevation, slope and aspect) for a list of (supposedly
        adjacent) SRTM tiles. It keeps CRS and extent of the SRTM tiles.

        """
        elevation, transform = self.__build_hgt(tiles)
        elevation = elevation[0, :, :]
        x, y = np.gradient(elevation.astype(np.float16))
        slope = np.degrees(np.pi / 2. - np.arctan(np.sqrt(x * x + y * y)))
        aspect = np.degrees(np.arctan2(-x, y))
        return DEM(elevation.astype(np.int16),
                   slope.astype(np.int16),
                   aspect.astype(np.int16),
                   crs='+proj=latlong',
                   transform=transform)

    def get_dem_for_mgrs_tile(self, tile: str) -> DEM:
        """Build a DEM (elevation, slope and aspect) for an MGRS tile with the
union of the intersecting SRTM tiles. It keeps CRS and extent of the
SRTM tiles.

        """
        srtm_tiles = srtm_tiles_from_mgrs_tile(tile)
        return self.get_dem_from_tiles(srtm_tiles)

    def get_dem_for_bbox(self, bbox: BoundingBox) -> DEM:
        """Build a DEM (elevation, slope and aspect) for a bounding box with the
union of the intersecting SRTM tiles. It keeps CRS and extent of the
SRTM tiles.

        """
        srtm_tiles = srtm_tiles_from_bbox(bbox)
        return self.get_dem_from_tiles(srtm_tiles)

    def __build_hgt(
            self, tiles: List[SRTMTileId]
    ) -> Tuple[np.ndarray, rio.transform.Affine]:
        file_names = [
            f"{self.base_dir}/{srtm_id_to_name(t)}.hgt" for t in tiles
        ]
        return merge(file_names)  # type: ignore
