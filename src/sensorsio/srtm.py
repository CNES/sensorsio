#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio as rio
from pyproj import CRS, Transformer
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon


class BoundingBox:
    """
    Class modeling a lat, long bounding box
    """
    def __init__(self, lonmin: float, latmin: float, lonmax: float,
                 latmax: float):
        self.lonmin = min(lonmax, lonmin)
        self.latmin = min(latmax, latmin)
        self.lonmax = max(lonmax, lonmin)
        self.latmax = max(latmax, latmin)


@dataclass(frozen=True)
class SRTMTileId:
    """ Class modeling a SRTM tile """
    lon: int
    lat: int

    def name(self) -> str:
        """ Build the name of the tile from the lon, lat values"""
        northing = 'S' if self.lat < 0 else 'N'
        easting = 'W' if self.lon < 0 else 'E'
        lat = str(np.abs(self.lat)).zfill(2)
        lon = str(np.abs(self.lon)).zfill(3)
        return f"{northing}{lat}{easting}{lon}"


def srtm_tiles_from_bbox(bbox: BoundingBox) -> List[SRTMTileId]:
    """ Return a list of SRTMTileId intersecting a bounding box"""
    latmin = int(np.floor(bbox.latmin))
    latmax = int(np.floor(bbox.latmax))
    lonmin = int(np.floor(bbox.lonmin))
    lonmax = int(np.floor(bbox.lonmax))
    return [
        SRTMTileId(lon, lat) for lat in range(latmin, latmax + 1)
        for lon in range(lonmin, lonmax + 1)
    ]


def get_polygon_mgrs_tile(tile: str) -> Polygon:
    """ Get the shapely.Polygon corresponding to a MGRS tile"""
    assert tile[0] != 'T'
    mgrs_df = gpd.read_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data/sentinel2/mgrs_tiles.shp'))
    return mgrs_df[mgrs_df.Name == tile].iloc[0].geometry


def get_bbox_mgrs_tile(tile: str) -> BoundingBox:
    """ Get a bounding box in '+proj=latlong' for a MGRS tile"""
    poly = get_polygon_mgrs_tile(tile)
    return BoundingBox(*poly.bounds)


def get_srtm_tiles_for_mgrs_tile(tile: str) -> List[SRTMTileId]:
    """ Get the list of SRTM tiles intersecting a MGRS tile"""
    return srtm_tiles_from_bbox(get_bbox_mgrs_tile(tile))


def get_transform_mgrs_tile(tile: str) -> rio.Affine:
    """ Get the rasterio.Affine transform for a MGRS tile in its own CRS"""
    ul, ur, ll, lr, _ = get_polygon_mgrs_tile(tile).exterior.coords
    transformer = Transformer.from_crs('+proj=latlong',
                                       get_crs_mgrs_tile(tile))
    x0, y0 = transformer.transform([ul[0]], [ul[1]])
    return rio.Affine(10.0, 0.0, np.round(x0[0]), 0.0, -10.0, np.round(y0[0]))


def get_crs_mgrs_tile(tile: str) -> CRS:
    """Get the pyproj.CRS for a MGRS tile.

    The tile is given as 31TCJ: the 2 digits correspond to the UTM zone
    and the first letter is used to determine the northing (N-Z are
    North hemisphere, see
    https://en.wikipedia.org/wiki/Military_Grid_Reference_System#Grid_zone_designation)

    """
    zone = int(tile[:2])
    south = tile[2] < 'N'
    return CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': south})


@dataclass(frozen=True)
class DEM:
    """Class modeling a 3 channel (elevation, slope, aspect) DEM
    """
    elevation: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    crs: Optional[str]
    transform: Optional[rio.Affine]

    def as_stack(self):
        """Get the DEM as a single 3 channel numpy array"""
        return np.stack([self.elevation, self.slope, self.aspect], axis=0)

    def write(self, out_file: str):
        """Write the DEM to a geotiff file"""
        with rio.open(out_file,
                      'w',
                      driver='GTiff',
                      height=self.elevation.shape[0],
                      width=self.elevation.shape[1],
                      count=3,
                      nodata=-32768.0,
                      dtype=self.elevation.dtype,
                      compress='lzw',
                      crs=self.crs,
                      transform=self.transform) as ds:
            ds.write(self.elevation, 1)
            ds.write(self.slope, 2)
            ds.write(self.aspect, 3)


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

    def get_dem_mgrs_tile(self, tile: str) -> DEM:
        """Build a DEM (elevation, slope and aspect) for an MGRS tile
        with the union of the intersecting SRTM tiles. It keeps CRS and
        extent of the SRTM tiles.
        """
        srtm_tiles = get_srtm_tiles_for_mgrs_tile(tile)
        return self.get_dem_from_tiles(srtm_tiles)

    def get_dem_for_bbox(self, bbox: BoundingBox) -> DEM:
        """Build a DEM (elevation, slope and aspect) for a bounding box
        with the union of the intersecting SRTM tiles. It keeps CRS and
        extent of the SRTM tiles.
        """
        srtm_tiles = srtm_tiles_from_bbox(bbox)
        return self.get_dem_from_tiles(srtm_tiles)

    def __build_hgt(self,
                    tiles: List[SRTMTileId]) -> Tuple[np.ndarray, rio.Affine]:
        file_names = [f"{self.base_dir}/{t.name()}.hgt" for t in tiles]
        return merge(file_names)  # type: ignore


def get_dem_mgrs_tile(tile: str) -> DEM:
    """ Get a 10m resolution DEM on the geometry of a MGRS tile"""
    mgrs_trsf = get_transform_mgrs_tile(tile)
    mgrs_crs = get_crs_mgrs_tile(tile)
    dst_dem = np.zeros((3, 10980, 10980))
    dem_handler = SRTM()
    srtm_dem = dem_handler.get_dem_mgrs_tile(tile)
    dst_dem, dst_dem_transform = reproject(srtm_dem.as_stack(),
                                           destination=dst_dem,
                                           src_transform=srtm_dem.transform,
                                           src_crs=srtm_dem.crs,
                                           dst_transform=mgrs_trsf,
                                           dst_crs=mgrs_crs,
                                           resampling=Resampling.cubic)
    mgrs_dem = DEM(dst_dem[0, :, :].astype(np.int16),
                   dst_dem[1, :, :].astype(np.int16),
                   dst_dem[2, :, :].astype(np.int16), mgrs_crs,
                   dst_dem_transform)
    return mgrs_dem
