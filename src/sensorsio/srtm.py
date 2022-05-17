#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject

import mgrs


def compute_latlon_bbox_from_region(bounds: BoundingBox,
                                    crs: str) -> BoundingBox:
    ul_from = (bounds.left, bounds.top)
    ur_from = (bounds.right, bounds.top)
    ll_from = (bounds.left, bounds.bottom)
    lr_from = (bounds.right, bounds.bottom)
    x_from = [p[0] for p in [ul_from, ur_from, ll_from, lr_from]]
    y_from = [p[1] for p in [ul_from, ur_from, ll_from, lr_from]]
    transformer = Transformer.from_crs(crs, '+proj=latlong')
    x_to, y_to = transformer.transform(x_from, y_from)
    return BoundingBox(np.min(x_to), np.min(y_to), np.max(x_to), np.max(y_to))


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


def get_srtm_tiles_for_mgrs_tile(tile: str) -> List[SRTMTileId]:
    """ Get the list of SRTM tiles intersecting a MGRS tile"""
    return srtm_tiles_from_bbox(mgrs.get_bbox_mgrs_tile(tile))


def srtm_tiles_from_bbox(bbox: BoundingBox) -> List[SRTMTileId]:
    """ Return a list of SRTMTileId intersecting a bounding box"""
    bottom = int(np.floor(bbox.bottom))
    top = int(np.floor(bbox.top))
    left = int(np.floor(bbox.left))
    right = int(np.floor(bbox.right))
    return [
        SRTMTileId(lon, lat) for lat in range(bottom, top + 1)
        for lon in range(left, right + 1)
    ]


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


def write_dem(dem: DEM, out_file: str):
    """Write a DEM to a geotiff file"""
    with rio.open(out_file,
                  'w',
                  driver='GTiff',
                  height=dem.elevation.shape[0],
                  width=dem.elevation.shape[1],
                  count=3,
                  nodata=-32768.0,
                  dtype=dem.elevation.dtype,
                  compress='lzw',
                  crs=dem.crs,
                  transform=dem.transform) as ds:
        ds.write(dem.elevation, 1)
        ds.write(dem.slope, 2)
        ds.write(dem.aspect, 3)


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

    def read_as_numpy(
        self,
        crs: str,
        resolution: float,
        bounds: BoundingBox,
        no_data_value: float = np.nan,
        algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
        dtype: np.dtype = np.float32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               str]:
        assert bounds is not None
        dst_transform = rio.Affine(resolution, 0.0, bounds.left, 0.0,
                                   -resolution, bounds.top)
        dst_size_x = int(np.ceil((bounds.right - bounds.left) / resolution))
        dst_size_y = int(np.ceil((bounds.top - bounds.bottom) / resolution))
        dst_dem = np.zeros((3, dst_size_y, dst_size_x))
        dem_handler = SRTM()
        bbox = compute_latlon_bbox_from_region(bounds, crs)
        srtm_dem = dem_handler.get_dem_for_bbox(bbox)
        dst_dem, dst_dem_transform = reproject(
            srtm_dem.as_stack(),
            destination=dst_dem,
            src_transform=srtm_dem.transform,
            src_crs=srtm_dem.crs,
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=algorithm)
        np_arr_height = dst_dem[0, :, :].astype(dtype)
        np_arr_slope = dst_dem[1, :, :].astype(dtype)
        np_arr_aspect = dst_dem[2, :, :].astype(dtype)
        print(np_arr_height.shape, dst_size_x, dst_size_y)
        xcoords = np.arange(bounds.left, bounds.right, resolution)
        ycoords = np.arange(bounds.top, bounds.bottom, -resolution)
        return (np_arr_height, np_arr_slope, np_arr_aspect, xcoords, ycoords,
                crs, dst_dem_transform)

    def read_as_xarray(
            self,
            crs: str,
            resolution: float,
            bounds: BoundingBox,
            no_data_value: float = np.nan,
            algorithm: rio.enums.Resampling = rio.enums.Resampling.cubic,
            dtype: np.dtype = np.float32) -> xr.Dataset:
        (np_arr_height, np_arr_slope, np_arr_aspect, xcoords, ycoords, crs,
         transform) = self.read_as_numpy(crs, resolution, bounds,
                                         no_data_value, algorithm, dtype)
        vars: Dict[str, Tuple[List[str], np.ndarray]] = {}
        vars['height'] = (["y", "x"], np_arr_height)
        vars['slope'] = (["y", "x"], np_arr_slope)
        vars['aspect'] = (["y", "x"], np_arr_aspect)
        xarr = xr.Dataset(vars,
                          coords={
                              'x': xcoords,
                              'y': ycoords
                          },
                          attrs={
                              'crs': crs,
                              'resolution': resolution,
                              'transform': transform
                          })
        return xarr


def get_dem_mgrs_tile(tile: str) -> DEM:
    """ Get a 10m resolution DEM on the geometry of a MGRS tile"""
    mgrs_trsf = mgrs.get_transform_mgrs_tile(tile)
    mgrs_crs = mgrs.get_crs_mgrs_tile(tile)
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