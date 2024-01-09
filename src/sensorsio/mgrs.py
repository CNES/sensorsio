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

import os
from functools import lru_cache

import fiona  # type: ignore
import numpy as np
import geopandas as gpd
import rasterio as rio
from pyproj import CRS, Transformer
from rasterio.coords import BoundingBox
from shapely import transform  # type: ignore
from shapely.geometry import Polygon  # type: ignore


@lru_cache
def get_polygon_mgrs_tile(tile: str) -> Polygon:
    """ Get the shapely.Polygon corresponding to a MGRS tile"""
    assert tile[0] != 'T'
    with fiona.open('/vsizip/' + os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'data/sentinel2/mgrs_tiles.gpkg.zip', 'mgrs_tiles.gpkg')) as f:
        tiles = list(filter(lambda t: t['properties']['Name'] == tile, f))
        return Polygon(tiles[0].geometry['coordinates'][0])


@lru_cache
def get_bbox_mgrs_tile(tile: str, latlon: bool = True) -> BoundingBox:
    """
    Get a bounding box in '+proj=latlong' for a MGRS tile. If latlon
    is False, the bounding box is given in the CRS of the MGRS tile
    """
    poly = get_polygon_mgrs_tile(tile)

    if latlon:
        return BoundingBox(*poly.bounds)
    else:
        transformer = Transformer.from_crs('+proj=latlong', get_crs_mgrs_tile(tile))
        utm_bounds = transform(
            poly, lambda x: np.stack(transformer.transform(x[:, 0], x[:, 1]), axis=-1)).bounds

        # Align on 10m grid
        utm_bounds = (10 * np.round(coord / 10.) for coord in utm_bounds)
        return BoundingBox(*utm_bounds)


@lru_cache
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


@lru_cache
def get_transform_mgrs_tile(tile: str) -> rio.Affine:
    """ Get the rasterio.Affine transform for a MGRS tile in its own CRS"""
    ul, ur, ll, lr, _ = get_polygon_mgrs_tile(tile).exterior.coords
    transformer = Transformer.from_crs('+proj=latlong', get_crs_mgrs_tile(tile))
    x0, y0 = transformer.transform([ul[0]], [ul[1]])
    return rio.Affine(10.0, 0.0, np.round(x0[0]), 0.0, -10.0, np.round(y0[0]))


@lru_cache
def get_mgrs_tiles_from_roi(roi_poly: Polygon, crs_poly: str = "4326") -> list[str]:
    """
    Get MGRS tile ID list which cover a ROI
    """
    # Transform polygon to GeoDataFrame
    roi = gpd.GeoDataFrame(data={'id':[1],'geometry':[roi_poly]},crs=crs_poly)
    mgrs_grid = gpd.read_file('/vsizip/' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/sentinel2/mgrs_tiles.gpkg.zip', 'mgrs_tiles.gpkg'))
    # Get tile IDs corresponding to the ROI
    mgrs_tiles = gpd.overlay(mgrs_grid,roi,how="intersection")
    tile_ids = list(mgrs_tiles.Name.values)
    return tile_ids

