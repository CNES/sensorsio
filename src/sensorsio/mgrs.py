#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

import os
from functools import lru_cache

import fiona
import geopandas as gpd
import numpy as np
import rasterio as rio
from pyproj import CRS, Transformer
from rasterio.coords import BoundingBox
from shapely import transform
from shapely.geometry import Polygon

from sensorsio import utils


@lru_cache
def get_polygon_mgrs_tile(tile: str) -> Polygon:
    """ Get the shapely.Polygon corresponding to a MGRS tile"""
    assert tile[0] != 'T'
    with fiona.open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data/sentinel2/mgrs_tiles.gpkg')) as f:
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
