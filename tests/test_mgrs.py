#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
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
"""
This module contains tests for the mgrs functions
"""

import unittest

import geopandas as gpd
import rasterio as rio
import shapely
from shapely.geometry import Polygon

from sensorsio import mgrs, utils


def test_get_mgrs_tiles_from_roi():
    """
    Test get_mgrs_tiles_from_roi function
    """
    test = unittest.TestCase()

    # Bounding box in WGS84 like MGRS grid
    roi_bbox_wgs84 = rio.coords.BoundingBox(-17.1, 13.8, -15.9, 15.0)
    roi_crs_wgs84 = 4326
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox_wgs84, roi_crs_wgs84)

    assert isinstance(mgrs_tiles, gpd.GeoDataFrame)
    assert mgrs_tiles.crs.to_epsg() == 4326
    test.assertCountEqual(list(mgrs_tiles.Name.values),
                          ['28PBA', '28PBB', '28PCA', '28PCB', '28PDA', '28PDB'])
    test.assertCountEqual(list(mgrs_tiles.columns),
                          ['Name', 'overlap_geometry', 'overlap_percentage', 'geometry'])
    poly = Polygon(((-16.8632889144, 15.3693449926), (-15.8404703036, 15.3755536177),
                    (-15.8366342042, 14.3829233976), (-16.8547891789, 14.3771328743),
                    (-16.8632889144, 15.3693449926)))
    shapely.equals_exact(mgrs_tiles.loc[mgrs_tiles.Name == "28PCB"].geometry.values[0], poly)
    # Bounding box in UTM
    roi_crs_utm = 32628
    roi_bbox_utm = utils.bb_transform(roi_crs_wgs84, roi_crs_utm, roi_bbox_wgs84)
    mgrs_tiles = mgrs.get_mgrs_tiles_from_roi(roi_bbox_utm, roi_crs_utm)

    assert isinstance(mgrs_tiles, gpd.GeoDataFrame)
    assert mgrs_tiles.crs.to_epsg() == 4326
    test.assertCountEqual(list(mgrs_tiles.Name.values),
                          ['28PBA', '28PBB', '28PCA', '28PCB', '28PDA', '28PDB'])
