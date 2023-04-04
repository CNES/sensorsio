#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests for the Sentinel2 driver
"""
import os

from sensorsio import mgrs, sentinel2


def get_sentinel2_l2a_theia_folder() -> str:
    """
    Retrieve SRTM folder from env var
    """
    return os.path.join(os.environ['SENSORSIO_TEST_DATA_PATH'], 'sentinel2', 'l2a_maja',
                        'SENTINEL2B_20230219-105857-687_L2A_T31TCJ_C_V3-1')


def test_get_theia_tiles():
    """
    Test the get theia tiles function
    """
    theia_tiles = sentinel2.get_theia_tiles()

    assert len(theia_tiles) == 1076


def test_find_tile_orbit_pairs():
    """
    Test the find tile orbit pairs function
    """
    tile_id = '31TCJ'
    tile_bounds = mgrs.get_bbox_mgrs_tile(tile_id, latlon=False)
    tile_crs = mgrs.get_crs_mgrs_tile(tile_id)

    tiles_orbits_df = sentinel2.find_tile_orbit_pairs(tile_bounds, tile_crs)
    print(tile_crs)
    print(tile_bounds)
    print(tiles_orbits_df)
    assert len(tiles_orbits_df) == 15

    most_covered = tiles_orbits_df[tiles_orbits_df.tile_coverage > 0.9].copy()

    assert len(most_covered) == 2

    assert set(most_covered.tile_id) == {'31TCJ'}
    assert set(most_covered.relative_orbit_number) == {8, 51}
