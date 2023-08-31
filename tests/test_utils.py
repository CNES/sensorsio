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
This module contains tests for the utility functions
"""

import numpy as np
import rasterio as rio
from sensorsio import utils


def test_rgb_render():
    """
    Test thr rgb render function
    """
    in_data = np.arange(0, 600).reshape((6, 10, 10))

    # Simple case, no normalisation
    out_data, out_min, out_max = utils.rgb_render(in_data, norm=False)
    assert out_data.shape == (10, 10, 3)
    assert out_min is None
    assert out_max is None
    assert out_data.min() == 0.
    assert out_data.max() == 299.

    # Simple case, norm and clip
    out_data, out_min, out_max = utils.rgb_render(in_data, norm=True, clip=0)
    assert out_data.shape == (10, 10, 3)
    np.testing.assert_equal(out_min, (200, 100, 0))
    np.testing.assert_equal(out_max, (299, 199, 99))
    assert out_data.min() == 0.
    assert out_data.max() == 1.

    # Simple case, norm and input dmin / dmax
    out_data, out_min, out_max = utils.rgb_render(in_data,
                                                  norm=True,
                                                  clip=0,
                                                  dmin=np.array([0, 0, 0]),
                                                  dmax=np.array([150, 150, 150]))
    assert out_data.shape == (10, 10, 3)
    np.testing.assert_equal(out_min, (0, 0, 0))
    np.testing.assert_equal(out_max, (150, 150, 150))
    assert out_data.min() == 0.
    assert out_data.max() == 1.


def test_generate_psf_kernel():
    """
    Test the generate_psf_kernel function
    """
    kernel = utils.generate_psf_kernel(1., 10., 0.1, half_kernel_width=3)
    assert kernel.shape == (7, 7)
    np.testing.assert_allclose(kernel.sum(), 1.)

    kernel = utils.generate_psf_kernel(1., 10., 0.1, half_kernel_width=None)
    assert kernel.shape == (21, 21)
    np.testing.assert_allclose(kernel.sum(), 1.)


def test_bb_intersect():
    """
    Test the bb_intersect function
    """
    bb1 = rio.coords.BoundingBox(0, 0, 10, 10)
    bb2 = rio.coords.BoundingBox(5, 5, 10, 10)
    bb3 = rio.coords.BoundingBox(20, 20, 30, 30)

    bb_out = utils.bb_intersect([bb1, bb2])

    assert bb_out == rio.coords.BoundingBox(5, 5, 10, 10)

    with np.testing.assert_raises(ValueError):
        utils.bb_intersect([bb1, bb3])


def test_bb_snap():
    """
    Test the snapping function
    """
    bb = rio.coords.BoundingBox(-0.4, 0.1, 9.9, 10.3)

    assert utils.bb_snap(bb, align=10) == rio.coords.BoundingBox(-10, 0, 10, 20)


def assert_within(box1: rio.coords.BoundingBox,
                  crs1: str,
                  box2: rio.coords.BoundingBox,
                  crs2: str,
                  margin: float = 0.):
    """
    Optionnaly reproject box1 to crs2, and assert that box1 is completely within box2 
    """
    if crs1 != crs2:
        box1 = utils.bb_transform(rio.crs.CRS.from_string(crs1), rio.crs.CRS.from_string(crs2),
                                  box1)

    assert box2.left - margin <= box1.left <= box2.right + margin
    assert box2.left - margin <= box1.right <= box2.right + margin
    assert box2.bottom - margin <= box1.bottom <= box2.top + margin
    assert box2.bottom - margin <= box1.bottom <= box2.top + margin


def test_bb_common():
    """
    Test the bb common function
    """
    bounds1 = rio.coords.BoundingBox(left=300000.0, bottom=4790220.0, right=409800.0, top=4900020.0)
    crs1 = 'epsg:32631'
    bounds2 = rio.coords.BoundingBox(left=500825.0, bottom=6241658.0, right=608783.0, top=6349610.0)
    crs2 = 'epsg:2154'

    # All default
    out_box, out_crs = utils.bb_common([bounds1, bounds2], [crs1, crs2])

    assert out_crs == crs1
    assert_within(out_box, out_crs, bounds1, crs1)
    # Need to add a margin of 200m for the test to pass. Due to reprojection ?
    assert_within(out_box, out_crs, bounds2, crs2)

    # Different target crs
    out_box, out_crs = utils.bb_common([bounds1, bounds2], [crs1, crs2], target_crs='epsg:4326')

    assert out_crs == 'epsg:4326'
    assert_within(out_box, out_crs, bounds1, crs1)
    # Need to add a margin of 200m for the test to pass. Due to reprojection ?
    assert_within(out_box, out_crs, bounds2, crs2)

    # snap to 10m
    out_box, out_crs = utils.bb_common([bounds1, bounds2], [crs1, crs2], snap=10)

    assert out_crs == crs1
    assert_within(out_box, out_crs, bounds1, crs1)
    # Need to add a margin of 200m for the test to pass. Due to reprojection ?
    assert_within(out_box, out_crs, bounds2, crs2, 10)

    # Check the snapping is effective
    for v in out_box:
        assert np.modf(v / 10)[0] == 0.


def test_compute_latlon_from_bbox():
    """
    Test the compute latlon from bbox function
    """
    bounds1 = rio.coords.BoundingBox(left=300000.0, bottom=4790220.0, right=409800.0, top=4900020.)
    crs1 = 'epsg:32631'

    wgs84_bounds_1 = utils.compute_latlon_bbox_from_region(bounds=bounds1, crs=crs1)
    wgs84_bounds_2 = utils.bb_transform(crs1, 'epsg:4326', bounds1, all_corners=True)
    assert wgs84_bounds_1 == wgs84_bounds_2


def test_extract_bitmask():
    """
    Test the extract bitmask function
    """
    mask = np.full((10, 10), 8)

    for b in range(0, 7):
        if b != 3:
            np.testing.assert_equal(utils.extract_bitmask(mask, bit=b), np.full_like(mask, False))
        else:
            np.testing.assert_equal(utils.extract_bitmask(mask, bit=b), np.full_like(mask, True))
