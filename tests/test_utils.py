#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains tests for the utility functions
"""
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import rasterio as rio
from pyproj import CRS
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


@dataclass(frozen=True)
class ImageConfig:
    """
    Dummy image configuration
    """
    size: Tuple[int, int] = (10, 10)
    nb_bands: int = 3
    dtype: np.dtype = np.dtype('int32')
    crs: str = 'EPSG:32631'
    resolution: float = 10.
    origin: Tuple[float, float] = (399960.0, 4800000.0)
    nodata: Optional[Union[np.float32, np.int16, int]] = -10000


def generate_dummy_image(
    file_path: str, cfg: ImageConfig = ImageConfig()) -> Tuple[str, np.ndarray]:
    """
    Code to generate dummy image
    """
    transform = rio.Affine(cfg.resolution, 0, cfg.origin[0], 0, -cfg.resolution, cfg.origin[1])
    with rio.open(file_path,
                  "w",
                  driver="GTiff",
                  height=cfg.size[0],
                  width=cfg.size[1],
                  count=cfg.nb_bands,
                  dtype=cfg.dtype,
                  crs=cfg.crs,
                  nodata=cfg.nodata,
                  transform=transform) as rio_ds:
        arr = np.arange(cfg.size[0] * cfg.size[1] * cfg.nb_bands).reshape((cfg.nb_bands, *cfg.size))
        if cfg.nodata is not None:
            arr[:, cfg.size[0] // 2:1 + cfg.size[0] // 2,
                cfg.size[1] // 2:1 + cfg.size[1] // 2] = cfg.nodata
        rio_ds.write(arr)

        return file_path, arr


def test_create_warped_vrt():
    """
    Test the create warped vrt function
    """
    with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file:
        cfg = ImageConfig()
        img_path, img_arr = generate_dummy_image(temporary_file.name, cfg)

        # Check default arguments
        vrt = utils.create_warped_vrt(img_path, 10)
        assert vrt.height == cfg.size[1]
        assert vrt.width == cfg.size[0]
        assert vrt.count == cfg.nb_bands
        assert vrt.crs == cfg.crs
        assert vrt.nodata == cfg.nodata

        # By default vrt should return the same array
        vrt_arr = vrt.read()
        np.testing.assert_equal(img_arr, vrt_arr)

        # Check src_nodata and nodata flag
        vrt = utils.create_warped_vrt(img_path, 10, src_nodata=-10)
        assert vrt.nodata == -10

        vrt = utils.create_warped_vrt(img_path, 10, src_nodata=-10, nodata=-100)
        assert vrt.nodata == -100

        # In this case the warped vrt should expose a different nodata
        vrt = utils.create_warped_vrt(img_path, 10, nodata=-100)
        assert vrt.nodata == -100
        vrt_arr = vrt.read()
        np.testing.assert_equal(
            vrt_arr[:, cfg.size[0] // 2:1 + cfg.size[0] // 2,
                    cfg.size[1] // 2:1 + cfg.size[1] // 2], -100)

        # Test different bounds
        dst_bounds = rio.coords.BoundingBox(cfg.origin[0], cfg.origin[1] - 5 * cfg.resolution,
                                            cfg.origin[0] + 5 * cfg.resolution, cfg.origin[1])

        vrt = utils.create_warped_vrt(img_path, 10, dst_bounds=dst_bounds)
        assert vrt.height == 5
        assert vrt.width == 5

        # Test different resolution
        vrt = utils.create_warped_vrt(img_path, 1)
        assert vrt.height == 100
        assert vrt.width == 100

        # Test different crs
        crs = 'epsg:2154'
        vrt = utils.create_warped_vrt(img_path, 10, dst_crs=crs)
        assert vrt.crs == crs
        assert vrt.height == 10
        assert vrt.width == 10


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


def test_read_as_numpy():
    """
   Test the read as numpy function 
    """
    with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file1,\
         tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file2:
        cfg1 = ImageConfig(nb_bands=3)
        img1_path, img1_arr = generate_dummy_image(temporary_file1.name, cfg1)
        cfg2 = ImageConfig(nb_bands=3)
        img2_path, img2_arr = generate_dummy_image(temporary_file2.name, cfg2)

        # All default
        out_stack, xcoords, ycoords, crs = utils.read_as_numpy([img1_path, img2_path],
                                                               resolution=10)
        assert out_stack.shape == (2, 3, 10, 10)
        assert xcoords.shape == (10, )
        assert ycoords.shape == (10, )
        assert crs == cfg1.crs

        # separate=true
        out_stack, _, _, _ = utils.read_as_numpy([img1_path, img2_path],
                                                 resolution=10,
                                                 separate=True)
        assert out_stack.shape == (3, 2, 10, 10)

        # scaling
        out_stack, _, _, _ = utils.read_as_numpy([img1_path, img2_path], resolution=10, scale=300)
        assert out_stack.max() <= 1.0

        # Use an additional image with different crs
        with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file3:
            cfg3 = ImageConfig(nb_bands=3, crs='epsg:2154', origin=(499820, 6350510))
            img3_path, img3_arr = generate_dummy_image(temporary_file3.name, cfg3)
            with rio.open(img3_path) as ds:
                assert ds.crs == 'epsg:2154'

            out_stack, xcoords, ycoords, crs = utils.read_as_numpy(
                [img1_path, img2_path, img3_path], resolution=10)
            assert out_stack.shape == (3, 3, 10, 10)
            assert xcoords.shape == (10, )
            assert ycoords.shape == (10, )
            assert crs == CRS.from_string(cfg1.crs)

            # Use a different target_crs
            out_stack, xcoords, ycoords, crs = utils.read_as_numpy(
                [img1_path, img2_path, img3_path], resolution=10, crs=cfg3.crs)
            assert out_stack.shape == (3, 3, 10, 10)
            assert xcoords.shape == (10, )
            assert ycoords.shape == (10, )
            assert crs == cfg3.crs

            # Restrict bounds
            out_stack, xcoords, ycoords, crs = utils.read_as_numpy(
                [img1_path, img2_path, img3_path],
                resolution=10,
                crs=cfg3.crs,
                bounds=(cfg3.origin[0], cfg3.origin[1] - 50, cfg3.origin[0] + 50, cfg3.origin[1]))
            assert out_stack.shape == (3, 3, 5, 5)
            assert xcoords.shape == (5, )
            assert ycoords.shape == (5, )
            np.testing.assert_allclose(
                xcoords,
                np.arange(cfg3.origin[0] + 5, cfg3.origin[0] + 5 + 10 * xcoords.shape[0], 10.))
            np.testing.assert_allclose(
                ycoords,
                np.arange(cfg3.origin[1] - 5, cfg3.origin[1] - 5 - 10 * ycoords.shape[0], -10.))

            assert crs == cfg3.crs

        # Test that using files with different number of bands raises ValueError
        with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file4:
            cfg4 = ImageConfig(nb_bands=1)
            img4_path, img4_arr = generate_dummy_image(temporary_file4.name, cfg4)

            with np.testing.assert_raises(ValueError):
                out_stack, xcoords, ycoords, crs = utils.read_as_numpy(
                    [img1_path, img2_path, img4_path], resolution=10)


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


def test_swath_resample():
    """
    Test the swath resample method
    """

    # Create WGS84 bounds from UTM 31N bounds
    bounds = rio.coords.BoundingBox(left=300000.0, bottom=4790220.0, right=409800.0, top=4900020.)
    crs = 'epsg:32631'
    wgs84_bounds = utils.bb_transform(crs, 'epsg:4326', bounds, all_corners=True)

    # Now generate irregularly sampled lat/lon
    lon_1d = np.array([
        wgs84_bounds.left + (wgs84_bounds.right - wgs84_bounds.left) / (1.5**p) for p in range(10)
    ])
    lat_1d = np.array([
        wgs84_bounds.bottom + (wgs84_bounds.top - wgs84_bounds.bottom) / (1.5**p) for p in range(10)
    ])

    # Generate 2d sampling grid
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    nb_discrete_vars = 2
    nb_continuous_vars = 3

    continuous_vars = np.arange(lon_2d.shape[0] * lon_2d.shape[1] * nb_continuous_vars).reshape(
        *lon_2d.shape, nb_continuous_vars).astype(float)

    discrete_vars = np.arange(lon_2d.shape[0] * lon_2d.shape[1] * nb_discrete_vars).reshape(
        *lon_2d.shape, nb_discrete_vars)

    out_dv, out_cv, xcoords, ycoords = utils.swath_resample(lat_2d,
                                                            lon_2d,
                                                            target_crs=crs,
                                                            target_bounds=bounds,
                                                            target_resolution=100.,
                                                            sigma=100.,
                                                            discrete_variables=discrete_vars,
                                                            continuous_variables=continuous_vars)

    assert out_cv.shape == (1098, 1098, nb_continuous_vars)
    assert out_dv.shape == (1098, 1098, nb_discrete_vars)
    assert xcoords.shape == (1098, )
    assert ycoords.shape == (1098, )

    assert (~np.isnan(out_cv)).sum() == 2970
    assert np.nanmax(out_cv) == 296.
    assert np.nanmin(out_cv) == 33.
