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
    dtype: np.dtype = np.int32
    crs: str = 'EPSG:32631'
    resolution: float = 10.
    origin: Tuple[float, float] = (399960.0, 4800000.0)
    nodata: Optional[Union[np.float32, np.int16]] = -10000


def generate_dummy_image(file_path: str, cfg: ImageConfig = ImageConfig()) -> str:
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
