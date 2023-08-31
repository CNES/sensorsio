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
This module contains tests for the regulargrid functions
"""
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import rasterio as rio
from pyproj import CRS
from sensorsio import regulargrid


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
        vrt = regulargrid.create_warped_vrt(img_path, 10)
        assert vrt.height == cfg.size[1]
        assert vrt.width == cfg.size[0]
        assert vrt.count == cfg.nb_bands
        assert vrt.crs == cfg.crs
        assert vrt.nodata == cfg.nodata

        # By default vrt should return the same array
        vrt_arr = vrt.read()
        np.testing.assert_equal(img_arr, vrt_arr)

        # Check src_nodata and nodata flag
        vrt = regulargrid.create_warped_vrt(img_path, 10, src_nodata=-10)
        assert vrt.nodata == -10

        vrt = regulargrid.create_warped_vrt(img_path, 10, src_nodata=-10, nodata=-100)
        assert vrt.nodata == -100

        # In this case the warped vrt should expose a different nodata
        vrt = regulargrid.create_warped_vrt(img_path, 10, nodata=-100)
        assert vrt.nodata == -100
        vrt_arr = vrt.read()
        np.testing.assert_equal(
            vrt_arr[:, cfg.size[0] // 2:1 + cfg.size[0] // 2,
                    cfg.size[1] // 2:1 + cfg.size[1] // 2], -100)

        # Test different bounds
        dst_bounds = rio.coords.BoundingBox(cfg.origin[0], cfg.origin[1] - 5 * cfg.resolution,
                                            cfg.origin[0] + 5 * cfg.resolution, cfg.origin[1])

        vrt = regulargrid.create_warped_vrt(img_path, 10, dst_bounds=dst_bounds)
        assert vrt.height == 5
        assert vrt.width == 5

        # Test different resolution
        vrt = regulargrid.create_warped_vrt(img_path, 1)
        assert vrt.height == 100
        assert vrt.width == 100

        # Test different crs
        crs = 'epsg:2154'
        vrt = regulargrid.create_warped_vrt(img_path, 10, dst_crs=crs)
        assert vrt.crs == crs
        assert vrt.height == 10
        assert vrt.width == 10


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
        out_stack, xcoords, ycoords, crs = regulargrid.read_as_numpy([img1_path, img2_path],
                                                                     resolution=10)
        assert out_stack.shape == (2, 3, 10, 10)
        assert xcoords.shape == (10, )
        assert ycoords.shape == (10, )
        assert crs == cfg1.crs

        # separate=true
        out_stack, _, _, _ = regulargrid.read_as_numpy([img1_path, img2_path],
                                                       resolution=10,
                                                       separate=True)
        assert out_stack.shape == (3, 2, 10, 10)

        # scaling
        out_stack, _, _, _ = regulargrid.read_as_numpy([img1_path, img2_path],
                                                       resolution=10,
                                                       scale=300)
        assert out_stack.max() <= 1.0

        # Use an additional image with different crs
        with tempfile.NamedTemporaryFile(suffix='.tif') as temporary_file3:
            cfg3 = ImageConfig(nb_bands=3, crs='epsg:2154', origin=(499820, 6350510))
            img3_path, img3_arr = generate_dummy_image(temporary_file3.name, cfg3)
            with rio.open(img3_path) as ds:
                assert ds.crs == 'epsg:2154'

            out_stack, xcoords, ycoords, crs = regulargrid.read_as_numpy(
                [img1_path, img2_path, img3_path], resolution=10)
            assert out_stack.shape == (3, 3, 10, 10)
            assert xcoords.shape == (10, )
            assert ycoords.shape == (10, )
            assert crs == CRS.from_string(cfg1.crs)

            # Use a different target_crs
            out_stack, xcoords, ycoords, crs = regulargrid.read_as_numpy(
                [img1_path, img2_path, img3_path], resolution=10, crs=cfg3.crs)
            assert out_stack.shape == (3, 3, 10, 10)
            assert xcoords.shape == (10, )
            assert ycoords.shape == (10, )
            assert crs == cfg3.crs

            # Restrict bounds
            out_stack, xcoords, ycoords, crs = regulargrid.read_as_numpy(
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
                out_stack, xcoords, ycoords, crs = regulargrid.read_as_numpy(
                    [img1_path, img2_path, img4_path], resolution=10)
