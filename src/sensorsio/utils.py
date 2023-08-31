#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright: (c) 2021 Centre National d'Etudes Spatiales
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
This module contains utilities function
"""

import math
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from pyproj import Transformer
from rasterio.coords import BoundingBox


def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: Optional[List[int]] = None,
    norm: bool = True,
    dmin: Optional[np.ndarray] = None,
    dmax: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for visualization with matplot lib

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    if bands is None:
        bands = [2, 1, 0]
    assert len(bands) == 1 or len(bands) == 3
    assert 0 <= clip <= 100

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)
    out_dmin = None
    out_dmax = None
    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            out_dmin = np.percentile(data_ready, clip, axis=(1, 2))
        else:
            out_dmin = dmin
        if dmax is None:
            out_dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        else:
            out_dmax = dmax
        data_ready = np.clip((np.einsum("ijk->jki", data_ready) - out_dmin) / (out_dmax - out_dmin),
                             0, 1)

    else:
        data_ready = np.einsum("ijk->jki", data_ready)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, out_dmin, out_dmax


def generate_psf_kernel(res: float,
                        mtf_res: float,
                        mtf_fc: float,
                        half_kernel_width: Optional[int] = None) -> np.ndarray:
    """
    Generate a gaussian PSF kernel sampled at given resolution

    :param res: The resolution at which to sample the kernel
    :param mtf_res: The resolution at which mtf_fc is expressed
    :param half_kernel_width: The half size of the kernel
                              (determined automatically if None)

    :return: The kernel as a ndarray of shape
             [2*half_kernel_width+1, 2*half_kernel_width+1]
    """
    sigma = (mtf_res / math.pi) * math.sqrt(-2 * math.log(mtf_fc))
    if half_kernel_width is None:
        half_kernel_width = int(math.ceil(mtf_res / (res)))
    kernel = np.zeros((2 * half_kernel_width + 1, 2 * half_kernel_width + 1))
    for i in range(0, half_kernel_width + 1):
        for j in range(0, half_kernel_width + 1):
            dist = res * math.sqrt(i**2 + j**2)
            psf = np.exp(-(dist * dist) / (2 * sigma * sigma)) / \
                (sigma * math.sqrt(2 * math.pi))
            kernel[half_kernel_width - i, half_kernel_width - j] = psf
            kernel[half_kernel_width - i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width + j] = psf
            kernel[half_kernel_width + i, half_kernel_width - j] = psf

    kernel = kernel / np.sum(kernel)
    kernel = kernel.astype(np.float32)
    return kernel


def bb_transform(source_crs: str,
                 target_crs: str,
                 bounding_box: BoundingBox,
                 all_corners: bool = False) -> BoundingBox:
    """
    Transform a bounding box by solely looking at its 2 corners (upper-left and lower-right)
    If all_corners is True, also include upper-right and lower-left
    """
    source_x = [bounding_box.left, bounding_box.right]
    source_y = [bounding_box.bottom, bounding_box.top]
    if all_corners:
        source_x += [bounding_box.left, bounding_box.right]
        source_y += [bounding_box.top, bounding_box.bottom]
    if source_crs != target_crs:
        target_x, target_y = rio.warp.transform(source_crs, target_crs, source_x, source_y)
        xmin = min(target_x)
        xmax = max(target_x)
        ymin = min(target_y)
        ymax = max(target_y)
        return BoundingBox(xmin, ymin, xmax, ymax)
    return bounding_box


def bb_intersect(bounding_box: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute the intersection of a list of bounding boxes

    :param bb: A list of BoundingBox objects
    :return: The intersection as a BoundingBox object
    """
    bb_iter = iter(bounding_box)
    first_elem = next(bb_iter)
    xmin = first_elem[0]
    xmax = first_elem[2]
    ymin = first_elem[1]
    ymax = first_elem[3]
    for box in bb_iter:
        if box[0] > xmax or box[2] < xmin or box[1] > ymax or box[3] < ymin:
            raise ValueError('Bounding Box intersection is empty!')

        xmin = max(xmin, box[0])
        xmax = min(xmax, box[2])
        ymin = max(ymin, box[1])
        ymax = min(ymax, box[3])

    return BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)


def bb_snap(bounding_box: BoundingBox, align: float = 20) -> BoundingBox:
    """
    Snap a bounding box to multiple of align parameter

    :param bb: The bounding box to snap as a BoundingBox object
    :param align: The step of the grip to align bounding box to

    :return: The snapped bounding box as a BoundingBox object
    """
    left = align * np.floor(bounding_box[0] / align)
    right = left + align * (1 + np.floor((bounding_box[2] - bounding_box[0]) / align))
    bottom = align * np.floor(bounding_box[1] / align)
    top = bottom + align * (1 + np.floor((bounding_box[3] - bounding_box[1]) / align))
    return BoundingBox(left=left, bottom=bottom, right=right, top=top)


def bb_common(bounds: List[BoundingBox],
              src_crs: List[str],
              snap: Optional[float] = None,
              target_crs: Optional[str] = None) -> Tuple[rio.coords.BoundingBox, str]:
    """
    Compute the common bounding box between a set of images.
    All bounding boxes are converted to crs before intersection.
    If crs is not set, crs from first image in list is used.
    After intersection, box is snapped to integer multiple of the snap parameter.

    param bounds: List of bounding boxes
    param src_crs: List of correponding crs
    param snap: Box is snaped to interger multiple of this parameter
    param target_crs: Common CRS for all boxes. If None, first src_crs

    returns: A tuple of box, crs
    """
    assert (len(bounds) == len(src_crs) and len(bounds) > 0)

    if target_crs is not None:
        out_target_crs = target_crs
    else:
        out_target_crs = src_crs[0]

    boxes = []
    for box, crs in zip(bounds, src_crs):
        crs_box = bb_transform(crs, out_target_crs, box)
        boxes.append(crs_box)

    # Intersect all boxes
    box = bb_intersect(boxes)
    # Snap to grid
    if snap is not None:
        box = bb_snap(box, align=snap)
    return box, out_target_crs


def compute_latlon_bbox_from_region(bounds: BoundingBox, crs: str) -> BoundingBox:
    """
    Compute WGS84 bounding box from bounding box
    """
    # TODO: Might be redundant with bb_transform(all_corners=True)
    ul_from = (bounds.left, bounds.top)
    ur_from = (bounds.right, bounds.top)
    ll_from = (bounds.left, bounds.bottom)
    lr_from = (bounds.right, bounds.bottom)
    x_from = [p[0] for p in [ul_from, ur_from, ll_from, lr_from]]
    y_from = [p[1] for p in [ul_from, ur_from, ll_from, lr_from]]
    transformer = Transformer.from_crs(crs, '+proj=latlong', always_xy=True)
    # pylint: disable=unpacking-non-sequence
    x_to, y_to = transformer.transform(x_from, y_from)
    return BoundingBox(np.min(x_to), np.min(y_to), np.max(x_to), np.max(y_to))


def extract_bitmask(mask: Union[xr.DataArray, np.ndarray], bit: int = 0) -> np.ndarray:
    """
    Extract a binary mask from the nth bit of a bit-encoded mask

    :param mask: the bit encoded mask
    :param bit: the index of the bit to extract
    :return: A binary mask of the nth bit of mask, with the same shape
    """
    if isinstance(mask, xr.DataArray):
        return mask.values.astype(int) >> bit & 1
    return mask.astype(int) >> bit & 1
