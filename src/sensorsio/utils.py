#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

"""
This module contains utilities function
"""

from typing import List, Tuple
import math
import numpy as np
from rasterio.enums import Resampling
from rasterio.coords import BoundingBox
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio import open as rio_open
from affine import Affine

def rgb_render(data: np.ndarray,
               clip: int = 2,
               bands: List[int] = [2,
                                   1,
                                   0],
               norm: bool = True,
               dmin: np.ndarray = None,
               dmax: np.ndarray = None) -> Tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.ndarray]:
    """
    Prepare data for visualization with matplot lib

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    assert(len(bands) == 1 or len(bands) == 3)
    assert(clip >= 0 and clip <= 100)

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)

    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            dmin = np.percentile(data_ready, clip, axis=(1, 2))
        if dmax is None:
            dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        data_ready = np.clip(
            (np.einsum("ijk->jki", data_ready) - dmin) / (dmax - dmin), 0, 1)

    else:
        data_ready = np.einsum("ijk->jki", data_ready)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, dmin, dmax


def generate_psf_kernel(
        res: float,
        mtf_res: float,
        mtf_fc: float,
        half_kernel_width: int = None) -> np.ndarray:
    """
    Generate a gaussian PSF kernel sampled at given resolution

    :param res: The resolution at which to sample the kernel
    :param mtf_res: The resolution at which mtf_fc is expressed
    :param half_kernel_width: The half size of the kernel
                              (determined automatically if None)

    :return: The kernel as a ndarray of shape
             [2*half_kernel_width+1, 2*half_kernel_width+1]
    """
    fc = 0.5 / mtf_res
    sigma = math.sqrt(-math.log(mtf_fc) / 2) / (math.pi * fc)
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


def create_warped_vrt(
        filename: str,
        resolution: float,
        dst_bounds: BoundingBox = None,
        dst_crs: str = None,
        src_nodata: float = None,
        nodata: float = None,
        shifts: Tuple[float] = None,
        resampling: Resampling = Resampling.cubic,
        dtype=None) -> WarpedVRT:
    """
    Create a warped vrt from filename, to change srs and resolution

    :param filename: Name of the image file
    :param resolution: Target resolution
    :param dst_bounds: Target bounds
    :param dst_crs: Target crs
    :param src_nodata: Value for missing data in source image
    :param nodata: Exposed value for missing data in VRT
    :param shifts: Shifts to apply to src origin for registration purposes
    :param resampling: Resampling method

    :return: A WarpedVRT object
    """
    with rio_open(filename) as src:
        target_bounds = src.bounds
        target_crs = src.crs
        if dst_crs is not None:
            target_crs = dst_crs
        if dst_bounds is not None:
            target_bounds = dst_bounds

        src_transform = src.transform
        if shifts is not None:
            src_res = src_transform[0]
            src_transform = Affine(src_res, 0.0, src_transform[2] - shifts[0],
                                   0.0, -src_res, src_transform[5] - shifts[1])

        # Compute optimized transform wrt. resolution and new crs
        left, bottom, right, top = target_bounds
        dst_width = (right - left) / resolution
        dst_height = (top - bottom) / resolution
        dst_transform = Affine(resolution, 0.0, left,
                               0.0, -resolution, top)

        vrt_options = {
            'resampling': resampling,
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            'crs': target_crs,
            'src_transform': src_transform

        }
        if src_nodata is not None:
            vrt_options['src_nodata'] = src_nodata
        elif src.nodata is not None:
            vrt_options['src_nodata'] = src.nodata
        if nodata is not None:
            vrt_options['nodata'] = nodata

        if dtype is not None:
            vrt_options['dtype'] = dtype

        vrt = WarpedVRT(src, **vrt_options)

        return vrt

def bb_intersect(bb: List[BoundingBox]) -> BoundingBox:
    """
    Compute the intersection of a list of bounding boxes

    :param bb: A list of BoundingBox objects
    :return: The intersection as a BoundingBox object
    """
    xmin = bb[0][0]
    xmax = bb[0][2]
    ymin = bb[0][1]
    ymax = bb[0][3]
    for b in bb[1:]:
        xmin = max(xmin, b[0])
        xmax = min(xmax, b[2])
        ymin = max(ymin, b[1])
        ymax = min(ymax, b[3])

    return BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)

def bb_snap(bb: BoundingBox, align: float = 20) -> BoundingBox:
    """
    Snap a bounding box to multiple of align parameter

    :param bb: The bounding box to snap as a BoundingBox object
    :param align: The step of the grip to align bounding box to

    :return: The snapped bounding box as a BoundingBox object
    """
    left = align * np.floor(bb[0] / align)
    right = align * np.ceil(bb[2] / align)
    bottom = align * np.floor(bb[1] / align)
    top = align * np.ceil(bb[3] / align)
    return BoundingBox(left=left, bottom=bottom, right=right, top=top)

def bb_common(imgs: List[str], snap: float = 20, crs: str = None):
    """
    Compute the common bounding box between a set of images.
    All bounding boxes are converted to crs before intersection.
    If crs is not set, crs from first image in list is used.
    After intersection, box is snapped to integer multiple of the snap parameter.

    param imgs: List of path to image files
    param snap: Box is snaped to interger multiple of this parameter
    param crs: Common CRS for all boxes. If None, crs from first image is used

    returns: A tuple of box, crs
    """
    boxes = []
    # Loop on images
    for img in imgs:
        # Open with rasterio
        with rio.open(img) as d:
            # Get crs from first image if None
            if crs is None:
                crs = d.crs
            box = d.bounds
            # Transform bounds to common crs
            crs_box = rio.warp.transform_bounds(d.crs, crs, *box)
            boxes.append(crs_box)
    # Intersect all boxes
    box = utils.bb_intersect(boxes)
    # Snap to grid
    box = utils.bb_snap(box, align=snap)
    return box, crs


def read_as_numpy(
        vrts: List[WarpedVRT],
        region: BoundingBox,
        dtype: np.dtype = np.float32,
        separate=False) -> np.ndarray:
    """
    Read a stack of VRTs as a numpy nd_array

    :param vrts: A list of WarpedVRT objects to stack
    :param region: The region to read as a BoundingBox object
    :param dtype: dtype of the output Tensor
    :param separate: If True, each WarpedVRT is considered to offer a single band

    :return: An array of shape [nb_vrts,nb_bands,w,h].
             If separate is True, shape is [1,nb_bands*nb_vrts,w, h]
    """
    # Convert region to window
    windows = [Window((region[0] - ds.bounds[0]) / ds.res[0],
                      (region[1] - ds.bounds[1]) / ds.res[1],
                      (region[2] - region[0]) / ds.res[0],
                      (region[3] - region[1]) / ds.res[1]) for ds in vrts]
    axis = 0
    # if vrts are bands of the same image
    if separate:
        axis = 1

    np_stack = np.stack([ds.read(window=w)
                         for (ds, w) in zip(vrts, windows)], axis=axis)

    # Convert to float before casting to final dtype
    np_stack = np_stack.astype(dtype)
    return np_stack
