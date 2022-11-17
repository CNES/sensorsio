#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains utilities function
"""

import math
from typing import List, Tuple, Union

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import Transformer
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

import time
from pyresample import geometry, kd_tree
from functools import partial
from concurrent.futures import ThreadPoolExecutor


def rgb_render(
        data: np.ndarray,
        clip: int = 2,
        bands: List[int] = [2, 1, 0],
        norm: bool = True,
        dmin: np.ndarray = None,
        dmax: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for visualization with matplot lib

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    assert (len(bands) == 1 or len(bands) == 3)
    assert (clip >= 0 and clip <= 100)

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


def generate_psf_kernel(res: float,
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


def create_warped_vrt(filename: str,
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

    with rio.open(filename) as src:
        target_bounds = None
        target_crs = src.crs
        if dst_crs is not None:
            target_crs = dst_crs
        if dst_bounds is not None:
            target_bounds = dst_bounds
        else:
            if target_crs != src.crs:
                target_bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
            else:
                target_bounds = src.bounds

        src_transform = src.transform
        if shifts is not None:
            src_res = src_transform[0]
            src_transform = Affine(src_res, 0.0, src_transform[2] - shifts[0],
                                   0.0, -src_res, src_transform[5] - shifts[1])

        # Compute optimized transform wrt. resolution and new crs
        left, bottom, right, top = target_bounds
        dst_width = int(np.floor((right - left) / resolution))
        dst_height = int(np.floor((top - bottom) / resolution))
        dst_transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)

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
        if b[0] > xmax or b[2] < xmin or b[1] > ymax or b[3] < ymin:
            raise ValueError('Bounding Box intersection is empty!')

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
    left = align * np.floor(2 * bb[0] / align) / 2
    right = left + align * (1 + np.floor((bb[2] - bb[0]) / align))
    bottom = align * np.floor(2 * bb[1] / align) / 2
    top = bottom + align * (1 + np.floor((bb[3] - bb[1]) / align))
    return BoundingBox(left=left, bottom=bottom, right=right, top=top)


def bb_common(bounds: List[BoundingBox],
              src_crs: List[str],
              snap: float = None,
              target_crs: str = None):
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
    assert (len(bounds) == len(src_crs))
    boxes = []
    for box, crs in zip(bounds, src_crs):
        if target_crs is None:
            target_crs = crs
        crs_box = rio.warp.transform_bounds(crs, target_crs, *box)
        boxes.append(crs_box)

    # Intersect all boxes
    box = bb_intersect(boxes)
    # Snap to grid
    if snap is not None:
        box = bb_snap(box, align=snap)
    return box, target_crs


def read_as_numpy(img_files: List[str],
                  crs: str = None,
                  resolution: float = 10,
                  offsets: Tuple[float, float] = None,
                  region: Union[Tuple[int, int, int, int],
                                rio.coords.BoundingBox] = None,
                  input_no_data_value: float = None,
                  output_no_data_value: float = np.nan,
                  bounds: rio.coords.BoundingBox = None,
                  algorithm=rio.enums.Resampling.cubic,
                  separate: bool = False,
                  dtype=np.float32,
                  scale: float = None) -> np.ndarray:
    """
    :param vrts: A list of WarpedVRT objects to stack
    :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
    :param dtype: dtype of the output Tensor
    :param separate: If True, each WarpedVRT is considered to offer a single band
    TODO
    """
    # Check if we need resampling or not
    need_warped_vrt = (offsets is not None)
    # If we change image bounds
    for f in img_files:
        with rio.open(f) as ds:
            if bounds is not None and ds.bounds != bounds:
                need_warped_vrt = True
            else:
                bounds = ds.bounds
            # If we change projection
            if crs is not None and crs != ds.crs:
                need_warped_vrt = True
            if ds.transform[0] != resolution:
                need_warped_vrt = True

    # If warped vrts are needed, create them
    if need_warped_vrt:
        datasets = [
            create_warped_vrt(f,
                              resolution,
                              dst_bounds=bounds,
                              dst_crs=crs,
                              nodata=input_no_data_value,
                              src_nodata=input_no_data_value,
                              resampling=algorithm,
                              shifts=offsets) for f in img_files
        ]

    else:
        datasets = [rio.open(f, 'r') for f in img_files]

    # Retrieve actual crs
    crs = datasets[0].crs

    # Read full img if region is None
    if region is None:
        region = datasets[0].bounds

    # Convert region to window
    if isinstance(region, BoundingBox):
        windows = [
            Window((region[0] - ds.bounds[0]) / ds.res[0],
                   (region[1] - ds.bounds[1]) / ds.res[1],
                   (region[2] - region[0]) / ds.res[0],
                   (region[3] - region[1]) / ds.res[1]) for ds in datasets
        ]
    else:
        windows = [
            Window(region[0], region[1], region[2] - region[0],
                   region[3] - region[1]) for ds in datasets
        ]

    axis = 0
    # if vrts are bands of the same image
    if separate:
        axis = 1
    np_stack = np.stack(
        [ds.read(window=w, masked=True) for (ds, w) in zip(datasets, windows)],
        axis=axis)

    # Close datasets
    for d in datasets:
        d.close()

    # If scaling is required, apply it
    if scale is not None:
        np_stack_mask = (np_stack == input_no_data_value)
        np_stack = np_stack / scale
        np_stack[np_stack_mask] = output_no_data_value

    # Convert to float before casting to final dtype
    np_stack = np_stack.astype(dtype)

    xcoords = np.linspace(
        bounds[0] + (0.5 + windows[0].col_off) * resolution, bounds[0] +
        (windows[0].col_off + np_stack.shape[3] - 0.5) * resolution,
        np_stack.shape[3])

    ycoords = np.linspace(
        bounds[3] - (windows[0].row_off + 0.5) * resolution, bounds[3] -
        (windows[0].row_off + np_stack.shape[2] - 0.5) * resolution,
        np_stack.shape[2])

    return np_stack, xcoords, ycoords, crs


def compute_latlon_bbox_from_region(bounds: BoundingBox,
                                    crs: str) -> BoundingBox:
    ul_from = (bounds.left, bounds.top)
    ur_from = (bounds.right, bounds.top)
    ll_from = (bounds.left, bounds.bottom)
    lr_from = (bounds.right, bounds.bottom)
    x_from = [p[0] for p in [ul_from, ur_from, ll_from, lr_from]]
    y_from = [p[1] for p in [ul_from, ur_from, ll_from, lr_from]]
    transformer = Transformer.from_crs(crs, '+proj=latlong')
    x_to, y_to = transformer.transform(x_from, y_from)
    return BoundingBox(np.min(x_to), np.min(y_to), np.max(x_to), np.max(y_to))


def swath_resample(latitudes: np.ndarray,
                   longitudes: np.ndarray,
                   target_crs: str,
                   target_bounds: rio.coords.BoundingBox,
                   target_resolution: float,
                   sigma: float,
                   nprocs: int = 6,
                   discrete_variables: np.ndarray = None,
                   continuous_variables: np.ndarray = None,
                   split_size: int = 1000,
                   fill_value: float = np.nan):
    """
    """

    nb_cols = int(
        np.round(
            (target_bounds.right - target_bounds.left - target_resolution) /
            target_resolution))
    nb_rows = int(
        np.round(
            (target_bounds.top - target_bounds.bottom - target_resolution) /
            target_resolution))

    print(nb_rows, nb_cols)

    # Define swath
    swath_def = geometry.SwathDefinition(lons=longitudes, lats=latitudes)

    start = time.perf_counter()

    # Define target area
    area_def = geometry.AreaDefinition('area', 'area', target_crs, target_crs,
                                       nb_cols, nb_rows, target_bounds)
    print(area_def)

    # Preprocess grid
    valid_input, valid_output, index_array, distance_array = kd_tree.get_neighbour_info(
        swath_def,
        area_def,
        2 * sigma,
        nprocs=nprocs,
        segments=1,
        reduce_data=True)

    preprocess_milestone = time.perf_counter()
    print(f'{preprocess_milestone-start=}')

    print(f"Starting a pool with {nprocs} threads")
    with ThreadPoolExecutor(max_workers=nprocs) as executor:
        # Use preprocessed grid on discrete variables
        if discrete_variables is not None:
            resample_function = partial(kd_tree.get_sample_from_neighbour_info,
                                        'nn', (nb_rows, nb_cols),
                                        valid_input_index=valid_input,
                                        valid_output_index=valid_output,
                                        index_array=np.take_along_axis(
                                            index_array,
                                            np.argmin(distance_array,
                                                      axis=-1,
                                                      keepdims=True),
                                            axis=-1)[:, 0],
                                        distance_array=None,
                                        weight_funcs=np.exp,
                                        fill_value=fill_value)
            out_dv = executor.map(resample_function, [
                discrete_variables[:, :, i]
                for i in range(discrete_variables.shape[-1])
            ])
            out_dv = np.stack((d for d in out_dv), axis=-1)
        else:
            out_dv = None

        # Use preprocessed grid on continuous variables
        if continuous_variables is not None:
            scaled_distances = -(distance_array / sigma)**2
            resample_function = partial(kd_tree.get_sample_from_neighbour_info,
                                        'custom', (nb_rows, nb_cols),
                                        valid_input_index=valid_input,
                                        valid_output_index=valid_output,
                                        index_array=index_array,
                                        distance_array=scaled_distances,
                                        weight_funcs=np.exp,
                                        fill_value=fill_value)
            out_cv = executor.map(resample_function, [
                continuous_variables[:, :, i]
                for i in range(continuous_variables.shape[-1])
            ])
            out_cv = np.stack((c for c in out_cv), axis=-1)
        else:
            out_cv = None

    xcoords = np.linspace(target_bounds[0] + target_resolution / 2,
                          target_bounds[2] - target_resolution / 2,
                          area_def.width)
    ycoords = np.linspace(target_bounds[3] - target_resolution / 2,
                          target_bounds[1] + target_resolution / 2,
                          area_def.height)

    end = time.perf_counter()
    print(f'{end-preprocess_milestone=}')

    return out_dv, out_cv, xcoords, ycoords
