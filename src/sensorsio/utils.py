#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
This module contains utilities function
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import rasterio as rio
import xarray as xr
from affine import Affine  # type: ignore
from pyproj import Transformer
from pyresample import geometry, kd_tree  # type: ignore
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window


def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: List[int] = [2, 1, 0],
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
    assert (len(bands) == 1 or len(bands) == 3)
    assert (clip >= 0 and clip <= 100)

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


def create_warped_vrt(filename: str,
                      resolution: float,
                      dst_bounds: Optional[BoundingBox] = None,
                      dst_crs: Optional[str] = None,
                      src_nodata: Optional[float] = None,
                      nodata: Optional[float] = None,
                      shifts: Optional[Tuple[float, float]] = None,
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
            src_transform = Affine(src_res, 0.0, src_transform[2] - shifts[0], 0.0, -src_res,
                                   src_transform[5] - shifts[1])

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


def bb_transform(source_crs: str,
                 target_crs: str,
                 bb: BoundingBox,
                 all_corners: bool = False) -> BoundingBox:
    """
    Transform a bounding box by solely looking at its 2 corners (upper-left and lower-right)
    If all_corners is True, also include upper-right and lower-left
    """
    source_x = [bb.left, bb.right]
    source_y = [bb.bottom, bb.top]
    if all_corners:
        source_x += [bb.left, bb.right]
        source_y += [bb.top, bb.bottom]
    if source_crs != target_crs:
        target_x, target_y = rio.warp.transform(source_crs, target_crs, source_x, source_y)
        xmin = min(target_x)
        xmax = max(target_x)
        ymin = min(target_y)
        ymax = max(target_y)
        return BoundingBox(xmin, ymin, xmax, ymax)
    return bb


def bb_intersect(bb: Iterable[BoundingBox]) -> BoundingBox:
    """
    Compute the intersection of a list of bounding boxes

    :param bb: A list of BoundingBox objects
    :return: The intersection as a BoundingBox object
    """
    bb_iter = iter(bb)
    first_elem = next(bb_iter)
    xmin = first_elem[0]
    xmax = first_elem[2]
    ymin = first_elem[1]
    ymax = first_elem[3]
    for b in bb_iter:
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
    left = align * np.floor(bb[0] / align)
    right = left + align * (1 + np.floor((bb[2] - bb[0]) / align))
    bottom = align * np.floor(bb[1] / align)
    top = bottom + align * (1 + np.floor((bb[3] - bb[1]) / align))
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
    assert (len(bounds) == len(src_crs))
    out_target_crs = target_crs
    boxes = []
    for box, crs in zip(bounds, src_crs):
        if out_target_crs is None:
            out_target_crs = crs
        crs_box = bb_transform(crs, out_target_crs, box)
        boxes.append(crs_box)

    # Intersect all boxes
    box = bb_intersect(boxes)
    # Snap to grid
    if snap is not None:
        box = bb_snap(box, align=snap)
    return box, out_target_crs


def read_as_numpy(img_files: List[str],
                  crs: Optional[str] = None,
                  resolution: float = 10,
                  offsets: Optional[Tuple[float, float]] = None,
                  region: Optional[Union[Tuple[int, int, int, int], rio.coords.BoundingBox]] = None,
                  input_no_data_value: Optional[float] = None,
                  output_no_data_value: float = np.nan,
                  bounds: Optional[rio.coords.BoundingBox] = None,
                  algorithm=rio.enums.Resampling.cubic,
                  separate: bool = False,
                  dtype=np.float32,
                  scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
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
    nb_bands = None
    out_crs = crs
    for f in img_files:
        with rio.open(f) as ds:
            if nb_bands is None:
                nb_bands = ds.count
            else:
                if nb_bands != ds.count:
                    raise ValueError("All image files need to have the same number of bands")
            if bounds is not None and ds.bounds != bounds:
                need_warped_vrt = True
                out_bounds = rio.coords.BoundingBox(*bounds)
            else:
                out_bounds = rio.coords.BoundingBox(*ds.bounds)
            # If we change projection
            if out_crs is None:
                out_crs = ds.crs
            if out_crs != ds.crs:
                need_warped_vrt = True
            if ds.transform[0] != resolution:
                need_warped_vrt = True

    # If warped vrts are needed, create them
    if need_warped_vrt:
        datasets = [
            create_warped_vrt(f,
                              resolution,
                              dst_bounds=out_bounds,
                              dst_crs=out_crs,
                              nodata=input_no_data_value,
                              src_nodata=input_no_data_value,
                              resampling=algorithm,
                              shifts=offsets) for f in img_files
        ]

    else:
        datasets = [rio.open(f, 'r') for f in img_files]

    axis = 0
    # if vrts are bands of the same image
    if separate:
        axis = 1

    # Read full img if region is None
    if region is not None and not need_warped_vrt:
        # Convert region to window
        if isinstance(region, BoundingBox):
            windows = []
            for ds in datasets:
                current_window = Window(
                    (region[0] - ds.bounds[0]) / ds.res[0], (region[1] - ds.bounds[1]) / ds.res[1],
                    (region[2] - region[0]) / ds.res[0], (region[3] - region[1]) / ds.res[1])
                windows.append(current_window)
        else:
            windows = [
                Window(region[0], region[1], region[2] - region[0], region[3] - region[1])
                for ds in datasets
            ]

        np_stack = np.stack([ds.read(window=w, masked=True) for (ds, w) in zip(datasets, windows)],
                            axis=axis)
    else:
        warnings.warn(
            'region parameter is set but read_as_numpy requires to use WarpedVRT. The region parameter will be ignored'
        )
        np_stack = np.stack([ds.read(masked=True) for ds in datasets], axis=axis)

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

    xcoords = np.linspace(out_bounds.left + 0.5 * resolution, out_bounds.right - 0.5 * resolution,
                          np_stack.shape[3])

    ycoords = np.linspace(out_bounds.top - 0.5 * resolution, out_bounds.bottom + 0.5 * resolution,
                          np_stack.shape[2])

    return np_stack, xcoords, ycoords, out_crs


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
    else:
        return mask.astype(int) >> bit & 1


def swath_resample(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_crs: str,
    target_bounds: rio.coords.BoundingBox,
    target_resolution: float,
    sigma: float,
    nthreads: int = 6,
    discrete_variables: Optional[np.ndarray] = None,
    continuous_variables: Optional[np.ndarray] = None,
    cutoff_sigma_mult: float = 2.,
    strip_size: int = 1500000,
    fill_value: float = np.nan,
    discrete_fill_value: int = 0,
    max_neighbours: int = 8
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    This function wraps and optimizes pyresample in order to resample
    swath data (i.e. observations indexed by an array of irregular latitudes and longitudes).

    :param latitudes: the array of latitudes of shape (in_height, in_width)
    :param longitudes: the array of longitudes of shape (in_height, in_width)
    :param target_crs: the target crs for resampling
    :param target_bounds: the target bounds for resampling as a rio.coords.BoundingBox object. Note that bounds define the outer edge of edge pixels
    :param target_resolution: the target resolution
    :param sigma: Width of the gausian weighting for the resampling
    :param cutoff_sigma_mult: Sigma multiplier for cutoff distance
    :param nthreads: Number of threads that will used. Can be higher than available cpus (but threads increase memory consumption, see strip_size parameter)
    :param discrete_variables: discrete variables to resample with nearest-neighbors, of shape (in_height, in_width, np_discrete). Can be None
    :param continuous_variables: continuous variables to resample with gaussian weighting, of shape (in_height, in_width, np_discrete)
    :param strip_size: Size of strip processed by a single thread, in pixels. Total memory is nthreads * memory required to process strip_size
    :param fill_value: Value to use for no-data in output arrays
    :param max_neighbours: Maximum number of neighbors considered

    :return: resampled discrete variables, resampled continuous variables, xcoords, ycoords
    """
    # Compute output number of rows and columns
    nb_cols = int(np.ceil((target_bounds.right - target_bounds.left) / target_resolution))
    nb_rows = int(np.ceil((target_bounds.top - target_bounds.bottom) / target_resolution))

    # Define swath
    swath_def = geometry.SwathDefinition(lons=longitudes, lats=latitudes)

    #start = time.perf_counter()

    # Define target area
    area_def = geometry.AreaDefinition('area', 'area', target_crs, target_crs, nb_cols, nb_rows,
                                       target_bounds)
    #print(area_def)

    # Preprocess grid
    valid_input, valid_output, index_array, distance_array = kd_tree.get_neighbour_info(
        swath_def,
        area_def,
        cutoff_sigma_mult * sigma,
        nprocs=nthreads,
        segments=1,
        reduce_data=True,
        neighbours=max_neighbours)

    # Scale distance by sigma, so that they are ready to be passed to np.exp
    scaled_distances = -(distance_array / sigma)**2

    #preprocess_milestone = time.perf_counter()
    #print(f'{preprocess_milestone-start=}')

    #print(f'{nthreads=}')
    # Start the threads pool
    with ThreadPoolExecutor(max_workers=nthreads) as executor:

        # Compute the height of the strips and the number of strips
        strip_size = max(1, int(np.floor(strip_size / nb_cols)))
        nb_strips = int(np.ceil(nb_rows / float(strip_size)))

        #print(f'{strip_size=}')
        #print(f'{nb_strips=}')

        # Those list will recieve the outputs of assynchronous map operations by the threads pool
        out_cvs_results = []
        out_dvs_results = []

        # Iterate on strips
        for s in range(nb_strips):

            # Compute parameters of current strip
            current_valid_output = valid_output[s * strip_size *
                                                nb_cols:min(nb_rows * nb_cols, (s + 1) *
                                                            strip_size * nb_cols)]
            current_index_array = index_array[s * strip_size *
                                              nb_cols:min(nb_rows * nb_cols, (s + 1) * strip_size *
                                                          nb_cols), :]
            current_distance_array = distance_array[s * strip_size *
                                                    nb_cols:min(nb_rows * nb_cols, (s + 1) *
                                                                strip_size * nb_cols), :]
            current_scaled_distances = scaled_distances[s * strip_size *
                                                        nb_cols:min(nb_rows * nb_cols, (s + 1) *
                                                                    strip_size * nb_cols), :]
            current_nb_rows = min(nb_rows, (s + 1) * strip_size) - s * strip_size

            # Process discrete variables
            if discrete_variables is not None:

                # Partial allows to use map directly with the pyresample function
                resample_function = partial(kd_tree.get_sample_from_neighbour_info,
                                            'nn', (current_nb_rows, nb_cols),
                                            valid_input_index=valid_input,
                                            valid_output_index=current_valid_output,
                                            index_array=np.take_along_axis(
                                                current_index_array,
                                                np.argmin(current_distance_array, axis=-1)[:, None],
                                                axis=-1)[:, 0],
                                            distance_array=None,
                                            weight_funcs=np.exp,
                                            fill_value=discrete_fill_value)

                # Map the partial function on each input discrete variable spearately
                # This call is assynchronous (computation runs in background)
                out_dv_futures = executor.map(
                    resample_function,
                    [discrete_variables[:, :, i] for i in range(discrete_variables.shape[-1])])
                # Keep tracks of pending results
                out_dvs_results.append(out_dv_futures)

            # Process continuous variables
            if continuous_variables is not None:
                # Same partial trick, see above
                resample_function = partial(kd_tree.get_sample_from_neighbour_info,
                                            'custom', (current_nb_rows, nb_cols),
                                            valid_input_index=valid_input,
                                            valid_output_index=current_valid_output,
                                            index_array=current_index_array,
                                            distance_array=current_scaled_distances,
                                            weight_funcs=np.exp,
                                            fill_value=fill_value)

                # Map call on each variable separately, assynchronous
                out_cv_futures = executor.map(
                    resample_function,
                    [continuous_variables[:, :, i] for i in range(continuous_variables.shape[-1])])
                # Keep track of pending results
                out_cvs_results.append(out_cv_futures)

    if continuous_variables is not None:
        # This code concatenates resulting variable for each strip, effectively joining all pending calculation
        out_cvs = [np.stack([c for c in v], axis=-1) for v in out_cvs_results]
        # And then stack strips
        out_cv: Optional[np.ndarray] = np.concatenate(out_cvs, axis=0)
    else:
        out_cv = None
    if discrete_variables is not None:
        # This code concatenates resulting variable for each strip, effectively joining all pending calculation
        out_dvs = [np.stack([c for c in v], axis=-1) for v in out_dvs_results]
        # And then stack strips
        out_dv = np.concatenate(out_dvs, axis=0)
    else:
        out_dv = None

    # Compute x and y coordinates at *center* of each output pixels
    xcoords = np.linspace(target_bounds[0] + target_resolution / 2,
                          target_bounds[2] - target_resolution / 2, area_def.width)
    ycoords = np.linspace(target_bounds[3] - target_resolution / 2,
                          target_bounds[1] + target_resolution / 2, area_def.height)

    #end = time.perf_counter()
    #print(f'{end-preprocess_milestone=}')

    return out_dv, out_cv, xcoords, ycoords
