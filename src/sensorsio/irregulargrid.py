#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
#
# Licensed under the Lesser GNU LESSER GENERAL PUBLIC
# LICENSE, Version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.gnu.org/licenses/lgpl-3.0.txt

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains utilities function
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Tuple

import numpy as np
from pyresample import geometry, kd_tree  # type: ignore
from rasterio.coords import BoundingBox


def swath_resample(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_crs: str,
    target_bounds: BoundingBox,
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
    """This function wraps and optimizes pyresample in order to resample
    swath data (i.e. observations indexed by an array of irregular
    latitudes and longitudes).

    :param latitudes: the array of latitudes of shape (in_height, in_width)
    :param longitudes: the array of longitudes of shape (in_height, in_width)
    :param target_crs: the target crs for resampling
    :param target_bounds: the target bounds for resampling as
     a rio.coords.BoundingBox object. Note that bounds define the outer edge
    of edge pixels
    :param target_resolution: the target resolution
    :param sigma: Width of the gausian weighting for the resampling
    :param cutoff_sigma_mult: Sigma multiplier for cutoff distance
    :param nthreads: Number of threads that will used. Can be higher
    than available cpus (but threads increase memory consumption, see
    strip_size parameter)
    :param discrete_variables: discrete variables to resample with
    nearest-neighbors, of shape (in_height, in_width,
    np_discrete). Can be None
    :param continuous_variables: continuous variables to resample with
    gaussian weighting, of shape (in_height, in_width, np_discrete)

    :param strip_size: Size of strip processed by a single thread, in
    pixels. Total memory is nthreads * memory required to process
    strip_size
    :param fill_value: Value to use for no-data in output arrays
    :param max_neighbours: Maximum number of neighbors considered

    :return: resampled discrete variables, resampled continuous
    variables, xcoords, ycoords
    """
    # Compute output number of rows and columns
    nb_cols = int(np.ceil((target_bounds.right - target_bounds.left) / target_resolution))
    nb_rows = int(np.ceil((target_bounds.top - target_bounds.bottom) / target_resolution))

    # Define swath
    swath_def = geometry.SwathDefinition(lons=longitudes, lats=latitudes)

    # Define target area
    area_def = geometry.AreaDefinition('area', 'area', target_crs, target_crs, nb_cols, nb_rows,
                                       target_bounds)
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

    # Start the threads pool
    with ThreadPoolExecutor(max_workers=nthreads) as executor:

        # Compute the height of the strips and the number of strips
        strip_size = max(1, int(np.floor(strip_size / nb_cols)))
        nb_strips = int(np.ceil(nb_rows / float(strip_size)))

        # Those list will recieve the outputs of assynchronous map
        # operations by the threads pool
        out_cvs_results = []
        out_dvs_results = []

        # Iterate on strips
        for strip in range(nb_strips):

            # Compute parameters of current strip
            current_valid_output = valid_output[strip * strip_size *
                                                nb_cols:min(nb_rows * nb_cols, (strip + 1) *
                                                            strip_size * nb_cols)]
            current_index_array = index_array[strip * strip_size *
                                              nb_cols:min(nb_rows * nb_cols, (strip + 1) *
                                                          strip_size * nb_cols), :]
            current_distance_array = distance_array[strip * strip_size *
                                                    nb_cols:min(nb_rows * nb_cols, (strip + 1) *
                                                                strip_size * nb_cols), :]
            current_scaled_distances = scaled_distances[strip * strip_size *
                                                        nb_cols:min(nb_rows * nb_cols, (strip + 1) *
                                                                    strip_size * nb_cols), :]
            current_nb_rows = min(nb_rows, (strip + 1) * strip_size) - strip * strip_size

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
        # This code concatenates resulting variable for each strip,
        # effectively joining all pending calculation
        out_cvs = [np.stack(list(v), axis=-1) for v in out_cvs_results]
        # And then stack strips
        out_cv: Optional[np.ndarray] = np.concatenate(out_cvs, axis=0)
    else:
        out_cv = None
    if discrete_variables is not None:
        # This code concatenates resulting variable for each strip,
        # effectively joining all pending calculation
        out_dvs = [np.stack(list(v), axis=-1) for v in out_dvs_results]
        # And then stack strips
        out_dv = np.concatenate(out_dvs, axis=0)
    else:
        out_dv = None

    # Compute x and y coordinates at *center* of each output pixels
    xcoords = np.linspace(target_bounds[0] + target_resolution / 2,
                          target_bounds[2] - target_resolution / 2, area_def.width)
    ycoords = np.linspace(target_bounds[3] - target_resolution / 2,
                          target_bounds[1] + target_resolution / 2, area_def.height)

    return out_dv, out_cv, xcoords, ycoords
