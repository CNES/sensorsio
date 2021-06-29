#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales

import warnings
from enum import Enum
from collections import namedtuple
import dateutil
from typing import List, Tuple, Union
import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import xml.etree.ElementTree as ET
import rasterio as rio
from shapely import geometry
from sensorsio import utils
from scipy import ndimage
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=RuntimeWarning, module='geopandas')

"""
This module contains Sentinel2 (L2A MAJA) related functions
"""

def get_theia_tiles():
    """
    Return a dataframe with tiles produced by Theia
    """
    return gpd.read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/sentinel2/theia_s2.gpkg')).set_index('Name')


def find_tile_orbit_pairs(bounds:rio.coords.BoundingBox, crs='epsg:4326'):
    """
    From bounding box and CRS, return a list of pairs of MGRS tiles
    and Sentinel2 relative orbits that covers the area, in the form of a dataframe.
    """
    # Convert bounds to 4326
    wgs84_bounds = rio.warp.transform_bounds(crs, 4326, *bounds)

    # Convert bounds to polygon
    aoi = geometry.Polygon([[wgs84_bounds[0], wgs84_bounds[1]],
                            [wgs84_bounds[0], wgs84_bounds[3]],
                            [wgs84_bounds[2], wgs84_bounds[3]],
                            [wgs84_bounds[2], wgs84_bounds[1]]])
    mgrs_df = gpd.read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/sentinel2/mgrs_tiles.shp'))
    orbits_df = gpd.read_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'data/sentinel2/orbits.gpkg'))
    intersections=[]
    for mgrs_id, mgrs_row in mgrs_df.iterrows():
        if aoi.intersects(mgrs_row.geometry):
            inter_mgrs_aoi = aoi.intersection(mgrs_row.geometry)
            mgrs_coverage = inter_mgrs_aoi.area/aoi.area
            orbits=[]
            for orbit_id,orbit_row in orbits_df.iterrows():
                # Last test is to exclude weird duplicates (malformed gpkg ?)
                if orbit_row.geometry.intersects(inter_mgrs_aoi) and not orbit_row.orbit_number in orbits:
                    orbits.append(orbit_row.orbit_number)
                    inter_mgrs_aoi_orbit = inter_mgrs_aoi.intersection(orbit_row.geometry)
                    mgrs_orbit_coverage = inter_mgrs_aoi_orbit.area/aoi.area
                    intersections.append((mgrs_row.Name,orbit_row.orbit_number,mgrs_coverage,mgrs_orbit_coverage))
    # Build a standard pandas df from tuples
    labels=['tile_id','relative_orbit_number','tile_coverage','tile_and_orbit_coverage']
    df = pd.DataFrame.from_records(intersections,columns=labels)
    return df


class Sentinel2:
    """
    Class for Sentinel2 L2A (MAJA format) product reading
    """     
    def __init__(self, product_dir:str, offsets:Tuple[float, float]=None, parse_xml=True):
        """
        Constructor

        :param product_dir: Path to product directory
        :param offsets: Shifts applied to image orgin (as computed by StackReg for instance)
        :param parse_xml: If True (default), parse additional information from xml metadata file
        """
        # Store product DIR
        self.product_dir = os.path.normpath(product_dir)
        self.product_name = os.path.basename(self.product_dir)

        # Look for xml file
        self.xml_file = self.build_xml_path()
                
        # Store offsets
        self.offsets = offsets

        # Get
        self.satellite = Sentinel2.Satellite(self.product_name[0:10])
                
        # Get tile
        self.tile = self.product_name[36:41]

        # Get acquisition date
        self.date = dateutil.parser.parse(self.product_name[11:19])
        self.year = self.date.year
        self.day_of_year = self.date.timetuple().tm_yday
        
        with rio.open(self.build_band_path(Sentinel2.B2)) as ds:
        # Get bounds
            self.bounds  = ds.bounds
            self.transform = ds.transform
        # Get crs
            self.crs = ds.crs

        # Init angles
        self.sun_angles = None
        self.incidence_angles = None
            
        # Parse xml file if requested
        if parse_xml:
            self.parse_xml()
            
    def __repr__(self):
        return f'{self.satellite.value}, {self.date}, {self.tile}'

    def parse_xml(self):
        """
        Parse metadata file
        """
        with open(self.xml_file) as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            # Parse cloud cover
            quality_node = root.find(".//*[@name='CloudPercent']")
            if quality_node is not None:
                self.cloud_cover = int(quality_node.text)
            # Parse orbit number
            orbit_node = root.find(".//ORBIT_NUMBER")
            if orbit_node is not None:
                self.orbit = int(orbit_node.text)
                self.relative_orbit_number = self.compute_relative_orbit_number(self.orbit)

            # Internal parsing function for angular grids
            def parse_angular_grid_node(node):
                    values=[]
                    for c in node.find('Values_List'):
                        values.append(np.array([float(t) for t in c.text.split()]))
                        values_array = np.stack(values)
                    return values_array

            # Parse sun angles
            Angles = namedtuple('Angles', 'zenith azimuth')

            self.sun_angles = Angles(parse_angular_grid_node(root.find('.//Angles_Grids_List/Sun_Angles_Grids/Zenith')),
                                     parse_angular_grid_node(root.find('.//Angles_Grids_List/Sun_Angles_Grids/Azimuth')))

            # Parse incidence angles
            self.incidence_angles = {}
            for b in root.find('.//Angles_Grids_List/Viewing_Incidence_Angles_Grids_List'):
                band_key = self.Band(b.attrib['band_id'])
                band_dict = {}
                for d in b.findall('Viewing_Incidence_Angles_Grids'):
                    det_key=self.Detector(int(d.attrib['detector_id']))
                    zen = parse_angular_grid_node(d.find('Zenith'))
                    az = parse_angular_grid_node(d.find('Azimuth'))
                    band_dict[det_key]=Angles(zen, az)
                self.incidence_angles[band_key]=band_dict
                                                                    
            
    def compute_relative_orbit_number(self, orbit):
        """
        Compute relative orbit number from absolute orbit and sensor id
        """
        phase = None
        if self.satellite is Sentinel2.Satellite.S2A:
            phase = 2
        else:
            phase = -27
        return ((orbit+phase)%143)+1
                            
                 
    # Enum class for sensor
    class Satellite(Enum):
        S2A = 'SENTINEL2A'
        S2B = 'SENTINEL2B'

    # Aliases
    S2A = Satellite.S2A
    S2B = Satellite.S2B

    # Enum class for Sentinel2 bands
    class Band(Enum):
        B2 = 'B2'
        B3 = 'B3'
        B4 = 'B4'
        B5 = 'B5'
        B6 = 'B6'
        B7 = 'B7'
        B8 = 'B8'
        B8A = 'B8A'
        B9 = 'B9'
        B10 = 'B10'
        B11 = 'B11'
        B12 = 'B12'
        
    # Aliases
    B2 = Band.B2
    B3 = Band.B3
    B4 = Band.B4
    B5 = Band.B5
    B6 = Band.B6
    B7 = Band.B7
    B8 = Band.B8
    B8A = Band.B8A
    B9 = Band.B9
    B10 = Band.B10
    B11 = Band.B11
    B12 = Band.B12


    # Enum class for Sentinel2 L2A masks
    class Mask(Enum):
        SAT = 'SAT'
        CLM = 'CLM'
        EDG = 'EDG'
        MG2 = 'MG2'

    # Aliases
    SAT = Mask.SAT
    CLM = Mask.CLM
    EDG = Mask.EDG
    MG2 = Mask.MG2
    
    # Enum class for mask and Atmos resolutions
    class Res(Enum):
        R1 = 'R1'
        R2 = 'R2'

    # Aliases for resolution
    R1 = Res.R1
    R2 = Res.R2
    
    # Atmosphere bands
    class Atmos(Enum):
        ATB = 'ATB'
    
    # Aliases for atmosphere bands 
    ATB = Atmos.ATB
  

    # Band groups
    GROUP_10M = [B2, B3, B4, B8]
    GROUP_20M = [B5, B6, B7, B8A, B11, B12]
    GROUP_60M = [B9, B10]
    ALL_MASKS = [SAT, CLM, EDG, MG2]
    ATMOS = [ATB]

    # Enum for BandType
    class BandType(Enum):
        FRE = 'FRE'
        SRE = 'SRE'
        
    # Aliases for band type
    FRE = BandType.FRE
    SRE = BandType.SRE

    # Detectors
    class Detector(Enum):
        D01 = 1
        D02 = 2
        D03 = 3
        D04 = 4
        D05 = 5
        D06 = 6
        D07 = 7
        D08 = 8
        D09 = 9
        D10 = 10
        D11 = 11
        D12 = 12
    
    # MTF
    MTF = {
        B2: 0.304,
        B3: 0.276,
        B4: 0.233,
        B5: 0.343,
        B6: 0.336,
        B7: 0.338,
        B8: 0.222,
        B8A: 0.325,
        B9: 0.39,
        B11: 0.21,
        B12: 0.19}

    # Resolution
    RES = {
        B2: 10,
        B3: 10,
        B4: 10,
        B5: 20,
        B6: 20,
        B7: 20,
        B8: 10,
        B8A: 20,
        B9: 60,
        B11: 60,
        B12: 60}

    def PSF(
        self,
        bands: List[Band],
        resolution: float = 0.5,
        half_kernel_width: int = None):
        """
        Generate PSF kernels from list of bands
    
        :param bands: A list of Sentinel2 Band Enum to generate PSF kernel for
        :param resolution: Resolution at which to sample the kernel
        :param half_kernel_width: The half size of the kernel
                                  (determined automatically if None)

        :return: The kernels as a Tensor of shape
                 [len(bands),2*half_kernel_width+1, 2*half_kernel_width+1]
        """
        return np.stack([(utils.generate_psf_kernel(resolution,
                                                    Sentinel2.RES[b],
                                                    Sentinel2.MTF[b],
                                                    half_kernel_width)) for b in bands])


    def build_detectors_masks_path(self):
        """
        Return a dictionnary of path to detector masks at both R1 and R2 resolution
        """
         # Sorted ensure masks come in the same order in both lists
        r1_masks = sorted(glob.glob(f"{self.product_dir}/MASKS/*DTF_R1-D*.tif"))
        r2_masks = sorted(glob.glob(f"{self.product_dir}/MASKS/*DTF_R2-D*.tif"))

        # Named tuple to store output
        DetectorMasks = namedtuple('DetectorMasks', 'r1 r2')

        output = {}

        # Collect all pairs of detectors
        for (r1, r2) in zip(r1_masks, r2_masks):
            detector_idx = self.Detector(int(r1[-6:-4]))
            output[detector_idx] = DetectorMasks(r1,r2)

        return output
        
    def build_xml_path(self) -> str:
        """
        Return path to root xml file
        """
        p = glob.glob(f"{self.product_dir}/*MTD_ALL.xml")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(f"Could not find root XML file in product directory {self.product_dir}")
        return p[0]
    
    def build_band_path(
        self,
        band: Band,
        band_type: BandType = FRE) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}/*{band_type.value}_{band.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(f"Could not find band {band.value} of type {band_type.value} in product directory {self.product_dir}")
        return p[0]

    def build_mask_path(
        self,
        mask: Mask,
        resolution: Res = R1) -> str:
        """
        Build path to a band for product
        :param band: The band to build path for as a Sentinel2.Band enum value
        :param prefix: The band prefix (FRE_ or SRE_)

        :return: The path to the band file
        """
        p = glob.glob(f"{self.product_dir}/MASKS/*{mask.value}_{resolution.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(f"Could not find mask {mask.value} of resolution {resolution.value} in product directory {self.product_dir}")
        return p[0]
    
    def build_atmos_path( ####################
        self,
        resolution: Res = R1) -> str:
        """
        Build path to a file containing WVC and AOT bands for product
        :param atmos: The band type to build path for Sentinel 2 atmosphere bands
        :param resolution: chosen resolution 

        :return: The path to the ATB file
        """
        p = glob.glob(f"{self.product_dir}/*ATB_{resolution.value}.tif")
        # Raise
        if len(p) == 0:
            raise FileNotFoundError(f"Could not find ATB of resolution {resolution.value} in product directory {self.product_dir}")
        return p[0]


    def read_as_numpy(self,
                      bands:List[Band],
                      band_type:BandType = FRE,
                      masks:List[Mask]=ALL_MASKS,
                      readAtmos:bool = False,
                      res:Res = Res.R1,
                      scale:float=10000,
                      crs: str=None,
                      resolution:float = 10,
                      region:Union[Tuple[int,int,int,int],rio.coords.BoundingBox]=None,
                      no_data_value:float=np.nan,
                      bounds:rio.coords.BoundingBox=None,
                      algorithm=rio.enums.Resampling.cubic,
                      dtype:np.dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Read bands from Sentinel2 products as a numpy ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param band_type: The band type (FRE or SRE)
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height],
                 The masks pixels as a np.ndarray of shape [masks, width, height],
                 The WVC band
                 The AOT band
                 The x coords as a np.ndarray of shape [width],
                 the y coords as a np.ndarray of shape [height],
                 the crs as a string
        """
        img_files = [self.build_band_path(b, band_type) for b in bands]
        np_arr, xcoords, ycoords, crs =  utils.read_as_numpy(img_files,
                                                             crs=crs,
                                                             resolution=resolution,
                                                             offsets=self.offsets,
                                                             region=region,
                                                             output_no_data_value = no_data_value,
                                                             input_no_data_value = -10000,
                                                             bounds = bounds,
                                                             algorithm = algorithm,
                                                             separate=True,
                                                             dtype = dtype,
                                                             scale = scale)

        # Skip first dimension
        np_arr = np_arr[0,...]

        # Read masks if needed
        np_arr_msk=None
        if len(masks)!=0:
            mask_files = [self.build_mask_path(m, res) for m in masks]
            np_arr_msk, _, _, _ =  utils.read_as_numpy(mask_files,
                                                    crs=crs,
                                                    resolution=resolution,
                                                    offsets=self.offsets,
                                                    region=region,
                                                    output_no_data_value = no_data_value,
                                                    input_no_data_value = -10000,
                                                    bounds = bounds,
                                                    algorithm = rio.enums.Resampling.nearest,
                                                    separate=True,
                                                    dtype = np.uint8,
                                                    scale = None)
            # Skip first dimension
            np_arr_msk = np_arr_msk[0,...]
        
        # Read atmosphere band
        np_arr_atm=None
        if readAtmos:
            atmos_file = [self.build_atmos_path(res)]
            np_arr_atm, _, _, _ = utils.read_as_numpy(atmos_file,
                                                    crs=crs,
                                                    resolution=resolution,
                                                    offsets=self.offsets,
                                                    region=region,
                                                    output_no_data_value = no_data_value,
                                                    input_no_data_value = -10000,
                                                    bounds = bounds,
                                                    algorithm = algorithm,
                                                    separate=True,
                                                    dtype = np.float,
                                                    scale = None)
            # Normalize
            np_arr_atm = np_arr_atm[:,0,...]
            np_arr_atm[1] = np_arr_atm[1]/200
        
        # Return plain numpy array
        return np_arr, np_arr_msk, np_arr_atm, xcoords, ycoords, crs

    def read_as_xarray(self,
                       bands:List[Band],
                       band_type:BandType = FRE,
                       masks:List[Mask]=ALL_MASKS,
                       res:Res = Res.R1,
                       scale:float=10000,
                       crs: str=None,
                       resolution:float = 10,
                       region:Union[Tuple[int,int,int,int],rio.coords.BoundingBox]=None,
                       no_data_value:float=np.nan,
                       bounds:rio.coords.BoundingBox=None,
                       algorithm=rio.enums.Resampling.cubic,
                       dtype:np.dtype=np.float32) -> xr.Dataset:
        """
        Read bands from Sentinel2 products as a numpy

        ndarray. Depending on the parameters, an internal WarpedVRT
        dataset might be used.

        :param bands: The list of bands to read
        :param band_type: The band type (FRE or SRE)
        :param scale: Scale factor applied to reflectances (r_s = r / scale). No scaling if set to None
        :param crs: Projection in which to read the image (will use WarpedVRT)
        :param resolution: Resolution of data. If different from the resolution of selected bands, will use WarpedVRT
        :param region: The region to read as a BoundingBox object or a list of pixel coords (xmin, ymin, xmax, ymax)
        :param no_data_value: How no-data will appear in output ndarray
        :param bounds: New bounds for datasets. If different from image bands, will use a WarpedVRT
        :param algorithm: The resampling algorithm to be used if WarpedVRT
        :param dtype: dtype of the output Tensor
        :return: The image pixels as a np.ndarray of shape [bands, width, height]
        """
        np_arr, np_arr_msk, xcoords, ycoords, crs = self.read_as_numpy(bands, band_type,
                                                                      masks, res,
                                                                      scale, crs,
                                                                      resolution, region,
                                                                      no_data_value, bounds,
                                                                      algorithm, dtype)    

        vars = {}
        for i in range(len(bands)):
            vars[bands[i].value]=(["t", "y", "x"] , np_arr[None,i,...])
            for i in range(len(masks)):
                vars[masks[i].value]=(["t", "y", "x"] , np_arr_msk[None,i,...])
            
            
        xarr = xr.Dataset(vars,
                          coords={'t' : [self.date], 'x' : xcoords, 'y':ycoords},
                          attrs= {'tile' : self.tile,
                                  'type' : band_type.value,
                                  'crs' : crs})
        return xarr

    #@profile
    def upsample_angular_grid(self, grid:np.ndarray, res:Res=Res.R1, order:int=1) -> np.ndarray:
        """
        upsample given angular grid at target resolution
        """
        if res == self.Res.R1:
            target_resolution=10.
        else:
            target_resolution=20.
        
        # Angular grids have a resolution of 5 km
        scale_factor=5000/target_resolution

        input_grid = np.array(grid)
                
        zoomed_grid = ndimage.zoom(input_grid, scale_factor, prefilter=False, order=1, mode='nearest', grid_mode=True)
        nb_pixels = int(10980*10/target_resolution)
        # We assume that angular center of first pixel correspond to center of first 10m pixel
        offset = int(scale_factor/2.)
        return zoomed_grid[offset:nb_pixels+offset, offset:nb_pixels+offset]

    #@profile
    def extrapolate_grid(self,grid):
        """
        """
        out_grid = np.copy(grid)
        x, y = np.indices(grid.shape)
        xvalid = x[~np.isnan(grid)]
        xvalid = x[~np.isnan(grid)]
        reg = LinearRegression().fit(np.stack((x[~np.isnan(grid)], y[~np.isnan(grid)]), axis=1),grid[~np.isnan(grid)])
        grid_filled =  reg.predict(np.stack((x.ravel(), y.ravel()), axis=1)).reshape(grid.shape)
        out_grid[np.isnan(grid)]=grid_filled[np.isnan(grid)]

        return out_grid
    
    #@profile
    def upsample_by_viewing_directions(self, zenith, azimuth, res:Res=Res.R1, order:int=1, extrapolate=False):
        """

        """
        # Copy input grids
        in_zenith=zenith
        in_azimuth=azimuth
        
        # Extrapolate nans if needed
        if extrapolate:
            in_zenith=self.extrapolate_grid(zenith)
            in_azimuth=self.extrapolate_grid(azimuth)
            
        # Use cartesian coordiantes to interpolate azimuth
        dx = np.tan(np.deg2rad(in_zenith))* np.sin(np.deg2rad(in_azimuth))
        dy = np.tan(np.deg2rad(in_zenith))* np.cos(np.deg2rad(in_azimuth))

        zoomed_dx = self.upsample_angular_grid(dx,res=res, order=order)
        zoomed_dy = self.upsample_angular_grid(dy,res=res, order=order)

        # General case
        zoomed_azimuth = np.arctan(zoomed_dx/zoomed_dy)
        zoomed_azimuth[zoomed_dy<0] += np.pi
        zoomed_azimuth[zoomed_azimuth<0] += 2*np.pi
        zoomed_zenith = np.arctan(zoomed_dy/np.cos(zoomed_azimuth))

        # dy == 0 but not dx
        mask =np.logical_and(zoomed_dy==0, zoomed_dx!=0)
        zoomed_azimuth[mask]=np.pi/(2*np.sign(zoomed_dx[mask]))
        zoomed_azimuth[np.logical_and(mask,zoomed_azimuth<0)]+=2*np.pi
        zoomed_zenith[mask]=np.arctan(np.abs(zoomed_dx[mask]))

        # dy == 0 and dxy == 0
        mask = np.logical_and(zoomed_dy==0, zoomed_dx==0)
        zoomed_azimuth[mask]=0.
        zoomed_zenith[mask]=0.

        # Final conversion to degrees
        zoomed_zenith=np.rad2deg(zoomed_zenith)
        zoomed_azimuth=np.rad2deg(zoomed_azimuth)
        
        return zoomed_zenith, zoomed_azimuth

    #@profile
    def read_solar_angles_as_numpy(self, res:Res = Res.R1, interpolation_order:int=1):
        """
        Return zenith and azimuth solar angle as a tuple fo 2 numpy arrays at requested resolution
        """
        # Ensure that xml is parsed
        if self.sun_angles is None:
            self.parse_xml()

        # Call up-sampling routine
        return self.upsample_by_viewing_directions(self.sun_angles.zenith, self.sun_angles.azimuth, res, order=interpolation_order)

    #@profile
    def read_incidence_angles_as_numpy(self, band:Band = Band.B2, res:Res = Res.R1, interpolation_order:int=1):
        """
        
        """
        # Ensure that xml is parsed
        if self.incidence_angles is None:
            self.parse_xml()

        band_angles = self.incidence_angles[band]

        # Get path to detector masks
        detector_masks = self.build_detectors_masks_path()

        # Derive output shape
        out_shape = (10980,10980)

        if res != self.Res.R1:
            out_shape=(5490,5490)

        # Init outputs
        odd_zenith_angles = np.full(out_shape,np.nan)
        odd_azimuth_angles = np.full(out_shape,np.nan)
        even_zenith_angles = np.full(out_shape,np.nan)
        even_azimuth_angles = np.full(out_shape,np.nan)

        # Loop on all detectors
        for det,angles in band_angles.items():

            if det in detector_masks:
                # Get path to detector mask
                current_detector_mask_path = detector_masks[det].r1
                if res != self.Res.R1:
                    current_detector_mask_path = detector_masks[det].r2

                # Read the mask
                with rio.open(current_detector_mask_path) as ds:
                    current_detector_mask = ds.read(1)

                zoomed_zenith, zoomed_azimuth = self.upsample_by_viewing_directions(angles.zenith, angles.azimuth, res=res, order=interpolation_order, extrapolate=True)

                # # Derive cartesian coordinates for azimyth
                # cartesian = angles.zenith * np.cos(angles.azimuth*np.pi/180.)

                # # Zoom zenith
                # zoomed_zenith = self.upsample_angular_grid(angles.zenith,res=res, order=interpolation_order, extrapolate=True)
                
                # # Zoom cartesian
                # zoomed_cartesian = self.upsample_angular_grid(cartesian,res=res, order=interpolation_order, extrapolate=True)
                
                # # Get back to azimuth
                # zoomed_azimuth = (180./np.pi)*np.arccos(zoomed_cartesian/zoomed_zenith)

                # Apply masking
                zoomed_zenith[current_detector_mask==0]=np.nan
                zoomed_azimuth[current_detector_mask==0]=np.nan

                # Build validy mask
                valid_zenith_mask = np.logical_not(np.isnan(zoomed_zenith))
                valid_azimuth_mask = np.logical_not(np.isnan(zoomed_azimuth))
                                               
                # Sort out detectors
                if det.value%2==1:
                    odd_zenith_angles[valid_zenith_mask] = zoomed_zenith[valid_zenith_mask]
                    odd_azimuth_angles[valid_azimuth_mask] = zoomed_azimuth[valid_azimuth_mask]
                else:
                    even_zenith_angles[valid_zenith_mask] = zoomed_zenith[valid_zenith_mask]
                    even_azimuth_angles[valid_azimuth_mask] = zoomed_azimuth[valid_azimuth_mask]

                # Clear
                del zoomed_zenith
                del zoomed_azimuth
                del valid_zenith_mask
                del valid_azimuth_mask
                
        return even_zenith_angles, odd_zenith_angles, even_azimuth_angles, odd_azimuth_angles
