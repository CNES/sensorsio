import numpy as np
import rasterio as rio
import src.sensorsio.srtm as srtm
from src.sensorsio import sentinel2


def test_srtm_id_to_name():
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(1, -2)) == "S02E001"
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(-1, -2)) == "S02W001"
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(-1, 12)) == "N12W001"


def test_crs_from_mgrs():
    assert srtm.crs_for_mgrs_tile('31TDH').to_authority() == ('EPSG', '32631')


def test_mgrs_transform():
    assert srtm.mgrs_transform('31TDH') == rio.Affine(10.0, 0.0, 399960.0, 0.0,
                                                      -10.0, 4800000.0)


def test_srtm_tiles_from_mgrs_tile():
    def build_tile_list(tile):
        return [
            srtm.srtm_id_to_name(tid)
            for tid in srtm.srtm_tiles_from_mgrs_tile(tile)
        ]

    assert build_tile_list("31TCJ") == [
        'N43E000', 'N43E001', 'N44E000', 'N44E001'
    ]

    assert build_tile_list("36TTM") == [
        'N41E029', 'N41E030', 'N42E029', 'N42E030'
    ]

    assert build_tile_list("35MMQ") == [
        'S06E026', 'S06E027', 'S05E026', 'S05E027'
    ]

    assert build_tile_list("19GEP") == [
        'S43W070', 'S43W069', 'S43W068', 'S42W070', 'S42W069', 'S42W068'
    ]


def test_generate_dem():
    dem_handler = srtm.SRTM()
    dem = dem_handler.get_dem_for_mgrs_tile("31TDH")
    with rio.open("/tmp/dem.tif",
                  'w',
                  driver='GTiff',
                  height=dem.elevation.shape[0],
                  width=dem.elevation.shape[1],
                  count=3,
                  nodata=-32768.0,
                  dtype=dem.elevation.dtype,
                  compress='lzw',
                  crs='+proj=latlong',
                  transform=dem.transform) as ds:
        ds.write(dem.elevation, 1)
        ds.write(dem.slope, 2)
        ds.write(dem.aspect, 3)


def test_dem_on_s2_tile():
    from rasterio.warp import Resampling, reproject
    RESOLUTION = 10
    s2_ds = sentinel2.Sentinel2(
        '/datalake/S2-L2A-THEIA/31TDH/2019/05/31/'
        'SENTINEL2B_20190531-105916-927_L2A_T31TDH_C_V2-2/')
    xrds = s2_ds.read_as_xarray([sentinel2.Sentinel2.B12],
                                resolution=RESOLUTION)

    dst_dem = np.zeros((3, xrds.sizes['x'], xrds.sizes['y']))
    dem_handler = srtm.SRTM()
    s2_dem = dem_handler.get_dem_for_mgrs_tile(s2_ds.tile)
    dst_dem, dst_dem_transform = reproject(s2_dem.as_stack(),
                                           destination=dst_dem,
                                           src_transform=s2_dem.transform,
                                           src_crs=s2_dem.crs,
                                           dst_transform=s2_ds.transform,
                                           dst_crs=s2_ds.crs,
                                           resampling=Resampling.cubic)
    s2_dem = srtm.DEM(dst_dem[0, :, :].astype(np.int16),
                      dst_dem[1, :, :].astype(np.int16),
                      dst_dem[2, :, :].astype(np.int16), s2_ds.crs,
                      dst_dem_transform)

    with rio.open("/tmp/s2_dem.tif",
                  'w',
                  driver='GTiff',
                  height=s2_dem.elevation.shape[0],
                  width=s2_dem.elevation.shape[1],
                  count=3,
                  nodata=-32768.0,
                  dtype=s2_dem.elevation.dtype,
                  compress='lzw',
                  crs=s2_dem.crs,
                  transform=s2_dem.transform) as ds:
        ds.write(s2_dem.elevation, 1)
        ds.write(s2_dem.slope, 2)
        ds.write(s2_dem.aspect, 3)


def test_dem_on_mgrs_tile():
    from rasterio.warp import Resampling, reproject
    TILE = '31TDH'

    mgrs_transform = srtm.mgrs_transform(TILE)
    mgrs_crs = srtm.crs_for_mgrs_tile(TILE)
    dst_dem = np.zeros((3, 10980, 10980))
    dem_handler = srtm.SRTM()
    s2_dem = dem_handler.get_dem_for_mgrs_tile(TILE)
    dst_dem, dst_dem_transform = reproject(s2_dem.as_stack(),
                                           destination=dst_dem,
                                           src_transform=s2_dem.transform,
                                           src_crs=s2_dem.crs,
                                           dst_transform=mgrs_transform,
                                           dst_crs=mgrs_crs,
                                           resampling=Resampling.cubic)
    s2_dem = srtm.DEM(dst_dem[0, :, :].astype(np.int16),
                      dst_dem[1, :, :].astype(np.int16),
                      dst_dem[2, :, :].astype(np.int16), mgrs_crs,
                      dst_dem_transform)

    with rio.open("/tmp/mgrs_dem.tif",
                  'w',
                  driver='GTiff',
                  height=s2_dem.elevation.shape[0],
                  width=s2_dem.elevation.shape[1],
                  count=3,
                  nodata=-32768.0,
                  dtype=s2_dem.elevation.dtype,
                  compress='lzw',
                  crs=s2_dem.crs,
                  transform=s2_dem.transform) as ds:
        ds.write(s2_dem.elevation, 1)
        ds.write(s2_dem.slope, 2)
        ds.write(s2_dem.aspect, 3)
