import pytest
import src.sensorsio.srtm as srtm
import rasterio as rio


def test_srtm_id_to_name():
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(1, -2)) == "S02E001"
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(-1, -2)) == "S02W001"
    assert srtm.srtm_id_to_name(srtm.SRTMTileId(-1, 12)) == "N12W001"


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
    dem = dem_handler.get_dem_for_mgrs_tile("31TCJ")
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
