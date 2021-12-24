import pytest
import src.sensorsio.srtm as srtm


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
