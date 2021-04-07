# sensorsio

`sensorsio` is a python library provides convenient functions to load Sentinel2 and other sensors data into `numpy` and `xarray`.

Using it is as simple as:

```python
# Import the sentinel 2 module
from sensorsio import sentinel2
# Create an instance of Sentinel2 class from the product path
dataset = sentinel2.Sentinel2('/datalake/S2-L2A-THEIA/31TDH/2019/05/31/SENTINEL2B_20190531-105916-927_L2A_T31TDH_C_V2-2/')
# Read bands, masks and coords to numpy
bands, masks, xcoords, ycoords, crs = dataset.read_as_numpy(sentinel2.Sentinel2.GROUP_10M)
# Read bands, masks and coords to xarray
xrds = dataset.read_as_xarray(sentinel2.Sentinel2.GROUP_10M)
```

```sensorsio``` provides a lot of flexibility and allows a lot more.

For Sentinel2, it offers:
*  Convenient attributes like day of year or sensor id
*  Selective read of desired bands and masks
*  On-the-fly resampling of 20m bands to 10m while reading
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting
*  Supports registration offsets computed by StackReg

See [this notebook](https://gitlab.cnes.fr/cesbio/sensorsio/-/blob/master/notebooks/sentinel2.ipynb) for an in depth review of the capabilties with Sentinel2 class.

## Installation

TODO

## Notes

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
