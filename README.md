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
print(xrds)
```
```
<xarray.Dataset>
Dimensions:  (t: 1, x: 10980, y: 10980)
Coordinates:
  * t        (t) datetime64[ns] 2019-05-31
  * x        (x) float64 4e+05 4e+05 4e+05 ... 5.097e+05 5.097e+05 5.098e+05
  * y        (y) float64 4.8e+06 4.8e+06 4.8e+06 ... 4.69e+06 4.69e+06 4.69e+06
Data variables:
    B2       (t, y, x) float32 0.0364 0.0378 0.0406 0.0393 ... nan nan nan nan
    SAT      (t, y, x) uint8 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0
    CLM      (t, y, x) uint8 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0
    EDG      (t, y, x) uint8 0 0 0 0 0 0 0 0 0 0 0 0 ... 1 1 1 1 1 1 1 1 1 1 1 1
    MG2      (t, y, x) uint8 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0
    B3       (t, y, x) float32 0.0652 0.068 0.0742 0.0752 ... nan nan nan nan
    B4       (t, y, x) float32 0.063 0.0712 0.0735 0.0728 ... nan nan nan nan
    B8       (t, y, x) float32 0.2343 0.2349 0.2369 0.2456 ... nan nan nan nan
Attributes:
    tile:     31TDH
    type:     FRE
    crs:      EPSG:32631
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
