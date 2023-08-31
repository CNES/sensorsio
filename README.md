# sensorsio

`sensorsio` is a python library that provides convenient functions to load Sentinel2 (Level 2A, MAJA format) and other sensors data into `numpy` and `xarray`. It supports on the fly reprojection to a user-defined grid, which makes it suitable for the building multi-modal datacubes.

# Licence

sensorsio is distributed under the Apache 2.0 licence, except from modules depending on pyresample, which are distributed under the LGPL v3 licence (irregulargrid.py, master.py, ecostress.py).

## Quickstart

### Reading a Sentinel2 product

Reading a **Sentinel2 L2A (MAJA format)** product is as simple as:

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
### Jointly reading multiple sensors on a common grid

It is also very simple to reproject several images from different sensors to a common grid for manipulation:

```python
# Create a Sentinel2 dataset
s2_ds = sentinel2.Sentinel2(s2)
# Create a Pleiades dataset
vns_ds = venus.Venus(vns)
# Find common grid
box, crs = utils.bb_common([s2_ds.bounds, vns_ds.bounds],[s2_ds.crs, vns_ds.crs],snap=10)

# Warp Sentinel2 on this grid:
s2_arr, _, _, _, _ = s2_ds.read_as_numpy(sentinel2.Sentinel2.GROUP_10M,
                                         resolution=10,
                                         crs=crs,
                                         bounds=box)
# Warp Pléiades on this grid:
vns_arr, _, _, _ = phr_ds.read_as_numpy(venus.Venus.GROUP_5M,
                                        resolution=10,
                                        crs=crs,
                                        bounds=box,
                                        algorithm=rio.enums.Resampling.cubic)
print(vns_arr.shape, s2_arr.shape)
```
```
((4, 5774, 2082), (4, 5774, 2082))
```

### Visualisation made easy

`sensorsio` also contains utilities to prepare `numpy` arrays for visualisation with `matplotlib`:

```python
# Prepare for rendering (band extraction and scaling)
dmin = np.array([0., 0., 0.])
dmax = np.array([0.2,0.2,0.2])
s2_rgb, dmin, dmax = utils.rgb_render(s2_arr, bands=[2,1,0], 
                                      dmin=dmin, 
                                      dmax=dmax)
phr_rgb, dmin, max = utils.rgb_render(phr_arr, bands=[2,1,0], 
                                      dmin=dmin, 
                                      dmax=dmax)
 # Call matplotlib
 fig, axes = plt.subplots(ncols=2, figsize=(int(25*phr_rgb.shape[1]/phr_rgb.shape[0]), 25))
axes[0].imshow(s2_rgb)
axes[0].set_title(str(s2_ds.satellite.value))
axes[1].imshow(phr_rgb)
axes[1].set_title(str(phr_ds.satellite.value))
fig.show()
 ```
 
 
## Available drivers
### Sentinel2 L2A (MAJA format)

For **Sentinel2 L2A** products from [Theia](https://theia.cnes.fr/atdistrib/rocket/#/home), it offers:
*  Convenient attributes like day of year or sensor id
*  Selective read of desired bands and masks
*  On-the-fly resampling of 20m bands to 10m while reading
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting
*  Supports registration offsets computed by StackReg
*  Access to recomposed solar and view angles

See [this notebook](notebooks/sentinel2.ipynb) for an in depth review of the capabilties with ```Sentinel2``` class.

See [this notebook](notebooks/sentinel2_angles.ipynb) for the access to solar and view angles.

### Venus L2A (MAJA format)

For **Venus L2A** products from [Theia](https://theia.cnes.fr/atdistrib/rocket/#/home), it offers:
*  Convenient attributes like day of year
*  Selective read of desired bands and masks
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting

See [this notebook](notebooks/venus.ipynb) for an in depth review of the capabilties with ```Venus``` class.

### Landsat8, Collection 2, Level 2 format

For *Landsat-8* products that can be downloaded from [EarthExplorer](https://earthexplorer.usgs.gov/), it offers:
*  Convenient attributes like day of year
*  Selective read of desired bands and masks
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting

### ECOSTRESS, Collection 1

For *Ecostress* products from Collection 1 that can be downloaded from [NASA LP DAAC](https://e4ftl01.cr.usgs.gov/ECOSTRESS/), you will need the LST file (```ECO2LST*```), the geom file (```ECO1BGEO*```), and optionally the cloud mask file (```ECO2CLD*```) and the rad file (```ECO1BRAD*```) for a given acquisition. The driver supports:
*  Convenient attributes like day of year
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting

### MASTER L1B and L2 products

For *MASTER* products from both L1B and L2 format that can be downloaded from [Master website](https://masterprojects.jpl.nasa.gov/Data_Products), you will require both the L1B and L2 products of a given track. It supports:
*  Convenient attributes like day of year
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting

### SRTM

For *SRTM*, it supports the *hgt* format, and from a directory containing SRTM tiles, it supports:
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting
* On-the-fly computation of Slope and Aspect

### WorldClim

For *Worldclim* data that can be downloaded from the [Worldclim website](https://worldclim.org/data/index.html), it supports:
*  Subsetting of variables
*  On-the-fly projection to a different Coordinates Reference System while reading
*  Image and geographical spatial subsetting

### Generic geotiff reading

The [regulargrid.py](src/sensorsio/regulargrid.py) offers a generic ```read_as_numpy()``` function that servers in most drivers and can be use to quickly stack, subsample and reproject any GeoTIFF file (or other Gdal supported image formats):

```python
def read_as_numpy(img_files: List[str],
                  crs: Optional[str] = None,
                  resolution: float = 10,
                  offsets: Optional[Tuple[float, float]] = None,
                  input_no_data_value: Optional[float] = None,
                  output_no_data_value: float = np.nan,
                  bounds: Optional[rio.coords.BoundingBox] = None,
                  algorithm=rio.enums.Resampling.cubic,
                  separate: bool = False,
                  dtype=np.float32,
                  scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
```

## Installation

Pass the path to cloned repository to ```pip install```:
```bash
$ pip install sensorsio
```

## Contributing

### Code quality

There is a Makefile at the root of the project that allows to run the following checkers:
* ```mypy```
* ```yapf```
* ```isort```
* ```ruff```
* ```pylint```

### Automated testing

* There are tests covering (almost) all modules. Some tests require test data that can be made available upon request due to their size. Those tests are flagged with ```@pytest.mark.requires_test_data```. Uncompress the test data to a folder and set the ```SENSORSIO_TEST_DATA_PATH``` environment variable before running the tests with ```pytest```

## Notes

This project has been set up using PyScaffold 4.0.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
