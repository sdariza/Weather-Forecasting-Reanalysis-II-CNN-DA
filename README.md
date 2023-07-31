# Weather-Forecasting-Reanalysis-II-CNN-DA

## Create and activate conda environment

```
conda create -n wf python=3.9
conda deactivate
conda activate wf
```

## Install main weather dependencies

```
conda install -c conda-forge iris
pip install basemap
```

## Download .nc data and create netCDF data:

```
python download-data.py <start_date> <end_date> #YYYY
```

## Install tensorflow with GPU
[Tensorflow - Guide](https://www.tensorflow.org)
