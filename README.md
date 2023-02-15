# PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation

This is an official implementation of PriSTI (ICDE 2023). We provided the codes about the experiments on air quality dataset AQI-36 and traffic speed datasets METR-LA and PEMS-BAY.


## Requirement

See `requirements.txt` for the list of packages.

## Dataset

All the datasets can be used in the experiments. The data of AQI-36 is in `./data/pm25/`. The data of METR-LA and PEMS-BAY can be downloaded from this [link](https://mega.nz/folder/Ei4SBRYD#ZjOinn0CzFPkiE_V9yVhJw).
The downloaded datasets are suggested to be stored in the `./data/`.

## Experiments

To train PriSTI on different datasets, you can run the scripts `exe_{dataset_name}.py` such as:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16
python exe_metrla.py --device 'cuda:0' --num_workers 16
python exe_pemsbay.py --device 'cuda:0' --num_workers 16
```

or directly using our provided pretrained model for imputation:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16 --modelfolders 'aqi36'
python exe_metrla.py --device 'cuda:0' --num_workers 16 --modelfolders 'metr_la'
python exe_pemsbay.py --device 'cuda:0' --num_workers 16 --modelfolders 'pems_bay'
```



