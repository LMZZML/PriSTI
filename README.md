# PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation

This is the code for our proposed framework GSTI. 
We only provided a part of code about the experiments on AQI-36.
The complete code will be released after the paper is published.

## Requirement

See `requirements.txt` for the list of packages.

## Dataset

The data of AQI-36 is in `./data/pm25/`.

## Experiments

An example of training GSTI on AQI-36 is:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16
```

or directly using our provided pretrained model for imputation:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16 --modelfolders 'aqi36'
```
