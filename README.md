# Bio-F1
Genetic Algorithms applied to F1 race strategy.

# Requirements
- tqdm
- numpy
- pandas
- plotly
- Python 

To create the environment use conda (better):
```python
conda create -n=<environment_name> python=3.9.7 pandas numpy tqdm plotly
```
or pip:
```python
python -m venv <environment_name> 
pip install -U pandas numpy tqdm plotly
```

Or you can use the requirements files:
```python
conda env create -f requirements.yml
```
```python
pip install -r requirements.txt
```

# Classes
## Extractor
This is intended to extract the data and return the various dataframe we are going to use. Notice that it could build a single dataframe containing *all* data. However it is very heavy computationally speaking (may take up to one hour and yes, it is not optimized (sorry Montresor)) and for this reason it is disabled.

## Utils
Contains some utility functions that can be useful for other classes. Moreover it contains some constants.

## Tyres
The class is intended to extract and analyze data about a stint (set) of tyre. There are two main classes: *Tyre* and *Tyres*, the first is used for a single tyre that can be FrontLeft, FrontRight, RearLeft or RearRight while the second is used to take the all four tyres in one single class.

## Fuel
The class is intended to extract and analyze data about fuel usage (it is based on the different stints approach). There is one main classes: *Fuel*.