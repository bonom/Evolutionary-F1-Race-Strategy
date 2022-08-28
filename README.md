# Bio-F1
>Formula 1 is a race competition that in the past years has been evolving year after year, becoming more and more complicated than ever. We develop an evolutionary algorithm that can simulate a race given the free practices data and find an optimal strategy for the race in a specific circuit. We are going to consider the weather and all the possible time lose from different aspects (different tyres, pit-stop, fuel weight, tyre wear, etc.) given by the simulations in F1 2021.


-----------------------------------------------------------------------------------------------------------------------

# Requirements

Needed packages to run the project:
- tqdm
- numpy
- pandas
- plotly
- Python 3.9

To create the environment use conda (better):
```python
conda create -n <environment_name> python=3.9.12 pandas numpy tqdm plotly
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

-----------------------------------------------------------------------------------------------------------------------

# Classes
## Car
A class where there is the elaboration of all the data regarding a car and store them in a *json* file.

## Genetic
This class is where the Evolutionary Algorithm and the Bruteforce are implemented.

## Race
This class checks if there are Race data to view in order to understand how the race has performed in the simulation.

## Utils
A class that contains the most general functions used in all the other scripts.

## Weather
A class where the weather is managed in order to pass it to the *Genetic* one.

-----------------------------------------------------------------------------------------------------------------------

# Usage

The script can be used with several flags in different ways, it starts with the default one:
```python
python main.py
```

Then flags are:
- --d: is a modality for retrieving more runs of the simulation
```python
python main.py --d
```
- --c <circuit>: better if specified, otherwise the script will run the script for all the circuits available. The circuits available are in the folders in the Data folder
```python
python main.py --c Monza
```
- --pop <int>: is the population size of the genetic algorithm
```python
python main.py --pop 100
```
- --mut <float>: is the mutation rate of the genetic algorithm
```python
python main.py --mut 0.1
```
- --cross <float>: is the crossover rate of the genetic algorithm
```python
python main.py --cross 0.1
```
- --i <int>: is the number of iterations of the genetic algorithm
```python
python main.py --i 100
```
- --w <weather file>: is the weather file to use in the simulation, if not specified the script will ask for the weather file to use. Notice that only the weather *txt*s in every circuit data folder are considered
```python
python main.py --w Sunny.txt
```

It is possible to use more than one flag, for instance:
```python
python main.py --c Monza --pop 100 --mut 0.1 --cross 0.1 --i 100 --w Sunny.txt --d
```
