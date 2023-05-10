<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">Evolutionary F1 Race Strategy [GECCO 2023]</h1>

  <h3 align="center">
    Official implementation of GECCO 2023 paper "Evolutionary F1 Race Strategy", it will be presented at the 8th Workshop on Industrial Applications of Metaheuristic (IAM 2023).
  </h3>

  ---
  <p align="center"> :fire: Follow the git to be notified when the paper will be available :fire:
  </p>
</div>

<!-- ----------------------------------------------------------------------------------------------------------------------- -->
<div align="center">
  <a href="https://scholar.google.com/citations?user=gnlDPxUAAAAJ">Andrea Bonomi</a>
  •
  <a href="https://www.w3schools.com/">Evelyn Turri</a>
  •
  <a href="https://scholar.google.it/citations?user=qSw6YfcAAAAJ&hl=en">Giovanni Iacca</a>
</div>
<!-- [Andrea Bonomi](#Introduction)
•
[Overview](#Overview)
•
[Project](#Project)
•
[Code](#Code)
# Authors
Authors: 
        [Andrea Bonomi](https://scholar.google.com/citations?user=gnlDPxUAAAAJ),
        [Evelyn Turri](),
        [Giovanni Iacca](https://scholar.google.it/citations?user=qSw6YfcAAAAJ&hl=en) -->

-----------------------------------------------------------------------------------------------------------------------
# Introduction
  Formula 1 is a highly competitive and ever-evolving sport, with teams constantly searching for ways to gain an edge over the competition. In order to meet this challenge, we propose a custom Genetic Algorithm that can simulate a race strategy given data from free practices and compute an optimal strategy for a specific circuit. The algorithm takes into account a variety of factors that can affect race performance, including weather conditions as well as tire choice, pit-stops, fuel weight, and tire wear. By simulating and computing multiple race strategies, the algorithm provides valuable insights and can help make informed strategic decisions, in order to optimize the performance on the track. The algorithm has been evaluated on both a video-game simulation and with real data on tire consumption provided by the tire manufacturer Pirelli. With the help of the race strategy engineers from Pirelli, we have been able to prove the real applicability of the proposed algorithm.


-----------------------------------------------------------------------------------------------------------------------

# Requirements

Needed packages to run the project:
- Python 3.11
- numpy
- pandas
- plotly
- tqdm

To create the environment use conda (better):
```bash
conda create -n <environment_name> python=3.11 pandas numpy tqdm plotly
```
or pip:
```bash
python -m venv <environment_name> 
pip install -U pandas numpy tqdm plotly
```

Or you can use the requirements file (inside the environment):
```bash
pip install -r requirements.txt
```

There should not be any problems with the packages versions, in case you have some troubles please use the `requirements.txt` file for the versions we used.

-----------------------------------------------------------------------------------------------------------------------

# Usage

The script can be used with several flags in different ways, it starts with the default one:
```bash
python main.py
```

Then flags are:
- `--d`: is a modality for retrieving more runs of the simulation
```bash
python main.py --d
```
- `--c <circuit>`: better if specified, otherwise the script will run the script for all the circuits available. The circuits available are the folders names in the Data folder
```bash
python main.py --c Monza
```
- `--pop <int>`: is the population size of the genetic algorithm
```bash
python main.py --pop 100
```
- `--mut <float>`: is the mutation rate of the genetic algorithm
```bash
python main.py --mut 0.1
```
- `--cross <float>`: is the crossover rate of the genetic algorithm
```bash
python main.py --cross 0.1
```
- `--i <int>`: is the number of iterations of the genetic algorithm
```bash
python main.py --i 100
```
- `--w <weather file>`: is the weather file to use in the simulation, if not specified the script will ask for the weather file to use. Notice that only the weather *txt*s in every circuit data folder are considered
```bash
python main.py --w Sunny.txt
```

It is possible to use more than one flag, for example:
```bash
python main.py --c Monza --pop 100 --mut 0.1 --cross 0.1 --i 100 --w Sunny.txt --d
```
<!-- -----------------------------------------------------------------------------------------------------------------------
# Citing our work
Please cite the following paper if you use our code:
```latex
@inproceedings{EAF1Strategy,
  title = {Evolutionary F1 Race Strategy},
  author = {Bonomi, Andrea and Turri, Evelyn and Iacca, Giovanni},
  booktitle = {Genetic and Evolutionary Computation Conference Companion (GECCO ’23 Companion)},
  year = {July 2023}
}
``` -->

-----------------------------------------------------------------------------------------------------------------------
# Acknowledgements
We would like to thank Simone Berra and Fernando Osuna from Pirelli for providing us the real data and giving us feedback on the numerical results.
