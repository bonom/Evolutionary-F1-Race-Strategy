import os
import sys 
import logging
import argparse

from typing import List
from datetime import datetime

from classes.Car import Car
from classes.Genetic import GeneticSolver
from classes.Utils import ms_to_time, get_basic_logger, CIRCUIT

MAX_RUNS = 30

parser = argparse.ArgumentParser(description='Parser to get the F1 data and other parameters.')
parser.add_argument('--c', type=str, default=None, help='Circuit path')
parser.add_argument('--pop', type=int, default=250, help='Population')
parser.add_argument('--mut', type=float, default=0.9, help='Mutation probability value')
parser.add_argument('--cross', type=float, default=0., help='Crossover probability value')
parser.add_argument('--i', type=int, default=1000, help='Iterations')
parser.add_argument('--d', action='store_true', default=False, help='Data Collection mode. Default is to collect a single run data.')
args = parser.parse_args()

logger = get_basic_logger(name="main", level=logging.INFO)
    
def main(
        population:int, 
        mutation_pr:float, 
        crossover_pr:float, 
        iterations:int, 
        circuits:List[str],
        base_path:str,
        index: int = -1,
    ) -> None:

    # In the case the c
    if args.c is None:
        circuits = list(CIRCUIT.keys())
    else:
        circuits = [args.c]
        
    
    for circuit in circuits:
        save_path = os.path.join(base_path, circuit, datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))

        # Check if folder exists, if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Create log file and write headers if the file does not exists
        csv_path = os.path.join(os.path.dirname(save_path), "log.csv")
        if not os.path.isfile(csv_path):
            with open(csv_path, "w") as f:
                f.write("Run,Population,Iterations,Mutation,Crossover,PitBest,EA_Fitness_1Pit,EA_Fitness_2Pit,EA_Fitness_3Pit,EA_Timing_1Pit,EA_Timing_2Pit,EA_Timing_3Pit,Timer,Save_Path\n")

        # Init the genetic solver
        genetic = GeneticSolver(population=population, mutation_pr=mutation_pr, crossover_pr=crossover_pr, iterations=iterations, car=Car(circuit), circuit=circuit, save_path=save_path,)
        best, best_idx, timer = genetic.run() 

        with open(csv_path, "a") as f:
            f.write(f"{index+1},{population},{iterations},{mutation_pr},{crossover_pr},{best_idx},{best[1]['TotalTime']},{best[2]['TotalTime']},{best[3]['TotalTime']},{ms_to_time(best[1]['TotalTime'])},{ms_to_time(best[2]['TotalTime'])},{ms_to_time(best[3]['TotalTime'])},{ms_to_time(timer)},{save_path}\n")

    return None #best, timer, save_path

if __name__ == "__main__":  
    population:int = args.pop
    iterations:int = args.i
    mutation_pr:float = args.mut
    crossover_pr:float = args.cross
    circuits:str = args.c

    # Preliminary checks
    if circuits is None:
        logger.warning(f"You did not specify a circuit. The script will automatically load ALL circuits available in the `DATA.json` file.")
    
    if mutation_pr > 1.0 or mutation_pr < 0.0:
        raise ValueError("Mutation probability value must be between 0 and 1. Inserted '{}'.".format(mutation_pr))
    
    if crossover_pr  > 1.0 or crossover_pr < 0.0:
        raise ValueError("Crossover probability value must be between 0 and 1. Inserted '{}'.".format(crossover_pr))
    

    # The --d flag enables data collection mode, this mode is designed to perform multiple runs with the same parameters and save the results in a single file.
    # To use this mode just use `python main.py <arguments> --d`
    if args.d:
        for i in range(0,MAX_RUNS):
            main(
                population=population,
                mutation_pr=mutation_pr, 
                crossover_pr=crossover_pr, 
                iterations=iterations, 
                circuits=circuits,
                base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs'),
                index=i,
            )
    else:
        main(
            population=population, 
            mutation_pr=mutation_pr, 
            crossover_pr=crossover_pr, 
            iterations=iterations, 
            circuits=circuits,
            base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs'),
        )
    sys.exit(0)
