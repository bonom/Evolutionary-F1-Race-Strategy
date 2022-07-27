import sys, os
from classes.Genetic import GeneticSolver
from classes.Car import get_car_data, Car
from classes.Race import RaceData, plot_best
from classes.Utils import ms_to_time
import plotly.express as px

import argparse

parser = argparse.ArgumentParser(description='Process F1 Data.')
parser.add_argument('--c', type=str, default=None, help='Circuit path')
args = parser.parse_args()

def printStrategy(strategy):
    for lap in range(len(strategy['TyreCompound'])):
        print(f"Lap {lap+1} -> Compound '{strategy['TyreCompound'][lap]}', Wear '{round(strategy['TyreWear'][lap]['FL'],2)}'% | '{round(strategy['TyreWear'][lap]['FR'],2)}'% | '{round(strategy['TyreWear'][lap]['RL'],2)}'% | '{round(strategy['TyreWear'][lap]['RR'],2)}'%, Fuel '{round(strategy['FuelLoad'][lap],2)}' Kg, PitStop '{'Yes' if strategy['PitStop'][lap] else 'No'}', Time '{ms_to_time(strategy['LapTime'][lap])}' ms")

def main():
    if args.c is None:
        circuits = [os.path.abspath(os.path.join('Data', path)) for path in os.listdir(os.path.abspath('Data'))]
        if '.DS_Store' in circuits:
            circuits.remove('.DS_Store')
    else:
        if 'Data' in args.c.split('/'):#.split("\\"):
            path = os.path.abspath(args.c)
        else:
            path = os.path.abspath(os.path.join('Data', args.c))
        if os.path.isdir(path):
            circuits = [path]
        else:
            circuits = []
            print(f"Invalid circuit path: {path}")
    
    for circuit in circuits:
        #plot_best([0],circuit)
        car:Car = get_car_data(circuit)
        #race_data:RaceData = RaceData(circuit)
        #race_data.plot(path=circuit)
        
        _circuit = circuit.split("\\")[-1] if os.name == 'nt' else circuit.split("/")[-1]

        genetic = GeneticSolver(car=car, population=300, iterations=1000,circuit=_circuit)
        #bruteforce_strategy = genetic.lower_bound()
        #printStrategy(bruteforce_strategy)

        best, best_eval, fitness_values = genetic.startSolver()

        print(f"Best strategy:")
        printStrategy(best)
        print(f"Best strategy fitness: {ms_to_time(best_eval)}")
        (px.line(x=list(range(len(fitness_values))), y=fitness_values, title=f"Fitness of the best strategy for {_circuit}",)).show()
        
        #genetic.lower_bound()
        
if __name__ == "__main__":
    main()
    sys.exit(0)