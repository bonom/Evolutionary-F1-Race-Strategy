from genericpath import isfile
import sys, os
from classes.Genetic import GeneticSolver
from classes.Car import get_car_data, Car
from classes.Race import RaceData, plot_best
from classes.Utils import CIRCUIT, ms_to_time
import plotly.express as px
import linecache
import os
import tracemalloc
import argparse

#
# For tracing RAM usage:
# tracemalloc.start()
# [...] CODE [...]
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
#

parser = argparse.ArgumentParser(description='Process F1 Data.')
parser.add_argument('--c', type=str, default=None, help='Circuit path')
args = parser.parse_args()

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

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
        car:Car = get_car_data(circuit)
        #race_data:RaceData = RaceData(circuit)
        #race_data.plot(path=circuit)
        
        _circuit = circuit.split("\\")[-1] if os.name == 'nt' else circuit.split("/")[-1]

        genetic = GeneticSolver(car=car, population=100, iterations=1000,circuit=_circuit)
        #genetic.fixed_strategy()
        #return
        bruteforce_save_path = os.path.join(circuit, "Bruteforce_strategy.txt")
        if os.path.isfile(bruteforce_save_path):
            print(f"Loading previous results for {_circuit}:")
            with open(bruteforce_save_path, "r") as f:
                for line in f:
                    print(line, end="")
        else:
            bruteforce_strategy = genetic.lower_bound()
            with open(bruteforce_save_path, "a") as f:
                laps = genetic.numLaps-1
                strategy, timing = bruteforce_strategy
                for lap in range(laps):
                    f.write(f"Lap {lap+1}/{laps} -> Compound: '{strategy[lap]['Compound']}', TyresAge: {strategy[lap]['TyresAge']} Laps, TyresWear: {strategy[lap]['TyresWear']}, FuelLoad: {strategy[lap]['FuelLoad']} Kg, PitStop: {'Yes' if strategy[lap]['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy[lap]['LapTime'])} (hh:)mm:ss.ms\n")
                f.write(f"Total time: {ms_to_time(timing)}")

        best, best_eval, fitness_values = genetic.startSolver()

        printStrategy(best)
        print(f"Best strategy fitness: {ms_to_time(best_eval)}")
        (px.line(x=list(range(len(fitness_values))), y=fitness_values, title=f"Fitness of the best strategy for {_circuit}",)).show()
        
if __name__ == "__main__":
    main()
    sys.exit(0)

