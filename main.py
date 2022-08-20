import copy
import sys, os

import pandas as pd
import numpy as np
from classes.Genetic import GeneticSolver
from classes.Car import get_car_data, Car
from classes.Race import RaceData, plot_best
from classes.Utils import ms_to_time, time_to_ms
import plotly.express as px
import linecache
import os
import tracemalloc
import argparse

### To suppress plotly warnings
import warnings
warnings.filterwarnings('ignore')
###

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
    
def main():
    print(f"\n---------------------START----------------------\n")
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
        print(f"\n-------------------{circuit}--------------------\n")

        car:Car = get_car_data(circuit)

        #race_data:RaceData = RaceData(circuit)
        #race_data.plot(path=circuit)
        
        _circuit = circuit.split("\\")[-1] if os.name == 'nt' else circuit.split("/")[-1]

        genetic = GeneticSolver(population=50, mutation_pr=0.9, crossover_pr=0.4, iterations=500, car=car, circuit=_circuit)

        bruteforce_save_path = os.path.join(circuit, "Bruteforce_strategy.log")
        if os.path.isfile(bruteforce_save_path):
            print(f"Bruteforce results for {_circuit} are already calculated in '{bruteforce_save_path}'.\n")
        else:
            bruteforce_strategy = genetic.lower_bound()
            with open(bruteforce_save_path, "a") as f:
                laps = genetic.numLaps
                strategy, timing = bruteforce_strategy
                for lap in range(laps):
                    f.write(f"Lap {lap+1}/{laps} -> Compound: '{strategy[lap]['Compound']}', TyreAge: {strategy[lap]['TyreAge']} Laps, TyreWear: FL:{round(strategy[lap]['TyreWear']['FL']*100,1)}% FR:{round(strategy[lap]['TyreWear']['FR']*100,1)}% RL:{round(strategy[lap]['TyreWear']['RL']*100,1)} RR:{round(strategy[lap]['TyreWear']['RR']*100,1)}%, FuelLoad: {strategy[lap]['FuelLoad']} Kg, PitStop: {'Yes' if strategy[lap]['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy[lap]['LapTime'])} (hh:)mm:ss.ms\n")
                t = f"{int(timing):,}".replace(",", " ")
                f.write(f"\nFitness: {t}\n")
                f.write(f"Total time: {ms_to_time(timing)}")

        with open(bruteforce_save_path, 'r') as f:
            lines = f.readlines()
        
        bf_time = lines[-1].split(" ")
        bf_time_in_ms = time_to_ms(bf_time[-1])

        best, best_eval, boxplot_data, fitness_data = genetic.startSolver(bf_time = bf_time_in_ms) 
        
        print(f"\n------------------------------------------------\n")
        print(f"EA timing: {ms_to_time(best_eval)}")
        print(f"Bruteforce give timing: {bf_time[-1]}")
        print(f"\n------------------------------------------------\n")

        print(f"\n------------------------------------------------\n")
        ea = f"{int(best_eval):,}".replace(",", " ")
        bf = f"{int(bf_time_in_ms):,}".replace(",", " ")
        print(f"EA fitness: {ea}\nBruteforce fitness: {bf}")
        print(f"\n------------------------------------------------\n")


        # Plots
        fit_gen_boxplot = px.box(boxplot_data, title="Boxplot fitnesses of every generation")
        fit_gen_boxplot.update_layout(xaxis_title="Generation", yaxis_title="Fitness")
        fit_gen_boxplot.write_html(os.path.join(circuit, "Boxplot_fitnesses.html"))

        y_values = []
        minutes_worst = int(max(fitness_data["Fitness"])/1000)//60 - 59
        minutes_best = min(int(best_eval/1000), int(bf_time_in_ms/1000))//60 - 61

        for i in range(minutes_best, minutes_worst+2):
            y_values.append((i+60)*60*1000)
        
        fitness_data['LapTime'] = []
        for val in fitness_data['Fitness']:
            fitness_data['LapTime'].append(ms_to_time(val))

        fit_line = px.line(fitness_data, x="Generation", y="Fitness", text="LapTime", title=f"Line plot fitnesses for {_circuit}")#, color="Fitness")
        fit_line.add_hline(y=bf_time_in_ms, line_color="red", annotation_text=f"Bruteforce time -> {bf_time[-1]}", annotation_position="top left")
        
        fit_line.update_traces(textposition='top center')
        fit_line.update_layout(
            xaxis={
                'title': 'Generation',
                'range': [-0.25, fitness_data['Generation'][-1]+0.25],
            },
            yaxis={
                "tickmode": "array",
                "tickvals": y_values,
                "ticktext": [ms_to_time(x) for x in y_values],
                "range" : [y_values[0], y_values[-1]]
            },
        )
        
        fit_line.write_html(os.path.join(circuit, "Line_plot_fitnesses.html"))

        if input(f"\nDo you want to see the plots for {_circuit}? (Y/n) ").lower() == "y":
            print(f"Plotting for {_circuit}...")
            fit_gen_boxplot.show()
            fit_line.show()
        
    
    print(f"\n----------------------END-----------------------\n")
    return

def foo2(df):
    frames = [df]
    N = len(df)
    for i in range(N):
        frames += [pd.Series(df, index=df.index)]
    return pd.concat(frames, axis=1)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main()
    sys.exit(0)
