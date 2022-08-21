import sys, os
import pandas as pd
from datetime import datetime
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
parser.add_argument('--pop', type=int, default=100, help='Population')
parser.add_argument('--mu', type=float, default=0.9, help='Mutation probability value')
parser.add_argument('--cross', type=float, default=0.5, help='Crossover probability value')
parser.add_argument('--i', type=int, default=1000, help='Iterations')
parser.add_argument('--w', type=str, default=None, help='Weather file')
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
    
def main(population:int, mutation_pr:float, crossover_pr:float, iterations:int, weather:str, base_path:str):
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
        _circuit = circuit.split("\\")[-1] if os.name == 'nt' else circuit.split("/")[-1]
        save_path = os.path.join(base_path, _circuit, datetime.now().strftime("%Y_%m_%d %H_%M_%S"))

        while not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        car:Car = get_car_data(circuit)

        #race_data:RaceData = RaceData(circuit)
        #race_data.plot(path=circuit)
        
        

        print(f"\n-------------------{_circuit}--------------------\n")

        genetic = GeneticSolver(population=population, mutation_pr=mutation_pr, crossover_pr=crossover_pr, iterations=iterations, car=car, circuit=_circuit, save_path=save_path, weather=weather)

        bruteforce_save_path = os.path.join(circuit, "Bruteforce_strategy.log")
        if not os.path.isfile(bruteforce_save_path):
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

        print(f"Lower bound: {ms_to_time(bf_time_in_ms)}\n")

        best, best_eval, boxplot_data, fitness_data = genetic.run(bf_time = bf_time_in_ms) 
        
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
        fit_gen_boxplot.write_html(os.path.join(save_path, "Boxplot_fitnesses.html"))

        y_values = []
        minutes_worst = int(max(fitness_data["Fitness"])/1000)//60 - 59
        minutes_best = min(int(best_eval/1000), int(bf_time_in_ms/1000))//60 - 61

        for i in range(minutes_best, minutes_worst+2):
            y_values.append((i+60)*60*1000)
        
        fitness_data['LapTime'] = []
        for val in fitness_data['Fitness']:
            fitness_data['LapTime'].append(ms_to_time(val))

        fit_line = px.line(fitness_data, x="Generation", y="Fitness", title=f"Line plot fitnesses for {_circuit}")#, color="Fitness")
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
        
        fit_line.write_html(os.path.join(save_path, "Line_plot_fitnesses.html"))

        if input(f"\nDo you want to see the plots for {_circuit}? (Y/n) ").lower() == "y":
            print(f"Plotting for {_circuit}...")
            fit_gen_boxplot.show()
            fit_line.show()
        
    
    print(f"\n----------------------END-----------------------\n")
    return

if __name__ == "__main__":    
    os.system('cls' if os.name == 'nt' else 'clear')
    main(population=args.pop, mutation_pr=args.mu, crossover_pr=args.cross, iterations=args.i, weather=args.w, base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs'))
    sys.exit(0)
