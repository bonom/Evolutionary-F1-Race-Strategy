import math
import sys, os
import numpy as np
from datetime import datetime
from classes.Genetic import GeneticSolver
from classes.Car import get_car_data, Car
from classes.Race import RaceData, plot_best
from classes.Utils import ms_to_time, time_to_ms
from classes.LocalSearch import LocalSearch
import plotly.express as px
import argparse

from classes.Weather import weather_summary

### To suppress plotly warnings
import warnings
warnings.filterwarnings('ignore')
###

parser = argparse.ArgumentParser(description='Process F1 Data.')
parser.add_argument('--c', type=str, default=None, help='Circuit path')
parser.add_argument('--pop', type=int, default=10, help='Population')
parser.add_argument('--mut', type=float, default=0.9, help='Mutation probability value')
parser.add_argument('--cross', type=float, default=0.6, help='Crossover probability value')
parser.add_argument('--i', type=int, default=50, help='Iterations')
parser.add_argument('--w', type=str, default=None, help='Weather file')
parser.add_argument('--d', action='store_true', default=False, help='Data Collection mode')
args = parser.parse_args()
    
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

        print(f"\n-------------------{_circuit}--------------------\n")

        while not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        car:Car = get_car_data(circuit)

        race_data:RaceData = RaceData(circuit)
        #race_data.plot(path=circuit)

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

        best, best_eval, boxplot_data, fitness_data, timer = genetic.run(bf_time = bf_time_in_ms) 
        
        print(f"\n------------------------------------------------\n")
        print(f"EA timing: {ms_to_time(best_eval)}")
        print(f"Bruteforce give timing: {bf_time[-1]}")
        print(f"\n------------------------------------------------\n")

        print(f"\n------------------------------------------------\n")
        ea = f"{int(best_eval):,}".replace(",", " ")
        bf = f"{int(bf_time_in_ms):,}".replace(",", " ")
        print(f"EA fitness: {ea}\nBruteforce fitness: {bf}")
        print(f"\n------------------------------------------------\n")

        localSearch = LocalSearch(best, genetic)
        finalStrategy, finalStrategy_eval = localSearch.run()
        
        ### Print the final strategy of local search
        print("\n------------------------------------------------\n")
        string = "Local Search Strategy:\n"
        for lap in range(genetic.numLaps):
            string += f"Lap {lap+1}/{genetic.numLaps} -> Compound: '{finalStrategy['TyreCompound'][lap]}' TyreAge: {finalStrategy['TyreAge'][lap]} Laps, TyreWear: FL:{round(finalStrategy['TyreWear'][lap]['FL']*100,1)}% FR:{round(finalStrategy['TyreWear'][lap]['FR']*100,1)}% RL:{round(finalStrategy['TyreWear'][lap]['RL']*100,1)}% RR:{round(finalStrategy['TyreWear'][lap]['RR']*100,1)}%, FuelLoad: {finalStrategy['FuelLoad'][lap]} Kg, PitStop: {'Yes' if finalStrategy['PitStop'][lap] else 'No'}, LapTime: {ms_to_time(finalStrategy['LapTime'][lap])} (hh:)mm:ss.ms\n" 

        string += f"\nLocal Search Timing: {ms_to_time(finalStrategy_eval)}"
        with open(os.path.join(save_path, "LocalSearch_strategy.log"), "a") as f:
            f.write(string)
        print(string)
        print(f"EA timing: {ms_to_time(best_eval)}")
        print(f"Bruteforce timing: {bf_time[-1]}")
        print("\n------------------------------------------------\n")

        # print(f"\n------------------------------------------------\n")
        # print(f"EA timing: {ms_to_time(best_eval)}")
        # print(f"LocalSearch timing : {ms_to_time(finalStrategy_eval)}")
        # print(f"\n------------------------------------------------\n")

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
            if not math.isnan(val) or not math.isinf(val):
                fitness_data['LapTime'].append(ms_to_time(val))
            else:
                fitness_data['LapTime'].append(np.nan)

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
        
    
    print(f"\n----------------------END-----------------------\n")
    return best, best_eval, bf_time_in_ms, save_path, timer

if __name__ == "__main__":  
    os.system('cls' if os.name == 'nt' else 'clear')

    population = args.pop
    iterations = args.i
    mutation_pr = args.mut
    crossover_pr = args.cross
    weather = args.w
    circuit = args.c

    wsummary = weather_summary(circuit=circuit, weather_file=weather)
    output_path = os.path.join("Outputs",circuit)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if not os.path.isfile(os.path.join(output_path, f"{circuit}.csv")):
        with open(os.path.join(output_path, f"{circuit}.csv"), "w") as f:
            f.write("Population,Iterations,Mutation,Crossover,EA Fitness,BF Fitness,EA Timing,BF Timing,Timer,Weather,Save Path\n")

    if args.d:
        counter = 0
        while True:
            counter += 1
            strategy, timing, bruteforce_time, log_path, timer = main(population=population, mutation_pr=mutation_pr, crossover_pr=crossover_pr, iterations=iterations, weather=weather, base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs'))
           
            log_path = log_path.replace("\\", "/").split("/")[-1]
        
            with open(os.path.join(output_path, f"{circuit}.csv"), "a") as f:
                f.write(f"{population},{iterations},{mutation_pr},{crossover_pr},{timing},{bruteforce_time},{ms_to_time(timing)},{ms_to_time(bruteforce_time)},{ms_to_time(round(timer*1000))},")
                for w in wsummary:
                    f.write(f"{w} ")
                f.write(f",{log_path}\n")

            if counter >= 10:
                break

    else:

        strategy, timing, bruteforce_time, log_path, timer = main(population=population, mutation_pr=mutation_pr, crossover_pr=crossover_pr, iterations=iterations, weather=weather, base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs'))

        log_path = log_path.replace("\\", "/").split("/")[-1]
        
        with open(os.path.join(output_path, f"{circuit}.csv"), "a") as f:
            f.write(f"{population},{iterations},{mutation_pr},{crossover_pr},{timing},{bruteforce_time},{ms_to_time(timing)},{ms_to_time(bruteforce_time)},{ms_to_time(round(timer*1000))},")
            for w in wsummary:
                f.write(f"({w})")
            f.write(f",{log_path}\n")



    sys.exit(0)
