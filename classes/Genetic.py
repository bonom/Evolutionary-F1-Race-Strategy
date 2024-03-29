import os
import copy
import math 
import time
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from random import SystemRandom

from classes.Car import Car
from classes.Weather import Weather
from classes.Utils import CIRCUIT, Log, ms_to_time, get_basic_logger

random = SystemRandom()
logger = get_basic_logger('Genetic', logging.INFO)

TYRE_WEAR_THRESHOLD = 0.3
BEST_TIME = np.inf
STRATEGY = None

def boxplot_insert(data_list:list, population:list):
    population = sorted(population, key=lambda x: x['TotalTime'])

    fitnesses = list()

    for strategy in population:
        timing = strategy['TotalTime']
        if strategy['Valid'] and timing > 0:
            fitnesses.append(strategy['TotalTime'])
        else:
            fitnesses.append(np.nan)
            
    data_list.append(fitnesses)
    return data_list

def orderOfMagnitude(number):
    order = 0
    if number < 1 and number > 0:
        while number<1.0:
            order-=1
            number*=10
    elif number > 0:
        while number>1.0:
            order+=1
            number/=10

    return order

def overLimit(values, limit):
    if not isinstance(values, list):
        values = list(values)
    for val in values:
        if val >= limit:
            return True
    
    return False

def changeTyre(tyresWear:dict):
    if all([x < TYRE_WEAR_THRESHOLD for x in tyresWear.values()]):
        return False

    boundary = random.random()
    for wear in tyresWear.values():
        if boundary < wear*2:
            return True
    return False

class GeneticSolver:

    def __init__(self, population:int=2, mutation_pr:float=0.75, crossover_pr:float=0.5, iterations:int=1, car:Car=None, circuit:str='', weather:str='', save_path:str='') -> None:
        self.circuit = circuit
        self.pitStopTime = CIRCUIT[circuit]['PitStopTime']
        self.availableTyres:dict = dict()
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.numLaps = CIRCUIT[circuit]['Laps']
        self.iterations = iterations
        self.car:Car = car
        self.weather = Weather(circuit) if weather == '' else Weather(circuit, weather)

        ### For the log file
        self.path = save_path
        self.log = Log(save_path, values={'Circuit':circuit, 'Weather': self.weather.filename, 'PitStopTime':self.pitStopTime, 'Mutation': mutation_pr, 'Crossover': crossover_pr, 'Population': population, 'Iterations':iterations})
        
        self.mu_decay = 0.99
        self.sigma_decay = 0.99
    

    # Function to run the algorithm
    def run(self,bf_time:int=0):
        start_timer = time.time()

        fitness_values = dict()
        stuck_counter = 0
        boxplot_list = list()
        prev = {}

        ### Initial population of random bitstring
        population = self.initSolver()

        print(f"\n-------------------------------------------------------------\nData for '{self.circuit}':\n\nPopulation = {self.population}\nIterations = {self.iterations}\nMutation = {self.sigma}\nCrossover = {self.mu}\nWeather = {self.weather.filename}\n-------------------------------------------------------------\n")
        
        ### Enumerate generations
        try:
            bar = tqdm(range(self.iterations))
            for gen in bar:
                
                ### Checking if there are duplicates, if so, we remove them
                to_pop = []
                for i in range(0, len(population)-1):
                    for j in range(i+1, len(population)):
                        if population[i] == population[j] and j not in to_pop:
                            to_pop.append(j)

                ### Removing duplicates by sorted indexes would return an error (we are changing length of the list) so we remove them in reversed order
                to_pop = sorted(to_pop, reverse=True)
                for i in to_pop:
                    population.pop(i)
                
                ### Gathering the first solution from population at gen^th generation
                if gen == 0:
                    best, best_eval = self.getBest(population)
                else:
                    best, best_eval = self.getBest(population, best)

                ### Storing data for boxplot
                boxplot_list = boxplot_insert(boxplot_list, population)

                ### Select parents
                selected = self.selection_dynamic_penalty(step=gen+1,population=population,threshold_quantile=2/13, best = best_eval)
                
                 ### Set as parents the selected individuals
                parents = copy.deepcopy(selected)
                
                ### Make a copy of the parents as new children
                children = copy.deepcopy(parents)

                ### Crossover and mutation steps
                for i in range(0, len(parents)-1, 2): 
                    p1, p2 = copy.deepcopy(parents[i]), copy.deepcopy(parents[i+1])

                    for c in self.crossover(p1, p2):
                        children.append(c)
                    
                    for l in self.mutation(parents[i]):
                        children.append(l)
                    
                    for l in self.mutation(parents[i+1]):
                        children.append(l)

                ### Add random children to the population if the population is not full
                for _ in range(self.population-len(children)):
                    children.append(self.randomChild())
                
                ### Replace old population
                population = copy.deepcopy(children)

                if prev == best_eval:
                    stuck_counter += 1
                else:
                    ### Check for new best solution
                    if not math.isinf(best_eval):
                        fitness_values[gen] = best_eval
                    stuck_counter = 0
                
                prev = best_eval

                ### Check if the solution is stucked
                if stuck_counter == 0:
                    threshold_quantile = 0.3

                if stuck_counter >= (self.iterations)//100:
                    stuck_counter = 0
                    quarter_pop = self.population//4
                    population = population[:self.population]
                    idx = random.randint(1, quarter_pop)
                    threshold_quantile = round(threshold_quantile - 0.05,2)
                    for i in range(3*quarter_pop+idx, self.population):
                        population[i] = self.randomChild()
                
                if threshold_quantile <= 0.01 or threshold_quantile >= 0.99:
                    threshold_quantile = round(random.uniform(0.3,0.99),2)

                valid_strategies = round(((sum([1 for x in children if x['Valid'] == True]))/len(children))*100,2)
                bar.set_description(f"Best: {ms_to_time(best_eval)}, Difference: {ms_to_time(best_eval-bf_time)}, Threshold: {threshold_quantile}, Stuck: {stuck_counter}, Valid strategies: {valid_strategies}%")
                bar.refresh()
                string = f'[EA] Generation {gen+1} - Bruteforce solution: {ms_to_time(bf_time)} -> best overall: {ms_to_time(best_eval)} - difference: {ms_to_time(best_eval-bf_time)} - valid strategies: {valid_strategies}% | threshold is {threshold_quantile} - Stuck Counter = {stuck_counter}/{(self.iterations)//100}'
                self.log.write(string+"\n")
                
        except KeyboardInterrupt:
            pass 

        end_timer = time.time() - start_timer

        fit_dict = {'Generation' : list(fitness_values.keys()), 'Fitness' : list(fitness_values.values())}

        strategy_path = os.path.join(self.path, 'Strategy.txt')
        
        string = f"Best Strategy fitness: {best_eval}\nBest Strategy time: {ms_to_time(best_eval)}\n\n vs \n\nBruteforce fitness: {bf_time}\nBruteforce time: {ms_to_time(bf_time)}\n\n\n"

        for lap in range(self.numLaps):
            string += f"Lap {lap+1}: Rain {best['Weather'][lap]}% -> Compound '{best['TyreCompound'][lap]}', TyresAge {best['TyreAge'][lap]}, Wear '{round(best['TyreWear'][lap]['FL']*100,1)}'% | '{round(best['TyreWear'][lap]['FR']*100,1)}'% | '{round(best['TyreWear'][lap]['RL']*100,1)}'% | '{round(best['TyreWear'][lap]['RR']*100,1)}'%, Fuel '{round(best['FuelLoad'][lap],2)}' Kg, PitStop '{'Yes' if best['PitStop'][lap] else 'No'}', Time '{ms_to_time(best['LapTime'][lap])}' ms\n"
        
        with open(strategy_path, 'w') as f:
            f.write(string)
        
        string += f"Time elapsed: {ms_to_time(round(end_timer*1000))}\n"
        
        print("\n\n"+string)
        self.log.write("\n\n"+string)

        boxplot_df = pd.DataFrame(index=range(len(boxplot_list)))
        for i in range(max([len(x) for x in boxplot_list])):
            for j in range(len(boxplot_list)):
                try:
                    boxplot_df.at[i,j] = boxplot_list[j][i]
                except:
                    pass

        boxplot_df.to_csv(os.path.join(self.path,'Boxplot.csv'))

        return best, best_eval, boxplot_df, fit_dict, end_timer
    

    # Function to initialize the population
    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategies.append(self.randomChild())

        return strategies
    

    # Function to build a random strategy
    def randomChild(self):
        strategy = {'TyreCompound': [], 'TyreAge':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'Weather':self.weather.get_weather_percentage_list(), 'Valid':False, 'TotalTime': np.inf}

        weather = strategy['Weather'][:1]

        ### Get a random compound
        compound = self.randomCompound()
        strategy['TyreCompound'].append(compound)
        tyresAge = 0 
        strategy['TyreAge'].append(tyresAge)
        strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))

        ### The fuel load can be inferred by the coefficient of the fuel consumption, we add a random value between -10 and 10 to get a little variation
        initialFuelLoad = round(random.uniform(0,110),2)
        strategy['FuelLoad'].append(initialFuelLoad)

        ### At first lap the pit stop is not made (PitStop list means that at lap i^th the pit stop is made at the beginning of the lap)
        strategy['PitStop'].append(False)

        ### Compute lapTime
        strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=0, fuel_load=initialFuelLoad, conditions=weather, drs=False, pitStop=False))

        ### For every lap we repeat the whole process
        for lap in range(1,self.numLaps):
            weather = strategy['Weather'][:lap+1]

            ### The fuel does not depend on the compound and/or pit stops => we compute it and leave it here
            fuelLoad = self.getFuelLoad(initial_fuel=initialFuelLoad, conditions=weather)
            strategy['FuelLoad'].append(fuelLoad)

            newTyre = random.choice([True, False])

            if newTyre:
                compound = self.randomCompound()
                tyresAge = 0
                pitStop = True
                strategy['NumPitStop'] += 1
            else: 
                compound = strategy['TyreCompound'][lap-1]
                tyresAge += 1
                pitStop = False
            
                
            strategy['TyreAge'].append(tyresAge)
            strategy['TyreCompound'].append(compound)
            strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))
            strategy['PitStop'].append(pitStop)
            strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad, conditions=weather, drs=False, pitStop=pitStop))
           
        strategy['TotalTime'] = sum(strategy['LapTime'])
        return strategy
    

    # Function to get a random compound
    def randomCompound(self,):
        return random.choice(['Soft', 'Medium', 'Hard','Inter','Wet'])
    

    # Function to get the TyreWear given the compound and the lap
    def getTyreWear(self, compound:str, lap:int):
        if lap == 0:
            return {'FL':0.0, 'FR':0.0, 'RL':0.0, 'RR':0.0}

        wear = self.car.predict_tyre_wear(compound, lap)
        
        for key, val in wear.items():
            wear[key] = val/100
        
        return wear
    

    # Function for computing the lap time 
    def getLapTime(self, compound:str, compoundAge:int, lap:int, fuel_load:float, conditions:list, drs:bool, pitStop:bool) -> int:
        conditions_int = conditions[-1]
        conditions_str = [self.weather.get_weather_string(c) for c in conditions[:-1]]

        time = self.car.predict_laptime(tyre=compound, tyre_age=compoundAge, lap=lap, start_fuel=fuel_load, conditions_str=conditions_str, conditions_int=conditions_int, drs=drs)

        if pitStop:
            time += self.pitStopTime

        if lap == 0:
            time += 2000

        return round(time) 
    

    # Function to get the fuel load
    def getFuelLoad(self, initial_fuel:float, conditions:list) :
        weather = [self.weather.get_weather_string(c) for c in conditions[:-1]]
        return round(self.car.predict_fuel_weight(initial_fuel, weather), 2)
    

    # Function to get the best individual/strategy in the population
    def getBest(self, population:list, best={'TotalTime':np.inf}):
        
        for strategy in population:
            self.checkValidity(strategy)
            
            if strategy['Valid']:
                if strategy['TotalTime'] < best['TotalTime']:
                    best = strategy
                
        return best, best['TotalTime']
    

    # Function for checking the validity of a strategy
    def checkValidity(self, strategy:dict):
        all_compounds = set(strategy['TyreCompound'])
        last_lap_fuel_load = self.getFuelLoad(strategy['FuelLoad'][0], strategy['Weather'])

        if any([x != 0 for x in strategy['Weather']]): 
            ### If weather is not completely Dry the constraint of changing tyre does not apply anymore
            if last_lap_fuel_load >= 0:
                strategy['Valid'] = True
                return True
        
        else:
            if len(all_compounds) > 1 and last_lap_fuel_load >= 0:
                strategy['Valid'] = True
                return True
        
        strategy['Valid'] = False 
        return False


    # Selection step with dynamic penalty
    def selection_dynamic_penalty(self, step:int, population:list, threshold_quantile:float, best:int):
        deltas = [abs(x['TotalTime'] - best) for x in population]
        max_delta = max(1,max(deltas))

        alpha = np.exp(1+(1/self.iterations)*step)
        penalty = [(delta/max_delta)*alpha for delta in deltas]

        quantile = np.quantile(penalty, threshold_quantile)

        for p, pop in zip(penalty, population):
            if not pop['Valid']:
                if pop['NumPitStop'] < 1 and all([x == 'Dry' for x in pop['Weather']]):
                    p *= alpha
                    if p == 0.0:
                        p = np.exp(alpha)
                last_lap_fuel_load = self.getFuelLoad(initial_fuel=pop['FuelLoad'][0], conditions=pop['Weather'])
                if last_lap_fuel_load < 0:
                    last_lap_fuel_load = abs(last_lap_fuel_load)
                    p *= np.exp(last_lap_fuel_load)
                    if p == 0.0:
                        p = np.exp(last_lap_fuel_load)
                
        for idx, x in enumerate(population):
            x['Penalty'] = penalty[idx]
        sortedPopulation = sorted(population, key=lambda x: x['Penalty'])
        selected = [x for _, x in enumerate(sortedPopulation) if x['Penalty'] < quantile]
        
        for x in selected:
            x.pop('Penalty')
        
        return selected
    

    # Total crossover step
    def crossover(self, p1:dict, p2:dict,):
        ### Children are copies of parents by default
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

        ### Check for recombination
        if random.random() < self.mu:
            c1, c2 = self.crossover_fuel(c1, c2)
        
        return [c1,c2]


    # Crossover step on the fuel
    def crossover_fuel(self, p1:dict, p2:dict):
        fuelLoad_p1 = p1['FuelLoad'][0]
        fuelLoad_p2 = p2['FuelLoad'][0]

        p1['FuelLoad'][0] = fuelLoad_p2
        p2['FuelLoad'][0] = fuelLoad_p1

        
        for lap in range(1, self.numLaps):
            fuelLoad_p1 = self.getFuelLoad(initial_fuel=fuelLoad_p2, conditions=p1['Weather'][:lap+1])
            fuelLoad_p2 = self.getFuelLoad(initial_fuel=fuelLoad_p1, conditions=p2['Weather'][:lap+1])
            p1['FuelLoad'][lap] = fuelLoad_p2
            p2['FuelLoad'][lap] = fuelLoad_p1
        
        return self.correct_strategy(p1), self.correct_strategy(p2)


    # Function to correct a strategy
    def correct_strategy(self, strategy:dict, index:int=0):
        initialFuelLoad = round(strategy['FuelLoad'][0],2)
        strategy['FuelLoad'][0] = initialFuelLoad
        tyre = strategy['TyreCompound'][0]
        strategy['LapTime'][0] = self.getLapTime(compound=tyre, compoundAge=0, lap=0, fuel_load=initialFuelLoad, conditions=strategy['Weather'][:1], drs=False, pitStop=strategy['PitStop'][0])
        pitStopCounter = 0
        
        if index != 0 and index != self.numLaps:
            compound = strategy['TyreCompound'][index-1]
            tyre_age = strategy['TyreAge'][index-1]
            while index < self.numLaps and strategy['PitStop'][index] == False:
                tyre_age += 1
                strategy['TyreAge'][index] = tyre_age
                strategy['TyreCompound'][index] = compound
                strategy['TyreWear'][index] = self.getTyreWear(strategy['TyreCompound'][index], strategy['TyreAge'][index])
                strategy['LapTime'][index] = self.getLapTime(strategy['TyreCompound'][index], strategy['TyreAge'][index], index, strategy['FuelLoad'][index], strategy['Weather'][:index+1], False, False)
                index += 1

            strategy['NumPitStop'] = sum([x for x in strategy['PitStop'] if x])
            strategy['TotalTime'] = sum(strategy['LapTime'])

            return strategy

        for lap in range(1, self.numLaps):
            weather = strategy['Weather'][:lap+1]

            ### FuelLoad keeps the same, it just needs to be corrected if changed
            fuelLoad = self.getFuelLoad(initial_fuel=initialFuelLoad, conditions=weather)
            strategy['FuelLoad'][lap] = fuelLoad

            ### Get if a pitstop is made and compound lap'
            pitStop = strategy['PitStop'][lap]
            old_compound = strategy['TyreCompound'][lap-1]
            compound = strategy['TyreCompound'][lap]
            tyresAge = strategy['TyreAge'][lap-1]
            
            ### We have two options: either there is a pitstop or the compound has changes, if so we have to recalculate all
            if pitStop or old_compound != compound or any([x >= 0.8 for x in strategy['TyreWear'][lap-1].values()]):
                tyresAge = 0
                pitStop = True
                pitStopCounter += 1
            else:
                tyresAge += 1
                
            tyreWear = self.getTyreWear(compound=compound, lap=tyresAge)
            timing = self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad,conditions=weather, drs=False, pitStop=pitStop)
            strategy['PitStop'][lap] = pitStop
            strategy['TyreWear'][lap] = tyreWear
            strategy['TyreAge'][lap] = tyresAge
            strategy['LapTime'][lap] = timing

        strategy['NumPitStop'] = pitStopCounter
        strategy['TotalTime'] = sum(strategy['LapTime'])

        return strategy


    # Total function for the mutation step
    def mutation(self,child:dict) -> list:
        childAllMutated = copy.deepcopy(child)
        children = []

        if random.random() < self.sigma:
            children.append(self.mutation_compound(copy.deepcopy(child)))
            childAllMutated = self.mutation_compound(childAllMutated)
        
        if random.random() < self.sigma:
            children.append(self.mutation_pitstop(copy.deepcopy(child)))
            children.append(self.mutation_pitstop_add(copy.deepcopy(child)))
        
            childAllMutated = self.mutation_pitstop(childAllMutated)
            childAllMutated = self.mutation_pitstop_add(childAllMutated)

        if random.random() < self.sigma:
            children.append(self.mutation_fuel_load(copy.deepcopy(child)))
            childAllMutated = self.mutation_fuel_load(childAllMutated)
        
        children.append(childAllMutated)
        
        return children


    # Mutation step on the compound
    def mutation_compound(self, child:dict, ):
        usedTyres = dict()
        usedTyres[0] = child['TyreCompound'][0]
        for lap in range(1, self.numLaps):
             if child['TyreCompound'][lap] != child['TyreCompound'][lap - 1]:
                usedTyres[lap] = child['TyreCompound'][lap]

        lapRandom = random.randint(0, len(usedTyres)-1)
        
        lap = list(usedTyres.keys())[lapRandom]
        oldCompound = usedTyres[lap]

        compoundRandom = self.randomCompound()

        while oldCompound == compoundRandom:
            compoundRandom = self.randomCompound()
        
        child['TyreCompound'][lap] = compoundRandom

        for i in range(lap + 1, self.numLaps):
            if not child['PitStop'][lap]:
                child['TyreCompound'][i] = compoundRandom
            else:
                return self.correct_strategy(child)
        
        return self.correct_strategy(child)


    # Mutation step on the pitstop
    def mutation_pitstop(self,child:dict):
        childPitNum = child['NumPitStop'] 

        ### Check if we cannot make different pitStops number
        if childPitNum < 1:
            return self.randomChild()
        
        ### There should be at least 1 pitstop
        if childPitNum == 1: 
            return child
        
        numRandomPitStop = random.randint(1,childPitNum)
        numPitStops = 0
        index = -1
        for lap in range(0, self.numLaps):
            if child['PitStop'][lap] == True:
                numPitStops +=1
                if numPitStops == numRandomPitStop:
                    child['PitStop'][lap] = False
                    child['NumPitStop'] -= 1
                    index = lap

        return self.correct_strategy(child, index)


    # Mutation step for adding a pitstop
    def mutation_pitstop_add(self, child:dict):
        random_lap = random.randint(1, self.numLaps-1)

        while child['PitStop'][random_lap] == True:
            random_lap = random.randint(1, self.numLaps-1)
        
        compound = self.randomCompound()
        
        tyre_age = 0
        child['PitStop'][random_lap] = True
        child['TyreAge'][random_lap] = tyre_age
        child['TyreWear'][random_lap] = self.getTyreWear(compound=compound, lap=tyre_age)
        child['TyreCompound'][random_lap] = compound
        child['LapTime'][random_lap] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=random_lap, fuel_load=child['FuelLoad'][random_lap], conditions=child['Weather'][:random_lap] if random_lap != 0 else child['Weather'][:random_lap], drs=False, pitStop=child['PitStop'][random_lap])
        child['NumPitStop'] += 1
        remaining = random_lap + 1
        tyre_age += 1
        while remaining < self.numLaps and child['PitStop'][remaining] == False:
            child['TyreWear'][remaining] = self.getTyreWear(compound=compound, lap=tyre_age)
            child['TyreCompound'][remaining] = compound
            child['TyreAge'][remaining] = tyre_age
            child['LapTime'][remaining] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=remaining, fuel_load=child['FuelLoad'][remaining], conditions=child['Weather'][:remaining+1], drs=False, pitStop=child['PitStop'][remaining])
            remaining += 1
            tyre_age += 1
        child['TotalTime'] = sum(child['LapTime'])
        
        return child
    

    # Mutation step on the fuel
    def mutation_fuel_load(self, child:dict, ):
        new_fuel = child['FuelLoad'][0]+random.uniform(-10,10)

        child['FuelLoad'][0] = new_fuel
        child['LapTime'][0] = self.getLapTime(compound=child['TyreCompound'][0], compoundAge=child['TyreAge'][0], lap=0, fuel_load=new_fuel, conditions=child['Weather'][:1], drs=False, pitStop=child['PitStop'][0])
        
        for lap in range(1,self.numLaps):
            fuel = self.getFuelLoad(initial_fuel=new_fuel, conditions=child['Weather'][:lap+1])
            timing = self.getLapTime(compound=child['TyreCompound'][lap], compoundAge=child['TyreAge'][lap], lap=lap, fuel_load=fuel, conditions=child['Weather'][:lap+1], drs=False, pitStop=child['PitStop'][lap])
            
            child['FuelLoad'][lap] = fuel
            child['LapTime'][lap] = timing

        child['TotalTime'] = sum(child['LapTime'])
        return child


    """
    Bruteforce algorithm
    """
    # Function to build the tree of the bruteforce algorithm
    def build_tree(self, temp_tree:list, tyres_age:int, lap:int):
        global BEST_TIME
        global STRATEGY

        weather = self.weather.get_weather_percentage_list()
        total_time = sum([x['LapTime'] for x in temp_tree])
        pitStop_count = sum([x['PitStop'] for x in temp_tree])
        initial_fuel = temp_tree[0]['FuelLoad']
        compound_set = set([x['Compound'] for x in temp_tree])

        if total_time > BEST_TIME or pitStop_count > 2:
            return {'Strategy':None, 'TotalTime':np.inf}

        if any([x >= 0.8 for x in temp_tree[-1]['TyreWear'].values()]):
            return {'Strategy':None, 'TotalTime':np.inf}

        if lap == self.numLaps:
            if total_time < BEST_TIME:
                if len(compound_set) > 1:
                    BEST_TIME = total_time
                    STRATEGY = copy.deepcopy(temp_tree)
                    return {'Strategy':copy.deepcopy(temp_tree), 'TotalTime':total_time}
            
            return {'Strategy':None, 'TotalTime':np.inf}
         
        fuel_load = self.getFuelLoad(initial_fuel,weather[:lap+1])
        w = weather[lap]

        if w < 20:
            idx = 1
            values = {1:None, 2:None, 3:None, 4:None}
            for compound in ['Soft', 'Medium','Hard']:
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True,False]:
                        node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                        idx+=1
                else:
                    pitStop = True
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
                    idx+=1
        elif w > 50 and w < 80:
            values = {1:None}
            idx = 1
            compound = 'Inter'
            if compound == temp_tree[-1]['Compound']:
                for pitStop in [True,False]:
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
            else:
                pitStop = True
                node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                temp_tree.append(node)
                values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                temp_tree.pop()
        elif w > 80:
            values = {1:None}
            idx = 1
            compound = 'Wet'
            if compound == temp_tree[-1]['Compound']:
                for pitStop in [True,False]:
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
            else:
                pitStop = True
                node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                temp_tree.append(node)
                values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                temp_tree.pop()
        else:
            values = {1:None, 2:None, 3:None, 4:None}
            idx = 1
            for compound in ['Inter','Soft', 'Medium','Hard']:
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True,False]:
                        node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                        idx+=1
                else:
                    pitStop = True
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
                    idx+=1

        ### Best Strategy
        to_remove = list()
        for key, val in values.items():
            if val is None:
                to_remove.append(key)
        for key in to_remove:
            values.pop(key)
        for val in values.values():
            if val['Strategy'] is not None:
                if len(val['Strategy']) < self.numLaps:
                    val['TotalTime'] = np.inf
        best_strategy = min(values, key=lambda x: values[x]['TotalTime'])


        return {'Strategy':copy.deepcopy(values[best_strategy]['Strategy']), 'TotalTime':values[best_strategy]['TotalTime']}


    # Function to get the result of the bruteforce algorithm
    def lower_bound(self,):
        ### Build the solution space as a tree
        temp_tree = []
        weather = self.weather.get_weather_percentage_list()
        initial_fuel = self.getInitialFuelLoad(weather)
        timer_start = time.time()
        w = weather[0]

        if w < 20:
            values = {1:None, 2:None, 3:None}
            
            ### Soft
            soft_timer = time.time()
            compound = 'Soft'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            soft_timer = ms_to_time(round(1000*(time.time() - soft_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {soft_timer}")

            ### Medium
            medium_timer = time.time()
            compound = 'Medium'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[2] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            medium_timer = ms_to_time(round(1000*(time.time() - medium_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {medium_timer}")

            ### Hard
            hard_timer = time.time()
            compound = 'Hard'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[3] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            hard_timer = ms_to_time(round(1000*(time.time() - hard_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {hard_timer}")
        elif w > 50 and w < 80:
            values = {1:None}
            ### Inter
            inter_timer = time.time()
            compound = 'Inter'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            inter_timer = ms_to_time(round(1000*(time.time() - inter_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {inter_timer}")
        elif w > 80:
            values = {1:None}

            ### Wet
            wet_timer = time.time()
            compound = 'Wet'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            wet_timer = ms_to_time(round(1000*(time.time() - wet_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {wet_timer}")
        else:
            values = {1:None, 2:None, 3:None, 4:None}
            
            ### Inter
            inter_timer = time.time()
            compound = 'Inter'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            inter_timer = ms_to_time(round(1000*(time.time() - inter_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {inter_timer}")

            ### Soft
            soft_timer = time.time()
            compound = 'Soft'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[2] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            soft_timer = ms_to_time(round(1000*(time.time() - soft_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {soft_timer}")

            ### Medium
            medium_timer = time.time()
            compound = 'Medium'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[3] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            medium_timer = ms_to_time(round(1000*(time.time() - medium_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {medium_timer}")

            ### Hard
            hard_timer = time.time()
            compound = 'Hard'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
            values[4] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            hard_timer = ms_to_time(round(1000*(time.time() - hard_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {hard_timer}")

        ### Best Strategy
        best_strategy_index = min(values, key=lambda x: values[x]['TotalTime'])
        best_strategy, best_laptime = values[best_strategy_index]['Strategy'], values[best_strategy_index]['TotalTime']

        if best_strategy is None:
            print(f"Strategy is none...")
            exit(-1)

        for lap, strategy in enumerate(best_strategy):
            print(f"Lap {lap+1} -> Compound '{strategy['Compound']}', TyresAge {strategy['TyreAge']}, Wear '{round(strategy['TyreWear']['FL']*100,1)}'% | '{round(strategy['TyreWear']['FR']*100,1)}'% | '{round(strategy['TyreWear']['RL']*100,1)}'% | '{round(strategy['TyreWear']['RR']*100,1)}'%, Fuel '{round(strategy['FuelLoad'],2)}' Kg, PitStop '{'Yes' if strategy['PitStop'] else 'No'}', Time '{ms_to_time(strategy['LapTime'])}' ms")
        print(f"Computed in time {ms_to_time(round(1000*(time.time()-timer_start)))}")
        print(f"Total time {ms_to_time(best_laptime)}")

        ### Find the best solution
        return best_strategy, best_laptime
    

    # Function to get the initial fuel load 
    def getInitialFuelLoad(self, conditions:list):
        weather = [self.weather.get_weather_string(c) for c in conditions[:-1]]
        return round(self.car.predict_starting_fuel(weather), 2)