import math
import time
import numpy as np
import pandas as pd
import os
from random import SystemRandom
import copy
from tqdm import tqdm

from classes.Car import Car
from classes.Weather import Weather
random = SystemRandom()

from classes.Utils import CIRCUIT, Log, ms_to_time

TYRE_WEAR_THRESHOLD = 0.3
BEST_TIME = np.inf
STRATEGY = None

def boxplot_insert(df:dict, pop_size:int, generation:int, population:list):
    fitnesses = list()
    for idx in range(pop_size):
        if idx < len(population):
            strategy = population[idx]
            if strategy['Valid']:
                fitnesses.append(strategy['TotalTime'])
            else:
                fitnesses.append(np.nan)
        else:
            fitnesses.append(np.nan)
            
    if df is not None:
        to_add = pd.DataFrame({generation + 1 : fitnesses}, index=df.index)
        return pd.concat([df, to_add], axis=1) 
        
    return pd.DataFrame({generation : fitnesses}, index=population)

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
    #valutare di prendere la media dei valori e fare il controllo con il random solo su quella
    for wear in tyresWear.values():
        #if wear > 0.3:
            if boundary < wear*2:
                return True
    return False

class GeneticSolver:
    def __init__(self, population:int=2, mutation_pr:float=0.75, crossover_pr:float=0.5, iterations:int=1, car:Car=None, circuit:str='') -> None:
        self.circuit = circuit
        self.pitStopTime = CIRCUIT[circuit]['PitStopTime']
        self.availableTyres:dict = dict()
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.numLaps = CIRCUIT[circuit]['Laps']
        self.iterations = iterations
        self.car:Car = car
        weather = Weather(circuit, self.numLaps)
        self.weather = weather.get_weather_list()

        # For the log.txt
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log = Log(os.path.join(path,'Data', circuit))
        
        self.mu_decay = 0.99
        self.sigma_decay = 0.99
    
    def getTyreWear(self, compound:str, lap:int):
        if lap == 0:
            return {'FL':0.0, 'FR':0.0, 'RL':0.0, 'RR':0.0}

        wear = self.car.predict_tyre_wear(compound, lap)
        
        for key, val in wear.items():
            wear[key] = val/100
        
        return wear
    
    def getBestLapTime(self,):
        return self.car.time_diff['Soft']

    def getFuelLoad(self, initial_fuel:float, conditions:list) :
        return round(self.car.predict_fuel_weight(initial_fuel, conditions), 2)
    
    def getInitialFuelLoad(self, conditions:list):
        return round(self.car.predict_starting_fuel(conditions), 2)

    def getWearTimeLose(self, compound:str, lap:int):
        return self.car.predict_tyre_time_lose(compound, lap)
        
    def getFuelTimeLose(self, lap:int=0, fuel_load:float=0, initial_fuel:float=0, conditions:list=None):
        if fuel_load == 0:
            fuel_load = self.getFuelLoad(lap, initial_fuel, conditions)
        
        return self.car.predict_fuel_time_lose(fuel_load)
    
    def getLapTime(self, compound:str, compoundAge:int, lap:int, fuel_load:float, conditions:list, drs:bool, pitStop:bool) -> int:
        time = self.car.predict_laptime(tyre=compound, tyre_age=compoundAge, lap=lap, start_fuel=fuel_load, conditions=conditions, drs=drs)

        if pitStop:
            time += self.pitStopTime

        if lap == 0:
            time += 2000

        return round(time)     

    def getBest(self, population:list, best={'TotalTime':np.inf}):
        idx = -1
        for strategy in population:
            idx += 1
            self.checkValidity(strategy)
            
            if strategy['Valid']:
                if strategy['TotalTime'] < best['TotalTime']:
                    best = strategy
                
        return best, best['TotalTime']

    def checkValidity(self, strategy:dict):
        all_compounds = set(strategy['TyreCompound'])
        last_lap_fuel_load = self.getFuelLoad(strategy['FuelLoad'][0], strategy['Weather'])
        
        if any([x != 'Dry' for x in strategy['Weather']]):
            if last_lap_fuel_load >= 0:
                strategy['Valid'] = True
                return True
        
        if len(all_compounds) > 1 and last_lap_fuel_load >= 0:
            strategy['Valid'] = True
            return True
        
        strategy['Valid'] = False 
        return False

    def startSolver(self,bf_time:int=0):
        start_timer = time.time()

        fitness_values = list()
        threshold_quantile = 0.3
        counter = 0
        stuck_value = 0
        boxplot_df = None
        prev = {}

        # initial population of random bitstring
        population = self.initSolver()
        
        # enumerate generations
        try:
            bar = tqdm(range(self.iterations))
            for gen in bar:
                
                # Checking if there are duplicates, if so, we remove them
                to_pop = []
                for i in range(0, len(population)-1):
                    for j in range(i+1, len(population)):
                        if population[i] == population[j] and j not in to_pop:
                            to_pop.append(j)

                # Removing duplicates by sorted indexes would return an error (we are changing length of the list) so we remove them in reversed order
                to_pop = sorted(to_pop, reverse=True)
                for i in to_pop:
                    population.pop(i)

                # Storing data for boxplot
                boxplot_df = boxplot_insert(boxplot_df, self.population, gen, population)
                
                # Gathering the first solution from population at gen^th generation
                if gen == 0:
                    best, best_eval = self.getBest(population)
                else:
                    best, best_eval = self.getBest(population, best)

                _, temp_best_eval = self.getBest(population)

                # Select parents
                selected = self.selection_dynamic_penalty(step=gen+1,population=population,threshold_quantile=threshold_quantile, best = best_eval)
                
                # Create the next generation
                children = [parent for parent in selected]

                if len(selected) > self.population:
                    selected = selected[:self.population]

                if len(selected) > 1:
                    for i in range(0, len(selected)-2, 2): # why not 1? I know there will be 2*population length - 2 but maybe it is good
                        # Get selected parents in pairs
                        p1, p2 = copy.deepcopy(selected[i]), copy.deepcopy(selected[i+1])
                        # Crossover 
                        for c in self.crossover(p1, p2):
                            # Mutation
                            children.append(c)

                        # Mutation
                        for l in self.mutation(selected[i]):
                            children.append(l)

                        for l in self.mutation(selected[i+1]):
                            children.append(l)

                non_random_pop = len(children)

                # Add random children to the population if the population is not full
                for _ in range(self.population-len(children)):
                    children.append(self.randomChild())
                
                # Replace population
                population = copy.deepcopy(children)

                # Check for new best solution
                fitness_values.append(temp_best_eval)

                if prev == best_eval:
                    counter += 1
                else:
                    counter = 0
                
                prev = best_eval

                if counter == 0:
                    threshold_quantile = 0.3
                    stuck_value = 0

                if counter >= (self.iterations)//100:
                    counter = 0
                    stuck_value += 1
                    quarter_pop = self.population//4
                    population = population[:self.population]
                    idx = random.randint(1, quarter_pop)
                    threshold_quantile = round(threshold_quantile - 0.05,2)
                    for i in range(3*quarter_pop+idx, self.population):
                        population[i] = self.randomChild()
                
                if threshold_quantile <= 0.01 or threshold_quantile >= 0.99:
                    threshold_quantile = round(random.uniform(0.3,0.99),2)

                if non_random_pop/self.population == 0.0 or non_random_pop == 1:
                    string = f"No valid individuals for generating new genetic material({non_random_pop}), stopping"
                    print("\n"+string)
                    self.log.write(string+"\n")
                    break
                    
                if best_eval <= bf_time:
                    string = f"Found the best possible solution in {gen+1} generations"
                    print("\n"+string)
                    self.log.write(string+"\n")                    
                    break

                bar.set_description(f"{gen}/{self.iterations} - BF: {ms_to_time(bf_time)}, Best: {ms_to_time(best_eval)}, Difference: {ms_to_time(best_eval-bf_time)}, Threshold: {threshold_quantile}, Stuck: {stuck_value}, Non-random: {round(len(population)/self.population,2)}%")
                bar.refresh()
                string = f'[EA] Generation {gen+1} - Bruteforce solution: {ms_to_time(bf_time)} -> best overall: {ms_to_time(best_eval)} - best of generation: {ms_to_time(temp_best_eval)} - population size ratio {round((len(population)/self.population)*100,1)}% | threshold is {threshold_quantile} - counter = {counter}/{(self.iterations)//100} - stuck value = {stuck_value}'
                #print("\n"+string)
                self.log.write(string+"\n")
                
        except KeyboardInterrupt:
            pass 

        end_timer = time.time() - start_timer
        
        fit_dict = {'Generation' : range(len(fitness_values)), 'Fitness' : fitness_values}
        
        string = f"\n\nBest Strategy: {ms_to_time(best_eval)} vs {ms_to_time(bf_time)}"
        print(string)
        self.log.write(string+"\n")


        for lap in range(len(best['TyreCompound'])):
            string = f"Lap {lap+1} -> Compound '{best['TyreCompound'][lap]}', TyresAge {best['TyreAge'][lap]}, Wear '{round(best['TyreWear'][lap]['FL']*100,1)}'% | '{round(best['TyreWear'][lap]['FR']*100,1)}'% | '{round(best['TyreWear'][lap]['RL']*100,1)}'% | '{round(best['TyreWear'][lap]['RR']*100,1)}'%, Fuel '{round(best['FuelLoad'][lap],2)}' Kg, PitStop '{'Yes' if best['PitStop'][lap] else 'No'}', Time '{ms_to_time(best['LapTime'][lap])}' ms"
            print(string)
            self.log.write(string+"\n")
        
        string = f"Time elapsed: {ms_to_time(round(end_timer*1000))}"
        print("\n"+string)
        self.log.write("\n"+string+"\n")

        return best, best_eval, boxplot_df, fit_dict

    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategies.append(self.randomChild())

        return strategies

    def randomChild(self):
        strategy = {'TyreCompound': [], 'TyreAge':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'Weather':self.weather.copy(), 'Valid':False, 'TotalTime': np.inf}

        weather = [strategy['Weather'][0]]

        ### Get a random compound and verify that we can use it, if so we update the used compounds list and add the compound to the strategy
        compound = self.randomCompound(weather[0])
        strategy['TyreCompound'].append(compound)

        ### If the compound is used we put a tyre wear of 2 laps (if it is used but available the compound has been used for 2/3 laps.
        ### However, 2 laps out of 3 are done very slowly and the wear is not as the same of 3 laps)
        ### If the compound is new tyre wear = 0
        tyresAge = 0 
        strategy['TyreAge'].append(tyresAge)
        strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))

        ### The fuel load can be inferred by the coefficient of the fuel consumption, we add a random value between -10 and 10 to get a little variation
        initialFuelLoad = random.uniform(0,110)
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
                compound = self.randomCompound(weather[lap])
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
        
        self.checkValidity(strategy)    
        strategy['TotalTime'] = sum(strategy['LapTime'])
        return strategy

    def randomCompound(self,weather:str):
        if weather == 'Wet':
            return 'Inter'
        elif weather == 'VWet':
            return 'Wet'
        elif weather == 'Dry/Wet':
            return random.choice(['Soft', 'Medium', 'Hard', 'Inter'])
        return random.choice(['Soft', 'Medium', 'Hard'])

    def selection(self,population, percentage:float=0.4):
        sortedPopulation = sorted(population, key=lambda x: x['TotalTime'])
        
        selected = [x for x in sortedPopulation if not math.isinf(x['TotalTime'])]
        
        if len(selected) >= int(len(population)*percentage):
            return selected[:int(len(population)*percentage)]
        
        return selected

    def selection_dynamic_penalty(self, step:int, population:list, threshold_quantile:float, best:int):
        deltas = [abs(x['TotalTime'] - best) for x in population]
        max_delta = max(1,max(deltas))

        penalty= [delta/max_delta for delta in deltas]

        alpha = np.exp(1+(1/self.iterations)*step)
        penalty = [p*alpha for p in penalty]

        quantile = np.quantile(penalty, threshold_quantile)

        for p, pop in zip(penalty, population):
            if not pop['Valid']:
                if pop['NumPitStop'] < 1:
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
        selected = [x for idx, x in enumerate(sortedPopulation) if x['Penalty'] < quantile]
        for x in sortedPopulation:
            x.pop('Penalty')

        if len(selected) <= 1:
            threshold_quantile+=0.01
            return self.selection_dynamic_penalty(step, population, threshold_quantile, best)
        
        
        return selected

    def mutation_fuel_load(self, child:dict, ):
        new_fuel = child['FuelLoad'][0]+random.uniform(-10,10)

        child['FuelLoad'][0] = new_fuel
        child['LapTime'][0] = self.getLapTime(compound=child['TyreCompound'][0], compoundAge=child['TyreAge'][0], lap=0, fuel_load=new_fuel, conditions=[child['Weather'][0]], drs=False, pitStop=child['PitStop'][0])
        
        for lap in range(1,len(child['FuelLoad'])):
            fuel = self.getFuelLoad(initial_fuel=new_fuel, conditions=child['Weather'][:lap])
            timing = self.getLapTime(compound=child['TyreCompound'][lap], compoundAge=child['TyreAge'][lap], lap=lap, fuel_load=fuel, conditions=child['Weather'][:lap], drs=False, pitStop=child['PitStop'][lap])
            
            child['FuelLoad'][lap] = fuel
            child['LapTime'][lap] = timing

        child['TotalTime'] = sum(child['LapTime'])
        return child

    def mutation_compound(self, child:dict, ):
        usedTyres = dict()
        usedTyres[0] = child['TyreCompound'][0]
        for lap in range(1, self.numLaps):
             if child['TyreCompound'][lap] != child['TyreCompound'][lap - 1]:
                usedTyres[lap] = child['TyreCompound'][lap]

        lapRandom = random.randint(0, len(usedTyres)-1)
        
        lap = list(usedTyres.keys())[lapRandom]
        oldCompound = usedTyres[lap]

        compoundRandom = self.randomCompound(child['Weather'][lap])
        weather = child['Weather'][lap]

        if weather in ['Dry','Dry/Wet']:
            while oldCompound == compoundRandom:
                compoundRandom = self.randomCompound(weather)
        else:
            compoundRandom = self.randomCompound(weather)
        
        child['TyreCompound'][lap] = compoundRandom

        for i in range(lap + 1, self.numLaps):
            if not child['PitStop'][lap]:
                child['TyreCompound'][i] = compoundRandom
            else:
                return self.correct_strategy(child)
        
        return self.correct_strategy(child)

    def mutation_pitstop(self,child:dict):
        childPitNum = child['NumPitStop'] 

        ### Check if we cannot make different pitStops number
        if childPitNum < 1:
            return self.randomChild()
        
        #There should be at least 1 pitstop
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
        #return self.correct_strategy_pitstop(strategy=child, indexPitStop=index)

    def mutation_pitstop_add(self, child:dict):
        #print(child)
        
        for lap in range(0, self.numLaps):
            #print(lap)
            if changeTyre(child['TyreWear'][lap]) and child['PitStop'][lap] == False:
                compound = self.randomCompound(child['Weather'][lap])
                #print("Sono entrato in add pitstop : giro ", lap, "vecchio compound ", child['TyreCompound'][lap], "nuovo compund ", compound)
                tyre_age = 0
                child['PitStop'][lap] = True
                child['TyreAge'][lap] = tyre_age
                child['TyreWear'][lap] = self.getTyreWear(compound=compound, lap=tyre_age)
                child['TyreCompound'][lap] = compound
                child['LapTime'][lap] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=lap, fuel_load=child['FuelLoad'][lap], conditions=child['Weather'][:lap] if lap != 0 else [child['Weather'][0]], drs=False, pitStop=child['PitStop'][lap])
                child['NumPitStop'] += 1
                remaining = lap + 1
                tyre_age += 1
                while remaining < self.numLaps and child['PitStop'][remaining] == False:
                    child['TyreWear'][remaining] = self.getTyreWear(compound=compound, lap=tyre_age)#, conditions=child['Weather'][:remaining])
                    child['TyreCompound'][remaining] = compound
                    child['TyreAge'][remaining] = tyre_age
                    child['LapTime'][remaining] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=remaining, fuel_load=child['FuelLoad'][remaining], conditions=child['Weather'][:remaining], drs=False, pitStop=child['PitStop'][remaining])
                    remaining += 1
                    tyre_age += 1
                child['TotalTime'] = sum(child['LapTime'])
                #print(child)
                return child
        return child
            
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

    def crossover_fuel(self, p1:dict, p2:dict):
        fuelLoad_p1 = p1['FuelLoad'][0]
        fuelLoad_p2 = p2['FuelLoad'][0]

        p1['FuelLoad'][0] = fuelLoad_p2
        p2['FuelLoad'][0] = fuelLoad_p1

        
        for lap in range(1, self.numLaps):
            fuelLoad_p1 = self.getFuelLoad(initial_fuel=fuelLoad_p2, conditions=p1['Weather'][:lap])
            fuelLoad_p2 = self.getFuelLoad(initial_fuel=fuelLoad_p1, conditions=p2['Weather'][:lap])
            p1['FuelLoad'][lap] = fuelLoad_p2
            p2['FuelLoad'][lap] = fuelLoad_p1
        
        return self.correct_strategy(p1), self.correct_strategy(p2)
        

    def crossover(self, p1:dict, p2:dict,):
        # children are copies of parents by default
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        # check for recombination
        if random.random() < self.mu:
            c1, c2 = self.crossover_fuel(c1, c2)
            return [c1, c2]
            #return [self.correct_strategy(c1), self.correct_strategy(c2)]
        
        return [p1,p2]

    def correct_strategy(self, strategy:dict, index:int=0):
        initialFuelLoad = strategy['FuelLoad'][0]
        pitStopCounter = 0
        
        if index != 0 and index != self.numLaps:
            compound = strategy['TyreCompound'][index-1]
            tyre_age = strategy['TyreAge'][index-1]
            while index < self.numLaps and strategy['PitStop'][index] == False:
                tyre_age += 1
                strategy['TyreAge'][index] = tyre_age
                strategy['TyreCompound'][index] = compound
                strategy['TyreWear'][index] = self.getTyreWear(strategy['TyreCompound'][index], strategy['TyreAge'][index])
                strategy['LapTime'][index] = self.getLapTime(strategy['TyreCompound'][index], strategy['TyreAge'][index], index, strategy['FuelLoad'][index], strategy['Weather'][:index], False, False)
                index += 1

            strategy['NumPitStop'] = sum([x for x in strategy['PitStop'] if x])
            strategy['TotalTime'] = sum(strategy['LapTime'])

            return strategy

        for lap in range(1, self.numLaps):
            ### FuelLoad keeps the same, it just needs to be corrected if changed
            fuelLoad = self.getFuelLoad(initial_fuel=initialFuelLoad, conditions=strategy['Weather'][:lap])
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
            strategy['PitStop'][lap] = pitStop
            strategy['TyreWear'][lap] = tyreWear
            strategy['TyreAge'][lap] = tyresAge
            weather = strategy['Weather'][:lap]
            strategy['LapTime'][lap] = self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad,conditions=weather, drs=False, pitStop=pitStop)

        strategy['NumPitStop'] = pitStopCounter
        strategy['TotalTime'] = sum(strategy['LapTime'])
        self.checkValidity(strategy)

        return strategy

    def fillRemainings(self, lap:int, strategy:dict):
        compound = strategy['TyreCompound'][lap-1]
        for _ in range(lap, self.numLaps):
            strategy['TyreAge'].append(0)
            strategy['TyreCompound'].append(compound)
            strategy['TyreWear'].append({'FL':0, 'FR':0, 'RL':0, 'RR':0})
            strategy['FuelLoad'].append(0)
            strategy['PitStop'].append(True)
            strategy['LapTime'].append(np.inf)

        return strategy

    def build_tree(self, temp_tree:list, tyres_age:int, lap:int):
        global BEST_TIME
        global STRATEGY

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
         
        fuel_load = self.getFuelLoad(initial_fuel,self.weather[:lap])


        if self.weather[lap] == "Dry":
            idx = 1
            values = {1:None, 2:None, 3:None, 4:None}
            for compound in ['Soft', 'Medium','Hard']:
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True,False]:
                        node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                        idx+=1
                else:
                    pitStop = True
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
                    idx+=1
            
        else:
            values = {1:None, 2:None, 3:None}
            for compound in ['Inter','Wet']:
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True,False]:
                        node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                        idx+=1
                else:
                    pitStop = True
                    node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                    temp_tree.append(node)
                    values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                    temp_tree.pop()
                    idx+=1

        ### Best Strategy
        for val in values.values():
            if val['Strategy'] is not None:
                if len(val['Strategy']) < self.numLaps:
                    val['TotalTime'] = np.inf
        best_strategy = min(values, key=lambda x: values[x]['TotalTime'])


        return {'Strategy':copy.deepcopy(values[best_strategy]['Strategy']), 'TotalTime':values[best_strategy]['TotalTime']}

    def lower_bound(self,):
        ### Build the solution space as a tree
        temp_tree = []
        initial_fuel = self.getInitialFuelLoad(self.weather)
        timer_start = time.time()
        
        if self.weather[0] == "Dry":
            values = {1:None, 2:None, 3:None}
            
            ### Soft
            soft_timer = time.time()
            compound = 'Soft'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            soft_timer = ms_to_time(round(1000*(time.time() - soft_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {soft_timer}")

            #return values[1]['Strategy'], values[1]['TotalTime']

            ### Medium
            medium_timer = time.time()
            compound = 'Medium'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
            values[2] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            medium_timer = ms_to_time(round(1000*(time.time() - medium_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {medium_timer}")

            ### Hard
            hard_timer = time.time()
            compound = 'Hard'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
            values[3] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            hard_timer = ms_to_time(round(1000*(time.time() - hard_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {hard_timer}")

        else:
            values = {1:None, 2:None}
            
            ### Inter
            inter_timer = time.time()
            compound = 'Inter'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
            values[1] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            inter_timer = ms_to_time(round(1000*(time.time() - inter_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {inter_timer}")

            ### Wet
            wet_timer = time.time()
            compound = 'Wet'
            print(f"[BruteForce] Computations starting with {compound}...")
            temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
            values[2] = self.build_tree(temp_tree, 0, 1)
            temp_tree.pop()
            wet_timer = ms_to_time(round(1000*(time.time() - wet_timer)))
            print(f"\033[A\033[K[BruteForce] {compound} computed in {wet_timer}")

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
    
    def fixed_strategy(self, compund_list:list, stop_lap:list):
        if len(stop_lap) != len(compund_list)-1:
            print(f"Either the compound list or the pit stop list are wrong!")
            exit(-1)
        stop_lap.append(self.numLaps)
        strategy = []
        start_fuel = self.getInitialFuelLoad(self.weather)
        idx_stop = 0
        idx_tyre = 0
        tyre_lap = 0
        for lap in range(self.numLaps):
            if lap != 0:
                fuel_load = self.getFuelLoad(start_fuel,self.weather[:lap])
            else:
                fuel_load = start_fuel

            if lap < stop_lap[idx_stop]:
                strategy.append({'Compound':compund_list[idx_tyre], 'TyresWear': self.getTyreWear(compund_list[idx_tyre], tyre_lap), 'TyresAge':tyre_lap, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound=compund_list[idx_tyre], compoundAge=tyre_lap, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap] if lap > 0 else [self.weather[0]], drs=False, pitStop=False)})
            elif lap == stop_lap[idx_stop]:
                idx_tyre += 1
                strategy.append({'Compound':compund_list[idx_tyre], 'TyresWear': self.getTyreWear(compund_list[idx_tyre], 0), 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound=compund_list[idx_tyre], compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap] if lap > 0 else [self.weather[0]], drs=False, pitStop=True)})
                idx_stop += 1
                tyre_lap = 0

            tyre_lap += 1
            
        total = 0
        for lap, strategy in enumerate(strategy):
            print(f"Lap {lap+1}/{self.numLaps} -> Compound: '{strategy['Compound']}', TyresAge: {strategy['TyresAge']} Laps, TyresWear: {strategy['TyresWear']}, FuelLoad: {strategy['FuelLoad']} Kg, PitStop: {'Yes' if strategy['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy['LapTime'])} (hh:)mm:ss.ms")
            total += strategy['LapTime']

        print(f"With a total time of {ms_to_time(total)} -> {total}")

        return
