import math
import time
import numpy as np
import pandas as pd
import os
from random import SystemRandom
import copy

from classes.Car import Car
from classes.Weather import Weather
random = SystemRandom()

from classes.Utils import CIRCUIT, Log, ms_to_time

MIN_LAP_TIME = np.inf
START_FUEL = 0
STRATEGY = None

def boxplot_insert(df:dict, pop_size:int, generation:int, population:list):
    fitnesses = list()

    # for strategy in population:
    #     if strategy['Valid']:
    #         fitnesses.append(strategy['TotalTime'])
    
    # dictionary['Generation'].append(generation)
    # dictionary['Fitness'].append(np.array(fitnesses))
        
    # return dictionary

    for idx in range(pop_size):
        if idx < len(population):
            strategy = population[idx]
            if strategy['Valid']:
                fitnesses.append(strategy['TotalTime'])
            else:
                fitnesses.append(np.nan)
        else:
            fitnesses.append(np.nan)
            
    to_add = pd.DataFrame({generation : fitnesses})
    new_df = to_add
    if df is not None:
        new_df = pd.concat([df, to_add], axis=1).copy()
        
    return new_df

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
    if all([x < 0.4 for x in tyresWear.values()]):
        return False

    boundary = random.random()
    for wear in tyresWear.values():
        if boundary < wear:
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

    def checkValidity(self, strategy:dict):
        all_compounds = set(strategy['TyreCompound'])
        last_lap_fuel_load = self.getFuelLoad(strategy['FuelLoad'][0], strategy['Weather'])
        
        if len(all_compounds) > 1 and last_lap_fuel_load >= 0:
            strategy['Valid'] = True
            return True
        
        strategy['Valid'] = False 
        return False
            

    def getBest(self, population:list, best={'TotalTime':np.inf}):
        idx = -1
        for strategy in population:
            idx += 1
            self.checkValidity(strategy)
            
            if strategy['Valid']:
                if strategy['TotalTime'] < best['TotalTime']:
                    best = strategy
                
        if best['FuelLoad'][-1] < 0:
            print(f"ERROR!!!!!")
        return best, best['TotalTime']


    def startSolver(self,):
        fitness_values = list()
        threshold_quantile = 0.3
        counter = 0
        stuck_value = 0
        #boxplot_dict = {'Generation':list(), 'Fitness':list()}
        boxplot_df = None

        # initial population of random bitstring
        population = self.initSolver()
        
        # enumerate generations
        try:
            for gen in range(self.iterations):
                
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
                    prev = {}
                else:
                    best, best_eval = self.getBest(population, best)

                _, temp_best_eval = self.getBest(population)

                if gen != 0 and prev_temp < temp_best_eval:
                    print("Something is wrong")

                prev_temp = temp_best_eval

                if prev == best_eval:
                    counter += 1
                else:
                    counter = 0
                
                prev = best_eval

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

                #if gen%10:
                string = f'Generation {gen+1} -> best overall: {ms_to_time(best_eval)} - best of generation: {ms_to_time(temp_best_eval)} - population size ratio {round(len(population)/self.population,2)}% | threshold is {threshold_quantile} - counter = {counter}/{(self.iterations)//100} - stuck value = {stuck_value}'
                print(string)
                self.log.write(string+"\n")
                #if (counter/((self.iterations)//100)) > 1:
                #    threshold_quantile = round(threshold_quantile + 0.01,2)

                if counter == 0:
                    threshold_quantile = 0.3
                    stuck_value = 0

                if stuck_value >= 5 and gen > self.iterations//4:
                    string = "Stopping because stucked (Stuck in local minima or global optimum found)"
                    print(string)
                    self.log.write(string+"\n")
                    break

                if counter >= (self.iterations)//100:
                    counter = 0
                    stuck_value += 1
                    quarter_pop = self.population//4
                    population = population[:self.population]
                    idx = random.randint(1, quarter_pop)
                    threshold_quantile = round(threshold_quantile + 0.05,2)
                    for i in range(3*quarter_pop+idx, self.population):
                        population[i] = self.randomChild()
                
                if threshold_quantile <= 0.01:
                    threshold_quantile = 0.3
                elif threshold_quantile >= 0.99:
                    threshold_quantile = 0.3

                if non_random_pop/self.population == 0.0 or non_random_pop == 1:
                    string = f"No valid individuals for generating new genetic material({non_random_pop}), stopping"
                    print(string)
                    self.log.write(string+"\n")
                    break
                
        except KeyboardInterrupt:
            pass 
        
        fit_dict = {'Generation' : range(len(fitness_values)), 'Fitness' : fitness_values}
        
        string = f"\n\nBest Strategy:\n\n"
        print(string)
        self.log.write(string+"\n")


        for lap in range(len(best['TyreCompound'])):
            string = f"Lap {lap+1} -> Compound '{best['TyreCompound'][lap]}', Wear '{round(best['TyreWear'][lap]['FL'],2)}'% | '{round(best['TyreWear'][lap]['FR'],2)}'% | '{round(best['TyreWear'][lap]['RL'],2)}'% | '{round(best['TyreWear'][lap]['RR'],2)}'%, Fuel '{round(best['FuelLoad'][lap],2)}' Kg, PitStop '{'Yes' if best['PitStop'][lap] else 'No'}', Time '{ms_to_time(best['LapTime'][lap])}' ms"
            print(string)
            self.log.write(string+"\n")

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
        initialFuelLoad = self.getInitialFuelLoad(conditions=self.weather)+random.uniform(-10,10)
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

        while oldCompound == compoundRandom:
            compoundRandom = self.randomCompound(child['Weather'][lap])
        
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
        for lap in range(0, self.numLaps):
            if changeTyre(child['TyreWear'][lap]) and child['PitStop'][lap] == False:
                compound = self.randomCompound(child['Weather'][lap])
                tyre_age = 0
                child['PitStop'][lap] = True
                child['TyreWear'][lap] = tyre_age
                child['TyreCompound'][lap] = compound
                child['NumPitStop'] += 1
                remaining = lap
                while remaining < self.numLaps and child['PitStop'][remaining] == True:
                    child['TyreWear'][remaining] = self.getTyreWear(compound=compound, lap=tyre_age)#, conditions=child['Weather'][:remaining])
                    child['TyreCompound'][remaining] = compound
                    child['TyreAge'][remaining] = tyre_age
                    child['LapTime'][remaining] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=remaining, fuel_load=child['FuelLoad'][remaining], conditions=child['Weather'][:remaining], drs=False, pitStop=child['PitStop'][remaining])
                    remaining += 1
                child['TotalTime'] = sum(child['LapTime'])

        return child
            
    def mutation(self,child:dict) -> list:
        childCompound = copy.deepcopy(child)
        childPitStop = copy.deepcopy(child)
        childFuelLoad = copy.deepcopy(child)
        children = []

        if random.random() < self.sigma:
            children.append(self.mutation_compound(childCompound))
        
        if random.random() < self.sigma:
            children.append(self.mutation_pitstop(childPitStop))
            children.append(self.mutation_pitstop_add(childPitStop))

        if random.random() < self.sigma:
            children.append(self.mutation_fuel_load(childFuelLoad))
        
        return children
    
    #def crossover(self, p1:dict, p2:dict,):
    #    # children are copies of parents by default
    #    c1, c2 = p1.copy(), p2.copy()
    #    # check for recombination
    #    if random.random() < self.mu:
    #        # select crossover point that is not on the end of the string
    #        pt = random.randint(1, len(p1['TyreCompound'])-2)
    #        # perform crossover
#
    #        ### {'TyresAvailability': self.availableTyres.copy(), 'TyreCompound': [], 'TyreStatus':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'LapsCompound':[], 'Weather':[], 'TotalTime': np.inf}
    #        c1 = {'TyresAvailability': copy.deepcopy(self.availableTyres),'TyreCompound': p1['TyreCompound'][:pt]+p2['TyreCompound'][pt:], 'TyreAge':p1['TyreAge'][:pt]+p2['TyreAge'][pt:], 'TyreWear': p1['TyreWear'][:pt]+p2['TyreWear'][pt:], 'FuelLoad': p1['FuelLoad'][:pt]+p2['FuelLoad'][pt:], 'PitStop': p1['PitStop'][:pt]+p2['PitStop'][pt:], 'LapTime': p1['LapTime'][:pt]+p2['LapTime'][pt:], 'LapsCompound': p1['LapsCompound'][:pt]+p2['LapsCompound'][pt:], 'Weather':p1['Weather'], 'NumPitStop': p1['NumPitStop'], 'TotalTime': p1['TotalTime']}
    #        c2 = {'TyresAvailability': copy.deepcopy(self.availableTyres),'TyreCompound': p2['TyreCompound'][:pt]+p1['TyreCompound'][pt:], 'TyreAge':p2['TyreAge'][:pt]+p1['TyreAge'][pt:], 'TyreWear': p2['TyreWear'][:pt]+p1['TyreWear'][pt:], 'FuelLoad': p2['FuelLoad'][:pt]+p1['FuelLoad'][pt:], 'PitStop': p2['PitStop'][:pt]+p1['PitStop'][pt:], 'LapTime': p2['LapTime'][:pt]+p1['LapTime'][pt:], 'LapsCompound': p2['LapsCompound'][:pt]+p1['LapsCompound'][pt:], 'Weather':p2['Weather'], 'NumPitStop': p2['NumPitStop'], 'TotalTime': p2['TotalTime']}
    #        
    #        return [self.correct_strategy(c1), self.correct_strategy(c2)]
    #    
    #    return []

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
        c1, c2 = p1.copy(), p2.copy()
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
            tyresAge = strategy['TyreAge'][lap]
            
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

    def build_tree(self, temp_tree:list, tyres_age, lap):
        global MIN_LAP_TIME
        global STRATEGY
        global START_FUEL

        total_time = sum([lap['LapTime'] for lap in temp_tree])
        pitStop_count = sum([x['PitStop'] for x in temp_tree])

        if total_time > MIN_LAP_TIME or pitStop_count > 2:
            return 

        if any([x >= 0.8 for x in temp_tree[-1]['TyresWear'].values()]):
            return 

        if lap == self.numLaps:
            if total_time < MIN_LAP_TIME:
                compound_set = set([x['Compound'] for x in temp_tree])
                if len(compound_set) > 1:
                    fuel_val = self.getFuelLoad(temp_tree[0]['FuelLoad'], self.weather)
                    if fuel_val >= 0:  
                        MIN_LAP_TIME = total_time
                        STRATEGY = copy.deepcopy(temp_tree)
            
            return
         
        fuel_load = self.getFuelLoad(START_FUEL,self.weather[:lap])
        if self.weather[lap] == "Dry":
            for index, compound in enumerate(['Soft', 'Medium', 'Hard']):
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True, False]:
                        node = {'Compound':compound, 'TyresWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyresAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        self.build_tree(temp_tree,tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                else:
                    node = {'Compound':compound, 'TyresWear': self.getTyreWear(compound, 0), 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':True, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=True)}
                    temp_tree.append(node)
                    self.build_tree(temp_tree, 0, lap+1)
                    temp_tree.pop()
        
        else:
            for index, compound in enumerate(['Inter', 'Wet']):
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True, False]:
                        node = {'Compound':compound, 'TyresWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyresAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
                        temp_tree.pop()
                else:
                    node = {'Compound':compound, 'TyresWear': self.getTyreWear(compound, 0), 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':True, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=True)}
                    temp_tree.append(node)
                    self.build_tree(temp_tree, 0, lap+1)
                    temp_tree.pop()

        return 

    def lower_bound(self):
        ### Build the solution space as a tree
        temp_tree = []
        global START_FUEL
        START_FUEL = self.getInitialFuelLoad(self.weather)
        timer_start = time.time()
        
        if self.weather[0] == "Dry":
            for compound in ['Soft','Medium']: #'Hard', 
                start = time.time()
                temp_tree.append({'Compound':compound, 'TyresWear': self.getTyreWear(compound, 0), 'TyresAge':0, 'FuelLoad':START_FUEL, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=START_FUEL, conditions=[self.weather[0]], drs=False, pitStop=False)})
                self.build_tree(temp_tree, 0, 1)
                print(f"{compound} done in {ms_to_time(time.time()-start)}")
                temp_tree.pop()
        else:
            for compound in ['Inter', 'Wet']:
                start = time.time()
                temp_tree.append({'Compound':compound, 'TyresWear': self.getTyreWear(compound, 0), 'TyresAge':0, 'FuelLoad':START_FUEL, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=START_FUEL, conditions=[self.weather[0]], drs=False, pitStop=False)})
                self.build_tree(temp_tree, 0, 1)
                print(f"{compound} done in {ms_to_time(time.time()-start)}")
                temp_tree.pop()

        for lap, strategy in enumerate(STRATEGY):
            print(f"Lap {lap+1}/{self.numLaps} -> Compound: '{strategy['Compound']}', TyresAge: {strategy['TyresAge']} Laps, TyresWear: {strategy['TyresWear']}, FuelLoad: {strategy['FuelLoad']} Kg, PitStop: {'Yes' if strategy['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy['LapTime'])} (hh:)mm:ss.ms")
        print(f"Computed in time {ms_to_time(round((time.time()-timer_start)*1000))}")
        print(f"Total time {ms_to_time(MIN_LAP_TIME)}")
        ### Find the best solution
        return STRATEGY, MIN_LAP_TIME
    
    def fixed_strategy(self):
        stop_lap = 16
        strategy = []
        start_fuel = self.getInitialFuelLoad(self.weather)
        for lap in range(self.numLaps):
            if lap != 0:
                fuel_load = self.getFuelLoad(start_fuel,self.weather[:lap])
            else:
                fuel_load = start_fuel
            if lap < stop_lap:
                strategy.append({'Compound':'Soft', 'TyresWear': self.getTyreWear('Soft', lap), 'TyresAge':lap, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound='Soft', compoundAge=lap, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap] if lap > 0 else [self.weather[0]], drs=False, pitStop=False)})
            elif lap == stop_lap:
                strategy.append({'Compound':'Medium', 'TyresWear': self.getTyreWear('Medium', 0), 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound='Medium', compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap] if lap > 0 else [self.weather[0]], drs=False, pitStop=True)})
            elif lap > stop_lap:
                strategy.append({'Compound':'Medium', 'TyresWear': self.getTyreWear('Medium', lap-9), 'TyresAge':lap-9, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound='Medium', compoundAge=lap-9, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap] if lap > 0 else [self.weather[0]], drs=False, pitStop=False)})

        total = 0
        for lap, strategy in enumerate(strategy):
            print(f"Lap {lap+1}/{self.numLaps} -> Compound: '{strategy['Compound']}', TyresAge: {strategy['TyresAge']} Laps, TyresWear: {strategy['TyresWear']}, FuelLoad: {strategy['FuelLoad']} Kg, PitStop: {'Yes' if strategy['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy['LapTime'])} (hh:)mm:ss.ms")
            total += strategy['LapTime']

        print(f"With a total time of {ms_to_time(total)}")

        return
