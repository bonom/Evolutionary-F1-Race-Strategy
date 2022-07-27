import math
import time
import numpy as np
from classes.Car import Car
from random import SystemRandom
import copy
# from multiprocessing import Process, Queue

from classes.Weather import Weather
random = SystemRandom()

from classes.Utils import CIRCUIT, ms_to_time

DRY_COMPOUNDS:list = ['Soft', 'Medium', 'Hard']
PITSTOP = [True, False]

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
        self.numLaps = CIRCUIT[circuit]['Laps']+1
        self.iterations = iterations
        self.car:Car = car
        weather = Weather(circuit)
        self.weather = weather.get_weather_list()
        
        self.mu_decay = 0.99
        self.sigma_decay = 0.99

        #self.availableTyres = self.get_available_tyres(circuit) #DISCONTINUED


    ### DISCONTINUED
    # def get_available_tyres(self, circuit:str):
    #     available_tyres = {'Soft':{'Used': 0, 'New': 0}, 'Medium':{'Used': 0, 'New': 0}, 'Hard':{'Used': 0, 'New': 0}, 'Inter':{'Used': 0, 'New': 0}, 'Wet':{'Used': 0, 'New': 0}}
    #     print(f"Please insert tyres available for '{circuit}':")
    #     for tyre in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
    #         #available_tyres[tyre]['Used'] = int(input(f"\t\t\t {tyre} Used: "))
    #         #available_tyres[tyre]['New'] = int(input(f"\t\t\t {tyre} New: "))
    #         available_tyres[tyre]['Used'] = 0
    #         available_tyres[tyre]['New'] = 0
        
    #     return available_tyres

    # DISCONTINUED OR MUST BE REWRITTEN
    # def print(self) -> str:
    #     string = ''
    #     for i in range(self.population):
    #         string+=f"---------- Individual {i+1} ----------\n"
    #         for lap in range(self.numLaps):
    #             string+=f"{lap+1}º LapTime: {ms_to_time(self.strategies[i]['LapTime'][lap])} | TyreCompound: {self.strategies[i]['TyreCompound'][lap]} | TyreWear: {self.strategies[i]['TyreWear'][lap]} | FuelLoad: {self.strategies[i]['FuelLoad'][lap]} | PitStop: {self.strategies[i]['PitStop'][lap]} | LapsCompound: {self.strategies[i]['LapsCompound'][lap]}\n"
    #         string+=f"TotalTime: {ms_to_time(self.strategies[i]['TotalTime'])}\n"
    #         string+="\n"

    #     return string
    
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
        return self.car.predict_fuel_weight(initial_fuel, conditions)
    
    def getInitialFuelLoad(self, conditions:list):
        return self.car.predict_starting_fuel(conditions)

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
        if math.isinf(best['TotalTime']):
            best = {'TotalTime':np.inf}
        idx = -1
        best_idx = -1
        for strategy in population:
            idx += 1
            if strategy['TotalTime'] < best['TotalTime']:
                #Check strategy is good
                all_compounds = set(strategy['TyreCompound'])
                if len(all_compounds) > 1 and strategy['FuelLoad'][-1] >= 1:
                    best_idx = idx 
                    best = strategy
        
        print(f"Best strategy index: {best_idx}")
        return best, best['TotalTime']


    def startSolver(self,):
        fitness_values = list()
        threshold_quantile = 0.3
        counter = 0

        # initial population of random bitstring
        population = self.initSolver()
        
        # keep track of best solution
        #best, best_eval = self.getBest(population)
        
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
                #selected = self.selection(population=population,percentage=0.4)
                #selected = self.selection_dynamic_penalty(population=population)
                selected = self.selection_dynamic_penalty(step=gen+1,population=population,threshold_quantile=threshold_quantile, best = best_eval)
                
                # Create the next generation
                children = [parent for parent in selected]

                if len(selected) > self.population:
                    selected = selected[:self.population]

                if len(selected) > 1:
                    for i in range(0, len(selected)-2, 2): # why not 1? I know there will be 2*population length - 2 but maybe it is good
                        # Get selected parents in pairs
                        p1, p2 = selected[i], selected[i+1]
                        # Crossover 
                        for c in self.crossover(p1, p2):
                            # Mutation
                            for l in self.mutation(c):
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
                print(f'Generation {gen+1}/{self.iterations} best overall: {ms_to_time(best_eval)}, best of generation: {ms_to_time(temp_best_eval)}, non random individuals: {round(non_random_pop/self.population,2)}% and threshold is {threshold_quantile} and counter = {counter}/{(self.iterations)//75}')

                if (counter/((self.iterations)//75)) > 1:
                    threshold_quantile = round(threshold_quantile - 0.01,2)

                if counter >= 100:
                    print("Stopping because of counter (Stuck in local minima or global optimum found)")
                    break
                if counter >= (self.iterations)//75:
                    #threshold_quantile = round(random.random(),2)
                    half_pop = self.population//2
                    population = population[:self.population]
                    idx = random.randint(1, half_pop)
                    for i in range(half_pop+idx, self.population):
                        population[i] = self.randomChild()
                
                if threshold_quantile <= 0.01:
                    while threshold_quantile <= 0.05:
                        threshold_quantile = 0.3

                if non_random_pop/self.population == 0.0 or non_random_pop == 1:
                    print(f"No valid individuals for generating new genetic material({non_random_pop}), stopping")
                    break

                prev_pop = copy.deepcopy(population)
                
        except KeyboardInterrupt:
            pass 
        
        return best, best_eval, {key+1:val for key, val in enumerate(fitness_values)}  

    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategies.append(self.randomChild())

        return strategies

    # def randomChild(self):
    #     strategy = {'TyreCompound': [], 'TyreAge':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'LapsCompound':[], 'Weather':self.weather.copy(), 'TotalTime': np.inf}

    #     weather = [strategy['Weather'][0]]

    #     ### Get a random compound and verify that we can use it, if so we update the used compounds list and add the compound to the strategy
    #     compound = self.randomCompound(weather[0])
    #     strategy['TyreCompound'].append(compound)

    #     ### If the compound is used we put a tyre wear of 2 laps (if it is used but available the compound has been used for 2/3 laps.
    #     ### However, 2 laps out of 3 are done very slowly and the wear is not as the same of 3 laps)
    #     ### If the compound is new tyre wear = 0
    #     tyresAge = 0 
    #     strategy['TyreAge'].append(tyresAge)
    #     strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))

    #     ### The fuel load can be inferred by the coefficient of the fuel consumption, we add a random value between -10 and 10 to get a little variation
    #     initialFuelLoad = self.getInitialFuelLoad(conditions=self.weather)+random.randint(-10,10)
    #     strategy['FuelLoad'].append(initialFuelLoad)

    #     ### At first lap the pit stop is not made (PitStop list means that at lap i^th the pit stop is made at the beginning of the lap)
    #     strategy['PitStop'].append(False)

    #     ### Compute lapTime
    #     strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=0, fuel_load=initialFuelLoad, conditions=weather, drs=False, pitStop=False))

    #     ### Add the laps counter of the compound on the car (+1 for all lap we complete with the compound, set to 0 when changing compound)
    #     strategy['LapsCompound'].append(tyresAge)

    #     ### For every lap we repeat the whole process
    #     for lap in range(1,self.numLaps):
    #         weather = strategy['Weather'][:lap+1]

    #         ### The fuel does not depend on the compound and/or pit stops => we compute it and leave it here
    #         fuelLoad = self.getFuelLoad(initial_fuel=initialFuelLoad, conditions=weather)
    #         strategy['FuelLoad'].append(fuelLoad)

    #         ### With probability of the tyre wear we make a pit stop (if tyre wear low we have low probability, else high)
    #         if changeTyre(strategy['TyreWear'][lap-1]):
    #             ### We have the case of the pitStop => new tyre (can be the same compound type of before!!!)
    #             compound = self.randomCompound(weather[lap])
    #             tyresAge = 0
    #             pitStop = True
    #             strategy['NumPitStop'] += 1
    #         else:
    #             ### No pitstop => same tyres of lap before
    #             compound = strategy['TyreCompound'][lap-1]
    #             tyresAge += 1
    #             pitStop = False
                
    #         strategy['TyreAge'].append(tyresAge)
    #         strategy['TyreCompound'].append(compound)
    #         strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))
    #         strategy['PitStop'].append(pitStop)
    #         strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad, conditions=weather, drs=False, pitStop=pitStop))
    #         strategy['LapsCompound'].append(tyresAge)
            
        
    #     ### Check that all constraints are ok and if so compute the Total Time
    #     allCompounds = set(strategy['TyreCompound'])
    #     if len(allCompounds) > 1 and strategy['FuelLoad'][-1] >= 1:
    #         strategy['TotalTime'] = sum(strategy['LapTime'])
    #         return strategy
    #     else:
    #         return strategy

    def randomChild(self):
        strategy = {'TyreCompound': [], 'TyreAge':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'LapsCompound':[], 'Weather':self.weather.copy(), 'TotalTime': np.inf}

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
        initialFuelLoad = self.getInitialFuelLoad(conditions=self.weather)+random.randint(-10,10)
        strategy['FuelLoad'].append(initialFuelLoad)

        ### At first lap the pit stop is not made (PitStop list means that at lap i^th the pit stop is made at the beginning of the lap)
        strategy['PitStop'].append(False)

        ### Compute lapTime
        strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=0, fuel_load=initialFuelLoad, conditions=weather, drs=False, pitStop=False))

        ### Add the laps counter of the compound on the car (+1 for all lap we complete with the compound, set to 0 when changing compound)
        strategy['LapsCompound'].append(tyresAge)

        ### For every lap we repeat the whole process
        for lap in range(1,self.numLaps):
            weather = strategy['Weather'][:lap+1]

            ### The fuel does not depend on the compound and/or pit stops => we compute it and leave it here
            fuelLoad = self.getFuelLoad(initial_fuel=initialFuelLoad, conditions=weather)
            strategy['FuelLoad'].append(fuelLoad)

            ### With probability of the tyre wear we make a pit stop (if tyre wear low we have low probability, else high)
            if random.random() > 0.5: #changeTyre(strategy['TyreWear'][lap-1]):
                ### We have the case of the pitStop => new tyre (can be the same compound type of before!!!)
                compound = self.randomCompound(weather[lap])
                tyresAge = 0
                pitStop = True
                strategy['NumPitStop'] += 1
            else:
                ### No pitstop => same tyres of lap before
                compound = strategy['TyreCompound'][lap-1]
                tyresAge += 1
                pitStop = False
                
            strategy['TyreAge'].append(tyresAge)
            strategy['TyreCompound'].append(compound)
            strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))
            strategy['PitStop'].append(pitStop)
            strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad, conditions=weather, drs=False, pitStop=pitStop))
            strategy['LapsCompound'].append(tyresAge)
            
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

    # def selection_dynamic_penalty(sel, population):
    #     sortedPopulation = sorted(population, key=lambda x: x['TotalTime'])
    #     penalty= []
    #     selected_first_quantile = []
    #     selected_second_quantile = []

    #     for p in sortedPopulation:
    #         penalty.append(p['TotalTime'] - sortedPopulation[0]['TotalTime'])

    #     treshold_first_quantile = np.quantile(penalty, 0.25)
    #     treshold_second_quantile = np.quantile(penalty, 0.50)
    #     print('PENALTY -> The best value is : ', penalty[0], ' and then ', penalty[1], ', The treshold is : ', treshold_second_quantile, ', while the max value is ',penalty[-1] )
    #     for i in range(0, len(penalty)):
    #         if penalty[i] < treshold_second_quantile:
    #             selected_second_quantile.append(sortedPopulation[i])    
    #         if penalty[i] < treshold_first_quantile:
    #             selected_first_quantile.append(sortedPopulation[i])

    #     if len(selected_second_quantile) > len(population)*0.5:
    #         print('solo primo quantile')
    #         return selected_first_quantile
    #     else: 
    #         print('anche secondo quantile')
    #         return selected_second_quantile
    
    def selection_dynamic_penalty(self, step:int, population:list, threshold_quantile:float, best:int):
        sortedPopulation = sorted(population, key=lambda x: x['TotalTime'])
        #best = sortedPopulation[0]['TotalTime']
        
        deltas = [abs(x['TotalTime'] - best) for x in sortedPopulation]
        max_delta = max(deltas)

        penalty= [delta/max_delta for delta in deltas]

        alpha = np.exp(1+(1/self.iterations)*step)
        penalty = [p*alpha for p in penalty]

        quantile = np.quantile(penalty, threshold_quantile)

        for p, pop in zip(penalty, sortedPopulation):
            if pop['NumPitStop'] < 1:
                p *= alpha
                if p == 0.0:
                    p = np.exp(alpha)
            if pop['FuelLoad'][-1] < 1:
                p *= np.exp(2-pop['FuelLoad'][-1])
                if p == 0.0:
                    p = np.exp(2-pop['FuelLoad'][-1])
        
        #sortedPenalty, sortedPopulation = [x for _, x in sorted(zip(penalty, sortedPopulation))]
        #selected = [x for idx, x in enumerate(sortedPopulation) if penalty[idx] < quantile]
        for idx, x in enumerate(sortedPopulation):
            x['Penalty'] = penalty[idx]
        sortedPopulation = sorted(sortedPopulation, key=lambda x: x['Penalty'])
        selected = [x for idx, x in enumerate(sortedPopulation) if x['Penalty'] < quantile]
        for x in sortedPopulation:
            x.pop('Penalty')

        if len(selected) <= 1:
            threshold_quantile+=0.01
            return self.selection_dynamic_penalty(step, population, threshold_quantile, best)
        
        
        return selected

    def mutation_fuel_load(self, child:dict, ):
        new_fuel = child['FuelLoad'][0]+random.randint(-10,10)

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
        ### Initialize lap from which we will mutate
        lap = random.randint(1,self.numLaps-1)
        mutationCompound = self.randomCompound(child['Weather'][lap])

        ### Until new pitStop we change the compound and then correct strategy will make everything ok
        #pitStop = child['PitStop'][lap]
        child['TyreCompound'][lap] = mutationCompound
        lap += 1
        if lap == self.numLaps:
            return self.correct_strategy(child)
             
        pitStop = child['PitStop'][lap]
        while not pitStop and lap < self.numLaps-1:
            child['TyreCompound'][lap] = mutationCompound
            pitStop = child['PitStop'][lap +1 ]
            lap += 1
        
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
        index = 0
        for lap in range(1, self.numLaps):
            if child['PitStop'][lap-1] == True:
                numPitStops +=1
                if numPitStops == numRandomPitStop:
                    child['PitStop'][lap-1] = False
                    child['NumPitStop'] -= 1
                    index = lap-1

        return self.correct_strategy_pitstop(strategy=child, indexPitStop=index)

    def correct_strategy_pitstop(self, strategy:dict, indexPitStop: int):
        tyresAge = 0
        if indexPitStop == 0:
            strategy['TyreCompound'][0] = strategy['TyreCompound'][1]
            strategy['TyreWear'][0] = self.getTyreWear(strategy['TyreCompound'][0], 0)
            strategy['LapTime'][0] = self.getLapTime(strategy['TyreCompound'][0], tyresAge, i, strategy['FuelLoad'][0], [strategy['Weather'][0]], False, False)
            for i in range(1, self.numLaps):
                    if strategy['PitStop'][i] == False:
                        tyresAge += 1
                        strategy['TyreAge'][i] = tyresAge
                        strategy['TyreCompound'][i] = strategy['TyreCompound'][i-1]
                        strategy['TyreWear'][i] = self.getTyreWear(strategy['TyreCompound'][i], tyresAge)
                        strategy['LapTime'][i] = self.getLapTime(strategy['TyreCompound'][i], tyresAge, i, strategy['FuelLoad'][i], strategy['Weather'][:i], False, False)
                    else:
                        tyresAge = 0
        else:
            i = indexPitStop
            # while strategy['TyreCompound'][i] == strategy['TyreCompound'][i-1] and i > 1:
            #     tyresAge += 1
            #     i -= 1
            tyresAge = strategy['TyreAge'][i-1]
            for i in range(indexPitStop, self.numLaps):
                    if strategy['PitStop'][i] == False:
                        tyresAge += 1
                        strategy['TyreAge'][i] = tyresAge
                        strategy['TyreCompound'][i] = strategy['TyreCompound'][i-1]
                        strategy['TyreWear'][i] = self.getTyreWear(strategy['TyreCompound'][i], tyresAge)
                        strategy['LapTime'][i] = self.getLapTime(strategy['TyreCompound'][i], tyresAge, i, strategy['FuelLoad'][i], strategy['Weather'][:i], False, False)
                    else:
                        tyresAge = 0

        strategy['NumPitStop'] = sum([x for x in strategy['PitStop'] if x])
        
        stints = set(strategy['TyreCompound'])
        if len(stints) > 1 and strategy['FuelLoad'][-1] >= 1:
            strategy['TotalTime'] = sum(strategy['LapTime'])
            return strategy
        
        return self.randomChild()

    def mutation(self,child:dict) -> list:
        childCompound = copy.deepcopy(child)
        childPitStop = copy.deepcopy(child)
        childFuelLoad = copy.deepcopy(child)
        children = []

        if random.random() < self.sigma:
            children.append(self.mutation_compound(childCompound))
        
        if random.random() < self.sigma:
            children.append(self.mutation_pitstop(childPitStop))

        if random.random() < self.sigma:
            children.append(self.mutation_fuel_load(childFuelLoad))
        
        return children
    
    def crossover(self, p1:dict, p2:dict,):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if random.random() < self.mu:
            # select crossover point that is not on the end of the string
            pt = random.randint(1, len(p1['TyreCompound'])-2)
            # perform crossover

            ### {'TyresAvailability': self.availableTyres.copy(), 'TyreCompound': [], 'TyreStatus':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'LapsCompound':[], 'Weather':[], 'TotalTime': np.inf}
            c1 = {'TyresAvailability': copy.deepcopy(self.availableTyres),'TyreCompound': p1['TyreCompound'][:pt]+p2['TyreCompound'][pt:], 'TyreAge':p1['TyreAge'][:pt]+p2['TyreAge'][pt:], 'TyreWear': p1['TyreWear'][:pt]+p2['TyreWear'][pt:], 'FuelLoad': p1['FuelLoad'][:pt]+p2['FuelLoad'][pt:], 'PitStop': p1['PitStop'][:pt]+p2['PitStop'][pt:], 'LapTime': p1['LapTime'][:pt]+p2['LapTime'][pt:], 'LapsCompound': p1['LapsCompound'][:pt]+p2['LapsCompound'][pt:], 'Weather':p1['Weather'], 'NumPitStop': p1['NumPitStop'], 'TotalTime': p1['TotalTime']}
            c2 = {'TyresAvailability': copy.deepcopy(self.availableTyres),'TyreCompound': p2['TyreCompound'][:pt]+p1['TyreCompound'][pt:], 'TyreAge':p2['TyreAge'][:pt]+p1['TyreAge'][pt:], 'TyreWear': p2['TyreWear'][:pt]+p1['TyreWear'][pt:], 'FuelLoad': p2['FuelLoad'][:pt]+p1['FuelLoad'][pt:], 'PitStop': p2['PitStop'][:pt]+p1['PitStop'][pt:], 'LapTime': p2['LapTime'][:pt]+p1['LapTime'][pt:], 'LapsCompound': p2['LapsCompound'][:pt]+p1['LapsCompound'][pt:], 'Weather':p2['Weather'], 'NumPitStop': p2['NumPitStop'], 'TotalTime': p2['TotalTime']}
            
            return [self.correct_strategy(c1), self.correct_strategy(c2)]
        
        return []

    def correct_strategy(self, strategy:dict):
        initialFuelLoad = strategy['FuelLoad'][0]
        pitStopCounter = 0
        
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
            else:
                tyresAge += 1
                
            tyreWear = self.getTyreWear(compound=compound, lap=tyresAge)
            strategy['PitStop'][lap] = pitStop
            strategy['TyreWear'][lap] = tyreWear
            strategy['TyreAge'][lap] = tyresAge
            strategy['LapsCompound'][lap] = tyresAge
            weather = strategy['Weather'][:lap]
            strategy['LapTime'][lap] = self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, fuel_load=fuelLoad,conditions=weather, drs=False, pitStop=pitStop)
            pass

        strategy['NumPitStop'] = pitStopCounter

        ### Check that all constraints are ok and if so compute the Total Time
        allCompounds = set(strategy['TyreCompound'])
        if len(allCompounds) > 1 and strategy['FuelLoad'][-1] >= 1:
            strategy['TotalTime'] = sum(strategy['LapTime'])

        return strategy

    def fillRemainings(self, lap:int, strategy:dict):
        compound = strategy['TyreCompound'][lap-1]
        fuelLoad = strategy['FuelLoad'][lap-1]
        for _ in range(lap, self.numLaps):
            strategy['TyreAge'].append(1)
            strategy['TyreCompound'].append(compound)
            strategy['TyreWear'].append({'FL':1.0, 'FR':1.0, 'RL':1.0, 'RR':1.0})
            strategy['FuelLoad'].append(1000)
            strategy['LapsCompound'].append(0)
            strategy['PitStop'].append(False)
            strategy['LapTime'].append(0)

        return strategy

    def build_tree(self, tree:list, temp_tree:list, tyres_age, start_fuel, lap):
        if lap == self.numLaps:
            ###Check if valid
            compound_set = set([x['Compound'] for x in temp_tree])
            if len(compound_set) > 1:
                if temp_tree[-1]['FuelLoad'] >= 0:
                    tree.append({'Strategy':temp_tree.copy(), 'Time': sum([lap['LapTime'] for lap in temp_tree])})
            return
        pitStop_count = sum([x['PitStop'] for x in temp_tree])
        if pitStop_count > 3:
            return
        fuel_load = self.getFuelLoad(start_fuel,self.weather[:lap])
        if self.weather[lap] == "Dry":
            for index, compound in enumerate(['Hard', 'Medium', 'Soft']):
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True, False]:
                        node = {'Compound':compound, 'TyresAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        self.build_tree(tree, temp_tree,tyres_age+1 if not pitStop else 0, start_fuel, lap+1)
                        temp_tree.pop()
                else:
                    node = {'Compound':compound, 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':True, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=True)}
                    temp_tree.append(node)
                    self.build_tree(tree, temp_tree,0, start_fuel, lap+1)
                    temp_tree.pop()
        
        else:
            for index, compound in enumerate(['Inter', 'Wet']):
                if compound == temp_tree[-1]['Compound']:
                    for pitStop in [True, False]:
                        node = {'Compound':compound, 'TyresAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=pitStop)}
                        temp_tree.append(node)
                        self.build_tree(tree, temp_tree, tyres_age+1 if not pitStop else 0, start_fuel, lap+1)
                        temp_tree.pop()
                    else:
                        node = {'Compound':compound, 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':True, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=lap, fuel_load=fuel_load, conditions=self.weather[:lap], drs=False, pitStop=True)}
                        temp_tree.append(node)
                        self.build_tree(tree, temp_tree, 0, start_fuel, lap+1)
                        temp_tree.pop()

    def lower_bound(self):
        ### Build the solution space as a tree
        temp_tree = []
        tree = []
        start_fuel = self.getInitialFuelLoad(self.weather)
        
        start = time.time()
        
        if self.weather[0] == "Dry":
            for compound in ['Hard', 'Medium', 'Soft']:
                temp_tree.append({'Compound':compound, 'TyresAge':0, 'FuelLoad':start_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=start_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
                self.build_tree(tree, temp_tree, 0, start_fuel, 1)
                print(f"{compound} done in {time.time()-start}")
                temp_tree.pop()
        else:
            for compound in ['Inter', 'Wet']:
                temp_tree.append({'Compound':compound, 'TyresAge':0, 'FuelLoad':start_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=start_fuel, conditions=[self.weather[0]], drs=False, pitStop=False)})
                self.build_tree(tree, temp_tree, 0, start_fuel, 1)
                print(f"{compound} done in {time.time()-start}")
                temp_tree.pop()

        #print(tree)
        sorted_tree = sorted(tree, key=lambda k: k['Time'])
        for lap in sorted_tree[0]['Strategy']:
            print(lap)
        print(f"With a total time of {sorted_tree[0]['Time']}")
        print(f"Computed in time {time.time()-start}")
        
        ### Find the best solution
        return sorted_tree[0]