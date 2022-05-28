import math
import numpy as np
from classes.Car import Car
from random import SystemRandom
random = SystemRandom()

from classes.Utils import get_basic_logger, STINTS, ms_to_time

log = get_basic_logger('Genetic')

DRY_STINTS:list = ['Soft', 'Medium', 'Hard']
PITSTOP = [True, False]

def overLimit(values, limit):
    if not isinstance(values, list):
        values = list(values)
    for val in values:
        if val >= limit:
            return True
    
    return False

class GeneticSolver:
    def __init__(self, population:int=2, mutation_pr:float=0.0, crossover_pr:float=0.0, iterations:int=1, numLaps:int=60, car:Car=None) -> None:
        self.bestLapTime = car.getBestLapTime()
        self.fuel_coeff = car.fuel_lose
        self.coeff = car.wear_coeff
        self.pitStopTime = 20000
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.numLaps = numLaps
        self.iterations = iterations
        self.car:Car = car
        #self.numPitStop = 0
        

        self.mu_decay = 0.99
        self.sigma_decay = 0.99


    def print(self) -> None:
        string = ''
        for i in range(self.population):
            string+=f"---------- Individual {i+1} ----------\n"
            for lap in range(self.numLaps):
                string+=f"{lap+1}º LapTime: {ms_to_time(self.strategies[i]['LapTime'][lap])} | TyreStint: {self.strategies[i]['TyreStint'][lap]} | TyreWear: {self.strategies[i]['TyreWear'][lap]} | FuelLoad: {self.strategies[i]['FuelLoad'][lap]} | PitStop: {self.strategies[i]['PitStop'][lap]} | LapsStint: {self.strategies[i]['LapsStint'][lap]}\n"
            string+=f"TotalTime: {ms_to_time(self.strategies[i]['TotalTime'])}\n"
            string+="\n"

        return string
    
    def getTyreWear(self, stint:str, lap:int):
        return self.car.getTyreWear(stint, lap)
    
    def getFuelLoad(self, lap:int, initial_fuel:float) :
        return self.car.getFuelLoad(lap, initial_fuel)
    
    def getWearTimeLose(self, stint:str, lap:int):
        return self.car.getWearTimeLose(stint, lap)
    
    def getFuelTimeLose(self, lap:int):
        return self.car.getFuelTimeLose(lap)

    def lapTime(self, stint:str, lap:int, fuel_load:float, pitStop:bool) -> int:
        if fuel_load < 0:
            fuel_load = 0

        time = self.bestLapTime + self.getWearTimeLose(stint, lap) + self.getFuelTimeLose(lap)
        if pitStop:
            time += self.pitStopTime
        return round(time) 
    
    def count_pitstop(self, strategy:dict):
        num_pitstop = 0
        for i in range(self.numLaps -1):
            if strategy['PitStop'][i] == True:
                num_pitstop +=1
        return num_pitstop
    
    def countLapsStint(self, strategy: dict):
        if len(strategy['LapsStint']) == 0:
            strategy['LapsStint'].append(1)
            for i in range(1, self.numLaps-1):
                if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
                    strategy['LapsStint'].append(strategy['LapsStint'][i-1] + 1)
                else:
                    strategy['LapsStint'].append(1)
        #print(strategy['LapsStint'])
        else:
            strategy['LapsStint'][0] = 1
            for i in range(1, self.numLaps-1):
                if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
                    strategy['LapsStint'][i] = strategy['LapsStint'][i-1] + 1
                else:
                    strategy['LapsStint'][i] = 1
        return strategy


    def correct_strategy(self, strategy:dict):
        for i in range(1,self.numLaps):
            stint = strategy['TyreStint'][i]
            if strategy['TyreStint'][i-1] != stint or strategy['PitStop'][i] == True:
                strategy['PitStop'][i] = True
                strategy['TyreWear'][i] = self.getTyreWear(stint, 0)
                strategy['LapTime'][i] = self.lapTime(stint, i, strategy['FuelLoad'][i], True)
                j = i+1
                while j < self.numLaps and (strategy['TyreWear'][j] == stint or strategy['PitStop'][j] == False):
                    strategy['TyreWear'][j] = self.getTyreWear(strategy['TyreStint'][j], j-i)
                    strategy['PitStop'][j] = False
                    strategy['LapTime'][j] = self.lapTime(stint, j-i, strategy['FuelLoad'][j], False)
                    j+=1
        
        strategy['numPitStop'] = self.count_pitstop(strategy)
        self.countLapsStint(strategy)
        stints = set(strategy['TyreStint'])
        if len(stints) > 0 and strategy['FuelLoad'][-1] >= 1:
            strategy['TotalTime'] = sum(strategy['LapTime'])
        else:
            strategy['TotalTime'] = np.inf


    def correct_strategy_pitstop(self, strategy:dict, indexPitStop: int):
        if indexPitStop == 0:
            strategy['TyreStint'][0] = strategy['TyreStint'][1]
            strategy['TyreWear'][0] = self.getTyreWear(strategy['TyreStint'][i], 0)
            strategy['LapTime'][0] = self.lapTime(strategy['TyreStint'][i], 0, strategy['FuelLoad'][0], False)
            for j in range(1, self.numLaps -1):
                if strategy['PitStop'][j] == False:
                    strategy['TyreStint'][j] = strategy['TyreStint'][j-1]
                    strategy['TyreWear'][j] = self.getTyreWear(strategy['TyreStint'][j], j)
                    strategy['LapTime'][j] = self.lapTime(strategy['TyreStint'][j], j, strategy['FuelLoad'][j], False)
        else:
            for i in range(indexPitStop, self.numLaps-1):
                    if strategy['PitStop'][i] == False:
                        strategy['TyreStint'][i] = strategy['TyreStint'][i-1]
                        strategy['TyreWear'][i] = self.getTyreWear(strategy['TyreStint'][i], self.countLapsStint(strategy)['LapsStint'][i])
                        strategy['LapTime'][i] = self.lapTime(strategy['TyreStint'][i], i, strategy['FuelLoad'][i], False)
                

        #log.debug(strategy['LapTime'][indexPitStop])
        strategy['numPitStop'] = self.count_pitstop(strategy)
        #self.countLapsStint(strategy)
        stints = set(strategy['TyreStint'])
        if len(stints) > 0 and strategy['FuelLoad'][-1] >= 1:
            strategy['TotalTime'] = sum(strategy['LapTime'])
        else:
            strategy['TotalTime'] = np.inf



    def mutation_stint(self, child:dict):
        if random.random() < (self.sigma)/2:
            mutationStint = random.choice(DRY_STINTS)
            iRandom = random.randint(0,len(child['TyreStint'])-1)
            #child['TyreStint'][iRandom] = mutationStint
            for i in range(iRandom, len(child['TyreStint'])-1):
                if child['PitStop'][i] == False:
                    child['TyreStint'][i] = mutationStint
                else:
                    break
            #for i in range(len(child['TyreStint'])):
            #    # check for a mutation
            #    if random.random() < self.sigma:
            #    	# flip the bit
            #        if child[]
            #        child['TyreStint'][i] = random.choice(DRY_STINTS)
            self.correct_strategy(child) 
            return child

    def mutation_pitstop(self,child:dict):
        num_PitStop = 0
        if random.random() < self.sigma:
            if child['numPitStop'] != 0:
                pitstop_random = random.randint(1,child['numPitStop'])
                for i in range(0, len(child['PitStop'])-1):
                    if child['PitStop'][i] == True:
                        num_PitStop +=1
                        if num_PitStop == pitstop_random:
                            child['PitStop'][i] = False
                            indexPitStop = i
                            self.correct_strategy_pitstop(child, indexPitStop)
                            break
                        
                return child
            

    def mutation(self,child:dict):
        numMutation = 0
        initialChildStint = child
        initialChildPitStop = child
        children = []
        if random.random() < self.sigma:
            children.append(self.mutation_stint(initialChildStint))
        
        if random.random() < self.sigma:
            children.append(self.mutation_pitstop(initialChildPitStop))
        
        return children


        #if random.random() < self.sigma:
        #    mutationStint = random.choice(DRY_STINTS)
        #    iRandom = random.randint(0,len(child['TyreStint'])-1)
        #    #child['TyreStint'][iRandom] = mutationStint
        #    for i in range(iRandom, len(child['TyreStint'])-1):
        #        if child['PitStop'][i] == False:
        #            child['TyreStint'][i] = mutationStint
        #        else:
        #            break
        #for i in range(len(child['TyreStint'])):
		#    # check for a mutation
        #    if random.random() < self.sigma:
		#    	# flip the bit
        #        if child[]
        #        child['TyreStint'][i] = random.choice(DRY_STINTS)  
        #pitstop_random = random.randint(1,child['numPitStop'])

        #self.correct_strategy(child)        
    
    
    def crossover(self, p1:dict, p2:dict,):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if random.random() < self.mu:
            # select crossover point that is not on the end of the string
            pt = random.randint(1, len(p1['TyreStint'])-2)
            # perform crossover
            c1 = {'TyreStint': p1['TyreStint'][:pt]+p2['TyreStint'][pt:], 'TyreWear': p1['TyreWear'][:pt]+p2['TyreWear'][pt:], 'FuelLoad': p1['FuelLoad'][:pt]+p2['FuelLoad'][pt:], 'PitStop': p1['PitStop'][:pt]+p2['PitStop'][pt:], 'LapTime': p1['LapTime'][:pt]+p2['LapTime'][pt:], 'LapsStint': [], 'TotalTime': None}
            c2 = {'TyreStint': p2['TyreStint'][:pt]+p1['TyreStint'][pt:], 'TyreWear': p2['TyreWear'][:pt]+p1['TyreWear'][pt:], 'FuelLoad': p2['FuelLoad'][:pt]+p1['FuelLoad'][pt:], 'PitStop': p2['PitStop'][:pt]+p1['PitStop'][pt:], 'LapTime': p2['LapTime'][:pt]+p1['LapTime'][pt:], 'LapsStint': [], 'TotalTime': None}
            
                
            self.count_pitstop(c1)
            self.count_pitstop(c2)
            self.countLapsStint(c1)
            self.countLapsStint(c2)
            self.correct_strategy(c1)
            self.correct_strategy(c2)
        return [c1, c2]

    def selection(self,population, scores):
        dict_map = {}
        temp = [score for score in scores if not math.isinf(score)]

        for index,score in enumerate(scores):
            new_idx = score
            
            if math.isinf(new_idx):
                while dict_map.get(new_idx) is not None:
                    new_idx = random.randint(3600000,4000000)
            
            dict_map[new_idx] = population[index]

        sorted_dict = sorted(dict_map.items(), key=lambda x: x[0])
        
        population_selected = []
        for i in range(0, round((40/100)*self.population)):
            population_selected.append(sorted_dict[i][1])
            
        return population_selected

    def randomChild(self,):
        strategy = {'TyreStint': [], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'LapsStint':[], 'TotalTime': np.inf}
            
        strategy['TyreStint'].append(random.choice(DRY_STINTS))
        strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][0], 0))
        strategy['FuelLoad'].append(random.randint(100, 120))
        strategy['PitStop'].append(False)
        strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][0], 0, strategy['FuelLoad'][0], False))
        strategy['LapsStint'].append(0)
        
            

        lap_count = 0
        for i in range(1,self.numLaps):
            lap_count += 1
            strategy['TyreStint'].append(random.choice(DRY_STINTS))
            if strategy['TyreStint'][i] == strategy['TyreStint'][i-1] and strategy['PitStop'][i-1] == False:
                strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][i], lap_count))
                strategy['PitStop'].append(False)
                strategy['LapsStint'].append(0)
                if overLimit(strategy['TyreWear'][i].values(), 80):
                    strategy['PitStop'].append(True)
                else:
                    strategy['PitStop'].append(False)
            else:
                lap_count = 0
                strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][i], lap_count))
                strategy['PitStop'].append(True)
                strategy['LapsStint'].append(0)

            strategy['FuelLoad'].append(self.getFuelLoad(i, strategy['FuelLoad'][0]))
            strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][i], i, strategy['FuelLoad'][i], strategy['PitStop'][i]))
        
        strategy['NumPitStop'] = self.count_pitstop(strategy)
        #print(strategy)
        self.countLapsStint(strategy)
        #print(strategy)
        stints = set(strategy['TyreStint'])
        
        if len(stints) > 0 and strategy['FuelLoad'][-1] >= 1:
            strategy['TotalTime'] = sum(strategy['LapTime'])
            self.correct_strategy(strategy)
        else:
            return self.randomChild()
        
        return strategy

    # def initPopulation(self,):
    #     strategy = {'TyreStint': [], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'TotalTime': np.inf}
    #     initial_stint = random.choice(DRY_STINTS)
    #     strategy['TyreStint'].append(initial_stint)
    #     strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][0], 0))
    #     strategy['FuelLoad'].append(random.randint(270,300))
    #     strategy['PitStop'].append(False)
    #     strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][0], i, strategy['FuelLoad'][0], False))
        
    #     for i in range(1,self.numLaps+1):
    #         if strategy['PitStop'][i-1] == True:
    #             initial_stint = random.choice(DRY_STINTS)
    #             strategy['TyreStint'].append(initial_stint)
    #         else: 
    #             strategy['TyreStint'].append(initial_stint)
    #         if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
    #             strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][i], i))
    #             if overLimit(strategy['TyreWear'][i].values(), 80):
    #                 strategy['PitStop'].append(True)
                    
    #             else:
    #                 strategy['PitStop'].append(False)
    #         else:
    #             strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][i], i))
    #             strategy['PitStop'].append(True)
            
    #         strategy['FuelLoad'].append(self.getFuelLoad(i))
    #         strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][i], i, strategy['FuelLoad'][i], strategy['PitStop'][i]))
        
    #     strategy['NumPitStop'] = self.count_pitstop(strategy)
    #     stints = set(strategy['TyreStint'])

    #     if len(stints) > 0 and strategy['FuelLoad'][-1] > 0:
    #         strategy['TotalTime'] = sum(strategy['LapTime'])
    #     else:
    #         strategy['TotalTime'] = np.inf

    #     return strategy

    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategies.append(self.randomChild())

        return strategies
        
    # 
    #                           Function taken from 
    # https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # 
    def startSolver(self,):
        fitness_values = list()

        # initial population of random bitstring
        pop = self.initSolver()

        # keep track of best solution
        best, best_eval = pop[0], pop[0]['TotalTime']
        
        # enumerate generations
        try:
            for gen in range(self.iterations):
                temp_best, temp_best_eval = 0, pop[0]['TotalTime']

                # evaluate all candidates in the population
                scores = [c['TotalTime'] for c in pop]
                # check for new best solution
                for i in range(self.population):
                    if scores[i] < best_eval:
                        best, best_eval = pop[i], scores[i]
                    if scores[i] < temp_best_eval:
                        temp_best, temp_best_eval = pop[i], scores[i]
                

                # select parents
                selected = self.selection(pop, scores)#[self.selection(pop, scores, x) for x in range(self.population)]
                
                #if len(selected) < self.population:
                #    for _ in range(self.population-len(selected)):
                #        selected.append(self.randomChild())
               
                # create the next generation
                children = list()
                for i in range(0, len(selected)-1):
                    children.append(selected[i])

                for i in range(0, len(selected)-2, 2):
                    # get selected parents in pairs
                    p1, p2 = selected[i], selected[i+1]
                    # crossover and mutation
                    for c in self.crossover(p1, p2):
                        # mutation
                        childrenMutated = self.mutation(c)
                        if len(childrenMutated) != 0:
                            for l in childrenMutated:
                                #log.debug(l)
                                if l is not None:
                                    children.append(l)
                        else:
                            children.append(c)
                        # store for next generation
                        #children.append(c)
                
                if len(children) < self.population:
                    for _ in range(self.population-len(children)):
                        children.append(self.randomChild())
                # replace population
                pop = children
                #log.debug("Population:\n{}".format(pop))

                #self.sigma = self.sigma * self.sigma_decay
                #self.mu = self.mu * self.mu_decay

                fitness_values.append(temp_best_eval)

                #if gen%10:
                log.info(f'Generation {gen+1}/{self.iterations} best overall: {ms_to_time(best_eval)}, best of generation: {ms_to_time(temp_best_eval)}')
                
        except KeyboardInterrupt:
            pass 
        
        return best, best_eval, {key+1:val for key, val in enumerate(fitness_values)}
    
