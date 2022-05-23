import math
import numpy as np
from classes.Car import Car
from random import SystemRandom
random = SystemRandom()

from classes.Utils import get_basic_logger

log = get_basic_logger('Genetic')

STINTS = ['Soft', 'Medium', 'Hard'] #'Inter', 'Wet'

def convertMillis(ms):
    if math.isinf(ms):
        return f"::. (infinite)"
    seconds=(ms/1000)%60
    minutes=(ms/(1000*60))%60
    hours=(ms/(1000*60*60))%24
    ms=ms % 1000
    
    if int(hours) < 1:
        return f"{int(minutes)}:{int(seconds)}.{ms}"
    
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}.{ms}"

class GeneticSolver:
    def __init__(self, population:int=2, mutation_pr:float=0.0, crossover_pr:float=0.0, iterations:int=1, numLaps:int=60, car:Car=None) -> None:
        self.bestLapTime = car.get_best_lap_time()
        self.tyre_lose = car.tyre_lose
        self.fuel_coeff = car.fuel_lose
        self.coeff = car.wear_coeff
        self.pitStopTime = 20000
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.numLaps = numLaps
        self.iterations = iterations
        self.car:Car = car

        self.mu_decay = 0.99
        self.sigma_decay = 0.99


    def print(self) -> None:
        string = ''
        for i in range(self.population):
            string+=f"---------- Individual {i+1} ----------\n"
            for lap in range(self.numLaps):
                string+=f"{lap+1}º LapTime: {convertMillis(self.strategies[i]['LapTime'][lap])} | TyreStint: {self.strategies[i]['TyreStint'][lap]} | TyreWear: {self.strategies[i]['TyreWear'][lap]} | FuelLoad: {self.strategies[i]['FuelLoad'][lap]} | PitStop: {self.strategies[i]['PitStop'][lap]}\n"
            string+=f"TotalTime: {convertMillis(self.strategies[i]['TotalTime'])}\n"
            string+="\n"

        return string
    
    def getTyreWear(self, stint:str, lap:int):
        return self.car.get_tyre_wear(stint, lap)
    
    def getFuelLoad(self, lap:int):
        return min([self.car.fuel[i].predict_fuelload(self.car.fuel[i].get_frame(lap)) for i in range(len(self.car.fuel))])

    def lapTime(self, stint:str, wear:float, fuel_load:float, pitStop:bool) -> int:

        if fuel_load < 0:
            fuel_load = 0

        time = self.bestLapTime + self.tyre_lose[stint] +  self.compute_wear_time_lose(wear_percentage=wear,tyre_compound=stint) + self.fuel_coeff * fuel_load + self.pitStopTime * pitStop
        
        return round(time) 
        

    def compute_wear_time_lose(self, wear_percentage:float, tyre_compound:str):
        if self.coeff[tyre_compound] == (0,0):
            if tyre_compound == 'Medium':
                self.coeff[tyre_compound] = ((self.coeff['Hard'][0]+self.coeff['Soft'][0])/2, (self.coeff['Soft'][1]+self.coeff['Soft'][1])/2)

        lose = self.coeff[tyre_compound][0] + self.coeff[tyre_compound][1] * wear_percentage 

        return lose

    def correct_strategy(self, strategy:dict):

        for i in range(1,self.numLaps):
            stint = strategy['TyreStint'][i]
            if strategy['TyreStint'][i-1] != stint:
                strategy['PitStop'][i] = True
                strategy['TyreWear'][i] = 0
                strategy['LapTime'][i] = self.lapTime(stint, 0, strategy['FuelLoad'][i], True)
                j = i+1
                while j < self.numLaps and strategy['TyreWear'][j] == stint:
                    strategy['TyreWear'][j] = self.getTyreWear(strategy['TyreStint'][j], j, strategy['TyreWear'][j-1])
                    strategy['PitStop'][j] = False
                    strategy['LapTime'][j] = self.lapTime(stint, strategy['TyreWear'][j], strategy['FuelLoad'][j], False)
                    j+=1
        
        stints = set(strategy['TyreStint'])
        if len(stints) > 1 and strategy['FuelLoad'][-1] > 0:
            strategy['TotalTime'] = sum(strategy['LapTime'])
        else:
            strategy['TotalTime'] = np.inf

    
    def mutation(self,child:dict):
        for i in range(len(child['TyreStint'])):
		    # check for a mutation
            if random.random() < self.sigma:
		    	# flip the bit
                child['TyreStint'][i] = random.choice(STINTS)
                
        self.correct_strategy(child)        
    
    def crossover(self, p1:dict, p2:dict,):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if random.random() < self.mu:
            # select crossover point that is not on the end of the string
            pt = random.randint(1, len(p1['TyreStint'])-2)
            # perform crossover
            c1 = {'TyreStint': p1['TyreStint'][:pt]+p2['TyreStint'][pt:], 'TyreWear': p1['TyreWear'][:pt]+p2['TyreWear'][pt:], 'FuelLoad': p1['FuelLoad'][:pt]+p2['FuelLoad'][pt:], 'PitStop': p1['PitStop'][:pt]+p2['PitStop'][pt:], 'LapTime': p1['LapTime'][:pt]+p2['LapTime'][pt:], 'TotalTime': None}
            c2 = {'TyreStint': p2['TyreStint'][:pt]+p1['TyreStint'][pt:], 'TyreWear': p2['TyreWear'][:pt]+p1['TyreWear'][pt:], 'FuelLoad': p2['FuelLoad'][:pt]+p1['FuelLoad'][pt:], 'PitStop': p2['PitStop'][:pt]+p1['PitStop'][pt:], 'LapTime': p2['LapTime'][:pt]+p1['LapTime'][pt:], 'TotalTime': None}
            
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
                    new_idx = random.randint(max(temp)*5, max(temp)*10)
            
            dict_map[new_idx] = population[index]

        sorted_dict = sorted(dict_map.items(), key=lambda x: x[0])

        return [val for _, val in sorted_dict]

    def randomChild(self,):
        strategy = {'TyreStint': [], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'TotalTime': np.inf}
            
        strategy['TyreStint'].append(random.choice(STINTS))
        strategy['TyreWear'].append(0)
        strategy['FuelLoad'].append(random.randint(120,200))
        strategy['PitStop'].append(False)
        strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][0], strategy['TyreWear'][0], strategy['FuelLoad'][0], False))
        
        for i in range(1,self.numLaps+1):
            strategy['TyreStint'].append(np.random.choice(STINTS))
            if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
                strategy['TyreWear'].append(self.getTyreWear(strategy['TyreStint'][i], i))
                strategy['PitStop'].append(False)
            else:
                strategy['TyreWear'].append(0)
                strategy['PitStop'].append(True)
            
            strategy['FuelLoad'].append(self.getFuelLoad(i))
            strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][i], strategy['TyreWear'][i], strategy['FuelLoad'][i], strategy['PitStop'][i]))
        
        stints = set(strategy['TyreStint'])

        if len(stints) > 1 and strategy['FuelLoad'][-1] > 0:
            strategy['TotalTime'] = sum(strategy['LapTime'])
        else:
            strategy['TotalTime'] = np.inf

        return strategy

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
        best, best_eval = 0, pop[0]['TotalTime']

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
                if len(selected) < self.population:
                    for _ in range(self.population-len(selected)):
                        selected.append(self.randomChild())
                # create the next generation
                children = list()
                for i in range(0, self.population, 2):
                    # get selected parents in pairs
                    p1, p2 = selected[i], selected[i+1]
                    # crossover and mutation
                    for c in self.crossover(p1, p2):
                        # mutation
                        self.mutation(c)
                        # store for next generation
                        children.append(c)
                # replace population
                pop = children
                #log.debug("Population:\n{}".format(pop))

                #self.sigma = self.sigma * self.sigma_decay
                #self.mu = self.mu * self.mu_decay

                fitness_values.append(temp_best_eval)

                log.debug(f'Generation {gen+1}/{self.iterations} best overall: {convertMillis(best_eval)}, best of generation: {convertMillis(temp_best_eval)}')
        except KeyboardInterrupt:
            pass 
        return (best, best_eval)
      