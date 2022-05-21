import math
from typing import List
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

        self.mu_decay = 1
        self.sigma_decay = 1


    def __str__(self) -> str:
        string = ''
        for i in range(self.population):
            string+=f"---------- Individual {i+1} ----------\n"
            for lap in range(self.numLaps):
                string+=f"{lap+1}º LapTime: {convertMillis(self.strategies[i]['LapTime'][lap])} | TyreStint: {self.strategies[i]['TyreStint'][lap]} | TyreWear: {self.strategies[i]['TyreWear'][lap]} | FuelLoad: {self.strategies[i]['FuelLoad'][lap]} | PitStop: {self.strategies[i]['PitStop'][lap]}\n"
            string+=f"TotalTime: {convertMillis(self.strategies[i]['TotalTime'])}\n"
            string+="\n"

        return string
    
    def getTyreWear(self, stint:str, lap:int, wear:float=0.0):
        return 2.5 if wear == 0 else wear+2.5

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

    def selection(self,population, scores, k):
        dict_map = {}
        temp = [score for score in scores if not math.isinf(score)]
        for index,score in enumerate(scores):
            if math.isinf(score):
                new_idx = random.randint(max(temp)*5, max(temp)*10)
                while dict_map.get(new_idx) is not None:
                    new_idx = random.randint(max(temp)*5, max(temp)*10)
            else:
                new_idx = score

            dict_map[new_idx] = population[index]
        
        #log.debug(f"Index: {index}, Len of dict_map: {len(dict_map.keys())} while population = {self.population}\n{dict_map.keys()}")
        
        sorted_dict = sorted(dict_map.items(), key=lambda x: x[0])

        try:
            return sorted_dict[k][1]
        except IndexError:
            return sorted_dict[-1][1]
        #return sorted.pop()[1]
        # selection_idx = random.randint(0, len(population)-1)
        # for i in range(0,len(population),k-1):
        #     if scores[i] < scores[selection_idx]:
        #         selection_idx = i
        # return population[selection_idx]

    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategy = {'TyreStint': [], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'TotalTime': np.inf}
            
            strategy['TyreStint'].append(random.choice(STINTS))
            strategy['TyreWear'].append(0)
            strategy['FuelLoad'].append(random.randint(120,200))
            strategy['PitStop'].append(False)
            strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][0], strategy['TyreWear'][0], strategy['FuelLoad'][0], False))
            
            for i in range(1,self.numLaps+1):
                strategy['TyreStint'].append(np.random.choice(STINTS))

                if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
                    strategy['TyreWear'].append(float(strategy['TyreWear'][i-1])+2.5)
                    strategy['PitStop'].append(False)
                else:
                    strategy['TyreWear'].append(0)
                    strategy['PitStop'].append(True)
                
                strategy['FuelLoad'].append(strategy['FuelLoad'][i-1]-3)
                strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][i], strategy['TyreWear'][i], strategy['FuelLoad'][i], strategy['PitStop'][i]))

            stints = set(strategy['TyreStint'])
            if len(stints) > 1 and strategy['FuelLoad'][-1] > 0:
                strategy['TotalTime'] = sum(strategy['LapTime'])
            else:
                strategy['TotalTime'] = np.inf

            strategies.append(strategy)

        return strategies
        
    
    # 
    #                           Function taken from 
    # https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
    # 
    def startSolver(self,):
        # initial population of random bitstring
        pop = self.initSolver()

        # keep track of best solution
        best, best_eval = 0, pop[0]['TotalTime']

        # enumerate generations
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
            selected = [self.selection(pop, scores, x) for x in range(self.population)]
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

            self.sigma = self.sigma * self.sigma_decay
            self.mu = self.mu * self.mu_decay

            log.debug(f'Generation {gen+1}/{self.iterations} best overall: {convertMillis(best_eval)}, best of generation: {convertMillis(temp_best_eval)}')
        
        return (best, best_eval)
        

"""

 https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

# objective function
def onemax(x):
	return -sum(x)

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
"""