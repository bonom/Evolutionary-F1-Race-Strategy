import math, time, copy, os
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm

from classes.Car import Car
random = SystemRandom()

from classes.Utils import CIRCUIT, Log, ms_to_time

BEST_ID = 0

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

class GeneticSolver:
    def __init__(self, population:int=2, mutation_pr:float=0.75, crossover_pr:float=0.5, iterations:int=1, car:Car=None, circuit:str='', save_path:str='') -> None:
        self.circuit = circuit
        self.pitStopTime = CIRCUIT[circuit]['PitStopTime']
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.numLaps = CIRCUIT[circuit]['Laps']
        self.iterations = iterations
        self.car:Car = car

        self.availableID = 0
        self.mapStrategies = dict()

        # For the log file
        self.path = save_path
        self.log = Log(save_path, values={'Circuit':circuit, 'PitStopTime':self.pitStopTime, 'Mutation': mutation_pr, 'Crossover': crossover_pr, 'Population': population, 'Iterations':iterations})
    
    def getTyreWear(self, compound:str, compoundAge:int) -> float:
        return self.car.predict_tyre_wear(compound, compoundAge)

    def getFuelLoad(self, lap:int) -> float:
        return self.car.predict_fuel_weight(lap)
    
    def getInitialFuelLoad(self) -> float:
        return self.car.initial_fuel

    def getWearTimeLose(self, compound:str, lap:int) -> int:
        return self.car.predict_tyre_time_lose(compound, lap)
        
    def getFuelTimeLose(self, lap:int) -> int:
        return self.car.predict_fuel_loss(lap)
    
    def getLapTime(self, compound:str, compoundAge:int, lap:int, pitStop:bool) -> int:
        time = self.car.predict_laptime(tyre=compound, lap=lap, tyresAge=compoundAge,)

        if pitStop:
            time += self.pitStopTime

        #if lap == 0:
        #    time += 2000

        return round(time)     

    def getBest(self, population:list, bests:dict):
        
        for strategy in population:
            self.checkValidity(strategy)
            
            if strategy['Valid']:
                idx = strategy['NumPitStop'] if strategy['NumPitStop'] < 4 else 0
                if strategy['TotalTime'] < bests[idx]['TotalTime']:
                    if idx == 1:
                        global BEST_ID
                        BEST_ID = strategy['ID']
                    bests[idx] = copy.deepcopy(strategy)
                
        return bests#, bests['TotalTime']

    def checkValidity(self, strategy:dict):
        all_compounds = set(strategy['TyreCompound'])
        last_lap_fuel_load = self.getFuelLoad(self.numLaps)

        if len(all_compounds) > 1 and last_lap_fuel_load >= 0 and all([x < 100.0 for x in strategy['TyreWear']]):
            strategy['Valid'] = True
            return True
        
        strategy['Valid'] = False 
        return False

    def get_compressed_version(self, strategy:dict) -> list:
        tyre_compressed_version = list()
        pit_compressed_version = list()
        
        prev_tyre_compound = strategy['TyreCompound'][0]

        for lap in range(1,self.numLaps):
            new_compound = strategy['TyreCompound'][lap]
            pit_stop = strategy['PitStop'][lap]
            
            if new_compound != prev_tyre_compound or lap == self.numLaps - 1 or pit_stop:
                #prev_tyre_age = strategy['TyreAge'][lap-1]+1
                if prev_tyre_compound == 'Soft':
                    prev_tyre_compound = 'S'
                elif prev_tyre_compound == 'Medium':
                    prev_tyre_compound = 'M'
                elif prev_tyre_compound == 'Hard':
                    prev_tyre_compound = 'H'
                
                tyre_compressed_version.append(prev_tyre_compound)
                
                if lap != self.numLaps - 1:
                    pit_compressed_version.append(str(lap+1)+'')
                
                prev_tyre_compound = new_compound

        return np.array(tyre_compressed_version), np.array(pit_compressed_version)

    def check_equality(self, strategy1:dict, strategy2:dict) -> bool:
        if strategy1['ID'] == strategy2['ID']:
            return True
        
        for key in strategy1.keys():
            if key != 'Parent' and key != 'ID' and strategy1[key] != strategy2[key]:
                return False
    
        return True

    def run(self):
        self.mapStrategies = dict()
        self.availableID = 0
        
        start_timer = time.time()

        #fitness_values = dict()
        # stuck_counter = 0
        boxplot_list = list()
        #prev = {}

        # initial population of random bitstring
        population = self.initSolver()
        best = {0: {'TotalTime':np.inf}, 1:{'TotalTime':np.inf}, 2:{'TotalTime':np.inf}, 3:{'TotalTime':np.inf}}

        #print(f"\n-------------------------------------------------------------\nData for '{self.circuit}':\n\nPopulation = {self.population}\nIterations = {self.iterations}\nMutation = {self.sigma}\nCrossover = {self.mu}\n-------------------------------------------------------------\n")
        
        # enumerate generations
        try:
            bar = tqdm(range(self.iterations))
            for gen in bar:   
                # Checking if there are duplicates, if so, we remove them
                to_pop = []
                for i in range(0, len(population)-1):
                    for j in range(i+1, len(population)):
                        if self.check_equality(strategy1=population[i], strategy2=population[j]) and j not in to_pop:
                            to_pop.append(j)

                # Removing duplicates by sorted indexes would return an error (we are changing length of the list) so we remove them in reversed order
                to_pop = sorted(to_pop, reverse=True)
                for i in to_pop:
                    population.pop(i)
                
                # Gathering the first solution from population at gen^th generation
                best = self.getBest(population, best)
                #_, temp_best_eval = self.getBest(population)

                # Storing data for boxplot
                boxplot_list = boxplot_insert(boxplot_list, population)

                # Select parents
                parents = self.selection_dynamic_penalty(step=gen+1,population=copy.deepcopy(population),threshold_quantile=1/5)#(len(population)//5)/len(population))
                
                ### Check if there is one with one pit and one with two
                found = False
                for p in parents:
                    if p['NumPitStop'] == 1 and p['Valid']:
                        found = True
                        break
                
                if not found and not math.isinf(best[1]['TotalTime']):
                    print("No one with one pit stop found")
                
                for parent in parents:
                    #sel['Generation'] = gen
                    if parent['ID'] not in self.mapStrategies.keys():
                        tyres, pits = self.get_compressed_version(copy.deepcopy(parent))
                        self.mapStrategies[parent['ID']] = {'Strategy_Tyre': tyres, 'Strategy_Pit':pits, 'Generation':gen, 'Parent':parent['Parent'], 'NumPitStop':parent['NumPitStop']}
                    
                children = copy.deepcopy(parents)

                for i in range(0, len(parents)-1, 2): 
                    #p1, p2 = copy.deepcopy(parents[i]), copy.deepcopy(parents[i+1])

                    #for c in self.crossover(p1, p2):
                    #    children.append(c)
                    
                    for l in self.mutation(parents[i]):
                        l['Generation'] = gen
                        l['Parent'] = parents[i]['ID']
                        children.append(l)
                        #self.mapStrategies[l['ID']] = self.get_compressed_version(l)
                    
                    for l in self.mutation(parents[i+1]):
                        l['Generation'] = gen
                        l['Parent'] = parents[i+1]['ID']
                        children.append(l)
                        #self.mapStrategies[l['ID']] = self.get_compressed_version(l)
                
                # Add random children to the population if the population is not full
                for _ in range(self.population-len(children)):
                    children.append(self.randomChild(gen=gen))

                # Replace population
                population = copy.deepcopy(children)

                # if stuck_counter == 0:
                #     threshold_quantile = 0.2

                # if stuck_counter >= (self.iterations)//100:
                #     stuck_counter = 0
                #     quarter_pop = self.population//4
                #     population = population[:self.population]
                #     idx = random.randint(1, quarter_pop)
                #     threshold_quantile = round(threshold_quantile - 0.05,2)
                #     for i in range(3*quarter_pop+idx, self.population):
                #         population[i] = self.randomChild()
                
                # if threshold_quantile <= 0.01 or threshold_quantile >= 0.99:
                #     threshold_quantile = round(random.uniform(0.3,0.99),2)

                for key, val in best.items():
                    if not math.isinf(val['TotalTime']) and key > 0 and key < 3 and val['ID'] not in [x['ID'] for x in population] and not any([x['TotalTime'] < val['TotalTime'] for x in population]):
                        print(f"Best {key} not in population!")

                valid_strategies = round(((sum([1 for x in population if x['Valid'] == True]))/len(population))*100,2)
                bar.set_description(f"Circuit:{self.circuit}, Pop:{self.population}, Iter:{self.iterations}, Mut:{self.sigma}, Cross:{self.mu} | ({best[0]['NumPitStop']})P: {ms_to_time(best[0]['TotalTime'])}, 1P: {ms_to_time(best[1]['TotalTime'])}, 2P: {ms_to_time(best[2]['TotalTime'])}, 3P: {ms_to_time(best[3]['TotalTime'])}")
                bar.refresh()
                string = f"[EA] Generation {gen+1} - ({best[0]['NumPitStop']})P: {ms_to_time(best[0]['TotalTime'])}, 1P: {ms_to_time(best[1]['TotalTime'])} 2P: {ms_to_time(best[2]['TotalTime'])} 3P: {ms_to_time(best[3]['TotalTime'])} - valid strategies: {valid_strategies}%"
                self.log.write(string+"\n")

        except KeyboardInterrupt:
            pass 


        end_timer = time.time() - start_timer
        string = f"Time elapsed: {ms_to_time(round(end_timer*1000))}\n"
        
        self.log.write("\n\n"+string)

        for pit in range(1,4):
            with open(os.path.join(self.path, f"Strategy_{pit}_pit.txt"), "w") as f:
                string = f"Fastest strategy for {self.circuit} with {pit} pit stops\n\n"
                string += f"Total time: {ms_to_time(best[pit]['TotalTime'])}\n"
                
                for lap in range(self.numLaps):
                    string += f"Lap {lap+1}: {best[pit]['TyreCompound'][lap]}, TyresAge {best[pit]['TyreAge'][lap]}, Wear {round(best[pit]['TyreWear'][lap],1)}%, Fuel {round(best[pit]['FuelLoad'][lap],2)} Kg, PitStop {'Yes' if best[pit]['PitStop'][lap] else 'No'}, TimeLost {ms_to_time(best[pit]['LapTime'][lap])}\n"
                #print(string)
                string += "\n"
                f.write(string)
                for individual in population:
                    self.checkValidity(individual)
                    if individual['Valid'] and individual['NumPitStop'] == pit and individual != best[pit] and individual['TotalTime'] <= best[pit]['TotalTime']:
                        f.write(f"\nStrategy for {self.circuit} with {pit} pit stops\n")
                        f.write(f"Total time: {ms_to_time(best[pit]['TotalTime'])}\n")
                        for lap in range(self.numLaps):
                            f.write(f"Lap {lap+1}: {individual['TyreCompound'][lap]}, TyresAge {individual['TyreAge'][lap]}, Wear {round(individual['TyreWear'][lap],1)}%, Fuel {round(individual['FuelLoad'][lap],2)} Kg, PitStop {'Yes' if individual['PitStop'][lap] else 'No'}, TimeLost {ms_to_time(individual['LapTime'][lap])}\n")
                f.close()
        
        best_idx = 0
        best_fit_temp = np.inf
        for key, values in best.items():
            if values['TotalTime'] < best_fit_temp:
                best_idx = key
                best_fit_temp = values['TotalTime']

        # with open(os.path.join(self.path, "Strategy.txt"), 'w') as f:
        #     string = f"Fastest strategy for {self.circuit} with {best_idx} pit stops\n\n"
        #     string += f"Total time: {ms_to_time(best[best_idx]['TotalTime'])}\n"
        #     for lap in range(self.numLaps):
        #         string += f"Lap {lap+1}: {best[best_idx]['TyreCompound'][lap]}, TyresAge {best[best_idx]['TyreAge'][lap]}, Wear {round(best[best_idx]['TyreWear'][lap],1)}%, Fuel {round(best[best_idx]['FuelLoad'][lap],2)} Kg, PitStop {'Yes' if best[best_idx]['PitStop'][lap] else 'No'}, TimeLost {ms_to_time(best[best_idx]['LapTime'][lap])}\n"
        #     print(string)
        #     string += "\n"
        #     f.write(string)
        
        with open(os.path.join(self.path, "Strategy_CSV.csv"), 'w') as f:
            f.write("Generation,key,Pits,Tyres,Pits\n")
            for key, value in best.items():
                if key > 0:
                    tyre, pit = self.get_compressed_version(value)
                    f.write(f"{value['Generation']},{key},{value['NumPitStop']},{tyre},{pit}\n")
                    parent_ID = value['Parent']
                    while parent_ID is not None:
                        f.write(f"{self.mapStrategies[parent_ID]['Generation']},{key},{self.mapStrategies[parent_ID]['NumPitStop']},{self.mapStrategies[parent_ID]['Strategy_Tyre']},{self.mapStrategies[parent_ID]['Strategy_Pit']}\n")
                        parent_ID = self.mapStrategies[parent_ID]['Parent']
            f.close()

        with open(os.path.join(self.path, "LastPopulation.csv"), 'w') as f:
            f.write("idx,S_ID,Valid,NumPitStops,Tyres,Pits\n")
            for idx, value in enumerate(population):
                tyre, pit = self.get_compressed_version(value)
                f.write(f"{idx+1},{value['ID']},{value['Valid']},{value['NumPitStop']},{tyre},{pit}\n")
            f.close()


        # with open(os.path.join(self.path, "Strategy_Log.csv"), 'w') as f:
        #     f.write("Generation, ID, nPit, Strategy,\n")
        #     for key, values in best.items():
        #         #f.write(f"Strategy for {self.circuit} with {key} pit stops\n")
        #         for val in self.mapStrategies[str(values['ID'])]:
        #             f.write(f"{val['Generation']}, {val['ID']}, {key}, {self.get_compressed_version(val)},\n")
        #     f.close()

        #     for key, values in best.items():
        #         string = f"Strategy for {self.circuit} with {key} pit stops\n\n"
        #         for val in self.mapStrategies[str(values['ID'])]:
        #             string += f"ID: {val['ID']} - Total time: {ms_to_time(val['TotalTime'])}\n"
        #             for lap in range(self.numLaps):
        #                 string += f"Lap {lap+1}: {val['TyreCompound'][lap]}, TyresAge {val['TyreAge'][lap]}, Wear {round(val['TyreWear'][lap],1)}%, Fuel {round(val['FuelLoad'][lap],2)} Kg, PitStop {'Yes' if val['PitStop'][lap] else 'No'}, TimeLost {ms_to_time(val['LapTime'][lap])}\n"
        #             string += "\n"
        #         f.write(string)

        boxplot_df = pd.DataFrame(index=range(len(boxplot_list)))
        for i in range(max([len(x) for x in boxplot_list])):
            for j in range(len(boxplot_list)):
                try:
                    boxplot_df.at[i,j] = boxplot_list[j][i]
                except:
                    pass

        boxplot_df.to_csv(os.path.join(self.path,'Boxplot.csv'))

        return best, best_idx, round(end_timer*1000)

    def initSolver(self,):
        strategies = []
        for _ in range(self.population):
            strategies.append(self.randomChild())

        return strategies

    def randomChild(self, gen:int=0, ID:int=0):

        strategy = {'ID': self.availableID,'Parent': None, 'Generation':gen, 'TyreCompound': [], 'TyreAge':[], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'NumPitStop': 0, 'Valid':False, 'TotalTime': np.inf}
        self.availableID += 1
        ### Get a random compound and verify that we can use it, if so we update the used compounds list and add the compound to the strategy
        compound = self.randomCompound()#(weather[0])
        strategy['TyreCompound'].append(compound)

        ### If the compound is used we put a tyre wear of 2 laps (if it is used but available the compound has been used for 2/3 laps.
        ### However, 2 laps out of 3 are done very slowly and the wear is not as the same of 3 laps)
        ### If the compound is new tyre wear = 0
        tyresAge = 0 
        strategy['TyreAge'].append(tyresAge)
        strategy['TyreWear'].append(self.getTyreWear(compound, tyresAge))

        ### The fuel load can be inferred by the coefficient of the fuel consumption, we add a random value between -10 and 10 to get a little variation
        strategy['FuelLoad'].append(self.car.initial_fuel)

        ### At first lap the pit stop is not made (PitStop list means that at lap i^th the pit stop is made at the beginning of the lap)
        strategy['PitStop'].append(False)

        ### Compute lapTime
        strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=0,pitStop=False))

        ### For every lap we repeat the whole process
        for lap in range(1,self.numLaps):
            ### The fuel does not depend on the compound and/or pit stops => we compute it and leave it here
            fuelLoad = self.getFuelLoad(lap=lap)
            strategy['FuelLoad'].append(fuelLoad)

            newTyre = random.choice([True, False])

            if newTyre:
                compound = self.randomCompound()#(weather[lap])
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
            strategy['LapTime'].append(self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, pitStop=pitStop))
        
        #self.checkValidity(strategy)    
        strategy['TotalTime'] = sum(strategy['LapTime'])

        self.mapStrategies[ID] = {'Strategy':self.get_compressed_version(strategy), 'Generation':gen, 'Parent':None}
        #self.mapStrategies[str(strategy['ID'])].append(strategy)

        return strategy

    def randomCompound(self,):
        return random.choice(['Soft', 'Medium', 'Hard',])#'Inter','Wet'

    def selection_dynamic_penalty(self, step:int, population:list, threshold_quantile:float):
        best = min([p['TotalTime'] for p in population if p['Valid']])
        penalty = [abs(x['TotalTime'] - best) for x in population]
        alpha = np.exp(1+(1/self.iterations)*step)

        #quantile = np.percentile(penalty, threshold_quantile*100)

        for p, pop in zip(penalty, population):
            if not pop['Valid']:
                highest_wear = max(pop['TyreWear'])
                
                if pop['NumPitStop'] < 1 or len(set(pop['TyreCompound'])) < 2:
                    p *= alpha
                    if p == 0.0:
                        p += np.exp(alpha)

                elif highest_wear > 100:
                    p *= np.exp(highest_wear/100)
                    if p == 0.0:
                        p += np.exp(highest_wear/100)

                #
                # NOT ACTIVE SINCE FUEL DATA NOT INCLUDED
                #
                # last_lap_fuel_load = self.getFuelLoad(lap=self.numLaps)
                # if last_lap_fuel_load < 0:
                #     last_lap_fuel_load = abs(last_lap_fuel_load)
                #     p *= np.exp(last_lap_fuel_load)
                #     if p == 0.0:
                #         p = np.exp(last_lap_fuel_load)
                
        
        for idx, x in enumerate(population):
            x['Penalty'] = penalty[idx]
        sortedPopulation = sorted(population, key=lambda x: x['Penalty'])
        #selected = [x for idx, x in enumerate(sortedPopulation) if x['Penalty'] <= quantile]
        selected = sortedPopulation[:int(threshold_quantile*self.population)]
        
        for x in selected:
            x.pop('Penalty')
        
        if BEST_ID > 0 and BEST_ID not in [x['ID'] for x in selected]:
            print(f'\nBest not in selected\n{self.mapStrategies[BEST_ID]}\n\n')

        return selected

    # def mutation_fuel_load(self, child:dict, ):
    #     new_fuel = child['FuelLoad'][0]+random.uniform(-10,10)

    #     child['FuelLoad'][0] = new_fuel
    #     child['LapTime'][0] = self.getLapTime(compound=child['TyreCompound'][0], compoundAge=child['TyreAge'][0], lap=0, fuel_load=new_fuel, drs=False, pitStop=child['PitStop'][0])
        
    #     for lap in range(1,self.numLaps):
    #         fuel = self.getFuelLoad(initial_fuel=new_fuel, conditions=child['Weather'][:lap+1])
    #         timing = self.getLapTime(compound=child['TyreCompound'][lap], compoundAge=child['TyreAge'][lap], lap=lap, fuel_load=fuel, drs=False, pitStop=child['PitStop'][lap])
            
    #         child['FuelLoad'][lap] = fuel
    #         child['LapTime'][lap] = timing

    #     child['TotalTime'] = sum(child['LapTime'])
    #     return child

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
        #weather = child['Weather'][lap]

        while oldCompound == compoundRandom:
            compoundRandom = self.randomCompound()
        
        child['TyreCompound'][lap] = compoundRandom

        for i in range(lap + 1, self.numLaps):
            if not child['PitStop'][lap]:
                child['TyreCompound'][i] = compoundRandom
            else:
                return self.correct_strategy(child)
        
        self.availableID += 1 
        child['ID'] = self.availableID       
        
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
        
        self.availableID += 1
        child['ID'] = self.availableID

        return self.correct_strategy(child, index)
    
    def mutation_pitstop_add(self, child:dict):
        random_lap = random.randint(1, self.numLaps-1)

        while child['PitStop'][random_lap] == True:
            random_lap = random.randint(1, self.numLaps-1)
        
        compound = self.randomCompound()
        
        tyre_age = 0
        child['PitStop'][random_lap] = True
        child['TyreAge'][random_lap] = tyre_age
        child['TyreWear'][random_lap] = self.getTyreWear(compound=compound, compoundAge=tyre_age)
        child['TyreCompound'][random_lap] = compound
        child['LapTime'][random_lap] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=random_lap, pitStop=child['PitStop'][random_lap])
        child['NumPitStop'] += 1
        remaining = random_lap + 1
        tyre_age += 1
        while remaining < self.numLaps and child['PitStop'][remaining] == False:
            child['TyreWear'][remaining] = self.getTyreWear(compound=compound, compoundAge=tyre_age)#, conditions=child['Weather'][:remaining])
            child['TyreCompound'][remaining] = compound
            child['TyreAge'][remaining] = tyre_age
            child['LapTime'][remaining] = self.getLapTime(compound=compound, compoundAge=tyre_age, lap=remaining, pitStop=child['PitStop'][remaining])
            remaining += 1
            tyre_age += 1
        child['TotalTime'] = sum(child['LapTime'])
        
        self.availableID += 1
        child['ID'] = self.availableID
        
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

        #if random.random() < self.sigma:
        #    children.append(self.mutation_fuel_load(copy.deepcopy(child)))
        #    childAllMutated = self.mutation_fuel_load(childAllMutated)
        
        children.append(childAllMutated)

        return children

    # def crossover_fuel(self, p1:dict, p2:dict):
    #     fuelLoad_p1 = p1['FuelLoad'][0]
    #     fuelLoad_p2 = p2['FuelLoad'][0]

    #     p1['FuelLoad'][0] = fuelLoad_p2
    #     p2['FuelLoad'][0] = fuelLoad_p1

        
    #     for lap in range(1, self.numLaps):
    #         fuelLoad_p1 = self.getFuelLoad(lap=lap)
    #         fuelLoad_p2 = self.getFuelLoad(lap=lap)
    #         p1['FuelLoad'][lap] = fuelLoad_p2
    #         p2['FuelLoad'][lap] = fuelLoad_p1
        
    #     return self.correct_strategy(p1), self.correct_strategy(p2)
        

    # def crossover(self, p1:dict, p2:dict,):
    #     # children are copies of parents by default
    #     c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
    #     # check for recombination
    #     if random.random() < self.mu:
    #         c1, c2 = self.crossover_fuel(c1, c2)
        
    #     return [c1,c2]

    def correct_strategy(self, strategy:dict, index:int=0):
        tyre = strategy['TyreCompound'][0]
        strategy['LapTime'][0] = self.getLapTime(compound=tyre, compoundAge=0, lap=0, pitStop=False)
        
        if index != 0 and index != self.numLaps:
            compound = strategy['TyreCompound'][index-1]
            tyre_age = strategy['TyreAge'][index-1]
            while index < self.numLaps and strategy['PitStop'][index] == False:
                tyre_age += 1
                strategy['TyreAge'][index] = tyre_age
                strategy['TyreCompound'][index] = compound
                strategy['TyreWear'][index] = self.getTyreWear(strategy['TyreCompound'][index], strategy['TyreAge'][index])
                strategy['LapTime'][index] = self.getLapTime(strategy['TyreCompound'][index], strategy['TyreAge'][index], index, pitStop=strategy['PitStop'][index])
                index += 1

            strategy['NumPitStop'] = sum([x for x in strategy['PitStop'] if x])
            strategy['TotalTime'] = sum(strategy['LapTime'])

            return strategy

        for lap in range(1, self.numLaps):
            ### Get if a pitstop is made and compound lap'
            pitStop = strategy['PitStop'][lap]
            old_compound = strategy['TyreCompound'][lap-1]
            compound = strategy['TyreCompound'][lap]
            tyresAge = strategy['TyreAge'][lap-1]
            
            ### We have two options: either there is a pitstop or the compound has changes, if so we have to recalculate all
            if pitStop or old_compound != compound:# or strategy['TyreWear'][lap-1] >= 100:
                tyresAge = 0
                pitStop = True
            else:
                tyresAge += 1
                
            tyreWear = self.getTyreWear(compound=compound, compoundAge=tyresAge)
            timing = self.getLapTime(compound=compound, compoundAge=tyresAge, lap=lap, pitStop=pitStop)
            strategy['PitStop'][lap] = pitStop
            strategy['TyreWear'][lap] = tyreWear
            strategy['TyreAge'][lap] = tyresAge
            strategy['LapTime'][lap] = timing

        strategy['NumPitStop'] = sum([x for x in strategy['PitStop'] if x])
        strategy['TotalTime'] = sum(strategy['LapTime'])
        #self.checkValidity(strategy)

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

    # def build_tree(self, temp_tree:list, tyres_age:int, lap:int):
    #     global BEST_TIME
    #     global STRATEGY

    #     total_time = sum([x['LapTime'] for x in temp_tree])
    #     pitStop_count = sum([x['PitStop'] for x in temp_tree])
    #     initial_fuel = temp_tree[0]['FuelLoad']
    #     compound_set = set([x['Compound'] for x in temp_tree])

    #     if total_time > BEST_TIME or pitStop_count > 2:
    #         return {'Strategy':None, 'TotalTime':np.inf}

    #     if any([x >= 0.8 for x in temp_tree[-1]['TyreWear'].values()]):
    #         return {'Strategy':None, 'TotalTime':np.inf}

    #     if lap == self.numLaps:
    #         if total_time < BEST_TIME:
    #             if len(compound_set) > 1:
    #                 BEST_TIME = total_time
    #                 STRATEGY = copy.deepcopy(temp_tree)
    #                 return {'Strategy':copy.deepcopy(temp_tree), 'TotalTime':total_time}
            
    #         return {'Strategy':None, 'TotalTime':np.inf}
         
    #     fuel_load = self.getFuelLoad(initial_fuel,weather[:lap+1])
    #     w = weather[lap]

    #     if w < 20:
    #         idx = 1
    #         values = {1:None, 2:None, 3:None, 4:None}
    #         for compound in ['Soft', 'Medium','Hard']:
    #             if compound == temp_tree[-1]['Compound']:
    #                 for pitStop in [True,False]:
    #                     node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                     temp_tree.append(node)
    #                     values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                     temp_tree.pop()
    #                     idx+=1
    #             else:
    #                 pitStop = True
    #                 node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                 temp_tree.append(node)
    #                 values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                 temp_tree.pop()
    #                 idx+=1
    #     elif w > 50 and w < 80:
    #         values = {1:None}
    #         idx = 1
    #         compound = 'Inter'
    #         if compound == temp_tree[-1]['Compound']:
    #             for pitStop in [True,False]:
    #                 node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                 temp_tree.append(node)
    #                 values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                 temp_tree.pop()
    #         else:
    #             pitStop = True
    #             node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #             temp_tree.append(node)
    #             values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #             temp_tree.pop()
    #     elif w > 80:
    #         values = {1:None}
    #         idx = 1
    #         compound = 'Wet'
    #         if compound == temp_tree[-1]['Compound']:
    #             for pitStop in [True,False]:
    #                 node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                 temp_tree.append(node)
    #                 values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                 temp_tree.pop()
    #         else:
    #             pitStop = True
    #             node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #             temp_tree.append(node)
    #             values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #             temp_tree.pop()
    #     else:
    #         values = {1:None, 2:None, 3:None, 4:None}
    #         idx = 1
    #         for compound in ['Inter','Soft', 'Medium','Hard']:
    #             if compound == temp_tree[-1]['Compound']:
    #                 for pitStop in [True,False]:
    #                     node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                     temp_tree.append(node)
    #                     values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                     temp_tree.pop()
    #                     idx+=1
    #             else:
    #                 pitStop = True
    #                 node = {'Compound':compound, 'TyreWear': self.getTyreWear(compound, tyres_age+1 if not pitStop else 0), 'TyreAge':tyres_age+1 if not pitStop else 0, 'FuelLoad':fuel_load, 'PitStop':pitStop, 'LapTime': self.getLapTime(compound=compound, compoundAge=tyres_age+1 if not pitStop else 0, lap=lap, fuel_load=fuel_load, conditions=weather[:lap+1], drs=False, pitStop=pitStop)}
    #                 temp_tree.append(node)
    #                 values[idx] = self.build_tree(temp_tree, tyres_age+1 if not pitStop else 0, lap+1)
    #                 temp_tree.pop()
    #                 idx+=1

    #     ### Best Strategy
    #     to_remove = list()
    #     for key, val in values.items():
    #         if val is None:
    #             to_remove.append(key)
    #     for key in to_remove:
    #         values.pop(key)
    #     for val in values.values():
    #         if val['Strategy'] is not None:
    #             if len(val['Strategy']) < self.numLaps:
    #                 val['TotalTime'] = np.inf
    #     best_strategy = min(values, key=lambda x: values[x]['TotalTime'])


    #     return {'Strategy':copy.deepcopy(values[best_strategy]['Strategy']), 'TotalTime':values[best_strategy]['TotalTime']}

    # def lower_bound(self,):
    #     ### Build the solution space as a tree
    #     temp_tree = []
    #     weather = self.weather.get_weather_percentage_list()
    #     initial_fuel = self.getInitialFuelLoad(weather)
    #     timer_start = time.time()
    #     w = weather[0]

    #     if w < 20:
    #         values = {1:None, 2:None, 3:None}
            
    #         ### Soft
    #         soft_timer = time.time()
    #         compound = 'Soft'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[1] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         soft_timer = ms_to_time(round(1000*(time.time() - soft_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {soft_timer}")

    #         #return values[1]['Strategy'], values[1]['TotalTime']

    #         ### Medium
    #         medium_timer = time.time()
    #         compound = 'Medium'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[2] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         medium_timer = ms_to_time(round(1000*(time.time() - medium_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {medium_timer}")

    #         ### Hard
    #         hard_timer = time.time()
    #         compound = 'Hard'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[3] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         hard_timer = ms_to_time(round(1000*(time.time() - hard_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {hard_timer}")
    #     elif w > 50 and w < 80:
    #         values = {1:None}
    #         ### Inter
    #         inter_timer = time.time()
    #         compound = 'Inter'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[1] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         inter_timer = ms_to_time(round(1000*(time.time() - inter_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {inter_timer}")
    #     elif w > 80:
    #         values = {1:None}
    #         ### Wet
    #         wet_timer = time.time()
    #         compound = 'Wet'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[1] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         wet_timer = ms_to_time(round(1000*(time.time() - wet_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {wet_timer}")
    #     else:
    #         values = {1:None, 2:None, 3:None, 4:None}
            
    #         ### Inter
    #         inter_timer = time.time()
    #         compound = 'Inter'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[1] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         inter_timer = ms_to_time(round(1000*(time.time() - inter_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {inter_timer}")

    #         ### Soft
    #         soft_timer = time.time()
    #         compound = 'Soft'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[2] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         soft_timer = ms_to_time(round(1000*(time.time() - soft_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {soft_timer}")

    #         ### Medium
    #         medium_timer = time.time()
    #         compound = 'Medium'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[3] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         medium_timer = ms_to_time(round(1000*(time.time() - medium_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {medium_timer}")

    #         ### Hard
    #         hard_timer = time.time()
    #         compound = 'Hard'
    #         print(f"[BruteForce] Computations starting with {compound}...")
    #         temp_tree.append({'Compound':compound, 'TyreWear': self.getTyreWear(compound, 0), 'TyreAge':0, 'FuelLoad':initial_fuel, 'PitStop':False, 'LapTime': self.getLapTime(compound=compound, compoundAge=0, lap=0, fuel_load=initial_fuel, conditions=[weather[0]], drs=False, pitStop=False)})
    #         values[4] = self.build_tree(temp_tree, 0, 1)
    #         temp_tree.pop()
    #         hard_timer = ms_to_time(round(1000*(time.time() - hard_timer)))
    #         print(f"\033[A\033[K[BruteForce] {compound} computed in {hard_timer}")

    #     ### Best Strategy
    #     best_strategy_index = min(values, key=lambda x: values[x]['TotalTime'])
    #     best_strategy, best_laptime = values[best_strategy_index]['Strategy'], values[best_strategy_index]['TotalTime']

    #     if best_strategy is None:
    #         print(f"Strategy is none...")
    #         exit(-1)

    #     for lap, strategy in enumerate(best_strategy):
    #         print(f"Lap {lap+1} -> Compound '{strategy['Compound']}', TyresAge {strategy['TyreAge']}, Wear '{round(strategy['TyreWear']['FL']*100,1)}'% | '{round(strategy['TyreWear']['FR']*100,1)}'% | '{round(strategy['TyreWear']['RL']*100,1)}'% | '{round(strategy['TyreWear']['RR']*100,1)}'%, Fuel '{round(strategy['FuelLoad'],2)}' Kg, PitStop '{'Yes' if strategy['PitStop'] else 'No'}', Time '{ms_to_time(strategy['LapTime'])}' ms")
    #     print(f"Computed in time {ms_to_time(round(1000*(time.time()-timer_start)))}")
    #     print(f"Total time {ms_to_time(best_laptime)}")
    #     ### Find the best solution
    #     return best_strategy, best_laptime

    def fixed_strategy(self, compund_list:list, stop_lap:list=[]):
        if len(stop_lap) != len(compund_list)-1:
            print(f"Either the compound list or the pit stop list are wrong!")
            exit(-1)
        stop_lap.append(self.numLaps)
        strategy = []
        start_fuel = self.car.initial_fuel
        idx_stop = 0
        idx_tyre = 0
        tyre_lap = 0
        for lap in range(self.numLaps):
            if lap != 0:
                fuel_load = self.getFuelLoad(lap+1)
            else:
                fuel_load = start_fuel

            if lap < stop_lap[idx_stop]:
                strategy.append({'Compound':compund_list[idx_tyre], 'TyresWear': self.getTyreWear(compund_list[idx_tyre], tyre_lap), 'TyresAge':tyre_lap, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound=compund_list[idx_tyre], compoundAge=tyre_lap, lap=lap,pitStop=False)})
            elif lap == stop_lap[idx_stop]:
                idx_tyre += 1
                strategy.append({'Compound':compund_list[idx_tyre], 'TyresWear': self.getTyreWear(compund_list[idx_tyre], 0), 'TyresAge':0, 'FuelLoad':fuel_load, 'PitStop':False, 'LapTime': self.getLapTime(compound=compund_list[idx_tyre], compoundAge=0, lap=lap, pitStop=True)})
                idx_stop += 1
                tyre_lap = 0

            tyre_lap += 1
            
        total = 0
        for lap, strategy in enumerate(strategy):
            print(f"Lap {lap+1}/{self.numLaps} - Compound: '{strategy['Compound']}', TyresAge: {strategy['TyresAge']} Laps, TyresWear: {strategy['TyresWear']}, FuelLoad: {strategy['FuelLoad']} Kg, PitStop: {'Yes' if strategy['PitStop'] else 'No'}, LapTime: {ms_to_time(strategy['LapTime'])} (hh:)mm:ss.ms")
            total += strategy['LapTime']

        print(f"With a total time of {ms_to_time(total)} -> {total}")

        exit(0)