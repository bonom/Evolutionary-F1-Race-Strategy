import copy
import numpy as np
from random import SystemRandom

from classes.Genetic import GeneticSolver

random = SystemRandom()

TYRE_WEAR_THRESHOLD = 0.3
BEST_TIME = np.inf
STRATEGY = None

class LocalSearch:
    def __init__(self, strategy:dict, genetic:GeneticSolver):
        self.strategy = strategy
        self.genetic = genetic

    def find_interval(self, indexPitstop: int):
        count = 0
        index = 0

        if indexPitstop == self.strategy['NumPitStop'] + 1 :
            return -1
        
        for i in range(0, self.genetic.numLaps):
            if self.strategy['PitStop'][i] == True:
                count = count + 1

            if count == indexPitstop:
                return i

        return index

    def shake(self, indexRandom, nextIndexRandom, randomCompound):
        """
        Shake is working on the compounds, so it changes randomly only one compound
        """
        shakeStrategy = copy.deepcopy(self.strategy)
        #shakeRandom = random.randint(0, self.strategy['NumPitStop'])
        #indexRandom = self.find_interval(shakeRandom)
        #nextIndexRandom = self.find_interval(shakeRandom+1)
        #randomCompound = self.genetic.randomCompound()
#
        if randomCompound == self.strategy['TyreCompound'][indexRandom]:
            return shakeStrategy

        shakeStrategy['TyreCompound'][indexRandom] = randomCompound
        if indexRandom!=self.genetic.numLaps-1:
            for i in range(indexRandom + 1, self.genetic.numLaps):
                if shakeStrategy['PitStop'][i] == False:
                    shakeStrategy['TyreCompound'][i] = randomCompound
                else:
                    if self.genetic.checkValidity(self.genetic.correct_strategy(shakeStrategy)):
                        return self.genetic.correct_strategy(shakeStrategy)

        if self.genetic.checkValidity(self.genetic.correct_strategy(shakeStrategy)):
            shakeStrategy = self.genetic.correct_strategy(shakeStrategy)
        else:
            shakeStrategy = copy.deepcopy(self.strategy)

        return self.genetic.correct_strategy(shakeStrategy)

    def local_search(self, strategy:dict, index:int, nextIndex:int):
        """
        LocalSearch is working on the pitstops of the shaked strategy.
        It is a BestImprovement local search.
        """
        localBest = copy.deepcopy(strategy)
        localStrategy_1 = copy.deepcopy(strategy)
        localStrategy_2 = copy.deepcopy(strategy)
        if index != 0:
            localStrategy_1['PitStop'][index] = False
            localStrategy_2['TyreCompound'][index] = localStrategy_2['TyreCompound'][index-1]
            for i in range(index, index-5, -1): 
                if i == 0:
                    localStrategy_1['PitStop'][i] = False
                    localStrategy_1['TyreCompound'][i] = localStrategy_1['TyreCompound'][index]

                    self.genetic.correct_strategy(localStrategy_1)

                    if localStrategy_1['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_1):
                        localBest = copy.deepcopy(localStrategy_1)

                if i > 0 and i < self.genetic.numLaps and i != index:
                    localStrategy_1['PitStop'][i] = True
                    localStrategy_1['TyreCompound'][i] = localStrategy_1['TyreCompound'][index]
                    self.genetic.correct_strategy(localStrategy_1)

                    if localStrategy_1['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_1):
                        localBest = copy.deepcopy(localStrategy_1)
                    else:
                        localStrategy_1['PitStop'][i] = False
                    # if i < index:
                    #     localStrategy_1['PitStop'][i] = True
                    #     localStrategy_1['TyreCompound'][i] = localStrategy_1['TyreCompound'][index]
                    #     self.genetic.correct_strategy(localStrategy_1)

                    #     if localStrategy_1['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_1):
                    #         localBest = copy.deepcopy(localStrategy_1)
                    #     else:
                    #         localStrategy_1['PitStop'][i] = False
            for i in range(index + 1, index + 6):
                if i == self.genetic.numLaps:
                    localStrategy_2['PitStop'][i-1] = False
                    self.genetic.correct_strategy(localStrategy_2)

                    if localStrategy_2['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_2):
                        localBest = copy.deepcopy(localStrategy_2)

                if i > 0 and i < self.genetic.numLaps and i != index:
                    localStrategy_2['PitStop'][i] = True
                    localStrategy_2['TyreCompound'][i] = localStrategy_2['TyreCompound'][i-1]
                    self.genetic.correct_strategy(localStrategy_2)

                    if localStrategy_2['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_2):
                        localBest = copy.deepcopy(localStrategy_2)
                    else:
                        localStrategy_2['PitStop'][i] = False

        if nextIndex != -1:
            localStrategy_3 = copy.deepcopy(strategy)
            localStrategy_4 = copy.deepcopy(strategy)
            localStrategy_3['PitStop'][nextIndex] = False
            localStrategy_4['TyreCompound'][nextIndex] = localStrategy_4['TyreCompound'][nextIndex-1]
            for i in range(nextIndex-1, nextIndex-6, -1): 
                if i == 0:
                    localStrategy_3['PitStop'][i] = False
                    localStrategy_3['TyreCompound'][i] = localStrategy_3['TyreCompound'][index]

                    self.genetic.correct_strategy(localStrategy_3)

                    if localStrategy_3['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_3):
                        localBest = copy.deepcopy(localStrategy_3)

                if i >= 0 and i < self.genetic.numLaps and i != nextIndex:
                    localStrategy_3['PitStop'][i] = True
                    localStrategy_3['TyreCompound'][i] = localStrategy_3['TyreCompound'][nextIndex]
                    self.genetic.correct_strategy(localStrategy_3)

                    if localStrategy_3['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_3):
                        localBest = copy.deepcopy(localStrategy_3)
                    else:
                        localStrategy_3['PitStop'][i] = False

            for i in range(nextIndex+1, nextIndex+6): 
                if i == self.genetic.numLaps:
                    localStrategy_4['PitStop'][i-1] = False
                    self.genetic.correct_strategy(localStrategy_4)

                    if localStrategy_4['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_2):
                        localBest = copy.deepcopy(localStrategy_4)

                if i >= 0 and i < self.genetic.numLaps and i != nextIndex:
                    localStrategy_4['PitStop'][i] = True
                    localStrategy_4['TyreCompound'][i] = localStrategy_4['TyreCompound'][i-1]
                    self.genetic.correct_strategy(localStrategy_4)

                    if localStrategy_4['TotalTime'] < localBest['TotalTime'] and self.genetic.checkValidity(localStrategy_4):
                        localBest = copy.deepcopy(localStrategy_4)
                    else:
                        localStrategy_4['PitStop'][i] = False

        return localBest
    
    def move_or_not(self, localSearchStrategy: dict):
        if self.strategy['TotalTime'] > localSearchStrategy['TotalTime']:
            newStrategy = copy.deepcopy(localSearchStrategy)
        else:
            newStrategy = copy.deepcopy(self.strategy)

        return newStrategy

    def run(self):
        best = copy.deepcopy(self.strategy)

        for t in ['Soft', 'Medium', 'Hard','Inter','Wet']:
            for p in range(0, self.strategy['NumPitStop']):
                indexRandom = self.find_interval(p)
                nextIndexRandom = self.find_interval(p+1)
                shakeStrategy = self.shake(indexRandom, nextIndexRandom, t)
                localSearchStrategy = self.local_search(shakeStrategy, indexRandom, nextIndexRandom)
                newStrategy = self.move_or_not(localSearchStrategy)
                if newStrategy['TotalTime'] < best['TotalTime']: #and self.genetic.checkValidity(newStrategy):
                    best = copy.deepcopy(newStrategy)

        return best, best['TotalTime']
