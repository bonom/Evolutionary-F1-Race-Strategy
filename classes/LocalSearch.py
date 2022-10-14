import math, time, copy, os
from threading import local
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm

from classes.Car import Car
from classes.Weather import Weather
from classes.Genetic import GeneticSolver
random = SystemRandom()

from classes.Utils import CIRCUIT, Log, ms_to_time

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
        for i in range(0, self.genetic.numLaps):
            if self.strategy['PitStop'][i] == True:
                count = count + 1

            if count == indexPitstop:
                return i

        return index

    def shake(self):
        """
        Shake is working on the compounds, so it changes randomly only one compound
        """
        shakeStrategy = copy.deepcopy(self.strategy)
        shakeRandom = random.randint(0, self.strategy['NumPitStop'])
        indexRandom = self.find_interval(shakeRandom)
        randomCompound = self.genetic.randomCompound()

        if randomCompound == self.strategy['TyreCompound'][indexRandom]:
            while randomCompound == self.strategy['TyreCompound'][indexRandom]:
                randomCompound = self.genetic.randomCompound()

        shakeStrategy['TyreCompound'][indexRandom] = randomCompound
        if indexRandom!=self.genetic.numLaps-1:
            for i in range(indexRandom + 1, self.genetic.numLaps):
                if shakeStrategy['PitStop'][i] == False:
                    shakeStrategy['TyreCompound'][i] = randomCompound
                else:
                    return indexRandom, self.genetic.correct_strategy(shakeStrategy)
                    
        return indexRandom, self.genetic.correct_strategy(shakeStrategy)

    def local_search(self, strategy:dict, index:int):
        """
        LocalSearch is working on the pitstops of the shaked strategy.
        It is a BestImprovement local strategy.
        """
        localBest = copy.deepcopy(strategy)
        localStrategy_1 = copy.deepcopy(strategy)
        localStrategy_2 = copy.deepcopy(strategy)
        localStrategy_1['PitStop'][index] = False
        localStrategy_2['TyreCompound'][index] = localStrategy_2['TyreCompound'][index-1]
        for i in range(index-5, index+5): 
            if i >= 0 and i < self.genetic.numLaps and i != index:
                if i < index:
                    localStrategy_1['PitStop'][i] = True
                    localStrategy_1['TyreCompound'][i] = localStrategy_1['TyreCompound'][index]
                    self.genetic.correct_strategy(localStrategy_1)

                    if localStrategy_1['TotalTime'] < localBest['TotalTime']:
                        localBest = copy.deepcopy(localStrategy_1)
                    else:
                        localStrategy_1['PitStop'][i] = False
                else:
                    localStrategy_1['PitStop'][i] = True
                    localStrategy_2['TyreCompound'][i] = localStrategy_2['TyreCompound'][i-1]
                    self.genetic.correct_strategy(localStrategy_2)

                    if localStrategy_2['TotalTime'] < localBest['TotalTime']:
                        localBest = copy.deepcopy(localStrategy_2)
                    else:
                        localStrategy_2['PitStop'][i] = False

        return localBest
    
    def move_or_not(self, localSearchStrategy: dict):
        if self.strategy['TotalTime'] > localSearchStrategy['TotalTime']:
            newStrategy = copy.deepcopy(localSearchStrategy)
        else:
            newStrategy = copy.deepcopy(self.strategy)
        return newStrategy

    def run(self):
        best = copy.deepcopy(self.strategy)
        k = 0

        while k < 200:
            index, shakeStrategy = self.shake()
            localSearchStrategy = self.local_search(shakeStrategy, index)
            newStrategy = self.move_or_not(localSearchStrategy)
            
            if newStrategy['TotalTime'] < best['TotalTime']:
                best = copy.deepcopy(newStrategy)
            
            k+=1

        return best, best['TotalTime']
    
