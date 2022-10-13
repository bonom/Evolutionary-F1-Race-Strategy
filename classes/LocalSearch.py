import math, time, copy, os
import numpy as np
import pandas as pd
from random import SystemRandom
from tqdm import tqdm

from classes.Car import Car
from classes.Weather import Weather
from classes.Genetic import Genetic
random = SystemRandom()

from classes.Utils import CIRCUIT, Log, ms_to_time

TYRE_WEAR_THRESHOLD = 0.3
BEST_TIME = np.inf
STRATEGY = None

class LocalSearch:

    def shake(strategy:dict):
        
        return strategy

    def local_search(strategy:dict):
        
        return strategy
    
    def move_or_not(strategy:dict):

        return strategy

    def run(self, strategy: dict):
        best = copy.deepcopy(strategy)
        k = 0

        while k < len(best['TyreCompound']):
            shakeStrategy = self.shake(strategy)
            localSearchStrategy = self.local_search(strategy)
            newStrategy = self.move_or_not(strategy)
            
            if newStrategy['TotalTime'] < best['TotalTime']:
                best = newStrategy

        return best
    
