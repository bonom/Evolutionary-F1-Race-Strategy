import numpy as np
from classes.Car import Car
import datetime

from classes.Utils import get_basic_logger

log = get_basic_logger('Genetic')

def convertMillis(ms):
    seconds=(ms/1000)%60
    minutes=(ms/(1000*60))%60
    hours=(ms/(1000*60*60))%24
    ms=ms % 1000
    
    if int(hours) < 1:
        return f"{int(minutes)}:{int(seconds)}.{ms}"
    
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}.{ms}"

class GeneticSolver:
    def __init__(self, population:int=1, mutation_pr:float=0.0, crossover_pr:float=0.0, car:Car=None) -> None:
        self.bestLapTime = car.get_best_lap_time()
        self.tyre_lose = car.tyre_lose
        self.fuel_coeff = car.fuel_lose
        self.coeff = car.wear_coeff
        self.pitStopTime = 20000
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.population = population
        self.strategies = []
        self.numLaps = 60

        self.initSolver()


    def __str__(self) -> str:
        string = ''
        for i in range(self.population):
            string+=f"---------- Individual {i+1} ----------\n"
            for lap in range(self.numLaps):
                string+=f"{lap+1}º LapTime: {convertMillis(self.strategies[i]['LapTime'][lap])} | TyreStint: {self.strategies[i]['TyreStint'][lap]} | TyreWear: {self.strategies[i]['TyreWear'][lap]} | FuelLoad: {self.strategies[i]['FuelLoad'][lap]} | PitStop: {self.strategies[i]['PitStop'][lap]}\n"
            string+=f"TotalTime: {convertMillis(self.strategies[i]['TotalTime'])}\n"
            string+="\n"

        return string
        

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
    
    def mutation(self,):
        """
        Self-adaptive (correlated mutations with multiple σ)
        """
        return
    
    def crossover(self,):
        return

    def initSolver(self,):
        for _ in range(self.population):
            strategy = {'TyreStint': [], 'TyreWear':[] , 'FuelLoad':[] , 'PitStop': [], 'LapTime':[], 'TotalTime': np.inf}
            
            strategy['TyreStint'].append(np.random.choice(['Soft', 'Medium', 'Hard']))
            strategy['TyreWear'].append(0)
            strategy['FuelLoad'].append(np.random.randint(120,200))
            strategy['PitStop'].append(False)
            strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][0], strategy['TyreWear'][0], strategy['FuelLoad'][0], False))
            
            for i in range(1,self.numLaps+1):
                #strategy['TyreStint'].append(np.random.choice(['Soft', 'Medium', 'Hard']))
                if i >= 20:
                    strategy['TyreStint'].append('Hard')
                else:
                    strategy['TyreStint'].append('Soft')

                if strategy['TyreStint'][i] == strategy['TyreStint'][i-1]:
                    strategy['TyreWear'].append(float(strategy['TyreWear'][i-1])+2.5)
                    strategy['PitStop'].append(False)
                else:
                    strategy['TyreWear'].append(0)
                    strategy['PitStop'].append(True)
                
                strategy['FuelLoad'].append(strategy['FuelLoad'][i-1]-3)
                strategy['LapTime'].append(self.lapTime(strategy['TyreStint'][i], strategy['TyreWear'][i], strategy['FuelLoad'][i], strategy['PitStop'][i]))

            strategy['TotalTime'] = sum(strategy['LapTime'])
            self.strategies.append(strategy.copy())
        
        
    def startSolver(self,):
        return

