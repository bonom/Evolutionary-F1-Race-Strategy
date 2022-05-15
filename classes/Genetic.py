import numpy as np
from classes.Utils import int_to_gray, gray_to_int

class GeneticSolver:
    def __init__(self, generations:int=1, mutation_pr:float=0.0, crossover_pr:float=0.0) -> None:
        self.bestLapTime = {'Soft': np.inf, 'Medium': np.inf, 'Hard': np.inf}
        self.coeff = {'Soft': (0,0), 'Medium': (0,0), 'Hard': (0,0)}
        self.pitStopTime = 0
        self.sigma = mutation_pr
        self.mu = crossover_pr
        self.tyres_history = [[] for _ in range(generations)]
        self.tyres_wear_history = [[] for _ in range(generations)]
        self.fuel_load_history = [[] for _ in range(generations)]
        self.pit_stop_history = [[] for _ in range(generations)]


    def lapTime(self, stint:str, wear:float, fuel_load:float, pitStop:bool) -> int:
        return round(self.bestLapTime[stint] + self.compute_wear_time_lose(wear_percentage=wear,tyre_compound=stint) + self.coeff * fuel_load + self.pitStopTime * pitStop)

    def compute_wear_time_lose(self, wear_percentage:float, tyre_compound:str):
        log_y = self.coeff[tyre_compound][0] + self.coeff[tyre_compound][1] * wear_percentage 
        return np.exp(log_y)
    
    def mutation(self,):
        """
        Self-adaptive (correlated mutations with multiple σ)
        """
        return
    
    def crossover(self,):
        return

    def init_solver(self,):
        return 
    
    def startSolver(self,):
        return