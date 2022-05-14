import math
import os
import pickle
import sys
from typing import Dict, List
import numpy as np
from pyparsing import col
from classes.Fuel import Fuel
from classes.Timing import Timing
from classes.Tyres import Tyres
import plotly.express as px
import pandas as pd

from classes.Utils import get_basic_logger

log = get_basic_logger('Cars')

class Car:
    def __init__(self,base_path:str=None, car_id:int=None, load_path:str=None):
        if load_path is None:
            self.car_id = car_id
            self.fuel_lose = 0
            self.tyre_lose = {'Soft':0, 'Medium':0, 'Hard':0, 'Inter':0, 'Wet':0}
            self.wear_coeff = {'Soft':0, 'Medium':0, 'Hard':0, 'Inter':0, 'Wet':0}
            self.fuel:List[Fuel] = list()
            self.tyres:List[Tyres] = list()
            self.timing:List[Timing] = list()

            if base_path is not None:
                self.add(base_path, car_id)
        else:
            data:Car = self.load(load_path)
            self.car_id = data.car_id
            self.fuel_lose = data.fuel_lose
            self.tyre_lose = data.tyre_lose
            self.wear_coeff = data.wear_coeff
            self.fuel = data.fuel
            self.tyres = data.tyres
            self.timing = data.timing



    def add(self,base_path:str, car_id:int):
        folders = os.listdir(base_path)
        if 'CarSaves' in folders:
            folders.remove('CarSaves')

        for folder in folders:
            path = os.path.join(base_path,folder+"/Saves/"+str(car_id))
            if not os.path.exists(path):
                print("Car {} does not exist in '{}'".format(car_id, path))
                return 

            fuel_data = list()
            tyres_data = list()
            timing_data = list()

            for file in os.listdir(path):
                if file.endswith(".json"):
                    if file.startswith("Fuel"):
                        fuel_data.append(file)
                    elif file.startswith("Timing"):
                        timing_data.append(file)
                    elif file.startswith("Tyres"):
                        tyres_data.append(file)

            for file in fuel_data:
                self.fuel.append(Fuel(load_path=os.path.join(path,file)))
            for file in timing_data:
                self.timing.append(Timing(load_path=os.path.join(path,file)))
            for file in tyres_data:
                self.tyres.append(Tyres(load_path=os.path.join(path,file)))

        fuel_lose = self.get_fuel_time_lose()
        if fuel_lose is not None and (self.fuel_lose > fuel_lose or self.fuel_lose == 0):
            self.fuel_lose = fuel_lose

        tyre_lose = self.get_tyre_time_lose()
        if tyre_lose is not None:
            for key in tyre_lose:
                if tyre_lose.get(key) is not None and (self.tyre_lose[key] > tyre_lose[key] or self.tyre_lose[key] == 0):
                    self.tyre_lose[key] = tyre_lose[key]

        self.get_wear_time_lose()

    def compute_wear_time_lose(self, wear_percentage, tyre_compound):
        log_y = self.wear_coeff[tyre_compound][0] + self.wear_coeff[tyre_compound][1] * wear_percentage 
        return np.exp(log_y)

    def get_wear_time_lose(self,):
        for idx, tyre in enumerate(self.tyres):
            time_lose = list()

            max_fuel_in_tank = round(max([x for x in self.fuel[idx].FuelInTank.values() if not math.isnan(x)]))
            best_time = int(self.timing[idx].BestLapTime - (max_fuel_in_tank * self.fuel_lose))
            
            if len(self.timing[idx].LapTimes)> 1:
                
                for jdx, time in enumerate(self.timing[idx].LapTimes):      
                    fuel_in_tank = self.fuel[idx].FuelInTank[self.fuel[idx].get_frame(jdx)]
                    if math.isnan(fuel_in_tank):
                        fuel_in_tank = max_fuel_in_tank
                        
                    
                    lose = time - (round(fuel_in_tank * self.fuel_lose) + best_time)
                    #print(f"{lose} = {time} - {round(fuel_in_tank * self.fuel_lose)} + {best_time}")
                    time_lose.append((round(lose),tyre.get_avg_wear(lap=jdx)))
                
                x = []
                for _, wear in time_lose:
                    if math.isnan(wear):
                        x.append(0)
                    else:
                        x.append(wear)

                y = []
                for time, _ in time_lose:
                    if time < 1:
                        y.append(1)
                    else:
                        y.append(time)
                
                
                try:
                    fit = np.polyfit(x, np.log(y), 1)
                except:
                    log.error("Unable to fit data, skipping it.\nx ({}) = {}\ny ({}) = {}".format(len(x),[round(_x) for _x in x],len(y),[round(_y) for _y in y]))
                
                coeff = self.wear_coeff[tyre.get_visual_compound()]
                if isinstance(coeff,int) and coeff == 0:
                    self.wear_coeff[tyre.get_visual_compound()] = (fit[1],fit[0])
                elif isinstance(coeff,tuple):
                    x = np.random.randint(0,80)
                    y_1 = np.exp(fit[1] + fit[0] * x)
                    y_2 = np.exp(coeff[1] + coeff[0] * x)

                    if y_1 > y_2:
                        self.wear_coeff[tyre.get_visual_compound()] = (fit[1],fit[0])

    def get_same_tyres(self,):
        for i in range(len(self.tyres)):
            for j in range(i, len(self.tyres)):
                if self.tyres[i].get_visual_compound() == self.tyres[j].get_visual_compound() and i != j:
                    return i, j

        return None, None

    def get_fuel_time_lose(self,):
        time_lose = 0
        fuel_diff = 0
        idx_1, idx_2 = self.get_same_tyres()
        
        if idx_1 is not None and idx_2 is not None:
            timing_1 = self.timing[idx_1].LapTimes
            timing_2 = self.timing[idx_2].LapTimes

            fuel_1_df = pd.DataFrame(self.fuel[idx_1].consumption())
            fuel_2_df = pd.DataFrame(self.fuel[idx_2].consumption())

            for row in fuel_1_df.index:
                try:
                    fuel_1_df.at[row,'Lap'] = int(fuel_1_df.at[row,'Lap'])
                except KeyError:
                    pass
            
            for row in fuel_2_df.index:
                try:
                    fuel_2_df.at[row,'Lap'] = int(fuel_2_df.at[row,'Lap'])
                except KeyError:
                    pass
            
            fuel_1_df.drop_duplicates(subset=['Lap'], keep='first', inplace=True)
            fuel_2_df.drop_duplicates(subset=['Lap'], keep='first', inplace=True)

            fuel_1 = list()
            fuel_2 = list()

            for lap in fuel_1_df['Lap'].values:
                value = fuel_1_df.loc[fuel_1_df['Lap'] == lap]['Fuel'].values[0]
                fuel_1.append(value if not math.isnan(value) else fuel_1_df.loc[fuel_1_df['Lap'] == lap+1]['Fuel'].values[0])
            
            for lap in fuel_2_df['Lap'].values:
                value = fuel_2_df.loc[fuel_2_df['Lap'] == lap]['Fuel'].values[0]
                fuel_2.append(value if not math.isnan(value) else 0)

            for i in range(min([len(fuel_1), len(fuel_2), len(timing_1), len(timing_2)])):
                time_lose += abs(timing_1[i] - timing_2[i])
                fuel_diff += abs(fuel_1[i] - fuel_2[i])
            
            return round(time_lose/fuel_diff)
        
        return None

    def get_same_fuel(self,) -> tuple:
        for i in range(len(self.fuel)):
            for j in range(i, len(self.fuel)):
                if i != j:
                    val1 = self.fuel[i].FuelInTank.first()[1]
                    if math.isnan(val1):
                        val1 = -1
                    val2 = self.fuel[j].FuelInTank.first()[1]
                    if math.isnan(val2):
                        val2 = -1
                    if round(val1) == round(val2):
                        return i, j
        
        return None, None

    def get_tyre_time_lose(self,):
        time_lose = 0

        idx_1, idx_2 = self.get_same_fuel()

        faster = 0

        if idx_1 is not None and idx_2 is not None:
            timing_1 = self.timing[idx_1].LapTimes
            timing_2 = self.timing[idx_2].LapTimes

            tyre_1 = self.tyres[idx_1].get_visual_compound()
            tyre_2 = self.tyres[idx_2].get_visual_compound()

            for i in range(min([len(timing_1), len(timing_2)])):
                time_lose += abs(timing_1[i] - timing_2[i])
                if timing_1[i] < timing_2[i]:
                    faster += 1
            
            time_lose /= min([len(timing_1), len(timing_2)])
            if faster > min([len(timing_1), len(timing_2)])/2:
                return {tyre_1:0, tyre_2: round(time_lose)}
            return {tyre_2:0, tyre_1: round(time_lose)}

    def save(self, save_path:str='', id:int=0) -> None:
        save_path = os.path.join(save_path,'Car_'+str(id)+'.json')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str=''):
        with open(path, 'rb') as f:
            return pickle.load(f)

def get_cars(path:str=None, load_path:str=None) -> dict:
    cars:Dict[int,Car] = dict()

    files = []
    if load_path is not None and os.path.exists(load_path):
        files = os.listdir(load_path)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
    

    if load_path is not None and len(files) > 0:
        log.info("Loading Cars Data from '{}'.".format(load_path))
        for idx in range(0,20):
            try:    
                cars[idx] = Car(load_path=os.path.join(load_path,'Car_'+str(idx)+'.json'))
                log.info("Car {} loaded successfully.".format(idx))
            except FileNotFoundError:
                log.info("Car {} not found.".format(idx))
                cars[idx] = Car()
    elif path is not None:
        path = os.path.abspath(path)
        if os.name == 'posix' and path.split('/')[-2] != 'Data':
            path = path.split('/')
            while path[-2] != 'Data':
                path = path[:-1]
            new_path = ''
            for p in path:
                new_path += p + '/'
            path = new_path
        elif os.name == 'nt' and path.split('\\')[-2] != 'Data':
            path = path.split('\\')
            while path[-2] != 'Data':
                path = path[:-1]
            new_path = ''
            for p in path:
                new_path += p + '\\'
            path = new_path
        
        
        save_path = os.path.join(path,'CarSaves')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        log.info("Ready to create Cars Data, will be saved in '{}'.".format(save_path))
        
        for idx in range(0,20):
            log.info("Creating Car {}...".format(idx))
            if cars.get(idx) is None:
                cars[idx] = Car()
            cars[idx].add(path, idx)
            cars[idx].save(save_path, idx)

        log.info("Cars Data created and saved successfully.")

    return cars

