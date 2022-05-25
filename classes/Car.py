import math
import os
import pickle
from typing import Dict, List
import numpy as np
from classes.Fuel import Fuel
from classes.Timing import Timing
from classes.Tyres import Tyres
import pandas as pd

from classes.Utils import STINTS, get_basic_logger

log = get_basic_logger('Cars')

def emptyTuple(d:dict):
    for _, val in d.items():
        if val != (0,0):
            return False
    return True

def getKey(dictionary:dict, value:int) -> int:
    for key, val in dictionary.items():
        if val == value:
            return key

class Car:
    def __init__(self,base_path:str=None, car_id:int=None, load_path:str=None):
        if load_path is None:
            self.car_id = car_id
            self.fuel_lose = 0.0
            self.tyre_lose = {'Soft':0, 'Medium':0, 'Hard':0, 'Inter':0, 'Wet':0}
            self.wear_coeff = {'Soft':None, 'Medium':None, 'Hard':None, 'Inter':None, 'Wet':None}
            self.fuel_coeff = (0,0)
            self.tyre_coeff = {'Soft':(0,0), 'Medium':(0,0), 'Hard':(0,0), 'Inter':(0,0), 'Wet':(0,0)}
            self.fuel:List[Fuel] = list()
            self.tyres:List[Tyres] = list()
            self.timing:List[Timing] = list()

            base_wear:dict = {'FL':(0,0), 'FR':(0,0), 'RL':(0,0), 'RR':(0,0)}
            for key, _ in self.wear_coeff.items():
                self.wear_coeff[key] = base_wear.copy()

            if base_path is not None:
                self.add(base_path, car_id)
        else:
            data:Car = self.load(load_path)
            self.car_id:int = data.car_id
            self.fuel_lose:float = data.fuel_lose
            self.tyre_lose:dict = data.tyre_lose
            self.wear_coeff:dict = data.wear_coeff
            self.fuel:List[Fuel] = data.fuel
            self.tyres:List[Tyres] = data.tyres
            self.timing:List[Timing] = data.timing

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

        fuel_coeff = [fuel.coeff for fuel in self.fuel]

        for coeff in fuel_coeff:
            self.fuel_coeff = (self.fuel_coeff[0]+coeff[0], self.fuel_coeff[1]+coeff[1])
        self.fuel_coeff = (self.fuel_coeff[0]/len(fuel_coeff), self.fuel_coeff[1]/len(fuel_coeff))
        
        count = {'Soft':0, 'Medium':0, 'Hard':0, 'Inter':0, 'Wet':0}

        for key, _ in self.wear_coeff.items():
            for tyre in self.tyres:
                if tyre.get_visual_compound() == key:
                    if emptyTuple(self.wear_coeff[key]):
                        self.wear_coeff[key] = tyre.wear_coeff
                        count[key] += 1
                    else:
                        self.wear_coeff[key]['FL'] = (self.wear_coeff[key]['FL'][0]+tyre.wear_coeff['FL'][0], self.wear_coeff[key]['FL'][1]+tyre.wear_coeff['FL'][1])
                        self.wear_coeff[key]['FR'] = (self.wear_coeff[key]['FR'][0]+tyre.wear_coeff['FR'][0], self.wear_coeff[key]['FR'][1]+tyre.wear_coeff['FR'][1])
                        self.wear_coeff[key]['RL'] = (self.wear_coeff[key]['RL'][0]+tyre.wear_coeff['RL'][0], self.wear_coeff[key]['RL'][1]+tyre.wear_coeff['RL'][1])
                        self.wear_coeff[key]['RR'] = (self.wear_coeff[key]['RR'][0]+tyre.wear_coeff['RR'][0], self.wear_coeff[key]['RR'][1]+tyre.wear_coeff['RR'][1])
                        count[key] += 1
                        
        for key, _ in self.wear_coeff.items():
            if count[key] > 1:
                self.wear_coeff[key]['FL'] = (self.wear_coeff[key]['FL'][0]/count[key], self.wear_coeff[key]['FL'][1]/count[key])
                self.wear_coeff[key]['FR'] = (self.wear_coeff[key]['FR'][0]/count[key], self.wear_coeff[key]['FR'][1]/count[key])
                self.wear_coeff[key]['RL'] = (self.wear_coeff[key]['RL'][0]/count[key], self.wear_coeff[key]['RL'][1]/count[key])
                self.wear_coeff[key]['RR'] = (self.wear_coeff[key]['RR'][0]/count[key], self.wear_coeff[key]['RR'][1]/count[key])
                
        for key, val in self.wear_coeff.items():
            if emptyTuple(val):
                idx = getKey(STINTS, key)
                
                if idx != 0 and idx < len(STINTS)-1 and not emptyTuple(self.wear_coeff[STINTS[idx-1]]) and not emptyTuple(self.wear_coeff[STINTS[idx+1]]):
                    self.wear_coeff[key]['FL'] = ((self.wear_coeff[STINTS[idx-1]]['FL'][0]+self.wear_coeff[STINTS[idx+1]]['FL'][0])/2, (self.wear_coeff[STINTS[idx-1]]['FL'][1]+self.wear_coeff[STINTS[idx+1]]['FL'][1])/2)
                    self.wear_coeff[key]['FR'] = ((self.wear_coeff[STINTS[idx-1]]['FR'][0]+self.wear_coeff[STINTS[idx+1]]['FR'][0])/2, (self.wear_coeff[STINTS[idx-1]]['FR'][1]+self.wear_coeff[STINTS[idx+1]]['FR'][1])/2)
                    self.wear_coeff[key]['RL'] = ((self.wear_coeff[STINTS[idx-1]]['RL'][0]+self.wear_coeff[STINTS[idx+1]]['RL'][0])/2, (self.wear_coeff[STINTS[idx-1]]['RL'][1]+self.wear_coeff[STINTS[idx+1]]['RL'][1])/2)
                    self.wear_coeff[key]['RR'] = ((self.wear_coeff[STINTS[idx-1]]['RR'][0]+self.wear_coeff[STINTS[idx+1]]['RR'][0])/2, (self.wear_coeff[STINTS[idx-1]]['RR'][1]+self.wear_coeff[STINTS[idx+1]]['RR'][1])/2)                   

        fuel_lose = self.get_fuel_time_lose()
        if fuel_lose is not None and (self.fuel_lose > fuel_lose or self.fuel_lose == 0):
            self.fuel_lose = fuel_lose

        tyre_lose = self.get_tyre_time_lose()
        if tyre_lose is not None:
            for key in tyre_lose:
                if tyre_lose.get(key) is not None and (self.tyre_lose[key] > tyre_lose[key] or self.tyre_lose[key] == 0):
                    self.tyre_lose[key] = tyre_lose[key]

        self.get_wear_time_lose()

    def get_best_lap_time(self,):
        return min(self.timing, key=lambda x: x.BestLapTime).BestLapTime
    
    def get_tyre_wear(self, tyre_compound:str, lap:int) -> dict:
        if lap == 0:
            return {'FL':0, 'FR':0, 'RL':0, 'RR':0}

        wear_FL = self.wear_coeff[tyre_compound]['FL'][0]*lap + self.wear_coeff[tyre_compound]['FL'][1]
        wear_FR = self.wear_coeff[tyre_compound]['FR'][0]*lap + self.wear_coeff[tyre_compound]['FR'][1]
        wear_RL = self.wear_coeff[tyre_compound]['RL'][0]*lap + self.wear_coeff[tyre_compound]['RL'][1]
        wear_RR = self.wear_coeff[tyre_compound]['RR'][0]*lap + self.wear_coeff[tyre_compound]['RR'][1]

        return {'FL':wear_FL, 'FR':wear_FR, 'RL':wear_RL, 'RR':wear_RR}
        

    def compute_wear_time_lose(self, wear_percentage, tyre_compound):
        log_y = self.wear_coeff[tyre_compound][0] + self.wear_coeff[tyre_compound][1] * wear_percentage 
        return np.exp(log_y)

    def get_wear_time_lose(self,):
        for time, fuel, tyre in zip(self.timing, self.fuel, self.tyres):            
            laps = list(time.lap_frames.keys())[1:-1]
            deltas = []
            
            x_FL = [self.get_tyre_wear(tyre.get_visual_compound(),lap)['FL'] for lap in laps]
            x_FR = [self.get_tyre_wear(tyre.get_visual_compound(),lap)['FR'] for lap in laps]
            x_RL = [self.get_tyre_wear(tyre.get_visual_compound(),lap)['RL'] for lap in laps]
            x_RR = [self.get_tyre_wear(tyre.get_visual_compound(),lap)['RR'] for lap in laps]



            for lap, delta in enumerate(time.Deltas):
                deltas.append(delta-(self.fuel_lose * fuel.FuelInTank[fuel.get_frame(lap)]))
            
            deltas_min = min(deltas)

            time_lose = []
            for delta in deltas:
                time_lose.append(round(delta+abs(deltas_min)))

            FL_wear_lose = np.polyfit(x_FL, np.log(time_lose), 1, w=np.sqrt(time_lose))
            FR_wear_lose = np.polyfit(x_FR, np.log(time_lose), 1, w=np.sqrt(time_lose))
            RL_wear_lose = np.polyfit(x_RL, np.log(time_lose), 1, w=np.sqrt(time_lose))
            RR_wear_lose = np.polyfit(x_RR, np.log(time_lose), 1, w=np.sqrt(time_lose))
            
            a = np.exp(FL_wear_lose[1])
            b = FL_wear_lose[0]
            y_fitted = a * np.exp(b * x_FL)
            
            import matplotlib.pyplot as plt
            poly = np.poly1d(FL_wear_lose)
            new_y = poly(x_FL)

            plt.plot(x_FL, np.log(time_lose), 'o')
            plt.show()
            plt.plot(x_FL, new_y, 'n')
            plt.show()
        exit()
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
                    fit = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
                    # from matplotlib import pyplot as plt
                    # plt.plot(x,y,'o')
                    # plt.show()
                    # trendpoly = np.poly1d(fit)
                    # plt.plot(x,trendpoly(x))
                    # plt.show()
                except:
                    log.error("Unable to fit data, skipping it.\nx ({}) = {}\ny ({}) = {}".format(len(x),[round(_x) for _x in x],len(y),[round(_y) for _y in y]))
                #exit(0)
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

def get_cars(path:str=None, load_path:str=None, car_idx:int = None) -> dict:
    cars:Dict[int,Car] = dict()

    files = []
    if load_path is not None and os.path.exists(load_path):
        files = os.listdir(load_path)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
    

    if load_path is not None and len(files) > 0:
        log.info("Loading Car(s) Data from '{}'.".format(load_path))
        if car_idx is None:
            for idx in range(0,20):
                try:    
                    cars[idx] = Car(load_path=os.path.join(load_path,'Car_'+str(idx)+'.json'))
                    log.info("Car {} loaded successfully.".format(idx))
                except FileNotFoundError:
                    log.info("Car {} not found.".format(idx))
                    cars[idx] = Car()
        else:
            return Car(load_path=os.path.join(load_path,'Car_'+str(car_idx)+'.json'))
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

        log.info("Ready to create Car(s) Data, will be saved in '{}'.".format(save_path))
        
        if car_idx is None:
            for idx in range(0,20):
                log.info("Creating Car {}...".format(idx))
                if cars.get(idx) is None:
                    cars[idx] = Car()
                cars[idx].add(path, idx)
                cars[idx].save(save_path, idx)
        else:
            car = Car()
            car.add(path, car_idx)
            #car.save(save_path, car_idx)
            car.get_wear_time_lose()

            log.info("Car Data created and saved successfully.")

            return car

        log.info("Cars Data created and saved successfully.")

    return cars

