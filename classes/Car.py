import math
import os
import pickle
from typing import Dict, List
import numpy as np
from classes.Fuel import Fuel
from classes.Timing import Timing
from classes.Tyres import Tyres
import pandas as pd

from classes.Utils import COMPOUNDS, get_basic_logger
log = get_basic_logger('Cars')

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

def emptyTuple(d:dict):
    for _, val in d.items():
        if val != (0,0):
            return False
    return True

def getKey(dictionary:dict, value:int) -> int:
    for key, val in dictionary.items():
        if val == value:
            return key
    
    return None

class Car:
    def __init__(self,base_path:str=None, car_id:int=None, load_path:str=None):
        if load_path is None:
            self.car_id = car_id
            self.fuel_lose:float = 0.0 # ms lose per kg
            self.fuel_coeff = (0,0) # coefficients for fuel consumption (should be decreasing => negative)
            self.wear_coeff = {'Soft':None, 'Medium':None, 'Hard':None, 'Inter':None, 'Wet':None} # coefficients for tyre wear (linear)
            self.tyre_coeff = {'Soft':None, 'Medium':None, 'Hard':None, 'Inter':None, 'Wet':None} # coefficients for ms lose per wear (exponential)
            self.fuel:List[Fuel] = list()
            self.tyres:List[Tyres] = list()
            self.timing:List[Timing] = list()

            _base_:dict = {'FL':(0,0), 'FR':(0,0), 'RL':(0,0), 'RR':(0,0)}
            for key, _ in self.wear_coeff.items():
                self.wear_coeff[key] = _base_.copy()
                self.tyre_coeff[key] = _base_.copy()

            if base_path is not None:
                self.add(base_path, car_id)
        else:
            data:Car = self.load(load_path)
            self.car_id:int = data.car_id
            self.fuel_lose:float = data.fuel_lose
            self.fuel_coeff:tuple = data.fuel_coeff
            self.wear_coeff:dict = data.wear_coeff
            self.tyre_coeff:dict = data.tyre_coeff
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


        self.compute_fuel_coeff()
        self.compute_wear_coeff()
        self.compute_fuel_time_lose()
        self.compute_wear_time_lose()

    def compute_wear_coeff(self,):
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
                idx = getKey(COMPOUNDS, key)
                
                if idx is not None and idx < len(COMPOUNDS)-1 and not emptyTuple(self.wear_coeff[COMPOUNDS[idx-1]]) and not emptyTuple(self.wear_coeff[COMPOUNDS[idx+1]]):
                    self.wear_coeff[key]['FL'] = ((self.wear_coeff[COMPOUNDS[idx-1]]['FL'][0]+self.wear_coeff[COMPOUNDS[idx+1]]['FL'][0])/2, (self.wear_coeff[COMPOUNDS[idx-1]]['FL'][1]+self.wear_coeff[COMPOUNDS[idx+1]]['FL'][1])/2)
                    self.wear_coeff[key]['FR'] = ((self.wear_coeff[COMPOUNDS[idx-1]]['FR'][0]+self.wear_coeff[COMPOUNDS[idx+1]]['FR'][0])/2, (self.wear_coeff[COMPOUNDS[idx-1]]['FR'][1]+self.wear_coeff[COMPOUNDS[idx+1]]['FR'][1])/2)
                    self.wear_coeff[key]['RL'] = ((self.wear_coeff[COMPOUNDS[idx-1]]['RL'][0]+self.wear_coeff[COMPOUNDS[idx+1]]['RL'][0])/2, (self.wear_coeff[COMPOUNDS[idx-1]]['RL'][1]+self.wear_coeff[COMPOUNDS[idx+1]]['RL'][1])/2)
                    self.wear_coeff[key]['RR'] = ((self.wear_coeff[COMPOUNDS[idx-1]]['RR'][0]+self.wear_coeff[COMPOUNDS[idx+1]]['RR'][0])/2, (self.wear_coeff[COMPOUNDS[idx-1]]['RR'][1]+self.wear_coeff[COMPOUNDS[idx+1]]['RR'][1])/2)                   
    
    def compute_wear_time_lose(self,):
        count = {'Soft':0, 'Medium':0, 'Hard':0, 'Inter':0, 'Wet':0}
        for time, fuel, tyre in zip(self.timing, self.fuel, self.tyres):       
            laps = list(time.lap_frames.keys())[1:-1]
            deltas = []
            stint = tyre.get_visual_compound()

            for lap, delta in enumerate(time.Deltas):
                lf = self.fuel_lose * fuel.FuelInTank[fuel.get_frame(lap)]
                if math.isnan(lf):
                    lf = self.fuel_lose * fuel.predict_fuelload(lap)
                deltas.append(delta-lf)


            if len(deltas) != len(laps):
                if len(deltas) < len(laps):
                    laps = laps[:len(deltas)]
                else:
                    deltas = deltas[:len(laps)]

            x_FL = [self.getTyreWear(stint,lap)['FL'] for lap in laps]
            x_FR = [self.getTyreWear(stint,lap)['FR'] for lap in laps]
            x_RL = [self.getTyreWear(stint,lap)['RL'] for lap in laps]
            x_RR = [self.getTyreWear(stint,lap)['RR'] for lap in laps]
            
            deltas_min = min(deltas)

            time_lose = []

            for delta in deltas:
                val = round(delta+abs(deltas_min))
                time_lose.append(val if val != 0 else 1)
            
            ### z is the contribution of the wear to the time loss
            z_FL = []
            z_FR = []
            z_RL = []
            z_RR = []
            
            for fl, fr, rl, rr in zip(x_FL, x_FR, x_RL, x_RR):
                total_wear = fl + fr + rl + rr
                z_FL.append(round(fl/total_wear,2))
                z_FR.append(round(fr/total_wear,2))
                z_RL.append(round(rl/total_wear,2))
                z_RR.append(round(rr/total_wear,2))
                
            ### y is the time loss due to wear
            y_FL = [tl*z_FL[ydx] for ydx, tl in enumerate(time_lose)]
            y_FR = [tl*z_FR[ydx] for ydx, tl in enumerate(time_lose)]
            y_RL = [tl*z_RL[ydx] for ydx, tl in enumerate(time_lose)]
            y_RR = [tl*z_RR[ydx] for ydx, tl in enumerate(time_lose)]

            FL_wear_lose = np.polyfit(x_FL, np.log(y_FL), 1, w=np.sqrt(y_FL))
            FR_wear_lose = np.polyfit(x_FR, np.log(y_FR), 1, w=np.sqrt(y_FR))
            RL_wear_lose = np.polyfit(x_RL, np.log(y_RL), 1, w=np.sqrt(y_RL))
            RR_wear_lose = np.polyfit(x_RR, np.log(y_RR), 1, w=np.sqrt(y_RR))

            if self.tyre_coeff[stint] == 0:
                self.tyre_coeff[stint] = {'FL':(FL_wear_lose[1],FL_wear_lose[0]), 'FR':(FR_wear_lose[1],FR_wear_lose[0]), 'RL':RL_wear_lose, 'RR':RR_wear_lose}
            else:
                FL_coeff_0, FL_coeff_1 = self.tyre_coeff[stint]['FL']
                FR_coeff_0, FR_coeff_1 = self.tyre_coeff[stint]['FR']
                RL_coeff_0, RL_coeff_1 = self.tyre_coeff[stint]['RL']
                RR_coeff_0, RR_coeff_1 = self.tyre_coeff[stint]['RR']
                self.tyre_coeff[stint]['FL'] = (FL_coeff_0+FL_wear_lose[1],FL_coeff_1+FL_wear_lose[0])
                self.tyre_coeff[stint]['FR'] = (FR_coeff_0+FR_wear_lose[1],FR_coeff_1+FR_wear_lose[0])
                self.tyre_coeff[stint]['RL'] = (RL_coeff_0+RL_wear_lose[1],RL_coeff_1+RL_wear_lose[0])
                self.tyre_coeff[stint]['RR'] = (RR_coeff_0+RR_wear_lose[1],RR_coeff_1+RR_wear_lose[0])
            
            count[stint] += 1

        for stint, val in count.items():
            if val > 0:
                FL_coeff_0, FL_coeff_1 = self.tyre_coeff[stint]['FL']
                FR_coeff_0, FR_coeff_1 = self.tyre_coeff[stint]['FR']
                RL_coeff_0, RL_coeff_1 = self.tyre_coeff[stint]['RL']
                RR_coeff_0, RR_coeff_1 = self.tyre_coeff[stint]['RR']

                self.tyre_coeff[stint]['FL'] = (FL_coeff_0/val, FL_coeff_1/val)
                self.tyre_coeff[stint]['FR'] = (FR_coeff_0/val, FR_coeff_1/val)
                self.tyre_coeff[stint]['RL'] = (RL_coeff_0/val, RL_coeff_1/val)
                self.tyre_coeff[stint]['RR'] = (RR_coeff_0/val, RR_coeff_1/val)
            
            if val == 0:
                idx = getKey(COMPOUNDS, stint)
                if idx is not None and idx < len(COMPOUNDS)-1 and not emptyTuple(self.tyre_coeff[COMPOUNDS[idx-1]]) and not emptyTuple(self.tyre_coeff[COMPOUNDS[idx+1]]):
                    self.tyre_coeff[stint]['FL'] = ((self.tyre_coeff[COMPOUNDS[idx-1]]['FL'][0]+self.tyre_coeff[COMPOUNDS[idx+1]]['FL'][0])/2, (self.tyre_coeff[COMPOUNDS[idx-1]]['FL'][1]+self.tyre_coeff[COMPOUNDS[idx+1]]['FL'][1])/2)
                    self.tyre_coeff[stint]['FR'] = ((self.tyre_coeff[COMPOUNDS[idx-1]]['FR'][0]+self.tyre_coeff[COMPOUNDS[idx+1]]['FR'][0])/2, (self.tyre_coeff[COMPOUNDS[idx-1]]['FR'][1]+self.tyre_coeff[COMPOUNDS[idx+1]]['FR'][1])/2)
                    self.tyre_coeff[stint]['RL'] = ((self.tyre_coeff[COMPOUNDS[idx-1]]['RL'][0]+self.tyre_coeff[COMPOUNDS[idx+1]]['RL'][0])/2, (self.tyre_coeff[COMPOUNDS[idx-1]]['RL'][1]+self.tyre_coeff[COMPOUNDS[idx+1]]['RL'][1])/2)
                    self.tyre_coeff[stint]['RR'] = ((self.tyre_coeff[COMPOUNDS[idx-1]]['RR'][0]+self.tyre_coeff[COMPOUNDS[idx+1]]['RR'][0])/2, (self.tyre_coeff[COMPOUNDS[idx-1]]['RR'][1]+self.tyre_coeff[COMPOUNDS[idx+1]]['RR'][1])/2)                   
                ### TODO: check if I have a different stint to use (wet, inter)
                # else:
                #     self.tyre_coeff[stint]['FL'] = (self.tyre_coeff[COMPOUNDS[idx-1]]['FL'][0]+2, self.tyre_coeff[COMPOUNDS[idx-1]]['FL'][1]-1)
                #     self.tyre_coeff[stint]['FR'] = (self.tyre_coeff[COMPOUNDS[idx-1]]['FR'][0]+2, self.tyre_coeff[COMPOUNDS[idx-1]]['FR'][1]-1)
                #     self.tyre_coeff[stint]['RL'] = (self.tyre_coeff[COMPOUNDS[idx-1]]['RL'][0]+2, self.tyre_coeff[COMPOUNDS[idx-1]]['RL'][1]-1)
                #     self.tyre_coeff[stint]['RR'] = (self.tyre_coeff[COMPOUNDS[idx-1]]['RR'][0]+2, self.tyre_coeff[COMPOUNDS[idx-1]]['RR'][1]-1)

    def compute_fuel_coeff(self,):
        fuel_coeff = [-abs(fuel.coeff[0]) for fuel in self.fuel]
        self.fuel_coeff = min(fuel_coeff)

    def compute_fuel_time_lose(self,):
        time_lose = 0
        fuel_diff = 0
        idx_1, idx_2 = self.getSameTyres()
        
        if idx_1 is not None and idx_2 is not None:
            try:    
                timing_1 = self.timing[idx_1].LapTimes
                timing_2 = self.timing[idx_2].LapTimes

                fuel_1_df = pd.DataFrame(self.fuel[idx_1].consumption())
                fuel_2_df = pd.DataFrame(self.fuel[idx_2].consumption())

                for row in fuel_1_df.index:
                    try:
                        fuel_1_df.at[row,'Lap'] = round(fuel_1_df.at[row,'Lap'])
                    except KeyError:
                        pass
                
                for row in fuel_2_df.index:
                    try:
                        fuel_2_df.at[row,'Lap'] = round(fuel_2_df.at[row,'Lap'])
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
                
                self.fuel_lose = round(time_lose/fuel_diff)
                return
            except:
                pass
        
        log.warning("No same tyres found, using standard fuel time lose coefficient (30 ms/kg)")
        self.fuel_lose = 30

    def getBestLapTime(self,):
        return min(self.timing, key=lambda x: x.BestLapTime).BestLapTime

    def getTyreWear(self, tyre_compound:str, lap:int) -> dict:
        if lap == 0:
            return {'FL':0, 'FR':0, 'RL':0, 'RR':0}

        wear_FL = self.wear_coeff[tyre_compound]['FL'][0]*lap + self.wear_coeff[tyre_compound]['FL'][1]
        wear_FR = self.wear_coeff[tyre_compound]['FR'][0]*lap + self.wear_coeff[tyre_compound]['FR'][1]
        wear_RL = self.wear_coeff[tyre_compound]['RL'][0]*lap + self.wear_coeff[tyre_compound]['RL'][1]
        wear_RR = self.wear_coeff[tyre_compound]['RR'][0]*lap + self.wear_coeff[tyre_compound]['RR'][1]

        return {'FL':wear_FL, 'FR':wear_FR, 'RL':wear_RL, 'RR':wear_RR}
        
    def getWearTimeLose(self, tyre_compound:str, lap:int):
        coeff1_FL, coeff2_FL = self.tyre_coeff[tyre_compound]['FL']
        coeff1_FL = np.exp(coeff1_FL)
        y_FL = coeff1_FL * np.exp(coeff2_FL*lap) 
        
        coeff1_FR, coeff2_FR = self.tyre_coeff[tyre_compound]['FR']
        coeff1_FR = np.exp(coeff1_FR)
        y_FR = coeff1_FR * np.exp(coeff2_FR*lap)

        coeff1_RL, coeff2_RL = self.tyre_coeff[tyre_compound]['RL']
        coeff1_RL = np.exp(coeff1_RL)
        y_RL = coeff1_RL * np.exp(coeff2_RL*lap)

        coeff1_RR, coeff2_RR = self.tyre_coeff[tyre_compound]['RR']
        coeff1_RR = np.exp(coeff1_RR)
        y_RR = coeff1_RR * np.exp(coeff2_RR*lap)

        return y_FL + y_FR + y_RL + y_RR
    
    def getInitialFuelLoad(self, total_laps:int) -> float:
        return abs(self.fuel_coeff*total_laps)
    
    def getFuelLoad(self, lap:int, initial_fuel:float) -> float:
        return initial_fuel-abs(self.fuel_coeff*lap)

    def getFuelTimeLose(self, lap:int):
        return self.fuel_lose*lap

    def getSameTyres(self,):
        for i in range(len(self.tyres)-1):
            for j in range(i+1, len(self.tyres)):
                if self.tyres[i].get_visual_compound() == self.tyres[j].get_visual_compound():
                    return i, j

        return None, None

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
            car.save(save_path, car_idx)
            
            log.info("Car Data created and saved successfully.")

            return car

        log.info("Cars Data created and saved successfully.")

    return cars

