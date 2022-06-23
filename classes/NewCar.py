from typing import List
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import os
import pickle

from classes.Utils import get_basic_logger, VISUAL_COMPOUNDS

P0 = {
    'Soft': (0.1, 270.0),
    'Medium': (0.075, 330.0),
    'Hard': (0.06, 455.0),
    'Inter': (0.05, 800.0),
    'Wet': (0.04, 800.0)
}

BOUNDS = {
    'Soft': ([0.075,250],[0.125,290]),
    'Medium': ([0.05,305],[0.1,355]),
    'Hard': ([0.045,430],[0.085,480]),
    'Inter': ([0.02,600],[1,1000]),
    'Wet':([0.01,600],[1,1000])
}


log = get_basic_logger('Car')

def exponential_fun(x, a, b):
    if isinstance(x, np.ndarray):
        return np.exp(a*x) * b
    return round(np.exp(a*x) * b)

class Car:
    def __init__(self, data:dict=None, load_path:str=None):
        self.data = None
        self.tyre_used:List[str] = []
        self.drs_lose:int = 0
        self.tyre_wear_coeff = {}    
        self.tyre_coeff = {}
        for key in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
            self.tyre_wear_coeff[key] = {}
            self.tyre_coeff[key] = 0
        
        ## Fuel lose

        if data is not None:
            self.data = data
            self.tyre_used = data.keys()
            self.compute_drs_lose(data)
            self.compute_tyre_wear_and_time_lose(data)

        if load_path is not None:
            self.data = data
            self.tyre_used = data.tyre_used
            self.drs_lose = data.drs_lose

    def compute_drs_lose(self, data:dict):
        self.drs_lose = 800

        for _, val in data.items():
            if len(val) > 1:
                drs = []
                no_drs = []
                for t_data in val:
                    if any(t_data['DRS'] == True):
                        drs = t_data[t_data['DRS'] == True]
                    if any(t_data['DRS'] == False): 
                        no_drs = t_data[t_data['DRS'] == False]

                if len(drs) > 0 and len(no_drs) > 0:
                    max_len = min(len(drs['LapTime'].values), len(no_drs['LapTime'].values))
                    self.drs_lose = round(np.mean(np.array(no_drs['LapTime'].values)[:max_len] - np.array(drs['LapTime'].values)[:max_len]))
        

    def compute_tyre_wear_and_time_lose(self, data:dict):
        alpha = .1
        tyre_wear = {}    
        for key in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
            tyre_wear[key] = {'FL':[], 'FR':[], 'RL':[], 'RR':[]}

        for tyre, val in data.items():
            for t_data in val:
                tyre_wear[tyre] = {'FL':[], 'FR':[], 'RL':[], 'RR':[]}
                for fl, fr, rl, rr in t_data[['FLWear', 'FRWear', 'RLWear', 'RRWear']].values:
                    tyre_wear[tyre]['FL'].append(fl)
                    tyre_wear[tyre]['FR'].append(fr) 
                    tyre_wear[tyre]['RL'].append(rl) 
                    tyre_wear[tyre]['RR'].append(rr) 
                
                t_wear_coeff = {'FL':0, 'FR':0, 'RL':0, 'RR':0}
                if len(self.tyre_wear_coeff[tyre]) > 0: # 
                    t_wear_coeff['FL'] = self.tyre_wear_coeff[tyre]['FL']
                    t_wear_coeff['FR'] = self.tyre_wear_coeff[tyre]['FR']
                    t_wear_coeff['RL'] = self.tyre_wear_coeff[tyre]['RL']
                    t_wear_coeff['RR'] = self.tyre_wear_coeff[tyre]['RR']

                for key in ['FL', 'FR', 'RL', 'RR']:
                    self.tyre_wear_coeff[tyre][key] = np.polyfit(np.arange(1,len(tyre_wear[tyre][key])+1), tyre_wear[tyre][key], 1)
                    
                    if not isinstance(t_wear_coeff[key], int):
                        self.tyre_wear_coeff[tyre][key] = [(self.tyre_wear_coeff[tyre][key][0] + t_wear_coeff[key][0])/2,(self.tyre_wear_coeff[tyre][key][1] + t_wear_coeff[key][1])/2]

                #self.tyre_coeff[tyre],_ = curve_fit(exponential_fun,  np.arange(1,len(softTimeLose)+1), softTimeLose, p0=P0[tyre],  bounds=BOUNDS[tyre], sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(softTimeLose)], maxfev=1000)
        
        #### CHECK WHAT TO DO IF WEAR COEFFS CANNOT BE COMPUTED
        ####
        # 
        # C'è un problema: va computato il time perso sulla base della percentuale di usura, inutile sapere che dopo 10 giri perdo 10 secondi se ho le gomme al 1% di usura
        # Va verificato se è esponenziale o lineare, perché ho qualche dubbio 
        #
        #### 

    def predict_tyre_wear(self, tyre:str, lap:int):
        fl = round(self.tyre_wear_coeff[tyre]['FL'][0] * lap + self.tyre_wear_coeff[tyre]['FL'][1])
        fr = round(self.tyre_wear_coeff[tyre]['FR'][0] * lap + self.tyre_wear_coeff[tyre]['FR'][1])
        rl = round(self.tyre_wear_coeff[tyre]['RL'][0] * lap + self.tyre_wear_coeff[tyre]['RL'][1])
        rr = round(self.tyre_wear_coeff[tyre]['RR'][0] * lap + self.tyre_wear_coeff[tyre]['RR'][1])

        return {'FL':fl, 'FR':fr, 'RL':rl, 'RR':rr}




def get_nearest_frame(df, frameList):
    framesReturn = []
    for frame in frameList:
        if frame in df['FrameIdentifier'].values:
            framesReturn.append(frame)
        else:
            notFound = True
            add = 1
            while notFound:
                if (frame + add) in df['FrameIdentifier'].values:
                    framesReturn.append(frame + add)
                    notFound = False
                add += 1

    return framesReturn

def get_data(folder:str, add_data:pd.DataFrame=None, ignore_frames:list=[]):
    lap = pd.read_csv(os.path.join(folder, "Lap.csv"))
    lap = lap.loc[lap["CarIndex"] == 19, ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")
    lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    to_drop = ignore_frames
    if to_drop == []:
        to_drop = input(f"Lap DataFrame is the following:\n{lap}\nIf there are some wrong frames, insert now separated by comma or press ENTER if None: ")
        if to_drop:
            to_drop = np.array(to_drop.split(','), dtype=int)

    for index in to_drop:
        if index in lap.index:
            lap = lap.drop(index)

    sub = min(lap['CurrentLapNum'])-1
    for i in lap.index:
        lap.at[i,'CurrentLapNum'] = int(lap.at[i,'CurrentLapNum'])-sub

    lap_frames = lap.index.values
    ### Must be corrected in 'if between frame start and frame end drs is opened then consider as opened for the whole lap'
    telemetry = pd.read_csv(os.path.join(folder, "Telemetry.csv"))
    telemetry = telemetry.loc[telemetry["CarIndex"] == 19, ['FrameIdentifier', 'DRS']].drop_duplicates(['FrameIdentifier'], keep="last")
    telemetry_frames = get_nearest_frame(telemetry, lap_frames)
    telemetry = telemetry.loc[telemetry['FrameIdentifier'].isin(telemetry_frames)].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData = pd.concat([lap, telemetry], axis=1)

    status = pd.read_csv(os.path.join(folder, "Status.csv"))
    status = status.loc[status["CarIndex"] == 19, ['FrameIdentifier','FuelInTank','VisualTyreCompound']].drop_duplicates(['FrameIdentifier'], keep="last")
    status_frames = get_nearest_frame(status, lap_frames)
    status = status.loc[status['FrameIdentifier'].isin(status_frames), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")
    
    for i in status.index:
        status.at[i,'VisualTyreCompound'] = VISUAL_COMPOUNDS[status.at[i,'VisualTyreCompound']]

    lap.index = status_frames
    concatData = pd.concat([concatData, status], axis=1)

    damage = pd.read_csv(os.path.join(folder, "Damage.csv"))
    damage = damage.loc[damage["CarIndex"] == 19, ['FrameIdentifier', 'TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR',]].drop_duplicates(['FrameIdentifier'], keep="last")
    damage_frames = get_nearest_frame(damage, lap_frames)
    damage = damage.loc[damage['FrameIdentifier'].isin(damage_frames), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData.index = damage_frames
    concatData = pd.concat([concatData, damage], axis=1)
    
    concatData.rename(columns={"CurrentLapNum": "Lap", "LastLapTimeInMS": "LapTime", 'FuelInTank':'Fuel', 'VisualTyreCompound':'Compound','TyresWearFL':'FLWear', 'TyresWearFR':'FRWear', 'TyresWearRL':'RLWear', 'TyresWearRR':'RRWear'}, inplace=True)

    tyres = concatData['Compound'].unique()

    ret = {}
    if add_data is not None:
        ret = add_data

    for tyre in tyres:
        if tyre not in ret.keys():
            ret[tyre] = []
        data = concatData.loc[concatData['Compound'] == tyre, :]
        sub = min(data['Lap'])-1
        if sub > 0:
            for i in data.index:
                data.at[i,'Lap'] = int(data.at[i,'Lap'])-sub

        ret[tyre].append(data)

    return ret


def get_car_data(path:str):
    car = None
    if os.path.isfile(os.path.join(path, 'Car.json')):
        car = Car(load_path=os.path.join(path, 'Car.json'))
    else:    
        if os.path.isfile(os.path.join(path, 'Data.json')):
            with open(os.path.join(path, 'Data.json'), 'rb') as f:
                data = pickle.load(f)

        else:   
            fp1_folder = os.path.join(path, 'FP1\Acquired_data')
            fp2_folder = os.path.join(path, 'FP2\Acquired_data')
            fp3_folder = os.path.join(path, 'FP3\Acquired_data')
            
            fp1 = get_data(fp1_folder, add_data=None, ignore_frames=[8207,14288,37444,43260])
            fp2 = get_data(fp2_folder, add_data=fp1, ignore_frames=[10,12731,12872])
            fp3 = get_data(fp3_folder, add_data=fp2, ignore_frames=[0,6838,6935,17777,32400,37341])

            concatenated = None
            for _, item in fp3.items():
                for i in item:
                    if concatenated is None:
                        concatenated = i
                    else:
                        concatenated = pd.concat([concatenated, i])
            
            concatenated.to_csv(os.path.join(path, 'FullData.csv'), index=False)

            with open(os.path.join(path, 'Data.json'), 'wb') as f:
                pickle.dump(fp3, f, pickle.HIGHEST_PROTOCOL)

            data = fp3.copy()

        car = Car(data=data)
    
    return car
