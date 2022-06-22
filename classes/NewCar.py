from typing import List
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import os
import pickle

from classes.Utils import get_basic_logger, VISUAL_COMPOUNDS

log = get_basic_logger('Car')

class Car:
    def __init__(self, data:dict=None, load_path:str=None):
        self.data = None
        self.tyre_used:List[str] = []
        self.drs_lose:int = 0
        self.tyre_wear_coeff = {}    
        for key in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
            self.tyre_wear_coeff[key] = {'FL':[], 'FR':[], 'RL':[], 'RR':[]}

        if data is not None:
            self.data = data
            self.tyre_used = data.keys()
            self.drs_lose = self.compute_drs_lose(data)
            self.tyre_wear = self.compute_tyre_wear(data)

        if load_path is not None:
            self.data = data
            self.tyre_used = data.tyre_used
            self.drs_lose = data.drs_lose

    def compute_drs_lose(self, data:pd.DataFrame):
        drs_lose = 800

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
                    drs_lose = round(np.mean(np.array(no_drs['LapTime'].values)[:max_len] - np.array(drs['LapTime'].values)[:max_len]))
        

        return drs_lose

    def compute_tyre_wear(self, data:pd.DataFrame):
        tyre_wear = {}    
        for key in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
            tyre_wear[key] = {'FL':[], 'FR':[], 'RL':[], 'RR':[]}

        for tyre, val in data.items():
            for t_data in val:
                for fl, fr, rl, rr in t_data[['FLWear', 'FRWear', 'RLWear', 'RRWear']].values:
                    tyre_wear[tyre]['FL'].append(fl)
                    tyre_wear[tyre]['FR'].append(fr) 
                    tyre_wear[tyre]['RL'].append(rl) 
                    tyre_wear[tyre]['RR'].append(rr) 
                
                if all(list(self.tyre_wear_coeff[tyre].values()) == []): #####################!!!!!!!!
                    tyre_wear_coeff_fl = self.tyre_wear_coeff[tyre]['FL']
                    tyre_wear_coeff_fr = self.tyre_wear_coeff[tyre]['FR']
                    tyre_wear_coeff_rl = self.tyre_wear_coeff[tyre]['RL']
                    tyre_wear_coeff_rr = self.tyre_wear_coeff[tyre]['RR']

                    print(tyre_wear_coeff_fl, tyre_wear_coeff_fr, tyre_wear_coeff_rl, tyre_wear_coeff_rr)
                    exit()
                for key in ['FL', 'FR', 'RL', 'RR']:
                    self.tyre_wear_coeff[tyre][key] = np.polyfit(t_data['Lap'], tyre_wear[tyre][key], 1)




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
