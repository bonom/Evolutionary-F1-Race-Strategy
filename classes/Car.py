from random import SystemRandom
import pandas as pd
import numpy as np
import os
import pickle
import plotly.express as px
import math

random = SystemRandom()

from classes.Utils import VISUAL_COMPOUNDS, ms_to_time

def linear_fun(x, a):
    if isinstance(x, np.ndarray):
        return a*x
    return round(a*x)

class Car:
    def __init__(self, data:dict=None, load_path:str=None):
        self.data = None
        self.tyre_used:list = []
        self.drs_lose:int = 0
        self.fuel_lose:int = 0
        self.fuel_consume_coeff:dict = {}
        self.time_diff:dict = {}
        self.tyre_wear_coeff:dict = {}
        self.tyre_coeff:dict = {}
        
        
        if data is not None:
            self.data = data
            self.drs_lose:int = random.randint(500,800)
            self.fuel_lose = random.randint(28,32)
            self.fuel_consume_coeff = {'Dry':0, 'Wet':0}
            self.time_diff = {'Medium':600, 'Hard':1100, 'Inter':6000, 'Wet':9000}
            self.tyre_coeff = {'Soft':{'FL':13, 'FR':13, 'RL':14, 'RR':14}, 'Medium':{'FL':11, 'FR':11, 'RL':12, 'RR':12}, 'Hard':{'FL':9, 'FR':9, 'RL':10, 'RR':10}, 'Inter':{'FL':6, 'FR':6, 'RL':5, 'RR':5}, 'Wet':{'FL':4, 'FR':4, 'RL':3, 'RR':3}}
            for key in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
                self.tyre_wear_coeff[key] = {'FL':0, 'FR':0, 'RL':0, 'RR':0}

            self.extract_tyre_used(data)
            self.compute_fuel_lose(data)
            self.compute_fuel_consume_coeff(data)
            self.compute_drs_lose(data)
            self.compute_tyre_wear_and_time_lose(data)  
            self.compute_time_compound(data)

        if load_path is not None:
            data = self.load(load_path)
            self.data = data.data
            self.tyre_used = data.tyre_used
            self.drs_lose = data.drs_lose
            self.fuel_lose = data.fuel_lose
            self.fuel_consume_coeff = data.fuel_consume_coeff
            self.tyre_wear_coeff = data.tyre_wear_coeff
            self.tyre_coeff = data.tyre_coeff
            self.time_diff = data.time_diff
    
    def extract_tyre_used(self, data:dict):
        for key in data.keys():
            self.tyre_used.append(key)

    def compute_fuel_lose(self, data:dict):
        for _, val in data.items():
            if len(val) > 1:
                fuel_lists = []
                time_list = []
                drs = []
                for idx, t_data in enumerate(val):
                    fuel_lists.append(t_data['Fuel'].values)
                    time_list.append(t_data['LapTime'].values)
                    if any(t_data['DRS'] == True):
                        drs.append(idx)
                
                for i, fi_data in enumerate(fuel_lists):
                    for j, fj_data in enumerate(fuel_lists[i:]):
                        if round(fi_data[0]) != round(fj_data[0]):
                            time_diff = []
                            for x in range(min(len(time_list[i]), len(time_list[j]))):
                                if (i in drs and j not in drs) or (j in drs and i not in drs):
                                    time_diff.append(time_list[i][x]-time_list[j][x]-self.drs_lose)
                                else:
                                    time_diff.append(time_list[i][x]-time_list[j][x])
                                    
                            fuel_diff = [fi_data[x] - fj_data[x] for x in range(min(len(fi_data), len(fj_data)))]
                
                            if self.fuel_lose == 0:
                                self.fuel_lose = np.mean(time_diff)/np.mean(fuel_diff) #Check formula
                            else:
                                old_fuel_lose = self.fuel_lose
                                self.fuel_lose = round((old_fuel_lose+(np.mean(time_diff)/np.mean(fuel_diff)))/2)

    def compute_drs_lose(self, data:dict):
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

                    ### Remove the time lost due to fuel component
                    drs_time = [x-self.predict_fuel_time_lose(fuel) for x, fuel in zip(drs['LapTime'].values[:max_len], drs['Fuel'].values[:max_len])]
                    no_drs_time = [x-self.predict_fuel_time_lose(fuel) for x, fuel in zip(no_drs['LapTime'].values[:max_len], no_drs['Fuel'].values[:max_len])]
                    
                    self.drs_lose = abs(round(np.mean(np.array(drs_time)-np.array(no_drs_time))))
    
    def compute_fuel_consume_coeff(self, data:dict):
        fuel_consume = {'Dry':[], 'Wet':[]}
        for tyre, val in data.items():
            if tyre in ['Soft', 'Medium', 'Hard']:
                for t_data in val:
                    fuel_consume['Dry'].append(t_data['Fuel'].values)
            elif tyre in ['Inter', 'Wet']:
                for t_data in val:
                    fuel_consume['Wet'].append(t_data['Fuel'].values)

        for key, vals in fuel_consume.items():
            for consume in vals:
                coefficients = np.polyfit(np.arange(1,len(consume)+1), consume, 1)
                
                if self.fuel_consume_coeff[key] == 0:
                    self.fuel_consume_coeff[key] = coefficients[0]
                else:
                    old_coeff = self.fuel_consume_coeff[key]
                    self.fuel_consume_coeff[key] = (old_coeff+coefficients[0])/2

    def compute_missing_wear_coeff(self,):
        tyres = ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']
        all_filled = [False for _ in tyres]
        while not all(all_filled):
            for idx, tyre in enumerate(tyres):
                if any([self.tyre_wear_coeff[tyre][x] == 0 for x in ['FL', 'FR', 'RL', 'RR']]):
                    ### Check if data are available, otherwise we will compute them afterwards the next while cycle
                    if idx > 0:
                        if all([self.tyre_wear_coeff[tyres[idx-1]][x] != 0 for x in ['FL', 'FR', 'RL', 'RR']]):
                            for x in ['FL', 'FR', 'RL', 'RR']:
                                self.tyre_wear_coeff[tyre][x] = self.tyre_wear_coeff[tyres[idx-1]][x]*35/40#np.exp(-idx-1)+1
                            all_filled[idx] = True
                        #if idx+1 < len(tyres) and tyre != 'Inter':
                        #    if all([self.tyre_wear_coeff[tyres[idx+1]][x] != 0 for x in ['FL', 'FR', 'RL', 'RR']]):
                        #        for x in ['FL', 'FR', 'RL', 'RR']:
                        #            self.tyre_wear_coeff[tyre][x] = self.tyre_wear_coeff[tyres[idx+1]][x]*np.log(idx+1.25)
                        #        all_filled[idx] = True
                    else:
                        if all([self.tyre_wear_coeff[tyres[idx+1]][x] != 0 for x in ['FL', 'FR', 'RL', 'RR']]):
                            for x in ['FL', 'FR', 'RL', 'RR']:
                                self.tyre_wear_coeff[tyre][x] = self.tyre_wear_coeff[tyres[idx+1]][x]*40/35#np.log(idx+1.25)
                            all_filled[idx] = True
                else:
                    all_filled[idx] = True
        
        pass

    def compute_tyre_wear_and_time_lose(self, data:dict):
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

                weighted_times = {'FL':[], 'FR':[], 'RL':[], 'RR':[]}
                drs = any(t_data['DRS'] == True)
                if not drs:
                    times = [x-self.predict_fuel_time_lose(fuel) for x, fuel in zip(t_data['LapTime'].values, t_data['Fuel'].values)]
                else:
                    times = [x-self.predict_fuel_time_lose(fuel)-self.drs_lose for x, fuel in zip(t_data['LapTime'].values, t_data['Fuel'].values)]
                best_time = min(times)
                
                for i,time in enumerate(times):
                    wear_sum = 0
                    for key in ['FL', 'FR', 'RL', 'RR']:
                        wear_sum += tyre_wear[tyre][key][i]

                    for key in ['FL', 'FR', 'RL', 'RR']:
                        weighted_times[key].append(round((time-best_time)*tyre_wear[tyre][key][i]/wear_sum))
                   

                for key in ['FL', 'FR', 'RL', 'RR']:
                    old_coeff = 0
                    if self.tyre_wear_coeff[tyre][key] != 0:
                        old_coeff = self.tyre_wear_coeff[tyre][key]
                    new_coeff = np.polyfit(np.arange(1,len(tyre_wear[tyre][key])+1), tyre_wear[tyre][key], 1)
                    self.tyre_wear_coeff[tyre][key] = new_coeff[0] if old_coeff == 0 else (old_coeff+new_coeff[0])/2

                    old_coeff = 0
                    if self.tyre_coeff[tyre][key] != 0:
                        old_coeff = self.tyre_coeff[tyre][key]
                    
                    mu = 0
                    sigma = 0
                    _w = [0]
                    for idx, time in enumerate(weighted_times[key][1:-1]):
                        mu = round((weighted_times[key][idx] + weighted_times[key][idx+2])/2)
                        sigma = abs(mu-time)
                        weight = abs(1 - (sigma/100))/100 * pow(idx+1, 1.1)
                        _w.append(weight)

                    last_w = abs((weighted_times[key][-1]-weighted_times[key][-2])/100)/100*pow(len(weighted_times[key]), 1.5)
                    _w.append(last_w if last_w > 0 else 0)
                    new_coeff = abs(np.polyfit(tyre_wear[tyre][key], weighted_times[key], 1, w=_w))
                    
                    self.tyre_coeff[tyre][key] = new_coeff[0] if old_coeff == 0 else (old_coeff+new_coeff[0])/2
                    
        self.compute_missing_wear_coeff()
    
    def compute_time_compound(self, data:pd.DataFrame):
        soft = data['Soft']
        best = {'Soft':np.inf, 'Medium':np.inf, 'Hard':np.inf, 'Inter':np.inf, 'Wet':np.inf}
        for df in soft:
            times = [x-self.predict_fuel_time_lose(fuel)-self.predict_tyre_time_lose('Soft',lap)['Total'] for x, fuel, lap in zip(df['LapTime'].values, df['Fuel'].values, df['Lap'].values)]
            if all(df['DRS'] == False):
                times = [x-self.drs_lose for x in times]
            best['Soft'] = min(times) if math.isinf(best['Soft']) else (best['Soft']+min(times))/2
        
        for tyre, val in data.items():
            if tyre != 'Soft':
                for t_data in val:
                    times = [x-self.predict_fuel_time_lose(fuel)-self.predict_tyre_time_lose(tyre,lap)['Total'] for x, fuel, lap in zip(t_data['LapTime'].values, t_data['Fuel'].values, t_data['Lap'].values)]
                    if all(t_data['DRS'] == False) and tyre not in ['Wet', 'Inter']:
                        times = [x-self.drs_lose for x in times]
                    best[tyre] = min(times) if math.isinf(best[tyre]) else (best[tyre]+min(times))/2
        
        for tyre, bestLap in best.items():
            if tyre == "Soft":
                self.time_diff[tyre] = bestLap
            else:
                if not math.isinf(bestLap):
                    self.time_diff[tyre] = (self.time_diff[tyre] + (bestLap-best['Soft']))/2
        
    def plot_tyres_time_lose(self,):
        lim_max = 51
        df = pd.DataFrame(columns=['Lap', 'Compound', 'TimeLost'])
        for tyre in ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']:
            for lap in range(1,lim_max):
                df.loc[len(df)] = [lap, tyre, self.predict_tyre_time_lose(tyre, lap)['Total']]
        
        fig = px.line(df, x="Lap", y="TimeLost", color="Compound", title="Tyre Time Loss")
        fig.show()

    def predict_starting_fuel(self, conditions:list):
        time = 0
        for condition in conditions:
            if condition == "Dry/Wet":
                time += abs((self.fuel_consume_coeff["Dry"] + self.fuel_consume_coeff["Wet"])/2)
            elif condition == "VWet":
                time += self.fuel_consume_coeff["Wet"]
            else:
                time += abs(self.fuel_consume_coeff[condition])
        
        return time
    
    def predict_fuel_weight(self, init_fuel:float, conditions:list):
        weight = init_fuel
        for condition in conditions:
            if condition == "Dry/Wet":
                weight-= abs((self.fuel_consume_coeff["Dry"] + self.fuel_consume_coeff["Wet"])/2)
            elif condition == "VWet":
                weight-= self.fuel_consume_coeff["Wet"]
            else:
                weight-= abs(self.fuel_consume_coeff[condition])
        
        return weight
        
    def predict_fuel_time_lose(self, fuel):
        return round(self.fuel_lose * fuel)

    def predict_tyre_wear(self, tyre:str, lap:int):
        fl = round(self.tyre_wear_coeff[tyre]['FL'] * lap)
        fr = round(self.tyre_wear_coeff[tyre]['FR'] * lap)
        rl = round(self.tyre_wear_coeff[tyre]['RL'] * lap)
        rr = round(self.tyre_wear_coeff[tyre]['RR'] * lap)

        return {'FL':fl, 'FR':fr, 'RL':rl, 'RR':rr}

    def predict_tyre_time_lose(self, tyre:str, lap:int=0, wear:dict=None):
        if wear is None:
            wear = self.predict_tyre_wear(tyre, lap)

        fl = round(self.tyre_coeff[tyre]['FL'] * wear['FL'] * (wear['FL']*wear['FL']*1/100*1/100+1))
        fr = round(self.tyre_coeff[tyre]['FR'] * wear['FR'] * (wear['FR']*wear['FR']*1/100*1/100+1))
        rl = round(self.tyre_coeff[tyre]['RL'] * wear['RL'] * (wear['RL']*wear['RL']*1/100*1/100+1))
        rr = round(self.tyre_coeff[tyre]['RR'] * wear['RR'] * (wear['RR']*wear['RR']*1/100*1/100+1))

        return {'FL':fl, 'FR':fr, 'RL':rl, 'RR':rr, 'Total':fl+fr+rl+rr}

    def predict_laptime(self, tyre:str, tyre_age:int, lap:int, start_fuel:float, conditions:list, drs:bool=False):
        compound_time_lose = self.time_diff[tyre] if tyre != "Soft" else 0
        fuel_time_lose = self.predict_fuel_time_lose(self.predict_fuel_weight(start_fuel, conditions))
        tyre_wear_time_lose = self.predict_tyre_time_lose(tyre, tyre_age)['Total']
        drs_lose = self.drs_lose if drs else 0

        return round(self.time_diff['Soft'] + compound_time_lose + fuel_time_lose + tyre_wear_time_lose + drs_lose)

    def save(self, path:str):
        with open(os.path.join(path,"Car.json"), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str):
        with open(os.path.join(path,"Car.json"), 'rb') as f:
            data = pickle.load(f)
        return data

def get_nearest_frame(df, frameList):
    framesReturn = []
    toRemove = []
    for frame in frameList:
        if frame in df['FrameIdentifier'].values:
            framesReturn.append(frame)
        else:
            notFound = True
            add = 1
            if frame > 40508:
                pass
            while notFound and ((frame + add) < max(df['FrameIdentifier'].values)):
                if (frame + add) in framesReturn:
                    pass
                elif (frame + add) in df['FrameIdentifier'].values:
                    framesReturn.append(frame + add)
                    notFound = False
                add += 1
            
            if notFound:
                toRemove.append(frame)

    return framesReturn, toRemove

def get_data(folder:str, add_data:pd.DataFrame=None, ignore_frames:list=[], race=False):
    if not os.path.isdir(folder):
        return add_data

    lap = pd.read_csv(os.path.join(folder, "Lap.csv"))
    car_index = lap['PlayerCarIndex'].unique()[0]

    lap = lap.loc[(lap["CarIndex"] == car_index) & (lap['DriverStatus'] > 0) & (lap['LastLapTimeInMS'] > 0), ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")#,'DriverStatus'
    if False:
        inserting = False
        temp = None
        runs = list()

        for frame in lap['FrameIdentifier'].values:
            val = lap.loc[lap['FrameIdentifier'] == frame, 'DriverStatus'].values[0]
            timing = lap.loc[lap['FrameIdentifier'] == frame, 'LastLapTimeInMS'].values[0]
            if val == 0:
                if inserting:
                    begin = temp
                    if frame-begin > 1000:
                        runs.append([begin, frame])
                    inserting = False
            else:
                if temp is None and timing != 0:
                    temp = frame
                    inserting = True
                elif not inserting and timing != 0:
                    temp = frame
                    inserting = True

    lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    ### DEBUG ###

    #status = pd.read_csv(os.path.join(folder, "Status.csv"))
    #status = status.loc[status["CarIndex"] == car_index, ['FrameIdentifier','FuelInTank','VisualTyreCompound']].drop_duplicates(['FrameIdentifier'], keep="last")
    #(px.line(status, x="FrameIdentifier", y="FuelInTank", title="Fuel", color="VisualTyreCompound", color_discrete_sequence=px.colors.sequential.RdBu)).show()
    
    pass
    ### DEBUG ###
    
    
    to_drop = ignore_frames
    if to_drop == []:
        if os.path.isfile(os.path.join(folder, "to_drop.txt")):
            with open(os.path.join(folder, "to_drop.txt"), "r") as f:
                to_drop = [int(x) for x in f.read().split(",")[:-1]]
        
        else:
            laps = lap['CurrentLapNum']
            duplicated_laps = lap[laps.isin(laps[laps.duplicated()])].sort_values("FrameIdentifier")
            print(f"Lap DataFrame is the following:")
            for idx, row in lap.iterrows():
                print(f"{idx}, {row['CurrentLapNum']} -> {ms_to_time(row['LastLapTimeInMS'])}")
            print(f"\nDuplicated ones are:\n{duplicated_laps}")
            to_drop = input("If there are some wrong frames, insert now separated by comma or press ENTER if None: ")
            if to_drop:
                to_drop = np.array(to_drop.split(','), dtype=int)

            with open(os.path.join(folder, "to_drop.txt"), "w") as f:
                for frame in to_drop:
                    f.write(f"{frame},")

    for index in to_drop:
        if index in lap.index:
            lap = lap.drop(index)

    sub = min(lap['CurrentLapNum'])-1
    for i in lap.index:
        lap.at[i,'CurrentLapNum'] = int(lap.at[i,'CurrentLapNum'])-sub

    lap_frames = lap.index.values
    telemetry = pd.read_csv(os.path.join(folder, "Telemetry.csv"))
    telemetry = telemetry.loc[telemetry["CarIndex"] == car_index, ['FrameIdentifier', 'DRS']].drop_duplicates(['FrameIdentifier'], keep="last")
    telemetry_frames, remove_frames = get_nearest_frame(telemetry, lap_frames)
    for frame in remove_frames:
        lap = lap.drop(frame)

    telemetry = telemetry.loc[telemetry['FrameIdentifier'].isin(telemetry_frames)].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    lap.index = telemetry_frames
    concatData = pd.concat([lap, telemetry], axis=1)

    status = pd.read_csv(os.path.join(folder, "Status.csv"))
    status = status.loc[status["CarIndex"] == car_index, ['FrameIdentifier','FuelInTank','VisualTyreCompound']].drop_duplicates(['FrameIdentifier'], keep="last")
    status_frames, remove_frames = get_nearest_frame(status, concatData.index.values)
    for frame in remove_frames:
        concatData = concatData.drop(frame)
    
    status = status.loc[status['FrameIdentifier'].isin(status_frames), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")
    
    for i in status.index:
        status.at[i,'VisualTyreCompound'] = VISUAL_COMPOUNDS[status.at[i,'VisualTyreCompound']]

    concatData.index = status_frames
    concatData = pd.concat([concatData, status], axis=1)

    damage = pd.read_csv(os.path.join(folder, "Damage.csv"))
    damage = damage.loc[damage["CarIndex"] == car_index, ['FrameIdentifier', 'TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR',]].drop_duplicates(['FrameIdentifier'], keep="last")
    damage_frames, remove_frames = get_nearest_frame(damage, concatData.index.values)
    for frame in remove_frames:
        concatData = concatData.drop(frame)

    damage = damage.loc[damage['FrameIdentifier'].isin(damage_frames), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData.index = damage_frames
    concatData = pd.concat([concatData, damage], axis=1)

    concatData.rename(columns={"CurrentLapNum": "Lap", "LastLapTimeInMS": "LapTime", 'FuelInTank':'Fuel', 'VisualTyreCompound':'Compound','TyresWearFL':'FLWear', 'TyresWearFR':'FRWear', 'TyresWearRL':'RLWear', 'TyresWearRR':'RRWear'}, inplace=True)

    tyres = concatData['Compound'].unique()
    
    for idx in concatData.index:
        concatData.at[idx, 'StringLapTime'] = ms_to_time(concatData.at[idx, 'LapTime'])

    if race:
        return concatData

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
                data.at[i,'Lap'] = round(data.at[i,'Lap'])-sub

        ret[tyre].append(data)

    #print(ret)
    return ret


def get_car_data(path:str):
    car = None
    if os.path.isfile(os.path.join(path, 'Car.json')):
        car = Car(load_path=path)
    else:    
        if os.path.isfile(os.path.join(path, 'Data.json')):
            with open(os.path.join(path, 'Data.json'), 'rb') as f:
                data = pickle.load(f)

        else:   
            fp1_folder = os.path.join(path, 'FP1')
            fp2_folder = os.path.join(path, 'FP2')
            fp3_folder = os.path.join(path, 'FP3')
            
            fp1 = get_data(fp1_folder, add_data=None)
            fp2 = get_data(fp2_folder, add_data=fp1)
            fp3 = get_data(fp3_folder, add_data=fp2)

            concatenated = None
            for _, item in fp3.items():
                for i in item:
                    if concatenated is None:
                        concatenated = i
                    else:
                        concatenated = pd.concat([concatenated, i])
            
            print(concatenated)
            concatenated.to_csv(os.path.join(path, 'FullData.csv'), index=False)

            with open(os.path.join(path, 'Data.json'), 'wb') as f:
                pickle.dump(fp3, f, pickle.HIGHEST_PROTOCOL)

            data = fp3.copy()
            
        car = Car(data=data)
        car.save(path)
        
    return car
