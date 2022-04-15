import pandas as pd
import numpy as np
import math
import pickle
import os
from classes.RangeDictionary import RangeDictionary
import plotly.express as px


class Fuel:
    def __init__(self, df:pd.DataFrame=None, load_path:str=None) -> None:
        if df is not None:
            indexes = list()
            for lap in df['NumLaps'].unique():
                if not math.isnan(lap):
                    indexes.append(min(df.loc[df['NumLaps'] == lap].notna().index.values))

            self.lap_frames = dict()
            max_lap_len = len(df.filter(like="lapTimeInMS").columns.to_list())
            for idx in range(max_lap_len):
                if idx < max_lap_len-1:
                    self.lap_frames[idx] = [i for i in range(indexes[idx],indexes[idx+1])]
                else:
                    self.lap_frames[idx] = [i for i in range(indexes[idx],df['FrameIdentifier'].iloc[-1])]
            
            self.lap_times = list()
            for col in df.filter(like="lapTimeInMS").columns.to_list():
                self.lap_times.append(max([int(value) for value in df[col].dropna().values]))

            self.lap_times = np.array(self.lap_times)

            self.Sector1InMS = list()
            for col in df.filter(like="sector1TimeInMS").columns.to_list():
                self.Sector1InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector1InMS = np.array(self.Sector1InMS)

            self.Sector2InMS = list()
            for col in df.filter(like="sector2TimeInMS").columns.to_list():
                self.Sector2InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector2InMS = np.array(self.Sector2InMS)

            self.Sector3InMS = list()
            for col in df.filter(like="sector3TimeInMS").columns.to_list():
                self.Sector3InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector3InMS = np.array(self.Sector3InMS)   

            self.FuelInTank = RangeDictionary(df['FuelInTank'].values)
            self.FuelCapacity = RangeDictionary(df['FuelCapacity'].values)
            self.FuelRemainingLaps = RangeDictionary(df['FuelRemainingLaps'].values)

        if df is None and load_path is not None:
            self = self.load(load_path)


    def __getitem__(self, idx) -> dict:
        lap = self.get_lap(idx)
        return {'NumLap': lap, 'LapTimeInMS': self.lap_times[lap], 'Sector1InMS': self.Sector1InMS[lap], 'Sector2InMS': self.Sector2InMS[lap], 'Sector3InMS': self.Sector3InMS[lap], 'FuelInTank': self.FuelInTank[idx], 'FuelCapacity': self.FuelCapacity[idx], 'FuelRemaining': self.FuelRemaining[idx]}

    def get_lap(self, frame, get_float:bool=False) -> int:
        first_value = list(self.lap_frames.values())[0][0]

        if get_float:
            for key, value in self.lap_frames.items():
                if frame+first_value in value:
                    try:
                        idx = value.index(frame+first_value)
                    except Exception as e:
                        input(e)
                        idx = 0
                    if idx == 0:
                        return key
                    return float(key)+(idx/len(value))

        else:
            for key, values in self.lap_frames.items():
                if idx in values:
                    return key
        
        return 0

    def fuel_consumption(self, display:bool=False) -> dict:
        
        fuel_consume = {'Frame':[int(value) for value in self.FuelRemainingLaps.keys()],'Fuel':[value for value in self.FuelRemainingLaps.values()]}

        if display:
            df = pd.DataFrame(fuel_consume)
            for row in df.index:
                df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)

            fig = px.line(df, x='Lap',y='Fuel', title='Fuel Consumption', range_y=[0,100]) #Need to check what is the maximum value of the fuel load
            
            #plotly.offline.plot(fig, filename='Fuel consumption.html')
            fig.show()


        return fuel_consume

    def fuel_timing(self, display:bool=False) -> dict:
        timing = {'Lap':[],'LapTimeInMS':[]}
        for lap in self.lap_frames.keys():
            value = self.lap_times[lap]
            if value != 0 and not math.isnan(value):
                timing['Lap'].append(lap+1)
                timing['LapTimeInMS'].append(value)
        
        if display:
            df = pd.DataFrame(timing)
            fig = px.line(df, x='Lap',y='LapTimeInMS', title='Lap Times',range_y=[min(timing['LapTimeInMS'])-1000,max(timing['LapTimeInMS'])+1000])
            #plotly.offline.plot(fig, filename='Fuel Timing.html')
            fig.show()
            
        return timing
    
    def save(self, path:str=''):
        numlaps = len(self.lap_frames.keys())
        path = os.path.join(path,'Fuel_'+str(numlaps)+'laps.json')
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str=''):
        with open(path, 'rb') as f:
            return pickle.load(f)

def get_fuel_data(df:pd.DataFrame, separators:dict) -> set:
    """
    Get the fuel data from the dataframe
    """
    fuel_data = set()
    columns = ['FrameIdentifier', 'NumLaps', 'FuelInTank', 'FuelCapacity','FuelRemainingLaps']

    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the laps we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x)]

        if len(numLaps) > 3:
            start = numLaps[0]
            end = numLaps[-1]

            fuel_columns = columns + ['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

            ### Get the fuel data of the laps we are considering
            data = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),fuel_columns]

            ### Add them to the set
            fuel_data.add((key,Fuel(df=data)))
            

    return fuel_data