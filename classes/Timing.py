from typing import Union
import pandas as pd
import numpy as np
import math
import pickle
import plotly
import plotly.express as px
import sys, os
from classes.Utils import get_basic_logger, get_host

log = get_basic_logger('TIMING')

class Timing:
    def __init__(self, df:pd.DataFrame=None, load_path:str=None) -> None:
        if df is not None:
            indexes = list()
            for lap in df['NumLaps'].unique():
                if not math.isnan(lap):
                    indexes.append(min(df.loc[df['NumLaps'] == lap].notna().index.values))

            self.lap_frames = dict()
            max_lap_len = len(indexes)
            for idx in range(max_lap_len):
                if idx < max_lap_len-1:
                    start = indexes[idx]
                    end = indexes[idx+1]
                else:
                    start = indexes[idx]
                    end = df['FrameIdentifier'].iloc[-1]
                
                for i in range(start,end):
                    self.lap_frames[i] = idx + round(((i - start)/(end - start)),2)
            
            self.LapTimes = list()
            for col in df.filter(like="lapTimeInMS").columns.to_list():
                values = [int(value) for value in df[col].dropna().drop_duplicates().values if int(value) > 0]
                if len(values) == 1:
                    self.LapTimes.append(values[0])
                elif len(values) > 1:
                    log.critical("Wrong number of times:\n\t\t\t\t\t\t{}".format(values))
             

            self.LapTimes = np.array(self.LapTimes)

            self.BestLapTime = min([lt for lt in self.LapTimes if lt > 0])
            self.BestLap = self.LapTimes.argmin() + 1

            self.Deltas = []
            for i in range(len(self.LapTimes)):
                self.Deltas.append(self.LapTimes[i]-self.BestLapTime)
            
            self.Deltas = np.array(self.Deltas)

            self.Sector1InMS = list()
            for col in df.filter(like="sector1TimeInMS").columns.to_list():
                self.Sector1InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector1InMS = np.array(self.Sector1InMS)

            self.BestSector1Time = min(self.Sector1InMS)
            self.BestSector1 = self.Sector1InMS.argmin() + 1

            self.Sector2InMS = list()
            for col in df.filter(like="sector2TimeInMS").columns.to_list():
                self.Sector2InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector2InMS = np.array(self.Sector2InMS)

            self.BestSector2Time = min(self.Sector2InMS)
            self.BestSector2 = self.Sector2InMS.argmin() + 1

            self.Sector3InMS = list()
            for col in df.filter(like="sector3TimeInMS").columns.to_list():
                self.Sector3InMS.append(max([int(value) for value in df[col].dropna().values]))

            self.Sector3InMS = np.array(self.Sector3InMS)   

            self.BestSector3Time = min(self.Sector3InMS)
            self.BestSector3 = self.Sector3InMS.argmin() + 1
            
        elif load_path is not None:
            data = self.load(load_path)
            self.lap_frames = data.lap_frames
            self.LapTimes = data.LapTimes
            self.BestLapTime = data.BestLapTime
            self.BestLap = data.BestLap
            self.Deltas = data.Deltas
            self.Sector1InMS = data.Sector1InMS
            self.BestSector1Time = data.BestSector1Time
            self.BestSector1 = data.BestSector1
            self.Sector2InMS = data.Sector2InMS
            self.BestSector2Time = data.BestSector2Time
            self.BestSector2 = data.BestSector2
            self.Sector3InMS = data.Sector3InMS
            self.BestSector3Time = data.BestSector3Time
            self.BestSector3 = data.BestSector3
    
    def __repr__(self):
        return f"LapTimes: {self.LapTimes}\nSector1InMS: {self.Sector1InMS}\nSector2InMS: {self.Sector2InMS}\nSector3InMS: {self.Sector3InMS}"
    
    def __str__(self):
        return self.__repr__(self)
    
    def __getitem__(self, key:int) -> dict:
        if key == -1:
            key = self.__len__() - 1

        #key -= list(self.lap_frames.keys())[0]
        lap = self.get_lap(key)
        return {'Lap':lap,'LapTimeInMS':self.LapTimes[lap], 'Sector1InMS':self.Sector1InMS[lap], 'Sector2InMS':self.Sector2InMS[lap], 'Sector3InMS':self.Sector3InMS[lap]}

    def __len__(self) -> int:
        return len(self.lap_frames.keys())
    
    def get_lap(self, frame, get_float:bool=False) -> Union[int,float]:
        first_value = list(self.lap_frames.keys())[0]

        if get_float:
            return self.lap_frames[frame+first_value]
        
        return int(self.lap_frames[frame+first_value])
    
    def get_frame(self, lap_num:Union[int,float]) -> int:
        for frame, lap in self.lap_frames.items():
            if lap == lap_num:
                return frame
        
        return -1
        
    def plot(self, display:bool=False) -> dict:
        timing = {'Lap':[],'LapTimeInMS':[]}
        for lap, lap_time in enumerate(self.LapTimes):
            if lap_time != 0:
                timing['Lap'].append(lap+1)
                timing['LapTimeInMS'].append(lap_time)
            elif lap != len(self.LapTimes)-1:
                log.critical("Lap {} has no time and it is not the last one!".format(lap+1))
        
        if display:
            df = pd.DataFrame(timing)
            fig = px.line(df, x='Lap',y='LapTimeInMS', title='Lap Times',markers=True,range_x=[-0.1,max(timing['Lap'])+1], range_y=[min(timing['LapTimeInMS'])-1000,max(timing['LapTimeInMS'])+1000])
            
            if get_host() == 'DESKTOP-KICFR1D':
                plotly.offline.plot(fig, filename='Plots/Timing.html')
            else:
                fig.show()
            
        return timing

    def save(self, save_path:str, id:int=0) -> None:
        save_path = os.path.join(save_path,'Timing_'+str(id)+'.json')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, load_path:str) -> dict:
        with open(load_path, 'rb') as f:
            return pickle.load(f)

def get_timing_data(df:pd.DataFrame, separators:dict, path:str=None) -> Timing:
    ### Initialize the set 
    timing_data = dict()
    
    if path is not None:
        log.info('Specified load path, trying to find Timing_*.json files...')
        files = [f for f in os.listdir(path) if f.endswith('.json') and f.startswith('Timing_')]
        if len(files) > 0:
            log.info('Specified load path with files inside. Loading Timing data from file...')
            for file in files:
                timing = Timing(load_path=os.path.join(path,file))
                idx = int(file.replace('Timing_','').replace('.json',''))
                timing_data[idx] = timing

            log.info('Loading completed.')
            return timing_data
                
    
    if path is not None:
        log.info(f'No timing_*.json files found in "{path}". Loading timing data from dataframe.')
    else:
        log.info('No load path specified. Loading timing data from dataframe.')

    ### Initialize the columns of interest (these are the common columns to all timings)
    basic_columns = ['FrameIdentifier','NumLaps']
    
    ### Cycle over all the times we box
    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the separator we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x)]
        
        if len(numLaps) > 3:
            start = numLaps[0]
            end = numLaps[-1]

            ### Initialize the columns of interest (these are the different columns to all timings)
            timing_columns = basic_columns + ['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

            ### Get the timing data of the separator we are considering
            data = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),timing_columns]

            ### Initialize the timing data and add it to the set
            timing = Timing(data) 
            timing.save(path,id=key)
            timing_data[key] = timing
        else:
            log.warning(f"Insufficient data (below 3 laps). Skipping {key+1}/{len(separators.keys())}.")

    return timing_data
