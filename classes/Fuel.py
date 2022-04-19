from typing import Union
import pandas as pd
import numpy as np
import math
import pickle
import sys, os
from classes.RangeDictionary import RangeDictionary
import plotly.express as px
import plotly
from sklearn.linear_model import LinearRegression

from classes.Utils import get_basic_logger

log = get_basic_logger('FUEL')

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
                    #self.lap_frames[idx] = [i for i in range(indexes[idx],indexes[idx+1])]
                    start = indexes[idx]
                    end = indexes[idx+1]
                else:
                    #self.lap_frames[idx] = [i for i in range(indexes[idx],df['FrameIdentifier'].iloc[-1])]
                    start = indexes[idx]
                    end = df['FrameIdentifier'].iloc[-1]
                
                for i in range(start,end):
                    self.lap_frames[i] = idx + round(((i - start)/(end - start)),2)
            
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
        elif load_path is not None:
            data = self.load(load_path)
            self.lap_frames = data.lap_frames
            self.lap_times = data.lap_times
            self.Sector1InMS = data.Sector1InMS
            self.Sector2InMS = data.Sector2InMS
            self.Sector3InMS = data.Sector3InMS
            self.FuelInTank = data.FuelInTank
            self.FuelCapacity = data.FuelCapacity
            self.FuelRemainingLaps = data.FuelRemainingLaps


    def __getitem__(self, idx) -> dict:
        lap = self.get_lap(idx)
        return {'NumLap': lap, 'LapTimeInMS': self.lap_times[lap], 'Sector1InMS': self.Sector1InMS[lap], 'Sector2InMS': self.Sector2InMS[lap], 'Sector3InMS': self.Sector3InMS[lap], 'FuelInTank': self.FuelInTank[idx], 'FuelCapacity': self.FuelCapacity[idx], 'FuelRemaining': self.FuelRemaining[idx]}

    def get_lap(self, frame, get_float:bool=False) -> Union[int,float]:
        first_value = list(self.lap_frames.keys())[0]
        if get_float:
            return self.lap_frames[frame+first_value]
        
        return int(self.lap_frames[frame+first_value])

    def consumption(self, display:bool=False) -> dict:
        
        fuel_consume = {'Frame':[int(value) for value in self.FuelInTank.keys()],'Fuel':[value for value in self.FuelInTank.values()]}

        fuel_consume = pd.DataFrame(fuel_consume)

        for row in fuel_consume.index:
            fuel_consume.at[row,'Lap'] = self.get_lap(fuel_consume.at[row,'Frame'],True)

        if display:
            fig = px.line(fuel_consume, x='Lap',y='Fuel', title='Fuel Consumption', range_y=[0,100]) #Need to check what is the maximum value of the fuel load
            #plotly.offline.plot(fig, filename='Fuel consumption.html')
            fig.show()


        return fuel_consume

    def timing(self, display:bool=False) -> dict:
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
    
    def predict_fuelload(self, x_predict:int) -> float:
        """
        Return the 2 coefficient beta_0 and beta_1 for the linear model that fits the data : Time/Fuel
        """
        x = np.array([int(key) for key in self.FuelInTank.keys()]).reshape((-1,1))
        y = np.array(list(self.FuelInTank.values()))
        if math.isnan(y[0]):
            y[0] = 0

        #print(x)
        #print(y)

        model = LinearRegression().fit(x,y)

        r_sq = model.score(x,y)
        #print('Coefficient of determination: ', r_sq)

        intercept = model.intercept_
        slope = model.coef_
        #print('Intercept : ', intercept)
        #print('Slope : ', slope)

        x_predict = np.array(x_predict).reshape(-1,1)
        y_predict = model.predict(x_predict)
        #print(y_predict)
        
        y_predict = round(y_predict[0],2)
        log.info(f"Predicted fuel consumption for lap {self.get_lap(int(x_predict))} (frame {int(x_predict)}) is {y_predict} %")
        
        return y_predict

    def save(self, path:str='', id:int=0) -> None:
        path = os.path.join(path,'Fuel_'+str(id)+'.json')
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str=''):
        with open(path, 'rb') as f:
            return pickle.load(f)

    

def get_fuel_data(df:pd.DataFrame, separators:dict, path:str=None) -> set:
    """
    Get the fuel data from the dataframe
    """
    ### Initialize the set 
    fuel_data = set()

    if path is not None:
        log.info('Specified load path, trying to find Fuel_*.json files...')
        files = [f for f in os.listdir(path) if f.endswith('.json') and f.startswith('Fuel_')]
        if len(files) > 0:
            log.info('Specified load path with files inside. Loading fuel data from file...')
            for file in files:
                fuel = Fuel(load_path=os.path.join(path,file))
                idx = int(file.replace('Fuel_','').replace('.json',''))
                fuel_data.add((idx,fuel))
                
            log.info('Loading completed.')
            return fuel_data
                
    
    if path is not None:
        log.info(f'No Fuel_*.json files found in "{path}". Loading fuel data from dataframe.')
    else:
        log.info('No load path specified. Loading fuel data from dataframe.')

    ### Initialize the common columns
    columns = ['FrameIdentifier', 'NumLaps', 'FuelInTank', 'FuelCapacity','FuelRemainingLaps']

    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the laps we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x)]

        if len(numLaps) > 3:
            start = numLaps[0]
            end = numLaps[-1]

            ### Get the columns data of the fuel range frames we are considering (these are particular, not common to all)
            fuel_columns = columns + ['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

            ### Get the fuel data of the laps we are considering
            data = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),fuel_columns]

            ### Add them to the set
            fuel = Fuel(df=data)
            fuel.save(path,id=key)
            fuel_data.add((key,fuel))
        else:
            log.warning(f"Insufficient data (below 3 laps). Skipping {key}/{len(separators.keys())}.")
            

    return fuel_data

if __name__ == "__main__":
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)