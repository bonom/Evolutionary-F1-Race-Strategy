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

            self.FuelInTank = RangeDictionary(df['FuelInTank'].values)
            self.FuelCapacity = RangeDictionary(df['FuelCapacity'].values)
            self.FuelRemainingLaps = RangeDictionary(df['FuelRemainingLaps'].values)

            ### MODEL ###
            x = np.array([int(key) for key in self.FuelInTank.keys()]).reshape((-1,1))
            y = np.array(list(self.FuelInTank.values()))
            if math.isnan(y[0]):
                y[0] = 0

            self.model = LinearRegression().fit(x,y)

            #r_sq = self.model.score(x,y)
            #intercept = self.model.intercept_
            #slope = self.model.coef_
            
        elif load_path is not None:
            data = self.load(load_path)
            self.lap_frames = data.lap_frames
            self.FuelInTank = data.FuelInTank
            self.FuelCapacity = data.FuelCapacity
            self.FuelRemainingLaps = data.FuelRemainingLaps
            self.model = data.model


    def __getitem__(self, idx) -> dict:
        lap = self.get_lap(idx)
        return {'NumLap': lap, 'FuelInTank': self.FuelInTank[idx], 'FuelCapacity': self.FuelCapacity[idx], 'FuelRemaining': self.FuelRemainingLaps[idx]}

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
        
        max_lap = int(max(fuel_consume['Lap']))
        fuel_consume = fuel_consume[fuel_consume['Lap'] <= max_lap]
        fuel_consume.drop_duplicates(subset=['Lap'], keep='first', inplace=True)

        if display:
            fig = px.line(fuel_consume, x='Lap',y='Fuel', title='Fuel Consumption', range_y=[0,100], range_x=[-0.1,max(fuel_consume['Lap'])+1]) #Need to check what is the maximum value of the fuel load
            
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Plots/Fuel consumption.html')
            else:
                fig.show()


        return fuel_consume
    
    def predict_fuelload(self, x_predict:int) -> float:
        """
        Return the 2 coefficient beta_0 and beta_1 for the linear model that fits the data : Time/Fuel
        """
        x_predict = np.array(x_predict).reshape(-1,1)
        y_predict = self.model.predict(x_predict)
        
        y_predict = round(y_predict[0],2)
        log.info(f"Predicted fuel consumption for lap {self.get_lap(int(x_predict))} (frame {int(x_predict)}) is {y_predict} %")
        
        return y_predict

    def save(self, save_path:str='', id:int=0) -> None:
        save_path = os.path.join(save_path,'Fuel_'+str(id)+'.json')
        with open(save_path, 'wb') as f:
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

    ### Initialize the columns of interest
    fuel_columns = ['FrameIdentifier', 'NumLaps', 'FuelInTank', 'FuelCapacity','FuelRemainingLaps']

    ### Cycle over all the times we box
    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the laps we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x)]

        if len(numLaps) > 3:
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