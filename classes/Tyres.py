import sys, os
from typing import Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import pickle
import plotly.express as px
import plotly
from classes.RangeDictionary import RangeDictionary
from classes.Utils import ACTUAL_COMPOUNDS, VISUAL_COMPOUNDS, TYRE_POSITION, get_basic_logger

log = get_basic_logger("TYRES")

class Tyre:
    """
    Class for a single tyre.

    Parameters:
    ----------
    position : str 
        The position of the tyre: 'FL' = Front Left, 'FR' = Right Front, 'RL' = Rear Left, 'RR' = Rear Right.
    wear : np.array or list
        It is a list of the wear percentages of the tyre every Frame.
    damage : np.array or list
        It is a list of the damage percentages of the tyre every Frame.
    pressure : np.array or list
        It is a list of the pressure of the tyre every Frame.
    inner_temperature : np.array or list
        It is a list of the inner temperature of the tyre every Frame.
    outer_temperature : np.array or list
        It is a list of the outer temperature of the tyre every Frame.
    
    Functions:
    ----------
    tyre_wear(self, display:bool=False) : 
        Returns the wear function of the tyre. If display is True, it will plot the graph of the tyre wear.
    
    cast_tyre_position(self, position) :
        Returns the position of the tyre in a string format.
    
    get_lap(self, frame, get_float) :
        Returns the lap of the tyre at the given frame. If get_float is true then it will cast the lap to a float (if we have two frames in the same lap this will return different float lap values).

    tyre_slip(self, display:bool=False) :
        Returns the slip of the tyre (notice that a slip = 1 means that the tyre rotate with no velocity while slip = -1 means the tyre does not rotate with a certain velocity). 
            If display is True, it will plot the graph of the tyre slip.
    """

    def __init__(self, position:str, wear:Union[np.array,list]=[0.0], damage:Union[np.array,list]=[0.0], pressure:Union[np.array,list]=[0.0], inner_temperature:Union[np.array,list]=[0.0], outer_temperature:Union[np.array,list]=[0.0], laps_data:dict={0:0}, slip:Union[np.array,list]=[0.0]) -> None:
        self.position = position
        self.lap_frames = laps_data
        self.wear = RangeDictionary(np.array(wear))
        self.pressure = RangeDictionary(np.array(pressure))
        self.inner_temperature = RangeDictionary(np.array(inner_temperature))
        self.outer_temperature = RangeDictionary(np.array(outer_temperature))
        self.damage = RangeDictionary(np.array(damage))
        self.slip = RangeDictionary(np.array(slip))


        ### MODEL ###
        x = np.array([int(key) for key in self.wear.keys()]).reshape((-1,1))
        y = np.array(list(self.wear.values()))
        if math.isnan(y[0]):
            y[0] = 0
        
        self.model = LinearRegression().fit(x,y)

        #r_sq = self.model.score(x,y)
        #intercept = self.model.intercept_
        #slope = self.model.coef_

    def __str__(self) -> str:
        to_ret = f"Tyre position: {self.position}\nTyre wear: ["
        for wear in self.wear:
            to_ret += f"{wear}%, "
        to_ret += f"]\nTyre damage: ["
        for damage in self.damage:
            to_ret += f"{damage}%, "
        to_ret += "Pressure: {self.pressure} psi\nInner Temperature: ["
        for temp in self.inner_temperature:
            to_ret += f"{temp}°C, "
        to_ret+= f"]\nOuter Temperature: ["
        for temp in self.outer_temperature:
            to_ret += f"{temp}°C, "
        to_ret += f"]"
        
        return to_ret
    
    def __getitem__(self,idx):
        """
        Returns a dict of data at the given index (must be a frame index).
        """
        if idx == -1:
            idx = self.__len__() 
        return {'TyrePosition':self.position, 'TyreWear':self.wear[idx], 'TyreDamage':self.damage[idx], 'TyrePressure':self.pressure[idx], 'TyreInnerTemperature':self.inner_temperature[idx], 'TyreOuterTemperature':self.outer_temperature[idx]}
    
    def get_lap(self, frame, get_float:bool=False) -> Union[int,float]:
        first_value = list(self.lap_frames.keys())[0]
        
        if get_float:
            return self.lap_frames[frame+first_value]
        
        return int(self.lap_frames[frame+first_value])

    def cast_tyre_position(self, position) -> str:
        return TYRE_POSITION[position]

    def tyre_wear(self, display:bool=False) -> dict:
        """
        Function that returns the wear of the tyre. If display is True, it will plot the graph of the tyre wear.
        """
        dict_items = {'Frame':[int(key) for key in self.wear.keys()],'Wear_'+str(self.position):[value for value in self.wear.values()]}
        
        if display:
            df = pd.DataFrame(dict_items)
            for row in df.index:
                df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)

            fig = px.line(df, x='Lap',y='Wear_'+str(self.position), title=self.cast_tyre_position(self.position)+' Tyre Wear', range_y=[0,100])
            
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Tyre'+str(self.position)+' Wear.html')
            else:
                fig.show()

        #df.set_index('Frame', inplace=True)
        return dict_items
    
    def tyre_slip(self, display:bool=False) -> dict:
        """
        Function that returns the slip of the tyre (notice that a slip = 1 means that the tyre rotate with no velocity while slip = -1 means the tyre does not rotate with a certain velocity).
        """
        dict_items = {'Frame':[int(key) for key in self.slip.keys()],'Slip_'+str(self.position):[value for value in self.slip.values()]}
        
        if display:
            df = pd.DataFrame(dict_items)
            for row in df.index:
                df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)

            fig = px.line(df, x='Lap',y='Slip_'+str(self.position), title=self.cast_tyre_position(self.position)+' Tyre Slip')
            fig.update(layout_yaxis_range = [-1.1,1.1])
            
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Tyre'+str(self.position)+' Slip.html')
            else:
                fig.show()

        return dict_items
    
    def predict_wear(self, x_predict:int) -> float:
        """
        Return the 2 coefficient beta_0 and beta_1 for the linear model that fits the data : Time/Fuel
        """
        x_predict = np.array(x_predict).reshape(-1,1)
        y_predict = self.model.predict(x_predict)

        y_predict = round(y_predict[0],2)
        return y_predict


class Tyres:
    """
    Super class of the Tyre class.

    Parameters:
    ----------
    df : pd.Dataframe
        The dataframe containing the data of the tyres. 
    
    Functions:
    ----------
    cast_visual_compound(self,visual_compound:int) : str 
        Cast the integer id of the visual compound to the string one.

    cast_actual_compound(self,actual_compound:int) : str 
        Cast the integer id of the actual compound to the string one.
    
    tyres_wear(self, display:bool=False) : 
        Returns the wear function of the tyres.

    tyres_timing(self, display:bool=False) :
        Returns the lap times of the used tyres.
    
    tyres_slip(self, display:bool=False) :
        Returns the slip of the tyres.
    
    get_lap(self,frame:int, get_float:bool=False) :
        Returns the lap based on the frame. If get_float is True it returns a float lap

    """

    def __init__(self, df:pd.DataFrame=None, load_path:str=None) -> None:
        if df is not None:
            visual_tyre_compound = np.array([int(x) for x in df['VisualTyreCompound'].unique() if not math.isnan(float(x)) and float(x) > 0])
            actual_tyre_compound = np.array([int(x) for x in df['ActualTyreCompound'].unique() if not math.isnan(float(x)) and float(x) > 0])

            if len(visual_tyre_compound) > 1 or len(actual_tyre_compound) > 1:
                log.critical("The dataframe contains more than one tyre compound:\nVisualTyreCompound contains {}\nActualTyreCompound contains {}".format(visual_tyre_compound, actual_tyre_compound))

            self.visual_tyre_compound = visual_tyre_compound.item() if len(visual_tyre_compound) == 1 else 0
            self.actual_tyre_compound = actual_tyre_compound.item() if len(actual_tyre_compound) == 1 else 0

            indexes = list()
            for lap in df['NumLaps'].unique():
                if not math.isnan(lap):
                    indexes.append(min(df.loc[df['NumLaps'] == lap].notna().index.values))

            self.lap_frames = dict()
            max_lap_len = len(indexes)
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
            """
            for lap in df['NumLaps'].unique():
                if not math.isnan(lap):
                    values = df.loc[df['NumLaps'] == lap].notna().index.values
                    indexes.append((min(values),max(values)))

            max_lap_len = len(indexes)
            
            self.lap_frames = dict()

            for idx in range(max_lap_len):
                if idx < max_lap_len-1:
                    start,_ = indexes[idx]
                    end,_ = indexes[idx+1]
                else:
                    #self.lap_frames[idx] = [i for i in range(indexes[idx],df['FrameIdentifier'].iloc[-1])]
                    start,end = indexes[idx]
                
                for i in range(start,end):
                    self.lap_frames[i] = idx + round(((i - start)/(end - start)),2)
            """
            

            self.FL_tyre = Tyre("FL", df["TyresWearFL"].values,df['TyresDamageFL'].values,df['FLTyrePressure'].values,df['FLTyreInnerTemperature'].values,df['FLTyreSurfaceTemperature'].values,self.lap_frames,df['FLWheelSlip'].values)
            self.FR_tyre = Tyre("FR", df["TyresWearFR"].values,df['TyresDamageFR'].values,df['FRTyrePressure'].values,df['FRTyreInnerTemperature'].values,df['FRTyreSurfaceTemperature'].values,self.lap_frames,df['FRWheelSlip'].values)
            self.RL_tyre = Tyre("RL", df["TyresWearRL"].values,df['TyresDamageRL'].values,df['RLTyrePressure'].values,df['RLTyreInnerTemperature'].values,df['RLTyreSurfaceTemperature'].values,self.lap_frames,df['RLWheelSlip'].values)
            self.RR_tyre = Tyre("RR", df["TyresWearRR"].values,df['TyresDamageRR'].values,df['RRTyrePressure'].values,df['RRTyreInnerTemperature'].values,df['RRTyreSurfaceTemperature'].values,self.lap_frames,df['RRWheelSlip'].values)

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

        elif load_path is not None:
            data = self.load(load_path)
            self.FL_tyre = data.FL_tyre
            self.FR_tyre = data.FR_tyre
            self.RL_tyre = data.RL_tyre
            self.RR_tyre = data.RR_tyre
            self.visual_tyre_compound = data.visual_tyre_compound
            self.actual_tyre_compound = data.actual_tyre_compound
            self.lap_frames = data.lap_frames
            self.lap_times = data.lap_times
            self.Sector1InMS = data.Sector1InMS
            self.Sector2InMS = data.Sector2InMS
            self.Sector3InMS = data.Sector3InMS


    def __str__(self) -> str:
        return str(self.FL_tyre) +"\n" + str(self.FR_tyre) +"\n" + str(self.RL_tyre) +"\n" + str(self.RR_tyre)+"\n"+str(self.lap_times)+"\n"+str(self.Sector1InMS)+"\n"+str(self.Sector2InMS)+"\n"+str(self.Sector3InMS)

    def __getitem__(self,idx) -> dict:
        if idx == -1:
            idx = self.__len__() 
        return {'Lap':self.get_lap(idx)+1,'FLTyre':self.FL_tyre[idx], 'FRTyre':self.FR_tyre[idx], 'RLTyre':self.RL_tyre[idx], 'RRTyre':self.RR_tyre[idx], 'LapTimeInMS':self.lap_times[self.get_lap(idx)], 'Sector1InMS':self.Sector1InMS[self.get_lap(idx)], 'Sector2InMS':self.Sector2InMS[self.get_lap(idx)], 'Sector3InMS':self.Sector3InMS[self.get_lap(idx)]}
    
    def __len__(self) -> int:
        if len(self.lap_frames) == 0:
            return 0
        return len(self.lap_frames.keys())
        
    def get_actual_compound(self, compound:int=None) -> str:
        return ACTUAL_COMPOUNDS[self.actual_tyre_compound if compound is None else compound]
    
    def get_visual_compound(self, compound:int=None) -> str:
        return VISUAL_COMPOUNDS[self.visual_tyre_compound if compound is None else compound]

    def timing(self, display:bool=False) -> dict:
        timing = {'Lap':[],'LapTimeInMS':[]}
        for lap, lap_time in enumerate(self.lap_times):
            if lap_time != 0:
                timing['Lap'].append(lap+1)
                timing['LapTimeInMS'].append(lap_time)
            elif lap != len(self.lap_times)-1:
                log.critical("Lap {} has no time and it is not the last one!".format(lap+1))
        
        if display:
            df = pd.DataFrame(timing)
            fig = px.line(df, x='Lap',y='LapTimeInMS', title='Lap Times on '+self.get_visual_compound()+" compound",markers=True,range_x=[-0.1,max(timing['Lap'])+1], range_y=[min(timing['LapTimeInMS'])-1000,max(timing['LapTimeInMS'])+1000])
            
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Tyres Timing.html')
            else:
                fig.show()
            
        return timing

    def wear(self, display:bool=False) -> pd.DataFrame:
        FL_Tyre_wear = self.FL_tyre.tyre_wear(display=False)
        FR_Tyre_wear = self.FR_tyre.tyre_wear(display=False)
        RL_Tyre_wear = self.RL_tyre.tyre_wear(display=False)
        RR_Tyre_wear = self.RR_tyre.tyre_wear(display=False)

        df_FL = pd.DataFrame(FL_Tyre_wear).set_index('Frame')
        df_FR = pd.DataFrame(FR_Tyre_wear).set_index('Frame')
        df_RL = pd.DataFrame(RL_Tyre_wear).set_index('Frame')
        df_RR = pd.DataFrame(RR_Tyre_wear).set_index('Frame')

        df = pd.concat([df_FL,df_FR,df_RL,df_RR], axis=1).reset_index()
        df = df[(df['Wear_FL'].notna()) & (df['Wear_FR'].notna()) & (df['Wear_RL'].notna()) & (df['Wear_RR'].notna())]
        
        for row in df.index:
            df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)
        
        max_lap = int(max(df['Lap']))
        df = df[df['Lap'] <= max_lap]
        df.drop_duplicates(subset=['Lap'], keep='first', inplace=True)
            
        if display:
            fig = px.line(df, x='Lap',y=['Wear_FL', 'Wear_FR', 'Wear_RL', 'Wear_RR'], title='Tyre Wear on '+self.get_visual_compound()+" compound",range_y=[0,100],range_x=[-0.1,max(df['Lap'])+1])
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Tyres Wear.html')
            else:
                fig.show()

        return df.to_dict()
    
    def predict_wears(self, x_predict:int) -> dict:
        FL_Tyre_model = self.FL_tyre.predict_wear(x_predict)
        FR_Tyre_model = self.FR_tyre.predict_wear(x_predict)
        RL_Tyre_model = self.RL_tyre.predict_wear(x_predict)
        RR_Tyre_model = self.RR_tyre.predict_wear(x_predict)

        predictions = dict({'FL' : FL_Tyre_model, 'FR' : FR_Tyre_model,'RL' : RL_Tyre_model, 'RR' : RR_Tyre_model})
        log.info(f"Tyres Wear predictions at lap {self.get_lap(x_predict, True)} (frame {x_predict}):\n\t\t\t\t\tFrontLeft Wear: {predictions['FL']} %,\n\t\t\t\t\tFrontRight Wear: {predictions['FR']} %,\n\t\t\t\t\tRearLeft Wear: {predictions['RL']} %,\n\t\t\t\t\tRearRight Wear: {predictions['RR']} %.")

        return predictions

    def slip(self, display:bool=False) -> pd.DataFrame:
        FL_Tyre_slip = self.FL_tyre.tyre_slip(display=False)
        FR_Tyre_slip = self.FR_tyre.tyre_slip(display=False)
        RL_Tyre_slip = self.RL_tyre.tyre_slip(display=False)
        RR_Tyre_slip = self.RR_tyre.tyre_slip(display=False)

        df_FL = pd.DataFrame(FL_Tyre_slip).set_index('Frame')
        df_FR = pd.DataFrame(FR_Tyre_slip).set_index('Frame')
        df_RL = pd.DataFrame(RL_Tyre_slip).set_index('Frame')
        df_RR = pd.DataFrame(RR_Tyre_slip).set_index('Frame')

        df = pd.concat([df_FL,df_FR,df_RL,df_RR], axis=1).reset_index()
        df = df[(df['Slip_FL'].notna()) & (df['Slip_FR'].notna()) & (df['Slip_RL'].notna()) & (df['Slip_RR'].notna())]
        
        for row in df.index:
            df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)
            
        max_lap = int(max(df['Lap']))
        df = df[df['Lap'] <= max_lap]
        df.drop_duplicates(subset=['Lap'], keep='first', inplace=True)

        if display:
            fig = px.line(df, x='Lap',y=['Slip_FL', 'Slip_FR', 'Slip_RL', 'Slip_RR'], title='Tyre Slip on '+self.get_visual_compound()+" compound",markers=True,range_y=[-1.1,1.1], range_x=[-0.1,max(df['Lap'])+1])
            
            if os.environ['COMPUTERNAME'] == 'PC-EVELYN':
                plotly.offline.plot(fig, filename='Tyres.html')
            else:
                fig.show()
    
        return df#.to_dict()


    def get_age(self,frame:int=0) -> Union[int,float]:
        return self.get_lap(frame)      

    def get_lap(self, frame, get_float:bool=False) -> Union[int,float]:
        first_value = list(self.lap_frames.keys())[0]

        if get_float:
            return self.lap_frames[frame+first_value]
        
        return int(self.lap_frames[frame+first_value])

    def save(self, path:str='', id:int=0) -> None:
        path = os.path.join(path,'Tyres_'+str(id)+'.json')
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str='') -> dict:
        with open(path, 'rb') as f:
            return pickle.load(f)

    

def get_tyres_data(df:pd.DataFrame, separators:dict, path:str=None) -> Tyres:
    """
    Function (wrapper) to get the data of the tyres.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataframe containing all data.

    Returns:
    ----------
    tyres_data : set(Tyres)
        The set of the tyres data based on the compound.
    """
    ### Initialize the set 
    tyres_data = set()
    
    if path is not None:
        log.info('Specified load path, trying to find Tyres_*.json files...')
        files = [f for f in os.listdir(path) if f.endswith('.json') and f.startswith('Tyres_')]
        if len(files) > 0:
            log.info('Specified load path with files inside. Loading tyres data from file...')
            for file in files:
                tyres = Tyres(load_path=os.path.join(path,file))
                idx = int(file.replace('Tyres_','').replace('.json',''))
                tyres_data.add((idx,tyres))

            log.info('Loading completed.')
            return tyres_data
                
    
    if path is not None:
        log.info(f'No Tyres_*.json files found in "{path}". Loading tyres data from dataframe.')
    else:
        log.info('No load path specified. Loading tyres data from dataframe.')

    
    
    ### Initialize the columns of the dataframe we are interested in (only the ones that are common to all compounds)
    columns = ['FrameIdentifier','NumLaps','TyresAgeLaps','FLTyreInnerTemperature','FLTyrePressure','FLTyreSurfaceTemperature','FRTyreInnerTemperature','FRTyrePressure','FRTyreSurfaceTemperature','RLTyreInnerTemperature','RLTyrePressure','RLTyreSurfaceTemperature','RRTyreInnerTemperature','RRTyrePressure','RRTyreSurfaceTemperature','TyresDamageFL','TyresDamageFR','TyresDamageRL','TyresDamageRR','TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR','VisualTyreCompound','ActualTyreCompound','RLWheelSlip', 'RRWheelSlip', 'FLWheelSlip', 'FRWheelSlip']
    

    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the compound we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x) and x > 0]
        
        if len(numLaps) > 3:
            start = numLaps[0]
            end = numLaps[-1]

            ### Get the columns data of the compound we are considering (these are particular, not common to all)  # Used to use --> ['tyreActualCompound['+str(key)+']','tyreVisualCompound['+str(key)+']']+  <-- but it is useless
            tyre_columns = columns + ['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

            ## Get the data from the specified columns
            data = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),tyre_columns]

            ### Initialize the tyres data and add it to the set
            tyres = Tyres(data) 
            tyres.save(path,id=key)
            tyres_data.add((key,tyres))
        else:
            ### In case the compound is used less than three laps we cannot deduce anything (we could only use it for the qualification purpose (...))
            compound = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'VisualTyreCompound'].unique()
            compound = [int(x) for x in compound if not math.isnan(x) and x > 0]
            compound = VISUAL_COMPOUNDS[compound[0] if len(compound) == 1 else 0]
            log.warning(f"Insufficient data for the compound '{compound}'. Data are below 3 laps.")

    log.info('Completed.')
    return tyres_data 


if __name__ == "__main__":
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)

    


