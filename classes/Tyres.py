import sys, os
from typing import Union
import pandas as pd
import numpy as np
import math
import pickle
import plotly.express as px
import plotly
from classes.RangeDictionary import RangeDictionary
from classes.Utils import ACTUAL_COMPOUNDS, VISUAL_COMPOUNDS, TYRE_POSITION, get_basic_logger, get_host

log = get_basic_logger("Tyres")

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

    def __init__(self, position:str, wear:pd.DataFrame=None, damage:pd.DataFrame=None, pressure:pd.DataFrame=None, inner_temperature:pd.DataFrame=None, outer_temperature:pd.DataFrame=None, laps_data:dict={0:0}, slip:pd.DataFrame=None, start:int=0) -> None:
                
        self.position = position
        self.lap_frames = laps_data
        self.wear = RangeDictionary(data=wear)
        self.pressure = RangeDictionary(pressure)
        self.inner_temperature = RangeDictionary(inner_temperature)
        self.outer_temperature = RangeDictionary(outer_temperature)
        self.damage = RangeDictionary(damage)
        self.slip = RangeDictionary(slip)


        ### MODEL ###
        x = np.array([self.get_lap(key) for key in self.wear.keys()])
        
        y = np.array(list(self.wear.values()))
        if math.isnan(y[0]):
            y[0] = 0
            
        self.coeff = np.polyfit(x, y, 1)

    def __len__(self) -> int:
        if len(self.lap_frames) == 0:
            return 0
        return len(self.lap_frames.keys())

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
        if get_float:
            return self.lap_frames[frame]
        
        return int(self.lap_frames[frame])

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
            
            if get_host() == 'DESKTOP-KICFR1D':
                plotly.offline.plot(fig, filename='Plots/Tyre'+str(self.position)+' Wear.html')
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
            
            if get_host() == 'DESKTOP-KICFR1D':
                plotly.offline.plot(fig, filename='Plots/Tyre'+str(self.position)+' Slip.html')
            else:
                fig.show()

        return dict_items
    
    def predict_wear(self, x_predict:int, intercept:float=0.0) -> float:
        """
        Return the 2 coefficient beta_0 and beta_1 for the linear model that fits the data : Time/Fuel
        """
    
        return self.coeff[0]*x_predict + self.coeff[1]


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

            if len(visual_tyre_compound) != 1 or len(actual_tyre_compound) != 1:
                log.warning(f"The visual and actual tyre compound have more compounds.\n\t\t\t\t\tVisual Tyre Compound: {visual_tyre_compound}\n\t\t\t\t\tActual Tyre Compound: {actual_tyre_compound}\n\t\t\t\t\tTrying to fix it...")
                col_visual = df.filter(like="tyreVisualCompound[").columns.to_list()
                col_actual = df.filter(like="tyreActualCompound[").columns.to_list()
                
                if len(col_visual) == 1 and len(col_actual) == 1:
                    col_visual = col_visual[0]
                    col_actual = col_actual[0]
                
                    visual_tyre_compound = np.array([int(x) for x in df[col_visual].unique() if not math.isnan(float(x)) and float(x) > 0])
                    actual_tyre_compound = np.array([int(x) for x in df[col_actual].unique() if not math.isnan(float(x)) and float(x) > 0])
                
                else:
                    log.critical("The dataframe contains more than one tyre compound and unable to fix it!")
                    visual_tyre_compound = visual_tyre_compound[-1:] if len(visual_tyre_compound) == 2 else visual_tyre_compound[-2:-1]
                    actual_tyre_compound = actual_tyre_compound[-1:] if len(actual_tyre_compound) == 2 else actual_tyre_compound[-2:-1]

                if len(visual_tyre_compound) > 1 or len(actual_tyre_compound) > 1:
                    log.critical("The dataframe contains more than one tyre compound!")
                    raise ValueError("VisualTyreCompound contains {}\nActualTyreCompound contains {}".format(visual_tyre_compound, actual_tyre_compound))

            self.visual_tyre_compound = visual_tyre_compound.item() if len(visual_tyre_compound) == 1 else 0
            self.actual_tyre_compound = actual_tyre_compound.item() if len(actual_tyre_compound) == 1 else 0

            indexes = list()
            laps = [int(x) for x in df['NumLaps'].unique() if not math.isnan(x)]
            for lap in laps:
                if not math.isnan(lap):
                    indexes.append((min(df.loc[df['NumLaps'] == lap,'FrameIdentifier'].values), max(df.loc[df['NumLaps'] == lap,'FrameIdentifier'].values)))
            
            self.frames_lap = dict()
            self.lap_frames = dict()
            max_lap_len = len(indexes)
            for idx in range(max_lap_len):
                if idx < max_lap_len-1:
                    start,_ = indexes[idx]
                    end,_ = indexes[idx+1]
                else:
                    start, end = indexes[idx]
                for i in range(start,end):
                    self.frames_lap[i] = idx + round(((i - start)/(end - start)),4)
                    try:
                        self.lap_frames[idx].append(i)
                    except:
                        self.lap_frames[idx] = [i]

            df = df.loc[(df['FrameIdentifier'] >= list(self.frames_lap.keys())[0]) & (df['FrameIdentifier'] <= list(self.frames_lap.keys())[-1])]

            self.FL_tyre = Tyre("FL", df[['FrameIdentifier','TyresWearFL']],df[['FrameIdentifier','TyresDamageFL']],df[['FrameIdentifier','FLTyrePressure']],df[['FrameIdentifier','FLTyreInnerTemperature']],df[['FrameIdentifier','FLTyreSurfaceTemperature']],self.frames_lap,df[['FrameIdentifier','FLWheelSlip']])
            self.FR_tyre = Tyre("FR", df[['FrameIdentifier','TyresWearFR']],df[['FrameIdentifier','TyresDamageFR']],df[['FrameIdentifier','FRTyrePressure']],df[['FrameIdentifier','FRTyreInnerTemperature']],df[['FrameIdentifier','FRTyreSurfaceTemperature']],self.frames_lap,df[['FrameIdentifier','FRWheelSlip']])
            self.RL_tyre = Tyre("RL", df[['FrameIdentifier','TyresWearRL']],df[['FrameIdentifier','TyresDamageRL']],df[['FrameIdentifier','RLTyrePressure']],df[['FrameIdentifier','RLTyreInnerTemperature']],df[['FrameIdentifier','RLTyreSurfaceTemperature']],self.frames_lap,df[['FrameIdentifier','RLWheelSlip']])
            self.RR_tyre = Tyre("RR", df[['FrameIdentifier','TyresWearRR']],df[['FrameIdentifier','TyresDamageRR']],df[['FrameIdentifier','RRTyrePressure']],df[['FrameIdentifier','RRTyreInnerTemperature']],df[['FrameIdentifier','RRTyreSurfaceTemperature']],self.frames_lap,df[['FrameIdentifier','RRWheelSlip']])

            self.wear_coeff = {'FL': self.FL_tyre.coeff, 'FR': self.FR_tyre.coeff, 'RL': self.RL_tyre.coeff, 'RR': self.RR_tyre.coeff}

        elif load_path is not None:
            data:Tyres = self.load(load_path)
            self.FL_tyre = data.FL_tyre
            self.FR_tyre = data.FR_tyre
            self.RL_tyre = data.RL_tyre
            self.RR_tyre = data.RR_tyre
            self.visual_tyre_compound = data.visual_tyre_compound
            self.actual_tyre_compound = data.actual_tyre_compound
            self.frames_lap = data.frames_lap
            self.lap_frames = data.lap_frames
            self.wear_coeff = data.wear_coeff


    def __str__(self) -> str:
        return str(self.FL_tyre) +"\n" + str(self.FR_tyre) +"\n" + str(self.RL_tyre) +"\n" + str(self.RR_tyre)+"\n"

    def __getitem__(self,idx) -> dict:
        if idx == -1:
            idx = self.__len__() - 1
        idx -= list(self.frames_lap.keys())[0]
        return {'Lap':self.get_lap(idx)+1,'FLTyre':self.FL_tyre[idx], 'FRTyre':self.FR_tyre[idx], 'RLTyre':self.RL_tyre[idx], 'RRTyre':self.RR_tyre[idx]}
    
    def __len__(self) -> int:
        if len(self.frames_lap) == 0:
            return 0
        return len(self.frames_lap.keys())
        
    def get_actual_compound(self, compound:int=None) -> str:
        return ACTUAL_COMPOUNDS[self.actual_tyre_compound if compound is None else compound]
    
    def get_visual_compound(self, compound:int=None) -> str:
        return VISUAL_COMPOUNDS[self.visual_tyre_compound if compound is None else compound]

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
            if get_host() == 'DESKTOP-KICFR1D':
                plotly.offline.plot(fig, filename='Plots/Tyres Wear.html')
            else:
                fig.show()

        return df.to_dict()
    
    def predict_wears(self, x_predict:int, single:bool=False) -> dict:
        FL_Tyre_model = self.FL_tyre.predict_wear(x_predict)
        FR_Tyre_model = self.FR_tyre.predict_wear(x_predict)
        RL_Tyre_model = self.RL_tyre.predict_wear(x_predict)
        RR_Tyre_model = self.RR_tyre.predict_wear(x_predict)

        predictions = dict({'FL' : FL_Tyre_model, 'FR' : FR_Tyre_model,'RL' : RL_Tyre_model, 'RR' : RR_Tyre_model})
        
        if single:
            return sum(predictions.values()) / 4
        
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
            
            if get_host() == 'DESKTOP-KICFR1D':
                plotly.offline.plot(fig, filename='Plots/Tyres Slip.html')
            else:
                fig.show()
    
        return df#.to_dict()


    def get_age(self,frame:int=0) -> Union[int,float]:
        return self.get_lap(frame)      

    def get_lap(self, frame, get_float:bool=False) -> Union[int,float]:
        if get_float:
            return self.frames_lap[frame]
        
        return int(self.frames_lap[frame])

    def get_frame(self, lap_num:Union[int,float]) -> int:
        if isinstance(lap_num, float):
            length = len(self.lap_frames[int(lap_num)])
            return self.lap_frames[int(lap_num)][int((length*lap_num)%length)]
        
        return self.lap_frames[lap_num][0]    

    def get_avg_wear(self, frame:int=0, lap:int=0) -> float:
        if frame == 0:
            frame = self.get_frame(lap)

        fl_wear = self.FL_tyre.wear[frame]
        fr_wear = self.FR_tyre.wear[frame]
        rl_wear = self.RL_tyre.wear[frame]
        rr_wear = self.RR_tyre.wear[frame]

        return (fl_wear + fr_wear + rl_wear + rr_wear) / 4

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
    tyres_data = dict()
    
    if path is not None:
        log.info('Specified load path, trying to find Tyres_*.json files...')
        files = [f for f in os.listdir(path) if f.endswith('.json') and f.startswith('Tyres_')]
        if len(files) > 0:
            log.info('Specified load path with files inside. Loading tyres data from file...')
            for file in files:
                tyres = Tyres(load_path=os.path.join(path,file))
                idx = int(file.replace('Tyres_','').replace('.json',''))
                tyres_data[idx] = tyres

            log.info('Loading completed.')
            return tyres_data
                
    
    if path is not None:
        log.info(f'No Tyres_*.json files found in "{path}". Loading tyres data from dataframe.')
    else:
        log.info('No load path specified. Loading tyres data from dataframe.')

    
    
    ### Initialize the columns of the dataframe we are interested in
    tyre_columns = ['FrameIdentifier','NumLaps','TyresAgeLaps','FLTyreInnerTemperature','FLTyrePressure','FLTyreSurfaceTemperature','FRTyreInnerTemperature','FRTyrePressure','FRTyreSurfaceTemperature','RLTyreInnerTemperature','RLTyrePressure','RLTyreSurfaceTemperature','RRTyreInnerTemperature','RRTyrePressure','RRTyreSurfaceTemperature','TyresDamageFL','TyresDamageFR','TyresDamageRL','TyresDamageRR','TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR','VisualTyreCompound','ActualTyreCompound','RLWheelSlip', 'RRWheelSlip', 'FLWheelSlip', 'FRWheelSlip']
    
    ### Cycle over all the times we box
    for key, (sep_start,sep_end) in separators.items():
        ### Get the numLap data of the compound we are considering
        numLaps = np.array(df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'NumLaps'].unique())
        numLaps = [int(x) for x in numLaps if not math.isnan(x)]
        
        if len(numLaps) > 3:
            temp_cols = list()

            temp_df = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end)]

            actual_cols = temp_df[temp_df.filter(like='tyreActualCompound').columns].notna().columns.to_list()
            visual_cols = temp_df[temp_df.filter(like='tyreVisualCompound').columns].notna().columns.to_list()

            for col in actual_cols:
                if str(key) in actual_cols:
                    temp_cols.append(col)
            
            for col in visual_cols:
                if str(key) in visual_cols:
                    temp_cols.append(col)

            ## Get the data from the specified columns            
            if len(temp_cols) > 0:
                data = temp_df[tyre_columns+temp_cols]
            else:
                data = temp_df[tyre_columns]

            ### Initialize the tyres data and add it to the set
            tyres = Tyres(df=data) 
            tyres.save(path,id=key)
            tyres_data[key] = tyres
        else:
            ### In case the compound is used less than three laps we cannot deduce anything (we could only use it for the qualification purpose (...))
            compound = df.loc[(df['FrameIdentifier'] >= sep_start) & (df['FrameIdentifier'] <= sep_end),'VisualTyreCompound'].unique()
            compound = [int(x) for x in compound if not math.isnan(x) and x > 0]
            compound = VISUAL_COMPOUNDS[compound[0] if len(compound) == 1 else 0]
            log.warning(f"Insufficient data (below 3 laps). Skipping {key+1}/{len(separators.keys())}. Compound -> {compound}.")

    return tyres_data 


if __name__ == "__main__":
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)

    


