import sys
from typing import Union
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
from classes.RangeDictionary import RangeDictionary
from classes.Utils import ACTUAL_COMPOUNDS, VISUAL_COMPOUNDS, TYRE_POSITION, get_basic_logger, separate_data

log = get_basic_logger("TYRES")

class Tyre:
    """
    Class for a single tyre.

    Parameters:
    ----------
    position : str 
        The position of the tyre: 'FL' = Front Left, 'FR' = Right Front, 'RL' = Rear Left, 'RR' = Rear Right.
    wear : np.array or list
        It is a list of the wear percentages of the tyre every 5 ms (which corresponds to a Frame).
    damage : np.array or list
        It is a list of the damage percentages of the tyre every 5 ms (every Frame).
    pressure : np.array or list
        It is a list of the pressure of the tyre every 5 ms (every Frame).
    inner_temperature : np.array or list
        It is a list of the inner temperature of the tyre every 5 ms (every Frame).
    outer_temperature : np.array or list
        It is a list of the outer temperature of the tyre every 5 ms (every Frame).
    
    Functions:
    ----------
    cast_visual_compound(self,visual_compound:int) : str 
        cast the integer id of the visual compound to the string one.

    cast_actual_compound(self,actual_compound:int) : str 
        cast the integer id of the actual compound to the string one.

    TODO:
    ----------
    tyre_wear(self, display:bool=False) : 
        returns the wear function of the tyre. 
    """

    def __init__(self, position:str, wear:Union[np.array,list]=[0.0], damage:Union[np.array,list]=[0.0], pressure:Union[np.array,list]=[0.0], inner_temperature:Union[np.array,list]=[0.0], outer_temperature:Union[np.array,list]=[0.0], laps_data:dict={0:0}, slip:Union[np.array,list]=[0.0]) -> None:
        self.position = position
        self.wear = RangeDictionary(np.array(wear))
        self.pressure = RangeDictionary(np.array(pressure))
        self.inner_temperature = RangeDictionary(np.array(inner_temperature))
        self.outer_temperature = RangeDictionary(np.array(outer_temperature))
        self.damage = RangeDictionary(np.array(damage))
        self.lap_frames = laps_data
        self.slip = RangeDictionary(np.array(slip))

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
        if idx == -1:
            idx = self.__len__() 
        return {'TyrePosition':self.position, 'TyreWear':self.wear[idx], 'TyreDamage':self.damage[idx], 'TyrePressure':self.pressure[idx], 'TyreInnerTemperature':self.inner_temperature[idx], 'TyreOuterTemperature':self.outer_temperature[idx]}
    
    def get_lap(self,frame:int, get_float:bool=False):
        first_value = list(self.lap_frames.values())[0][0]
        if get_float:
            for key, value in self.lap_frames.items():
                if frame+first_value in value:
                    try:
                        idx = value.index(frame+first_value)
                    except Exception as e:
                        idx = 0
                    if idx == 0:
                        return key
                    return float(key)+(idx/len(value))

        for key, value in self.lap_frames.items():
            if frame+first_value in value:
                return key
        
        return 0

    def cast_tyre_position(self, position) -> str:
        return TYRE_POSITION[position]

    def tyre_wear(self, display:bool=False) -> dict:
        dict_items = {'Frame':[int(key) for key in self.wear.keys()],'Wear_'+str(self.position):[value for value in self.wear.values()]}
        
        if display:
            df = pd.DataFrame(dict_items)
            for row in df.index:
                df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)

            fig = px.line(df, x='Lap',y='Wear_'+str(self.position), title=self.cast_tyre_position(self.position)+' Tyre Wear')
            fig.update(layout_yaxis_range = [0,100])
            #fig.update(layout_yaxis_range = [0,max(df['Wear_'+str(self.position)])])
            #plotly.offline.plot(fig, filename='Tyre'+str(self.position)+' Wear.html')
            fig.show()

        #df.set_index('Frame', inplace=True)
        return dict_items
    
    def tyre_slip(self, display:bool=False) -> dict:
        dict_items = {'Frame':[int(key) for key in self.slip.keys()],'Slip_'+str(self.position):[value for value in self.slip.values()]}
        
        if display:
            df = pd.DataFrame(dict_items)
            for row in df.index:
                df.loc[row,'Lap'] = self.get_lap(df.loc[row,'Frame'],True)

            fig = px.line(df, x='Lap',y='Slip_'+str(self.position), title=self.cast_tyre_position(self.position)+' Tyre Slip')
            fig.update(layout_yaxis_range = [-1.1,1.1])
            #plotly.offline.plot(fig, filename='Tyre'+str(self.position)+' Slip.html')
            fig.show()

        return dict_items

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
        cast the integer id of the visual compound to the string one.

    cast_actual_compound(self,actual_compound:int) : str 
        cast the integer id of the actual compound to the string one.
    
    TODO:
    ----------
    tyres_wear(self, display:bool=False) : 
        returns the wear function of the tyres.
    """

    def __init__(self, df:pd.DataFrame=None) -> None:
        visual_tyre_compound = df[df.filter(like='tyreVisualCompound').columns.item()].unique()
        actual_tyre_compound = df[df.filter(like='tyreActualCompound').columns.item()].unique()

        visual_tyre_compound = np.array([int(x) for x in visual_tyre_compound if int(x) > 0])
        actual_tyre_compound = np.array([int(x) for x in actual_tyre_compound if int(x) > 0]) 

        self.visual_tyre_compound = visual_tyre_compound.item()
        self.actual_tyre_compound = actual_tyre_compound.item()

        if len(visual_tyre_compound) > 1 or len(actual_tyre_compound) > 1:
            log.critical("The dataframe contains more than one tyre compound:\nVisualTyreCompound contains {}\nActualTyreCompound contains {}".format(visual_tyre_compound, actual_tyre_compound))

        indexes = list()
        for lap in df['NumLaps'].unique():
            indexes.append(df.loc[df['NumLaps'] == lap].index.values[0])

        if len(indexes) < 3:
            df = None

        self.lap_frames = dict()
        if df is not None:
            max_lap_len = len(df.filter(like="lapTimeInMS").columns.to_list())
            for idx in range(max_lap_len):
                if idx < max_lap_len-1:
                    self.lap_frames[idx] = [i for i in range(indexes[idx],indexes[idx+1])]
                else:
                    self.lap_frames[idx] = [i for i in range(indexes[idx],df['FrameIdentifier'].iloc[-1])]
        
        self.FL_tyre = Tyre("FL") if df is None else Tyre("FL", df["TyresWearFL"].values,df['TyresDamageFL'].values,df['FLTyrePressure'].values,df['FLTyreInnerTemperature'].values,df['FLTyreSurfaceTemperature'].values,self.lap_frames,df['FLWheelSlip'].values)
        self.FR_tyre = Tyre("FR") if df is None else Tyre("FR", df["TyresWearFR"].values,df['TyresDamageFR'].values,df['FRTyrePressure'].values,df['FRTyreInnerTemperature'].values,df['FRTyreSurfaceTemperature'].values,self.lap_frames,df['FRWheelSlip'].values)
        self.RL_tyre = Tyre("RL") if df is None else Tyre("RL", df["TyresWearRL"].values,df['TyresDamageRL'].values,df['RLTyrePressure'].values,df['RLTyreInnerTemperature'].values,df['RLTyreSurfaceTemperature'].values,self.lap_frames,df['RLWheelSlip'].values)
        self.RR_tyre = Tyre("RR") if df is None else Tyre("RR", df["TyresWearRR"].values,df['TyresDamageRR'].values,df['RRTyrePressure'].values,df['RRTyreInnerTemperature'].values,df['RRTyreSurfaceTemperature'].values,self.lap_frames,df['RRWheelSlip'].values)

        self.lap_times = [0]
        if df is not None:
            self.lap_times = list()
            for col in df.filter(like="lapTimeInMS").columns.to_list():
                self.lap_times.append(df[col].iloc[-1])
        
        self.lap_times = np.array(self.lap_times)
        
        self.Sector1InMS = [0]
        if df is not None:
            self.Sector1InMS = list()
            for col in df.filter(like="sector1TimeInMS").columns.to_list():
                self.Sector1InMS.append(df[col].iloc[-1])
        
        self.Sector1InMS = np.array(self.Sector1InMS)

        self.Sector2InMS = [0]
        if df is not None:
            self.Sector2InMS = list()
            for col in df.filter(like="sector2TimeInMS").columns.to_list():
                self.Sector2InMS.append(df[col].iloc[-1])
        
        self.Sector2InMS = np.array(self.Sector2InMS)

        self.Sector3InMS = [0]
        if df is not None:
            self.Sector3InMS = list()
            for col in df.filter(like="sector3TimeInMS").columns.to_list():
                self.Sector3InMS.append(df[col].iloc[-1])
        
        self.Sector3InMS = np.array(self.Sector3InMS)    


    def __str__(self) -> str:
        return str(self.FL_tyre) +"\n" + str(self.FR_tyre) +"\n" + str(self.RL_tyre) +"\n" + str(self.RR_tyre)+"\n"+str(self.lap_times)+"\n"+str(self.Sector1InMS)+"\n"+str(self.Sector2InMS)+"\n"+str(self.Sector3InMS)

    def __getitem__(self,idx) -> dict:
        if idx == -1:
            idx = self.__len__() 
        return {'Lap':self.get_lap(idx)+1,'FLTyre':self.FL_tyre[idx], 'FRTyre':self.FR_tyre[idx], 'RLTyre':self.RL_tyre[idx], 'RRTyre':self.RR_tyre[idx], 'LapTimeInMS':self.lap_times[self.get_lap(idx)], 'Sector1InMS':self.Sector1InMS[self.get_lap(idx)], 'Sector2InMS':self.Sector2InMS[self.get_lap(idx)], 'Sector3InMS':self.Sector3InMS[self.get_lap(idx)]}
    
    def __len__(self) -> int:
        if len(self.lap_frames) == 0:
            return 0
        return list(self.lap_frames[list(self.lap_frames.keys())[-1]])[-1]

    
    def cast_actual_compound(self, compound) -> str:
        return ACTUAL_COMPOUNDS[compound]
    
    def cast_visual_compound(self, compound) -> str:
        return VISUAL_COMPOUNDS[compound]

    def tyres_timing(self, display:bool=False) -> dict:
        timing = {'Lap':[],'LapTimeInMS':[]}
        for lap in self.lap_frames.keys():
            timing['Lap'].append(lap+1)
            if self.lap_times[lap] != 0:
                timing['LapTimeInMS'].append(self.lap_times[lap])
            else:
                timing['LapTimeInMS'].append(np.nan)
        
        if display:
            df = pd.DataFrame(timing)
            fig = px.line(df, x='Lap',y='LapTimeInMS', title='Lap Times on '+self.cast_visual_compound(self.visual_tyre_compound)+" compound",markers=True,range_y=[min(timing['LapTimeInMS'])-1000,max(timing['LapTimeInMS'])+1000])
            #plotly.offline.plot(fig, filename='Tyres Timing.html')
            fig.show()
            
        return timing

    def tyres_wear(self, display:bool=False) -> dict:
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
            
        if display:
            fig = px.line(df, x='Lap',y=['Wear_FL', 'Wear_FR', 'Wear_RL', 'Wear_RR'], title='Tyre Wear on '+self.cast_visual_compound(self.visual_tyre_compound)+" compound",markers=True,range_y=[0,100])
            #fig.update(layout_yaxis_range = [0,max(max(df['Wear_FL']),max(df['Wear_FR']),max(df['Wear_RL']),max(df['Wear_RR']))])
            #plotly.offline.plot(fig, filename='Tyres Wear.html')
            fig.show()

        return df.to_dict()
    
    def tyres_slip(self, display:bool=False)->dict:
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
            
        if display:
            fig = px.line(df, x='Lap',y=['Slip_FL', 'Slip_FR', 'Slip_RL', 'Slip_RR'], title='Tyre Slip on '+self.cast_visual_compound(self.visual_tyre_compound)+" compound",markers=True,range_y=[-1.1,1.1])
            #plotly.offline.plot(fig, filename='Tyres.html')
            fig.show()
    
        return df.to_dict()


    def get_tyres_age(self,frame:int=0) -> set:
        return self.get_lap(frame)      

    def get_lap(self,frame:int, get_float:bool=False) -> Union[int,float]:
        first_value = list(self.lap_frames.values())[0][0]
        
        if get_float:
            for key, value in self.lap_frames.items():
                if frame+first_value in value:
                    try:
                        idx = value.index(frame+first_value)
                    except Exception as e:
                        idx = 0
                    if idx == 0:
                        return key
        
                    return float(key)+(idx/len(value))

        for key, value in self.lap_frames.items():
            if frame+first_value in value:
                return key
        
        return 0

def get_tyres_data(df:pd.DataFrame) -> Tyres:
    """
    Function to get the data of the tyres.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataframe containing all data.

    Returns:
    ----------
    tyres_data : set(Tyres)
        The set of the tyres data based on the compound.
    """
    log.info("Separating compounds data.")
    separators = separate_data(df)
    
    columns = ['FrameIdentifier','NumLaps','TyresAgeLaps','FLTyreInnerTemperature','FLTyrePressure','FLTyreSurfaceTemperature','FRTyreInnerTemperature','FRTyrePressure','FRTyreSurfaceTemperature','RLTyreInnerTemperature','RLTyrePressure','RLTyreSurfaceTemperature','RRTyreInnerTemperature','RRTyrePressure','RRTyreSurfaceTemperature','TyresDamageFL','TyresDamageFR','TyresDamageRL','TyresDamageRR','TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR','VisualTyreCompound','ActualTyreCompound','RLWheelSlip', 'RRWheelSlip', 'FLWheelSlip', 'FRWheelSlip']
    tyres_data = set()

    for key, values in separators.items():
        numLaps = np.array(df.loc[values[0]: values[-1],'NumLaps'].unique())

        if len(numLaps) > 3:
            start = numLaps[0]
            end = numLaps[-1]

            tyre_columns = columns + ['tyreActualCompound['+str(key)+']','tyreVisualCompound['+str(key)+']']+['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

            data = df.loc[(df['FrameIdentifier'] >= values[0]) & (df['FrameIdentifier'] <= values[-1]),tyre_columns]
            
            tyres = Tyres(data)
            tyres_data.add((key,tyres))
        else:
            compound = VISUAL_COMPOUNDS[df.loc[df['FrameIdentifier'] == values[0],'VisualTyreCompound'].unique().item()]
            log.warning(f"Insufficient data for the compound '{compound}'. Data are below 3 laps.")

    return tyres_data

"""
    #############################################################################################################################
    # 
    #                                       OLD ONE - Working well but very inefficient
    #
    #############################################################################################################################
def get_tyres_data(df:pd.DataFrame) -> Tyres:

    columns = ['FrameIdentifier','NumLaps','TyresAgeLaps','FLTyreInnerTemperature','FLTyrePressure','FLTyreSurfaceTemperature','FRTyreInnerTemperature','FRTyrePressure','FRTyreSurfaceTemperature','RLTyreInnerTemperature','RLTyrePressure','RLTyreSurfaceTemperature','RRTyreInnerTemperature','RRTyrePressure','RRTyreSurfaceTemperature','TyresDamageFL','TyresDamageFR','TyresDamageRL','TyresDamageRR','TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR','VisualTyreCompound','ActualTyreCompound','RLWheelSlip', 'RRWheelSlip', 'FLWheelSlip', 'FRWheelSlip']
    tyres_data = set()

    tyres_used = list()
    tyres_compounds = [int(compound) for compound in df['VisualTyreCompound'].unique() if int(compound) != 0]
    
    for idx, compound in enumerate(tyres_compounds):
        tyres_used.append((compound, df.loc[df['VisualTyreCompound'] == compound,'FrameIdentifier'].values[0]))
    
    for idx,(compound,frame) in enumerate(tyres_used):    
        numLaps = np.array(df.loc[frame: tyres_used[idx+1][1]-1 if idx < len(tyres_used)-1 else len(df),'NumLaps'].unique())
        start = numLaps[0]
        end = numLaps[-1]

        tyre_columns = columns + ['endLap['+str(idx)+']','tyreActualCompound['+str(idx)+']','tyreVisualCompound['+str(idx)+']']+['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

        data = df.loc[(df['FrameIdentifier'] >= frame) & (df['FrameIdentifier'] < tyres_used[idx+1][1] if idx != len(tyres_used)-1 else 1),tyre_columns]
        
        tyres = Tyres(data)
        
        if len(tyres)!= 0:
            tyres_data.add((idx,tyres))
        else:
            log.warning(f"Insufficient data for compound '{VISUAL_COMPOUNDS[compound]}'. Data are below 3 laps.")
    
    return tyres_data
    """
        
    

if __name__ == "__main__":
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)

    


