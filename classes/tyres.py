import pandas as pd
import numpy as np
import plotly.express as px
from typing import Union

ACTUAL_COMPOUNDS: dict = {
    0:"N/A",
    1:"N/A",
    2:"N/A",
    3:"N/A",
    4:"N/A",
    5:"N/A",
    6:"N/A",
    7:"Inter",
    8:"Wet",
    9:"Dry",
    10:"Wet",
    11:"Super Soft",
    12:"Soft",
    13:"Medium",
    14:"Hard",
    15:"Wet",
    16:"C5",
    17:"C4",
    18:"C3",
    19:"C2",
    20:"C1",
}

VISUAL_COMPOUNDS: dict = {
    0:"N/A",
    1:"N/A",
    2:"N/A",
    3:"N/A",
    4:"N/A",
    5:"N/A",
    6:"N/A",
    7:"Inter",
    8:"Wet",
    9:"Dry",
    10:"Wet",
    11:"Super Soft",
    12:"Soft",
    13:"Medium",
    14:"Hard",
    15:"Wet",
    16:"Soft",
    17:"Medium",
    18:"Hard",
    19:"C2",
    20:"C1",
}

TYRE_POSITION: dict ={
    'FL':'Front Left',
    'FR':'Front Right',
    'RL':'Rear Left',
    'RR':'Rear Right',
}
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
    visual_compound : int
        Integer defining the visual compound of the tyre, to get the str version call ``cast_visual_compound(visual_compound)``.
    actual_compound : int
        Integer defining the actual compound of the tyre, to get the str version call ``cast_actual_compound(actual_compound)``.
    age : np.array or list
        It is a list of the age of the tyre every 5 ms (every Frame).
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

    def __init__(self, position:str, wear:Union[np.array,list]=[0.0], damage:Union[np.array,list]=[0.0], visual_compound:int=0, actual_compound:int=0, age:Union[np.array,list]=[0], pressure:Union[np.array,list]=[0.0], inner_temperature:Union[np.array,list]=[0.0], outer_temperature:Union[np.array,list]=[0.0]) -> None:
        self.position = position
        self.wear = np.array(wear)
        self.visual_compound = visual_compound
        self.actual_compound = actual_compound
        self.age = np.array(age)
        self.pressure = np.array(pressure)
        self.inner_temperature = np.array(inner_temperature)
        self.outer_temperature = np.array(outer_temperature)
        self.damage = np.array(damage)

    def __str__(self) -> str:
        to_ret = f"Tyre position: {self.position}\nTyre actual compound: {self.cast_actual_compound(self.actual_compound)}\nTyre visual compound: {self.cast_visual_compound(self.visual_compound)}\nTyre wear: ["
        for wear in self.wear:
            to_ret += f"{wear}%, "
        to_ret += f"]\nTyre age lap(s): ["
        for age in self.age:
            to_ret += f"{age}, "
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
    
    def __index__(self,idx):
        return {'TyrePosition':self.position, 'TyreWear':self.wear[idx], 'TyreDamage':self.damage[idx], 'TyreAge':self.age[idx], 'TyrePressure':self.pressure[idx], 'TyreInnerTemperature':self.inner_temperature[idx], 'TyreOuterTemperature':self.outer_temperature[idx]}

    def cast_actual_compound(self, compound) -> str:
        return ACTUAL_COMPOUNDS[compound]
    
    def cast_visual_compound(self, compound) -> str:
        return VISUAL_COMPOUNDS[compound]

    def cast_tyre_position(self, position) -> str:
        return TYRE_POSITION[position]

    def tyre_wear(self, display:bool=True):
        df = pd.DataFrame({'Wear':self.wear, 'Frame':[i for i in range(len(self.wear))]})
        if display:
            fig = px.line(df, x='Frame',y='Wear', title=self.cast_tyre_position(self.position)+' Tyre Wear')
            fig.show()
        return df

class Tyres:
    """
    Super class of the Tyre class.

    Parameters:
    ----------
    df : pd.Dataframe
        The dataframe containing the data of the tyres. Columns involved are: []
    
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

        self.FL_tyre = Tyre("FL") if df is None else Tyre("FL", df["TyresWearFL"].values,df['TyresDamageFL'].values,visual_tyre_compound.item(),actual_tyre_compound.item(),df['TyresAgeLaps'].values,df['FLTyrePressure'].values,df['FLTyreInnerTemperature'].values,df['FLTyreSurfaceTemperature'].values)
        self.FR_tyre = Tyre("FR") if df is None else Tyre("FR", df["TyresWearFR"].values,df['TyresDamageFR'].values,visual_tyre_compound.item(),actual_tyre_compound.item(),df['TyresAgeLaps'].values,df['FRTyrePressure'].values,df['FRTyreInnerTemperature'].values,df['FRTyreSurfaceTemperature'].values)
        self.RL_tyre = Tyre("RL") if df is None else Tyre("RL", df["TyresWearRL"].values,df['TyresDamageRL'].values,visual_tyre_compound.item(),actual_tyre_compound.item(),df['TyresAgeLaps'].values,df['RLTyrePressure'].values,df['RLTyreInnerTemperature'].values,df['RLTyreSurfaceTemperature'].values)
        self.RR_tyre = Tyre("RR") if df is None else Tyre("RR", df["TyresWearRR"].values,df['TyresDamageRR'].values,visual_tyre_compound.item(),actual_tyre_compound.item(),df['TyresAgeLaps'].values,df['RRTyrePressure'].values,df['RRTyreInnerTemperature'].values,df['RRTyreSurfaceTemperature'].values)
        
        indexes = list()
        for lap in df['NumLaps'].unique():
            indexes.append(df.loc[df['NumLaps'] == lap].index.values[0])

        #print(f"{indexes} {df['FrameIdentifier'].iloc[-1]}")
        self.lap_frames = {lap-1:[i for i in range(indexes[idx],indexes[idx+1] if idx < len(df['NumLaps'].unique()) -1 else df['FrameIdentifier'].iloc[-1])] for idx, lap in enumerate(df['NumLaps'].unique())}
        if 0 in self.lap_frames.keys():
            self.lap_frames.pop(0)
        #for key, values in self.lap_frames.items():
        #    print(key, values[:5])
        print(df["TyresWearFL"].values)

    def __str__(self) -> str:
        return str(self.FL_tyre) +"\n" + str(self.FR_tyre) +"\n" + str(self.RL_tyre) +"\n" + str(self.RR_tyre)
    
    def __index__(self,idx):
        return {'FLTyre':self.FL_tyre[idx], 'FRTyre':self.FR_tyre[idx], 'RLTyre':self.RL_tyre[idx], 'RRTyre':self.RR_tyre[idx]}

    def tyres_wear(self, display:bool=False):
        tyre_wear = pd.concat([self.FL_tyre.tyre_wear(display=False),self.FR_tyre.tyre_wear(display=False),self.RL_tyre.tyre_wear(display=False),self.RR_tyre.tyre_wear(display=False)], axis=1)
        print(tyre_wear.head())

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
    columns = ['FrameIdentifier','NumLaps','TyresAgeLaps','FLTyreInnerTemperature','FLTyrePressure','FLTyreSurfaceTemperature','FRTyreInnerTemperature','FRTyrePressure','FRTyreSurfaceTemperature','RLTyreInnerTemperature','RLTyrePressure','RLTyreSurfaceTemperature','RRTyreInnerTemperature','RRTyrePressure','RRTyreSurfaceTemperature','TyresDamageFL','TyresDamageFR','TyresDamageRL','TyresDamageRR','TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR','VisualTyreCompound','ActualTyreCompound']
    tyres_data = set()
    
    tyres_used = list()
    tyre_columns = df.filter(like='tyreVisualCompound').columns
    for idx, col in enumerate(tyre_columns):
        tyres_used.append((int(df.loc[df[col] > 0,col].unique().item()), df.loc[df[col].first_valid_index(),'FrameIdentifier']))
    
    #print(tyres_used)
    for idx,(compound,frame) in enumerate(tyres_used):    
        numLaps = np.array(df.loc[frame: tyres_used[idx+1][1]-1 if idx < len(tyres_used)-1 else len(df),'NumLaps'].unique())
        start = numLaps[0]
        end = numLaps[-1]

        tyre_columns = columns + ['endLap['+str(idx)+']','tyreActualCompound['+str(idx)+']','tyreVisualCompound['+str(idx)+']']+['lapTimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector1TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector2TimeInMS['+str(lap)+']' for lap in range(start,end)]+['sector3TimeInMS['+str(lap)+']' for lap in range(start,end)]

        data = df.loc[(df['FrameIdentifier'] >= frame) & (df['FrameIdentifier'] < tyres_used[idx+1][1] if idx != len(tyres_used)-1 else 1),tyre_columns]
        tyres_data.add((idx,Tyres(data)))
        #print("\n\n")

    return tyres_data
        
    

if __name__ == "__main__":
    get_tyres_data(pd.read_csv('Car_19_Data.csv'))

    


