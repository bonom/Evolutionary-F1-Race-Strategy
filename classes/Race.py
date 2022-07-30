from classes.Car import get_data as gd
import os
import plotly.express as px
import plotly.graph_objects as go
import pickle
import pandas as pd
import numpy as np
from classes.Utils import ms_to_time

class RaceData():
    def __init__(self, path) -> None:
        if os.path.isfile(os.path.join(path,"RaceData.json")):
            loaded = self.load(path)
            self.data = loaded.data
            self.total_time = loaded.total_time
        else:
            if os.path.exists(os.path.join(path, 'Race')):
                self.data = gd(os.path.join(path, 'Race'), None, [], True)
                for index in self.data.index:
                    self.data.at[index, 'StringLapTime'] = ms_to_time(self.data.at[index, 'LapTime'])

                self.total_time = ms_to_time(self.data['LapTime'].sum())
                self.save(path)
            else:
                print(f"No race data in {path}, skipping step")

    def plot(self, path):
        if not os.path.exists(os.path.join(path, 'Race')):
            return

        circuit_name = path.split("\\")[-1]
        fig = px.line(data_frame=self.data, x='Lap', y='LapTime', title=circuit_name + ' -> Total time: '+self.total_time, text='StringLapTime', color="Compound", color_discrete_map={'Soft': 'Red', 'Medium':'Yellow', 'Hard':'White', 'Inter':'Green', 'Wet':'Blue'})#, template="simple_white")
        
        if circuit_name == "Monza":
            fig.add_trace(
                go.Scatter(
                    x = [24, 23.5, 30, 30.5, 35, 38.5],
                    y = [105000, 106000, 113000, 120000, 134000, 125000],
                    text = ['PitStop', 'SafetyCar', 'PitStop', 'SafetyCar', 'SafetyCar', 'SafetyCar'],
                    mode = 'text',
                    name = 'Events',
                )
            )
            fig.add_shape(type="rect",
                x0=22.5, y0=91000,
                x1=24.5, y1=105000,
                line=dict(
                    color="Orange",
                    width=3,
                    dash="dash"),
                fillcolor="Orange",
                opacity=0.5,
                
                )
            
            fig.add_shape(type="rect",
                x0=28.5, y0=84000,
                x1=32.5, y1=119000,
                line=dict(
                    color="Orange",
                    width=3,
                    dash="dash"),
                fillcolor="Orange",
                opacity=0.5,
                )

            fig.add_shape(type="rect",
                x0=33.5, y0=94000,
                x1=36.5, y1=133000,
                line=dict(
                    color="Orange",
                    width=3,
                    dash="dash"),
                fillcolor="Orange",
                opacity=0.5,
                )  

            fig.add_shape(type="rect",
                x0=37.5, y0=117000,
                x1=39.5, y1=124000,
                line=dict(
                    color="Orange",
                    width=3,
                    dash="dash"),
                fillcolor="Orange",
                opacity=0.5,
                )  
        
        elif circuit_name == "Spielberg":
            fig.add_trace(
                go.Scatter(
                    x = [2, 26.5, 19, 40],
                    y = [78000, 93100, 92500, 83500],
                    text = ['VirtualSafetyCar', 'Rain', 'PitStop', 'PitStop'],
                    mode = 'text',
                    name = 'Events',
                )
            )
            
            
            fig.add_shape(type="rect",
                x0=1.5, y0=78000,
                x1=2.5, y1=79000,
                line=dict(
                    color="Yellow",
                    width=3,
                    dash="dash"),
                fillcolor="Yellow",
                opacity=0.5,
                
                )

            fig.add_shape(type="rect",
                x0=16.5, y0=69000,
                x1=36.5, y1=93000,
                line=dict(
                    color="Blue",
                    width=3,
                    dash="dash"),
                fillcolor="Blue",
                opacity=0.25,
                
                )
        
        else:
            fig.show()
            return
        
        fig.show()
        fig.write_html(os.path.join(path,"RaceData.html"))

    def save(self, path:str):
        with open(os.path.join(path,"RaceData.json"), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    def load(self, path:str):
        with open(os.path.join(path,"RaceData.json"), 'rb') as f:
            data = pickle.load(f)
        return data


def plot_best(values:list, folder:str):
    folder = os.path.join(folder,"Race")
    if not os.path.isdir(folder):
        return 

    data = pd.DataFrame(columns=['Car', 'TotalLapTime'])

    for car_index in range(22):
        lap = pd.read_csv(os.path.join(folder, "Lap.csv"))
        lap = lap.loc[lap["CarIndex"] == car_index, ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")
        lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

        to_drop = []
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
        
        data.at[car_index, 'TotalLapTime'] = lap['LastLapTimeInMS'].sum()

    ### MISSING FIGURE MANAGE
