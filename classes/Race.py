from classes.Car import get_data as gd
import os
import plotly.express as px
import plotly.graph_objects as go
import pickle

from classes.Utils import ms_to_time

class RaceData():
    def __init__(self, path) -> None:
        if os.path.isfile(os.path.join(path,"RaceData.json")):
            loaded = self.load(path)
            self.data = loaded.data
            self.total_time = loaded.total_time
        else:
            self.data = gd(os.path.join(path, 'Race'), None, [], True)
            for index in self.data.index:
                self.data.at[index, 'StringLapTime'] = ms_to_time(self.data.at[index, 'LapTime'])
            
            self.total_time = ms_to_time(self.data['LapTime'].sum())
            self.save(path)
            

    def plot(self, path):
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