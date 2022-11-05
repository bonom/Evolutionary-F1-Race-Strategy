import json

from classes.Utils import CIRCUIT

class Car:
    def __init__(self, circuit:str):
        self.circuit = circuit

        load_data = json.load(open('DATA.json', 'r'))
        self.data = load_data[circuit]

        self.initial_fuel:float = 110.0
        self.fuel_consumption:float = 110/CIRCUIT[circuit]['Laps']
        self.fuel_loss:int = 30 # Milliseconds

        self.tyre_wear_coefficients = {
            'Soft': 100/self.data["Wear"]["Soft"],
            'Medium': 100/self.data["Wear"]["Medium"],
            'Hard': 100/self.data["Wear"]["Hard"],
        }

    def predict_fuel_weight(self, lap:int) -> float:
        return self.initial_fuel - self.fuel_consumption*lap
    
    def predict_fuel_loss(self, lap:int) -> int:
        return self.initial_fuel*self.fuel_loss - self.fuel_loss*lap

    def predict_tyre_time_lose(self, compound:str, lap:int) -> int:
        return self.data['Degradation'][compound]*lap

    def predict_laptime(self, tyre:str, lap:int, tyresAge:int) -> int:
        time = self.data['Delta'][tyre] + self.predict_tyre_time_lose(tyre, tyresAge) + self.predict_fuel_loss(lap)
        return time
    
    def predict_tyre_wear(self, compound:str, compoundAge:int) -> float:
        return self.tyre_wear_coefficients[compound]*compoundAge
    
