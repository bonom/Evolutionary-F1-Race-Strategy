import os
import json
import logging 

from random import SystemRandom

from classes.Utils import get_basic_logger, CIRCUIT

random = SystemRandom()

logger = get_basic_logger(name='Car', level=logging.INFO)

class Car:
    def __init__(self, circuit:str):
        self.circuit = circuit
        self.data = None

        if os.path.isfile('DATA.json'):
            load_data = json.load(open('DATA.json', 'r'))
            if circuit in load_data.keys():
                self.data = load_data[circuit]
            else:
                logger.warning(f"Circuit '{circuit}' not found in the DATA.json file. Random data will be generated.")
        else:
            logger.warning("Unfortunately, the real data is not publicly available. Random data will be generated.")
        
        if self.data is None:
            self.data = {
                "Delta": self.__generate_random_deltas(),
                "Degradation": self.__generate_random_degradations(),
                "Wear": self.__generate_random_wears()
            }

        self.initial_fuel:float = 110.0
        self.fuel_consumption:float = 110/CIRCUIT[circuit]['Laps']
        self.fuel_loss:int = 30 # Milliseconds

        self.tyre_wear_coefficients = {
            'Soft': 100/self.data["Wear"]["Soft"],
            'Medium': 100/self.data["Wear"]["Medium"],
            'Hard': 100/self.data["Wear"]["Hard"],
        }

    def __generate_random_deltas(self):
        soft: int = 0 
        medium: int = random.randint(100, 500)
        hard: int = random.randint(medium, 1000)

        return {"Soft" : soft, "Medium" : medium, "Hard" : hard}

    def __generate_random_degradations(self):
        soft: int = random.randint(50, 200) 
        medium: int = random.randint(25, soft)
        hard: int = random.randint(10, medium)

        return {"Soft" : soft, "Medium" : medium, "Hard" : hard}
    
    def __generate_random_wears(self):
        soft: int = random.randint(20, 35) # soft tyre has 0 as delta since it is the default value referring to
        medium: int = random.randint(soft, 45)
        hard: int = random.randint(medium, 60)

        return {"Soft" : soft, "Medium" : medium, "Hard" : hard}

    def predict_fuel_weight(self, lap:int) -> float:
        return self.initial_fuel - self.fuel_consumption*lap
    
    def predict_fuel_loss(self, lap:int) -> int:
        return self.initial_fuel*self.fuel_loss - self.fuel_loss*lap

    def predict_tyre_time_lose(self, compound:str, tyresAge:int) -> int:
        return self.data['Degradation'][compound]*tyresAge

    def predict_laptime(self, tyre:str, lap:int, tyresAge:int) -> int:
        time = self.data['Delta'][tyre] + self.predict_tyre_time_lose(tyre, tyresAge) + self.predict_fuel_loss(lap)
        return time
    
    def predict_tyre_wear(self, compound:str, compoundAge:int) -> float:
        return self.tyre_wear_coefficients[compound]*compoundAge
    
