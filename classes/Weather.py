import numpy as np
from random import SystemRandom
random = SystemRandom()

from classes.Utils import CIRCUIT


class Weather:
    def __init__(self, circuit) -> None:
        self.numLaps = CIRCUIT[circuit]['Laps']+1
        self.weather = {i:0 for i in  range(1, self.numLaps + 1)}

        ### HARDCODED FOR DEBUG PURPOSES:
        for lap in range(1, self.numLaps + 1):
            self.weather[lap] = 0
        return 
        if input(f"Do you want to insert manually the weather data for '{circuit}'? (y/n) ") in ['y', 'Y', 'S', 's']:
            if input(f"Do you want the race to be completely sunny or completely wet? (y/n) ") in ['y', 'Y', 'S', 's']:
                if input(f"Do you want a total SUNNY race? (y/n) ") in ['y', 'Y', 'S', 's']:
                    for lap in range(1, self.numLaps + 1):
                        self.weather[lap] = 0
                else:
                    for lap in range(1, self.numLaps + 1):
                        self.weather[lap] = random.randint(70,100)
            else:
                for lap in range(1, self.numLaps + 1):
                    self.weather[lap] = input(f"Lap {lap} has rain in percentage (0-100): ") 
        else:
            for lap in range(1, self.numLaps + 1):
                self.weather[lap] = random.randint(0, 100)

    def get_weather(self, lap):
        w = self.weather[lap]

        if w < 30:
            return 'Dry'
        elif w > 50 and w < 80:
            return 'Wet'
        elif w >= 80:
            return 'VWet'

        return 'Dry/Wet'

    def get_weather_percentage(self, lap):
        return self.weather[lap]
    
    def get_weather_percentage_list(self):
        return self.weather
    
    def get_weather_list(self):
        return [self.get_weather(i) for i in self.weather.keys()]