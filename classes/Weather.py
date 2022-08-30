import os, sys
from classes.Utils import CIRCUIT
from random import SystemRandom
random = SystemRandom()

def weather_summary(circuit:str, weather_file:str):
    """
    Returns a string with the weather summary.
    """
    weather = Weather(circuit=circuit,filename=weather_file)

    wlist = weather.get_weather_list()

    to_ret = []
    for index, weather in enumerate(wlist):
        if index == 0 or weather != wlist[index-1]:
            to_ret.append((index+1, weather))

    return to_ret

class Weather:
    def __init__(self, circuit:str, filename:str=None) -> None:

        if filename is None:
            path = os.path.join('Data',circuit,'Weather')

            if not os.path.exists(path):
                print(f"Path '{path}' does not exists.")
                sys.exit(1)

            files = os.listdir(path)

            if '.DS_Store' in files:
                files.remove('.DS_Store')

            if len(files) == 0:
                print(f"There are no available weathers for circuit '{circuit}'. Insert them and run again")
                sys.exit(1)

            print(f"Available weathers for circuit '{circuit}': ")
            for idx, w in enumerate(files):
                print(f"{idx+1}. {w}")
            
            index = int(input(f"\nSelect weather by number: "))
            file = os.path.join(path, files[index-1])

            self.filename = files[index-1]
        
        else:
            file = os.path.join('Data',circuit,'Weather',filename)
            if not os.path.exists(file):
                print(f"Path '{file}' does not exists.")
                sys.exit(1)
            self.filename = filename[:-4]
            
        self.weather = []
        
        with open(file, 'r') as f:
            for line in f:
                self.weather.append(int(line.strip()))

        if len(self.weather)-1 != CIRCUIT[circuit]['Laps']:
            print(f"Weather file '{self.filename}' has {len(self.weather)} laps but circuit '{circuit}' has {CIRCUIT[circuit]['Laps']} laps.")

    def get_weather_string(self, w):
        if w < 20:
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
        return [self.get_weather_string(i) for i in self.weather]
