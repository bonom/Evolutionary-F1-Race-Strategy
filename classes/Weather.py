import os
import sys
import logging

from random import SystemRandom

from classes.Utils import CIRCUIT, get_basic_logger

random = SystemRandom()
logger = get_basic_logger(name='Weather', level=logging.INFO)

def weather_summary(circuit:str, weather_file:str):
    """
    Returns a string with the weather summary.
    """
    weather = Weather(circuit=circuit,filename=weather_file)

    wlist = weather.get_weather_list()

    to_ret = []
    for index, weather in enumerate(wlist):
        if index == 0 or weather != wlist[index-1]:
            to_ret.append(str(index+1)+"-"+str(weather))

    return to_ret

class Weather:
    def __init__(self, circuit:str, filename:str=None) -> None:
        
        handler = logging.StreamHandler()
        handler.terminator = ''
        logger.addHandler(handler)

        if filename is None:
            path = os.path.join('Data',circuit,'Weather')

            if not os.path.exists(path):
                raise FileExistsError(f"Path '{path}' does not exist.")

            files = os.listdir(path)

            if '.DS_Store' in files:
                files.remove('.DS_Store')

            if len(files) == 0:
                raise FileExistsError(f"No available weathers files in '{path}' for circuit '{circuit}'")

            logger.info(f"Available weathers for circuit '{circuit}': \n")
            for idx, w in enumerate(files):
                logger.info(f"{idx+1}. {w}\n")
            
            index = int(logger.info(f"\nSelect weather by number: "))
            file = os.path.join(path, files[index-1])

            self.filename = files[index-1]
        
        else:
            file = os.path.join('Data',circuit,'Weather',filename)
            if not os.path.exists(file):
                raise ValueError(f"Path '{file}' does not exists.")
            self.filename = filename[:-4]
            
        self.weather = []
        
        with open(file, 'r') as f:
            for line in f:
                self.weather.append(int(line.strip()))

        if len(self.weather)-1 != CIRCUIT[circuit]['Laps']:
            raise ValueError(f"Weather file '{self.filename}' has {len(self.weather)} laps but circuit '{circuit}' has {CIRCUIT[circuit]['Laps']} laps!")


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
