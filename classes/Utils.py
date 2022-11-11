import sys, os, math
from datetime import datetime

CIRCUIT: dict = {
    'Monza': {'Laps': 53, 'PitStopTime':24000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Spielberg' : {'Laps': 71, 'PitStopTime':21000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Montreal' : {'Laps': 70, 'PitStopTime':24000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Portimao' : {'Laps': 66, 'PitStopTime':25000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Bahrain' : {'Laps': 55, 'PitStopTime':23000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Zandvoort' : {'Laps': 72, 'PitStopTime':21500, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Silverstone' : {'Laps': 52, 'PitStopTime':20000, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
    'Barcelona' : {'Laps': 66, 'PitStopTime':22500, 'Tyres':{'SoftNew': 0, 'SoftUsed': 2, 'MediumNew': 1, 'MediumUsed':1, 'HardNew': 1, 'HardUsed': 1}},
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

class Log():
    def __init__(self, path:str, values:dict):
        self.path = os.path.join(path, 'Log.log')

        if not os.path.exists(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))

        if os.path.exists(self.path):
            print(f"Log file already exists at {self.path}, do you want to overwrite? [Y/n]", end=" ")
            if input().lower() == 'y':
                os.remove(self.path)
            elif input().lower() == 'n':
                sys.exit(0)

        self.write(f"Log file of circuit {values['Circuit']} at {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        self.write(f"Population: {values['Population']}\nIterations: {values['Iterations']}\nMutation: {values['Mutation']}\nCrossover: {values['Crossover']}\nPitStopTime: {ms_to_time(values['PitStopTime'])}\n\n")


    def write(self, msg:str):
        with open(self.path, 'a') as f:
            f.write(msg)

def time_to_ms(string):
    values = string.split(':')
    seconds, milliseconds = values[-1].split('.')[0], values[-1].split('.')[1]

    if len(values) > 2:
        minutes = values[1]
        hours = values[0]
    else:
        minutes = values[0]
        hours = 0
    
    return int(hours)*3600000 + int(minutes)*60000 + int(seconds)*1000 + int(milliseconds)

def ms_to_time(ms):
    if math.isinf(ms):
        return "Inf"
    
    if ms < 0:
        return "- " + ms_to_time(-ms)

    if isinstance(ms, float):
        ms = int(ms)

    milliseconds = str(ms)[-3:]

    while len(milliseconds) < 3:
        milliseconds = "0" + milliseconds

    # if int(milliseconds) < 100:
    #     if int(milliseconds) < 10:
    #         milliseconds = '0' + milliseconds
    #     milliseconds = '0' + milliseconds

    ms = ms-int(milliseconds)
    seconds = int((ms/1000)%60)
    minutes = int((ms/(1000*60))%60)
    hours = int((ms/(1000*60*60))%24)

    if seconds < 10:
        seconds = f"0{int(seconds)}"
    if minutes < 10:
        minutes = f"0{int(minutes)}"
    if hours < 10 and hours > 0:
        hours = f"{int(hours)}"
    
    if int(hours) < 1:
        if int(minutes) < 1:
            if int(seconds) < 10:
                return f"{int(seconds)}.{milliseconds}"
            return f"{seconds}.{milliseconds}"
        elif int(minutes) < 10:
            return f"{int(minutes)}:{seconds}.{milliseconds}"
        return f"{minutes}:{seconds}.{milliseconds}"
    
    return f"{hours}:{minutes}:{seconds}.{milliseconds}"
