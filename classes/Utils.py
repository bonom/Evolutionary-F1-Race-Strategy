import logging
from typing import Union
import pandas as pd
from tqdm import tqdm
import math
import sys, os
from datetime import datetime
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import socket

STINTS: dict = {
    0: 'Soft',
    1: 'Medium',
    2: 'Hard',
    3: 'Inter',
    4: 'Wet',

}

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

_xor = {("0", "0"): "0",
        ("0", "1"): "1",
        ("1", "0"): "1",
        ("1", "1"): "0"}

def tograystr(binary):
    result = prec = binary[0]
    for el in binary[1:]:
        result += _xor[el, prec]
        prec = el
    return result

def tobinarystr(gray):
    result = prec = gray[0]
    for el in gray[1:]:
        prec = _xor[prec, el]
        result += prec
    return result

def int_to_gray(n:int) -> str:
    """
    Convert an integer to a gray code string
    """
    return tograystr(bin(n)[2:])

def gray_to_int(n:str) -> int:
    """
    Convert a gray code string to an integer
    """
    return int(tobinarystr(n), 2)


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = datetime.now().strftime("%H:%M:%S")+" - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class MultiPlot:
    def __init__(self, rows:int=1,cols:int=1, titles:Union[np.array,list]=[]):
        self.x = rows
        self.y = cols
        self.count = 0
        self.fig = make_subplots(rows=rows, cols=cols,subplot_titles=titles)
        
    def add_trace(self, fig=None, row:int=1, col:int=1, traces=None):
        if traces is None:
            traces = []
            for trace in range(len(fig["data"])):
                traces.append(fig["data"][trace])

        for trace in traces:
            self.fig.add_trace(trace, row=row, col=col)
        
        self.count += 1
    
    def show(self, filename:str=None):
        if get_host() == 'DESKTOP-KICFR1D':
            return plotly.offline.plot(self.fig, filename=filename)
        return self.fig.show()
    
    def save(self, filename:str):
        self.fig.write_html(filename)

    def reset(self, rows:int = 1, cols:int = 1):
        backup = set()
        for data in self.fig.data:
            backup.add(data)

        self.x = rows
        self.y = cols
        self.fig = make_subplots(rows=self.x, cols=self.y)
        self.count = 0  

        for data in backup:
            self.add_trace(traces=data)  
    
    def set_title(self,title:str):
        self.fig.update_layout(title_text=title)


def ms_to_time(ms:int) -> str:
    date = datetime.fromtimestamp(ms/1000 - 3600) # - 3600 for UTC (I think)
    return date.strftime("%H:%M:%S.%f")[:-3]

def get_car_name(id:int=19, path:str=None) -> str:
    data_path = os.path.join(path,'ConcatData/Names.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df.loc[df['CarIndex']==id,'Name'].values.item()

    data_path = os.path.join(path,'Acquired_data/Participant.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path).loc[:,['CarIndex','Name']]
        df.drop_duplicates(subset=['CarIndex'], inplace=True)
        df.to_csv(os.path.join(path,'ConcatData/Names.csv'), index=False)
        return df.loc[df['CarIndex']==id,'Name'].values.item()
    
    return 'N/A'

def list_circuits(path:str='Data') -> str:
    """
    Function that takes a path and returns the list of circuits data in that folder.
    """
    log = get_basic_logger('UTILS')
    folders = os.listdir(path)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    if len(folders) == 1:
        log.info(f"Only one folder ({folders[0]}) found in {path}. Using it.")
        return os.path.join(path,folders[0])

    print(f"Select the folder (circuit) to use:")
    for idx,folder in enumerate(folders):
        print(f" {idx} for {folder}")
    
    folder_id = int(input("Enter the folder id: "))
    while folder_id < 0 or folder_id >= len(folders):
        folder_id = int(input("Invalid input. Folder selected must be between 0 and {} Enter a valid folder id: ".format(len(folders)-1)))
    
    folder = folders[folder_id]
    log.info('Using "{}".'.format(folder))

    return os.path.join(path,folder)

def list_data(directory:str) -> str:
    """
    Function that takes a directory and returns the list of subfolders in that folder.
    """
    log = get_basic_logger('UTILS')
    folders = os.listdir(directory)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    if len(folders) == 1:
        log.info(f"Only one folder ({folders[0]}) found in {directory}. Using it.")
        return os.path.join(directory,folders[0])

    print(f"Select the folder data to use from circuit '{directory}':")
    for idx,folder in enumerate(folders):
        print(f" {idx} for {folder}")
    
    folder_id = int(input("Enter the folder id: "))
    while folder_id < 0 or folder_id >= len(folders):
        folder_id = int(input("Invalid input. Folder selected must be between 0 and {} Enter a valid folder id: ".format(len(folders)-1)))
    
    folder = folders[folder_id]

    return os.path.join(directory,folder)

def get_basic_logger(logger_name="MAIN"):
    """
    Returns a basic logger with the given name.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    return logger

def get_host():
    """
    Returns the hostname of the machine.
    """
    return socket.gethostname()

def separate_data(df:pd.DataFrame) -> dict:
    """
    Separates the dataframe into different dataframes based on the column 'DriverStatus'. This column
    indicates the status of the driver (0 is in pit, >=1 on track). The idea is that every time we are 
    in pit we change tyres. With this information we take the frame where we exit the pit and the frame 
    we enter the pit and we use it as range for the laps we have done.

    (It can be not cleaer => I will explain better)
    """

    separator = dict()
    tmp_df = df.copy()

    start = int(tmp_df.index[0])
    count = 0    

    for it in tqdm(range(start+1, len(tmp_df))):
        value = tmp_df.loc[it,'DriverStatus']
        before = tmp_df.loc[it-1,'DriverStatus']

        if not math.isnan(value):
            if value > 0:
                if before == 0:
                    separator[count] = int(tmp_df.loc[it,'FrameIdentifier'])
            elif value == 0:
                if before > 0:
                    sframe = separator[count] 
                    separator[count] = (sframe, int(tmp_df.loc[it-1,'FrameIdentifier']))
                    count += 1
        else:
            tmp_df.at[it, 'DriverStatus'] = tmp_df.loc[it-1,'DriverStatus']

    try:
        separator[count]
        if isinstance(separator[count], int):
            sframe = separator[count] 
            separator[count] = (sframe, int(tmp_df.loc[it,'FrameIdentifier']))
    except KeyError:
        pass
    
    return separator

if __name__ == "__main__":
    log = get_basic_logger('Utils')
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)
