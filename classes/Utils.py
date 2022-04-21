import logging
import pandas as pd
from tqdm import tqdm
import math
import sys, os
from datetime import datetime

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

def get_car_name(id:int=19, path:str=None) -> str:
    data = pd.read_csv(os.path.join(path,'Acquired_data/Participant.csv')).loc[:,['CarIndex','Name']]
    data.drop_duplicates(subset=['CarIndex'], inplace=True)
    
    return data.loc[data['CarIndex']==id,'Name'].values.item()

def list_circuits(path:str='Data') -> str:
    """
    Function that takes a path and returns the list of circuits data in that folder.
    """
    log = get_basic_logger('MAIN')
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
    log = get_basic_logger('MAIN')
    folders = os.listdir(directory)
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    if len(folders) == 1:
        log.info(f"Only one folder ({folders[0]}) found in {directory}. Using it.")
        return os.path.join(directory,folders[0])

    print(f"Select the folder data to use:")
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
                    separator[count] = (sframe, int(tmp_df.loc[it,'FrameIdentifier']))
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
    
    ### Left here for debugging purposes
    #for key, values in separator.items():
    #    print(key, values)
    #
    #exit()

    return separator

if __name__ == "__main__":
    log = get_basic_logger('UTILS')
    log.warning("This module is not intended to be used as a standalone script. Run 'python main.py' instead.")
    sys.exit(1)



#Left here for plot debugging purposes (and maybe in the future can be useful)
"""
    import plotly.express as px
    from plotly.subplots import make_subplots

    import plotly

    for fpath in ['Data/Monza_Hard','Data/Monza_Soft_LongRun','Data/Monza_Soft_8_Laps']:
        df = pd.DataFrame()
        history = pd.read_csv(os.path.join(fpath,'History.csv'),low_memory=False)
        status = pd.read_csv(os.path.join(fpath,'Status.csv'),low_memory=False)
        damage = pd.read_csv(os.path.join(fpath,'Damage.csv'),low_memory=False)
        history = history.loc[history['CarIndex'] == 19]
        status = status.loc[status['CarIndex'] == 19]
        damage = damage.loc[damage['CarIndex'] == 19]
        best = 100000

        #print(status.columns.to_list())
        #print(status['FuelInTank'].values)
        #exit()

        for idx,col in enumerate(history.filter(like='lapTimeInMS').columns):
            laptime = max(history[col].values)
            if laptime == '-' or int(laptime) == 0:
                #print(f"At index {idx} we have a laptime of {laptime}")
                break
            laptime = int(laptime)
            if laptime < best:
                best = laptime
            df.at[idx,'LapTimeInMS'] = laptime
            df.at[idx,'LapNum'] = idx+1

        for idx, value in enumerate(df['LapTimeInMS'].values):
            #print(f"Lap {idx+1} - LapTimeInMS: {value} which is {value-best} of the best")
            df.at[idx,'Delta'] = value-best

        #df.to_csv('Test.csv')
        best = datetime.fromtimestamp(best/1000.0).strftime('%M:%S:%f')[:-3]
        
        fig2 = px.line(df, x='LapNum', y='Delta')
        fig1 = px.line(df, x='LapNum', y='LapTimeInMS')
        fig4 = px.line(status, x='FrameIdentifier', y='FuelInTank')
        fig3 = px.line(damage, x='FrameIdentifier', y=['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR'])

        fig = make_subplots(rows=2, cols=2, subplot_titles=(f'Laptimes',f'Delta wrt {best}','TyresWear','FuelInTank'))

        fig1_traces = []
        fig2_traces = []
        fig3_traces = []
        fig4_traces = []

        for trace in range(len(fig1["data"])):
            fig1_traces.append(fig1["data"][trace])

        for trace in range(len(fig2["data"])):
            fig2_traces.append(fig2["data"][trace])

        for trace in range(len(fig3["data"])):
            fig3_traces.append(fig3["data"][trace])
        
        for trace in range(len(fig4["data"])):
            fig4_traces.append(fig4["data"][trace])

        for traces in fig1_traces:
            fig.append_trace(traces, row=1, col=1)
        for traces in fig2_traces:
            fig.append_trace(traces, row=1, col=2)
        for traces in fig3_traces:
            fig.append_trace(traces, row=2, col=1)
        for traces in fig4_traces:
            fig.append_trace(traces, row=2, col=2)

        
        #fig.show()
        newpath = fpath.replace('Data/Monza','Tyres')
        plotly.offline.plot(fig, filename=f'{newpath}.html')

    exit()
    """
    