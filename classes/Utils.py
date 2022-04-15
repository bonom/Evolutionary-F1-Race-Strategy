import logging
import pandas as pd
from tqdm import tqdm
import math
import os

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

def list_data(directory:str='Data'):
    """
    Function that takes a directory and returns the list of subfolders in that folder.
    """
    folders = os.listdir(directory)
    folders.remove('.DS_Store')
    print(f"Select the folder data to use:")
    for idx,folder in enumerate(folders):
        print(f" {idx} for {folder}")
    
    folder_id = int(input("Enter the folder id: "))
    while folder_id < 0 or folder_id >= len(folders):
        folder_id = int(input("Invalid input. Enter a valid folder id: "))
    
    folder = folders[folder_id]

    return "Data/{}".format(folder)

def get_basic_logger(logger_name="default"):
    """
    Returns a basic logger with the given name.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
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
    
    #for key, values in separator.items():
    #    print(key, values)
    #
    #exit()

    return separator
