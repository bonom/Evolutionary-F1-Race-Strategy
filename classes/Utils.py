import logging
import pandas as pd
from tqdm import tqdm
import math

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

def get_basic_logger(logger_name="default"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def separate_data(df:pd.DataFrame) -> dict:
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
