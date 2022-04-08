import logging
import pandas as pd
from tqdm import tqdm

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

def separate_data(df:pd.DataFrame):
    separator = dict()
    count = 0

    df = df[df['DriverStatus'] != 0]
    start = df['FrameIdentifier'].values[0]
    for it in tqdm(range(start+1, len(df))):
        before = df['FrameIdentifier'].values[it-1]
        if before+1 == df['FrameIdentifier'].values[it]:
            try:
                separator[count].append(before+1)
            except KeyError:
                separator[count] = [before+1]
        else:
            count += 1

    #for key, values in separator.items():
    #    print(key, values[0], values[1], values[-1])
    #
    #exit()

    return separator
