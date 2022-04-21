import sys, os
import pandas as pd
from classes.Timing import get_timing_data
from classes.Tyres import get_tyres_data
from classes.Fuel import get_fuel_data
from classes.Extractor import extract_data, remove_duplicates
from classes.Utils import get_basic_logger, list_data, separate_data, list_circuits

import argparse

parser = argparse.ArgumentParser(description='Process F1 Data.')
parser.add_argument('--i', type=int, default=19, help='Car ID')
parser.add_argument('--d', type=str, default='Data', help='Data folder')
parser.add_argument('--c', type=str, help='Circuit path')
parser.add_argument('--f', type=str, help='Exact data folder')
args = parser.parse_args()

log = get_basic_logger('MAIN')


def main(car_id:int=19,data_folder:str='Data',circuit:str='',folder:str=''):
    """
    Main wrapper, takes the folder where the csv's are stores and car id as input and runs the whole process.
    """
    ### Getting data from the 'folder'/'circuit' path
    log.info(f"Getting data for car '{car_id}'...")
    if folder == '' or folder is None:
        ### There is no specific folder of data => we use all the data in a given circuit
        if circuit == '' or circuit is None:
            ### There is no specific circuit => We have to get it from the user
            circuit = list_circuits(data_folder) # Returns the path to the circuit folder
        
        folder = list_data(circuit) # Returns the path to the data folder
    
    concat_path = os.path.join(folder,'ConcatData')
    if not os.path.exists(concat_path):
        os.makedirs(concat_path)
    
    if os.path.isfile(os.path.join(concat_path,'CarId_{}.csv'.format(car_id))):
        log.info(f"Found concatenated data for car '{car_id}' in '{concat_path}'")
        df = pd.read_csv(os.path.join(concat_path,'CarId_{}.csv'.format(car_id)))
    else:
        acquired_data_folder = os.path.join(folder,'Acquired_data')
        log.info(f"No existing concatenated data found. Concatenating data for car '{car_id}'...")
        
        ### This function removes duplicates of the dataframe and returns the dataframe with the unique rows (based on 'FrameIdentifier')
        remove_duplicates(acquired_data_folder) 

        damage, history, lap, motion, session, setup, setup, telemetry = extract_data(path=acquired_data_folder, idx=car_id)

        ### Creating a single dataframe with all the data
        ### In order to concatenate all data in a single dataframe (which is more easier to deal with) we need to set the FrameIdentifier (which is unique) as index
        damage.set_index('FrameIdentifier',inplace=True)
        history.set_index('FrameIdentifier',inplace=True)
        lap.set_index('FrameIdentifier',inplace=True)
        motion.set_index('FrameIdentifier',inplace=True)
        session.set_index('FrameIdentifier',inplace=True)
        setup.set_index('FrameIdentifier',inplace=True)
        telemetry.set_index('FrameIdentifier',inplace=True)

        log.info("Concatenating data...")
        df = pd.concat([damage, history, lap, motion, session, setup, setup, telemetry], axis=1)
        log.info("Saving concatenated data...")
        df.drop(columns=['CarIndex'],inplace=True) # CarIndex is not needed anymore because it is in the file name
        df = df.loc[:,~df.columns.duplicated()] #Remove duplicated columns  
        df.sort_index(inplace=True) #Sort the dataframe by the index (in this case FrameIdentifier)
        df.reset_index(inplace=True) #Reset the index to 0,1,2,3... instead of FrameIdentifier
        df.to_csv(os.path.join(concat_path,'CarId_{}.csv'.format(car_id)),index=False) #Save the dataframe as a csv file in order to have it for future use

        log.info(f"Complete unification of data for car '{car_id}' and saved it as 'ConcatData_Car{car_id}.csv'.")
    
    saves = os.path.join(folder,f'Saves\{car_id}')
    if not os.path.exists(saves):
        os.makedirs(saves)
    ### Separating the dataframe into different dataframes
    log.info(f"Separating data for car '{car_id}'...")
    separators = separate_data(df)
    log.info(f"Separation complete.")

    ### Getting the tyres data
    log.info(f"Getting the data for the times ({len(separators.keys())})...")
    timing_data = get_timing_data(df, separators=separators,path=saves)
    log.info(f"Complete getting the data for the times.")

    log.info(f"Getting all the tyres used ({len(separators.keys())})...")
    tyres_data = get_tyres_data(df, separators=separators,path=saves)
    log.info(f"Complete getting all the tyres used.")

    log.info(f"Getting the data for the fuel consumption ({len(separators.keys())})...")
    fuel_data = get_fuel_data(df, separators=separators,path=saves)
    log.info(f"Complete getting the data for the fuel consumption.")


    ### Plotting the data
    for idx, times in timing_data:
        times.plot(True)

    for idx, tyres in tyres_data:
        tyres.wear(True)

    for idx, fuel in fuel_data:
        fuel.consumption(True)
    
if __name__ == "__main__":
    main(args.i,args.d,args.c,args.f)
    sys.exit(0)
    