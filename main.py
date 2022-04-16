import sys, os
import pandas as pd
from classes.Tyres import get_tyres_data
from classes.Fuel import get_fuel_data
from classes.Extractor import extract_data, remove_duplicates
from classes.Utils import get_basic_logger, list_data, separate_data, list_circuits
import plotly.express as px

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
    
    if os.path.isfile(f'{folder}/ConcatData_Car{car_id}.csv'):
        log.info(f"Found concatenated data for car '{car_id}' in '{folder}'")
        df = pd.read_csv(f'{folder}/ConcatData_Car{car_id}.csv')
    else:
        acquired_data_folder = os.path.join(folder,'Acquired_data')
        log.info(f"No existing concatenated data found. Concatenating data for car '{car_id}'...")
        
        ### This function removes duplicates of the dataframe and returns the dataframe with the unique rows (based on 'FrameIdentifier')
        remove_duplicates(acquired_data_folder) 

        damage, history, lap, motion, session, setup, setup, telemetry = extract_data(path=acquired_data_folder)

        ### Creating a single dataframe with all the data
        ### In order to concatenate all data in a single dataframe (which is more easier to deal with) we need to set the FrameIdentifier (which is unique) as index
        damage.set_index('FrameIdentifier',inplace=True)
        history.set_index('FrameIdentifier',inplace=True)
        lap.set_index('FrameIdentifier',inplace=True)
        motion.set_index('FrameIdentifier',inplace=True)
        session.set_index('FrameIdentifier',inplace=True)
        setup.set_index('FrameIdentifier',inplace=True)
        telemetry.set_index('FrameIdentifier',inplace=True)

        df = pd.concat([damage, history, lap, motion, session, setup, setup, telemetry], axis=1)
        df.drop(columns=['CarIndex'],inplace=True) # CarIndex is not needed anymore because it is in the file name
        df = df.loc[:,~df.columns.duplicated()] #Remove duplicated columns  
        df.sort_index(inplace=True) #Sort the dataframe by the index (in this case FrameIdentifier)
        df.reset_index(inplace=True) #Reset the index to 0,1,2,3... instead of FrameIdentifier
        df.to_csv(f'{folder}/ConcatData_Car{car_id}.csv',index=False) #Save the dataframe as a csv file in order to have it for future use

        log.info(f"Complete unification of data for car '{car_id}' and saved it as 'ConcatData_Car{car_id}.csv'.")
    
    saves = os.path.join(folder,'Saves')
    ### Separating the dataframe into different dataframes
    log.info(f"Separating data for car '{car_id}'...")
    separators = separate_data(df)
    log.info(f"Separation complete.")

    ### Getting the tyres data
    log.info(f"Getting all the tyres used ({len(separators.keys())})...")
    tyres_data = get_tyres_data(df, separators=separators,path=saves)
    log.info(f"Complete getting all the tyres used.")

    log.info(f"Getting the data for the fuel consumption ({len(separators.keys())})...")
    fuel_data = get_fuel_data(df, separators=separators,path=saves)
    log.info(f"Complete getting the data for the fuel consumption.")

    ### Plotting the data
    for idx, tyre in tyres_data:
        tyre.wear(True)

    for idx, fuel in fuel_data:
        fuel.consumption(True)
    
if __name__ == "__main__":
    main(args.i,args.d,args.c,args.f)
    sys.exit(0)
    
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
    