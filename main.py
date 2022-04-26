import sys, os
import pandas as pd
import numpy as np
from classes.Timing import get_timing_data, Timing
from classes.Tyres import get_tyres_data, Tyres
from classes.Fuel import get_fuel_data, Fuel
from classes.Extractor import extract_data, remove_duplicates
from classes.Utils import MultiPlot, get_basic_logger, get_car_name, list_data, ms_to_m, separate_data, list_circuits

import plotly.express as px
from plotly.subplots import make_subplots
import plotly

from datetime import datetime 

import argparse

parser = argparse.ArgumentParser(description='Process F1 Data.')
parser.add_argument('--i', type=int, default=None, help='Car ID')
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
        #remove_duplicates(acquired_data_folder) 

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
    
    saves = os.path.join(folder,f'Saves/{car_id}')
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
    #fig = list()
    #fig_traces = {}
    #for idx, times in timing_data.items():
    #    times.plot(True)
    #
    #plotly.offline.plot(fig, filename='Plots/Timing.html')

    #for idx, tyres in tyres_data.items():
    #    tyres.wear(True)

    #for idx, fuel in fuel_data.items():
    #    fuel.consumption(True)

    ### Return data
    to_ret = {'Times':timing_data,'Tyres':tyres_data,'Fuel':fuel_data}

    tmp = pd.DataFrame(columns=['Lap','Delta','Wear_FL','Wear_FR','Wear_RL','Wear_RR','Fuel'])

    for key,value in timing_data.items():
        best = min([x for x in value.LapTimes if x > 0])
        for idx, delta in enumerate(value.LapTimes): #Deltas
            delta = delta - best
            
            ### Get Frame to use indexing (__getitem__)
            frame = value.get_frame(idx+1)
            
            ### Tyres
            tyres_wear = {'FL':0, 'FR':0, 'RL':0, 'RR':0}
            for tyre in ['FL', 'FR', 'RL', 'RR']:
                wear = tyres_data[key][frame][tyre+'Tyre']['TyreWear']
                tyres_wear[tyre] = wear
            
            fuel_consume = fuel_data[key][frame]['FuelInTank']
            
            tmp.loc[idx] = [idx,delta,tyres_wear['FL'],tyres_wear['FR'],tyres_wear['RL'],tyres_wear['RR'],fuel_consume]

            #log.debug(f"Lap: {idx}, Delta: {delta}, Wear: {tyres_wear}, Fuel: {fuel_consume}")
        
        if tmp.at[len(tmp)-1,'Delta'] < 0:
            tmp = tmp[:-1]

        x = tmp['Lap'].values
        y = tmp['Delta'].values
        #y = [np.log(y) if y > 0 else 0 for y in tmp['Delta'].values]

        coefficients = np.polyfit(x,y,3)
        
        poly = np.poly1d(coefficients)

        new_x = np.linspace(x[0], x[-1])
        new_y = poly(new_x)

        #fig = make_subplots(rows=4, cols=2)
        fig = MultiPlot(4,2,titles=['TimeDelta w.r.t '+ms_to_m(best), 'LapDeltaPolyFit', 'TyresWear', 'Fuel Consumption', 'Delta/Wear', 'Delta/Fuel', 'FuelPolyFit'])
        

        fig1 = px.line(tmp, x='Lap', y='Delta', title='Delta')
        fig2 = px.line(pd.DataFrame({'Lap':new_x, 'Delta':new_y}), x='Lap', y='Delta', title='Delta')
        fig3 = px.line(tmp, x='Lap', y=['Wear_FL','Wear_FR','Wear_RL','Wear_RR'], title='Tyres Wear')
        fig4 = px.line(tmp, x='Lap', y='Fuel', title='Fuel Consumption')

        tmp = tmp.sort_values(by='Delta')
        fig5 = px.line(tmp,x='Delta',y=['Wear_FL','Wear_FR','Wear_RL','Wear_RR'])
        fig6 = px.line(tmp,x='Delta',y='Fuel')

        x = tmp['Delta'].values
        y = tmp['Fuel'].values

        coefficients = np.polyfit(x, y, 1)
        poly = np.poly1d(coefficients)

        new_x = np.linspace(x[0], x[-1])
        new_y = poly(new_x)

        fig7 = px.line(pd.DataFrame({'Delta':new_x, 'Fuel':new_y}), x='Delta', y='Fuel', title='Fuel Consumption')

        fig.add_trace(fig1, row=1, col=1)
        fig.add_trace(fig2, row=1, col=2)
        fig.add_trace(fig3, row=2, col=1)
        fig.add_trace(fig4, row=2, col=2)
        fig.add_trace(fig5, row=3, col=1)
        fig.add_trace(fig6, row=3, col=2)
        fig.add_trace(fig7, row=4, col=1)

        if os.name == 'posix':
            path = folder.split('/')[1:]
        else:
            path = folder.split('\\')[1:]
        plots_path = os.path.join('Plots/',path[0],path[1])
        fig.set_title(f"Car_{car_id}")
        fig.save(os.path.join(plots_path,f'{car_id}.html'))
        fig.show()
        
    return to_ret
    
if __name__ == "__main__":
    if not os.path.exists('Plots'):
        os.mkdir('Plots')

    if args.i is not None:
        main(args.i,args.d,args.c,args.f)
        sys.exit(0)

    for i in range(0,20):
        if not os.path.exists('Plots'):
            os.mkdir('Plots')
        data = main(i,args.d,args.c,args.f)
    """
        max_value = max([len(data['Times'].keys()),len(data['Tyres'].keys()),len(data['Fuel'].keys())])
        if max_value == 0:
            path = args.f.split('/')
            path = os.path.join('Plots',path[2],path[3])
            if not os.path.exists(path):
                os.makedirs(path)
            open(f'{path}/Car{i}_NoData.html', mode='a').close()
        else:
            title = []
            times_title = []
            tyres_title = []
            fuel_title = []

            for key in data['Tyres'].keys():
                compound = data['Tyres'][key].get_visual_compound()
                times_title += [f'Times id {key}']
                tyres_title += [f'{compound} Compound Wear #{key}']
                fuel_title += [f'FuelInTank #{key}']

            fig = make_subplots(rows=3, cols=max_value, subplot_titles=times_title + tyres_title + fuel_title)
            for idx,(key, value) in enumerate(data['Times'].items()):
                times = value.plot(False)
                times = pd.DataFrame(times)
                tmp_fig = px.line(times, x='Lap', y='LapTimeInMS', title=f"#{key} - Time for car {idx}")
                traces = []
                for trace in range(len(tmp_fig["data"])):
                    traces.append(tmp_fig["data"][trace])

                for t in traces:
                    fig.append_trace(t, 1, idx+1)

            for idx,(key, value) in enumerate(data['Tyres'].items()):
                tyres = value.wear(False)
                tmp_fig = px.line(tyres, x='Lap', y=['Wear_FL', 'Wear_FR', 'Wear_RL', 'Wear_RR'], title=f"#{key} - Tyres wear for car {idx} on {value.get_visual_compound()}")
                traces = []
                for trace in range(len(tmp_fig["data"])):
                    traces.append(tmp_fig["data"][trace])

                for t in traces:
                    fig.append_trace(t, 2, idx+1)


            for idx,(key, value) in enumerate(data['Fuel'].items()):
                fuel = value.consumption(False)
                fuel = pd.DataFrame(fuel)
                tmp_fig = px.line(fuel, x='Lap', y='Fuel', title=f"#{key} - Fuel consumption for car {idx}")
                traces = []
                for trace in range(len(tmp_fig["data"])):
                    traces.append(tmp_fig["data"][trace])

                for t in traces:
                    fig.append_trace(t, 3, idx+1)

            if os.name == 'posix':
                path = folder.split('/')[1:]
            else:
                path = folder.split('\\')[1:]
            path = os.path.join('Plots',path[2],path[3])
            if not os.path.exists(path):
                os.makedirs(path)

            fig.update_layout(title_text=f"Car {i} -> {get_car_name(i,path=args.f)}")
            plotly.offline.plot(fig, filename=f'{path}/Car{i}.html')
"""
        
    sys.exit(0)
    