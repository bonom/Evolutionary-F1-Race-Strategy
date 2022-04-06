from classes.Tyres import get_tyres_data
from classes.Extractor import extract_data,unify_car_data
from classes.Utils import get_basic_logger
import pandas as pd
import os

log = get_basic_logger('MAIN')

def main(car_id:int=19):
    if os.path.exists('Car_{}_data.csv'.format(car_id)):
        log.info('Car_{}_data.csv already exists, using it.'.format(car_id))
        df = pd.read_csv('Car_{}_data.csv'.format(car_id))        
    else:
        log.info('Car_{}_data.csv not found, building it.'.format(car_id))
        damage, history, lap, motion, session, setup, status, telemetry, min_frame, max_frame = extract_data()
        df = unify_car_data(car_id,damage, history, lap, motion, session, setup, status, telemetry, max_frame,min_frame)
    
    tyres_data = get_tyres_data(df)

    for idx, data in tyres_data:
        data.tyres_wear(display=True)
        data.tyres_timing(display=True)

if __name__ == "__main__":
    main()