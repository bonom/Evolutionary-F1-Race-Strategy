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
        data.tyres_slip(display=False)
        data.tyres_wear(display=False)
        data.tyres_timing(display=False)
    

if __name__ == "__main__":
    main()



    
    """
    import plotly.express as px
    #import plotly #if for offline save
    from plotly.subplots import make_subplots

    # HOW TO PLOT SUBPLOTS WITH PLOTLY EXPRESS

    fig = make_subplots(rows=3, cols=2)
    damage = pd.read_csv('Data/Damage.csv')
        
    fig1 = px.line(damage.loc[(damage['CarIndex']==19) & (damage['FrameIdentifier'] < 32450)],x="FrameIdentifier",y=['TyresWearRL','TyresWearRR','TyresWearFL','TyresWearFR'], range_y=[0,100])
    fig2 = px.line(damage.loc[(damage['CarIndex']==19) & (damage['FrameIdentifier'] < 32450)],x="FrameIdentifier",y=['TyresDamageRL','TyresDamageRR','TyresDamageFL','TyresDamageFR'], range_y=[0,100])

    motion = pd.read_csv('Data/Motion.csv')
    fig3 = px.line(motion.loc[(motion['CarIndex'] == 19) & (motion['FrameIdentifier'] < 32450)], y=['RLWheelSlip', 'RRWheelSlip', 'FLWheelSlip', 'FRWheelSlip'], x='FrameIdentifier', range_y=[-1.1,1.1])
    fig4 = px.line(motion.loc[(motion['CarIndex'] == 19) & (motion['FrameIdentifier'] < 32450)], y=['Roll'], x='FrameIdentifier')
    
    telemetry = pd.read_csv('Data/Telemetry.csv')
    fig5 = px.line(telemetry.loc[(telemetry['CarIndex'] == 19) & (telemetry['FrameIdentifier'] < 32450)], y=['RLTyreSurfaceTemperature', 'RRTyreSurfaceTemperature', 'FLTyreSurfaceTemperature', 'FRTyreSurfaceTemperature'], x='FrameIdentifier')
    fig6 = px.line(telemetry.loc[(telemetry['CarIndex'] == 19) & (telemetry['FrameIdentifier'] < 32450)], y=['RLTyreInnerTemperature', 'RRTyreInnerTemperature', 'FLTyreInnerTemperature', 'FRTyreInnerTemperature'], x='FrameIdentifier')

    fig1_traces = []
    fig2_traces = []
    fig3_traces = []
    fig4_traces = []
    fig5_traces = []
    fig6_traces = []

    for trace in range(len(fig1["data"])):
        fig1_traces.append(fig1["data"][trace])

    for trace in range(len(fig2["data"])):
        fig2_traces.append(fig2["data"][trace])
    
    for trace in range(len(fig3["data"])):
        fig3_traces.append(fig3["data"][trace])
    
    for trace in range(len(fig4["data"])):
        fig4_traces.append(fig4["data"][trace])
    
    for trace in range(len(fig5["data"])):
        fig5_traces.append(fig5["data"][trace])
    
    for trace in range(len(fig6["data"])):
        fig6_traces.append(fig6["data"][trace])
    
    for traces in fig1_traces:
        fig.append_trace(traces, row=1, col=1)
    for traces in fig2_traces:
        fig.append_trace(traces, row=1, col=2)
    for traces in fig3_traces:
        fig.append_trace(traces, row=2, col=1)
    for traces in fig4_traces:
        fig.append_trace(traces, row=2, col=2)
    for traces in fig5_traces:
        fig.append_trace(traces, row=3, col=1)
    for traces in fig6_traces:
        fig.append_trace(traces, row=3, col=2)

    
    fig.show()

    #plotly.offline.plot(fig, filename='tyres_data.html')
    """