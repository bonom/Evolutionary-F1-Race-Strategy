import pandas as pd
import numpy as np
from tqdm import tqdm

def fixer(df:pd.DataFrame, frame:int, col:str, before):
    if len(df.loc[df['FrameIdentifier'] == frame, col].to_numpy()) != 1:
        return before
    return df.loc[df['FrameIdentifier'] == frame, col].to_numpy()[0]

def unify_car_data(damage:pd.DataFrame,history:pd.DataFrame,lap:pd.DataFrame,motion:pd.DataFrame,session:pd.DataFrame,setup:pd.DataFrame,status:pd.DataFrame,telemetry:pd.DataFrame):
    damage_cols = damage.columns.to_numpy()
    history_cols = history.columns.to_numpy()
    lap_cols = lap.columns.to_numpy()
    motion_cols = motion.columns.to_numpy()
    session_cols = session.columns.to_numpy()
    setup_cols = setup.columns.to_numpy()
    status_cols = status.columns.to_numpy()
    telemetry_cols = telemetry.columns.to_numpy()
    
    columns = set()
    for i in damage_cols:
        if i not in columns:
            columns.add(i)
    for i in history_cols:
        if i not in columns:
            columns.add(i)
    for i in lap_cols:
        if i not in columns:
            columns.add(i)
    for i in motion_cols:
        if i not in columns:
            columns.add(i)
    for i in session_cols:
        if i not in columns:
            columns.add(i)
    for i in status_cols:
        if i not in columns:
            columns.add(i)
    for i in setup_cols:
        if i not in columns:
            columns.add(i)
    for i in telemetry_cols:
        if i not in columns:
            columns.add(i)
    
    
    add = {col:[] for col in columns}
    for i in tqdm(pHeader['FrameIdentifier'].to_numpy()):
        for col in columns:
            if col in head_cols:
                add[col].append(fixer(pHeader, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
            elif col in participant_cols:
                add[col].append(fixer(pParticipant, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
            elif col in lap_cols:   
                add[col].append(fixer(pLap, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
            elif col in telemetry_cols:
                add[col].append(fixer(pTelemetry, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
            elif col in status_cols:
                add[col].append(fixer(pStatus, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
            elif col in setup_cols:
                add[col].append(fixer(pSetup, i, col, add[col][-1] if len(add[col]) != 0 else np.nan))
     
    
    df = pd.DataFrame(columns=columns)
    for col in columns:
        df[col] = add[col]
    df.to_csv(f"Car_{car_index}_DATA.csv")

def extract_data(idx=19):
    """
    Extracts the very essentials data from the csv
    """
    max_frame = max(pd.read_csv('Data/Header.csv').replace('-', np.nan)['FrameIdentifier'])

    damage = pd.read_csv('Data/Damage.csv').replace('-', np.nan)
    damage = damage.loc[damage['CarIndex']==idx,['FrameIdentifier','CarIndex','TyresWearRL','TyresWearRR','TyresWearFL','TyresWearFR','TyresDamageRL','TyresDamageRR','TyresDamageFL','TyresDamageFR']]
    
    history = pd.read_csv('Data/History.csv').replace('-', np.nan)
    history = history.loc[history['CarIndex']==idx].drop(['PacketFormat','GameMajorVersion','GameMinorVersion','PacketVersion','PacketId','SessionUID','SessionTime','PlayerCarIndex','SecondaryPlayerCarIndex'], axis=1)
    for column in list(history.columns):
        if history[column].isnull().values.all():
            history.drop(column, axis=1, inplace=True)

    lap = pd.read_csv('Data/Lap.csv').replace('-', np.nan)
    lap = lap.loc[lap['CarIndex']==idx,['FrameIdentifier','CarIndex','LastLapTimeInMS','CurrentLapTimeInMS','Sector1TimeInMS','Sector2TimeInMS','LapDistance','TotalDistance','CurrentLapNum','Sector','PitStopShouldServePen']]

    lap_frames = dict()
    num_laps = lap['CurrentLapNum'].iloc[-1]
    for i in range(1,num_laps):
        frame_range = lap.loc[lap['CurrentLapNum']==i,'FrameIdentifier']
        if len(frame_range) > 0:
            last_frame = frame_range.iloc[-1]
            lap_frames[i] = last_frame
        else:
            lap_frames[i] = 0

    motion = pd.read_csv('Data/Motion.csv').replace('-', np.nan)
    motion = motion.loc[motion['CarIndex']==idx, ['FrameIdentifier','CarIndex','WorldPositionX','WorldPositionY','WorldPositionZ']]

    session = pd.read_csv('Data/Session.csv').replace('-', np.nan).drop(['PacketFormat','GameMajorVersion','GameMinorVersion','PacketVersion','PacketId','SessionUID','SessionTime','PlayerCarIndex','SecondaryPlayerCarIndex','PitSpeedLimit','GamePaused','IsSpectating','SpectatorCarIndex','SliProNativeSupport','NetworkGame','AIDifficulty','SeasonLinkIdentifier','WeekendLinkIdentifier','SessionLinkIdentifier','SteeringAssist','BrakingAssist','GearboxAssist','PitAssist','PitReleaseAssist','ERSAssist','DRSAssist','DynamicRacingLine','DynamicRacingLineType'], axis=1)
    for column in list(session.columns):
        if session[column].isnull().values.all():
            session.drop(column, axis=1, inplace=True)

    # The commented .drop() is for when we will manage the complete setup of the car
    # If uncommenting .drop() then uncomment the next line and comment the next one

    setup = pd.read_csv('Data/Setup.csv').replace('-', np.nan)#.drop('PacketFormat','GameMajorVersion','GameMinorVersion','PacketVersion','PacketId','SessionUID','SessionTime','PlayerCarIndex','SecondaryPlayerCarIndex',)
    #setup = setup.loc[setup['CarIndex']==idx]
    setup = setup.loc[setup['CarIndex']==idx, ['FrameIdentifier','CarIndex','RearLeftTyrePressure','RearRightTyrePressure','FrontLeftTyrePressure','FrontRightTyrePressure','FuelLoad']]
    
    status = pd.read_csv('Data/Status.csv').replace('-', np.nan)
    status = status.loc[status['CarIndex'] == idx, ['FrameIdentifier','CarIndex','FuelInTank','FuelCapacity','FuelRemainingLaps','ActualTyreCompound','VisualTyreCompound','TyresAgeLaps','VehicleFIAFlags','ERSStoreEnergy','ERSDeployMode','ERSHarvestedThisLapMGUK','ERSHarvestedThisLapMGUH','ERSDeployedThisLap']]
    
    telemetry = pd.read_csv('Data/Telemetry.csv').replace('-', np.nan)
    telemetry = telemetry.loc[telemetry['CarIndex'] == idx].drop(['PacketFormat','GameMajorVersion','GameMinorVersion','PacketVersion','PacketId','SessionUID','SessionTime','PlayerCarIndex','SecondaryPlayerCarIndex','Speed','Throttle','Steer','Brake','Clutch','Gear','EngineRPM','DRS','RevLightsPercent','RevLightsBitValue','RLSurfaceType','RRSurfaceType','FLSurfaceType','FRSurfaceType','MFD','MFDSecondaryPlayer','SuggestedGear'], axis=1, )
    
    min_frame = max(min(damage['FrameIdentifier']), min(history['FrameIdentifier']), min(lap['FrameIdentifier']), min(motion['FrameIdentifier']), min(session['FrameIdentifier']), min(setup['FrameIdentifier']), min(status['FrameIdentifier']), min(telemetry['FrameIdentifier']))

    for frame in range(min_frame, max_frame+1):
        tmp_damage = damage.loc[damage['FrameIdentifier']==frame]
        tmp_history = history.loc[history['FrameIdentifier']==frame]
        tmp_lap = lap.loc[lap['FrameIdentifier']==frame]
        tmp_motion = motion.loc[motion['FrameIdentifier']==frame]
        tmp_session = session.loc[session['FrameIdentifier']==frame]
        tmp_setup = setup.loc[setup['FrameIdentifier']==frame]
        tmp_status = status.loc[status['FrameIdentifier']==frame]
        tmp_telemetry = telemetry.loc[telemetry['FrameIdentifier']==frame]

        print(tmp_damage, tmp_history, tmp_lap, tmp_motion, tmp_session, tmp_setup, tmp_status, tmp_telemetry)
        
    
    

if __name__ == "__main__":
    extract_data()