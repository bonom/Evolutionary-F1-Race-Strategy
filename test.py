import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import os
import matplotlib.pyplot as plt

FP1 = "D:\Projects\Bio-F1\Data\Austria\FP1\Acquired_data" # Hard/Medium
FP2 = "D:\Projects\Bio-F1\Data\Austria\FP2\Acquired_data" # Inter/Soft NO DRS
FP3 = "D:\Projects\Bio-F1\Data\Austria\FP3\Acquired_data" # Wet/Soft YES DRS

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

def searchForNearest(df, frameList):
    framesReturn = []
    for frame in frameList:
        if frame in df['FrameIdentifier'].values:
            framesReturn.append(frame)
        else:
            notFound = True
            add = 1
            while notFound:
                if (frame + add) in df['FrameIdentifier'].values:
                    framesReturn.append(frame + add)
                    notFound = False
                add += 1

    return framesReturn

def extract():
    #### FP1 ####
    toDrop = [8207,14288,37444,43260]

    lap = pd.read_csv(os.path.join(FP1, "Lap.csv"))
    lap = lap.loc[lap["CarIndex"] == 19, ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")
    lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    #to_drop = np.array(input(f"Lap DataFrame is the following:\n{lap}\nIf there are some wrong frames, insert now separated by comma: ").split(','), dtype=int)
    #to_drop = [int(i) for i in to_drop]
    #print(to_drop)

    for index in toDrop:
        if index in lap.index:
            lap = lap.drop(index)

    sub = min(lap['CurrentLapNum'])
    for i in lap.index:
        lap.at[i,'CurrentLapNum'] = int(lap.at[i,'CurrentLapNum'])-sub

    framesOfInterest = lap.index.values

    

    status = pd.read_csv(os.path.join(FP1, "Status.csv"))
    status = status.loc[status["CarIndex"] == 19, ['FrameIdentifier','FuelInTank','VisualTyreCompound','DRSAllowed','DRSActivationSystem']].drop_duplicates(['FrameIdentifier'], keep="last")
    status_frames = searchForNearest(status, framesOfInterest)
    status = status.loc[status['FrameIdentifier'].isin(status_frames), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")
    

    for i in status.index:
        status.at[i,'VisualTyreCompound'] = VISUAL_COMPOUNDS[status.at[i,'VisualTyreCompound']]

    lap.index = status_frames
    concatData = pd.concat([lap, status], axis=1)

    damage = pd.read_csv(os.path.join(FP1, "Damage.csv"))
    damage = damage.loc[damage["CarIndex"] == 19, ['FrameIdentifier', 'TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR',]].drop_duplicates(['FrameIdentifier'], keep="last")
    damageFramesOfInterest = searchForNearest(damage, framesOfInterest)
    damage = damage.loc[damage['FrameIdentifier'].isin(damageFramesOfInterest), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData.index = damageFramesOfInterest
    concatData = pd.concat([concatData, damage], axis=1)
    
    hard = concatData[concatData['VisualTyreCompound'] == 'Hard']
    medium = concatData[concatData['VisualTyreCompound'] == 'Medium']

    hardBestTime = min(hard['LastLapTimeInMS'].values)
    mediumBestTime = min(medium['LastLapTimeInMS'].values)

    for i in hard.index:
        hard.loc[i,'Delta'] = hard.loc[i,'LastLapTimeInMS'] - hardBestTime
        hard.at[i,'DRS'] = True

    for i in medium.index:
        medium.loc[i,'CurrentLapNum'] = int(medium.loc[i,'CurrentLapNum'] - 11)
        medium.loc[i,'Delta'] = int(medium.loc[i,'LastLapTimeInMS'] - mediumBestTime)
        medium.at[i,'DRS'] = True

    #### FP2 ####
    toDrop = [10,12731,12872]

    lap = pd.read_csv(os.path.join(FP2, "Lap.csv"))
    lap = lap.loc[lap["CarIndex"] == 19, ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")
    lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")


    for index in toDrop:
        if index in lap.index:
            lap = lap.drop(index)

    for i in lap.index:
        lap.at[i,'CurrentLapNum'] = int(lap.at[i,'CurrentLapNum'])-1

    framesOfInterest = lap.index.values

    telemetry = pd.read_csv(os.path.join(FP2, "Telemetry.csv"))
    telemetry = telemetry.loc[telemetry["FrameIdentifier"].isin(framesOfInterest), ['FrameIdentifier', 'CarIndex', 'DRS']]
    telemetry = telemetry.loc[telemetry['CarIndex'] == 19]

    status = pd.read_csv(os.path.join(FP2, "Status.csv"))
    status = status.loc[status["CarIndex"] == 19, ['FrameIdentifier','FuelInTank','VisualTyreCompound']].drop_duplicates(['FrameIdentifier'], keep="last")
    statusFramesOfInterest = searchForNearest(status, framesOfInterest)
    status = status.loc[status['FrameIdentifier'].isin(framesOfInterest), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    for i in status.index:
        status.at[i,'VisualTyreCompound'] = VISUAL_COMPOUNDS[status.at[i,'VisualTyreCompound']]

    lap.index = statusFramesOfInterest
    concatData = pd.concat([lap, status], axis=1)

    damage = pd.read_csv(os.path.join(FP2, "Damage.csv"))
    damage = damage.loc[damage["CarIndex"] == 19, ['FrameIdentifier', 'TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR',]].drop_duplicates(['FrameIdentifier'], keep="last")
    damageFramesOfInterest = searchForNearest(damage, framesOfInterest)
    damage = damage.loc[damage['FrameIdentifier'].isin(damageFramesOfInterest), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData.index = damageFramesOfInterest
    concatData = pd.concat([concatData, damage], axis=1)

    inter = concatData[concatData['VisualTyreCompound'] == 'Inter']
    soft = concatData[concatData['VisualTyreCompound'] == 'Soft']

    interBestTime = min(inter['LastLapTimeInMS'].values)
    softBestTime = min(soft['LastLapTimeInMS'].values)

    for i in inter.index:
        inter.at[i,'Delta'] = int(inter.at[i,'LastLapTimeInMS'] - interBestTime)
        inter.at[i,'DRS'] = False

    for i in soft.index:
        soft.at[i,'CurrentLapNum'] -= 10
        soft.at[i,'Delta'] = int(soft.at[i,'LastLapTimeInMS'] - softBestTime)
        soft.at[i,'DRS'] = False

    #### FP3 ####
    toDrop = [0,6838,6935,17777,32400,37341]

    lap = pd.read_csv(os.path.join(FP3, "Lap.csv"))
    lap = lap.loc[lap["CarIndex"] == 19, ['CurrentLapNum', 'FrameIdentifier', 'LastLapTimeInMS']].drop_duplicates(['FrameIdentifier'], keep="last")
    lap = lap.drop_duplicates(['CurrentLapNum','LastLapTimeInMS'], keep='first').sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")



    for index in toDrop:
        if index in lap.index:
            lap = lap.drop(index)

    for i in lap.index:
        lap.at[i,'CurrentLapNum'] = int(lap.at[i,'CurrentLapNum'])-1


    framesOfInterest = lap.index.values

    status = pd.read_csv(os.path.join(FP3, "Status.csv"))
    status = status.loc[status["CarIndex"] == 19, ['FrameIdentifier','FuelInTank','VisualTyreCompound']].drop_duplicates(['FrameIdentifier'], keep="last")
    statusFramesOfInterest = searchForNearest(status, framesOfInterest)
    status = status.loc[status['FrameIdentifier'].isin(framesOfInterest), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    for i in status.index:
        status.at[i,'VisualTyreCompound'] = VISUAL_COMPOUNDS[status.at[i,'VisualTyreCompound']]

    lap.index = statusFramesOfInterest
    concatData = pd.concat([lap, status], axis=1)

    damage = pd.read_csv(os.path.join(FP3, "Damage.csv"))
    damage = damage.loc[damage["CarIndex"] == 19, ['FrameIdentifier', 'TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR',]].drop_duplicates(['FrameIdentifier'], keep="last")
    damageFramesOfInterest = searchForNearest(damage, framesOfInterest)
    damage = damage.loc[damage['FrameIdentifier'].isin(damageFramesOfInterest), :].sort_values(by=['FrameIdentifier']).set_index("FrameIdentifier")

    concatData.index = damageFramesOfInterest
    concatData = pd.concat([concatData, damage], axis=1)

    wet = concatData[concatData['VisualTyreCompound'] == 'Wet']
    softDRS = concatData[concatData['VisualTyreCompound'] == 'Soft']

    wetBestTime = min(wet['LastLapTimeInMS'].values)
    softDRSBestTime = min(softDRS['LastLapTimeInMS'].values)

    for i in wet.index:
        wet.at[i,'Delta'] = int(wet.at[i,'LastLapTimeInMS'] - wetBestTime)
        wet.at[i,'DRS'] = False

    for i in softDRS.index:
        softDRS.at[i,'CurrentLapNum'] -= 10
        softDRS.at[i,'Delta'] = int(softDRS.at[i,'LastLapTimeInMS'] - softDRSBestTime)
        softDRS.at[i,'DRS'] = True




    concatAll = pd.concat([softDRS, soft, medium, hard, inter, wet, ], axis=0)
    
    concatAll.to_csv(os.path.join(FP3, "concatAll.csv"), index=False)
    print(concatAll)

data = extract()
data = pd.read_csv("D:\Projects\Bio-F1\Data\Austria\concatAll.csv")
datas = dict()
tyres = ['Soft', 'Medium', 'Hard', 'Inter', 'Wet']
for tyre in tyres:
    datas[tyre] = data[data['VisualTyreCompound'] == tyre]

# for data in datas.items():
#     print(data[0])
#     print(data[1])

#### DRS Difference:
softNoDRS = datas['Soft'][datas['Soft']['DRS'] == False]
softDRS = datas['Soft'][datas['Soft']['DRS'] == True]

timeDifferenceDRS = np.mean(softNoDRS['LastLapTimeInMS'].values - softDRS['LastLapTimeInMS'].values)
print(f"DRS makes difference for {timeDifferenceDRS} ms\n")


#### Tyres Difference:

bestSoftDRS = min(softDRS['LastLapTimeInMS'].values)
bestSoftNoDRS = min(softNoDRS['LastLapTimeInMS'].values)
bestMedium = min(datas['Medium']['LastLapTimeInMS'].values)
bestHard = min(datas['Hard']['LastLapTimeInMS'].values)
bestInter = min(datas['Inter']['LastLapTimeInMS'].values)
bestWet = min(datas['Wet']['LastLapTimeInMS'].values)

mediumDiff = bestMedium - bestSoftDRS
hardDiff = bestHard - bestSoftDRS
interDiff = bestInter - bestSoftDRS
wetDiff = bestWet - bestSoftDRS

interDiffNoDRS = bestInter - bestSoftNoDRS
wetDiffNoDRS = bestWet - bestSoftNoDRS

print(f"Soft time = {bestSoftDRS} ms")
print(f"Medium time = {bestMedium} ms -> difference of {mediumDiff} ms")
print(f"Hard time = {bestHard} ms -> difference of {hardDiff} ms")
print(f"Inter time = {bestInter} ms -> difference of {interDiff} ms with DRS, withour: {interDiffNoDRS} ms")
print(f"Wet time = {bestWet} ms -> difference of {wetDiff} ms with DRS, withour: {wetDiffNoDRS} ms\n")



#### WearPrediction
### coeffs are coeff[0] = m and coeff[1] = q

### Soft:
softNoDRSWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}
softDRSWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

for fl, fr, rl, rr in softNoDRS[['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    softNoDRSWear['FL'].append(fl)
    softNoDRSWear['FR'].append(fr)
    softNoDRSWear['RL'].append(rl)
    softNoDRSWear['RR'].append(rr)

for fl, fr, rl, rr in softDRS[['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    softDRSWear['FL'].append(fl)
    softDRSWear['FR'].append(fr)
    softDRSWear['RL'].append(rl)
    softDRSWear['RR'].append(rr)

coeffNoDRS = {'FL': None, 'FR': None, 'RL': None, 'RR': None}
coeffDRS = {'FL': None, 'FR': None, 'RL': None, 'RR': None}

for key in coeffNoDRS.keys():
    coeffNoDRS[key] = np.polyfit(softNoDRS['CurrentLapNum'], softNoDRSWear[key], 1)
    coeffDRS[key] = np.polyfit(softDRS['CurrentLapNum'], softDRSWear[key], 1)

### Medium:
mediumWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

for fl, fr, rl, rr in datas['Medium'][['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    mediumWear['FL'].append(fl)
    mediumWear['FR'].append(fr)
    mediumWear['RL'].append(rl)
    mediumWear['RR'].append(rr)

coeffMedium = {'FL': None, 'FR': None, 'RL': None, 'RR': None}
for key in coeffMedium.keys():
    coeffMedium[key] = np.polyfit(datas['Medium']['CurrentLapNum'], mediumWear[key], 1)

### Hard:
hardWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

for fl, fr, rl, rr in datas['Hard'][['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    hardWear['FL'].append(fl)
    hardWear['FR'].append(fr)
    hardWear['RL'].append(rl)
    hardWear['RR'].append(rr)

coeffhard = {'FL': None, 'FR': None, 'RL': None, 'RR': None}
for key in coeffhard.keys():
    coeffhard[key] = np.polyfit(datas['Hard']['CurrentLapNum'], hardWear[key], 1)

### Inter:
interWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

for fl, fr, rl, rr in datas['Inter'][['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    interWear['FL'].append(fl)
    interWear['FR'].append(fr)
    interWear['RL'].append(rl)
    interWear['RR'].append(rr)

coeffInter = {'FL': None, 'FR': None, 'RL': None, 'RR': None}
for key in coeffInter.keys():
    coeffInter[key] = np.polyfit(datas['Inter']['CurrentLapNum'], interWear[key], 1)

### Wet:
wetWear = {'FL': [], 'FR': [], 'RL': [], 'RR': []}

for fl, fr, rl, rr in datas['Wet'][['TyresWearFL','TyresWearFR','TyresWearRL','TyresWearRR']].values:
    wetWear['FL'].append(fl)
    wetWear['FR'].append(fr)
    wetWear['RL'].append(rl)
    wetWear['RR'].append(rr)

coeffWet = {'FL': None, 'FR': None, 'RL': None, 'RR': None}
for key in coeffWet.keys():
    coeffWet[key] = np.polyfit(datas['Wet']['CurrentLapNum'], wetWear[key], 1)


print(f"Soft no DRS Coefficients: {coeffNoDRS}")
print(f"Soft DRS Coefficients: {coeffDRS}")
print(f"Medium Coefficients: {coeffMedium}")
print(f"Hard Coefficients: {coeffhard}")
print(f"Inter Coefficients: {coeffInter}")
print(f"Wet Coefficients: {coeffWet}\n")


### Prediction

laps = 20

prediction = {'FL': coeffNoDRS['FL'][0]*laps + coeffNoDRS['FL'][1], 'FR': coeffNoDRS['FR'][0]*laps + coeffNoDRS['FR'][1], 'RL': coeffNoDRS['RL'][0]*laps + coeffNoDRS['RL'][1], 'RR': coeffNoDRS['RR'][0]*laps + coeffNoDRS['RR'][1]}
print(f"Prediction of Soft (No DRS) wear at {laps} laps: {prediction}")

prediction = {'FL': coeffDRS['FL'][0]*laps + coeffDRS['FL'][1], 'FR': coeffDRS['FR'][0]*laps + coeffDRS['FR'][1], 'RL': coeffDRS['RL'][0]*laps + coeffDRS['RL'][1], 'RR': coeffDRS['RR'][0]*laps + coeffDRS['RR'][1]}
print(f"Prediction of Soft (DRS) wear at {laps} laps: {prediction}")

prediction = {'FL': coeffMedium['FL'][0]*laps + coeffMedium['FL'][1], 'FR': coeffMedium['FR'][0]*laps + coeffMedium['FR'][1], 'RL': coeffMedium['RL'][0]*laps + coeffMedium['RL'][1], 'RR': coeffMedium['RR'][0]*laps + coeffMedium['RR'][1]}
print(f"Prediction of Medium wear at {laps} laps: {prediction}")

prediction = {'FL': coeffhard['FL'][0]*laps + coeffhard['FL'][1], 'FR': coeffhard['FR'][0]*laps + coeffhard['FR'][1], 'RL': coeffhard['RL'][0]*laps + coeffhard['RL'][1], 'RR': coeffhard['RR'][0]*laps + coeffhard['RR'][1]}
print(f"Prediction of Hard wear at {laps} laps: {prediction}")

prediction = {'FL': coeffInter['FL'][0]*laps + coeffInter['FL'][1], 'FR': coeffInter['FR'][0]*laps + coeffInter['FR'][1], 'RL': coeffInter['RL'][0]*laps + coeffInter['RL'][1], 'RR': coeffInter['RR'][0]*laps + coeffInter['RR'][1]}
print(f"Prediction of Inter wear at {laps} laps: {prediction}")

prediction = {'FL': coeffWet['FL'][0]*laps + coeffWet['FL'][1], 'FR': coeffWet['FR'][0]*laps + coeffWet['FR'][1], 'RL': coeffWet['RL'][0]*laps + coeffWet['RL'][1], 'RR': coeffWet['RR'][0]*laps + coeffWet['RR'][1]}
print(f"Prediction of Wet wear at {laps} laps: {prediction}\n")


### Time lose
softTimeLose = [x-bestSoftDRS for x in softDRS['LastLapTimeInMS'].values]
mediumTimeLose = [x-bestMedium for x in datas['Medium']['LastLapTimeInMS'].values]
hardTimeLose = [x-bestHard for x in datas['Hard']['LastLapTimeInMS'].values]
interTimeLose = [x-bestInter for x in datas['Inter']['LastLapTimeInMS'].values]
wetTimeLose = [x-bestWet for x in datas['Wet']['LastLapTimeInMS'].values]

def func(x, a, b):
    if isinstance(x, np.ndarray):
        return np.exp(a*x) * b
    return round(np.exp(a*x) * b)

alpha = 0.5
#coeffSoft, _ = curve_fit(func,  [0,10,15,20,35], [0, 400, 1000, 2500, 9200], maxfev=100000)
#coeffMedium, _ = curve_fit(func,  [0,10,15,20,35,45], [0, 600,900, 1500, 4600,9200], maxfev=100000)
#coeffHard, _ = curve_fit(func,  [0,10,15,20,50], [0, 700,1150, 1800, 9000], maxfev=100000)
coeffSoft, _ = curve_fit(func,  np.arange(1,len(softTimeLose)+1), softTimeLose, p0=(0.1, 270.0),  bounds=([0.075,250],[0.125,290]), sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(softTimeLose)], maxfev=1000)
coeffMedium, _ = curve_fit(func,  np.arange(1,len(mediumTimeLose)+1),  mediumTimeLose, p0=(0.075, 330.0), bounds=([0.05,305],[0.1,355]), sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(mediumTimeLose)], maxfev=1000)
coeffHard, _ = curve_fit(func,  np.arange(1,len(hardTimeLose)+1),  hardTimeLose, p0=(0.06, 455.0), bounds=([0.045,430],[0.085,480]), sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(hardTimeLose)], maxfev=1000)
coeffInter, _ = curve_fit(func,  np.arange(1,len(interTimeLose)+1),  interTimeLose, p0=(0.05, 800.0), bounds=([0.02,600],[1,1000]), sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(interTimeLose)], maxfev=1000)
coeffWet, _ = curve_fit(func,  np.arange(1,len(wetTimeLose)+1),  wetTimeLose, p0=(0.04, 800.0), bounds=([0.01,600],[1,1000]), sigma=[round(pow(alpha,i)*time)+1 for i,time in enumerate(wetTimeLose)], maxfev=1000)

lim_max = 51
plt.plot(np.arange(0,lim_max), func(np.arange(0,lim_max), *coeffSoft), 'r--', label='fit: a=%5.3f, b=%5.3f' % tuple(coeffSoft))#
plt.plot(np.arange(0,lim_max), func(np.arange(0,lim_max), *coeffMedium), 'g--', label='fit: a=%5.3f, b=%5.3f' % tuple(coeffMedium))
plt.plot(np.arange(0,lim_max), func(np.arange(0,lim_max), *coeffHard), 'b--', label='fit: a=%5.3f, b=%5.3f' % tuple(coeffHard))
plt.plot(np.arange(0,lim_max), func(np.arange(0,lim_max), *coeffInter), 'y--', label='fit: a=%5.3f, b=%5.3f' % tuple(coeffInter))
plt.plot(np.arange(0,lim_max), func(np.arange(0,lim_max), *coeffWet), 'c--', label='fit: a=%5.3f, b=%5.3f' % tuple(coeffWet))
plt.xlabel('Lap')
plt.ylabel('Delta')
plt.legend()
ax = plt.gca()
ax.set_xlim([-1, lim_max])
ax.set_ylim([-1, 8000])

laps = 20
print(f"Prediction of Soft (DRS) time lose at {laps} laps: {func(laps, *coeffSoft)}")
print(f"Prediction of Medium time lose at {laps} laps: {func(laps, *coeffMedium)}")
print(f"Prediction of Hard time lose at {laps} laps: {func(laps, *coeffHard)}")
print(f"Prediction of Inter time lose at {laps} laps: {func(laps, *coeffInter)}")
print(f"Prediction of Wet time lose at {laps} laps: {func(laps, *coeffWet)}\n")

plt.show()