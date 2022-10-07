import pandas as pd
import numpy as np
import os
import pathlib
import sys
import torch

def load_data(path, recordingIds):
    df = []
    largest_objectId = 0
    n_objects_total = 0
    for filename in np.sort(os.listdir(path)): # To make more robust you could load according to the order in recordingId, but this requires parsing of filename
        if filename[-3:]=='npy':
            continue
        recId = int(filename.split('_')[-2])
        if ~np.isin(recId,recordingIds):
            continue
        tmp = pd.read_csv(path  / filename,delim_whitespace=True
                    ,names=['frame','originalObjectId','xCenter','yCenter','heading'])
        tmp['recordingId'] = recId
        originalObjectIds = tmp['originalObjectId'].unique()
        tmp['locationId'] = get_locationId(recId)
        tmp = tmp.sort_values(['recordingId','originalObjectId','frame'],axis=0) ########
        tmp['objectId'] = np.nan
        nextObjectId = largest_objectId + 1
        n_objects = len(originalObjectIds)
        objectIds = range(nextObjectId,nextObjectId+n_objects)
        for originalId, id in zip(originalObjectIds,objectIds):
            tmp.loc[tmp['originalObjectId'] ==originalId,'objectId'] = id
        largest_objectId = largest_objectId +n_objects
        n_objects_total += n_objects
        df.append(tmp)
    df = pd.concat(df)
    assert(np.all(df.groupby('objectId').count()==20))
    assert(n_objects_total==df['objectId'].iloc[-1])
    return df

def get_locationId(recordingId):
    if recordingId in range(7,18):
        return 1
    elif recordingId in range(18,30):
        return 2
    elif recordingId in range(30,33):
        return 3
    elif recordingId in range(7):
        return 4

def evaluate_GRIP(recordingIds_test):
    dfs = {}
    tmp = load_data(pathlib.Path('./trajectories_InD'),recordingIds_test)
    dfs['truth'] = tmp
    dfs['truth'] = tmp.groupby('objectId').apply(lambda g: g.iloc[8:])
    dfs['truth'] = dfs['truth'].droplevel('objectId')
    dfs['predictions']=pd.read_csv('prediction_result.txt'
                            ,names=['recordingId','frame','originalObjectId','locationId','objectId','xCenter','yCenter']
                            ,delim_whitespace=True)

    #print(dfs['truth'])
    #print(dfs['predictions'])
    df = dfs['predictions']
    df['errors'] = np.linalg.norm(dfs['predictions'][['xCenter','yCenter']].to_numpy()-dfs['truth'][['xCenter','yCenter']].to_numpy(),axis=1)
    ADEs = df.groupby(['locationId','objectId'])\
            .apply(lambda g: np.average(g['errors']))\
            .mean(level='locationId')
    print('ADEs:')
    print(ADEs)
    FDEs = df.groupby(['locationId','objectId'])\
            .apply(lambda g: g['errors'].iloc[-1])\
                .mean(level='locationId')
    print('FDEs:')
    print(FDEs)

RECORDING_ID = {1:range(7,18)
                ,2:range(18,30)
                ,3:range(30,33)
                ,4:range(7)
                }
RECORDING_ID_TEST = {1:[14,15,16,17]
                ,2:[26,27,28,29]
                ,3:[32]
                ,4:[5,6]
                }
