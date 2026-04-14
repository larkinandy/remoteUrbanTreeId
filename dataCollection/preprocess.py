# preprocess.py
# Author: Andrew Larkin
# Date Created: April 13, 2026
# Summary: Preprocess records for deep learning
# Note: ChatGPT was used for code assist

# import libraries
import ee
from datetime import datetime, timedelta
import sys
import os
import pandas as ps
from dotenv import load_dotenv

# import custom classes and environments
GIT_PATH = "C:/Users/larki/Documents/GitHub/remoteUrbanTreeId/dataCollection/"
sys.path.append(GIT_PATH)

load_dotenv(dotenv_path=GIT_PATH + ".env")

# get list of weekly Sentinel-2 files to combine
# OUTPUTS:
#    df (pandas dataframe) - contains combined Sentinel-2 time series
def getFilesToCombine():
    individualFiles = os.listdir(os.getenv("GEE_LOCAL_FOLDER"))
    pandasArr = []
    for file in individualFiles:
        data = ps.read_csv(os.getenv("GEE_LOCAL_FOLDER")+file)
        pandasArr.append(data)
    df = ps.concat(pandasArr)
    df['date'] = ps.to_datetime(df['date'])
    print(df.head())
    return(df)

# interpolate Sentinel-2 bands when cloud cover results in missing values
# OUTPUTS:
#    df (pandas dataframe) - Sentinel-2 time series with interpolated values 
def interpolateSentinel():
    df = df.sort_values(['uniqueID', 'date']).set_index('date')
    cols = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8','B8A','B11','B12']
    df = (
        df.groupby('uniqueID').apply(lambda g: g.resample('5D').mean())
    )
    df[cols] = (
        df.groupby('uniqueID')[cols].apply(lambda g: g.interpolate(
            method='spline',
            limit=10,              # max gap size
            limit_area='inside'   # no extrapolation
        ))
    )
    df = df.reset_index()
    return(df)