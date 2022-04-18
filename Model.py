import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import re

#Load players data
data = pd.read_csv('player-value-prediction.csv')
#checking if data read correctly
#print(data.head())
#Checking for missing values
#print(data.isna().sum())
#Pre-processing

##Converting Date String to Date to apply Processes on it
#data['birth_date']=pd.to_datetime(data['birth_date'])
##Changing Date format to days wh(ich could be represented as one meaningful number
#now = datetime.now()
#data['birth_date']=now - data['birth_date'] #subtracting current date from it to get age in days

##splitting position column into separate columns
data[['w','x','y','z']]= data['positions'].str.split(',',expand=True)
##Replacing 'None' entries with 0's
data['w'].fillna(0 ,inplace=True)
data['x'].fillna(0 ,inplace=True)
data['y'].fillna(0 ,inplace=True)
data['z'].fillna(0 ,inplace=True)
##Replacing nominal entries with 1's
data.loc[data['w'] != 0,'w']=1
data.loc[data['x'] != 0,'x']=1
data.loc[data['y'] != 0,'y']=1
data.loc[data['z'] != 0,'z']=1
##Merging the three tables values into position column and droping temporary columns
data['positions']=data['w']+data['x']+data['y']+data['z']
data.drop(columns=['w','x','y','z'],inplace=True,axis=1)
##Droping near empty columns
data.drop(columns=['national_team','national_rating','national_team_position','national_jersey_number'],inplace=True,axis=1)
##Droping rows with missing values
data.dropna(inplace=True,axis=0)
print(data.isna().any())

#Features
X=data.iloc[:,4:91]
print(X.head())
print(X['positions'])

#label
Y=data['value']