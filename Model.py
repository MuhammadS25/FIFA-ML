import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#Load players data
data = pd.read_csv('player-value-prediction.csv')
#checking if data read correctly
print(data.head())
#Checking for missing values
print(data.isna().sum())
#Pre-processing

##Converting Date String to Date to apply Processes on it
#data['birth_date']=pd.to_datetime(data['birth_date'])
##Changing Date format to days which could be represented as one meaningful number
#now = datetime.now()
#data['birth_date']=now - data['birth_date'] #subtracting current date from it to get age in days

##splitting position column into separate columns

#Features
X=data.iloc[:,4:91]
print(X)
#label
Y=data['value']