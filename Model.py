import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


pd.set_option("display.max_rows",5, "display.max_columns", 5)
#Load players data
data = pd.read_csv('player-value-prediction.csv')
#checking if data read correctly
#print(data.head())
#print(data.isna().sum())
#Pre-processing
##Droping near empty columns
droped_columns=[]
maxMissingValues=7000
for column in data:
    if data[column].isna().sum()> maxMissingValues:
        droped_columns.append(column)
data.drop(columns=droped_columns,inplace=True,axis=1)


##Droping rows with missing values
data.dropna(inplace=True,axis=0)

##Converting Date String to Date to apply Processes on it
data['birth_date']=pd.to_datetime(data['birth_date'])

##Changing Date format to days wh(ich could be represented as one meaningful number
now = datetime.now()
data['birth_date']=now - data['birth_date'] #subtracting current date from it to get age in days

##splitting position column into separate columns
newColumns=['w','x','y','z']
data[newColumns]= data['positions'].str.split(',',expand=True)

##Replacing 'None' entries with 0's
for col in newColumns:
    data[col].fillna(0 ,inplace=True)
    ##Replacing nominal entries with 1's
    data.loc[data[col] != 0,col]=1

##Merging the three tables values into position column and droping temporary columns
data['positions']=data['w']+data['x']+data['y']+data['z']
data.drop(columns=newColumns,inplace=True,axis=1)

#label
##splitting work_rate column to Attacking_Work_Rate and Defensive Work Rate
data[['Attacking Work Rate','Defensive Work Rate']]=data['work_rate'].str.split('/',expand=True)
data.drop(columns=['work_rate'],inplace=True,axis=1)
data['Attacking Work Rate']=data['Attacking Work Rate'].replace({"Low":1, "Medium":2, "High":3})
data['Defensive Work Rate']=data['Defensive Work Rate'].replace({" Low":1, " Medium":2, " High":3})
valueCol=data['value']
data.drop(columns=['value'],inplace=True,axis=1)
data.insert(loc=len(data.columns),column='value', value=valueCol)
##bodyType
lbl = LabelEncoder()
lbl.fit(list(data['body_type'].values))
data['body_type'] = lbl.transform(list(data['body_type'].values))
lbl.fit(list(data['club_position'].values))
data['club_position'] = lbl.transform(list(data['club_position'].values))

#Features
X=data.iloc[:,4:-1]
print(X.head())
Y=data['value']
print(Y)