import re

import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import pandas as pd

pd.set_option('display.max_columns', None)


def pre_processing(data,predict_column):
    # Pre-processing
    # Fill Missing Values(with data mean if number / data mode if category)
    for column in data.columns:
        if data.dtypes[column] == np.int64 or data.dtypes[column] == np.float64:
            data[column] = data[column].replace(np.NaN, data[column].mean())
        else:
            data[column] = data[column].replace(np.NaN, data[column].mode()[0])

    # Dynamically splitting position column into separate columns
    splittedPositions= data['positions'].str.split(',', expand=True)
    # Replacing 'None' entries with 0's
    for col in splittedPositions.columns:
        splittedPositions[col].fillna(0, inplace=True)
        # Replacing nominal entries with 1's
        splittedPositions.loc[splittedPositions[col] != 0, col] = 1
    # Adding the sums of splitted columns to positions columns
    data['positions'] = splittedPositions.sum(axis=1)



    # splitting work_rate column to Attacking_Work_Rate and Defensive Work Rate
    data[['Attacking Work Rate', 'Defensive Work Rate']] = data['work_rate'].str.split('/', expand=True)
    data.drop(columns=['work_rate'], inplace=True, axis=1)
    data['Attacking Work Rate'] = data['Attacking Work Rate'].replace({"Low": 1, "Medium": 2, "High": 3})
    data['Defensive Work Rate'] = data['Defensive Work Rate'].replace({" Low": 1, " Medium": 2, " High": 3})
    valueCol = data[predict_column]
    data.drop(columns=[predict_column], inplace=True, axis=1)
    data.insert(loc=len(data.columns), column=predict_column, value=valueCol)

    # bodyType
    data['body_type'] = data['body_type'].replace({"Stocky": 1, "Normal": 2, "Lean": 3})
    # For Strange Data
    data['body_type'] = data['body_type'].replace(r'^[^123]', 2, regex=True)

    # Encoding Nationality , club_team , preferred_foot , club_position columns
    lbl = LabelEncoder()
    data['nationality'] = lbl.fit_transform(list(data['nationality'].values))
    data['club_team'] = lbl.fit_transform(list(data['club_team'].values))
    data['preferred_foot'] = lbl.fit_transform(list(data['preferred_foot'].values))
    data['club_position'] = lbl.fit_transform(list(data['club_position'].values))

    # Handling +2 in columns
    index_LS = data.columns.get_loc("LS")
    index_RB = data.columns.get_loc("RB")
    temp = data.iloc[:, index_LS:index_RB + 1]

    for col in temp:
        temp[col] = temp[col].str[0:2]
        temp[col] = temp[col].astype(np.float64)

    data.iloc[:, index_LS:index_RB + 1] = temp

    # using last 2 digits of year in club_join_date & contract_end_year
    data['club_join_date'] =data['club_join_date'].astype(np.str)
    data['club_join_date'] = data['club_join_date'].str[-2:]
    data['club_join_date'] = data['club_join_date'].astype(np.float64)
    data['contract_end_year'] = data['contract_end_year'].astype(np.str)
    data['contract_end_year'] = data['contract_end_year'].str[-2:]
    data['contract_end_year'] = data['contract_end_year'].astype(np.float64)

    return data


def Correlation_Plotting(data,predict_column,saved_classification=False,saved_prediction=False):
    #Get the correlation between the features
    corr = data.corr()
    saved_pred_features=['overall_rating', 'potential', 'wage', 'international_reputation(1-5)','release_clause_euro', 'club_rating', 'reactions']
    saved_class_features=['overall_rating', 'potential', 'release_clause_euro', 'club_rating',
                        'short_passing', 'reactions', 'composure', 'LS', 'ST', 'RS', 'LW', 'LF',
                        'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM']
    top_features=None
    if saved_prediction:
        top_features=saved_pred_features
    elif saved_classification:
        top_features = saved_class_features
    else:
        # Top 50% Correlation training features with the Value
        top_features = corr.index[abs(corr[predict_column]) > 0.5]
        top_features = top_features.delete(-1)

    # Correlation Plotting
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    # scaling all the features between 0 and 1 values --> [Normalization]
    min_max_scaler = preprocessing.MinMaxScaler()
    data[top_features] = min_max_scaler.fit_transform(data[top_features])


    return data, top_features
