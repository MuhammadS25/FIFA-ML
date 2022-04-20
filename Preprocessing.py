import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def pre_processing(data):
    # Pre-processing
    # Dropping near empty columns
    droped_columns = []
    maxMissingValues = 7000
    for column in data:
        if data[column].isna().sum() > maxMissingValues:
            droped_columns.append(column)
    data.drop(columns=droped_columns, inplace=True, axis=1)

    # Droping rows with missing values
    data.dropna(inplace=True, axis=0)

    # splitting position column into separate columns
    newColumns = ['w', 'x', 'y', 'z']
    data[newColumns] = data['positions'].str.split(',', expand=True)

    # Replacing 'None' entries with 0's
    for col in newColumns:
        data[col].fillna(0, inplace=True)
        # Replacing nominal entries with 1's
        data.loc[data[col] != 0, col] = 1

    # Merging the three tables values into position column and droping temporary columns
    data['positions'] = data['w'] + data['x'] + data['y'] + data['z']
    data['positions'] = data['positions'].astype(np.int64)
    data.drop(columns=newColumns, inplace=True, axis=1)

    # splitting work_rate column to Attacking_Work_Rate and Defensive Work Rate
    data[['Attacking Work Rate', 'Defensive Work Rate']] = data['work_rate'].str.split('/', expand=True)
    data.drop(columns=['work_rate'], inplace=True, axis=1)
    data['Attacking Work Rate'] = data['Attacking Work Rate'].replace({"Low": 1, "Medium": 2, "High": 3})
    data['Defensive Work Rate'] = data['Defensive Work Rate'].replace({" Low": 1, " Medium": 2, " High": 3})
    valueCol = data['value']
    data.drop(columns=['value'], inplace=True, axis=1)
    data.insert(loc=len(data.columns), column='value', value=valueCol)

    # bodyType
    data['body_type'] = data['body_type'].replace(
        {"Stocky": 1, "Normal": 2, "Lean": 3, "Akinfenwa": 1, "Neymar": 3, "C. Ronaldo": 3, "PLAYER_BODY_TYPE_25": 3})

    # Encoding Nationality , club_team , preferred_foot , club_position columns
    lbl = LabelEncoder()
    lbl.fit_transform(list(data['nationality'].values))
    lbl.fit_transform(list(data['club_team'].values))
    lbl.fit_transform(list(data['preferred_foot'].values))
    lbl.fit_transform(list(data['club_position'].values))

    # Handling +2 in columns
    index_LS = data.columns.get_loc("LS")
    index_RB = data.columns.get_loc("RB")
    temp = data.iloc[:, index_LS:index_RB + 1]

    for col in temp:
        temp[col] = temp[col].str[0:2]
        temp[col] = temp[col].astype(np.int64)

    data.iloc[:, index_LS:index_RB + 1] = temp

    # using last 2 digits of year in club_join_date & contract_end_year
    data['club_join_date'] = data['club_join_date'].str[-2:]
    data['club_join_date'] = data['club_join_date'].astype(np.int64)
    data['contract_end_year'] = data['contract_end_year'].str[-2:]
    data['contract_end_year'] = data['contract_end_year'].astype(np.int64)

    return data
