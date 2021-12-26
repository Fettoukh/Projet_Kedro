import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split


def remove_Empty_Columns(data: pd.DataFrame) -> pd.DataFrame:
    missing_rate = data.isna().sum()/data.shape[0]

    blood_columns = list(data.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
    viral_columns = list(data.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])

    key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

    df = data[key_columns + blood_columns + viral_columns]

    return df


def encodage(data: pd.DataFrame) -> pd.DataFrame:
    code = {'negative': 0,
            'positive': 1,
            'not_detected': 0,
            'detected': 1}
    for col in data.select_dtypes('object').columns:
        data.loc[:, col] = data[col].map(code)

    return data


def feature_engineering(data: pd.DataFrame, intitaldata: pd.DataFrame) -> pd.DataFrame:
    missing_rate = intitaldata.isna().sum()/intitaldata.shape[0]
    viral_columns = list(intitaldata.columns[(
        missing_rate < 0.80) & (missing_rate > 0.75)])

    data['est malade'] = data[viral_columns].sum(axis=1) >= 1
    df = data.drop(viral_columns, axis=1)
    return df

def imputation(df):
    df = df.dropna(axis=0)
    return  df

def data_split(data: pd.DataFrame) -> pd.DataFrame:
    trainset, testset = train_test_split(data, test_size=0.2, random_state=0)
    return trainset, testset
