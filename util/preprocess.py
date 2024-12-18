import os
import pandas as pd
import numpy as np


def get_all_sales_data(data_dir):
    sales_off = pd.read_csv(os.path.join(data_dir, "sales.csv"), delimiter=',')
    sales_on = pd.read_csv(os.path.join(data_dir, "online.csv"), delimiter=',')
    sales_off['date'] = pd.to_datetime(sales_off['date'])
    sales_on['date'] = pd.to_datetime(sales_on['date'])
    sales_off =sales_off.drop(columns=['Unnamed: 0'])
    sales_on =sales_on.drop(columns=['Unnamed: 0'])
    sales_off['source'] = 'offline'
    sales_on['source'] = 'online'
    combined_sales = pd.concat([sales_off, sales_on], ignore_index=True)

    return combined_sales


def get_test_data(data_dir):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), delimiter=';')
    test['date'] = pd.to_datetime(test['date'], format='%d.%m.%Y')
    test = test.drop(columns=['row_id'])
    return test

def add_date_time_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df['year'] = df["date"].dt.year
    df['day_of_month'] = df["date"].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_end_of_month'] = (df['date'] == df['date'] + pd.offsets.MonthEnd(0)).astype(int)
    return df


def add_average_quantity(df, attribute=None):
    if attribute:
        average_quantity = df.groupby(['item_id', attribute], as_index=False)['quantity'].mean()
    else:
        average_quantity = df.groupby('item_id', as_index=False)['quantity'].mean()

    average_quantity.rename(columns={'quantity': 'average_quantity'}, inplace=True)

    return average_quantity

def add_features(df):
    df = add_date_time_features(df)
    return df
