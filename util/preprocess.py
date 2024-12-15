import os
import pandas as pd


def get_all_sales_data(data_dir):
    sales_off = pd.read_csv(os.path.join(data_dir, "sales.csv"), delimiter=',')
    sales_on = pd.read_csv(os.path.join(data_dir, "online.csv"), delimiter=',')
    combined_sales = pd.concat([sales_off, sales_on], ignore_index=True)
    return combined_sales


def get_test_data(data_dir):
    return pd.read_csv(os.path.join(data_dir, "test.csv"), delimiter=';')


def add_days(df):
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.day
    return df


def add_average_quantity(df, attribute=None):
    if attribute:
        # Group by item_id and attribute, calculate the mean quantity for each combination
        average_quantity = df.groupby(['item_id', attribute], as_index=False)['quantity'].mean()
    else:
        average_quantity = df.groupby('item_id', as_index=False)['quantity'].mean()

    # Merge the calculated average_quantity back to the original dataframe
    average_quantity.rename(columns={'quantity': 'average_quantity'}, inplace=True)

    return average_quantity


def add_features(df):
    df = add_days(df)

    return df
