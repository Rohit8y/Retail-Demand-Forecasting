import os
import pandas as pd
import numpy as np


def get_all_sales_data(data_dir):
    sales_off = pd.read_csv(os.path.join(data_dir, "sales.csv"), delimiter=",")
    sales_on = pd.read_csv(os.path.join(data_dir, "online.csv"), delimiter=",")
    sales_off["date"] = pd.to_datetime(sales_off["date"])
    sales_on["date"] = pd.to_datetime(sales_on["date"])
    sales_off = sales_off.drop(columns=["Unnamed: 0"])
    sales_on = sales_on.drop(columns=["Unnamed: 0"])
    sales_off["source"] = "offline"
    sales_on["source"] = "online"
    combined_sales = pd.concat([sales_off, sales_on], ignore_index=True)
    return combined_sales


def get_discount_history(data_dir):
    discount_history_data = pd.read_csv(os.path.join(data_dir, "discounts_history.csv"), delimiter=",")
    discount_history_data["date"] = pd.to_datetime(discount_history_data["date"])
    return discount_history_data


def get_test_data(data_dir):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), delimiter=";")
    test["date"] = pd.to_datetime(test["date"], format="%d.%m.%Y")
    test = test.drop(columns=["row_id"])
    return test


def add_discount_features(sales_df, discount_df):
    # Add a discount column to the discount DataFrame
    discount_df['discount'] = discount_df['sale_price_time_promo'] < discount_df['sale_price_before_promo']

    # Merge the two DataFrames
    merged_df = sales_df.merge(
        discount_df[['date', 'item_id', 'store_id', 'discount']],
        on=['date', 'item_id', 'store_id'],
        how='left'
    )

    # Fill missing values with False (assumes no discount if not found)
    merged_df['discount'] = merged_df['discount'].fillna(False)
    return merged_df


def add_date_time_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["year"] = df["date"].dt.year
    # df["day_of_month"] = df["date"].dt.day
    # df["day_of_year"] = df["date"].dt.dayofyear
    # df["week_of_year"] = df["date"].dt.isocalendar().week
    # df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # df["is_end_of_month"] = (df["date"] == df["date"] + pd.offsets.MonthEnd(0)).astype(int)
    return df


def add_class_subclass_features(data_dir, df):
    catalog = pd.read_csv(os.path.join(data_dir, "translated_catalog.csv"), delimiter=",")
    catalog = catalog.drop(
        columns=["weight_volume", "weight_netto", "fatness", "Unnamed: 0"]
    )
    merged_df = df.merge(catalog, on="item_id", how="left")
    unmatched = merged_df[merged_df["dept_name"].isna()]
    unmatched_count = unmatched["item_id"].nunique()
    print(f"Number of unmatched item_ids: {unmatched_count}")
    merged_df[["dept_name", "class_name", "subclass_name", "item_type"]] = merged_df[
        ["dept_name", "class_name", "subclass_name", "item_type"]
    ].apply(lambda col: col.fillna(col.mode()[0]))
    return merged_df


def add_average_quantity(df, attribute=None):
    if attribute:
        average_quantity = df.groupby(["item_id", attribute], as_index=False)[
            "quantity"
        ].mean()
    else:
        average_quantity = df.groupby("item_id", as_index=False)["quantity"].mean()

    average_quantity.rename(columns={"quantity": "average_quantity"}, inplace=True)

    return average_quantity


def add_features(data_dir, df):
    df = add_date_time_features(df)
    df = add_class_subclass_features(data_dir, df)
    return df
