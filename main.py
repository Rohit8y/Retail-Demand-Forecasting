import argparse
import logging

from method import Average
from method import BaselineRandomForest
from method import BaselineCatBoost
from util.preprocess import get_all_sales_data, add_features, get_test_data

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def prepare_data(data_path, apply_filters=True):
    logger.info("Starting data preparation...")
    raw_data = get_all_sales_data(data_path)
    logger.info("Raw data loaded successfully.")

    # Add features to the raw data
    all_data = add_features(raw_data)
    logger.info("Features added successfully.")

    # Apply filters
    if apply_filters:
        filtered_df = all_data[['day', 'weekday', 'item_id', 'quantity', 'store_id']]
        logger.info("Data filtering completed.")
        return filtered_df
    else:
        logger.info("Skipping Data filtering.")
        return all_data


def prepare_test_data(data_dir):
    test = get_test_data(data_dir)
    logger.info("Test data loaded successfully.")
    test_data = add_features(test)
    logger.info("Features added to test data")

    return test_data


# def get_store_df(df):
#     logger.info("Creating store-wise sales DataFrames...")
#     # Create a dictionary to hold sales data for each store
#     store_sales = {f"store_{store_id}_sales": df[df['store_id'] == store_id] for store_id in df['store_id'].unique()}
#     logger.info("Store-wise DataFrames created successfully.")
#     return store_sales


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Config for baseline")
    parser.add_argument('--data', default='data', type=str,
                        help='Path to the data directory')
    parser.add_argument('--method', default='avg', type=str,
                        help='Main method to use: avg|random_forest')
    parser.add_argument('--output', default=None, type=str,
                        help='Output path to save submission file')

    args = parser.parse_args()

    if args.output is None:
        raise Exception("Output path is None")

    logger.info("Starting script execution...")
    df = prepare_data(args.data,apply_filters=False)
    logger.info("Data preparation completed successfully.")

    test_data = prepare_test_data(data_dir=args.data)

    # init method
    logger.info("initialising method...")
    if args.method == 'avg':
        method = Average(df, test_data, args.output)
    elif args.method == 'rf_base':
        method = BaselineRandomForest(df, test_data, args.output)
    elif args.method == 'cb_base':
        method = BaselineCatBoost(df, test_data, args.output)
    else:
        logger.info(f"{args.method} not defined")
        raise Exception(f"{args.method} not defined")

    logger.info("Executing core method")
    method.run()

    logger.info("Generating submission file")
    method.generate_submission()
