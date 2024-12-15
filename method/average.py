import pandas as pd

from skeleton import Process
from tqdm import tqdm

from util.preprocess import add_average_quantity


class Average(Process):
    def __init__(self, all_data, store_data, test_data, output_file):
        super().__init__(all_data, store_data, test_data, output_file)
        self.store_avg_dfs = None

    def run(self):
        self.store_avg_dfs = []

        for store_name, store_df in self.store_data.items():
            # Apply the add_average_quantity function to the store DataFrame
            processed_df = add_average_quantity(store_df, attribute='weekday')  # Example attribute
            self.store_avg_dfs.append(processed_df)

    def generate_submission(self):
        generate_submission_weekday_smart(self.test_data, self.store_avg_dfs, self.output_file)


def generate_submission_weekday_smart(test_df, store_avg_dfs, output_file='submission_avg.csv'):
    """
    Generate weekday-based sales predictions for submission.

    Args:
        test_df (pd.DataFrame): Test data with columns ['store_id', 'item_id', 'weekday', 'row_id'].
        store_avg_dfs (list): List of DataFrames containing average quantities by item and weekday for each store.
        store_avg_base_dfs (list): List of DataFrames containing base averages for each store.
        output_file (str): File path to save the output CSV.

    Returns:
        int: Count of rows where no matching data was found.
    """
    not_found_count = 0

    # Map store_id to their respective average DataFrames
    store_avg_map = {i + 1: store_avg_dfs[i] for i in range(len(store_avg_dfs))}

    # Pre-index store average DataFrames for efficient lookups
    for store_id, avg_df in store_avg_map.items():
        avg_df.set_index(['item_id', 'weekday'], inplace=True)

    # Prepare results list
    results = []

    # Iterate through the test DataFrame
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing test data"):
        store_id = row['store_id']
        item_id = row['item_id']
        weekday = row['weekday']
        row_id = row['row_id']

        avg_quantity = 0  # Default quantity when no match is found

        # Retrieve the store's average DataFrame
        store_avg_df = store_avg_map.get(store_id, None)

        if store_avg_df is not None:
            # Check if the (item_id, weekday) combination exists in the indexed DataFrame
            try:
                avg_quantity = store_avg_df.loc[(item_id, weekday), 'average_quantity']
            except KeyError:
                # Log missing item for debugging
                not_found_count += 1
        else:
            raise Exception(f"Store ID {store_id} not found in store average map")

        # Add result
        results.append({'row_id': row_id, 'quantity': avg_quantity})

    # Convert results to a DataFrame
    submission_df = pd.DataFrame(results)

    # Save submission as a CSV file
    submission_df.to_csv(output_file, index=False)

    return not_found_count
