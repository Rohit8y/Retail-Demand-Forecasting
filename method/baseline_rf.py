import pandas as pd
from method.skeleton import Process
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np



class BaselineRandomForest(Process):
    def __init__(self, all_data, test_data, output_file):
        super().__init__(all_data, test_data, output_file)
        self.combined_sales = all_data
        self.test = test_data
        self.output_file = output_file
        print(self.combined_sales.shape)
        print(self.test.shape)

    def process_test_data(self):
        all_items = set(self.combined_sales["item_id"].to_list())
        test_items = set(self.test["item_id"].to_list())
        missing_items = test_items - all_items
        missing_indices= self.test[self.test["item_id"].isin(missing_items)].index
        valid_indices = self.test.index.difference(missing_indices)
        test_processed = self.test.drop(index=missing_indices)
        test_processed = test_processed.reset_index(drop=True)
        return test_processed,missing_indices,valid_indices

    def partition_data(self, df, test_df, features, target):
        train_data = df[df["date"] < "2024-06-01"]
        valid_data = df[df["date"] >= "2024-06-01"]

        X_train = train_data[features]
        y_train = train_data[target]
        X_valid = valid_data[features]
        y_valid = valid_data[target]
        X_test = test_df[features]

        return X_train, X_valid, X_test, y_train, y_valid

    def encode(self, train_series, target, test_series, n_splits=5, random_state=42):
        """
        Perform target encoding on a categorical feature.

        Parameters:
        - train_series: pd.Series, categorical feature from training data
        - target: pd.Series, target variable from training data
        - test_series: pd.Series, categorical feature from test data
        - n_splits: int, number of K-Fold splits
        - random_state: int, random seed

        Returns:
        - encoded_train: pd.Series, target-encoded training feature
        - encoded_test: pd.Series, target-encoded test feature
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        encoded_train = pd.Series(np.nan, index=train_series.index)

        for train_idx, val_idx in kf.split(train_series):
            means = target.iloc[train_idx].groupby(train_series.iloc[train_idx]).mean()
            encoded_train.iloc[val_idx] = train_series.iloc[val_idx].map(means)

        global_mean = target.mean()
        encoded_train.fillna(global_mean, inplace=True)

        means_full = target.groupby(train_series).mean()
        encoded_test = test_series.map(means_full)
        encoded_test.fillna(global_mean, inplace=True)

        return encoded_train, encoded_test

    def run(self):
        self.test_processed,self.missing,self.valid = self.process_test_data()
        self.combined_sales["item_id_enc"], self.test_processed["item_id_enc"] = self.encode(
            self.combined_sales["item_id"],
            self.combined_sales["quantity"],
            self.test_processed["item_id"],
        )
        exclude_cols = [
            "date",
            "item_id",
            "sum_total",
            "source",
            "quantity",
            "price_base",
        ]
        features = [
            col for col in self.combined_sales.columns if col not in exclude_cols
        ]
        target = "quantity"
        print("Selected Features:", features)
        X_train, X_valid, X_test, y_train, y_valid = self.partition_data(
            self.combined_sales, self.test_processed, features, target
        )
        rf_model = RandomForestRegressor(
            n_estimators=55,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            verbose=2,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train)
        preds_valid = rf_model.predict(X_valid)
        rmse = root_mean_squared_error(y_valid, preds_valid)
        print(f"Validation RMSE: {rmse}")

        self.predictions = rf_model.predict(X_test)

    def generate_submission(self):

        all_indices = np.arange(self.test.shape[0])
        pred_map = dict(zip(self.valid, self.predictions))

        final_quantities = []
        for i in all_indices:
            if i in pred_map:
                final_quantities.append(pred_map[i])
            else:
                final_quantities.append(0)

        submission_df = pd.DataFrame(
            {"row_id": all_indices, "quantity": final_quantities}
        )

        submission_df.to_csv('submission.csv', index=False)
