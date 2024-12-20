import pandas as pd
from method.skeleton import Process
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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
    
    def scale(self,X_train,X_valid,X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled,X_valid_scaled,X_test_scaled
    
    def label_encode(self, train_df, test_df):
        for col in ['dept_name', 'class_name', 'subclass_name', 'item_type']:
            label_encoder = LabelEncoder() 
            train_df[col] = label_encoder.fit_transform(train_df[col])
            test_labels = test_df[col].values 
            valid_mask = np.isin(test_labels, label_encoder.classes_) 
            encoded_labels = np.full_like(test_labels, fill_value=-1, dtype=np.int32) 
            encoded_labels[valid_mask] = label_encoder.transform(test_labels[valid_mask])  
            test_df[col] = encoded_labels
        return train_df, test_df

    def one_hot_encode(self, df, column):
        encoded_df = pd.get_dummies(df, columns=[column], prefix=[column])
        return encoded_df   

    def partition_data(self, df, test_df, features, target):
        train_data = df[df["date"] < "2024-06-01"]
        valid_data = df[df["date"] >= "2024-06-01"]
        X_train = train_data[features]
        y_train = train_data[target]
        X_valid = valid_data[features]
        y_valid = valid_data[target]
        X_test = test_df[features]

        return X_train, X_valid, X_test, y_train, y_valid

    def hashing_encode(self,series, n_buckets=50021):
        return series.apply(lambda x: hash(x) % n_buckets)
    
    def frequency_encode(self,train_series, test_series):
        freq_map = train_series.value_counts().to_dict()
        train_encoded = train_series.map(freq_map).fillna(0)
        test_encoded = test_series.map(freq_map).fillna(0)
        return train_encoded, test_encoded

    def target_encode(self, train_series, target, test_series, n_splits=5, random_state=42):
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
        for col in ['dept_name', 'class_name', 'subclass_name', 'item_type', 'item_id']:
            self.combined_sales[col], self.test_processed[col] = self.target_encode(
                self.combined_sales[col],
                self.combined_sales["quantity"],
                self.test_processed[col],
            )

        self.combined_sales = self.one_hot_encode(self.combined_sales, "store_id")
        self.test_processed= self.one_hot_encode(self.test_processed, "store_id")
        exclude_cols = [
            "date",
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

        X_train, X_valid, X_test, y_train, y_valid = self.partition_data(        #change this line when using test_processed
            self.combined_sales, self.test_processed, features, target
        )

        X_train,X_valid,X_test = self.scale(X_train, X_valid, X_test)

        rf_model = RandomForestRegressor(
            n_estimators=30,
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

    def generate_submission_item_id(self):

        all_indices = np.arange(self.test.shape[0])
        pred_map = dict(zip(self.valid, self.predictions))

        final_quantities = []
        for i in all_indices:
            if i in pred_map:
                final_quantities.append(pred_map[i])
            else:
                final_quantities.append(3)

        submission_df = pd.DataFrame(
            {"row_id": all_indices, "quantity": final_quantities}
        )

        submission_df.to_csv('submission.csv', index=False)

    def generate_submission(self):
        self.generate_submission_item_id()
        # all_indices = np.arange(self.test.shape[0])
        # submission_df = pd.DataFrame(
        #     {"row_id": all_indices, "quantity": self.predictions}
        # )
        # submission_df.to_csv('submission.csv', index=False)



