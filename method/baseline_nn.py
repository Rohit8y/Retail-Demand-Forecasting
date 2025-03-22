import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from method.skeleton import Process
from torchinfo import summary
import os
from tqdm import tqdm


class MLZoomCampSalesDataset(Dataset):
    def __init__(self, df, numeric_features, cat_features, target=None):
        self.df = df
        self.numeric_features = numeric_features
        self.cat_features = cat_features
        self.target = target

        self.X_num = torch.tensor(df[self.numeric_features].values, dtype=torch.float32)
        self.X_cat = []
        for c in self.cat_features:
            self.X_cat.append(torch.tensor(df[c].values, dtype=torch.long))
        self.X_cat = torch.stack(self.X_cat, dim=1) if self.X_cat else None

        self.y = None
        if target is not None:
            self.y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_num = self.X_num[idx]
        x_cat = self.X_cat[idx] if self.X_cat is not None else None
        if self.y is not None:
            return x_num, x_cat, self.y[idx]
        else:
            return x_num, x_cat


class BaselineEmbeddingNN(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        embedding_dims,
        num_numeric,
        hidden_dims=[64, 32],
        dropout=0.1,
    ):
        super(BaselineEmbeddingNN, self).__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim)
                for cardinality, dim in zip(cat_cardinalities, embedding_dims)
            ]
        )

        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + num_numeric

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x_num, x_cat):
        if x_cat is not None:
            embedded = []
            for i, emb in enumerate(self.embeddings):
                embedded.append(emb(x_cat[:, i]))
            x_emb = torch.cat(embedded, dim=1)
            x = torch.cat([x_num, x_emb], dim=1)
        else:
            x = x_num

        out = self.model(x)
        return out


class BaselineNeuralNet(Process):
    def __init__(self, all_data, test_data, output_file):
        super().__init__(all_data, test_data, output_file)
        self.combined_sales = all_data
        self.test = test_data
        self.output_file = output_file

    def process_test_data(self):
        all_items = set(self.combined_sales["item_id"].to_list())
        test_items = set(self.test["item_id"].to_list())
        missing_items = test_items - all_items
        missing_indices = self.test[self.test["item_id"].isin(missing_items)].index
        valid_indices = self.test.index.difference(missing_indices)
        test_processed = self.test.drop(index=missing_indices)
        test_processed = test_processed.reset_index(drop=True)
        return test_processed, missing_indices, valid_indices

    def partition_data(self, df, test_df, features, target="quantity"):
        train_data = df[df["date"] < "2024-06-01"]
        valid_data = df[df["date"] >= "2024-06-01"]
        X_train = train_data[features]
        y_train = train_data[target]
        X_valid = valid_data[features]
        y_valid = valid_data[target]
        X_test = test_df[features]

        return X_train, X_valid, X_test, y_train, y_valid

    def rmse(self, y_true, y_pred):
        return torch.sqrt(((y_true - y_pred) ** 2).mean())

    def save_checkpoint(
        self, model, optimizer, epoch, best_rmse, file_path="models/checkpoint.pth"
    ):
        state = {
            "epoch": epoch,
            "best_rmse": best_rmse,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(state, file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, model, optimizer, file_path="models/checkpoint.pth"):
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]
            best_rmse = checkpoint["best_rmse"]
            print(f"Loaded checkpoint from {file_path} at epoch {epoch}")
            return model, optimizer, epoch, best_rmse
        else:
            print(f"No checkpoint found at {file_path}. Training from scratch.")
            return model, optimizer, 0, float("inf")

    def train_one_epoch(self, model, train_loader, optimizer, criterion, device):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training", leave=False)

        for x_num, x_cat, y in train_bar:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x_num, x_cat)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_num.size(0)
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        return train_loss

    def validate_one_epoch(self, model, valid_loader, criterion, device):
        model.eval()
        valid_loss = 0.0
        valid_preds = []
        valid_targets = []
        valid_bar = tqdm(valid_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for x_num, x_cat, y in valid_bar:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                preds = model(x_num, x_cat)
                loss = criterion(preds, y)
                valid_loss += loss.item() * x_num.size(0)
                valid_preds.append(preds.cpu())
                valid_targets.append(y.cpu())
                valid_bar.set_postfix(loss=loss.item())

        valid_loss /= len(valid_loader)
        valid_preds = torch.cat(valid_preds)
        valid_targets = torch.cat(valid_targets)
        return valid_loss, valid_preds, valid_targets

    def run_training(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
        n_epochs=100,
        checkpoint_path="models/checkpoint.pth",
        resume_training=True,
    ):
        model = model.to(device)

        if resume_training:
            model, optimizer, start_epoch, best_rmse = self.load_checkpoint(
                model, optimizer, checkpoint_path
            )

        best_rmse = float("inf")
        start_epoch = 0

        for epoch in range(start_epoch, n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")

            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            valid_loss, valid_preds, valid_targets = self.validate_one_epoch(
                model, valid_loader, criterion, device
            )
            valid_rmse = self.rmse(valid_targets, valid_preds).item()
            print(
                f"Train Loss: {train_loss:.4f} - Valid Loss: {valid_loss:.4f} - Valid RMSE: {valid_rmse:.4f}"
            )

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            self.save_checkpoint(
                model, optimizer, epoch + 1, best_rmse, file_path=checkpoint_path
            )

        model, _, _, _ = self.load_checkpoint(model, optimizer, checkpoint_path)

        return model, best_rmse

    def run(self):
        test_processed, missing_indices, valid_indices = self.process_test_data()
        exclude_cols = [
            "date",
            "sum_total",
            "source",
            "quantity",
            "price_base",
        ]
        cat_features = [
            "item_id",
            "dept_name",
            "class_name",
            "subclass_name",
            "item_type",
            "store_id",
        ]

        testandtrain = pd.concat(
            [self.combined_sales, test_processed], axis=0, ignore_index=True
        )

        for c in cat_features:
            testandtrain[c], _ = pd.factorize(testandtrain[c].astype(str))

        train_len = self.combined_sales.shape[0]
        self.combined_sales = testandtrain.iloc[:train_len].copy()
        test_processed = testandtrain.iloc[train_len:].copy().reset_index(drop=True)

        num_features = [
            col
            for col in self.combined_sales.columns
            if col not in exclude_cols and col not in cat_features
        ]

        X_train, X_valid, X_test, y_train, y_valid = self.partition_data(
            self.combined_sales, test_processed, num_features + cat_features
        )

        scaler = StandardScaler()
        X_train_numeric = scaler.fit_transform(X_train[num_features])
        X_valid_numeric = scaler.transform(X_valid[num_features])
        X_test_numeric = scaler.transform(X_test[num_features])

        X_train_scaled = X_train.copy()
        X_train_scaled[num_features] = X_train_numeric
        X_valid_scaled = X_valid.copy()
        X_valid_scaled[num_features] = X_valid_numeric
        X_test_scaled = X_test.copy()
        X_test_scaled[num_features] = X_test_numeric

        cat_cardinalities = []
        for c in cat_features:
            max_val = max(
                X_train_scaled[c].max(), X_valid_scaled[c].max(), X_test_scaled[c].max()
            )
            cat_cardinalities.append(int(max_val + 1))

        print(cat_cardinalities)

        embedding_dims = [min(100, (card // 2) + 1) for card in cat_cardinalities]

        train_dataset = MLZoomCampSalesDataset(
            X_train_scaled, num_features, cat_features, target=y_train
        )
        valid_dataset = MLZoomCampSalesDataset(
            X_valid_scaled, num_features, cat_features, target=y_valid
        )
        test_dataset = MLZoomCampSalesDataset(X_test_scaled, num_features, cat_features)

        train_loader_1sample = DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        sample_batch = next(iter(train_loader_1sample))
        print(sample_batch)
        sample_continuous, sample_categorical, sample_label = sample_batch

        model = BaselineEmbeddingNN(
            cat_cardinalities=cat_cardinalities,
            embedding_dims=embedding_dims,
            num_numeric=len(num_features),
            hidden_dims=[128, 64],
            dropout=0.1,
        )

        summary(model, input_data=(sample_continuous, sample_categorical))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.MSELoss()

        model, best_rmse = self.run_training(
            model, train_loader, valid_loader, optimizer, criterion, device, 10
        )

        print(best_rmse)

    def generate_submission(self):
        pass
