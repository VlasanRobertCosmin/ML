import os
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # NEW: for plots


# ============================
# Configuration
# ============================
DATA_PATH = "AllData25sym2019.txt"  # make sure this file is in the same folder
SYMBOL = "SNP"                      # your chosen symbol
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
USE_CUDA = torch.cuda.is_available()


# ============================
# Dataset
# ============================
class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        data: numpy array of shape (num_samples, num_features) - already scaled.
        We will create sequences of length seq_len to predict the next day's ClosePrice.
        ClosePrice is assumed to be at index 1 of features.
        """
        self.seq_len = seq_len

        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i: i + seq_len, :])
            # Predict next day's ClosePrice (index 1)
            y.append(data[i + seq_len, 1])

        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# GRU Model
# ============================
class GRUStockModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)          # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]   # last time step
        out = self.fc(last_hidden)    # (batch, 1)
        return out


# ============================
# Utility functions
# ============================
def prepare_data(df_symbol):
    """
    df_symbol: dataframe filtered on one symbol, sorted by date.
    Returns scaled data array + scaler for inverse transform.
    """
    feature_cols = ["OpenPrice", "ClosePrice", "MinPrice", "MaxPrice", "MedPrice"]
    data = df_symbol[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler, feature_cols


def train_val_test_split(data_scaled, val_split=0.15, test_split=0.15):
    n = len(data_scaled)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_data = data_scaled[:n_train]
    # add overlap of SEQ_LEN for sequence continuity
    val_data = data_scaled[n_train - SEQ_LEN: n_train + n_val]
    test_data = data_scaled[n_train + n_val - SEQ_LEN:]

    return train_data, val_data, test_data


def create_dataloaders(train_data, val_data, test_data):
    train_ds = StockDataset(train_data, SEQ_LEN)
    val_ds = StockDataset(val_data, SEQ_LEN)
    test_ds = StockDataset(test_data, SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(loader.dataset)


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # Load and filter data
    # ----------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, sep="\t")

    # Parse dates and sort
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df_symbol = df[df["Symbol"] == SYMBOL].copy().sort_values("Date")

    if len(df_symbol) < SEQ_LEN + 10:
        raise ValueError(f"Not enough data for symbol {SYMBOL}")

    print(f"Number of rows for symbol {SYMBOL}: {len(df_symbol)}")

    # ----------------------------
    # Prepare scaled data
    # ----------------------------
    data_scaled, scaler, feature_cols = prepare_data(df_symbol)
    print(f"Using features: {feature_cols}")

    # ----------------------------
    # Train/val/test split
    # ----------------------------
    train_data, val_data, test_data = train_val_test_split(
        data_scaled, val_split=VAL_SPLIT, test_split=TEST_SPLIT
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data
    )

    input_size = train_data.shape[1]
    model = GRUStockModel(input_size=input_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ----------------------------
    # Training loop (with saving losses)
    # ----------------------------
    print("Starting training...")
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"- Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
        )

    # Save loss curves to CSV
    loss_df = pd.DataFrame({
        "epoch": range(1, EPOCHS + 1),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    loss_filename = f"gru_training_results_{SYMBOL}.csv"
    loss_df.to_csv(loss_filename, index=False)
    print(f"Saved training results to {loss_filename}")

    # ----------------------------
    # Plot loss curve
    # ----------------------------
    plt.figure()
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train loss")
    plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"GRU Training vs Validation Loss ({SYMBOL})")
    plt.legend()
    loss_plot_name = f"gru_loss_curve_{SYMBOL}.png"
    plt.savefig(loss_plot_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve plot to {loss_plot_name}")

    # ----------------------------
    # Evaluate on test set
    # ----------------------------
    test_loss = eval_model(model, test_loader, criterion, device)
    print(f"\nFinal TEST MSE loss: {test_loss:.6f}")

    # ----------------------------
    # Example prediction + saving predictions
    # ----------------------------
    model.eval()
    with torch.no_grad():
        X_last, y_last = next(iter(test_loader))  # first batch from test set
        X_last = X_last.to(device)
        y_last = y_last.to(device)
        preds = model(X_last)

    # Inverse-scale a few examples (ClosePrice is feature index 1)
    close_min = scaler.data_min_[1]
    close_max = scaler.data_max_[1]

    def inv_scale_close(x):
        return x * (close_max - close_min) + close_min

    y_true = inv_scale_close(y_last.squeeze().cpu().numpy())
    y_pred = inv_scale_close(preds.squeeze().cpu().numpy())

    print("\nSample predictions (ClosePrice):")
    for i in range(min(5, len(y_true))):
        print(f"True: {y_true[i]:.4f} | Pred: {y_pred[i]:.4f}")

    # Save some predictions to CSV
    n_save = min(50, len(y_true))
    pred_df = pd.DataFrame({
        "index": list(range(n_save)),
        "true_close": y_true[:n_save],
        "pred_close": y_pred[:n_save],
    })
    pred_filename = f"gru_predictions_{SYMBOL}.csv"
    pred_df.to_csv(pred_filename, index=False)
    print(f"Saved sample predictions to {pred_filename}")

    # ----------------------------
    # Plot true vs predicted
    # ----------------------------
    plt.figure()
    plt.plot(pred_df["index"], pred_df["true_close"], label="True Close")
    plt.plot(pred_df["index"], pred_df["pred_close"], label="Predicted Close")
    plt.xlabel("Sample index (test batch)")
    plt.ylabel("Close price")
    plt.title(f"True vs Predicted Close Price ({SYMBOL})")
    plt.legend()
    pred_plot_name = f"gru_pred_vs_true_{SYMBOL}.png"
    plt.savefig(pred_plot_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to {pred_plot_name}")

    # ----------------------------
    # Save model weights
    # ----------------------------
    model_filename = f"gru_model_{SYMBOL}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Saved model to {model_filename}")

    print("\nTraining complete.")
    print("For the lab, you now have:")
    print(f"  * {loss_filename}  (loss values)")
    print(f"  * {loss_plot_name} (loss curve PNG)")
    print(f"  * {pred_filename}  (true vs pred values)")
    print(f"  * {pred_plot_name} (true vs pred PNG)")
    print(f"  * {model_filename} (saved GRU model)")


if __name__ == "__main__":
    main()
