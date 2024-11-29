# src/model_training.py

import logging
import torch
import torch.nn as nn
from src.utils import check_gpu

class SequentialLSTM(nn.Module):
    """
    LSTM model for sequential data.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of output features.
        pred_steps (int): Number of future steps to predict.
        dropout (float): Dropout rate to prevent overfitting.

    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_steps, dropout=0.2):
        super(SequentialLSTM, self).__init__()
        self.pred_steps = pred_steps
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size * pred_steps)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, pred_steps, output_size).
        """
        output, (hn, cn) = self.lstm(x)
        out = output[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)      # Linear layer
        out = out.view(-1, self.pred_steps, 1)  # Reshape for multi-step prediction
        return out

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience (int): Number of epochs to wait before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            logging.debug(f"Initial validation loss set to {val_loss:.6f}.")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logging.debug(f"No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered.")
        else:
            self.best_loss = val_loss
            self.counter = 0
            logging.debug(f"Validation loss improved to {val_loss:.6f}.")
            
def train_model_with_early_stopping(
    model, X_train, y_train, X_val, y_val,
    epochs=500, batch_size=32, learning_rate=0.001, patience=5, verbose=True
):
    """
    Trains the model with early stopping.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation labels.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Patience for early stopping.
        verbose (bool): Whether to print training progress.

    Returns:
        model (nn.Module): The trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Training on device: {device}")

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i + batch_size].to(device)
            targets = y_train_tensor[i:i + batch_size].to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_loss = criterion(val_outputs, y_val_tensor.to(device)).item()

        # Calculate average losses
        avg_train_loss = epoch_loss / len(X_train_tensor)

        if verbose:
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, filepath="best_model_temp.pth")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            if verbose:
                logging.info("Early stopping triggered.")
            break

    # Load the best model weights
    model.load_state_dict(torch.load("best_model_temp.pth"))
    logging.info("Loaded the best model weights.")
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The trained PyTorch model.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation labels.

    Returns:
        val_loss (float): Validation loss.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        criterion = nn.MSELoss()
        val_loss = criterion(val_outputs, y_val_tensor).item()
    logging.debug(f"Validation loss: {val_loss:.6f}")
    return val_loss

def save_checkpoint(model, filepath="best_model.pth"):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (nn.Module): The PyTorch model to save.
        filepath (str): Path to the file where the model will be saved.
    """
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")

def load_model(model: nn.Module, load_path: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Loads the model state dictionary from a file.

    Args:
        model (nn.Module): The model instance to load the state into.
        load_path (str): Path to the saved model file.
        device (torch.device): Device to map the model to.

    Returns:
        model (nn.Module): The model with loaded state dictionary.
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {load_path}")
    return model

# Define the objective function for Optuna
def objective(trial, pred_steps, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        pred_steps (int): Number of future steps to predict.
        X_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation labels.

    Returns:
        val_loss (float): Validation loss after training.
    """
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    logging.info(f"Trial {trial.number}: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, learning_rate={learning_rate}, batch_size={batch_size}")

    # Instantiate the model
    model = SequentialLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        pred_steps=pred_steps,
        dropout=dropout
    )

    # Train the model
    model = train_model_with_early_stopping(
        model, X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=5,
        verbose=False  # Suppress output during optimization
    )

    # Evaluate the model
    val_loss = evaluate_model(model, X_val, y_val)

    logging.info(f"Trial {trial.number}: Validation Loss={val_loss:.6f}")

    return val_loss

def perform_inference(model, scaler, sequence_length, pred_steps, data_normalized):
    """
    Performs inference to predict future stock prices.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        scaler (MinMaxScaler): Scaler used for data normalization.
        sequence_length (int): Length of input sequences.
        pred_steps (int): Number of future steps to predict.
        data_normalized (np.ndarray): Normalized stock price data.
    
    Returns:
        future_prices (np.ndarray): Predicted future stock prices.
    """
    device = check_gpu()
    model.to(device)
    model.eval()

    # Prepare the last sequence from the data for inference
    last_sequence = data_normalized[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)  # Shape: (1, sequence_length, input_size)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)

    # Predict future prices
    logging.info(f"Predicting the next {pred_steps} days of stock prices.")
    with torch.no_grad():
        future_predictions = model(last_sequence_tensor)  # Shape: (1, pred_steps, output_size)
        future_predictions = future_predictions.cpu().numpy().reshape(-1, 1)

    # Inverse transform to get actual prices
    future_prices = scaler.inverse_transform(future_predictions).flatten()

    # Print predicted future prices
    logging.info(f"Predicted prices for the next {pred_steps} days:")
    for i, price in enumerate(future_prices, 1):
        logging.info(f"Day {i}: ${price:.2f}")

    return future_prices