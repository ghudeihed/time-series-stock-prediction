# main.py

import logging
import os
from functools import partial

import numpy as np
import optuna
import torch

from src.data_preprocessing import create_sequences, preprocess_data, split_data
from src.model_training import SequentialLSTM, train_model_with_early_stopping, evaluate_model, save_checkpoint, objective
from src.utils import load_data, set_seed, check_gpu, download_stock_data
from src.visualization import plot_predictions, plot_data

def main():
    # Set up logging for detailed tracking of progress
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("project_log.log")  # Log to a file as well
        ]
    )
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Download and dave data
    symbols = ["AAPL", "MSFT", "NVDA"]
    download_stock_data(symbols)
    logging.info(f"Downloading data for {symbols}...")
    
    # Load the stock data
    symbol = "AAPL"
    data_file_path = f"data/{symbol}_stock_data.parquet"
    target_column = f"Adj_Close_{symbol}"
    sequence_length = 60  # Number of past days to use for each prediction
    pred_steps = 5        # Number of future days to predict
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Download stock data if not already present (optional)
    # Uncomment the following lines if you have implemented download_stock_data
    # from src.utils import download_stock_data
    # symbols = ["AAPL"]
    # download_stock_data(symbols, start_date="2015-01-01", end_date=None, data_dir="data")
    
    # Load the stock data
    df = load_data(data_file_path)
    
    # Preprocess the data
    data_normalized, scaler = preprocess_data(df, target_column)
    
    # Create sequences and labels
    sequences, labels = create_sequences(data_normalized, sequence_length, pred_steps)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = split_data(sequences, labels, test_size=0.2)
    
    # Define the objective function with additional arguments using partial
    study_objective = partial(objective, pred_steps=pred_steps, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    
    # Create the Optuna study
    study = optuna.create_study(direction='minimize', study_name='LSTM_Hyperparameter_Optimization')
    
    # Run the hyperparameter optimization
    logging.info("Starting hyperparameter optimization with Optuna.")
    study.optimize(study_objective, n_trials=20, timeout=None)  # Adjust n_trials as needed
    
    # Retrieve the best hyperparameters
    best_hyperparams = study.best_params
    logging.info("Best hyperparameters found:")
    for key, value in best_hyperparams.items():
        logging.info(f"{key}: {value}")
    
    # Instantiate the best model
    best_model = SequentialLSTM(
        input_size=1,
        hidden_size=best_hyperparams['hidden_size'],
        num_layers=best_hyperparams['num_layers'],
        output_size=1,
        pred_steps=pred_steps,
        dropout=best_hyperparams['dropout']
    )
    
    # Train the best model with the best hyperparameters
    logging.info("Training the best model with the best hyperparameters.")
    best_model = train_model_with_early_stopping(
        best_model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=best_hyperparams['batch_size'],
        learning_rate=best_hyperparams['learning_rate'],
        patience=5,
        verbose=True
    )
    
    # Save the best model
    save_checkpoint(best_model, filepath="best_model.pth")
    
    # Final evaluation on the validation set
    val_loss = evaluate_model(best_model, X_val, y_val)
    logging.info(f"Final Validation Loss of the Best Model: {val_loss:.6f}")
    
    # Perform inference for future predictions
    device = check_gpu()
    best_model.to(device)
    best_model.eval()
    
    # Prepare the last sequence from the data for inference
    last_sequence = data_normalized[-sequence_length:]
    last_sequence = last_sequence.reshape(1, sequence_length, 1)  # Shape: (1, sequence_length, input_size)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    
    # Predict future prices
    logging.info(f"Predicting the next {pred_steps} days of stock prices.")
    with torch.no_grad():
        future_predictions = best_model(last_sequence_tensor)  # Shape: (1, pred_steps, output_size)
        future_predictions = future_predictions.cpu().numpy().reshape(-1, 1)
    
    # Inverse transform to get actual prices
    future_prices = scaler.inverse_transform(future_predictions).flatten()
    
    # Print predicted future prices
    logging.info(f"Predicted prices for the next {pred_steps} days:")
    for i, price in enumerate(future_prices, 1):
        logging.info(f"Day {i}: ${price:.2f}")
    
    # Plot the last 60 days and the predicted next days
    last_60_days_prices = scaler.inverse_transform(data_normalized[-sequence_length:].reshape(-1, 1)).flatten()
    plot_prices = np.concatenate((last_60_days_prices, future_prices))
    
    plot_data(
        data=plot_prices,
        title=f"{symbol} Last {sequence_length} Days and Predicted Next {pred_steps} Days",
        xlabel="Days",
        ylabel="Price"
    )

if __name__ == "__main__":
    main()
