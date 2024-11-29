# src/visualization.py

import logging
import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_predictions(y_trues, y_preds, labels, title="Comparison of Predictions", dates=None):
    """
    Plots multiple sets of actual vs predicted prices for comparison.

    Args:
        y_trues (list of np.ndarray): List of actual prices arrays.
        y_preds (list of np.ndarray): List of predicted prices arrays.
        labels (list of str): List of labels for each prediction set.
        title (str): Title of the plot.
        dates (np.ndarray): Dates corresponding to the prices.

    Returns:
        None
    """
    logging.info("Plotting multiple actual vs predicted prices for comparison.")
    plt.figure(figsize=(12, 6))
    for y_true, y_pred, label in zip(y_trues, y_preds, labels):
        if dates is not None:
            plt.plot(dates, y_true, label=f'Actual - {label}')
            plt.plot(dates, y_pred, label=f'Predicted - {label}')
        else:
            plt.plot(y_true, label=f'Actual - {label}')
            plt.plot(y_pred, label=f'Predicted - {label}')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_data(data, title="Stock Prices", xlabel="Time", ylabel="Price"):
    """
    Plots the stock prices.

    Args:
        data (np.ndarray): Stock prices data.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    logging.info("Plotting stock prices.")
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_stock_prices(symbols, data_dir, target_column="Adj_Close"):
    """
    Plots the stock prices for the given symbols.

    Args:
        symbols (list): List of stock symbols.
        data_dir (str): Directory where the stock data files are stored.
        target_column (str): Column name of the stock prices to plot.

    Returns:
        None
    """
    logging.info("Plotting stock prices for symbols.")
    plt.figure(figsize=(12, 6))
    for symbol in symbols:
        filepath = os.path.join(data_dir, f"{symbol}_stock_data.parquet")
        df = pd.read_parquet(filepath)
        plt.plot(df[target_column], label=symbol)
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred, dates=None, title="Actual vs Predicted Prices"):
    """
    Plots the actual vs predicted prices.

    Args:
        y_true (np.ndarray): Actual prices.
        y_pred (np.ndarray): Predicted prices.
        dates (np.ndarray): Dates corresponding to the prices.
        title (str): Title of the plot.

    Returns:
        None
    """
    logging.info("Plotting actual vs predicted prices.")
    plt.figure(figsize=(10, 6))
    if dates is not None:
        plt.plot(dates, y_true, label='Actual')
        plt.plot(dates, y_pred, label='Predicted')
    else:
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def plot_final_predictions(last_60_days, future_prices, symbol, sequence_length, pred_steps):
    """
    Plots the last 60 days of actual stock prices and the predicted next days.
    
    Args:
        last_60_days (np.ndarray): Array of the last 60 days of actual prices.
        future_prices (np.ndarray): Array of predicted future prices.
        symbol (str): Stock symbol.
        sequence_length (int): Number of past days used for prediction.
        pred_steps (int): Number of future days predicted.
    
    Returns:
        None
    """
    logging.info("Plotting the final predictions.")
    plot_prices = np.concatenate((last_60_days, future_prices))

    plt.figure(figsize=(12, 6))
    plt.plot(range(sequence_length), last_60_days, label='Last 60 Days')
    plt.plot(range(sequence_length, sequence_length + pred_steps), future_prices, label='Predicted Next Days', linestyle='--', marker='o')
    plt.title(f"{symbol} Last {sequence_length} Days and Predicted Next {pred_steps} Days")
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.show()