import os
import matplotlib.pyplot as plt
import numpy as np
import logging

from src.utils import load_data

# Set up logging for detailed tracking of progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("project_log.log")  # Log to a file as well
    ]
)

def plot_data(data, dates=None, title="Stock Prices", xlabel="Date", ylabel="Price"):
    """
    Plot stock data.
    """
    plt.figure(figsize=(12, 8))
    if dates is not None:
        plt.plot(dates, data, label="Stock Prices", linestyle='-', marker='o', color='teal')
    else:
        plt.plot(data, label="Stock Prices", linestyle='-', marker='o', color='teal')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting function for stock prices from multiple symbols
def plot_stock_prices(symbols, data_dir, target_column="Close"):
    plt.figure(figsize=(14, 8))

    for symbol in symbols:
        target_column_name = f"{target_column}_{symbol}"
        data_file_path = os.path.join(data_dir, f"{symbol}_stock_data.parquet")
        try:
            # Load the data for each symbol
            df = load_data(data_file_path)

            if target_column_name not in df.columns:
                logging.error(f"Column '{target_column_name}' not found in DataFrame for {symbol}")
                continue

            dates = df.index  # Assuming the DataFrame index is the date
            prices = df[target_column_name].values

            # Plot the stock prices for the current symbol
            plt.plot(dates, prices, label=f"{symbol} Prices")

        except Exception as e:
            logging.error(f"Failed to load or plot data for {symbol}: {e}")
    
    # Add plot details
    plt.title("Stock Prices for Multiple Symbols")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, dates=None):
    """
    Plot true vs predicted stock prices with error visualization.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    plt.figure(figsize=(12, 8))

    if dates is not None:
        plt.plot(dates, y_true, label="Actual Prices", color="blue", linestyle='-', marker='o')
        plt.plot(dates, y_pred, label="Predicted Prices", color="orange", linestyle='-', marker='x')
        plt.fill_between(dates, y_true, y_pred, color='gray', alpha=0.2, label="Prediction Error")
    else:
        plt.plot(y_true, label="Actual Prices", color="blue", linestyle='-', marker='o')
        plt.plot(y_pred, label="Predicted Prices", color="orange", linestyle='-', marker='x')
        plt.fill_between(range(len(y_true)), y_true, y_pred, color='gray', alpha=0.2, label="Prediction Error")

    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Time" if dates is None else "Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_predictions(y_trues, y_preds, labels, title="Comparison of Predictions", dates=None):
    """
    Plot multiple true vs predicted stock price series for comparison.
    """
    plt.figure(figsize=(12, 8))
    for i in range(len(y_trues)):
        if dates is not None:
            plt.plot(dates, y_trues[i], label=f"Actual {labels[i]}", linestyle='-', color=f"C{i}", marker='o')
            plt.plot(dates, y_preds[i], label=f"Predicted {labels[i]}", linestyle='--', color=f"C{i}", alpha=0.8, marker='x')
        else:
            plt.plot(y_trues[i], label=f"Actual {labels[i]}", linestyle='-', color=f"C{i}", marker='o')
            plt.plot(y_preds[i], label=f"Predicted {labels[i]}", linestyle='--', color=f"C{i}", alpha=0.8, marker='x')

    plt.title(title)
    plt.xlabel("Date" if dates is not None else "Time")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    y_true = np.sin(np.linspace(0, 10, 100))
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    
    # Plotting predictions with error visualization
    plot_predictions(y_true, y_pred)

    # Example data for multiple predictions comparison
    y_trues = [y_true, y_true * 0.8]
    y_preds = [y_pred, y_true * 0.8 + np.random.normal(0, 0.15, 100)]
    labels = ["Model 1", "Model 2"]

    # Plotting multiple predictions for comparison
    plot_multiple_predictions(y_trues, y_preds, labels)
