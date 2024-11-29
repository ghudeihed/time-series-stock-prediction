# src/utils.py

import logging
import numpy as np
import pandas as pd
import random
import torch
import yfinance as yf
import os

# Function to download stock data for a list of symbols
def download_stock_data(symbols, start_date="2015-01-01", end_date=None, data_dir="data"):
    """
    Downloads stock data for the given symbols and saves them as parquet files.

    Args:
        symbols (list): List of stock symbols to download.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format. If None, defaults to today's date.
        data_dir (str): Directory to save the downloaded data.

    Returns:
        None
    """
    logging.info("Starting stock data download.")
    ensure_dir_exists(data_dir)
    
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    for symbol in symbols:
        logging.info(f"Downloading data for {symbol} from {start_date} to {end_date}.")
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            
            # Flatten MultiIndex if present and clean column names
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = ['_'.join(col).strip().replace(' ', '_') for col in stock_data.columns]

            # Save data to Parquet format for efficient storage
            file_path = os.path.join(data_dir, f"{symbol}_stock_data.parquet")
            stock_data.to_parquet(file_path)
            logging.info(f"Data for {symbol} saved to {file_path}")
        
        except Exception as e:
            logging.error(f"Error downloading data for {symbol}: {e}")

# Load data function
def load_data(filepath):
    """
    Loads stock data from a parquet file.

    Args:
        filepath (str): Path to the parquet file.

    Returns:
        df (pd.DataFrame): DataFrame containing the stock data.
    """
    logging.info(f"Loading data from {filepath}")   
    try:
        df = pd.read_parquet(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    return df

def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensures that the specified directory exists; creates it if it does not.
    
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logging.info(f"Directory created: {dir_path}")
    else:
        logging.debug(f"Directory already exists: {dir_path}")

# Check if GPU is available
def check_gpu():
    """
    Checks if a GPU is available and returns the appropriate device.

    Returns:
        device (torch.device): 'cuda' if GPU is available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("No GPU detected. Using CPU.")
    
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Using device: {device}")

    return device

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    logging.info(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
