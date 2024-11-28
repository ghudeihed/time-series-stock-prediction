import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to download stock data for a list of symbols
def download_stock_data(symbols, start_date="2015-01-01", end_date=None):
    """
    Download stock data for a list of symbols using yfinance.
    :param symbols: List of stock symbols.
    :param start_date: Start date for the data download.
    :param end_date: End date for the data download.
    """
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

# Preprocessing function: Normalizes the target data
def preprocess_data(df, target_column="Close", feature_columns=None, scaler=None):
    """
    Normalize the target column and return normalized data and scaler.
    
    Args:
        df (pd.DataFrame): DataFrame containing the stock data.
        target_column (str): The column to be used as the target for predictions.
        feature_columns (list): List of feature columns to include. Defaults to target column only.
        scaler (sklearn.preprocessing): Optional scaler instance to use.

    Returns:
        np.ndarray: Normalized data.
        sklearn.preprocessing: Fitted scaler.
    """
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in DataFrame.")

    # Select relevant features for LSTM
    feature_columns = feature_columns or [target_column]
    data = df[feature_columns].values

    # Handle missing values (forward and backward fill)
    data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values

    # Initialize scaler if none provided
    if scaler is None:
        scaler = MinMaxScaler()

    # Normalize data to the range [0, 1]
    data_normalized = scaler.fit_transform(data)
    
    logging.info(f"Data normalization complete. Shape: {data_normalized.shape}")
    return data_normalized, scaler

# Sequence creation function
def create_sequences(data, sequence_length):
    """
    Create sequences for LSTM training.
    
    Args:
        data (np.ndarray): Normalized data.
        sequence_length (int): Length of each sequence.

    Returns:
        np.ndarray: Array of input sequences.
        np.ndarray: Array of corresponding labels.
    """
    if len(data) <= sequence_length:
        raise ValueError("Data length must be greater than the sequence length.")

    # Create sequences and corresponding labels
    num_sequences = len(data) - sequence_length
    sequences = np.array([data[i:i + sequence_length] for i in range(num_sequences)])
    labels = np.array([data[i + sequence_length] for i in range(num_sequences)])

    logging.info(f"Sequences created. Number of sequences: {len(sequences)}, Sequence length: {sequence_length}")
    return sequences, labels

# Splitting the data function
def split_data(sequences, labels, test_size=0.2):
    """
    Split data into training and validation sets.
    
    Args:
        sequences (np.ndarray): Input sequences for training.
        labels (np.ndarray): Output labels for each sequence.
        test_size (float): Fraction of the dataset to be used as validation data.

    Returns:
        tuple: Split training and validation datasets.
    """
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_size, random_state=42)
    logging.info(f"Data split complete. Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val

# Preprocessing function to normalize and create sequences for each stock symbol
def preprocess_and_create_sequences(symbols, sequence_length, target_column="Close"):
    all_X_train = []
    all_X_val = []
    all_y_train = []
    all_y_val = []

    for symbol in symbols:
        target_column_name = f"{target_column}_{symbol}"
        # Load and preprocess data for each symbol
        data_file_path = os.path.join(data_dir, f"{symbol}_stock_data.parquet")
        try:
            df = load_data(data_file_path)

            if target_column_name not in df.columns:
                logging.error(f"Column '{target_column_name}' not found in DataFrame for {symbol}")
                continue

            # Normalize the target column
            data = df[target_column_name].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data)

            # Create sequences and labels for LSTM
            sequences, labels = create_sequences(data_normalized, sequence_length)

            # Split into training and validation datasets
            X_train, X_val, y_train, y_val = split_data(sequences, labels)

            # Append the data for each symbol
            all_X_train.append(X_train)
            all_X_val.append(X_val)
            all_y_train.append(y_train)
            all_y_val.append(y_val)

        except Exception as e:
            logging.error(f"Failed to load or preprocess data for {symbol}: {e}")

if __name__ == "__main__":
    from src.utils import load_data
    # Load and preprocess data
    file_path = "../data/stock_data.parquet"
    sequence_length = 60

    try:
        # Step 1: Load data
        df = load_data(file_path)

        # Step 2: Preprocess data
        data_normalized, scaler = preprocess_data(df, target_column="Close_AAPL")

        # Step 3: Create sequences
        sequences, labels = create_sequences(data_normalized, sequence_length)

        # Step 4: Split data into training and validation sets
        X_train, X_val, y_train, y_val = split_data(sequences, labels)

        logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
