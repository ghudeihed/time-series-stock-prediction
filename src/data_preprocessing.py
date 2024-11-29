# src/data_preprocessing.py

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df, target_column="Adj_Close", feature_columns=None, scaler=None):
    """
    Preprocesses the data by selecting the target column and normalizing it.

    Args:
        df (pd.DataFrame): DataFrame containing the stock data.
        target_column (str, optional): The name of the target column. Defaults to "Adj_Close".
        feature_columns (_type_, optional): _description_. Defaults to None.
        scaler (_type_, optional): _description_. Defaults to None.

    Returns:
        data_normalized (np.ndarray): Normalized data.
        scaler (MinMaxScaler): Scaler used for normalization.
    """
    logging.info("Starting data preprocessing.")
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

def create_sequences(data, sequence_length, pred_steps):
    """
    Creates input sequences and corresponding labels for multi-step prediction.

    Args:
        data (np.ndarray): The normalized data array.
        sequence_length (int): Length of each input sequence.
        pred_steps (int): Number of future steps to predict.

    Returns:
        sequences (np.ndarray): Array of input sequences.
        labels (np.ndarray): Array of corresponding labels (future values).
    """
    logging.info("Creating sequences.")
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length - pred_steps + 1):
        seq = data[i:i + sequence_length]
        label = data[i + sequence_length:i + sequence_length + pred_steps]
        sequences.append(seq)
        labels.append(label)
    sequences = np.array(sequences)
    labels = np.array(labels)
    logging.info(f"Sequences created. Number of sequences: {len(sequences)}, Sequence length: {sequence_length}")
    return sequences, labels

def split_data(sequences, labels, test_size=0.2):
    """
    Splits the sequences and labels into training and validation sets.

    Args:
        sequences (np.ndarray): Input sequences.
        labels (np.ndarray): Corresponding labels.
        test_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        X_train (np.ndarray): Training sequences.
        X_val (np.ndarray): Validation sequences.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
    """
    logging.info(f"Splitting data with test size {test_size}.")
    split_index = int(len(sequences) * (1 - test_size))
    X_train = sequences[:split_index]
    y_train = labels[:split_index]
    X_val = sequences[split_index:]
    y_val = labels[split_index:]
    logging.info(f"Data split into {len(X_train)} training and {len(X_val)} validation samples.")
    return X_train, X_val, y_train, y_val
