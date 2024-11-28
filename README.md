# Time-Series Stock Prediction using LSTM Networks

This project demonstrates how to use LSTM (Long Short-Term Memory) neural networks for predicting stock prices using historical data. The project employs data from multiple stocks, downloaded using the Yahoo Finance API, and processed with PyTorch for training a sequential prediction model. The main goal is to showcase time-series forecasting using deep learning techniques.

## Project Structure

```
├── data/
│   └── *.parquet                 # Stock data for different symbols
├── notebooks/
│   └── stock_prediction.ipynb    # Notebook with the end-to-end implementation
├── src/
│   ├── data_preprocessing.py     # Data preprocessing functions
│   ├── model_training.py         # Model definition and training functions
│   ├── visualization.py          # Plotting functions
│   └── utils.py                  # Utility functions such as directory setup, GPU check
├── README.md                     # Project documentation
└── requirements.txt              # Project dependencies
```

## Project Overview
This project aims to predict future stock prices using LSTMs, a type of Recurrent Neural Network (RNN) suited for time-series data due to their ability to retain sequential dependencies. We leverage historical daily stock data from Yahoo Finance, train a model, and evaluate its ability to forecast stock prices for major technology companies.

### Key Features
- **Data Collection**: Downloads historical data using Yahoo Finance for multiple stocks.
- **Data Preprocessing**: Handles missing values, normalizes the data, and creates sequences for model training.
- **Model Training**: Implements an LSTM model using PyTorch, with features like early stopping and gradient clipping.
- **Visualization**: Plots the stock price trends, predictions vs actual values, and other related data.

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- `pip` package installer
- GPU with CUDA support (optional but recommended)

### Installation
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/ghudeihed/time-series-stock-prediction.git
   cd time-series-stock-prediction
   ```

2. **Create and Activate a Virtual Environment**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

### Directory Setup
- **Data Directory**: Ensure the `data/` directory exists. This directory is where all stock data files are saved in Parquet format.

### GPU Setup (Optional)
To utilize GPU for model training, ensure you have CUDA installed and that PyTorch is configured to use GPU. You can verify CUDA availability using the `nvidia-smi` command.

## Running the Project

1. **Download Stock Data**
   - Run the following script to download historical data for specified stocks:
     ```python
     from src.utils import download_stock_data
     stock_symbols = ["AAPL", "MSFT", "NVDA"]
     download_stock_data(stock_symbols)
     ```

2. **Preprocess Data**
   - Load and preprocess the stock data for training. This includes handling missing values, normalizing the dataset, and creating sequences.

3. **Train the Model**
   - The LSTM model is trained using the prepared sequences. You can modify hyperparameters such as sequence length, learning rate, batch size, etc. The model training includes:
     - Early stopping to prevent overfitting.
     - Checkpointing to save the best model during training.

4. **Evaluate the Model**
   - Use the trained model to make predictions on the validation set and visualize the actual vs predicted prices.

5. **Visualization**
   - Use the provided plotting utilities to visualize stock trends and the performance of your model predictions.

## Project Notebook
The `notebooks/stock_prediction.ipynb` notebook provides an interactive, step-by-step walkthrough of the entire process from data collection to model evaluation.

## Example Usage
To run the main steps of the project:
```python
from src.utils import ensure_dir_exists
from src.model_training import train_model_with_early_stopping
from src.data_preprocessing import preprocess_data, create_sequences, split_data
from src.visualization import plot_predictions

# Download stock data for selected symbols
symbols = ["AAPL", "MSFT", "NVDA"]
download_stock_data(symbols)

# Load, preprocess and create training/validation sequences
data = load_data("data/AAPL_stock_data.parquet")
data_normalized, scaler = preprocess_data(data, target_column="Close_AAPL")
sequences, labels = create_sequences(data_normalized, sequence_length=60)
X_train, X_val, y_train, y_val = split_data(sequences, labels)

# Train the LSTM model
model = SequentialLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1)
train_model_with_early_stopping(model, X_train, y_train, X_val, y_val)

# Visualize predictions
plot_predictions(y_val, model(X_val))
```

## Future Enhancements
- **Hyperparameter Tuning**: Implement grid search or random search to optimize hyperparameters.
- **Ensemble Methods**: Combine multiple models to enhance the accuracy of stock price predictions.
- **Different Model Architectures**: Experiment with GRUs or Transformer-based architectures for improved results.

## Known Issues
- **Data Quality**: Stock data often contains missing values or anomalies that can impact model accuracy. Make sure to preprocess properly.
- **Overfitting**: LSTMs can overfit on small datasets. Using dropout and early stopping helps mitigate this issue.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Yahoo Finance** for providing easy access to historical stock data.
- **PyTorch and Scikit-learn** for offering robust libraries that made this project possible.