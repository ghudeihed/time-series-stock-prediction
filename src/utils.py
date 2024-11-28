import os
import torch
import pandas as pd
from torch import nn

import logging

# Set up logging for detailed tracking of progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("project_log.log")  # Log to a file as well
    ]
)

# Load data function


def load_data(parquet_path):
    """
    Load stock data from a Parquet file.
    
    Args:
        parquet_path (str): Path to the Parquet file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Data loaded successfully from {parquet_path}")
    except FileNotFoundError:
        logging.error(f"File not found: {parquet_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    return df

def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensure a directory exists, and create it if it doesn't.
    
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created at: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

# Check if GPU is available and print CUDA version and PyTorch version
def check_gpu():
    if torch.cuda.is_available():
        logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU detected. Using CPU.")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"PyTorch version: {torch.__version__}")

def save_model(model: nn.Module, save_path: str) -> None:
    """
    Save the trained model to a specified path.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        save_path (str): Path to save the model.
    """
    ensure_dir_exists(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")


def load_model(model: nn.Module, load_path: str, device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    Load a model from a specified path.
    
    Args:
        model (nn.Module): The PyTorch model instance to load the weights into.
        load_path (str): Path to the saved model file.
        device (torch.device): Device to load the model on (default is CPU).

    Returns:
        nn.Module: The model loaded with the specified weights.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at '{load_path}'")
    
    try:
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)
        model.eval()  # Set to evaluation mode after loading
        print(f"Model successfully loaded from {load_path}")
    except Exception as e:
        print(f"Error loading model from {load_path}: {e}")
        raise e
    
    return model


# Example usage
if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Example paths
    model_save_path = "./saved_models/my_model.pth"

    # Initialize model
    model = SimpleModel()

    # Save model
    save_model(model, model_save_path)

    # Load model
    loaded_model = load_model(SimpleModel(), model_save_path, device=torch.device("cpu"))
