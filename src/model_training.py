import torch
import torch.nn as nn
import numpy as np


class SequentialLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SequentialLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stopping to terminate training when validation loss stops improving.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(model, filepath="best_model.pth"):
    """
    Save the model checkpoint.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model_with_early_stopping(
    model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5, checkpoint_path="best_model.pth"
):
    # Detect device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create DataLoader objects for training and validation
    train_data = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break


# Example usage
if __name__ == "__main__":
    # Example data (replace with your actual data)
    X_train = np.random.rand(800, 60, 1)
    y_train = np.random.rand(800, 1)
    X_val = np.random.rand(200, 60, 1)
    y_val = np.random.rand(200, 1)

    # Model initialization
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model = SequentialLSTM(input_size, hidden_size, num_layers, output_size)

    # Train the model with early stopping
    train_model_with_early_stopping(
        model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=5, checkpoint_path="best_lstm_model.pth"
    )
