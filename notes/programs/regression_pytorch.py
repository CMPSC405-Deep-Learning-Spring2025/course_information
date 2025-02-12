import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

def plot_loss(loss_values, filename='loss_plot.png'):
    """
    Plot and save the loss values over epochs.
    """
    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.savefig(filename)
    plt.close()

def plot_predictions(y_true, y_pred, filename='predictions_plot.png'):
    """
    Plot and save the true vs predicted values.
    """
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.savefig(filename)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Load and preprocess the data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Initialize the model, loss function, and optimizer
    model = LinearRegressionTorch(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    epochs = 1000
    loss_values = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if epoch % 100 == 0:  # Print loss every 100 epochs
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}')

    # Plot and save the loss values
    plot_loss(loss_values)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        plot_predictions(y_test, predictions)
