import torch
from torch import nn
import torch.optim as optim

# Define a model using a modular D2L-style approach
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = None  # Lazy initialization to adapt dynamically

    def forward(self, x):
        if self.net is None:
            self.net = nn.Sequential(nn.Linear(x.shape[1], 3))  # Initialize dynamically
        return self.net(x)

# Create model instance
model = MyModel()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Dummy input tensor
x = torch.randn(5, 5)  # Batch size = 5, Feature size = 5

# Forward pass to initialize the model lazily
model(x)

# Print initial weights
print("Initial Weights (Before Training):")
for name, param in model.named_parameters():
    print(name, param.data)

# Training loop for a few steps
optimizer.zero_grad()
output = model(x)  # Forward pass
loss = output.sum()  # Compute a dummy loss
loss.backward()  # Backpropagation
optimizer.step()  # Update weights

# Print weights after training
print("\nWeights After Training:")
for name, param in model.named_parameters():
    print(name, param.data)

# Save model parameters
torch.save(model.state_dict(), "saved_model.pth")

# Reload model into a new instance
new_model = MyModel()
new_model.load_state_dict(torch.load("saved_model.pth"))

# Print weights after reloading
print("\nWeights After Reloading:")
for name, param in new_model.named_parameters():
    print(name, param.data)

# Verify that weights remain the same
for param_old, param_new in zip(model.parameters(), new_model.parameters()):
    assert torch.equal(param_old, param_new), "Weights do not match after reloading!"

print("\nModel successfully saved, reloaded, and verified.")
