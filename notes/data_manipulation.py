import numpy as np
import torch
import matplotlib.pyplot as plt

# Create a 2D NumPy array
np_array = np.array([[1, 2, 3], [4, 5, 6]])

# Create a 2D PyTorch tensor with the same data
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Plotting the NumPy array
plt.subplot(1, 2, 1)
plt.imshow(np_array, cmap='viridis', interpolation='none')
plt.title('NumPy Array')
plt.colorbar()

# Plotting the PyTorch tensor
plt.subplot(1, 2, 2)
plt.imshow(torch_tensor.numpy(), cmap='viridis', interpolation='none')
plt.title('PyTorch Tensor')
plt.colorbar()

plt.show()

print("\n Part 1: Creating Arrays and Tensors \n")

# Create a NumPy array
np_array = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy Array:\n", np_array)

# Create a PyTorch tensor
torch_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("PyTorch Tensor:\n", torch_tensor)

print("\n Part 2: Common Operations \n")

# Element-wise addition
np_array2 = np.ones((2, 3))  # NumPy
print("np array 2: \n ", np_array2)
torch_tensor2 = torch.ones((2, 3))  # PyTorch
print("torch tensor 2: \n ", torch_tensor2)

print("NumPy Addition:\n", np_array + np_array2)
print("PyTorch Addition:\n", torch_tensor + torch_tensor2)

# Matrix multiplication
np_mat1 = np.array([[1, 2], [3, 4]])
print("np mat1: \n ", np_mat1)
np_mat2 = np.array([[5, 6], [7, 8]])
print("np mat2: \n ", np_mat2)
print("NumPy Matrix Multiplication:\n", np.dot(np_mat1, np_mat2))

torch_mat1 = torch.tensor([[1, 2], [3, 4]])
torch_mat2 = torch.tensor([[5, 6], [7, 8]])
print("PyTorch Matrix Multiplication:\n", torch.matmul(torch_mat1, torch_mat2))

print("\n Part 3: Broadcasting Example \n")

# Broadcasting in NumPy
np_array3 = np.array([[1, 2, 3], [4, 5, 6]])
np_broadcast = np.array([10, 20, 30])
print("Broadcasted Addition in NumPy:\n", np_array3 + np_broadcast)

# Broadcasting in PyTorch
torch_array3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch_broadcast = torch.tensor([10, 20, 30])
print("Broadcasted Addition in PyTorch:\n", torch_array3 + torch_broadcast)

print("\n Part 4: Reshaping Example \n")

# Reshaping in NumPy
np_array4 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Reshaped NumPy Array:\n", np_array4.reshape(1, -1))

# Reshaping in PyTorch
torch_array4 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Reshaped PyTorch Tensor:\n", torch_array4.view(1, -1))