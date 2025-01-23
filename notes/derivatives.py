import torch
import matplotlib.pyplot as plt
import numpy as np

# Function: y = x^2
def f(x):
    return x**2

# Derivative: dy/dx = 2x
def df(x):
    return 2 * x

# Gradient descent visualization
x = np.linspace(-2, 2, 100)
y = f(x)

# Start at an initial point
current_x = 1.5
learning_rate = 0.1
steps = []

for _ in range(5):  # Take 5 gradient descent steps
    steps.append(current_x)
    current_x -= learning_rate * df(current_x)

plt.plot(x, y, label="y = x^2")
plt.scatter(steps, [f(x) for x in steps], color="red", label="Steps")
plt.legend()
plt.title("Gradient Descent Steps")
plt.xlabel("x")
plt.ylabel("y")
plt.show()