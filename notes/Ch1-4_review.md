# Review Game

## Round 1: Multiple Choice Questions
### Format: Each team answers 5 multiple-choice questions. 
### Scoring: 1 points for correct answers, 0 points for incorrect answers.

### Questions:
1. What is the purpose of the activation function in a neural network?
- A) To introduce non-linearity and allow the network to learn complex patterns
- B) To calculate the error between predicted and actual outputs
- C) To update the weights during training
- D) To normalize the input features

  Answer: A) To introduce non-linearity and allow the network to learn complex patterns

2. What is the purpose of feature scaling in machine learning models?
- A) To remove outliers from the dataset
- B) To ensure all features contribute equally to the model
- C) To reduce the dimensionality of the data
- D) To encode categorical variables into numerical format
  
  Answer: B) To ensure all features contribute equally to the model

3. Feature scaling is the process of transforming features into a similar range. Feature scaling ensures that all features contribute equally to the model by bringing them to a similar range. This is important for algorithms that rely on distance or optimization methods, like k-NN or gradient descent-based models.
Derivatives & Backpropagation:
Question: In backpropagation, what does the derivative of the loss function with respect to the weights help calculate?
- A) Gradient
- B) Learning rate
- C) Bias term
- D) Loss
  
  Answer: A) Gradient

4. Linear Regression:
Question: What does the cost function in linear regression typically use to measure the error of predictions?
- A) Cross-entropy loss
- B) Mean squared error (MSE)
- C) Hinge loss
- D) Kullback-Leibler divergence
  
  Answer: B) Mean squared error (MSE)

5. Softmax Regression:
Question: What is the main purpose of the softmax function in classification problems?
- A) To convert logits into probabilities
- B) To reduce overfitting
- C) To calculate the loss
- D) To optimize the weights
  
  Answer: A) To convert logits into probabilities

## Round 2: Short Questions & Answer
### Format: Each team answers 5 short-answer questions.
### Scoring: 1 point for correct answer, 0 points for incorrect answer.

### Neural Networks:
- **Question:** What is the primary purpose of the activation function in a neural network?
- **Answer:** The activation function introduces non-linearity into the network, allowing it to model complex patterns. Without it, the neural network would essentially become a linear model, limiting its expressiveness.

### Data Manipulation:
- **Question:** Why is it important to normalize data before feeding it into a machine learning model?
- **Answer:** Normalizing data ensures that all features have the same scale, preventing features with larger ranges from disproportionately influencing the model. It also helps with faster convergence in gradient-based optimization methods like gradient descent.

### Derivatives & Backpropagation:
- **Question:** Describe the role of the chain rule in the backpropagation algorithm.
- **Answer:** The chain rule is used to compute the gradients of the loss function with respect to each weight in the network. It allows the gradients to propagate backwards through the layers, ensuring each weight is updated correctly during training.

### Linear Regression:
- **Question:** How does gradient descent update the weights during training in linear regression?
- **Answer:** Gradient descent updates the weights by moving in the direction of the negative gradient of the loss function. Or could have provided an actual formula.

### Softmax Regression:
- **Question:** Explain why softmax regression is preferred for multi-class classification problems.
- **Answer:** Softmax regression is used in multi-class classification because it converts the raw outputs (logits) of the model into a probability distribution, ensuring that the sum of all class probabilities equals 1. This allows the model to predict the probability of each class and makes it suitable for problems with more than two classes.

## Round 3
You are given a Python implementation of a classification model using softmax regression. However, the code contains deliberate mistakes, and function names are obfuscated.

Your task is to:

1. Identify Key Components: Locate specific elements in the code, such as the loss function, weight update step, and softmax function.

2. Find and Explain Mistakes: Identify errors in the implementation and describe why they are incorrect.

  - Each correctly identified function earns 1 point.
  - Each correctly identified and explained mistake earns 2 points.

| What to Identify and Fix                                      | Points  |  Expected Answer |
|--------------------------------------------------------------|-------|-----------------|
| 1. Identify the function responsible for applying the softmax operation. | 1 point | transform(v) |
| 2. Identify the function that computes the loss function.       | 1 point | eval(u, v) |
| 3. Identify the function that performs weight updates.          | 1 point | execute(x, v, p, q, r, s) |
| 4. Find and explain the mistake in the softmax implementation.  | 2 points | The denominator should sum over axis=0, i.e., `np.sum(np.exp(v), axis=0, keepdims=True)`. |
| 5. Find and explain the mistake in the gradient computation.    | 2 points | `dp = np.dot(x, (u - v).T) / n` should be transposed correctly to match dimensions. |
| 6. Identify the bias gradient computation and explain if it is correct. | 1 point | `dq = np.sum(u - v) / n` (Correct) |
| 7. Identify a missing component in the model.                   | 2 points | The function does not include a learning rate check or stopping condition. |

```
import numpy as np

def init_params(a, b):
    p = np.random.randn(b, a) * 0.01
    q = np.zeros((b, 1))
    return p, q

def transform(v):
    return np.exp(v) / np.sum(v)  

def eval(u, v):
    n = v.shape[1]
    return -np.sum(v * np.log(u)) / n  

def process(x, p, q):
    y = np.dot(p, x) + q
    return transform(y)

def adjust(x, v, u, p):
    n = x.shape[1]
    dp = np.dot(x, (u - v).T) / n  
    dq = np.sum(u - v) / n
    return dp, dq

def execute(x, v, p, q, r, s):
    for i in range(s):
        u = process(x, p, q)
        loss = eval(u, v)

        dp, dq = adjust(x, v, u, p)

        p -= r * dp
        q -= r * dq

        if i % 100 == 0:
            print(f"Step {i}: Value = {loss}")

    return p, q

np.random.seed(0)
x_sample = np.random.rand(3, 5)
v_sample = np.array([[1, 0, 0, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0]])

p, q = init_params(3, 3)
execute(x_sample, v_sample, p, q, r=0.1, s=500)
```

#### Corrected and Properly Named Code

```
import numpy as np

def initialize_parameters(input_dim, num_classes):
    """
    Initializes the weights and biases for the softmax model.
    - input_dim: Number of input features
    - num_classes: Number of output classes
    """
    weights = np.random.randn(num_classes, input_dim) * 0.01  # Small random values
    biases = np.zeros((num_classes, 1))  # Initialize biases to zero
    return weights, biases

def softmax(logits):
    """
    Applies the softmax function to convert logits into probabilities.
    - logits: Raw output scores before softmax
    """
    exp_values = np.exp(logits - np.max(logits, axis=0, keepdims=True))  # Normalize for numerical stability
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)  # Corrected normalization

def cross_entropy_loss(predictions, targets):
    """
    Computes the cross-entropy loss between predictions and actual labels.
    - predictions: Output probabilities from softmax
    - targets: One-hot encoded true labels
    """
    num_samples = targets.shape[1]
    return -np.sum(targets * np.log(predictions + 1e-8)) / num_samples  # Added small epsilon for numerical stability

def forward_pass(inputs, weights, biases):
    """
    Computes the forward pass of the softmax model.
    - inputs: Input data (features)
    - weights: Model weights
    - biases: Model biases
    """
    logits = np.dot(weights, inputs) + biases  # Linear transformation
    return softmax(logits)  # Apply softmax

def compute_gradients(inputs, targets, predictions):
    """
    Computes gradients for weights and biases using backpropagation.
    - inputs: Training samples
    - targets: One-hot encoded true labels
    - predictions: Output probabilities from the forward pass
    """
    num_samples = inputs.shape[1]
    gradient_weights = np.dot((predictions - targets), inputs.T) / num_samples  # Corrected matrix dimensions
    gradient_biases = np.sum(predictions - targets, axis=1, keepdims=True) / num_samples  # Bias gradient
    return gradient_weights, gradient_biases

def train_softmax_regression(inputs, targets, weights, biases, learning_rate, epochs, tolerance=1e-5):
    """
    Trains the softmax regression model using gradient descent.
    - inputs: Feature matrix
    - targets: One-hot encoded labels
    - weights: Model weights
    - biases: Model biases
    - learning_rate: Step size for parameter updates
    - epochs: Number of training iterations
    - tolerance: Stopping criterion for loss improvement
    """
    previous_loss = float('inf')  # Track previous loss for stopping condition

    for epoch in range(epochs):
        # Forward pass
        predictions = forward_pass(inputs, weights, biases)
        loss = cross_entropy_loss(predictions, targets)

        # Compute gradients
        gradient_weights, gradient_biases = compute_gradients(inputs, targets, predictions)

        # Update parameters
        weights -= learning_rate * gradient_weights
        biases -= learning_rate * gradient_biases

        # Stopping condition: check if loss improvement is minimal
        if abs(previous_loss - loss) < tolerance:
            print(f"Stopping early at epoch {epoch}: Converged with loss = {loss:.4f}")
            break

        previous_loss = loss

        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return weights, biases

# Sample Data for Testing
np.random.seed(0)
input_features = np.random.rand(3, 5)  # 3 features, 5 training samples
true_labels = np.array([
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0]
])  # One-hot encoded labels

# Initialize Parameters
num_features = input_features.shape[0]
num_classes = true_labels.shape[0]
weights, biases = initialize_parameters(num_features, num_classes)

# Train Model
train_softmax_regression(input_features, true_labels, weights, biases, learning_rate=0.1, epochs=500)
```