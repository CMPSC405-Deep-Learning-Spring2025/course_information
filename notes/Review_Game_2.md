# Review Game 2

## Round 1: Multiple Choice Questions
### Format: Each team answers 5 multiple-choice questions. 
### Scoring: 1 points for correct answers, 0 points for incorrect answers.

### Questions:
1. What is the primary advantage of using a multilayer perceptron (MLP) over a linear model? (D2L Chapter 5: Multilayer Perceptrons)

A) MLPs can approximate complex non-linear functions
B) MLPs require fewer parameters than linear models
C) MLPs do not require activation functions
D) MLPs can only be used for binary classification

Answer: A) MLPs can approximate complex non-linear functions

2. In the forward pass of an MLP, what operation is applied to the weighted sum of inputs at each layer? (D2L Chapter 5: Multilayer Perceptrons)

A) Softmax
B) Activation function
C) Mean squared error
D) Backpropagation

Answer: B) Activation function

3. Why is weight initialization important in deep learning models? (D2L Chapter 12: The Builders Guide to Training Deep Networks)

A) It speeds up training and prevents vanishing or exploding gradients
B) It eliminates the need for optimization algorithms
C) It ensures that all neurons have the same initial output
D) It prevents overfitting by reducing model capacity

Answer: A) It speeds up training and prevents vanishing or exploding gradients

4. What is the primary function of the backpropagation algorithm? (D2L Chapter 5: Multilayer Perceptrons)

A) To compute the gradient of the loss function with respect to each weight
B) To randomly adjust weights until the model works
C) To generate new training data
D) To evaluate model performance on a test set

Answer: A) To compute the gradient of the loss function with respect to each weight

5. Why do we use ReLU instead of sigmoid in hidden layers of deep networks? (D2L Chapter 6: Computation)

A) ReLU prevents vanishing gradients by maintaining a positive derivative for positive inputs
B) Sigmoid is computationally faster
C) ReLU always produces better results
D) Sigmoid is only used in classification problems

Answer: A) ReLU prevents vanishing gradients by maintaining a positive derivative for positive inputs

## Round 2: Short Questions & Answer
### Format: Each team answers 5 short-answer questions.
### Scoring: 1 point for correct answer, 0 points for incorrect answer.

1. Linear Classification: What is the role of the normalization factor in softmax regression? (D2L Chapter 5: Softmax Regression)

Answer: The normalization factor ensures that the output probabilities sum to one, making it a valid probability distribution for multi-class classification.

2. Multilayer Perceptrons: How does dropout improve neural network generalization? (D2L Chapter 5: Multilayer Perceptrons)

Answer: Dropout randomly deactivates neurons during training, forcing the network to learn redundant and more robust features, reducing overfitting.

3. Backpropagation: How does automatic differentiation simplify the training process? (D2L Chapter 6: Computation)

Answer: Automatic differentiation computes gradients efficiently by keeping track of operations in a computational graph, enabling backpropagation without manually deriving derivatives.

4. Custom Layers: Why might you implement a custom layer instead of using built-in PyTorch layers? (D2L Chapter 6: Computation)

Answer: Custom layers allow for specialized operations not available in built-in layers, enabling greater flexibility in designing architectures for specific tasks.

5. Vanishing Gradients: How does the vanishing gradient problem affect deep networks, and what techniques mitigate it? (D2L Chapter 5: Multilayer Perceptrons)

Answer: Vanishing gradients slow down training by making weight updates negligible. Using activation functions like ReLU and employing batch normalization can help mitigate this issue.

## Round 3

You are given a PyTorch implementation of an MLP. However, the code contains deliberate mistakes.

Your task is to:

1. Identify Key Components – Locate specific elements in the code, such as activation functions, loss function, and weight update steps.
2. Find and Explain Mistakes – Identify errors in the implementation and describe why they are incorrect.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Added softmax activation
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)  # Added softmax activation
        return x

model = MLP(3, 5, 3)
criterion = nn.CrossEntropyLoss()  # Corrected loss function
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

| What to Identify and Fix                              | Points  | Detailed Explanation                                                                                     |
|-------------------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------|
| Identify the activation function used in the model    | 1 point| The model correctly uses ReLU after the first layer, which helps introduce non-linearity.                 |
| Identify the function that computes the loss function | 1 point| The loss function used is MSELoss, which is incorrect for classification. The correct choice is nn.CrossEntropyLoss(). |
| Identify the function that performs weight updates    | 1 point| The optimizer optim.SGD updates model parameters using gradients computed from the loss.                  |
| Add missing softmax activation in the output layer    | 2 points| The final layer should include nn.Softmax(dim=1) to properly convert logits into probabilities.            |
| Fix the incorrect loss function                       | 2 points| MSELoss should be replaced with CrossEntropyLoss, which is appropriate for classification problems.       |
| Add missing backpropagation step in training          | 2 points| The model lacks a .backward() call to compute gradients after loss calculation.                           |


