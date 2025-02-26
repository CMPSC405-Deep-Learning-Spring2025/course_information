# Midterm Review

The midterm exam will take place during the lab session on February 24th starting at 2:30pm and ending at 4pm. Those wanting to attend a meeting with Karen Skarupski at 2:30pm in Alden, can take the exam either before or after that gathering but need to allocate 1.5 uninterupted hours. 

The exam will consist of 5 multiple-choice questions (3 points each), 5 short-answer questions (5 points each), 2 look at the code segments with 4-5 questions in each (3-5 points for each question) and 2 scenario-based questions (10 points each). Partial points are given when possible for partial answers.

## Steps of NN 
- Inputs, Weights, Bias
- Feed forward
- Forward pass & backward pass
- Backpropagation updates (error, gradient, learning rate)
- Gradient Descent algorithm

## Data Manipulation & Tensors
- Arrays vs. Tensors
- Matrix element-wise operations
- Dot product
- Broadcasting
- Reshaping
- autograd (automatic differentiation)

## Linear Regression
- Linear vs. Non-Linear Models
- Basic idea of fitting a line
- Least squares error
- Mean Squared Error (MSE)
- Gradient Descent for parameter optimization

## Softmax Regression
- Softmax function for classification probabilities
- Cross-Entropy Loss and its interpretation
- Generalization & Regularization (e.g., Weight Decay)

## Building MLP in PyTorch
- Using `nn.Linear` for fully connected layers
- Stacking layers with `nn.Sequential`
- Creating custom models with `nn.Module`

## Multilayer Perceptrons (MLPs)
- XOR problem and why linear models are insufficient
- Universal Approximation Theorem
- Vanishing & Exploding Gradients
- Dropout & Weight Initialization (Xavier, He)
- Activation functions: ReLU, Sigmoid, Tanh

## Scenario-Based Questions

1. Your organization wants to predict how many units of a product will sell next month. You have features such as recent sales, ad budget, and seasonality indicators. The current linear regression model shows signs of underfitting and errors remain high. How would you improve predictions, and what regularization technique you might consider to avoid potential overfitting?

2. You are asked to build a neural network that classifies whether images contain defects in car parts on an assembly line. The dataset is large but has imbalanced classes (many times more "good" parts than "defective" parts). How will you make to ensure robust performance under class imbalance?

