import numpy as np

class LinearRegression:
    def __init__(self, input_size, lr=0.01):
        self.weights = np.zeros(input_size + 1)
        self.lr = lr

    def predict(self, x):
        return self.weights.dot(np.insert(x, 0, 1))

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            for i in range(y.shape[0]):
                x = np.insert(X[i], 0, 1)
                y_pred = self.predict(X[i])
                error = y[i] - y_pred
                self.weights += self.lr * error * x

# Example usage
if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 3, 4, 5])
    model = LinearRegression(input_size=1)
    model.fit(X, y)
    print("Predictions:", [model.predict(x) for x in X])