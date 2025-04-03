import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * 0.1 for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(size, 1) * 0.1 for size in layer_sizes[1:]]

    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
            activations.append(x)
        return activations

    def backward(self, activations, y):
        deltas = [activations[-1] - y] # error of outer layer 
        for i in range(self.num_layers - 2, 0, -1):
            deltas.append(np.dot(self.weights[i].T, deltas[-1]) * sigmoid_derivative(activations[i]))
        deltas.reverse()
        return deltas

    def update_weights(self, activations, deltas, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(deltas[i], activations[i].T)
            self.biases[i] -= learning_rate * deltas[i]

    def train(self, X, Y, epochs, learning_rate):
        mse_list = []

        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x, y = x.reshape(-1, 1), y.reshape(-1, 1) # convert to col vector (-1 means infer)
                activations = self.forward(x)
                deltas = self.backward(activations, y)
                self.update_weights(activations, deltas, learning_rate)
            
            if epoch % 100 == 0:
                mse = np.mean((self.predict(X) - Y) ** 2)
                mse_list.append(mse)
            if epoch % 1000 == 0:
                mse = np.mean((self.predict(X) - Y) ** 2)
                print(f"Epoch {epoch}, Loss: {mse:.4f}")
        
        return mse_list 

    def predict(self, X):
        return np.array([self.forward(x.reshape(-1, 1))[-1].flatten() for x in X])

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
