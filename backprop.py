"""
PSYCH 420: Backpropagation

To customize number of epochs (training cycles) and/or the alpha (learning rate), please add arguments when running. 
The below are using the default values: 
    -e=10000 --epochs=10000
    -a=0.1 --alpha=0.1
"""
import numpy as np
import argparse
import sys

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

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
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                x, y = x.reshape(-1, 1), y.reshape(-1, 1) # convert to col vector (-1 means infer)
                activations = self.forward(x)
                deltas = self.backward(activations, y)
                self.update_weights(activations, deltas, learning_rate)
            
            if epoch % 1000 == 0:
                loss = np.mean((self.predict(X) - Y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.array([self.forward(x.reshape(-1, 1))[-1].flatten() for x in X])

# XOR-specific training
def train_xor(epochs, learning_rate):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    
    nn = NeuralNetwork([2, 3, 1])  # 2 input neurons, 3 hidden, 1 output
    nn.train(X, Y, epochs, learning_rate)
    
    predictions = nn.predict(X)
    print("Final Predictions:")
    for x, p in zip(X, predictions):
        print(f"Input: {x} => Predicted Output: {p[0]:.4f}")

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    return parser.parse_args()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            sys.exit(1)
    args = parse_arguments()
    
    train_xor(args.epochs, args.alpha)

if __name__ == "__main__":
    main()
