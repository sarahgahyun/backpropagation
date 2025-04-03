# AND-specific training
def train_and(epochs, learning_rate):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [0], [0], [1]])  
    
    nn = NeuralNetwork([2, 3, 1])  # 2 input neurons, 3 hidden, 1 output
    nn.train(X, Y, epochs, learning_rate)

    predictions = nn.predict(X)
    print("\nFinal Predictions")
    for x, p in zip(X, predictions):
        print(f"Input: {x} => Predicted Output: {p[0]:.4f}")