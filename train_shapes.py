import numpy as np 
from network import NeuralNetwork
from plot import plot_mse

def train_shapes(epochs, learning_rate): 
    X = np.array([square, square2, triangle, cross, triangle2, cross2, cross3, square3, triangle3])
    # Quick output encoding: let it be 1 for square, triangle, cross, respectively
    Y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])  
    X2 = np.array([square4, triangle4, cross4])
    Y2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    nn = NeuralNetwork([16, 8, 3])  # 16 input (4x4), 8 hidden, 3 output
    mses = nn.train(X, Y, epochs, learning_rate)

    predictions = np.round(nn.predict(X2), 3)
    print("\nFinal Predictions")
    for i, (p, y) in enumerate(zip(predictions, Y2)):
        print(f"Predicted {p} <= Expected {y}")
    print()

    plot_mse(mses, "Shapes")


# Flattened 4x4 grayscale images of shapes
# SQUARES
square = np.array([  
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1]
]).flatten()

square2 = np.array([  
    [1, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0]
]).flatten()

square3 = np.array([  
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1]
]).flatten()

square4 = np.array([  
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [0, 0, 0, 0]
]).flatten()

# TRIANGLES
triangle = np.array([
    [0, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1]
]).flatten()

triangle2 = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
]).flatten()

triangle3 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0]
]).flatten()

triangle4 = np.array([
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
]).flatten()

# CROSSES
cross = np.array([
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
]).flatten()

cross2 = np.array([
    [0, 0, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
]).flatten()

cross3 = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0]
]).flatten()

cross4 = np.array([
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 0, 0]
]).flatten()