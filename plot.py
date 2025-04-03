import matplotlib.pyplot as plt

# Plotting
def plot_mse(mses):
    plt.plot(mses)
    plt.xlabel('Epochs (per 100)')
    plt.ylabel('Mean squared error (MSE)')
    plt.title('MSE Values Throughout Training')
    plt.grid(True)
    plt.show()