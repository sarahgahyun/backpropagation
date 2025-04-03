import matplotlib.pyplot as plt

# Plotting
def plot_mse(mses, desc):
    plt.plot(mses)
    plt.xlabel('Epochs (per 100)')
    plt.ylabel('Mean squared error (MSE)')
    plt.title('MSE Values Throughout Training for ' + desc)
    plt.grid(True)
    plt.show()