import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(X, y, w, b, save_path=None):
    """
    Plot 2D dataset and perceptron decision boundary.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Input features.
    y : np.ndarray of shape (n_samples,)
        Labels in {-1, +1}.
    w : np.ndarray of shape (2,)
        Learned weight vector.
    b : float
        Learned bias.
    save_path : str or None
        If given, save the figure to this path
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)

    # Separate the two classes

    spam = X[y == 1]
    ham = X[y == -1]

    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(ham[:, 0], ham[:, 1], label="HAM (-1)", marker="o")
    plt.scatter(spam[:, 0], spam[:, 1], label="SPAM (+1)", marker="x")

    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b) / w2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_values = np.linspace(x_min, x_max, 200)

    if abs(w[1]) > 1e-12:
        y_values = -(w[0] * x_values + b) / w[1]
        plt.plot(x_values, y_values, label="Decision Boundary")
    else:
        # Special case: vertical line
        x_vertical = -b / w[0]
        plt.axvline(x=x_vertical, label="Decision Boundary")
    
    plt.xlabel("Number of Links")
    plt.ylabel("Number of CAPS Words")
    plt.title("Perceptron Decision Boundary on Toy Spam Dataset")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()