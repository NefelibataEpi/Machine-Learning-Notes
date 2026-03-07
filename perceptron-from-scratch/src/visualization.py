import os
import tempfile

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(X, y, w, b, save_path=None, show=True, title=None):
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

    # Fix figure size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    plt.scatter(ham[:, 0], ham[:, 1], label="HAM (-1)", marker="o")
    plt.scatter(spam[:, 0], spam[:, 1], label="SPAM (+1)", marker="x")

    # Fix the range of axis 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Plot decision boundary: w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b) / w2
    x_values = np.linspace(x_min, x_max, 200)

    if abs(w[1]) > 1e-12:
        y_values = -(w[0] * x_values + b) / w[1]
        ax.plot(x_values, y_values, label="Decision Boundary")
    elif abs(w[0] > 1e-12):
        # Special case: vertical line
        x_vertical = -b / w[0]
        ax.axvline(x=x_vertical, label="Decision Boundary")
    else:
        # w = [0,0] with no boundary
        pass

    
    ax.set_xlabel("Number of Links")
    ax.set_ylabel("Number of CAPS Words")
    ax.set_title("Perceptron Decision Boundary on Toy Spam Dataset")
    ax.legend()
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

def save_training_gif(X, y, history, gif_path, duration=0.4):
    """
    Save perceptron training process as a GIF

    Parameters
    ----------
    X, y : dataset
    history : list of (w, b)
        Saved parameter states during training
    gif_path : str
        Output GIF path
    duration : float
        Duration per frame in seconds
    """

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    frame_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for step, (w, b) in enumerate(history):
            frame_path = os.path.join(temp_dir, f"frame_{step:03d}.png")

            title = f"Perceptron Training - Step {step}"
            plot_decision_boundary(
                X, y, w, b,
                save_path=frame_path,
                show=False,
                title=title
            )

            frame_paths.append(frame_path)
        
        images = [imageio.imread(path) for path in frame_paths]
        imageio.mimsave(gif_path, images, duration=duration)