import os
import sys
import numpy as np

# Make `src/` importable no matter where you run from
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.append(SRC_DIR)

from perceptron import Perceptron
from dataset import make_toy_spam_dataset
from visualization import plot_decision_boundary, save_training_gif

if __name__ == "__main__":
    X, y = make_toy_spam_dataset(n=100, seed=0)

    model = Perceptron().fit(X, y, epochs=50)

    print("Learned w =", model.w)
    print("Learned b =", model.b)
    print("Updates   =", model.updates)
    print("Train acc =", model.score(X, y))

    new_X = np.array([
        [0, 1],     # likely ham
        [5, 8],     # likely spam
        [2, 3],     # borderline
    ], dtype=float)

    new_pred = model.predict(new_X)
    print("\nNew prediction:")
    for x, p in zip(new_X, new_pred):
        label = "SPAM" if p == 1 else "HAM"
        print(f" x={x} -> {label}")

    # Save final decision boundary image
    image_path = os.path.join(RESULTS_DIR, "decision_boundary.png")
    plot_decision_boundary(X, y, model.w, model.b, save_path=image_path)
    print(f"\nFigure saved to: {image_path}")

    # Save training GIF
    gif_path = os.path.join(RESULTS_DIR, "perceptron_training.png")
    save_training_gif(X, y, model.history, gif_path=gif_path, duration=2.0)
    print(f"Training GIF saved to: {gif_path}")