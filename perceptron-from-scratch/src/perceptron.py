import numpy as np

class Perceptron:
    """
    Perceptron binary classifier (labels must be -1 / +1)

    Decision rule:
        sign(w*x + b)
    Update rule:
        w = w + y * x
        b = b + y
    """

    def __init__(self):
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.updates: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        self.updates = 0

        for epoch in range(epochs):
            changed = False
            for i in range(n):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin <= 0:
                    self.w = self.w + y[i] * X[i]
                    self.b = self.b + y[i]
                    self.updates += 1
                    changed = True
            
            if not changed:
                break

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        
        X = np.asarray(X, dtype=float)
        scores = X @ self.w + self.b
        return np.where(scores > 0, 1, -1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=int)
        pred = self.predict(X)
        return float((pred == y).mean())