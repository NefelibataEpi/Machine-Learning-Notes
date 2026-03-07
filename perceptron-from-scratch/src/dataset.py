import numpy as np

def make_toy_spam_dataset():
    """
    Toy 'spam' dataset with 2 features:
        x1 = num_links
        x2 = num_caps_words
    
    label:
        +1 = spam
        -1 = ham
    """

    X = np.array([
        [0, 0],
        [0, 2],
        [1, 1],
        [1, 3],
        [2, 2],
        [2, 4],
        [3, 3],
        [3, 5],
        [4, 4],
        [4, 6],
    ], dtype=float)

    y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=int)
    return X, y