import numpy as np

def make_toy_spam_dataset(n=100, seed=None):
    """
    Generate a synthetic spam dataset

    Features:
        x1 = num_links
        x2 = num_caps_words
    
    Labels:
        +1 = spam
        -1 = ham
    """

    if seed is not None:
        np.random.seed(seed)

    # HAM emails (normal emails)
    ham_links = np.random.normal(loc=1.5, scale=0.8, size=n//2)
    ham_caps = np.random.normal(loc=1.0, scale=0.7, size=n//2)

    ham = np.column_stack((ham_links, ham_caps))
    ham_labels = -np.ones(n//2)

    # SPAM emails
    spam_links = np.random.normal(loc=3.5, scale=0.8, size=n//2)
    spam_caps = np.random.normal(loc=4.5, scale=0.8, size=n//2)

    spam = np.column_stack((spam_links, spam_caps))
    spam_labels = np.ones(n//2)

    # Combine dataset
    X = np.vstack((ham, spam))
    y = np.concatenate((ham_labels, spam_labels))

    return X, y