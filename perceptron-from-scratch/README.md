# Perceptron From Scratch

This project implements the **Perceptron algorithm from scratch** using Python and Numpy.

The implementation is inspired by the Perceptron algorithm taught in **MIT 6.036 - Introduction to Machine Learning**.

---

## What is Perceptron?

Perceptron is one of the earliest machine learning algorithms.
It is a **binary linear classifier** that separates two classes using a line (in 2D) or a hyperplane (in higher dimensions).

The classifier has the form:

$$
h(x) = \text{ sign}(w \cdot x + b)
$$

Where:

- `w` = weight vector
- `b` = bias
- `x` = feature vector

Prediction rule:

- if $w \cdot x + b > 0$ $\to$ class +1
- if $w \cdot x + b \leq 0$ $\to$ class -1


---

## Update Rule

When a sample is misclassified, the perceptron updates its parameters:

$$
\begin{aligned}
w & = w + y \cdot x \\
b & = b + y
\end{aligned}
$$

This moves the decision boundary toward the correct classification.

The algorithm keeps updating until:

- all samples are correctly classified, or
- a maximum number of epochs is reached

---

# Project Structure

```
perceptron-from-scratch
│
├── src
│ ├── perceptron.py     # Perceptron implementation
│ ├── dataset.py        # Synthetic spam dataset generator
│ └── visualization.py  # Decision boundary & training GIF
│
├── results
│ └── decision_boundary.png
│ └── perceptron_training.gif
│
└── train.py            # Training script
└── README.md
```

---

# Dataset

This project uses a **synthetic spam detection dataset**.

Each email is represented by two features:

- `x1` : number of links in the email
- `x2` : number of capitalized words

Labels:

- `+1` = spam
- `-1` = ham (normal email)

The dataset is randomly generated using Gaussian distributions

Example intuition:

Spam emails usually contain:

- more links
- more capitalized words

---

# Training Result

Example decision boundary learned by the Perceptron:

<img src="results/decision_boundary.png" alt="Decision Boundary" style="zoom:50%;" />

The classifier successfully separated the two classes.

---

# Training Process Visualization

The training process can also be visualized as an animation.

Each frame represents one updates of the Perceptron parameters.

<img src="results/perceptron_training.gif" alt="Decision Boundary" style="zoom:50%;" />

This animation shows how the decision boundary gradually moves until it separates the data.

---

# Example output

```
Learned w = [-0.47 6.94]
Learned b = -17.0
Updates = 37
Train acc = 1.0
```