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

---

# Training Result

After training, the perceptron learns a decision boundary that separates spam from normal emails.

Example decision boundary:

<img src="results/decision_boundary.png" alt="Decision Boundary" style="zoom:50%;" />

---

# Example output

```
Learned w = [5. 1.]
Learned b = -13.0
Train acc = 1.0
```