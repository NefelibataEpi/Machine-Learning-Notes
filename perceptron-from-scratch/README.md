# Perceptron From Scratch

This project implements the **Perceptron algorithm from scratch** using Python and Numpy.

The implementation is inspired by the **Perceptron algorithm taught in MIT 6.036**.

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

If $w \cdot x + b > 0$, the prediction is **+1**.
Otherwise the prediction is **-1**.

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

---

# Project Structure

```
perceptron-from-scratch
│
├── src
│ ├── perceptron.py # Perceptron implementation
│ ├── dataset.py # Toy spam dataset generator
│ └── visualization.py # Decision boundary plotting
│
├── results
│ └── decision_boundary.png
│
└── train.py # Training script
```

---

# Dataset

This project uses a **toy spam detection dataset**.

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