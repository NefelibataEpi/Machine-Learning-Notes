# Logistic Regression Breast Cancer Classification

## 📌 Project Overview

This project implements a binary classification model using Logistic Regression on the Breast Cancer Wisconsin dataset. The goal is to classify tumors as malignant or benign based on numerical features extracted from digitized images.

---

## 📊 Dataset

* Source: scikit-learn built-in dataset
* Samples: 569
* Features: 30 numerical features
* Classes:

  * Malignant (0)
  * Benign (1)

---

## ⚙️ Methodology

1. Data loading and exploration
2. Train-test split (80/20)
3. Feature standardization using StandardScaler
4. Model training using Logistic Regression
5. Model evaluation using:

   * Accuracy
   * Confusion Matrix
   * Precision, Recall, F1-score

6. Feature importance analysis via model coefficients

---

## 📈 Results

* Accuracy: **0.97**
* Strong performance on both classes
* Minimal misclassification:

  * 2 malignant cases misclassified as benign
  * 1 benign case misclassified as malignant

---

## 🔍 Key Insights

* Features such as **worst texture**, **radius error**, and **concavity** are strong indicators of malignant tumors.
* Features like **compactness** and **symmetry** are associated with benign tumors.
* Logistic Regression provides interpretable coefficients that help understand feature importance.

---

## 📂 Project Structure

```
logistic-regression-breast-cancer/
│
├── logistic_regression.ipynb
├── images/
│   └── confusion_matrix.png
└── README.md
```

---

## 🚀 Future Work

* Try other models (SVM, Random Forest)
* Perform feature selection
* Tune hyperparameters
* Deploy as a simple web app
