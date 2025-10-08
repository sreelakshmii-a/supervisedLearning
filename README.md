## Supervised Learning INT395 Hands-on Repository

Welcome to the **supervisedLearning** repository, a hands-on resource for **Supervised Learning (INT395)**. This repository contains practical implementations and explorations of fundamental and advanced supervised machine learning concepts, algorithms, and techniques.

This resource is designed to complement the course material by providing practical code examples for data manipulation, model building, and evaluation.

---

## 📚 Table of Contents

* [Overview](#-overview)
* [Practical List](#-practical-list)
* [Textbooks](#-textbooks)
* [References](#-references)
* [Setup and Usage](#-setup-and-usage)

---

## 💡 Overview

This repository focuses on key areas of supervised learning, including:

* **Data Preprocessing and Visualization** 📊: Using libraries like **Pandas** and **Seaborn** to clean, prepare, and understand data.
* **Feature Engineering and Selection** ⚙️: Techniques for creating meaningful features and reducing dimensionality (**PCA**).
* **Model Implementation** 💻: Building and applying various models such as **Logistic Regression**, **Decision Trees**, **k-NN**, **SVM**, **Naïve Bayes**, and **Ensemble Methods** (Random Forest, AdaBoost, Gradient Boosting).
* **Model Evaluation and Tuning** 📈: Assessing model performance using metrics (Precision, Recall, F1-score, ROC-AUC) and optimizing hyperparameters (**Grid Search**, **Random Search**).
* **Regression** 📉: Implementing different regression models (Linear, Polynomial, Ridge) and **Time Series Forecasting** (ARIMA/SARIMA).
* **Deployment** 🚀: Utilizing **Pickle** and **Streamlit** for basic model deployment.

---

## 📝 Practical List

The following practical exercises are included in this repository:

1.  Write a program to explore and visualize a dataset using **Pandas** and **Seaborn**.
2.  Write a program to preprocess data by handling **missing values, outliers, scaling, and encoding**.
3.  Write a program to perform **feature engineering** and **feature selection** using statistical methods and **PCA**.
4.  Write a program to split data into training and testing sets and apply **k-fold cross-validation**.
5.  Write a program to implement **logistic regression** for binary classification.
6.  Write a program to build a **decision tree classifier** and evaluate it using a **confusion matrix**.
7.  Write a program to compare the performance of **k-NN, SVM, and Naïve Bayes** classifiers.
8.  Write a program to evaluate a classification model using **precision, recall, F1-score, and ROC-AUC**.
9.  Write a program to implement **Random Forest classification** using **Bagging** technique.
10. Write a program to apply **boosting algorithms** like **AdaBoost and Gradient Boosting** for classification.
11. Write a program to **tune hyperparameters** using **Grid Search and Random Search** methods.
12. Write a program to implement **linear, polynomial, and ridge regression** models for prediction.
13. Write a program to forecast **time series data** using **ARIMA or SARIMA** models.
14. Write a program to **deploy a machine learning model** using **Pickle and Streamlit**.

---

## 📖 Textbooks

The primary textbook for reference:

1.  **MACHINE LEARNING-I** by **CHANDRA S.S, VINOD**, PHI Learning

---

## 📎 References

For supplementary reading and deeper understanding:

1.  **MACHINE LEARNING** by **ETHEM ALPAYDIN**, MIT Press

---

## 🛠️ Setup and Usage

To run the practicals in this repository, you will typically need the following libraries. It's recommended to use a virtual environment.

### Prerequisites

* Python (3.x recommended)
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/alensomaxx/supervisedLearning.git](https://github.com/alensomaxx/supervisedLearning.git)
    cd supervisedLearning
    ```

2.  **Install the necessary packages:**
    (A `requirements.txt` file is usually included in a production repository, but based on the practicals, you'll need the following):
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn statsmodels streamlit
    ```

3.  **Run the individual practicals:**
    Navigate to the directory of the practical you wish to run and execute the Python script.

    *Example:*
    ```bash
    python practical_1_data_viz.py
    ```

Happy Learning! 🚀
