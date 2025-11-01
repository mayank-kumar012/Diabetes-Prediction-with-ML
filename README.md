# 🩺 Diabetes Prediction using Machine Learning

## ✨ Project Overview

This project implements a machine learning model to predict the likelihood of a person having diabetes based on several diagnostic measurements. The model is trained using the **Pima Indians Diabetes Dataset** and deployed as an interactive web application using **Streamlit**, allowing users to input their health data and receive an instant prediction.

### Key Components:

  * **Data Analysis & Model Training:** Performed in a Jupyter notebook (`Diabetes_model_training.ipynb`).
  * **Model:** **Support Vector Classifier (SVC)** with a linear kernel.
  * **Deployment:** Interactive web application built with **Streamlit** (`app.py`).

-----

## 💾 Dataset Details

The model is trained on the **Pima Indians Diabetes Dataset**, which contains diagnostic measurements from female patients at least 21 years old of Pima Indian heritage.

| Feature | Description |
| :--- | :--- |
| **Pregnancies** | Number of times pregnant. |
| **Glucose** | Plasma glucose concentration a 2 hours in an oral glucose tolerance test. |
| **BloodPressure** | Diastolic blood pressure (mm Hg). |
| **SkinThickness** | Triceps skin fold thickness ($\text{mm}$). |
| **Insulin** | 2-Hour serum insulin ($\text{mu U/ml}$). |
| **BMI** | Body mass index (weight in $\text{kg}/(\text{height in }\text{m})^2$). |
| **DiabetesPedigreeFunction** | Diabetes pedigree function. |
| **Age** | Age (years). |
| **Outcome** | Class variable (0: Not diabetic, 1: Diabetic) - **Target Variable**. |

## 🛠️ Model Training and Evaluation

The Support Vector Classifier (SVC) was used for this binary classification problem.

### Steps in `Diabetes_model_training.ipynb`:

1.  **Data Loading & Initial Exploration:** Loaded the data and examined its shape (`768` rows, `9` columns) and statistical description.
2.  **Class Distribution:** Checked the balance of the target variable (`Outcome`):
      * **0 (Not Diabetic):** 500 records
      * **1 (Diabetic):** 268 records
3.  **Feature Separation:** The dataset was split into features ($\mathbf{X}$) and the target ($\mathbf{Y}$).
4.  **Train-Test Split:** The data was split into **80% training** and **20% testing** sets, using **stratified sampling** to maintain the same proportion of classes in both sets.
5.  **Model Training:** A **Support Vector Classifier ($\text{SVC}$) with a linear kernel** was trained on the training data.
6.  **Evaluation Metrics:** The model was evaluated using accuracy, precision, recall, and F1-score on both the training and test sets.

### Model Performance:

| Metric | Training Data | Test Data |
| :--- | :--- | :--- |
| **Accuracy Score** | $0.7834$ | $0.7727$ |
| **Precision** | - | $0.7568$ |
| **Recall** | - | $0.5185$ |
| **F1-score** | - | $0.6154$ |

The close accuracy between the training and test sets suggests the model is **well-generalized** and **not severely overfitted**.


-----

## 🚀 Deployment (Streamlit App)

The trained model is saved as `diabetes_model.sav` and is used by the `app.py` script to create a web-based prediction tool.

### Prerequisites

To run the Streamlit app locally, you need the following Python libraries:

```bash
pip install pandas scikit-learn streamlit
```

### How to Run the App

1.  Clone this repository:
    ```bash
    git clone [Your-Repo-Link]
    cd [Your-Repo-Name]
    ```
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open automatically in your web browser.
