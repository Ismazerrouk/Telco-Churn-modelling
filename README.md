# Telco Churn Modelling 📊📉

This project aims to analyze and predict customer churn for a telecommunications company using machine learning techniques. 🧠💡 The dataset used for this project is the Telco Customer Churn dataset. 📁

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)


## Introduction
Customer churn is a critical issue for telecommunications companies. This project focuses on identifying the factors that influence customer churn and building predictive models to classify customers as churned or not churned.

## Dataset
The dataset used in this project is the Telco Customer Churn dataset, which contains information about customer demographics, account information, and services subscribed. The dataset includes the following columns:
- `customerID`: Customer ID
- `gender`: Whether the customer is a male or a female
- `SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)
- `Partner`: Whether the customer has a partner or not (Yes, No)
- `Dependents`: Whether the customer has dependents or not (Yes, No)
- `tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Whether the customer has phone service or not (Yes, No)
- `MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)
- `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)
- `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)
- `OnlineBackup`: Whether the customer has online backup or not (Yes, No, No internet service)
- `DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
- `TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)
- `StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)
- `StreamingMovies`: Whether the customer has streaming movies or not (Yes, No, No internet service)
- `Contract`: The contract term of the customer (Month-to-month, One year, Two year)
- `PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)
- `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer
- `Churn`: Whether the customer churned or not (Yes, No)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Ismazerrouk/Telco-Churn-modelling.git
   cd Telco-Churn-modelling ```

2. Create and activate a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install the required dependencies:
```sh
pip install -r requirements.txt
```
## Usage
To run the analysis and generate the results, execute the Jupyter Notebook:

1. Start Jupyter Notebook:
```sh
jupyter notebook
```

2.Open AN.ipynb and run the cells to perform the analysis and visualize the results.

## Features 

- Data preprocessing and exploration 🔍
- Feature engineering 🛠️
- Model training and evaluation 🏋️‍♂️
- Feature importance analysis 📈
- Data visualization 🎨

## Results
The analysis reveals that customers with shorter tenure and month-to-month contracts are more likely to churn. The Random Forest model performed the best in predicting customer churn with an accuracy of 85.36%. The most important feature influencing churn is MonthlyCharges

## Feature Importance
The top 10 most important features identified by the Random Forest model are:

1. MonthlyCharges
2. TotalCharges
3. tenure
4. Contract
5. InternetService
6. OnlineSecurity
7. TechSupport
8. PaymentMethod
9. StreamingTV
10. StreamingMovies

## Dashboard (Streamlit)
Launch an interactive retention dashboard that reuses the Telco churn dataset:

1. Ensure the Telco CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is available locally in this folder (or note its path).
2. Install minimal deps:
   ```sh
   pip install streamlit pandas plotly
   ```
3. Run the app:
   ```sh
   streamlit run streamlit_app.py
   ```
4. In the sidebar, point `Data path` to the CSV if it is not in the project root. Use the filters to slice by contract type, services, tenure, and billing. The dashboard surfaces churn KPIs, risk tables, and the existing visuals from the analysis.
