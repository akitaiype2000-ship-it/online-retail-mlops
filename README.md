# Online Retail Customer Segmentation & High-Value Customer Prediction (MLOps)

## Project Overview

This project builds an end-to-end Machine Learning pipeline to identify **High-Value Repeat Customers** from an online retail dataset.

The pipeline cleans raw transactional data, performs customer-level feature engineering, trains multiple machine learning models, evaluates their performance, and selects the best-performing model.

The project follows an MLOps-style modular architecture for scalability and maintainability.

---

## Problem Statement

The business wants to predict which customers are likely to become **High-Value Repeat Customers**.

A customer is considered a High-Value Repeat Customer if:

- Number of purchases > 1
- Total revenue is above the 75th percentile

This prediction helps marketing teams target valuable customers more effectively.

---

## Dataset

**Dataset:** Online Retail Dataset

The dataset contains invoice-level transaction records including:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

---

## Project Workflow

```
Raw Excel Data
        │
        ▼
Data Ingestion
        │
        ▼
Data Validation
        │
        ▼
Data Transformation
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
```

---

## Project Structure

```
online-retail-mlops/

│

├── data/

├── notebook/

├── src/

│   └── components/

│       ├── data_ingestion.py

│       ├── data_validation.py

│       ├── data_transformation.py

│       ├── model_trainer.py

│       └── model_evaluation.py

│

├── test.py

├── requirements.txt

└── README.md
```

---

## Data Cleaning

The following preprocessing steps were performed:

- Combined both Excel sheets
- Renamed inconsistent columns
- Removed missing Customer IDs
- Removed cancelled invoices
- Removed duplicate records
- Removed invalid prices
- Created Revenue feature

---

## Feature Engineering

Customer-level features created:

- Total Revenue
- Number of Purchases
- Purchase Frequency
- Average Order Value
- Recency (days)

---

## Target Variable

A customer is labeled as High-Value Repeat Customer when:

- Number of Purchases > 1
- Total Revenue > 75th Percentile

---

## Models Used

- Logistic Regression
- Random Forest Classifier

---

## Evaluation Metrics

Models were evaluated using:

- Confusion Matrix
- Classification Report
- F1 Score
- ROC-AUC Score
- Cross Validation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## How to Run

Clone the repository

```bash
git clone https://github.com/akitaiype2000-ship-it/online-retail-mlops.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the pipeline

```bash
python test.py
```

---

## Future Improvements

- Configuration Management
- Logging
- Exception Handling
- MLflow Integration
- FastAPI Deployment
- Docker Containerization
- CI/CD Pipeline

---

## Author

Akita