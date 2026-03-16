# E-Commerce-Customer-Churn-Prediction

Predicting customer churn using Google Cloud Vertex AI AutoML on a real-world e-commerce dataset, with business-driven threshold selection to maximize retention impact.

# Overview
Customer acquisition costs five to seven times more than retention, making early identification of at-risk customers a high-value business problem. This project builds a binary classification model to predict churn using 5,630 customer records and 20 behavioral and demographic features, then selects an operating threshold based on the asymmetric cost structure of churn prediction rather than classifier defaults.

# Dataset
- Source: E-Commerce Customer Churn Dataset (Kaggle)
- Records: 5,630
- Features: 20 (tenure, order count, satisfaction score, preferred payment method, days since last order, cashback amount, and more)
- Target: Binary churn label (1 = churned, 0 = retained)


# Tools & Technologies
- Google Cloud Vertex AI AutoML
- Google Cloud Storage
- Sampled Shapley feature attribution


# Methodology
1. Data Preparation
- Removed CustomerID column prior to training to prevent spurious correlations between arbitrary identifiers and churn labels
- Set all feature transformations to Automatic, allowing Vertex AI to determine appropriate encoding and scaling per column type
- Configured training objective as binary classification with Churn as the target variable

2. Model Training
- Trained using Vertex AI AutoML with a 1-node-hour budget and early stopping enabled
- Early stopping triggered before budget exhaustion, indicating model stabilization
- AutoML performed architecture search across multiple candidate configurations, selecting the best-performing model based on validation set performance

3. Evaluation
- Selected precision-recall curve as the primary evaluation tool due to class imbalance in the dataset. ROC AUC can produce misleadingly optimistic results when the negative class dominates, whereas precision-recall curves focus specifically on the positive (churner) class.

<img width="255" height="255" alt="Screenshot 2026-03-16 at 3 24 04 PM" src="https://github.com/user-attachments/assets/ac75a11a-c556-493d-8864-f777ca74bb2c" />

4. Threshold Selection
Rather than using the default threshold of 0.5, I selected 0.46 based on the asymmetric cost structure of churn prediction. Missing a churner (false negative) is substantially more costly than incorrectly flagging a loyal customer (false positive) with a low-cost retention offer. At threshold 0.46, recall for the positive class first reaches 100% while precision remains at 99.6%, making it the optimal operating point on the precision-recall curve.

5. Feature Importance
- Feature attribution was computed using the Sampled Shapley method:

<img width="641" height="669" alt="Screenshot 2026-03-15 at 4 15 44 PM" src="https://github.com/user-attachments/assets/d5a55c89-a3ff-4726-9fac-5d9cf38da796" />

- Customer longevity and complaint history are the strongest predictors of churn risk in this dataset.

# Key Takeaways
- Automated pipelines do not eliminate the need for deliberate feature preparation — excluding CustomerID was a necessary manual step that AutoML could not perform autonomously
- Near-perfect evaluation results (PR AUC = 1.0) warrant critical interpretation, not just celebration; they may reflect a highly separable dataset rather than guaranteed generalization
- Threshold selection is a business decision, not a technical default — the optimal threshold requires knowing the relative cost of each error type in the specific operational context


# References
Google Cloud. (2026). AutoML tabular training. Google Cloud Documentation. https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/train-model

He, X., Zhao, K., & Chu, X. (2021). AutoML: A survey of the state-of-the-art. Knowledge-Based Systems, 212, 106622. https://doi.org/10.1016/j.knosys.2020.106622

Nasir, Suphan. (2015). Customer Relationship Management Strategies in the Digital Era. Customer Relationship Management Strategies in the Digital Era. 1-322.10.4018/978-1-4666-8231-3.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Verma, A. (2021). Ecommerce customer churn analysis and prediction [Data set]. Kaggle. https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
