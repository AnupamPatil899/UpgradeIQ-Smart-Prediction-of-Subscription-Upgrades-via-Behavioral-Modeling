# UpgradeIQ: Smart Prediction of Subscription Upgrades via Behavioral Modeling

## Introduction
UpgradeIQ is a machine learning pipeline designed to analyze customer behavioral data and predict subscription changes (such as churn or potential upgrades). By leveraging customer engagement metrics, viewing habits, and support interactions, this project provides actionable insights to help service providers improve retention and tailor subscription offerings.

## Key Features
- **Extensive Feature Engineering:** Calculates advanced metrics such as Engagement Score, Support Intensity, Recent Activity Drop, and Total Risk Score.
- **Handling Imbalanced Data:** Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalances in the dataset.
- **Robust Modeling:** Implements and compares Logistic Regression and XGBoost classifiers.
- **Comprehensive Evaluation:** Evaluates models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and AUPRC. It also generates visualizations for ROC curves, Precision-Recall curves, and Feature Importances.
- **Pipeline Export:** Automatically saves trained models, scalers, and encoders via `joblib` for easy deployment.

## Installation
Clone the repository:
```bash
git clone https://github.com/AnupamPatil899/UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling.git
cd UpgradeIQ-Smart-Prediction-of-Subscription-Upgrades-via-Behavioral-Modeling
```

Install the required dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost imbalanced-learn joblib
```

## Dataset
This project requires train and test datasets containing user behavior features such as:
- `MonthlyCharges`
- `ViewingHoursPerWeek`
- `AccountAge`
- `ContentDownloadsPerMonth`
- `SupportTicketsPerMonth`
- `UserRating`
- `SubscriptionType`
- `Churn` (Target Variable)

*Note: You may need to adapt the file paths in `upgradeiq.py` to match the location of your datasets.*

## Usage
Run the main script to start the training and evaluation process:
```bash
python upgradeiq.py
```

The script will:
1. Load and preprocess the training data.
2. Perform feature engineering and create composite risk/engagement scores.
3. Transform categorical variables using One-Hot Encoding.
4. Balance the training data using SMOTE.
5. Train Logistic Regression and XGBoost models using cross-validation.
6. Evaluate the models and generate ROC & AUPRC curves.
7. Save the best models and pipeline tools to `models/trained_models/`.

## Contributing
Contributions and suggestions are always welcome! Feel free to open an issue or submit a pull request with new ideas, features, or bug fixes.

## License
[Add your specific license here, e.g., MIT, GPLv3]
