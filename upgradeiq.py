#libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#data
train_df = pd.read_csv('/content/drive/MyDrive/UpgradeIQ_Subscription/Datasets/Train_test_all/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/UpgradeIQ_Subscription/Datasets/Train_test_all/test.csv')

#Preprocessing
train_df.drop(columns=['CustomerID'],inplace=True)
cat_cols=[i for i in train_df.columns if train_df[i].dtype=='object']
num_cols=[i for i in train_df.columns if train_df[i].dtype!='object']

#FeatureEngineering
train_df['valueperhourmonthly']=train_df['MonthlyCharges']/(train_df['ViewingHoursPerWeek']*4)
train_df['avgmonthlyusage']=(train_df['ViewingHoursPerWeek']*4)/train_df['AccountAge']
train_df['EngagementScore'] = (train_df['ContentDownloadsPerMonth'] + train_df['WatchlistSize'] + (train_df['ViewingHoursPerWeek']*4)) / 3
train_df['SupportIntensity'] = train_df['SupportTicketsPerMonth'] / (train_df['AccountAge'] + 1)
train_df['RecentActivityDrop'] = ((train_df['ViewingHoursPerWeek'] < train_df['ViewingHoursPerWeek'].quantile(0.25)) &(train_df['AccountAge'] > 6)).astype(int)
train_df['Highwatching']=(train_df['ViewingHoursPerWeek']>train_df['ViewingHoursPerWeek'].quantile(0.75).astype(int))
train_df['HighSatisfaction'] = (train_df['UserRating'] >= 4.0).astype(int)
train_df['ChargesToAge_Ratio'] = train_df['MonthlyCharges'] / (train_df['AccountAge'] + 1)
train_df['EngagementSatisfaction'] = train_df['ViewingHoursPerWeek'] * train_df['UserRating']

#Risk scores
train_df['Low_view_monthly']=((train_df['ViewingHoursPerWeek']*4)<(train_df['ViewingHoursPerWeek']*4).quantile(0.25).astype(int))
train_df['Low_view_session']=((train_df['ViewingHoursPerWeek']*4)<(train_df['ViewingHoursPerWeek']*4).quantile(0.25).astype(int))
train_df['LowSatisfaction'] = (train_df['UserRating'] <= 2.0).astype(int)
train_df['HighSupport'] = (train_df['SupportTicketsPerMonth'] > train_df['SupportTicketsPerMonth'].quantile(0.75)).astype(int)
train_df['RecentActivityDrop'] = ((train_df['ViewingHoursPerWeek'] < train_df['ViewingHoursPerWeek'].quantile(0.25)) &
                              (train_df['AccountAge'] > 6)).astype(int)
train_df['Total_risk_score']=train_df['Low_view_monthly']+train_df['Low_view_session']+train_df['LowSatisfaction']+train_df['HighSupport']+train_df['RecentActivityDrop']

train_df.replace({'SubscriptionType':{'Basic':0,'Standard':1,'Premium':2}},inplace=True)

cat_cols=[i for i in train_df.columns if train_df[i].dtype=='object']
num_cols=[i for i in train_df.columns if train_df[i].dtype!='object']

y = train_df['Churn']
X = train_df.drop(columns='Churn')

cat_cols=[i for i in X.columns if train_df[i].dtype=='object']
num_cols=[i for i in X.columns if train_df[i].dtype!='object']

#Transforming cat cols into num cols
from sklearn.preprocessing import OneHotEncoder
OHEncoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = OHEncoder.fit_transform(X)

#Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,test_size=0.2,random_state=42,stratify=y)

#Handle imbalance data
from imblearn.over_sampling import SMOTE
sampler = SMOTE(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

#Log_model
log_model=LogisticRegression(random_state=42, max_iter=1000,n_jobs=-1)
cv_scores = cross_val_score(log_model,X_resampled,y_resampled, cv=5, scoring='roc_auc')
print(f"{LogisticRegression}: ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
log_model.fit(X_resampled,y_resampled)

#XGB model
xgb_model = XGBClassifier(random_state=42,eval_metric='logloss',n_jobs=-1)
print("Starting XGBoost cross-validation...")
cv_scores = cross_val_score(xgb_model, X_resampled, y_resampled, cv=5, scoring='roc_auc')
print("Cross-validation finished.")
print(f"XGBoost: ROC-AUC = {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
print("\nFitting final XGBoost model on all data...")
xgb_model.fit(X_resampled, y_resampled)
print("Final model is trained and ready.")

models = {#'Logistic Regression': log_model,
          'XGBoost': xgb_model}

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

def evaluate_models(models, X_test, y_test):

    results = {}

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'auprc': average_precision_score(y_test, y_prob)
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC: {results[name]["roc_auc"]:.3f})')

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(recall, precision, label=f'{name} (AUPRC: {results[name]["auprc"]:.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', label='No Skill')
    axes[0].set_title('ROC Curves')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    axes[1].set_title('Precision-Recall Curves')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()

    results_df = pd.DataFrame(results).T
    best_model_name = results_df['roc_auc'].idxmax()
    best_model_obj = models[best_model_name]

    if hasattr(best_model_obj, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': best_model_obj.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        axes[2].barh(feature_importance['feature'], feature_importance['importance'])
        axes[2].set_title(f'Top 10 Features for {best_model_name}')
        axes[2].invert_yaxis()
    else:
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, f'{best_model_name}\nhas no feature importances',
                     ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    # --- Print summary and final result ---
    print("\nModel Performance Summary:")
    print(results_df.round(4))

    print(f"\nBest model selected: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f})")


    return results_df, best_model_obj

results_summary, best_model = evaluate_models(models, X_test, y_test)

import os
import joblib
os.chdir('/content/drive/MyDrive/UpgradeIQ_Subscription/Datasets/Train_test_all')

def save_model_pipeline(model, scaler, ohe,model_path='models/trained_models/log_model_1/'):
    """Saves the complete model pipeline including encoders and scaler."""
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, f'{model_path}/best_model.pkl')
    joblib.dump(scaler, f'{model_path}/scaler.pkl')
    joblib.dump(ohe, f'{model_path}/one_hot_encoder.pkl')

    print(f"Model pipeline components saved to {model_path}")
save_model_pipeline(log_model, sampler, OHEncoder)

def save_model_pipeline(model, scaler, ohe, model_path='models/trained_models/xgb_model_1/'):
    """Saves the complete model pipeline including encoders and scaler."""
    os.makedirs(model_path, exist_ok=True)

    joblib.dump(model, f'{model_path}/best_model.pkl')
    joblib.dump(scaler, f'{model_path}/scaler.pkl')
    joblib.dump(ohe, f'{model_path}/one_hot_encoder.pkl')

    print(f"Model pipeline components saved to {model_path}")

save_model_pipeline(xgb_model, sampler, OHEncoder)

#Real-world-data
#test_df

#Preprocessing
#test_df.drop(columns=['CustomerID'],inplace=True)
cat_cols=[i for i in test_df.columns if test_df[i].dtype=='object']
num_cols=[i for i in test_df.columns if test_df[i].dtype!='object']

#FeatureEngineering
test_df['valueperhourmonthly']=test_df['MonthlyCharges']/(test_df['ViewingHoursPerWeek']*4)
test_df['avgmonthlyusage']=(test_df['ViewingHoursPerWeek']*4)/test_df['AccountAge']
test_df['EngagementScore'] = (test_df['ContentDownloadsPerMonth'] + test_df['WatchlistSize'] + (test_df['ViewingHoursPerWeek']*4)) / 3
test_df['SupportIntensity'] = test_df['SupportTicketsPerMonth'] / (test_df['AccountAge'] + 1)
test_df['RecentActivityDrop'] = ((test_df['ViewingHoursPerWeek'] < test_df['ViewingHoursPerWeek'].quantile(0.25)) &(test_df['AccountAge'] > 6)).astype(int)
test_df['Highwatching']=(test_df['ViewingHoursPerWeek']>test_df['ViewingHoursPerWeek'].quantile(0.75).astype(int))
test_df['HighSatisfaction'] = (test_df['UserRating'] >= 4.0).astype(int)
test_df['ChargesToAge_Ratio'] = test_df['MonthlyCharges'] / (test_df['AccountAge'] + 1)
test_df['EngagementSatisfaction'] = test_df['ViewingHoursPerWeek'] * test_df['UserRating']

#Risk scores
test_df['Low_view_monthly']=((test_df['ViewingHoursPerWeek']*4)<(test_df['ViewingHoursPerWeek']*4).quantile(0.25).astype(int))
test_df['Low_view_session']=((test_df['ViewingHoursPerWeek']*4)<(test_df['ViewingHoursPerWeek']*4).quantile(0.25).astype(int))
test_df['LowSatisfaction'] = (train_df['UserRating'] <= 2.0).astype(int)
test_df['HighSupport'] = (test_df['SupportTicketsPerMonth'] > test_df['SupportTicketsPerMonth'].quantile(0.75)).astype(int)
test_df['RecentActivityDrop'] = ((test_df['ViewingHoursPerWeek'] < test_df['ViewingHoursPerWeek'].quantile(0.25)) &
                              (test_df['AccountAge'] > 6)).astype(int)
test_df['Total_risk_score']=test_df['Low_view_monthly']+test_df['Low_view_session']+test_df['LowSatisfaction']+test_df['HighSupport']+train_df['RecentActivityDrop']

test_df.replace({'SubscriptionType':{'Basic':0,'Standard':1,'Premium':2}},inplace=True)

cat_cols=[i for i in test_df.columns if test_df[i].dtype=='object']
num_cols=[i for i in test_df.columns if test_df[i].dtype!='object']

X = test_df

cat_cols=[i for i in X.columns if test_df[i].dtype=='object']
num_cols=[i for i in X.columns if test_df[i].dtype!='object']

X_encoded = OHEncoder.transform(X)

y_preds=log_model.predict(X_encoded)
print(y_preds)

y_preds.shape


#Unseen data prediction
y_train_prob = log_model.predict_proba(X_resampled)[:, 1]

y_test_prob = log_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

train_roc_auc = roc_auc_score(y_resampled, y_train_prob)

test_roc_auc = roc_auc_score(y_test, y_test_prob)

print(f"Training Data ROC-AUC Score: {train_roc_auc:.4f}")
print(f"Testing Data ROC-AUC Score:  {test_roc_auc:.4f}")

y_train_prob = xgb_model.predict_proba(X_resampled)[:, 1]

y_test_prob = xgb_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

train_roc_auc = roc_auc_score(y_resampled, y_train_prob)

test_roc_auc = roc_auc_score(y_test, y_test_prob)

print(f"Training Data ROC-AUC Score: {train_roc_auc:.4f}")
print(f"Testing Data ROC-AUC Score:  {test_roc_auc:.4f}")


