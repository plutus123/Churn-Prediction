# src/functions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE
import joblib


def load_data(filepath):
    """
    Load the dataset from the specified filepath.
    """
    data = pd.read_csv(filepath)
    return data


def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    """
    duplicate_rows = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    if duplicate_rows > 0:
        data = data.drop_duplicates()
        print(f"Duplicates removed. New dataset shape: {data.shape}")
    else:
        print("No duplicate rows found.")
    return data


def handle_missing_values(data):
    """
    Handle missing values in the dataset.
    (Currently assumes no missing values; modify as needed)
    """
    missing_values = data.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    # Implement handling if necessary
    return data


def encode_categorical(data, categorical_columns):
    """
    Convert categorical columns to one-hot encoded variables.
    """
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data


def convert_bool_to_int(data):
    """
    Convert boolean columns to integers.
    """
    bool_cols = data.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        data[bool_cols] = data[bool_cols].astype(int)
        print(f"Boolean columns converted to integers: {bool_cols.tolist()}")
    return data


def scale_features(data, numerical_columns):
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data, scaler


def feature_engineering(data):
    """
    Create additional features and perform feature engineering.
    """
    data['play_time_per_session'] = data['total_play_time'] / (data['number_of_sessions'] + 1)
    data['purchases_per_session'] = data['in_game_purchases'] / (data['number_of_sessions'] + 1)
    data['engagement_score'] = (
            data['total_play_time'] * 0.4 +
            data['in_game_purchases'] * 0.3 +
            data['social_interactions'] * 0.2 +
            data['achievement_points'] * 0.1
    )

    # Replace negative values with zero
    data['purchases_per_session'] = data['purchases_per_session'].apply(lambda x: x if x >= 0 else 0)
    data['engagement_score'] = data['engagement_score'].apply(lambda x: x if x >= 0 else 0)

    # Log Transformation
    data['log_purchases_per_session'] = np.log1p(data['purchases_per_session'])
    data['log_engagement_score'] = np.log1p(data['engagement_score'])

    # Interaction Features
    data['play_time_purchases_interaction'] = data['total_play_time'] * data['in_game_purchases']
    # Assuming 'device_type_Mobile' and 'favorite_game_mode_Multiplayer' are already one-hot encoded
    if 'device_type_Mobile' in data.columns and 'favorite_game_mode_Multiplayer' in data.columns:
        data['device_game_mode'] = data['device_type_Mobile'] * data['favorite_game_mode_Multiplayer']
    if 'game_genre_Adventure' in data.columns and 'subscription_status_Yes' in data.columns:
        data['genre_subscription'] = data['game_genre_Adventure'] * data['subscription_status_Yes']

    return data


def apply_smote(X, y):
    """
    Apply SMOTE to handle class imbalance.
    """
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    print(f"\nAfter SMOTE, counts of label '1' (Churn): {sum(y_smote == 1)}")
    print(f"After SMOTE, counts of label '0' (No Churn): {sum(y_smote == 0)}")
    return X_smote, y_smote


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train a Random Forest Classifier.
    """
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', **kwargs)
    rf_clf.fit(X_train, y_train)
    return rf_clf


def train_xgboost(X_train, y_train, **kwargs):
    """
    Train an XGBoost Classifier.
    """
    xgb_clf = XGBClassifier(
        random_state=63,
        use_label_encoder=False,
        eval_metric='logloss',
        enable_categorical=True,
        **kwargs
    )
    xgb_clf.fit(X_train, y_train)
    return xgb_clf


def train_mlp(X_train, y_train, **kwargs):
    """
    Train an MLP Classifier.
    """
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=63,
        **kwargs
    )
    mlp_clf.fit(X_train, y_train)
    return mlp_clf


def train_lightgbm(X_train, y_train, **kwargs):
    """
    Train a LightGBM Classifier.
    """
    lgbm_clf = LGBMClassifier(
        random_state=63,
        **kwargs
    )
    lgbm_clf.fit(X_train, y_train)
    return lgbm_clf


def train_catboost(X_train, y_train, **kwargs):
    """
    Train a CatBoost Classifier.
    """
    cat_clf = CatBoostClassifier(
        random_state=63,
        verbose=0,
        **kwargs
    )
    cat_clf.fit(X_train, y_train)
    return cat_clf


def train_voting_classifier(X_train, y_train, **kwargs):
    """
    Train a Voting Classifier with XGBoost, LightGBM, and Random Forest.
    """
    xgb_clf = XGBClassifier(random_state=63, use_label_encoder=False, eval_metric='logloss')
    lgbm_clf = LGBMClassifier(random_state=63)
    rf_clf = RandomForestClassifier(random_state=63)

    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('lgbm', lgbm_clf),
            ('rf', rf_clf)
        ],
        voting='soft',
        **kwargs
    )
    voting_clf.fit(X_train, y_train)
    return voting_clf


def evaluate_model(model, X_test, y_test, model_name, save_plots=False, plot_dir='plots/'):
    """
    Evaluate the model and print metrics. Optionally save plots.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Evaluation ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'{model_name}_confusion_matrix.png'))
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, f'{model_name}_roc_curve.png'))
    plt.show()

    return roc_auc


def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform Stratified K-Fold Cross-Validation and return ROC-AUC scores.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=skf, scoring='roc_auc', n_jobs=-1
    )
    print("\nCross-Validation ROC-AUC Scores:")
    print(cv_scores)
    print(f"Mean ROC-AUC Score: {cv_scores.mean():.4f}")
    return cv_scores


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract and plot feature importances from the model.
    """
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nTop 20 Important Features:")
    print(feature_importances.head(top_n))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(top_n), palette='viridis')
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    return feature_importances.head(top_n)


def recursive_feature_elimination(model, X_train, y_train, n_features=20):
    """
    Perform Recursive Feature Elimination (RFE) to select top features.
    """
    rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    print(f"Selected Features ({len(selected_features)}): {selected_features.tolist()}")
    return selected_features


def save_model(model, model_name, models_dir='src/models/'):
    """
    Save the trained model to the specified directory.
    """
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
