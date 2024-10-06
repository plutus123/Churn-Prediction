# main.py

import os
from src.functions import (
    load_data,
    remove_duplicates,
    handle_missing_values,
    encode_categorical,
    convert_bool_to_int,
    scale_features,
    feature_engineering,
    apply_smote,
    split_data,
    train_random_forest,
    train_xgboost,
    train_mlp,
    train_lightgbm,
    train_catboost,
    train_voting_classifier,
    evaluate_model,
    cross_validate_model,
    get_feature_importance,
    recursive_feature_elimination,
    save_model
)

import pandas as pd


def main():
    # Paths
    data_path = os.path.join('data', 'game_user_churn.csv')
    models_dir = os.path.join('src', 'models')

    # 1. Load Data
    data = load_data(data_path)
    print("First 5 rows of the dataset:")
    print(data.head())

    # 2. Remove Duplicates
    data = remove_duplicates(data)

    # 3. Handle Missing Values
    data = handle_missing_values(data)

    # 4. Convert Categorical Columns to 'category' dtype
    categorical_columns = [
        'gender', 'country', 'game_genre',
        'subscription_status', 'device_type', 'favorite_game_mode'
    ]
    for col in categorical_columns:
        data[col] = data[col].astype('category')

    # 5. Encode Categorical Variables
    data = encode_categorical(data, categorical_columns)

    # 6. Convert Boolean Columns to Integers
    data = convert_bool_to_int(data)

    # 7. Feature Scaling
    numerical_columns = [
        'age', 'total_play_time', 'avg_session_time', 'games_played',
        'in_game_purchases', 'last_login', 'friend_count',
        'max_level_achieved', 'daily_play_time', 'number_of_sessions',
        'social_interactions', 'achievement_points'
    ]
    data, scaler = scale_features(data, numerical_columns)

    # 8. Feature Engineering
    data = feature_engineering(data)

    # 9. Final Verification
    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nData types after converting boolean columns:")
    print(data.dtypes)

    # 10. Prepare Feature Set and Target Variable
    X = data.drop(['churn'], axis=1)
    y = data['churn']

    # 11. Handle Class Imbalance with SMOTE
    X_smote, y_smote = apply_smote(X, y)

    # 12. Train-Test Split
    X_train, X_test, y_train, y_test = split_data(X_smote, y_smote)

    # 13. Initialize and Train Models
    print("\nTraining Random Forest Classifier...")
    rf_clf = train_random_forest(X_train, y_train)
    save_model(rf_clf, "RandomForest")

    print("\nTraining XGBoost Classifier...")
    xgb_clf = train_xgboost(X_train, y_train)
    save_model(xgb_clf, "XGBoost")

    print("\nTraining MLP Classifier...")
    mlp_clf = train_mlp(X_train, y_train)
    save_model(mlp_clf, "MLP_Classifier")

    print("\nTraining LightGBM Classifier...")
    lgbm_clf = train_lightgbm(X_train, y_train)
    save_model(lgbm_clf, "LightGBM")

    print("\nTraining CatBoost Classifier...")
    cat_clf = train_catboost(X_train, y_train)
    save_model(cat_clf, "CatBoost")

    print("\nTraining Voting Classifier...")
    voting_clf = train_voting_classifier(X_train, y_train)
    save_model(voting_clf, "Voting_Classifier")

    # 14. Evaluate Models
    models = {
        'RandomForest': rf_clf,
        'XGBoost': xgb_clf,
        'LightGBM': lgbm_clf,
        'CatBoost': cat_clf,
        'Voting Classifier': voting_clf,
        'MLP_Classifier': mlp_clf
    }

    roc_auc_scores = {}

    for name, model in models.items():
        roc_auc = evaluate_model(model, X_test, y_test, name)
        roc_auc_scores[name] = roc_auc

    # 15. Display ROC-AUC Scores
    roc_auc_df = pd.DataFrame(list(roc_auc_scores.items()), columns=['Model', 'ROC-AUC'])
    print("\nROC-AUC Scores for All Models:")
    print(roc_auc_df)

    # 16. Cross-Validation for Random Forest
    print("\nPerforming Cross-Validation for Random Forest...")
    cross_validate_model(rf_clf, X_train, y_train, cv=5)

    # 17. Feature Importance for Random Forest
    print("\nAnalyzing Feature Importances for Random Forest...")
    feature_importances = get_feature_importance(rf_clf, X_train.columns)

    # 18. Recursive Feature Elimination (RFE)
    print("\nPerforming Recursive Feature Elimination (RFE)...")
    selected_features = recursive_feature_elimination(rf_clf, X_train, y_train, n_features=20)

    # 19. Train Random Forest with Selected Features
    X_train_rfe = X_train[selected_features]
    X_test_rfe = X_test[selected_features]

    print("\nTraining Random Forest with RFE-selected Features...")
    rf_rfe = train_random_forest(X_train_rfe, y_train)
    save_model(rf_rfe, "RandomForest_RFE")

    # 20. Evaluate the RFE Model
    evaluate_model(rf_rfe, X_test_rfe, y_test, "RandomForest with RFE Features")

    print("\nBest Model: RandomForest")
    # Optionally, you can set RandomForest as the best model and proceed with it


if __name__ == "__main__":
    main()
