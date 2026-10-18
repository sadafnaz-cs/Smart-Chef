"""
Phase 6.3: Cooking Time Prediction
Predict 'minutes' from ingredients + numeric features.
"""

import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv('data/cleaned_recipes.csv')
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))

    # Filter outliers
    df = df[(df['minutes'] > 0) & (df['minutes'] < 300)]
    return df


def build_features(df):
    """Combine TF-IDF ingredients + numeric features."""
    vectorizer = TfidfVectorizer(max_features=2000, min_df=5, stop_words='english')
    ing_vec = vectorizer.fit_transform(df['ingredients_str'])

    numeric = df[['n_ingredients', 'ingredient_chars']].values

    from scipy.sparse import hstack
    X = hstack([ing_vec, numeric]).tocsr()
    y = df['minutes'].values

    return X, y, vectorizer


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Ridge
    print("\n--- Ridge Regression ---")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    preds = ridge.predict(X_test)
    results['Ridge'] = evaluate(y_test, preds)

    # Random Forest
    print("\nRandom Forest")
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=15, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results['RandomForest'] = evaluate(y_test, preds)

    return results, ridge, rf


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  MAE:  {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  R²:   {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_predictions(model, X_test, y_test, name):
    preds = model.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.3, s=5)
    plt.plot([0, 300], [0, 300], 'r--', label='Perfect prediction')
    plt.xlabel('Actual minutes')
    plt.ylabel('Predicted minutes')
    plt.title(f'{name} — Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'screenshots/regression_{name.lower()}.png', dpi=100)
    print(f"Saved regression_{name.lower()}.png")


if __name__ == "__main__":
    print("PHASE 6.3: REGRESSION")

    os.makedirs('models', exist_ok=True)
    os.makedirs('screenshots', exist_ok=True)

    df = load_data()
    print(f"Loaded: {df.shape}")

    X, y, vectorizer = build_features(df)
    print(f"Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nRidge")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_results = evaluate(y_test, ridge.predict(X_test))

    print("\nRandom Forest")
    rf = RandomForestRegressor(n_estimators=50, max_depth=15,
                               n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_results = evaluate(y_test, rf.predict(X_test))

    plot_predictions(ridge, X_test, y_test, 'Ridge')
    plot_predictions(rf, X_test, y_test, 'RandomForest')

    # Save best model
    with open('models/regression_model.pkl', 'wb') as f:
        pickle.dump((rf, vectorizer), f)

    print("\n✅ Phase 6.3 complete.")