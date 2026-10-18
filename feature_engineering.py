"""
Phase 5: Feature Engineering
- TF-IDF on ingredients (for content-based recommendation)
- Save TF-IDF matrix + vectorizer
"""

import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def load_clean():
    df = pd.read_csv('data/cleaned_recipes.csv')
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    print(f"Loaded: {df.shape}")
    return df


def build_ingredient_string(df):
    """Turn ['egg', 'flour'] → 'egg flour' for TF-IDF."""
    df = df.copy()
    df['ingredients_str'] = df['ingredients'].apply(lambda lst: ' '.join(lst))
    return df


def build_tfidf(df, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=5,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def save_artifacts(df, tfidf_matrix, vectorizer):
    with open('data/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('data/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    df[['name', 'ingredients_str']].to_csv(
        'data/recipes_for_tfidf.csv', index=False
    )
    print("✅ Saved: tfidf_matrix.pkl, tfidf_vectorizer.pkl, recipes_for_tfidf.csv")


if __name__ == "__main__":
    df = load_clean()
    df = build_ingredient_string(df)
    matrix, vec = build_tfidf(df)
    save_artifacts(df, matrix, vec)
    print("\n✅ Phase 5 complete.")