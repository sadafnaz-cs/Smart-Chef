"""
Data Cleaning
- Remove null values
- Select useful columns
- Create ingredient features
- Save cleaned_recipes.csv
"""

import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')


# Step 1: Load 
def load_data(path='data/RAW_recipes.csv'):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")
    return df


# Step 2: Remove nulls 
def remove_nulls(df):
    essential_cols = [c for c in ['name', 'ingredients', 'steps'] if c in df.columns]
    df = df.dropna(subset=essential_cols)
    for col in ['description', 'tags']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    print(f"After null removal: {df.shape}")
    return df


# Step 3: Select useful columns 
def select_columns(df):
    useful_cols = ['name', 'ingredients', 'steps', 'minutes',
                   'n_ingredients', 'tags', 'nutrition']
    keep = [c for c in useful_cols if c in df.columns]
    df = df[keep].copy()
    print(f"Selected columns: {list(df.columns)}")
    return df


# Step 4: Parse & clean ingredients 
def parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return [i.strip() for i in str(x).split(',')]


def clean_ingredient(ing):
    ing = str(ing).lower().strip().replace('-', ' ')
    return ' '.join(ing.split())


def process_ingredients(df):
    df['ingredients'] = df['ingredients'].apply(parse_list)
    df['ingredients'] = df['ingredients'].apply(
        lambda lst: [clean_ingredient(i) for i in lst if i]
    )
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(parse_list)
    print(f"Ingredients parsed. Example: {df['ingredients'].iloc[0][:3]}")
    return df


# Step 5: Feature engineering 
def create_features(df):
    df['n_ingredients'] = df['ingredients'].apply(len)
    df['n_steps'] = df['steps'].apply(
        lambda s: len(ast.literal_eval(s)) if isinstance(s, str) and s.startswith('[') else 1
    )
    df['ingredient_chars'] = df['ingredients'].apply(
        lambda x: sum(len(i) for i in x)
    )
    print(f"Features created. Shape: {df.shape}")
    return df


# Step 6: Remove duplicates & outliers 
def clean_outliers(df):
    before = len(df)
    # Convert ingredients list -> tuple for hashability, then drop
    df['_ing_key'] = df['ingredients'].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )
    df = df.drop_duplicates(subset=['name', '_ing_key'])
    df = df.drop(columns=['_ing_key'])
    print(f"Removed {before - len(df)} duplicates")

    df = df[df['n_ingredients'] > 0]
    if 'minutes' in df.columns:
        df = df[(df['minutes'] > 0) & (df['minutes'] < 1440)]
    return df


# Step 7: Save 
def save_clean(df, path='data/cleaned_recipes.csv'):
    df = df.reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Saved → {path} | Shape: {df.shape}")


# Main pipeline 
if __name__ == "__main__":
    df = load_data()
    df = remove_nulls(df)
    df = select_columns(df)
    df = process_ingredients(df)
    df = create_features(df)
    df = clean_outliers(df)
    save_clean(df)

    print("\n--- Sanity Checks ---")
    print("Nulls:", df[['name', 'ingredients']].isnull().sum().to_dict())
    print("Sample ingredients type:", type(df['ingredients'].iloc[0]))
    print(df[['n_ingredients', 'n_steps']].describe())