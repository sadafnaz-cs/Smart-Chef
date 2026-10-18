"""
Phase 1: Data Collection
Source: Food.com Recipes and Interactions Dataset (Kaggle)
Files:
    - RAW_recipes.csv         → Recipe metadata
    - RAW_interactions.csv    → User ratings & reviews
"""

import pandas as pd

def describe_source():
    recipes = pd.read_csv('data/RAW_recipes.csv')
    interactions = pd.read_csv('data/RAW_interactions.csv')

    print("PHASE 1: DATA COLLECTION SUMMARY")

    print(f"\n RAW_recipes.csv")
    print(f"   Shape: {recipes.shape}")
    print(f"   Columns: {recipes.columns.tolist()}")
    print(f"   Sample:")
    print(recipes.head(3))

    print(f"\n RAW_interactions.csv")
    print(f"   Shape: {interactions.shape}")
    print(f"   Columns: {interactions.columns.tolist()}")
    print(f"   Sample:")
    print(interactions.head(3))

    print("\n Data source documented.")


if __name__ == "__main__":
    describe_source()