"""
Phase 2: Exploratory Data Analysis
- Distribution of ingredients, steps, cooking time
- Top ingredients
- Most common tags
- Rating distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the screenshots directory if it doesn't exist
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')
    
sns.set_style("whitegrid")


def load():
    recipes = pd.read_csv('data/RAW_recipes.csv')
    interactions = pd.read_csv('data/RAW_interactions.csv')
    return recipes, interactions


def basic_stats(recipes, interactions):
    
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
  
    print(f"\n Recipes shape: {recipes.shape}")
    print(f" Interactions shape: {interactions.shape}")

    print("\n Numeric Summary (recipes)")
    print(recipes[['minutes', 'n_steps', 'n_ingredients']].describe())

    print("\n Rating distribution")
    print(interactions['rating'].value_counts().sort_index())


def plot_distributions(recipes, interactions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Ingredient count
    axes[0, 0].hist(recipes['n_ingredients'], bins=40, color='skyblue')
    axes[0, 0].set_title('Ingredients per Recipe')
    axes[0, 0].set_xlabel('Count')

    # 2. Steps count
    axes[0, 1].hist(recipes['n_steps'], bins=40, color='salmon')
    axes[0, 1].set_title('Steps per Recipe')
    axes[0, 1].set_xlabel('Count')

    # 3. Cooking time (log scale for readability)
    axes[1, 0].hist(recipes['minutes'].clip(0, 300), bins=50, color='lightgreen')
    axes[1, 0].set_title('Cooking Time (0-300 min)')
    axes[1, 0].set_xlabel('Minutes')

    # 4. Rating distribution
    interactions['rating'].value_counts().sort_index().plot(
        kind='bar', ax=axes[1, 1], color='orange'
    )
    axes[1, 1].set_title('Rating Distribution')
    axes[1, 1].set_xlabel('Rating (0-5)')

    plt.tight_layout()
    plt.savefig('screenshots/eda_distributions.png', dpi=100)
    import os
    os.makedirs('screenshots', exist_ok=True) # This will create the folder if missing, and do nothing if it already exists
    plt.savefig('screenshots/eda_distributions.png', dpi=100)
    plt.show()
    print("\n Saved: screenshots/eda_distributions.png")


def top_ingredients(recipes):
    import ast
    from collections import Counter

    all_ing = []
    for lst in recipes['ingredients'].dropna():
        try:
            all_ing.extend(ast.literal_eval(lst))
        except (ValueError, SyntaxError):
            continue

    top20 = Counter(all_ing).most_common(20)
    names, counts = zip(*top20)

    plt.figure(figsize=(10, 6))
    plt.barh(names[::-1], counts[::-1], color='coral')
    plt.title('Top 20 Ingredients Across Recipes')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('screenshots/top_ingredients.png', dpi=100)
    plt.show()
    print("✅ Saved: screenshots/top_ingredients.png")


if __name__ == "__main__":
    recipes, interactions = load()
    basic_stats(recipes, interactions)
    plot_distributions(recipes, interactions)
    top_ingredients(recipes)