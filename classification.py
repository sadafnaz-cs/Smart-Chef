"""
Phase 6.2 : Multi-Target Recipe Classification
Trains THREE separate classifiers:
  1. Cuisine     → Italian, Mexican, Pakistani, Indian, Chinese, ...
  2. Meal-Time   → Breakfast, Lunch, Dinner, Dessert, Snack, Beverage
  3. Course      → Dessert, Savoury, Main, Side, Appetizer, Condiment
"""

import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# LABEL DEFINITIONS
CUISINE_KEYWORDS = {
    'italian':        ['italian', 'pasta', 'pizza', 'risotto', 'lasagna', 'carbonara'],
    'mexican':        ['mexican', 'taco', 'burrito', 'enchilada', 'quesadilla'],
    'pakistani':      ['pakistani', 'karahi', 'biryani', 'nihari', 'haleem', 'chapli kabab'],
    'indian':         ['indian', 'curry', 'masala', 'tandoori', 'dal', 'paneer', 'samosa'],
    'chinese':        ['chinese', 'stir-fry', 'stir fry', 'wok', 'dumpling', 'szechuan'],
    'japanese':       ['japanese', 'sushi', 'ramen', 'teriyaki', 'miso', 'tempura'],
    'thai':           ['thai', 'pad thai', 'tom yum'],
    'french':         ['french', 'baguette', 'croissant', 'ratatouille', 'quiche'],
    'american':       ['american', 'bbq', 'burger', 'barbecue', 'mac and cheese'],
    'mediterranean':  ['mediterranean', 'greek', 'hummus', 'falafel', 'tzatziki'],
    'middle_eastern': ['middle eastern', 'shawarma', 'kebab', 'tabbouleh', 'persian'],
    'korean':         ['korean', 'kimchi', 'bulgogi', 'bibimbap', 'gochujang'],
}

MEAL_KEYWORDS = {
    'breakfast':  ['breakfast', 'brunch', 'morning', 'pancake', 'waffle', 'omelette'],
    'lunch':      ['lunch', 'midday', 'sandwich', 'wrap'],
    'dinner':     ['dinner', 'main dish', 'main course', 'supper', 'entree', 'entrée'],
    'dessert':    ['dessert', 'sweet', 'cake', 'cookie', 'pie', 'pudding', 'ice cream'],
    'snack':      ['snack', 'appetizer', 'starter', 'finger food', 'dip'],
    'beverage':   ['beverage', 'drink', 'smoothie', 'cocktail', 'juice', 'tea', 'coffee'],
}

COURSE_KEYWORDS = {
    'dessert':    ['dessert', 'sweet', 'cake', 'cookie', 'pie', 'candy', 'chocolate'],
    'savoury':    ['savoury', 'savory', 'salty', 'spicy'],
    'main':       ['main dish', 'main course', 'entree', 'entrée', 'dinner'],
    'side':       ['side dish', 'side', 'accompaniment'],
    'appetizer':  ['appetizer', 'starter', "hors d'oeuvre"],
    'condiment':  ['condiment', 'sauce', 'dressing', 'spread', 'dip', 'seasoning'],
}

# DATA LOADING
def load_data():
    df = pd.read_csv('data/cleaned_recipes.csv')
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['tags'] = df['tags'].apply(ast.literal_eval)
    df['ingredients_str'] = df['ingredients'].apply(lambda x: ' '.join(x))
    df['tags_str'] = df['tags'].apply(lambda x: ' '.join(x).lower())
    return df

# LABELING FUNCTION (single, reusable)
def assign_label(tags_str, keyword_map, default='other'):
    """
    Assign a single label based on keyword matching.
    If multiple match, pick the one whose keyword appears earliest
    in the tags string (proxy for most relevant).
    """
    best_label = default
    best_pos = float('inf')

    for label, keywords in keyword_map.items():
        for kw in keywords:
            pos = tags_str.find(kw)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_label = label

    return best_label

# TRAIN A SINGLE CLASSIFIER (reusable for all 3 targets)
def train_one_classifier(df, target_col, name, min_samples=50, top_n=10):
    """Train & evaluate a single classifier for a given label column."""
    print(f"  TRAINING: {name.upper()} CLASSIFIER")

    # Drop 'other' if it dominates, keep only well-populated classes
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= min_samples].head(top_n).index.tolist()
    subset = df[df[target_col].isin(valid_classes)].copy()

    print(f"Classes kept ({len(valid_classes)}): {valid_classes}")
    print(f"Rows: {len(subset):,}")
    print(f"Distribution:\n{subset[target_col].value_counts().to_string()}\n")

    # Features & target
    X = subset['ingredients_str']
    y = subset[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, min_df=3, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1:.4f}\n")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs('models', exist_ok=True)
    path = f'models/{name}_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"✅ Saved → {path}")

    # Confusion matrix
    os.makedirs('screenshots', exist_ok=True)
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=model.classes_, yticklabels=model.classes_,
                cmap='Oranges')
    plt.title(f'{name.capitalize()} Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'screenshots/classification_{name}_confusion.png', dpi=100)
    plt.close()
    print(f"✅ Saved confusion matrix → screenshots/classification_{name}_confusion.png")

    return model, vectorizer, acc, f1

# MAIN
if __name__ == "__main__":
    print("PHASE 6.2: MULTI-TARGET CLASSIFICATION")

    df = load_data()
    print(f"Loaded: {df.shape}")

    # Create all 3 label columns
    df['cuisine']   = df['tags_str'].apply(lambda t: assign_label(t, CUISINE_KEYWORDS))
    df['meal_time'] = df['tags_str'].apply(lambda t: assign_label(t, MEAL_KEYWORDS))
    df['course']    = df['tags_str'].apply(lambda t: assign_label(t, COURSE_KEYWORDS))

    # Quick preview of label distributions
    print("\n--- Cuisine distribution ---")
    print(df['cuisine'].value_counts().head(10))
    print("\n--- Meal-time distribution ---")
    print(df['meal_time'].value_counts().head(10))
    print("\n--- Course distribution ---")
    print(df['course'].value_counts().head(10))

    # Train all 3
    results = {}
    results['cuisine']   = train_one_classifier(df, 'cuisine',   'cuisine')
    results['meal_time'] = train_one_classifier(df, 'meal_time', 'meal_time')
    results['course']    = train_one_classifier(df, 'course',    'course')

    # Summary
    print("  FINAL SUMMARY")
    print(f"{'Target':<15} {'Accuracy':<12} {'F1 (weighted)':<15}")
    for target, (_, _, acc, f1) in results.items():
        print(f"{target:<15} {acc:<12.4f} {f1:<15.4f}")

    # Save target definitions for app.py to reuse
    with open('models/label_definitions.pkl', 'wb') as f:
        pickle.dump({
            'cuisine': CUISINE_KEYWORDS,
            'meal_time': MEAL_KEYWORDS,
            'course': COURSE_KEYWORDS,
        }, f)
    print("\n✅ Saved label definitions → models/label_definitions.pkl")

    # Test predictions
    print("\nSample predictions")
    sample = "tomato mozzarella basil olive oil"
    for target, (model, vec, _, _) in results.items():
        pred = model.predict(vec.transform([sample]))[0]
        print(f"  {target:<12}: {pred}")

    print("\n✅ Phase 6.2 complete.")