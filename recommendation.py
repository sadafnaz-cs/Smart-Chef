import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# 1. SETUP PATHS
# Define where files should go
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Make sure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. CONTENT-BASED FUNCTIONS (From Screenshot 1)
def build_tfidf_model(df):
    print("Building TF-IDF (may take a few minutes)...")
    vectorizer = TfidfVectorizer(
        max_features=5000, min_df=5, stop_words='english'
    )
    
    # Fit the vectorizer to the ingredients column
    matrix = vectorizer.fit_transform(df['ingredients'])
    
    # Define paths for saving
    matrix_path = os.path.join(DATA_DIR, 'tfidf_matrix.pkl')
    vec_path = os.path.join(DATA_DIR, 'tfidf_vectorizer.pkl')
    
    # Save the matrix and vectorizer
    with open(matrix_path, 'wb') as f:
        pickle.dump(matrix, f)
        
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saved TF-IDF matrix: {matrix.shape}")
    return matrix, vectorizer

# 3. COLLABORATIVE FILTERING FUNCTIONS (From Screenshot 2)
def build_svd_model(interactions_df):
    print("Starting SVD training...")
    
    # Surprise library requires specific data format
    reader = Reader(rating_scale=(1, 5)) # Adjust scale if your ratings are different!
    data = Dataset.load_from_df(interactions_df[['user_id', 'recipe_id', 'rating']], reader)
    
    # Split data
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train model
    model = SVD(n_factors=50, random_state=42)
    model.fit(trainset)
    
    # Save model directly to models folder
    model_path = os.path.join(MODELS_DIR, 'svd_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print("✅ SVD model trained and saved.")
    return model

# 4. MAIN EXECUTION BLOCK
if __name__ == "__main__":
    print("PHASE 6.1: RECOMMENDATION SYSTEM")
    
    # Step A: Load your Data 
    print("Loading data...")
    # Adjust these paths based on where your actual CSV files are
    recipes_df = pd.read_csv(os.path.join(DATA_DIR, 'RAW_recipes.csv'))
    interactions_df = pd.read_csv(os.path.join(DATA_DIR, 'RAW_interactions.csv'))

    # Step B: Run Content-Based Model 
    print("\nBuilding Content-Based Model")
    tfidf_matrix, vectorizer = build_tfidf_model(recipes_df)

    # Step C: Run Collaborative Model
    print("\nBuilding Collaborative Model")
    try:
        # NOTE: Ensure column names below match your CSV headers exactly!
        # e.g., if your CSV uses 'user' instead of 'user_id', change it here.
        svd_model = build_svd_model(interactions_df) 
    except Exception as e:
        print(f"❌ ERROR: SVD training failed. Reason: {e}")

    print("\n✅ Phase 6.1 complete.")

import pickle
import os

# 1. Load the saved model
model_path = os.path.join('models', 'svd_model.pkl')

with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

print("✅ Model loaded successfully!")

# 2. Make a prediction
# Let's pretend User ID 1 wants to know how they would rate Recipe ID 500
# NOTE: You must use actual IDs that exist in your RAW_interactions.csv file!
user_id = 1       # Change this to a real user ID from your data
recipe_id = 500   # Change this to a real recipe ID from your data

prediction = loaded_model.predict(user_id, recipe_id)
print(f"Predicted rating for User {user_id} on Recipe {recipe_id}: {prediction.est:.2f} stars")

print("\nTesting Saved SVD Model")
    
    # 1. Load the saved model back into memory
model_path = os.path.join(MODELS_DIR, 'svd_model.pkl')
with open(model_path, 'rb') as f:
        loaded_svd_model = pickle.load(f)
        
print("✅ Model loaded successfully from pickle file.")

    # 2. Make a test prediction
    # IMPORTANT: Change '1' and '500' to real IDs from your dataset!
test_user_id = 1  
test_recipe_id = 500 
    
prediction = loaded_svd_model.predict(test_user_id, test_recipe_id)
print(f"Prediction for User {test_user_id} on Recipe {test_recipe_id}: {prediction.est:.2f} stars")

    # 3. Get Top Recommendations for a User
print(f"\nTop 5 Recommendations for User {test_user_id}")
    
    # We need to test a list of recipes the user hasn't seen yet.
    # For this test, let's just grab 10 random recipes from your recipes_df
import random
sample_recipes = recipes_df.sample(10) 
    
predictions_list = []
for index, row in sample_recipes.iterrows():
        recipe_id = row['id'] # Assuming your recipes_df has an 'id' column
        # Predict rating
        pred = loaded_svd_model.predict(test_user_id, recipe_id)
        predictions_list.append((recipe_id, row['name'], pred.est))
        
    # Sort by predicted rating (highest first)
predictions_list.sort(key=lambda x: x[2], reverse=True)
    
    # Print the top 5
for i, (r_id, r_name, r_rating) in enumerate(predictions_list[:5]):
        print(f"{i+1}. {r_name} (ID: {r_id}) - Predicted Rating: {r_rating:.2f} stars")