import pandas as pd
import pickle
import os

# Define the directory where your files are located
DATA_DIR = r"C:\Users\sadaf naz\Desktop\Smart Chef Project\data'" \
""

def inspect():
    
    print("PHASE 3: PREPROCESSING SUMMARY")

    # 1. Load PP_recipes.csv
    print("\n[1] Loading PP_recipes.csv...")
    recipes_path = os.path.join(DATA_DIR, "PP_recipes.csv")
    if os.path.exists(recipes_path):
        pp_recipes = pd.read_csv(recipes_path)
        print(f"    Shape: {pp_recipes.shape}")
        print(f"    Columns: {list(pp_recipes.columns)}")
        print(pp_recipes.head(2))
    else:
        print(f"    File not found: {recipes_path}")

    # 2. Load PP_users.csv
    print("\n[2] Loading PP_users.csv...")
    users_path = os.path.join(DATA_DIR, "PP_users.csv")
    if os.path.exists(users_path):
        pp_users = pd.read_csv(users_path)
        print(f"    Shape: {pp_users.shape}")
        print(f"    Columns: {list(pp_users.columns)}")
        print(pp_users.head(2))
    else:
        print(f"    File not found: {users_path}")

    # 3. Load ingr_map.pkl (THE FIX)
    print("\n[3] Loading ingr_map.pkl...")
    ingr_map_path = os.path.join(DATA_DIR, "ingr_map.pkl")
    if os.path.exists(ingr_map_path):
        try:
            # Using pd.read_pickle() instead of pickle.load() to handle pandas 2.0 compatibility
            ingr_map = pd.read_pickle(ingr_map_path)
            print(f"    Shape: {ingr_map.shape}")
            print(f"    Columns: {list(ingr_map.columns)}")
            print(ingr_map.head(2))
            
            # Optional: Check for missing values in the mapping
            print(f"    Missing values in 'replaced': {ingr_map['replaced'].isna().sum()}")
            
        except Exception as e:
            print(f"    Error loading ingr_map.pkl: {e}")
            print("    Try installing an older version of pandas: pip install 'pandas<2.0.0'")
    else:
        print(f"    File not found: {ingr_map_path}")

    # 4. Load interactions_train.csv
    print("\n[4] Loading interactions_train.csv...")
    train_path = os.path.join(DATA_DIR, "interactions_train.csv")
    if os.path.exists(train_path):
        train_interactions = pd.read_csv(train_path)
        print(f"    Shape: {train_interactions.shape}")
        print(f"    Columns: {list(train_interactions.columns)}")
        print(train_interactions.head(2))
    else:
        print(f"    File not found: {train_path}")

    # 5. Load interactions_test.csv
    print("\n[5] Loading interactions_test.csv...")
    test_path = os.path.join(DATA_DIR, "interactions_test.csv")
    if os.path.exists(test_path):
        test_interactions = pd.read_csv(test_path)
        print(f"    Shape: {test_interactions.shape}")
        print(f"    Columns: {list(test_interactions.columns)}")
        print(test_interactions.head(2))
    else:
        print(f"    File not found: {test_path}")

    print("Phase 3 Inspection Complete.")

# Run the inspection
if __name__ == "__main__":
    inspect()