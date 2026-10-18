import os
import pickle
import gzip

models_dir = 'models'
for filename in os.listdir(models_dir):
    if filename.endswith('.pkl'):
        filepath = os.path.join(models_dir, filename)
        print(f"Compressing {filename}...")
        
        # Load the normal pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        # Overwrite it with a compressed version
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
print("Done! Models are now much smaller.")