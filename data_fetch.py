import pandas as pd
import numpy as np
import os

# 1. On définit la fonction (La recette)
def generate_synthetic_data(n_samples=2000):
    print("Tentative de génération des données...")
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.random.seed(42)
    data = {
        'rainfall_mm': np.random.gamma(2, 10, n_samples),
        'river_level_m': np.random.normal(3.0, 1.5, n_samples),
        'soil_moisture_index': np.random.uniform(0, 100, n_samples),
        'elevation_m': np.concatenate([np.random.normal(50, 10, n_samples//2), np.random.normal(5, 2, n_samples//2)]),
        'drainage_capacity': np.random.choice([0.3, 0.5, 0.8], n_samples)
    }
    
    df = pd.DataFrame(data)
    risk_score = (df['rainfall_mm'] * 0.4) + (df['river_level_m'] * 10) - (df['elevation_m'] * 2)
    df['flood_occurred'] = (risk_score > 25).astype(int)
    
    file_path = os.path.join(data_dir, 'abidjan_flood_data.csv')
    df.to_csv(file_path, index=False)
    print(f"TERMINÉ : Le fichier a été créé dans {file_path}")

# 2. On appelle la fonction (On tourne la clé)
# C'EST ICI QUE TU METS LE CODE :
if __name__ == "__main__":
    generate_synthetic_data()