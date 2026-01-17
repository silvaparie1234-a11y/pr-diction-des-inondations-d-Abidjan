import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    print("--- Début de l'entraînement de l'IA ---")
    
    # Vérification du fichier de données
    data_path = 'data/abidjan_flood_data.csv'
    if not os.path.exists(data_path):
        print("ERREUR : Le fichier de données n'existe pas dans data/")
        return

    # Chargement
    df = pd.read_csv(data_path)
    X = df.drop('flood_occurred', axis=1)
    y = df['flood_occurred']
    
    # Séparation Entraînement / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Création du modèle
    print("Analyse des données en cours...")
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    
    # Score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SUCCÈS : Modèle entraîné avec une précision de : {acc*100:.2f}%")
    
    # Sauvegarde
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/flood_xgboost.pkl')
    print("FICHIER CRÉÉ : models/flood_xgboost.pkl")

# LA CLÉ DE CONTACT (Indispensable !)
if __name__ == "__main__":
    train_model()