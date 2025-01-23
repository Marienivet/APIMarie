import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

app = FastAPI()

# Détection de l'environnement
if "GITHUB_ACTIONS" in os.environ:
    project_path = "."  # Le répertoire racine du dépôt dans GitHub Actions
else:
    project_path = "/content/drive/MyDrive/OCProjet7/git"

model_path = os.path.join(project_path, "xgb_model_optimized.pkl")
data_path = os.path.join(project_path, "df_git_train.csv")  # Ajusté pour correspondre au nom exact

# Chargement des fichiers
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fichier introuvable : {model_path}")
    model = joblib.load(model_path)
    print("Modèle chargé avec succès")
except Exception as e:
    raise Exception(f"Erreur lors du chargement du modèle : {e}")

try:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    df_clean = pd.read_csv(data_path)
    print("Dataset chargé avec succès")
except Exception as e:
    raise Exception(f"Erreur lors du chargement du dataset : {e}")

# Routes FastAPI
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction d'accord/refus de prêt !"}

@app.get("/predict/{client_id}")
def predict(client_id: int):
    client_data = df_clean[df_clean["SK_ID_CURR"] == client_id]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé dans le dataset.")

    features = client_data.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    prediction = model.predict(features)
    result = "Accordé" if prediction[0] >= 0.5 else "Refusé"

    return {"client_id": client_id, "prediction": result}
