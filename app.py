from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI()

# Détection de l'environnement (local vs CI/CD)
if "GITHUB_ACTIONS" in os.environ:  # CI/CD (GitHub Actions)
    project_path = "OCProjet7/git"
else:  # Local (Google Colab ou autre)
    project_path = "/content/drive/MyDrive/OCProjet7/git"

# Chemins des fichiers
model_path = os.path.join(project_path, "xgb_model_optimized.pkl")
data_path = os.path.join(project_path, "df_train_clean.csv")

# Chargement du modèle et des données
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fichier introuvable : {model_path}")
    model = joblib.load(model_path)
except FileNotFoundError as e:
    raise Exception(f"Modèle introuvable au chemin {model_path}. Vérifiez le chemin ou le fichier.") from e

try:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")
    df_clean = pd.read_csv(data_path)
except FileNotFoundError as e:
    raise Exception(f"Dataset nettoyé introuvable au chemin {data_path}. Vérifiez le chemin ou le fichier.") from e

# Routes API
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

