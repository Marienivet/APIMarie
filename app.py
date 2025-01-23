from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI()

# Chemins des fichiers pour Google Colab
project_path = "/content/drive/My Drive/OCProjet7/git"
model_path = os.path.join(project_path, "xgb_model_optimized.pkl")
data_path = os.path.join(project_path, "df_train_clean.csv")

# Chargement du modèle et des données
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise Exception(f"Modèle introuvable au chemin {model_path}. Vérifiez le chemin ou le fichier.")

try:
    df_clean = pd.read_csv(data_path)
except FileNotFoundError:
    raise Exception(f"Dataset nettoyé introuvable au chemin {data_path}. Vérifiez le chemin ou le fichier.")

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
