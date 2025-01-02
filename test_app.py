from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction d'accord/refus de prêt !"}

def test_predict_valid_client():
    # Exemple d'un client qui existe
    client_id = 353213  # Remplacez par un ID valide présent dans `df_git_train.csv`
    response = client.get(f"/predict/{client_id}")
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["client_id"] == client_id

def test_predict_invalid_client():
    # Exemple d'un client inexistant
    client_id = 99999999  # Assurez-vous que cet ID n'existe pas dans `df_git_train.csv`
    response = client.get(f"/predict/{client_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Client non trouvé dans le dataset."}
