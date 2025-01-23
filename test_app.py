from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API de prédiction d'accord/refus de prêt !"}

def test_predict_valid_client():
    client_id = 353213 	  # Exemple d'un client qui existe dans le dataset
    response = client.get(f"/predict/{client_id}")
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["client_id"] == client_id

def test_predict_invalid_client():
    client_id = 99999999  # Client inexistant
    response = client.get(f"/predict/{client_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Client non trouvé dans le dataset."}
