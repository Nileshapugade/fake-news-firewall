import sys
import os
# Add the nested fake-news-firewall/ folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))


from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_classify():
    response = client.post("/classify", json={"text": "This is a test article"})
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()
    assert "explanation" in response.json()

def test_feedback():
    response = client.post("/feedback", json={"text": "Test article", "label": "credible"})
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback saved"}