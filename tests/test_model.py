from api.models.classifier import NewsClassifier

def test_classifier():
    classifier = NewsClassifier()
    prediction, confidence = classifier.predict("This is a test article")
    assert prediction in ["credible", "misleading", "fake"]
    assert 0 <= confidence <= 1