import joblib
import os
import numpy as np
from scipy.sparse import hstack
from app.core.config import settings
from app.core.logging import logger
from typing import List, Dict, Any

class InferenceService:
    def __init__(self):
        self.tfidf_title = None
        self.tfidf_body = None
        self.mlb = None
        self.model = None
        self.is_loaded = False

    def load_model(self):
        """Loads the bundled model artifact."""
        if self.is_loaded:
            return

        try:
            model_path = os.path.join(settings.MODEL_PATH, settings.MODEL_FILENAME)

            logger.info(f"Loading model from {model_path}...")
            
            if os.path.exists(model_path):
                artifacts = joblib.load(model_path)
                self.tfidf_title = artifacts["tfidf_title"]
                self.tfidf_body = artifacts["tfidf_body"]
                self.mlb = artifacts["mlb"]
                self.model = artifacts["model"]
                
                # PATCH: Fix sklearn version mismatch
                # Newer sklearn forbids having both 'base_estimator' and 'estimator'/'classifier'
                # But requires 'estimator' to be present.
                if hasattr(self.model, "base_estimator") and not hasattr(self.model, "estimator"):
                    self.model.estimator = self.model.base_estimator
                    self.model.base_estimator = "deprecated"
                
                self.is_loaded = True
                logger.info("Model loaded successfully.")
            else:
                logger.warning(f"Model file not found at {model_path}. Inference will fail.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def preprocess(self, text: str) -> str:
        """Basic preprocessing: lowercase, strip. Handles None."""
        if text is None:
            return ""
        return text.lower().strip()

    def predict(self, title: str, body: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()
            if not self.is_loaded:
                 raise RuntimeError("Models are not loaded.")

        processed_title = self.preprocess(title)
        processed_body = self.preprocess(body)
        
        # Vectorize
        # Note: transform expects an iterable (list), so we wrap strings in a list
        X_title = self.tfidf_title.transform([processed_title])
        X_body = self.tfidf_body.transform([processed_body])
        
        # Combine features
        X_combined = hstack([X_title, X_body])
        
        # Predict
        prediction = self.model.predict(X_combined)
        
        # Calculate Confidence Scores
        # predict_proba returns an array of shape (n_samples, n_classes)
        probabilities = self.model.predict_proba(X_combined)
        
        # Handle sparse matrix if necessary
        if hasattr(probabilities, "toarray"):
            probabilities = probabilities.toarray()
            
        # Inverse transform to get labels
        predicted_labels = self.mlb.inverse_transform(prediction)
        
        # predicted_labels is a list of tuples (because input was a list), we take the first one
        labels_list = list(predicted_labels[0])
        
        # Map probabilities to labels
        scores = {}
        if len(probabilities) > 0:
            probs = probabilities[0]
            for idx, label in enumerate(self.mlb.classes_):
                # Convert numpy float to python float
                scores[label] = float(probs[idx])
        
        return {
            "labels": labels_list,
            "confidence_scores": scores
        }
