from functools import lru_cache
from app.modules.inference.service import InferenceService

@lru_cache()
def get_inference_service() -> InferenceService:
    service = InferenceService()
    # Optionally load model on startup or lazy load on first request
    # service.load_model() 
    return service
