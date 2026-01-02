from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    API_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI Multi-Label NLP Engine"
    DEBUG_MODE: bool = False
    MODEL_PATH: str = "models/"
    
    # Model filenames
    MODEL_FILENAME: str = "github_classifier_cc_lgbm_stratified.pkl"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
