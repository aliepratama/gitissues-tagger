from fastapi import APIRouter, Depends, HTTPException
from app.modules.inference.schemas import PredictionInput, PredictionOutput
from app.modules.inference.service import InferenceService
from app.modules.inference.dependencies import get_inference_service

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    service: InferenceService = Depends(get_inference_service)
):
    try:
        result = service.predict(input_data.title, input_data.body)
        return PredictionOutput(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
