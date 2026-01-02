from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.exceptions import global_exception_handler, http_exception_handler
from app.core.logging import logger
from app.modules.inference.router import router as inference_router

def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG_MODE,
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routers
    application.include_router(inference_router, prefix=settings.API_PREFIX, tags=["Inference"])

    # Exception Handlers
    application.add_exception_handler(Exception, global_exception_handler)
    application.add_exception_handler(HTTPException, http_exception_handler)

    return application

app = create_application()

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url=f"{settings.API_PREFIX}/docs")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "app_name": settings.PROJECT_NAME}

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
