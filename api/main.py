from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes.prediction_routes import router as prediction_router

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="SmartRent API")
app.include_router(prediction_router)
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend/index.html")
