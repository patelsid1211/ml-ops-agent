import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

from agent.orchestrator import run_agent_mock
from training.evaluate import evaluate
from pipeline.feature_store import list_versions, get_latest_version

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML-Ops Agent API",
    description="REST API for triggering the ML monitoring agent and querying pipeline status",
    version="1.0.0"
)


# --- Request / Response models ---
class MonitorRequest(BaseModel):
    message: str = "Please monitor the ML pipeline and take appropriate action."


class MonitorResponse(BaseModel):
    status:    str
    report:    str
    timestamp: str


class EvaluateResponse(BaseModel):
    status:   str
    rmse:     float
    mae:      float
    r2:       float
    run_id:   str
    version:  str
    reasons:  list


class HealthResponse(BaseModel):
    status:          str
    timestamp:       str
    latest_version:  str | None
    total_versions:  int


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and pipeline are operational."""
    versions = list_versions()
    return HealthResponse(
        status         = "ok",
        timestamp      = datetime.now().isoformat(),
        latest_version = get_latest_version(),
        total_versions = len(versions)
    )


@app.post("/monitor", response_model=MonitorResponse)
def monitor(request: MonitorRequest = MonitorRequest()):
    """Trigger the agent monitoring cycle.
    
    The agent will:
    1. Query the feature store
    2. Run model evaluation
    3. Decide and execute the appropriate action
    """
    try:
        logger.info("Monitor endpoint triggered")
        report = run_agent_mock()
        return MonitorResponse(
            status    = "completed",
            report    = report,
            timestamp = datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate", response_model=EvaluateResponse)
def run_evaluate():
    """Run the evaluation harness and return current model metrics."""
    try:
        logger.info("Evaluate endpoint triggered")
        report = evaluate()
        return EvaluateResponse(
            status  = report["status"],
            rmse    = report["metrics"]["rmse"],
            mae     = report["metrics"]["mae"],
            r2      = report["metrics"]["r2"],
            run_id  = report["run_id"],
            version = report["version"],
            reasons = report["reasons"]
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)