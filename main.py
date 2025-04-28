import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Load environment variables from .env
load_dotenv()

# Credentials and instance setup
INSTANCE = os.getenv("INSTANCE")
USERNAME = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("CLIENT_SECRET")

# Initialize FastAPI app
app = FastAPI()

# --- Root Health Check ---
@app.get("/")
def root():
    return {"message": "ServiceNow connector is running!"}

# --- Get Full Incident by Number ---
@app.get("/incident/{incident_number}")
def get_incident_by_number(incident_number: str):
    """
    Get the full raw JSON of an incident by number (e.g., INC2017_009780).
    """
    url = f"{INSTANCE}/api/now/table/incident?sysparm_query=number={incident_number}"
    response = requests.get(
        url,
        auth=(USERNAME, PASSWORD),
        headers={"Accept": "application/json"}
    )
    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content={"error": response.text})

    data = response.json()
    if "result" in data and len(data["result"]) > 0:
        return {"status": "success", "incident": data["result"][0]}
    else:
        return {"status": "not found", "message": f"No incident found for number {incident_number}"}

# --- Get Summary of an Incident ---
@app.get("/incident_summary/{incident_number}")
def summarize_incident(incident_number: str):
    """
    Return a clean summary of key fields for an incident.
    """
    url = f"{INSTANCE}/api/now/table/incident?sysparm_query=number={incident_number}"
    response = requests.get(
        url,
        auth=(USERNAME, PASSWORD),
        headers={"Accept": "application/json"}
    )
    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content={"error": response.text})

    data = response.json()
    if "result" not in data or len(data["result"]) == 0:
        return {"status": "not found", "message": f"No incident found for number {incident_number}"}

    incident = data["result"][0]
    summary = {
        "number": incident.get("number"),
        "short_description": incident.get("short_description"),
        "description": incident.get("description"),
        "category": incident.get("category"),
        "state": incident.get("state"),
        "priority": incident.get("priority"),
        "urgency": incident.get("urgency"),
        "impact": incident.get("impact"),
        "opened_at": incident.get("opened_at"),
        "resolved_at": incident.get("resolved_at"),
        "close_notes": incident.get("close_notes"),
    }
    return {"status": "success", "summary": summary}

# --- App Entry Point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)