import csv
import json
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException, status, Response
from pydantic import BaseModel, ConfigDict
from pathlib import Path
import httpx

from load_json import run_ingestion_loop, load_config
from ai_summary import generate_summary


async def fetch_all_pages(client: httpx.AsyncClient, url: str) -> dict:
    """Collect all entries across HAPI pages and return them as a single merged bundle."""
    all_entries = []
    next_url: str | None = url
    while next_url:
        resp = await client.get(next_url)
        if resp.status_code != 200:
            break
        bundle = resp.json()
        all_entries.extend(bundle.get("entry", []))
        # HAPI signals more pages with a link entry where relation == "next".
        next_url = next(
            (lnk["url"] for lnk in bundle.get("link", []) if lnk.get("relation") == "next"),
            None,
        )
    return {"entry": all_entries}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_OUTPUT_FILE = str(_PROJECT_ROOT / "json_out" / "ai_summary.json")
METRICS_JSON_FILE = str(_PROJECT_ROOT / "json_out" / "ai_summary_metrics.json")
METRICS_CSV_FILE = str(_PROJECT_ROOT / "csv_out" / "ai_summary_metrics.csv")

# Module-level handle so the lifespan shutdown hook can cancel the ingestion task.
ingestion_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_task
    # create_task schedules the ingestion loop as a background task so it runs alongside the HTTP server in the same event loop.
    ingestion_task = asyncio.create_task(run_ingestion_loop())
    yield
    if ingestion_task:
        ingestion_task.cancel()


app = FastAPI(
    title="FHIR Patient History API",
    lifespan=lifespan
)

# Pydantic models define the shape of the API response — FastAPI uses them to validate and serialize the output.

# Minimal shape for conditions and allergies: the display name and the FHIR resource reference (e.g. Condition/abc123).
class ResourceDetail(BaseModel):
    display: str
    source: str

# Extends ResourceDetail with medication-specific fields; all three extras are Optional because only MedicationRequest has them.
class MedicationDetail(BaseModel):
    display: str
    source: str
    status: Optional[str] = None       # e.g. "active", "stopped", "completed", "on-hold"
    authoredOn: Optional[str] = None   # date the prescription was originally written
    practioner: Optional[str] = None   # prescribing provider name (intentional typo matches field in data)

# Top-level response model; extra="allow" lets config-driven fields pass through without a schema change.
class PatientHistoryResponse(BaseModel):
    patient_id: str
    conditions: List[ResourceDetail] = []
    active_medications: List[MedicationDetail] = []
    historical_medications: List[MedicationDetail] = []
    allergies: List[ResourceDetail] = []
    summary: Optional[str] = None      # AI-generated prior-auth narrative from Ollama
    missing: List[str] = []            # flags resource types that returned zero results


# Registers GET /fhir/Patient/{patient_id} with FastAPI. 
# The decorators tell FastAPI to validate the output against PatientHistoryResponse
# response_model_exclude_none=True strips None fields from the output so the JSON stays clean.
@app.get(
    "/fhir/Patient/{patient_id}",
    response_model=PatientHistoryResponse,
    response_model_exclude_none=True
)
async def get_patient_history(
        patient_id: str,
        model: Optional[str] = Query(default="llama3.2:3b"),  # LLM model section from llm_param.ini
        _pretty: Optional[bool] = Query(default=True)          # ?_pretty=false returns compact JSON
):
    """Looks up a patient in HAPI, fetches their conditions, medications, and allergies,
    calls Ollama to generate a prior-auth summary, and returns the full assembled packet."""
    # Re-read config on each request so HAPI URL changes take effect without restarting the server.
    config = load_config()
    hapi_url = config.get("hapi_url", "http://localhost:8080/fhir")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            patient_resp = await client.get(f"{hapi_url}/Patient/{patient_id}")

            if patient_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "PatientNotFound",
                        "message": f"Patient '{patient_id}' not found.",
                        "status": "Check if ingestion has completed for this file."
                    }
                )

            # Fetch all pages for each resource type; _count=1000 reduces round-trips for large histories.
            cond_bundle = await fetch_all_pages(client, f"{hapi_url}/Condition?patient=Patient/{patient_id}&_count=1000")
            med_bundle = await fetch_all_pages(client, f"{hapi_url}/MedicationRequest?patient=Patient/{patient_id}&_count=1000")
            alg_bundle = await fetch_all_pages(client, f"{hapi_url}/AllergyIntolerance?patient=Patient/{patient_id}&_count=1000")

        except httpx.HTTPError:
            # Covers ConnectError (HAPI not running) and ReadTimeout (HAPI too slow to respond).
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to connect to local HAPI FHIR server."
            )

    def parse_bundle(bundle_json, resource_type):
        items = []
        if not bundle_json or "entry" not in bundle_json:
            return items

        for entry in bundle_json.get("entry", []):
            resource = entry.get("resource", {})
            res_id = resource.get("id", "")

            display_name = "Unknown"
            code_obj = resource.get("code") or resource.get("medicationCodeableConcept") or {}
            if "text" in code_obj:
                display_name = code_obj["text"]
            elif "coding" in code_obj and len(code_obj["coding"]) > 0:
                display_name = code_obj["coding"][0].get("display", "Unknown")

            item = {"display": display_name, "source": f"{resource_type}/{res_id}"}

            if resource_type == "MedicationRequest":
                item["status"] = resource.get("status", "")
                item["authoredOn"] = resource.get("authoredOn", "")
                item["practioner"] = resource.get("requester", {}).get("display", "")

            items.append(item)
        return items

    conditions = parse_bundle(cond_bundle, "Condition")
    medications = parse_bundle(med_bundle, "MedicationRequest")
    allergies = parse_bundle(alg_bundle, "AllergyIntolerance")

    # != "active" catches stopped, completed, on-hold, and unknown status so nothing falls through to the wrong list.
    active_medications = [m for m in medications if m.get("status") == "active"]
    historical_medications = [m for m in medications if m.get("status") != "active"]

    missing_list = []
    if not conditions:
        missing_list.append("No conditions on file")
    if not medications:
        missing_list.append("No medications on file")
    if not allergies:
        missing_list.append("No active allergies logged")

    # Call Ollama with the parsed clinical lists; capture metrics for eval output files.
    summary_text, metrics = await generate_summary(conditions, active_medications, historical_medications, allergies, model or "llama3.2:3b")

    # Assemble the full response dict; PatientHistoryResponse validates and serializes it on return.
    base_response = {
        "patient_id": patient_id,
        "conditions": conditions,
        "active_medications": active_medications,
        "historical_medications": historical_medications,
        "allergies": allergies,
        "summary": summary_text,
        "missing": missing_list
    }

    # Write eval artifacts on every request — browser, curl, or eval.py all produce the same output.
    os.makedirs(os.path.dirname(PAYLOAD_OUTPUT_FILE), exist_ok=True)
    with open(PAYLOAD_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(base_response, f, indent=2)

    os.makedirs(os.path.dirname(METRICS_JSON_FILE), exist_ok=True)
    with open(METRICS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump({"patient_id": patient_id, "performance_metrics": metrics}, f, indent=2)

    csv_row = {"patient_id": patient_id, **metrics}
    os.makedirs(os.path.dirname(METRICS_CSV_FILE), exist_ok=True)
    file_exists = os.path.isfile(METRICS_CSV_FILE)
    with open(METRICS_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    response_obj = PatientHistoryResponse(**base_response)

    if _pretty:
        return Response(content=response_obj.model_dump_json(indent=2), media_type="application/json")

    return response_obj


if __name__ == "__main__":
    import uvicorn

    # uvicorn runs the fast api app and listens for incoming HTTP requests
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
