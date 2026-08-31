import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException, status, Response
from pydantic import BaseModel, ConfigDict
import httpx

from load_json import run_ingestion_loop, load_config
from ai_summary import generate_summary

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
    model_config = ConfigDict(extra="allow")
    patient_id: str
    original_patient_id: Optional[str] = None  # Synthea UUID recovered from the Patient identifier array
    hapi_patient_id: str                        # HAPI's internal resource ID used for all sub-queries
    conditions: List[ResourceDetail] = []
    active_medications: List[MedicationDetail] = []
    historical_medications: List[MedicationDetail] = []
    allergies: List[ResourceDetail] = []
    summary: Optional[str] = None      # AI-generated prior-auth narrative from Ollama
    missing: List[str] = []            # flags resource types that returned zero results


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

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            patient_resource = None

            # Try the ID as a direct HAPI resource ID first; fall back to a FHIR identifier search so both ID formats work.
            patient_resp = await client.get(f"{hapi_url}/Patient/{patient_id}")

            if patient_resp.status_code == 200:
                patient_resource = patient_resp.json()
            else:
                # ?identifier= searches every Patient's identifier array for a matching value (covers Synthea UUIDs).
                search_resp = await client.get(f"{hapi_url}/Patient?identifier={patient_id}")
                search_data = search_resp.json() if search_resp.status_code == 200 else {}

                if search_data.get("total", 0) > 0:
                    patient_resource = search_data["entry"][0]["resource"]
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail={
                            "error": "PatientNotFound",
                            "message": f"Patient '{patient_id}' not found by Resource ID or Identifier.",
                            "status": "Check if ingestion has completed for this file."
                        }
                    )

            # resolved_id is HAPI's internal ID — used for all sub-queries regardless of how the patient was looked up.
            resolved_id = patient_resource.get("id", patient_id)

            # Walk the identifier array to recover the original Synthea UUID (hyphens indicate UUID format).
            original_uuid = None
            for ident in patient_resource.get("identifier", []):
                val = ident.get("value")
                if val and ("-" in val or ident.get("system") == "urn:ietf:rfc:3986"):
                    original_uuid = val
                    break

            # Fetch the three clinical resource types linked to this patient using FHIR search syntax.
            cond_resp = await client.get(f"{hapi_url}/Condition?patient=Patient/{resolved_id}")
            med_resp = await client.get(f"{hapi_url}/MedicationRequest?patient=Patient/{resolved_id}")
            alg_resp = await client.get(f"{hapi_url}/AllergyIntolerance?patient=Patient/{resolved_id}")

        except httpx.ConnectError:
            # Raised when HAPI is not running; return 503 with a clear message instead of an internal 500.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to connect to local HAPI FHIR server."
            )

    def parse_bundle(bundle_json, resource_type):
        """Extracts display name and source reference from each entry in a HAPI search bundle,
        adding status/authoredOn/prescriber for MedicationRequest resources."""
        # Defined inside the route handler so it's co-located with the code that uses it; only this route needs it.
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

    # Only parse the body if HAPI returned 200; pass {} otherwise so parse_bundle returns an empty list safely.
    conditions = parse_bundle(cond_resp.json() if cond_resp.status_code == 200 else {}, "Condition")
    medications = parse_bundle(med_resp.json() if med_resp.status_code == 200 else {}, "MedicationRequest")
    allergies = parse_bundle(alg_resp.json() if alg_resp.status_code == 200 else {}, "AllergyIntolerance")

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

    # Call Ollama with the parsed clinical lists; returns summary string + performance metrics (metrics discarded here).
    summary_text, _ = await generate_summary(conditions, active_medications, historical_medications, allergies, model or "llama3.2:3b")

    # Assemble the full response dict; PatientHistoryResponse validates and serializes it on return.
    base_response = {
        "patient_id": patient_id,
        "original_patient_id": original_uuid if original_uuid else patient_id,
        "hapi_patient_id": resolved_id,
        "conditions": conditions,
        "active_medications": active_medications,
        "historical_medications": historical_medications,
        "allergies": allergies,
        "summary": summary_text,
        "missing": missing_list
    }

    # Returning a raw Response with indent=2 bypasses FastAPI's compact serializer so the output is human-readable by default.
    if _pretty:
        return Response(content=json.dumps(base_response, indent=2), media_type="application/json")

    return base_response


if __name__ == "__main__":
    import uvicorn

    # "main:app" tells uvicorn to import the app object from this module; reload=True restarts on file changes.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
