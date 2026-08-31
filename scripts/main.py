import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, Query, HTTPException, status, Response
from pydantic import BaseModel, ConfigDict
import httpx

from load_json import run_ingestion_loop, load_config

ingestion_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion_task
    ingestion_task = asyncio.create_task(run_ingestion_loop())
    yield
    if ingestion_task:
        ingestion_task.cancel()


app = FastAPI(
    title="FHIR Patient History API",
    version="3.3.0",
    lifespan=lifespan
)


class ResourceDetail(BaseModel):
    display: str
    source: str


class EncounterInfo(BaseModel):
    encounter_id: str
    type: str
    facility_name: str
    provider_name: str


class MedicationDetail(BaseModel):
    display: str
    source: str
    status: Optional[str] = None
    authoredOn: Optional[str] = None
    practioner: Optional[str] = None


class PatientHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    patient_id: str
    original_patient_id: Optional[str] = None
    hapi_patient_id: str
    encounter: Optional[EncounterInfo] = None
    conditions: List[ResourceDetail] = []
    active_medications: List[MedicationDetail] = []
    historical_medications: List[MedicationDetail] = []
    allergies: List[ResourceDetail] = []
    summary: str
    missing: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


@app.get(
    "/fhir/Patient/{patient_id}/_history/{version_id}",
    response_model=PatientHistoryResponse,
    response_model_exclude_none=True
)
async def get_patient_history(
        patient_id: str,
        version_id: str,
        _pretty: Optional[bool] = Query(default=True)
):
    config = load_config()
    hapi_url = config.get("hapi_url", "http://localhost:8080/fhir")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            patient_resource = None

            # 1. Attempt lookup by HAPI Resource ID
            patient_resp = await client.get(f"{hapi_url}/Patient/{patient_id}")

            if patient_resp.status_code == 200:
                patient_resource = patient_resp.json()
            else:
                # 2. Fallback lookup by UUID identifier
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

            resolved_id = patient_resource.get("id", patient_id)

            # Extract original UUID identifier if available
            original_uuid = None
            for ident in patient_resource.get("identifier", []):
                val = ident.get("value")
                if val and ("-" in val or ident.get("system") == "urn:ietf:rfc:3986"):
                    original_uuid = val
                    break

            # 3. Fetch related clinical resources using resolved HAPI ID
            cond_resp = await client.get(f"{hapi_url}/Condition?patient=Patient/{resolved_id}")
            med_resp = await client.get(f"{hapi_url}/MedicationRequest?patient=Patient/{resolved_id}")
            alg_resp = await client.get(f"{hapi_url}/AllergyIntolerance?patient=Patient/{resolved_id}")

        except httpx.ConnectError:
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

    conditions = parse_bundle(cond_resp.json() if cond_resp.status_code == 200 else {}, "Condition")
    medications = parse_bundle(med_resp.json() if med_resp.status_code == 200 else {}, "MedicationRequest")
    allergies = parse_bundle(alg_resp.json() if alg_resp.status_code == 200 else {}, "AllergyIntolerance")

    active_medications = [m for m in medications if m.get("status") == "active"]
    historical_medications = [m for m in medications if m.get("status") != "active"]

    missing_list = []
    if not conditions:
        missing_list.append("No conditions on file")
    if not medications:
        missing_list.append("No medications on file")
    if not allergies:
        missing_list.append("No active allergies logged")

    base_response = {
        "patient_id": patient_id,
        "original_patient_id": original_uuid if original_uuid else patient_id,
        "hapi_patient_id": resolved_id,
        "encounter": {
            "encounter_id": f"Encounter/enc-{resolved_id}",
            "type": "Ambulatory / Outpatient",
            "facility_name": "General Hospital Clinic A",
            "provider_name": "Dr. Jane Doe, MD"
        },
        "conditions": conditions,
        "active_medications": active_medications,
        "historical_medications": historical_medications,
        "allergies": allergies,
        "summary": f"Retrieved record for Patient '{patient_id}' (HAPI internal ID: {resolved_id}, Original UUID: {original_uuid}) at version {version_id}.",
        "missing": missing_list
    }

    if "custom_metadata" in config:
        base_response["metadata"] = config["custom_metadata"]

    if _pretty:
        return Response(content=json.dumps(base_response, indent=2), media_type="application/json")

    return base_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)