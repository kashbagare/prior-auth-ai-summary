import asyncio
import json
import csv
import configparser
import time
from string import Template
import httpx
import re

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent

HAPI_URL = "http://localhost:8080/fhir"
OLLAMA_URL = "http://localhost:11434/api/generate"
CONFIG_FILE = str(_PROJECT_ROOT / "config_dir" / "llm_param.ini")

PAYLOAD_OUTPUT_FILE = str(_PROJECT_ROOT / "json_out" / "ai_summary.json")
METRICS_JSON_FILE = str(_PROJECT_ROOT / "json_out" / "ai_summary_metrics.json")
METRICS_CSV_FILE = str(_PROJECT_ROOT / "csv_out" / "ai_summary_metrics.csv")


def extract_json_payload(raw_response: str) -> str:
    """Strips thinking tags, markdown formatting, and prelude text to isolate valid JSON."""
    if not raw_response:
        return ""

    cleaned = raw_response.strip()

    # 1. Remove reasoning/thinking tags emitted by models like Qwen3
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # 2. Remove markdown code fence blocks if present
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()

    # 3. Extract JSON object bounded by outermost { and }
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0).strip()

    # 4. Escape raw unescaped newlines/tabs inside string values that break json.loads
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
    cleaned = re.sub(r'(?<!\\)\t', r'\\t', cleaned)

    return cleaned


def parse_summary_from_response(raw_text: str) -> str:
    """Multi-tiered parser ensuring all models return a string summary without failing."""
    sanitized_str = extract_json_payload(raw_text)

    # Attempt 1: Standard JSON parsing
    try:
        data = json.loads(sanitized_str)
        if isinstance(data, dict) and "summary" in data:
            return str(data["summary"]).strip()
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strict regex extraction for "summary": "..."
    match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_text, re.DOTALL)
    if match:
        return match.group(1).replace('\\n', ' ').strip()

    # Attempt 3: Fallback — Strip thinking tags and return raw text as summary directly
    clean_fallback = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    clean_fallback = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", clean_fallback, flags=re.DOTALL).strip()
    if clean_fallback:
        return clean_fallback

    return "Summary unavailable."


def load_model_config(section_name: str) -> tuple[dict, str]:
    """Loads configuration options and prompt template for a specific model section."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if section_name not in config:
        raise ValueError(f"Section [{section_name}] not found in {CONFIG_FILE}")

    section = config[section_name]

    options = {
        "temperature": float(section.get("temperature", 0.1)),
        "top_p": float(section.get("top_p", 0.9)),
        "top_k": int(section.get("top_k", 20)),
        "num_predict": int(section.get("num_predict", 256)),
        "presence_penalty": float(section.get("presence_penalty", 0.0)),
        "frequency_penalty": float(section.get("frequency_penalty", 0.0))
    }

    prompt_template = section.get("prompt")
    model_name = section.get("model", section_name)

    return {"model": model_name, "options": options}, prompt_template


def parse_bundle(bundle: dict, resource_type: str) -> list[dict]:
    items = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        res_id = resource.get("id", "")
        code_obj = (
                resource.get("code")
                or resource.get("medicationCodeableConcept")
                or {}
        )
        if "text" in code_obj:
            display = code_obj["text"]
        elif code_obj.get("coding"):
            display = code_obj["coding"][0].get("display", "Unknown")
        else:
            display = "Unknown"

        item = {
            "display": display,
            "source": f"{resource_type}/{res_id}"
        }

        # Include additional detail fields for MedicationRequest resources
        if resource_type == "MedicationRequest":
            item["status"] = resource.get("status", "")
            item["authoredOn"] = resource.get("authoredOn", "")
            item["practioner"] = resource.get("requester", {}).get("display", "")

        items.append(item)
    return items


async def generate_summary(conditions: list, active_meds: list, historical_meds: list, allergies: list, model_section: str = "llama3.2:3b") -> tuple[str, dict]:
    def fmt_conditions_or_allergies(items):
        return ", ".join(i["display"] for i in items) if items else "none"

    def fmt_med_list(items):
        if not items:
            return "none"
        formatted_meds = []
        for m in items:
            details = m["display"]
            meta = []
            if m.get("authoredOn"):
                meta.append(f"Authored: {m['authoredOn']}")
            if m.get("practioner"):
                meta.append(f"Prescriber: {m['practioner']}")
            if meta:
                details += f" ({', '.join(meta)})"
            formatted_meds.append(details)
        return "; ".join(formatted_meds)

    def fmt_medications(active, historical):
        return (
            f"Active medications: {fmt_med_list(active)} | "
            f"Historical medications (stopped, completed, on-hold): {fmt_med_list(historical)}"
        )

    model_params, prompt_template = load_model_config(model_section)

    tmpl = Template(
        prompt_template.replace("{conditions}", "$conditions")
        .replace("{medications}", "$medications")
        .replace("{allergies}", "$allergies")
    )

    formatted_prompt = tmpl.substitute(
        conditions=fmt_conditions_or_allergies(conditions),
        medications=fmt_medications(active_meds, historical_meds),
        allergies=fmt_conditions_or_allergies(allergies)
    )

    payload = {
        "model": model_params["model"],
        "prompt": formatted_prompt,
        "system": "You are a JSON generator. You must output raw JSON only matching the exact schema requested. Do not output thinking tags or commentary.",
        "format": "json",
        "stream": False,
        "options": model_params["options"]
    }

    start_time = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()

        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        res_data = resp.json()

        raw_text = res_data.get("response", "")

        # Resilient multi-tier extraction
        summary_text = parse_summary_from_response(raw_text)

        # Extract timing and token metrics
        prompt_tokens = res_data.get("prompt_eval_count", 0)
        eval_tokens = res_data.get("eval_count", 0)
        total_tokens = prompt_tokens + eval_tokens

        total_duration_ms = round(res_data.get("total_duration", 0) / 1e6, 2)
        eval_duration_ms = round(res_data.get("eval_duration", 0) / 1e6, 2)

        metrics = {
            "model_section": model_section,
            "ollama_model": res_data.get("model", model_params["model"]),
            "wall_clock_time_seconds": elapsed_seconds,
            "ollama_total_duration_ms": total_duration_ms,
            "ollama_eval_duration_ms": eval_duration_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": eval_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(eval_tokens / (eval_duration_ms / 1000), 2) if eval_duration_ms > 0 else 0
        }

        return summary_text, metrics

    except Exception as exc:
        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        print(f"\n[DEBUG Error for {model_section}]: {exc}")

        if 'raw_text' in locals():
            print(f"[DEBUG Raw Response]: {raw_text}\n")

        metrics = {
            "model_section": model_section,
            "ollama_model": model_params["model"],
            "wall_clock_time_seconds": elapsed_seconds,
            "error": str(exc)
        }
        return f"Summary unavailable ({type(exc).__name__}).", metrics


async def fetch_patient(patient_id: str, model_section: str = "llama3.2:3b"):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{HAPI_URL}/Patient/{patient_id}")
        if resp.status_code == 404:
            print(f"Patient '{patient_id}' not found.")
            return
        resp.raise_for_status()

        cond_resp = await client.get(f"{HAPI_URL}/Condition?patient={patient_id}")
        med_resp = await client.get(f"{HAPI_URL}/MedicationRequest?patient={patient_id}")
        alg_resp = await client.get(f"{HAPI_URL}/AllergyIntolerance?patient={patient_id}")

    conditions = parse_bundle(cond_resp.json() if cond_resp.status_code == 200 else {}, "Condition")
    medications = parse_bundle(med_resp.json() if med_resp.status_code == 200 else {}, "MedicationRequest")
    allergies = parse_bundle(alg_resp.json() if alg_resp.status_code == 200 else {}, "AllergyIntolerance")

    active_meds = [m for m in medications if m.get("status") == "active"]
    historical_meds = [m for m in medications if m.get("status") != "active"]

    print(f"Generating summary via Ollama using [{model_section}]...")
    summary, metrics = await generate_summary(conditions, active_meds, historical_meds, allergies, model_section)

    missing = []
    if not conditions:
        missing.append("No conditions on file")
    if not medications:
        missing.append("No medications on file")
    if not allergies:
        missing.append("No active allergies logged")

    packet = {
        "patient_id": patient_id,
        "conditions": conditions,
        "active_medications": active_meds,
        "historical_medications": historical_meds,
        "allergies": allergies,
        "summary": summary,
        "missing": missing,
    }

    with open(PAYLOAD_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)

    metrics_packet = {
        "patient_id": patient_id,
        "performance_metrics": metrics
    }

    with open(METRICS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_packet, f, indent=2)

    csv_row = {"patient_id": patient_id, **metrics}

    with open(METRICS_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)

    print(f"\n--- Output Files Generated ---")
    print(f"Payload JSON saved to : {PAYLOAD_OUTPUT_FILE}")
    print(f"Metrics JSON saved to : {METRICS_JSON_FILE}")
    print(f"Metrics CSV saved to  : {METRICS_CSV_FILE}\n")
    print(json.dumps(packet, indent=2))


if __name__ == "__main__":
    patient_id = input("Enter patient ID: ").strip()
    selected_model = input("Enter model section (default: llama3.2:3b): ").strip() or "llama3.2:3b"

    if patient_id:
        asyncio.run(fetch_patient(patient_id, selected_model))