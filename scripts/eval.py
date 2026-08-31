import httpx
import json

# Thin CLI wrapper — all pipeline logic lives in main.py.
# Prerequisite: main.py must be running (python scripts/main.py).

BASE_URL = "http://127.0.0.1:8000"


def run_eval(patient_id: str, model: str) -> None:
    url = f"{BASE_URL}/fhir/Patient/{patient_id}?model={model}"
    print(f"\nCalling: {url}\n")

    # Ollama inference on CPU can take several seconds — keep timeout generous.
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(url)

    if resp.status_code == 404:
        print(f"Patient '{patient_id}' not found. Check that ingestion has completed.")
        return

    if resp.status_code == 503:
        print("HAPI FHIR server is unreachable. Is Docker running?")
        return

    resp.raise_for_status()

    print(json.dumps(resp.json(), indent=2))
    # Artifacts are written by the server as a side effect of the request — no extra steps needed.
    print("\nEval artifacts written to json_out/ and csv_out/ by the server.")


if __name__ == "__main__":
    patient_id = input("Enter patient ID: ").strip()
    model = input("Enter model section (default: llama3.2:3b): ").strip() or "llama3.2:3b"

    if patient_id:
        run_eval(patient_id, model)
