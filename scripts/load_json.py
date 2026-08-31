import asyncio
import glob
import json
import os
import shutil
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config_dir" / "config.json"


def load_config():
    # Falls back to hardcoded defaults if config.json doesn't exist so the service boots without a config file.
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "input_dir": "data/input",
            "processed_dir": "data/processed",
            "delete_after_upload": False,
            "poll_interval_seconds": 5,
            "max_concurrent_uploads": 10,
            "hapi_url": "http://localhost:8080/fhir"
        }
    # Strip the leading "./" and join to PROJECT_ROOT so paths resolve correctly from any working directory.
    for key in ("input_dir", "processed_dir"):
        cfg[key] = str(PROJECT_ROOT / cfg[key].lstrip("./"))
    return cfg


def preserve_synthea_ids(bundle_data):
    """By default, HAPI ignores Synthea's IDs and assigns its own sequential ones (Patient/1, Patient/2...).
    This function rewrites each bundle entry to use PUT instead of POST, and explicitly sets the resource ID
    to the Synthea UUID — forcing HAPI to store the resource under that UUID.
    This matters because the API response includes a source field (e.g. Condition/7978d71c-...) that links
    back to the original record; if HAPI replaced the UUID with its own ID, that link would be wrong.
    """
    for entry in bundle_data.get("entry", []):
        full_url = entry.get("fullUrl", "")
        if not full_url.startswith("urn:uuid:"):
            continue
        # Slice off the "urn:uuid:" prefix to get the bare UUID that HAPI will store as the resource ID.
        resource_id = full_url[len("urn:uuid:"):]
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")
        if not resource_type:
            continue
        resource["id"] = resource_id
        # Grab the existing request dict Synthea put in the entry, then overwrite method and url to use PUT with our UUID.
        request = entry.setdefault("request", {})
        request["method"] = "PUT"
        request["url"] = f"{resource_type}/{resource_id}"
    return bundle_data


async def upload_and_manage_file(client, file_path, semaphore, config):
    # Semaphore blocks here if max_concurrent_uploads slots are taken, preventing HAPI from being overwhelmed.
    async with semaphore:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                bundle_data = json.load(f)

            bundle_data = preserve_synthea_ids(bundle_data)

            # 1. Upload bundle to FHIR server
            response = await client.post(
                config["hapi_url"],
                json=bundle_data,
                headers={"Content-Type": "application/fhir+json"}
            )

            if response.status_code in [200, 201]:
                print(f"[Ingestion] Successfully uploaded {file_name}: Status {response.status_code}")

                # 2. File relocation or deletion logic
                if config.get("delete_after_upload", False):
                    os.remove(file_path)
                    print(f"[Ingestion] Deleted {file_name}")
                else:
                    processed_dir = config["processed_dir"]
                    os.makedirs(processed_dir, exist_ok=True)
                    target_path = os.path.join(processed_dir, file_name)

                    # Move file out of incoming folder
                    shutil.move(file_path, target_path)
                    print(f"[Ingestion] Moved {file_name} -> {target_path}")
            else:
                print(f"[Ingestion Error] Server returned {response.status_code} for {file_name}")
                print(f"[Ingestion Error] Response: {response.text[:500]}")

        except Exception as e:
            # Catch-all so one bad file (malformed JSON, network error) never crashes the ingestion loop.
            print(f"[Ingestion Exception] Failed processing {file_name}: {e}")


async def run_ingestion_loop():
    """Continuous polling loop that watches for incoming files."""
    print("[Ingestion Engine] Service started. Polling directory...")

    while True:
        # Reload config on every iteration so changes to poll interval or directories take effect without a restart.
        config = load_config()
        input_dir = config.get("input_dir", "./data/input")
        poll_interval = config.get("poll_interval_seconds", 5)
        concurrency = config.get("max_concurrent_uploads", 10)

        os.makedirs(input_dir, exist_ok=True)
        files = glob.glob(os.path.join(input_dir, "*.json"))

        if files:
            print(f"[Ingestion Engine] Found {len(files)} new files to process.")
            # Semaphore is a counter starting at max_concurrent_uploads; each upload decrements it and releases it when done.
            semaphore = asyncio.Semaphore(concurrency)

            async with httpx.AsyncClient(timeout=120.0) as client:
                tasks = [upload_and_manage_file(client, f, semaphore, config) for f in files]
                # gather starts all upload coroutines concurrently; the semaphore inside each one enforces the concurrency cap.
                await asyncio.gather(*tasks)

        # Sleep before checking for new files again
        await asyncio.sleep(poll_interval)