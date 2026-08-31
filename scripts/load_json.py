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
    # Resolve input/processed dirs relative to project root
    for key in ("input_dir", "processed_dir"):
        cfg[key] = str(PROJECT_ROOT / cfg[key].lstrip("./"))
    return cfg


def preserve_synthea_ids(bundle_data):
    """Convert POST entries to PUT so HAPI keeps the Synthea UUIDs as resource IDs.

    fullUrl stays as urn:uuid:... so intra-bundle cross-references keep resolving.
    Only request.method/url change to tell HAPI which ID to store the resource under.
    """
    for entry in bundle_data.get("entry", []):
        full_url = entry.get("fullUrl", "")
        if not full_url.startswith("urn:uuid:"):
            continue
        resource_id = full_url[len("urn:uuid:"):]
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")
        if not resource_type:
            continue
        resource["id"] = resource_id
        request = entry.setdefault("request", {})
        request["method"] = "PUT"
        request["url"] = f"{resource_type}/{resource_id}"
    return bundle_data


async def upload_and_manage_file(client, file_path, semaphore, config):
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
            print(f"[Ingestion Exception] Failed processing {file_name}: {e}")


async def run_ingestion_loop():
    """Continuous polling loop that watches for incoming files."""
    print("[Ingestion Engine] Service started. Polling directory...")

    while True:
        config = load_config()
        input_dir = config.get("input_dir", "./data/input")
        poll_interval = config.get("poll_interval_seconds", 5)
        concurrency = config.get("max_concurrent_uploads", 10)

        os.makedirs(input_dir, exist_ok=True)
        files = glob.glob(os.path.join(input_dir, "*.json"))

        if files:
            print(f"[Ingestion Engine] Found {len(files)} new files to process.")
            semaphore = asyncio.Semaphore(concurrency)

            async with httpx.AsyncClient(timeout=120.0) as client:
                tasks = [upload_and_manage_file(client, f, semaphore, config) for f in files]
                await asyncio.gather(*tasks)

        # Sleep before checking for new files again
        await asyncio.sleep(poll_interval)