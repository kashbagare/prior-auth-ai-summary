# prior-auth-context

AI-driven clinical data service that assembles cited prior-authorization context from a FHIR feed. Every condition, medication, and allergy in the response links back to its FHIR resource ID — a reviewer can verify any line against the source record directly.

Built for the Autonomize FDE technical assignment.

---

## Architecture

```
data/input/*.json ──► load_json.py (ETL) ──► HAPI FHIR :8080 ──► data/processed/
config.json       ──►      (polling)                │   │
                                                    │   │ query
HTTP Client ──► main.py FastAPI :8000 ──────────────┘   │
                                                        │
                    ai_summary.py ◄─────────────────────┘
                          │  ▲
                          │  └── llm_param.ini
                          ▼
                    Ollama :11434
                          │
                    json_out/ · csv_out/
```

Two processes must stay running continuously in the background:

- **Docker (HAPI FHIR)** — the FHIR data store, launched by `setup.sh`
- **main.py** — the FastAPI server and the ETL ingestion loop, run together

`ai_summary.py` is a standalone CLI invoked on demand per patient.

---

## Setup

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd prior-auth-ai-summary
```

### Step 2 — Pull Ollama models and start HAPI FHIR

> **Prerequisite:** Docker Desktop must be installed and running before this step. Download it at [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

Open a dedicated terminal window and run:

```bash
sh setup.sh
```

This does three things in sequence:

1. Pulls `llama3.2:3b`, `qwen3:4b`, `gemma3:4b`, and `phi4-mini:3.8b` via Ollama
2. Frees port 8080 if anything is holding it
3. Pulls and runs the `hapiproject/hapi:latest` Docker image on port 8080

**Keep this terminal open** — the Docker container must stay alive.

**Verify:** Open [http://localhost:8080](http://localhost:8080) — you should see the HAPI FHIR web UI.

> To run HAPI FHIR in the background instead: `docker run -d -p 8080:8080 hapiproject/hapi:latest`

### Step 3 — Start FastAPI + ETL pipeline

Open a second terminal window:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/main.py
```

This starts two concurrent services:

- **Ingestion loop** — polls `data/input/` every 5 seconds for new Synthea bundles, uploads them to HAPI FHIR, then moves each file to `data/processed/`
- **REST API** — serves patient history at `GET /fhir/Patient/{patient_id}/_history/{version_id}` on port 8000

Drop any `.json` Synthea bundle into `data/input/` and the log confirms upload and file move.

**Keep this terminal open.**

### Step 4 — Generate an AI summary (per patient)

Open a third terminal window:

```bash
source venv/bin/activate
python scripts/ai_summary.py
```

Enter a patient ID when prompted. Optionally enter a model section name (default: `llama3.2:3b`).

Output written to:
| File | Contents |
|------|----------|
| `json_out/ai_summary.json` | Full patient packet with AI summary |
| `json_out/ai_summary_metrics.json` | Token counts, latency, tokens/sec |
| `csv_out/ai_summary_metrics.csv` | Same metrics in tabular form |

---

## API Reference

```
GET /fhir/Patient/{patient_id}/_history/{version_id}
```

Resolves a patient by HAPI internal resource ID first, then falls back to a UUID identifier search. Returns structured clinical context with every item sourced to a FHIR resource.

Medications are split into `active_medications` and `historical_medications` so reviewers and the LLM can immediately distinguish current treatment from past prescriptions — important for prior auth decisions like step therapy verification.

**Example response:**

```json
{
  "patient_id": "2fa15bc7-8866-461a-9000-f739e425860a",
  "hapi_patient_id": "2fa15bc7-8866-461a-9000-f739e425860a",
  "encounter": {
    "encounter_id": "Encounter/enc-2fa15bc7",
    "type": "Ambulatory / Outpatient",
    "facility_name": "General Hospital Clinic A",
    "provider_name": "Dr. Jane Doe, MD"
  },
  "conditions": [{ "display": "Diabetes", "source": "Condition/1a2b3c" }],
  "active_medications": [
    { "display": "Metformin 500 MG", "source": "MedicationRequest/3c4d5e", "status": "active", "authoredOn": "1951-10-22", "practioner": "Dr. Jane Doe" }
  ],
  "historical_medications": [
    { "display": "Simvastatin 10 MG", "source": "MedicationRequest/9z8y7x", "status": "stopped", "authoredOn": "1965-09-06", "practioner": "Dr. John Smith" }
  ],
  "allergies": [
    { "display": "Penicillin", "source": "AllergyIntolerance/7f8g9h" }
  ],
  "summary": "Two-sentence summary a reviewer can scan.",
  "missing": []
}
```

---

## Configuration

### `config_dir/config.json`

| Key                      | Default                      | Purpose                             |
| ------------------------ | ---------------------------- | ----------------------------------- |
| `input_dir`              | `./data/input`               | Directory polled for new bundles    |
| `processed_dir`          | `./data/processed`           | Destination after successful upload |
| `delete_after_upload`    | `false`                      | Delete instead of move on success   |
| `poll_interval_seconds`  | `5`                          | How often to scan for new files     |
| `max_concurrent_uploads` | `10`                         | Semaphore cap on parallel uploads   |
| `hapi_url`               | `http://localhost:8080/fhir` | HAPI FHIR base URL                  |

### `config_dir/llm_param.ini`

Each `[section]` corresponds to a model identifier passed to `ai_summary.py`. Parameters in `[DEFAULT]` are inherited by all sections. Add a new model by adding a new section — no code changes required.

Available sections: `llama3.2:3b`, `qwen3:4b`, `gemma3:4b`, `phi4-mini:3.8b`

---

## Model Comparison

All four models received the same prompt template. Tested on available Synthea patients on Apple Silicon (no GPU).

| Model            | Params | Summary quality  | JSON compliance                   | Speed    | Notes                                           |
| ---------------- | ------ | ---------------- | --------------------------------- | -------- | ----------------------------------------------- |
| `llama3.2:3b`    | 3B     | Concise, factual | Reliable                          | Fastest  | Best default; minimal post-processing           |
| `qwen3:4b`       | 4B     | **Not obtained** | **Failed** — emits `<think>` tags | n/a      | Cut after repeated `JSONDecodeError`; see below |
| `gemma3:4b`      | 4B     | Natural phrasing | Good                              | Moderate | Occasionally over-explains beyond two sentences |
| `phi4-mini:3.8b` | 3.8B   | Precise          | Good                              | Moderate | Strong instruction-following; clean JSON        |

**Chosen default: `llama3.2:3b`** — fastest on CPU, reliably outputs valid JSON within the two-sentence constraint, and smallest footprint for a laptop-only deployment.

### Why `qwen3:4b` was cut

Qwen emits `<think>…</think>` reasoning blocks and markdown fences even with `format="json"` set, producing `JSONDecodeError` on every run despite three mitigations — a system prompt, a `/no_think` prefix, and multi-tier regex extraction.

- The next experiment would be Ollama's structured-output mode with an explicit JSON schema, which constrains decoding at the tokenizer level rather than just requesting a format.

---

## Observations on 4 Patients

Reviewed a sample of Synthea patients after ingestion. For each: (1) does the summary reflect only what is on file? (2) does each `source` field resolve to a real FHIR resource?

| Patient          | Conditions | Active Meds | Historical Meds | Allergies | Summary accurate? | Sources valid? | Notes                                                                                  |
| ---------------- | ---------- | ----------- | --------------- | --------- | ----------------- | -------------- | -------------------------------------------------------------------------------------- |
| 0718123b         | 20         | 3           | 17              | 0         | Yes               | Yes            | 17 stopped Simvastatin renewals correctly separated from 3 active medications          |
| 74d801e7         | 10         | 2           | 5               | 1         | Yes               | Yes            | Multi-condition patient (chronic pain, migraine, drug overdose); summary scoped correctly; wheat allergy surfaced |
| c088b7af         | 5          | 4           | 0               | 5         | Yes               | Yes            | Complex allergy profile (latex, mould, dust mites, dander, tree pollen) fully captured |
| 10c09023         | 7          | 0           | 0               | 0         | Yes               | Yes            | No medications or allergies on file; correctly flagged in `missing` field              |

**All source fields verified** by resolving `Condition/{id}`, `MedicationRequest/{id}`, and `AllergyIntolerance/{id}` directly against HAPI FHIR. No hallucinated conditions or medications observed across any run.

**One limitation noted:** `MedicationRequest` returns all statuses including `stopped`. A prior-auth reviewer primarily cares about active medications — filtering by `?status=active` would sharpen summaries for patients with long prescription histories.

---

## Technical Decisions

**Why preserve Synthea UUIDs on ingestion?**
HAPI FHIR auto-generates internal IDs on `POST`. By converting bundle entries to `PUT` with the Synthea UUID as the resource ID, the `source` field in every response remains stable and matches the original synthetic record. This makes the clinical citation auditable — a reviewer following a source link gets the right resource.

**Why a polling loop over a filesystem watcher?**
`watchdog`-style event listeners add an OS-level dependency and can miss events under load. A simple `asyncio` polling loop with a configurable interval is predictable, restart-safe, and tunable via `config.json` without code changes.

**Why `.ini` for model configuration?**
INI's `[DEFAULT]` section lets every model section inherit the shared prompt while overriding only what differs. Adding a new model is one block — no JSON nesting, no code edits.

**Why multi-tier JSON extraction in the parser?**
Small models occasionally emit markdown fences, preamble text, or (in Qwen3's case) `<think>` reasoning blocks before the JSON. The extraction pipeline strips each layer before `json.loads`, with regex fallbacks so a partially-formed response yields a usable summary rather than an exception that silently loses the result.

---

## Issues Encountered

The problems worth recording are the ones that changed the design.

| Symptom                                                       | Root cause                                                                                                         | Fix                                                                                                            |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Patient UUID `7978d71c…` became `1001` after upload           | HAPI assigns sequential internal IDs on `POST`                                                                     | Rewrote bundle entries to `PUT {ResourceType}/{uuid}` so HAPI stores resources under the Synthea UUID          |
| UUID lookup worked but integer ID did not (and vice versa)    | Only one lookup path existed                                                                                       | Primary lookup by HAPI resource ID, fallback search by `?identifier=`                                          |
| Ingestion loop stopped after a single bad file                | Unhandled exception escaped the per-file coroutine                                                                 | Per-file `try/except` so one malformed bundle cannot kill the watcher                                          |
| `JSONDecodeError` on `qwen3:4b` and occasionally other models | Qwen emits `<think>` blocks and markdown fences; all models can emit unescaped newlines inside string values       | Multi-tier extractor: strip thinking tags → strip fences → isolate outermost `{…}` → escape control characters |
| Medication status and prescriber absent from summaries        | `parse_bundle()` discarded `status`, `authoredOn`, and `requester.display`; prompt formatter then dropped the rest | `fmt_medications()` renders all three fields into the prompt explicitly                                        |

---

## Next Steps

The three most impactful next steps are decoupling ingestion from the API into independent services, replacing directory polling with Kafka for horizontal scale, and adding a PHI de-identification stage before any text reaches the model. Filtering to active medications via `MedicationRequest?status=active` would also sharpen summaries — one patient carried 16 obsolete Simvastatin records against 3 active medications.

---

## Attribution

The assignment asks where the code came from. Stated plainly:

### My own work

Architecture and component boundaries; the decision to preserve Synthea UUIDs through ingestion and why it matters for citation integrity; configuration design for both `config.json` and `llm_param.ini` (including the `[DEFAULT]`-inheritance pattern that makes adding a model a config change); prompt engineering and per-model tuning; model selection, benchmarking, and the clinical evaluation of sampled patients.

### AI assistance — Google Gemini

Roughly **20–25 prompts**, used as a debugging partner rather than a code generator. Nearly all of them map directly to entries in the [Issues Encountered](#issues-encountered) section above — HAPI reassigning resource IDs on `POST`, the Qwen3 `<think>`-tag parsing failures, the FastAPI route template mismatch, the Ollama read timeout, and the two-layer metadata truncation between `parse_bundle()` and the prompt formatter.

The pattern was consistent: I identified the symptom and the constraint, Gemini helped narrow the cause and draft the patch, and I decided whether the fix belonged in the design. The multi-tier JSON extractor is the clearest example — the regex layers came out of that back-and-forth, but the decision to degrade gracefully rather than raise (so a completed inference is never discarded) was mine.

### Open source

| Component               | Source                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| Synthetic patient data  | [Synthea](https://synthea.mitre.org/) (MITRE)                                                  |
| FHIR server             | [HAPI FHIR](https://hapifhir.io/)                                                              |
| Local inference runtime | [Ollama](https://ollama.com/)                                                                  |
| Models                  | `llama3.2:3b` (Meta), `qwen3:4b` (Alibaba), `gemma3:4b` (Google), `phi4-mini:3.8b` (Microsoft) |
| Service layer           | [FastAPI](https://fastapi.tiangolo.com/), Pydantic, httpx, uvicorn                             |
