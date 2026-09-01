# prior-auth-context

AI-driven clinical data service that assembles cited prior-authorization context from a FHIR feed. Every condition, medication, and allergy in the response links back to its FHIR resource ID — a reviewer can verify any line against the source record directly.

---

## Architecture

```
Data flow — end to end

  Synthea .json files
       │
       ▼ dropped into
  data/input/
       │
       ▼ polled every 5 s (background task started by lifespan hook)
  load_json.py  ──► preserve_synthea_ids()  ──► POST bundle to HAPI FHIR :8080
                                                        │
                                               data/processed/ (file moved here)

  main.py (FastAPI :8000)
       │  on GET /fhir/Patient/{id}?model=llama3.2:3b
       ├─► paginate HAPI for Condition, MedicationRequest, AllergyIntolerance (_count=1000, follow next links)
       ├─► split medications into active vs historical
       ├─► call ai_summary.generate_summary()
       │         └─► build prompt via llm_param.ini template
       │              (historical meds collapsed by drug name before prompt is built)
       │         └─► POST to Ollama :11434  ──► LLM generates summary
       ├─► return full packet: clinical data + AI summary (all source refs preserved)
       └─► write eval artifacts on every request:
               json_out/ai_summary.json         (full response payload)
               json_out/ai_summary_metrics.json (Ollama performance metrics)
               csv_out/ai_summary_metrics.csv   (appended row — builds comparison table across models)
```

Two processes must stay running:

- **Docker (HAPI FHIR)** — the FHIR data store, launched by `setup.sh`
- **main.py** — the FastAPI server and ingestion loop, run together

---

## Setup

### Step 1 — Clone the repo

```bash
git clone <repo-url>
cd prior-auth-ai-summary
```

### Step 2 — Pull Ollama models and start HAPI FHIR

> **Prerequisites:**
> - **Ollama** must be installed and its daemon running. Download it at [https://ollama.com/download](https://ollama.com/download).
> - **Docker Desktop** must be installed and running. Download it at [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

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
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/main.py
```

This starts two concurrent services:

- **Ingestion loop** — polls `data/input/` every 5 seconds for new Synthea bundles, uploads them to HAPI FHIR, then moves each file to `data/processed/`
- **REST API** — serves patient history with AI summary at `GET /fhir/Patient/{patient_id}` on port 8000

Drop any `.json` Synthea bundle into `data/input/` and the log confirms upload and file move.

**Keep this terminal open.**

### Step 4 — Query a patient

Open a browser or run curl with a patient ID and optional model:

```
http://127.0.0.1:8000/fhir/Patient/{patient_id}?model=llama3.2:3b
```

The response includes conditions, medications (active and historical), allergies, and an AI-generated prior-auth summary — all in one call.

---

## API Reference

```
GET /fhir/Patient/{patient_id}?model=llama3.2:3b&_pretty=true
```

| Parameter    | Default        | Description                                              |
|--------------|----------------|----------------------------------------------------------|
| `patient_id` | —              | HAPI resource ID or Synthea UUID                         |
| `model`      | `llama3.2:3b`  | Model section name from `llm_param.ini`                  |
| `_pretty`    | `true`         | Indent JSON output                                       |

Resolves the patient, fetches conditions/medications/allergies from HAPI, calls Ollama for a summary, and returns everything in one response. Medications are split into `active_medications` and `historical_medications`.

**Example response:**

```json
{
  "patient_id": "2fa15bc7-8866-461a-9000-f739e425860a",
  "conditions": [{ "display": "Diabetes", "source": "Condition/1a2b3c" }],
  "active_medications": [
    {
      "display": "Metformin 500 MG",
      "source": "MedicationRequest/3c4d5e",
      "status": "active",
      "authoredOn": "1951-10-22",
      "practioner": "Dr. Jane Doe"
    }
  ],
  "historical_medications": [
    {
      "display": "Simvastatin 10 MG",
      "source": "MedicationRequest/9z8y7x",
      "status": "stopped",
      "authoredOn": "1965-09-06",
      "practioner": "Dr. John Smith"
    }
  ],
  "allergies": [
    { "display": "Penicillin", "source": "AllergyIntolerance/7f8g9h" }
  ],
  "summary": "Patient is being treated for Diabetes with active Metformin therapy. Step therapy history shows prior Simvastatin use, now stopped.",
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

Each `[section]` maps to a model identifier passed via `?model=`. Parameters in `[DEFAULT]` are inherited by all sections — adding a new model is one config block, no code changes.

Available sections: `llama3.2:3b`, `gemma3:4b`, `phi4-mini:3.8b`, `qwen3:4b`

---

## Model Comparison

Evaluated on patient `0718123b` (Floyd Jerde): 23 conditions, 10 active medications, 1,265 historical medications. Apple Silicon, no GPU.

| Model            | JSON | Factual grounding | Completeness | Conciseness | Clinical utility | Wall clock | Tokens/sec |
| ---------------- | ---- | ----------------- | ------------ | ----------- | ---------------- | ---------- | ---------- |
| `llama3.2:3b`    | Pass | Pass              | 3/3          | Pass        | 3/3              | 3.13 s     | 89.00      |
| `gemma3:4b`      | Pass | Pass              | 2/3          | Pass        | 2/3              | 5.14 s     | 71.10      |
| `phi4-mini:3.8b` | Pass | Pass              | 3/3          | Pass        | 3/3              | 1.17 s     | 73.98      |
| `qwen3:4b`       | Pass | **Fail**          | 3/3          | Pass        | 1/3              | 5.24 s     | 71.01      |

Wall-clock times vary run to run with machine load; tokens/sec is the more stable comparison.

**Chosen default: `llama3.2:3b`**

> *"Patient has multiple chronic conditions including hypertension, diabetes, kidney disease, and osteoarthritis, and is currently taking several medications including metformin, insulin, and metoprolol, with a history of prostate cancer and Alzheimer's disease."*

**Why:** it's the only model that states high-stakes diagnoses directly (prostate cancer, Alzheimer's) rather than requiring the reviewer to infer them from drug names — the safer default for a PA-reviewer audience. It also ties for the top score on every rubric column.

- **`phi4-mini:3.8b`** — ties llama3.2 on every column and is faster, but only implies diagnoses via drug names (DOCEtaxel, Leuprolide).
- **`gemma3:4b`** — passes grounding but inconsistent run to run; sometimes drops the highest-stakes findings entirely.
- **`qwen3:4b`** — routes output to a `thinking` field, not `response` (fixed via fallback); still fails grounding on Simvastatin's active order.

---

## Observations on 2 Patients

### Patient 1 — Floyd Jerde (complex history)

23 conditions, 10 active medications, 1,265 historical medications, 0 allergies.

| Model | Factual grounding | Clinical utility | Key finding |
|---|---|---|---|
| `llama3.2:3b` | Pass | 3/3 | Named prostate cancer and Alzheimer's directly alongside core chronic disease burden |
| `gemma3:4b` | Pass | 2/3 | Run-to-run variance: one run omitted prostate cancer and Alzheimer's entirely, naming "surgical interventions" instead |
| `phi4-mini:3.8b` | Pass | 3/3 | Named all 10 active meds correctly, incl. oncology/dementia drugs (DOCEtaxel, Leuprolide, Donepezil); no hallucinations this run |
| `qwen3:4b` | **Fail** | 1/3 | Listed simvastatin as historical only, omitting its active order; named all 23 conditions but in one unstructured run-on sentence |

### Patient 2 — Alesha Marks (allergy-dominant)

5 conditions, 4 active medications, 0 historical, 5 allergies (latex, mould, dust mites, dander, tree pollen).

| Model | Factual grounding | Clinical utility | Key finding |
|---|---|---|---|
| `llama3.2:3b` | Pass | 2/3 | Named all 4 medications and all 5 allergies by name; omitted viral sinusitis as a separate condition |
| `gemma3:4b` | Pass | 2/3 | Correctly tied hydrochlorothiazide to hypertension; grouped the other 3 drugs by class (corticosteroid/bronchodilator/antihistamine) without naming them |
| `phi4-mini:3.8b` | Pass | 2/3 | Named all 4 medications correctly; no indication mapping |
| `qwen3:4b` | Pass | 2/3 | Named all 4 medications correctly; no indication mapping; omitted the allergy list entirely |

### Key takeaway

**Complexity, not simplicity, is where models separate.**

- All four models pass on Alesha's simple record; only Floyd's harder one differentiates them.
- Two previously-documented failures (gemma3, phi4-mini) didn't reproduce after the prompt/parser fixes.
- `qwen3:4b` still fails on Floyd — reads Simvastatin's active order as historical-only, a data ambiguity, not a fabrication.
- `llama3.2:3b` and `phi4-mini:3.8b` tie on Floyd; llama3.2 states diagnoses directly, phi4-mini is faster.
- All fields verified against HAPI FHIR source.

**Eval scope:** Two patients, developer-scored — a production eval would need a clinician-reviewed golden dataset with automated hallucination checks.

---

## Technical Decisions

**Eval artifacts on every request** — every browser hit or curl call produces the same metrics automatically. No separate eval mode to invoke; CSV appends build a comparison table across model runs without orchestration.

**Historical med grouping in the prompt** — after fixing pagination, Floyd Jerde's 1,265 historical entries doubled the prompt size (1,134 → 2,050 tokens). The model ignored conditions and fixated on one drug. Grouping collapses repeats: `Simvastatin 10 MG (×63, 1965–1980)` instead of 63 lines. Prompt dropped to 796 tokens; quality recovered.

**HAPI pagination** — default page size is 20. Floyd Jerde has 1,275 medication entries — a single un-paginated call silently returned 20 with no error. `fetch_all_pages()` requests `_count=1000` and follows `link[relation=next]` until the bundle is exhausted.

**UUID preservation on ingestion** — HAPI auto-generates IDs on `POST`. Converting to `PUT` with the Synthea UUID keeps every `source` field stable and auditable.

**Polling loop over filesystem watcher** — predictable, configurable without code changes, works across any environment. The 5-second interval feels near-immediate for a demo. A watcher fires faster but adds an OS dependency and can miss events under load.

**INI for model config** — `[DEFAULT]` inheritance means the shared prompt is written once; each model section overrides only what differs.

**Multi-tier JSON extraction** — `format: json` constrains Ollama's sampler to emit a bare JSON document, so there are no fences or preamble to strip. The remaining risk is unescaped control characters inside string values (literal newlines), which `json.loads` rejects. A character scanner fixes these in-string only — escaping structural newlines between tokens would corrupt valid pretty-printed JSON. A regex fallback on the raw text catches any remaining cases where the outer structure is still malformed.

---

## Issues Encountered

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Patient UUID became an integer after upload | HAPI assigns sequential IDs on `POST` | Rewrote bundle entries to `PUT {ResourceType}/{uuid}` |
| UUID lookup worked, integer ID didn't (and vice versa) | Only one lookup path existed | Primary lookup by resource ID, fallback by `?identifier=` |
| Ingestion loop stopped on one bad file | Unhandled exception escaped the per-file coroutine | Per-file `try/except` |
| Empty summaries from `qwen3:4b` | Ollama routes reasoning-model output to a `thinking` field, not `response`; code only read `response` | Read `thinking` as fallback when `response` is empty |
| `JSONDecodeError` on some models | Literal newlines inside JSON string values — `json.loads` rejects unescaped control characters | Character scanner escapes control chars inside strings only; regex fallback catches remaining malformed output |
| Medication status and prescriber missing from summaries | `parse_bundle()` discarded those fields | `fmt_medications()` renders all three fields into the prompt |
| 1,275 medications but model only saw 20 | HAPI's default page size is 20; no error is returned | `fetch_all_pages()` with `_count=1000` and `link[relation=next]` traversal |
| Model quality regressed after pagination fix | 1,265 historical entries doubled prompt size; model fixated on one mid-list drug | Group historical meds by drug name before building the prompt |

---

## Next Steps

To move this toward production:

- **Decouple ingestion from the API** — independent services that scale separately; replace polling with a message queue (e.g. Kafka).
- **Add PHI de-identification before inference** — scrub identifiers from the prompt before any text reaches the model, even on-premise.
- **Authenticate and authorize every request** — OAuth2/JWT on the endpoint with patient-level access control.
- **Persist model outputs to a database** — replace flat files with Postgres or DynamoDB for audit trails and retrieval without re-running inference.
- **Schema-constrained LLM decoding** — enforce JSON at the token level (Ollama schema mode or Outlines), eliminating the regex fallback extractor.
- **Golden evaluation dataset** — clinician-reviewed reference summaries with automated factual-grounding checks in CI.

---

## Attribution

### My own work

Architecture and component boundaries; UUID preservation design; configuration structure for `config.json` and `llm_param.ini`; prompt engineering; model selection and clinical evaluation.

Diagnosis of every issue was mine — I identified the symptom, traced the root cause, and defined what the fix had to do. AI helped move from diagnosis to working code faster, which is exactly the point of a 4–8 hour assignment.

### AI assistance

**Google Gemini** (~20–25 prompts) — used as a debugging partner after the foundation was in place. Key examples: I spotted the pagination gap by comparing the raw bundle count (1,275) to the API output (20); Gemini drafted `fetch_all_pages()`. I spotted the post-fix regression by comparing summaries before and after; Gemini helped implement the `fmt_historical_meds()` grouper. In both cases the diagnosis was mine; Gemini compressed the implementation time. Also handled per-model parameter tuning in `llm_param.ini`, and helped structure and format this README.
