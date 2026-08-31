import json
import configparser
import time
from collections import defaultdict
from string import Template
import httpx
import re

from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent

# Ollama runs locally so PHI never leaves the machine.
OLLAMA_URL = "http://localhost:11434/api/generate"
CONFIG_FILE = str(_PROJECT_ROOT / "config_dir" / "llm_param.ini")


def extract_json_payload(raw_response: str) -> str:
    """Strips thinking tags, markdown formatting, and prelude text to isolate valid JSON."""
    if not raw_response:
        return ""

    cleaned = raw_response.strip()

    # \1 keeps only the content inside the fence, discarding the backticks and optional "json" label.
    cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", cleaned, flags=re.DOTALL).strip()

    # Greedy .* finds the outermost braces, discarding any preamble text the model wrote before the JSON.
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0).strip()

    # Negative lookbehind (?<!\\) ensures only truly raw newlines/tabs are escaped — json.loads rejects them as-is.
    cleaned = re.sub(r'(?<!\\)\n', r'\\n', cleaned)
    cleaned = re.sub(r'(?<!\\)\t', r'\\t', cleaned)

    return cleaned


def parse_summary_from_response(raw_text: str) -> str:
    """Multi-tiered parser ensuring all models return a string summary without failing."""
    sanitized_str = extract_json_payload(raw_text)

    # Tier 1: clean JSON parse, the happy path for well-behaved models.
    try:
        data = json.loads(sanitized_str)
        if isinstance(data, dict) and "summary" in data:
            return str(data["summary"]).strip()
    except json.JSONDecodeError:
        pass

    # Tier 2: regex on the raw text in case brace isolation failed but the key is still findable.
    match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_text, re.DOTALL)
    if match:
        return match.group(1).replace('\\n', ' ').strip()

    # Tier 3: last resort — return cleaned plain text so a completed inference is never silently discarded.
    clean_fallback = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw_text, flags=re.DOTALL).strip()
    if clean_fallback:
        return clean_fallback

    return "Summary unavailable."


def load_model_config(section_name: str) -> tuple[dict, str]:
    """Loads configuration options and prompt template for a specific model section."""
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    if section_name not in config:
        raise ValueError(f"Section [{section_name}] not found in {CONFIG_FILE}")

    # Accessing a section via config[name] automatically merges [DEFAULT] values with section-specific overrides.
    section = config[section_name]

    options = {
        "temperature": float(section.get("temperature", 0.1)),
        "top_p": float(section.get("top_p", 0.9)),
        "top_k": int(section.get("top_k", 20)),
        "num_predict": int(section.get("num_predict", 256)),
        "presence_penalty": float(section.get("presence_penalty", 0.0)),
        "frequency_penalty": float(section.get("frequency_penalty", 0.0))
    }

    prompt_template = section.get("prompt") or ""
    model_name = section.get("model", section_name)

    return {"model": model_name, "options": options}, prompt_template


async def generate_summary(conditions: list, active_meds: list, historical_meds: list, allergies: list, model_section: str = "llama3.2:3b") -> tuple[str, dict]:
    # Inner helpers format the clinical data into plain strings the prompt template can embed.
    def fmt_conditions_or_allergies(items):
        return ", ".join(i["display"] for i in items) if items else "none"

    def fmt_active_meds(items):
        if not items:
            return "none"
        parts = []
        for m in items:
            details = m["display"]
            meta = []
            if m.get("authoredOn"):
                meta.append(f"since {m['authoredOn'][:10]}")
            if m.get("practioner"):
                meta.append(f"by {m['practioner']}")
            if meta:
                details += f" ({', '.join(meta)})"
            parts.append(details)
        return "; ".join(parts)

    def fmt_historical_meds(items):
        # Group by drug name and collapse repeats into a count + date range to keep the prompt compact.
        if not items:
            return "none"
        groups: dict[str, list[str]] = defaultdict(list)
        for m in items:
            raw_date = m.get("authoredOn", "")
            groups[m["display"]].append(raw_date[:10] if raw_date else "")
        parts = []
        for display, dates in groups.items():
            dates = sorted(d for d in dates if d)
            count = len(groups[display])
            if count == 1:
                parts.append(f"{display} (×1{', ' + dates[0] if dates else ''})")
            else:
                date_range = f", {dates[0]} to {dates[-1]}" if dates else ""
                parts.append(f"{display} (×{count}{date_range})")
        return "; ".join(parts)

    def fmt_medications(active, historical):
        return (
            f"Active medications: {fmt_active_meds(active)} | "
            f"Historical medications (stopped/completed, grouped by drug): {fmt_historical_meds(historical)}"
        )

    model_params, prompt_template = load_model_config(model_section)

    # INI uses {variable} placeholders; string.Template uses $variable — convert before substituting.
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

    # "format": "json" biases Ollama's token sampling toward valid JSON; "stream": False waits for the full response before returning.
    payload = {
        "model": model_params["model"],
        "prompt": formatted_prompt,
        "system": "You are a JSON generator. You must output raw JSON only matching the exact schema requested. Do not output thinking tags or commentary.",
        "format": "json",
        "stream": False,
        "options": model_params["options"]
    }

    # perf_counter gives sub-millisecond resolution; initialized before the try so elapsed is always measurable.
    start_time = time.perf_counter()
    raw_text = ""

    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()

        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        res_data = resp.json()

        # Ollama puts the model's full text output in the "response" key.
        raw_text = res_data.get("response", "")

        # Run the multi-tier extractor to get a clean summary string regardless of model output format.
        summary_text = parse_summary_from_response(raw_text)

        # Ollama returns all durations in nanoseconds; divide by 1e6 to convert to milliseconds.
        prompt_tokens = res_data.get("prompt_eval_count", 0)
        eval_tokens = res_data.get("eval_count", 0)
        total_tokens = prompt_tokens + eval_tokens

        total_duration_ms = round(res_data.get("total_duration", 0) / 1e6, 2)
        eval_duration_ms = round(res_data.get("eval_duration", 0) / 1e6, 2)

        # tokens_per_second = completion tokens / eval duration in seconds; guard against division by zero.
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
        # Return a fallback metrics dict rather than crashing so the caller always gets some output.
        elapsed_seconds = round(time.perf_counter() - start_time, 3)
        print(f"\n[DEBUG Error for {model_section}]: {exc}")

        if raw_text:
            print(f"[DEBUG Raw Response]: {raw_text}\n")

        metrics = {
            "model_section": model_section,
            "ollama_model": model_params["model"],
            "wall_clock_time_seconds": elapsed_seconds,
            "error": str(exc)
        }
        return f"Summary unavailable ({type(exc).__name__}).", metrics
