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


# Control characters JSON forbids unescaped inside a string literal.
_STRING_ESCAPES = {"\n": "\\n", "\t": "\\t", "\r": "\\r"}


def extract_json_payload(raw_response: str) -> str:
    """Escapes raw control characters inside JSON string values so json.loads accepts them."""
    if not raw_response:
        return ""

    # "format": "json" constrains Ollama's sampler to emit a bare JSON document, so there are no
    # markdown fences or prose preamble to strip. The one thing the grammar does not guarantee is
    # that a literal newline inside a string value gets escaped, which json.loads rejects.
    # A character scanner (not a regex) because only newlines *inside* a string need escaping —
    # escaping the structural ones between tokens (e.g. pretty-printed JSON) makes it invalid.
    out = []
    in_string = False   # True while the cursor is inside a JSON string literal
    escaped = False     # True for one character after a backslash, to skip \" without closing the string

    for ch in raw_response.strip():
        if in_string:
            if escaped:
                # Previous char was a backslash — this char is part of an escape sequence, pass it through.
                escaped = False
            elif ch == "\\":
                # Start of an escape sequence; the next character should not be interpreted.
                escaped = True
            elif ch == '"':
                # Closing quote — exit string mode.
                in_string = False
            elif ch in _STRING_ESCAPES:
                # Raw control character inside a string — replace with its JSON escape sequence.
                out.append(_STRING_ESCAPES[ch])
                continue
        elif ch == '"':
            # Opening quote — enter string mode.
            in_string = True
        out.append(ch)

    return "".join(out)


def parse_summary_from_response(raw_text: str) -> str:
    """Multi-tiered parser ensuring all models return a string summary without failing."""
    sanitized_str = extract_json_payload(raw_text)

    # Tier 1: standard JSON parse after sanitization — the expected path for all models.
    try:
        data = json.loads(sanitized_str)
        if isinstance(data, dict) and "summary" in data:
            return str(data["summary"]).strip()
    except json.JSONDecodeError:
        pass

    # Tier 2: key-value regex on the raw text. Catches cases where the outer JSON is malformed
    # but the "summary" key and its value are still findable (e.g. trailing comma, extra whitespace).
    match = re.search(r'"summary"\s*:\s*"(.*?)"', raw_text, re.DOTALL)
    if match:
        return match.group(1).replace('\\n', ' ').strip()

    # Tier 3: last resort — strip any markdown fences and return raw text so a completed
    # inference is never silently discarded as "Summary unavailable."
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

        # Ollama puts the model's full text output in the "response" key. Reasoning models such as
        # qwen3 route theirs to "thinking" instead and leave "response" empty, so fall back to it
        # rather than discarding a completed inference as a failure.
        raw_text = res_data.get("response") or res_data.get("thinking") or ""

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
