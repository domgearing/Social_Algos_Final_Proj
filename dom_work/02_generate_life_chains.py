#!/usr/bin/env python3
"""
02_generate_life_chains.py

Phase 3: For each persona, generate a sequential life-event chain that
updates latent belief variables via the domain-relevance map.

Architecture per persona
------------------------
1. Load Level 1 demographics + initialized Beta priors (from Phase 2)
2. For each of N_EVENTS (default 7) life events:
   a. Call LLM (high temperature) with rolling context window
      → returns structured JSON: event_text, event_type, valence, simulated_age
   b. Validate hard constraints; resample up to MAX_CONSTRAINT_RETRIES times
   c. Apply stochastic latent-variable updates from domain_relevance_map
      Negativity bias: negative events draw from upper magnitude range
   d. Track full trajectory of deltas per variable
   e. After every COMPRESS_EVERY events: run cheap compression prompt
      Compressor is explicitly instructed to PRESERVE cross-cutting details
3. Run final persona distillation → ~150-token persona card
4. Save JSON chain file + update flat summary CSV

Outputs
-------
  life_chains/<respondent_id>.json   — full chain with events, trajectories, card
  life_chains_summary.csv           — flat table of final latent means per persona

Usage
-----
  python 02_generate_life_chains.py --sample 200 --seed 42
  python 02_generate_life_chains.py --sample 200 --model anthropic/claude-haiku-4-5
  python 02_generate_life_chains.py --priors gss2024_latent_priors.csv  # full dataset

Set OPENROUTER_API_KEY environment variable before running.
"""

import os
import re
import sys
import json
import time
import random
import argparse
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from domain_relevance_map import get_updates_for_event, list_event_types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL     = "anthropic/claude-haiku-4-5"
N_EVENTS_MIN      = 6
N_EVENTS_MAX      = 8
COMPRESS_EVERY    = 4       # compress after this many events
MAX_RETRIES       = 3       # API retries per call
MAX_CONSTRAINT_RETRIES = 3  # resample attempts on constraint violation
NEGATIVITY_BIAS   = 1.5     # magnitude multiplier for negative events
ROLLING_WINDOW    = 3       # how many recent events to keep in context

LATENT_VARS = [
    "moral_traditionalism",
    "economic_redistribution",
    "racial_attitudes",
    "institutional_trust",
    "outgroup_affect",
]

LATENT_VAR_LABELS = {
    "moral_traditionalism":    "Moral Traditionalism (0=progressive → 1=traditional)",
    "economic_redistribution": "Economic Redistribution (0=anti-redistrib → 1=pro-redistrib)",
    "racial_attitudes":        "Racial Attitudes (0=individualist → 1=structuralist)",
    "institutional_trust":     "Institutional Trust (0=low → 1=high)",
    "outgroup_affect":         "Outgroup Affect (0=closed → 1=open/welcoming)",
}

VALID_EVENT_TYPES = list_event_types() + ["unmodeled"]

# ---------------------------------------------------------------------------
# Hard-constraint rules (age-gated / geographic / temporal)
# ---------------------------------------------------------------------------

# Events that require the persona to be at least MIN_AGE at the time
AGE_GATED_EVENTS = {
    "child_or_close_family_member_comes_out_as_lgbtq":         30,
    "child_or_family_member_addiction_incarceration_or_crisis": 30,
    "significant_upward_mobility_or_wealth_accumulation":       28,
    "military_service":                                          18,
    "divorce_or_marital_dissolution":                           20,
    "active_civic_or_political_participation":                  18,
}

# Phrases that suggest the persona immigrated to the US (incoherent for US-born)
IMMIGRATION_PHRASES = [
    "immigrated to america",
    "immigrated to the united states",
    "moved from another country to america",
    "came to the united states from",
    "arrived in america from",
    "left my home country for america",
]

# Phrases that suggest retirement (incoherent if simulated_age < 52)
RETIREMENT_PHRASES = ["retired", "retirement"]
MIN_RETIREMENT_AGE = 52

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def call_openrouter(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 1.0,
    max_tokens: int = 400,
    timeout: int = 45,
) -> Optional[str]:
    """
    Call the OpenRouter chat completions endpoint.
    Returns the assistant message content, or None on failure.
    Retries with exponential backoff up to MAX_RETRIES times.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            content = data["choices"][0]["message"]["content"]
            return content.strip() if content else None

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    [API error after {MAX_RETRIES} attempts] {e}", file=sys.stderr)
                return None

    return None


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object from a string.
    Handles LLMs that wrap JSON in markdown code fences.
    """
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Latent variable updater
# ---------------------------------------------------------------------------

def apply_update(
    current_mean: float,
    direction: str,
    magnitude_range: tuple[float, float],
    valence: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Compute a stochastic update to a latent variable mean.

    Negativity bias: negative-valence events draw from the upper portion of
    the magnitude range and are scaled by NEGATIVITY_BIAS.

    Returns (new_mean, actual_delta).
    """
    lo, hi = magnitude_range

    if valence == "negative":
        # Draw from upper half of range, then apply bias multiplier
        raw = rng.uniform((lo + hi) / 2.0, hi) * NEGATIVITY_BIAS
        magnitude = min(raw, hi * 2.0)  # cap to prevent runaway updates
    elif valence == "positive":
        magnitude = rng.uniform(lo, (lo + hi) / 2.0)
    else:  # neutral
        magnitude = rng.uniform(lo, hi)

    if direction == "up":
        delta = magnitude
    elif direction == "down":
        delta = -magnitude
    elif direction == "mixed":
        # 55% up, 45% down (slight positive bias for mixed events like diversity exposure)
        delta = magnitude if rng.random() < 0.55 else -magnitude
    else:
        delta = 0.0

    new_mean = float(np.clip(current_mean + delta, 0.03, 0.97))
    actual_delta = new_mean - current_mean
    return new_mean, actual_delta


def update_latent_state(
    state: dict[str, float],
    trajectory: dict[str, list[float]],
    event_type: str,
    valence: str,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Apply all domain-relevance-map updates for event_type to state in-place.
    Appends deltas to trajectory.
    Returns list of applied update dicts (for saving in event record).
    """
    updates = get_updates_for_event(event_type)
    applied = []

    for rule in updates:
        var = rule["variable"]
        if var not in state:
            continue
        old = state[var]
        new, delta = apply_update(old, rule["direction"], rule["magnitude_range"], valence, rng)
        state[var] = new
        trajectory[var].append(round(delta, 5))
        applied.append({
            "variable":  var,
            "direction": rule["direction"],
            "delta":     round(delta, 5),
            "old_mean":  round(old, 4),
            "new_mean":  round(new, 4),
        })

    # For variables not touched by this event, record a zero delta so
    # trajectory length stays in sync with event count.
    for var in LATENT_VARS:
        if not any(a["variable"] == var for a in applied):
            trajectory[var].append(0.0)

    return applied


# ---------------------------------------------------------------------------
# Hard constraint validator
# ---------------------------------------------------------------------------

def check_constraints(event_dict: dict, persona_age: Optional[int]) -> tuple[bool, str]:
    """
    Returns (is_valid, reason_if_invalid).
    Checks geographic, temporal, and biological coherence.
    """
    text = (event_dict.get("event_text") or "").lower()
    event_type = event_dict.get("event_type", "unmodeled")
    simulated_age = event_dict.get("simulated_age")

    # Geographic: no impossible immigration for implicitly US-born personas
    # (All GSS respondents are US residents, and the personas don't indicate
    # foreign birth, so we treat them as US-born by default.)
    for phrase in IMMIGRATION_PHRASES:
        if phrase in text:
            return False, f"Geographic incoherence: '{phrase}' in event text"

    # Temporal: simulated_age must be ≤ persona's current age (they're living backwards otherwise)
    if simulated_age is not None and persona_age is not None:
        if simulated_age > persona_age:
            return False, (
                f"Temporal incoherence: event age {simulated_age} > persona age {persona_age}"
            )
        if simulated_age < 5:
            return False, f"Temporal incoherence: event age {simulated_age} < 5"

    # Biological: age-gated event types
    if event_type in AGE_GATED_EVENTS:
        min_age = AGE_GATED_EVENTS[event_type]
        if simulated_age is not None and simulated_age < min_age:
            return False, (
                f"Biological incoherence: {event_type} requires age ≥ {min_age}, "
                f"got {simulated_age}"
            )
        if persona_age is not None and persona_age < min_age:
            return False, (
                f"Biological incoherence: persona age {persona_age} too young for {event_type}"
            )

    # Temporal: retirement age check
    if any(phrase in text for phrase in RETIREMENT_PHRASES):
        if simulated_age is not None and simulated_age < MIN_RETIREMENT_AGE:
            return False, f"Temporal incoherence: retired at age {simulated_age}"
        if persona_age is not None and persona_age < MIN_RETIREMENT_AGE:
            return False, f"Temporal incoherence: persona age {persona_age} too young to retire"

    return True, ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _latent_state_summary(state: dict[str, float]) -> str:
    lines = []
    for var, mean in state.items():
        label = LATENT_VAR_LABELS.get(var, var)
        lines.append(f"  {label}: {mean:.2f}")
    return "\n".join(lines)


def build_event_prompt(
    demographics: str,
    latent_state: dict[str, float],
    recent_context: str,
    event_index: int,
    total_events: int,
    persona_age: Optional[int],
) -> list[dict]:
    """
    Build the messages list for a life-event generation call.
    High temperature (1.0) — this is where idiosyncratic variance is generated.
    """
    event_types_list = "\n".join(f"  - {et}" for et in VALID_EVENT_TYPES)

    age_guidance = ""
    if persona_age is not None:
        # Space events roughly across life — event_index / total_events × persona_age
        min_age_for_event = max(15, int((event_index / total_events) * persona_age * 0.6))
        max_age_for_event = persona_age
        age_guidance = (
            f"The event should have occurred when this person was roughly "
            f"between {min_age_for_event} and {max_age_for_event} years old. "
            f"Their current age is {persona_age}.\n"
        )

    system = (
        "You are simulating a realistic life history for a survey respondent. "
        "Generate ONE specific, plausible life event for this person. "
        "Write in first-person ('I...'). Be concrete and specific — avoid generic platitudes."
    )

    user = f"""PERSON'S DEMOGRAPHICS:
{demographics}

CURRENT BELIEF STATE (internal context — do not mention in event text):
{_latent_state_summary(latent_state)}

RECENT LIFE CONTEXT:
{recent_context if recent_context else "(No prior events yet — this is the first event)"}

TASK:
Generate life event {event_index + 1} of {total_events} for this person.
{age_guidance}
HARD RULES (violations will be rejected and resampled):
- Age-coherent: no adult children if too young, no retirement before 52, etc.
- Geographically coherent: do NOT have this US-resident person "immigrate to America"
- Temporally sequential: the event must fit logically after any prior events shown above
- Do NOT repeat an event type already well-covered above
- Write events as they would naturally occur in a real American's life

CLASSIFICATION:
Choose the single best event_type from this list, or "unmodeled" if none fit:
{event_types_list}

OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
{{
  "event_text": "<2-3 sentence first-person narrative>",
  "event_type": "<one type from the list above or 'unmodeled'>",
  "valence": "<positive|negative|neutral>",
  "simulated_age": <integer age when this event occurred>
}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_compression_prompt(recent_events: list[str]) -> list[dict]:
    """
    Prompt to compress 3-4 raw events into a 3-sentence summary.
    Explicitly instructed to preserve cross-cutting and contradictory details.
    """
    events_text = "\n\n".join(f"Event {i+1}: {e}" for i, e in enumerate(recent_events))

    system = (
        "You are compressing life events into a concise summary. "
        "Your job is to retain sociologically relevant signal, not literary detail."
    )

    user = f"""Compress the following life events into EXACTLY 3 sentences.

CRITICAL INSTRUCTIONS:
- Preserve cross-cutting details and contradictions — if this person has experiences
  that pull in opposite directions (e.g., conservative on economics but progressive
  on social issues due to a specific experience), keep BOTH sides explicitly.
- A naive summary will drop contradictions in favor of the dominant theme. DO NOT do this.
- Retain any high-salience rupture events (sudden tragedies, dramatic reversals).
- Use third person ("This person...").
- Do NOT add interpretation — just compress the facts.

EVENTS TO COMPRESS:
{events_text}

Output ONLY the 3-sentence summary, no preamble."""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_distillation_prompt(
    demographics: str,
    full_compressed_history: str,
    final_latent_state: dict[str, float],
) -> list[dict]:
    """
    Prompt to produce the ~150-token persona card used in Phase 4.
    This card is all that survives the chain into the survey phase.
    """
    system = (
        "You are producing a compact worldview summary for a survey respondent. "
        "This summary will be used to predict how they answer attitudinal survey questions."
    )

    user = f"""DEMOGRAPHICS:
{demographics}

LIFE HISTORY SUMMARY:
{full_compressed_history}

CURRENT BELIEF INDICATORS:
{_latent_state_summary(final_latent_state)}

TASK:
Write a 5-6 sentence persona card describing this person's core worldview.
Cover ALL of the following dimensions — each in approximately one sentence:
1. Religious orientation and relationship to organized religion
2. Economic outlook (redistribution, government's role, class consciousness)
3. Racial attitudes (structural vs. individual explanations for inequality)
4. Trust in institutions (government, press, science, legal system)
5. Feelings toward social outgroups (immigrants, racial minorities, LGBTQ, other religions)

CRITICAL: Preserve internal contradictions. If this person is traditional on some
dimensions but progressive on others, say so explicitly. Do not flatten them into
a consistent ideological type. A cross-cutting persona is NOT inconsistent — it is
realistic.

Output ONLY the 5-6 sentence persona card, no preamble or metadata."""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_event_narrative_prompt(
    demographics: str,
    events: list[dict],
) -> list[dict]:
    """
    Prompt for prompt_mode=3: produces an experience-based first-person summary
    that describes what happened to the person, NOT their resulting worldview.
    Used by Phase 4 Option 3 to avoid injecting explicit ideology into survey prompts.
    """
    system = (
        "You are writing a brief life history for a survey respondent. "
        "Describe only concrete experiences, events, and facts — "
        "never infer or state beliefs, attitudes, or ideological positions."
    )

    events_text = "\n\n".join(
        f"[Age {e['simulated_age']}] {e['event_text']}" for e in events
    )

    user = f"""DEMOGRAPHICS:
{demographics}

LIFE EVENTS:
{events_text}

TASK:
Write a 5-6 sentence first-person summary of the key experiences from this person's life.

STRICT RULES:
- Write in first person ("I...").
- Describe only what HAPPENED — not what the person believes, values, or feels as a result.
- Never use evaluative or ideological terms: "worldview", "believes", "trusts", "distrusts",
  "values", "conservative", "liberal", "progressive", "traditional", "attitudes", "outlook".
- Translate implied beliefs back into the events that produced them.
  WRONG: "I distrust government institutions."
  RIGHT: "I was laid off when the plant closed after trade legislation passed, and my appeal for
          unemployment benefits was rejected on a technicality."
- If two events pulled in opposite directions, include both — do not resolve the tension.
- Be concrete: include ages, places, institutions, and outcomes.

Output ONLY the 5-6 sentence summary, no preamble or metadata."""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ---------------------------------------------------------------------------
# Life chain runner (one persona)
# ---------------------------------------------------------------------------

def run_life_chain(
    respondent_id: int,
    demographics: str,
    initial_state: dict[str, float],
    persona_age: Optional[int],
    model: str,
    api_key: str,
    rng: np.random.Generator,
    n_events: int,
) -> dict:
    """
    Run the full life-event chain for one persona.
    Returns the complete chain record (to be saved as JSON).
    """
    # Deep-copy so we don't mutate the input
    latent_state = deepcopy(initial_state)
    trajectory: dict[str, list[float]] = {var: [] for var in LATENT_VARS}

    events: list[dict] = []
    compressed_summaries: list[dict] = []

    # Rolling context: we keep the last ROLLING_WINDOW raw event texts,
    # or the latest compressed summary if one exists.
    rolling_raw: list[str]  = []   # last few raw event texts
    latest_compression: Optional[str] = None

    def _rolling_context() -> str:
        if latest_compression:
            header = "COMPRESSED HISTORY:\n" + latest_compression
        else:
            header = ""
        if rolling_raw:
            recent = "\n\n".join(
                f"Recent event {i+1}: {t}" for i, t in enumerate(rolling_raw)
            )
            return (header + "\n\nRECENT EVENTS:\n" + recent).strip()
        return header.strip()

    # ---- Generate life events ----
    for step in range(n_events):
        context = _rolling_context()
        prompt_messages = build_event_prompt(
            demographics   = demographics,
            latent_state   = latent_state,
            recent_context = context,
            event_index    = step,
            total_events   = n_events,
            persona_age    = persona_age,
        )

        event_dict = None
        constraint_reason = ""

        for attempt in range(MAX_CONSTRAINT_RETRIES):
            raw = call_openrouter(
                messages    = prompt_messages,
                model       = model,
                api_key     = api_key,
                temperature = 1.0,
                max_tokens  = 350,
            )
            parsed = extract_json(raw) if raw else None

            if not parsed:
                continue  # retry on parse failure

            # Normalise event_type
            et = (parsed.get("event_type") or "unmodeled").strip().lower().replace(" ", "_")
            if et not in VALID_EVENT_TYPES:
                et = "unmodeled"
            parsed["event_type"] = et

            # Normalise valence
            val = (parsed.get("valence") or "neutral").strip().lower()
            if val not in ("positive", "negative", "neutral"):
                val = "neutral"
            parsed["valence"] = val

            # Hard constraint check
            ok, reason = check_constraints(parsed, persona_age)
            if ok:
                event_dict = parsed
                break
            else:
                constraint_reason = reason

        if event_dict is None:
            # All retries exhausted — insert a placeholder unmodeled event
            event_dict = {
                "event_text":    f"[Step {step+1} skipped: constraint could not be satisfied — {constraint_reason}]",
                "event_type":    "unmodeled",
                "valence":       "neutral",
                "simulated_age": persona_age or 30,
            }

        # Apply latent variable updates
        applied_updates = update_latent_state(
            state      = latent_state,
            trajectory = trajectory,
            event_type = event_dict["event_type"],
            valence    = event_dict["valence"],
            rng        = rng,
        )

        event_record = {
            "step":               step + 1,
            "simulated_age":      event_dict.get("simulated_age"),
            "event_type":         event_dict["event_type"],
            "valence":            event_dict["valence"],
            "event_text":         event_dict.get("event_text", ""),
            "updates":            applied_updates,
            "latent_state_after": {v: round(latent_state[v], 4) for v in LATENT_VARS},
        }
        events.append(event_record)

        # Update rolling window
        rolling_raw.append(event_dict.get("event_text", ""))
        if len(rolling_raw) > ROLLING_WINDOW:
            rolling_raw.pop(0)

        # Compression every COMPRESS_EVERY events
        if (step + 1) % COMPRESS_EVERY == 0 and step + 1 < n_events:
            # Compress the last COMPRESS_EVERY raw event texts
            texts_to_compress = [e["event_text"] for e in events[-COMPRESS_EVERY:]]
            comp_messages = build_compression_prompt(texts_to_compress)
            summary = call_openrouter(
                messages    = comp_messages,
                model       = model,
                api_key     = api_key,
                temperature = 0.3,
                max_tokens  = 200,
            )
            if summary:
                compressed_summaries.append({
                    "after_event": step + 1,
                    "summary":     summary,
                })
                latest_compression = summary
                rolling_raw = []  # reset rolling window — compression replaces raw events

    # ---- Final persona distillation ----
    # Build a consolidated compressed history for the distillation prompt
    if compressed_summaries:
        recent_events = [e["event_text"] for e in events[-(COMPRESS_EVERY):]]
        if recent_events and recent_events[0] != compressed_summaries[-1].get("summary"):
            # Compress trailing events that weren't covered
            comp_messages = build_compression_prompt(recent_events)
            tail_summary = call_openrouter(
                messages    = comp_messages,
                model       = model,
                api_key     = api_key,
                temperature = 0.3,
                max_tokens  = 200,
            )
            if tail_summary:
                compressed_summaries.append({"after_event": n_events, "summary": tail_summary})

        full_history = "\n\n".join(s["summary"] for s in compressed_summaries)
    else:
        # Fewer events than COMPRESS_EVERY — just use raw texts
        full_history = "\n\n".join(
            f"Event {e['step']}: {e['event_text']}" for e in events
        )

    distill_messages = build_distillation_prompt(
        demographics        = demographics,
        full_compressed_history = full_history,
        final_latent_state  = latent_state,
    )
    persona_card = call_openrouter(
        messages    = distill_messages,
        model       = model,
        api_key     = api_key,
        temperature = 0.4,
        max_tokens  = 250,
    )
    if not persona_card:
        persona_card = f"[Distillation failed for respondent {respondent_id}]"

    # ---- Event chain (raw first-person events, for prompt_mode=2) ----
    event_chain = "\n\n".join(
        f"[Age {e['simulated_age']}] {e['event_text']}" for e in events
    )

    # ---- Compressed chain (joined periodic summaries, for prompt_mode=1) ----
    compressed_chain = (
        "\n\n".join(s["summary"] for s in compressed_summaries)
        if compressed_summaries else full_history
    )

    # ---- Event narrative (experience-based distillation, for prompt_mode=3) ----
    narrative_messages = build_event_narrative_prompt(
        demographics = demographics,
        events       = events,
    )
    event_narrative = call_openrouter(
        messages    = narrative_messages,
        model       = model,
        api_key     = api_key,
        temperature = 0.4,
        max_tokens  = 250,
    )
    if not event_narrative:
        event_narrative = f"[Event narrative failed for respondent {respondent_id}]"

    # ---- Assemble output record ----
    chain_record = {
        "respondent_id":    respondent_id,
        "model":            model,
        "demographics":     demographics,
        "n_events":         n_events,
        "events":           events,
        "compressed_summaries": compressed_summaries,
        "persona_card":     persona_card,
        "event_chain":      event_chain,
        "compressed_chain": compressed_chain,
        "event_narrative":  event_narrative,
        "initial_latent_state":  {v: round(initial_state[v], 4) for v in LATENT_VARS},
        "final_latent_state":    {v: round(latent_state[v], 4)  for v in LATENT_VARS},
        "latent_trajectories":   {v: trajectory[v]              for v in LATENT_VARS},
    }
    return chain_record


# ---------------------------------------------------------------------------
# Summary CSV helpers
# ---------------------------------------------------------------------------

def chain_to_summary_row(chain: dict) -> dict:
    """Flatten a chain record into a single-row dict for the summary CSV."""
    row = {"respondent_id": chain["respondent_id"], "model": chain["model"]}
    for var in LATENT_VARS:
        init  = chain["initial_latent_state"].get(var, float("nan"))
        final = chain["final_latent_state"].get(var, float("nan"))
        traj  = chain["latent_trajectories"].get(var, [])
        row[f"{var}_initial"]     = round(init, 4)
        row[f"{var}_final"]       = round(final, 4)
        row[f"{var}_total_shift"] = round(final - init, 4)
        row[f"{var}_n_updates"]   = sum(1 for d in traj if d != 0.0)
        row[f"{var}_max_single_shift"] = round(max((abs(d) for d in traj), default=0.0), 4)
    row["persona_card"]     = chain["persona_card"]
    row["event_chain"]      = chain.get("event_chain", "")
    row["compressed_chain"] = chain.get("compressed_chain", "")
    row["event_narrative"]  = chain.get("event_narrative", "")
    return row


def load_summary(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def append_to_summary(row: dict, path: Path) -> None:
    existing = load_summary(path)
    new_row = pd.DataFrame([row])
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate life-event chains and update latent belief variables."
    )
    parser.add_argument(
        "--priors",
        default="gss2024_latent_priors.csv",
        help="Path to priors CSV from Phase 2 (default: gss2024_latent_priors.csv)"
    )
    parser.add_argument(
        "--personas",
        default="gss2024_personas.csv",
        help="Path to personas CSV (default: gss2024_personas.csv)"
    )
    parser.add_argument(
        "--output-dir",
        default="life_chains",
        help="Directory to save per-persona JSON chains (default: life_chains/)"
    )
    parser.add_argument(
        "--summary",
        default="life_chains_summary.csv",
        help="Path to flat summary CSV (default: life_chains_summary.csv)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only this many personas (takes first N from priors file; use --seed for ordering)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for event stochasticity and sampling (default: 42)"
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=None,
        help=f"Fixed number of events per persona (default: random {N_EVENTS_MIN}-{N_EVENTS_MAX})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip personas that already have a completed chain file (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Overwrite existing chain files"
    )
    args = parser.parse_args()

    # API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable.")

    # Paths
    priors_path   = Path(args.priors)
    personas_path = Path(args.personas)
    output_dir    = Path(args.output_dir)
    summary_path  = Path(args.summary)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading priors from {priors_path} ...")
    priors_df = pd.read_csv(priors_path)

    print(f"Loading personas from {personas_path} ...")
    personas_df = pd.read_csv(personas_path).set_index("respondent_id")

    if args.sample is not None:
        priors_df = priors_df.head(args.sample)
    print(f"Processing {len(priors_df)} personas with model {args.model}\n")

    # Global RNG (per-persona RNGs derived from this + respondent_id for reproducibility)
    global_rng = np.random.default_rng(args.seed)

    # ---- Main loop ----
    skipped = 0
    failed  = 0

    for _, row in tqdm(priors_df.iterrows(), total=len(priors_df), unit="persona"):
        rid = int(row["respondent_id"])
        chain_path = output_dir / f"{rid}.json"

        # Resume: skip if already done
        if args.resume and chain_path.exists():
            skipped += 1
            continue

        # Persona-specific RNG — deterministic given seed + respondent_id
        persona_rng = np.random.default_rng(args.seed + rid)

        # Build initial latent state from Beta prior means
        initial_state = {
            var: float(row[f"{var}_mean"]) for var in LATENT_VARS
        }

        # Persona demographics text
        if rid in personas_df.index:
            demographics = str(personas_df.loc[rid, "persona"])
        else:
            demographics = f"Respondent {rid} — demographics unavailable"

        # Parse age from demographics row
        persona_age: Optional[int] = None
        age_val = row.get("demo_age")
        if age_val is not None and not pd.isna(age_val):
            persona_age = int(age_val)

        # Number of events for this persona
        if args.n_events is not None:
            n_events = args.n_events
        else:
            n_events = int(persona_rng.integers(N_EVENTS_MIN, N_EVENTS_MAX + 1))

        try:
            chain = run_life_chain(
                respondent_id = rid,
                demographics  = demographics,
                initial_state = initial_state,
                persona_age   = persona_age,
                model         = args.model,
                api_key       = api_key,
                rng           = persona_rng,
                n_events      = n_events,
            )

            # Save JSON chain
            with open(chain_path, "w") as f:
                json.dump(chain, f, indent=2)

            # Append to summary CSV
            summary_row = chain_to_summary_row(chain)
            append_to_summary(summary_row, summary_path)

        except Exception as e:
            failed += 1
            print(f"\n  [ERROR] Persona {rid}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue

    total = len(priors_df)
    done  = total - skipped - failed
    print(f"\nDone. Completed: {done}  |  Skipped (resume): {skipped}  |  Failed: {failed}")
    print(f"Chain files: {output_dir}/")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
