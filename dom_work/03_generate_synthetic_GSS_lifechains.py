#!/usr/bin/env python3
"""
03_generate_synthetic_GSS_lifechains.py

Phase 4: Query LLMs with the same 52 GSS attitudinal items as the baseline
(01_generate_synthetic_GSS.py), but using life-chain persona cards instead
of static demographic strings.

Prompt structure per survey item
---------------------------------
  BACKGROUND:          (Level 1 demographics, ~50 tokens)
  LIFE HISTORY:        (persona card from Phase 3, ~150 tokens)
  Question + options   (~30 tokens)
  → answer: single digit

Temperature is set to 1.0, matching the baseline replication script.

Everything else mirrors the baseline script:
  - Same 52 GSS questions
  - Same resume logic (skips completed persona/question/run triples)
  - Same parallel execution with ThreadPoolExecutor
  - Same batch saving
  - Identical output schema (plus chain_model column for provenance)

Outputs
-------
  synthetic_data_lifechains/year_<year>/<model_filename>.csv

Usage
-----
  python 03_generate_synthetic_GSS_lifechains.py --year 2024 --models openai/gpt-4o-mini --personas 200
  python 03_generate_synthetic_GSS_lifechains.py --year 2024 --all-models --personas 200
  python 03_generate_synthetic_GSS_lifechains.py --year 2024 --models mistralai/mistral-nemo \\
      --summary life_chains_summary.csv --chains-dir life_chains

Set OPENROUTER_API_KEY before running.
"""

import os
import sys
import json
import time
import random
import importlib.util
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import question bank + model list from the baseline script to avoid
# duplicating ~300 lines of static survey data.
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).parent
BASELINE_PATH = SCRIPT_DIR / "01_generate_synthetic_GSS.py"

_spec = importlib.util.spec_from_file_location("baseline_gss", BASELINE_PATH)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

GSS_QUESTIONS_COMPREHENSIVE = _baseline.GSS_QUESTIONS_COMPREHENSIVE
GSS_QUESTIONS_CULTUREWAR    = _baseline.GSS_QUESTIONS_CULTUREWAR
GSS_QUESTIONS_NONCULTUREWAR = _baseline.GSS_QUESTIONS_NONCULTUREWAR
POPULAR_MODELS              = _baseline.POPULAR_MODELS
OPENROUTER_API_URL          = _baseline.OPENROUTER_API_URL

del _spec, _baseline

# ---------------------------------------------------------------------------
# Latent-state formatter (used by prompt mode 1)
# ---------------------------------------------------------------------------

_LATENT_LABELS = {
    "moral_traditionalism":    "Moral traditionalism    (0=progressive → 1=traditional)",
    "economic_redistribution": "Economic redistribution (0=anti-redistribution → 1=pro)",
    "racial_attitudes":        "Racial attitudes        (0=individualist → 1=structuralist)",
    "institutional_trust":     "Institutional trust     (0=low → 1=high)",
    "outgroup_affect":         "Outgroup affect         (0=closed → 1=open/welcoming)",
}


def format_latent_state(card_row: "pd.Series") -> str:
    """Format final latent-state values from a summary CSV row into a readable string."""
    lines = []
    for var, label in _LATENT_LABELS.items():
        col = f"{var}_final"
        val = card_row.get(col)
        if val is not None and pd.notna(val):
            lines.append(f"- {label}: {float(val):.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Survey query — same interface as baseline, temperature matches baseline (1.0)
# ---------------------------------------------------------------------------

def query_openrouter(
    model: str,
    demographics: str,
    persona_card: str,
    question: str,
    options: Dict[int, str],
    api_key: str,
    year: int,
    prompt_mode: int = 1,
    event_chain: str = "",
    compressed_chain: str = "",
    event_narrative: str = "",
    latent_state: str = "",
    timeout: int = 30,
    max_retries: int = 3,
) -> Dict:
    """
    Query OpenRouter for a single GSS item using the life-chain prompt.

    prompt_mode controls what context is sent:
      1 — persona card (worldview distillation) + compressed chain + latent state values
      2 — raw first-person life events + cross-cutting contradiction instruction
      3 — experience-based narrative (events described, no worldview labels)
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    if prompt_mode == 2:
        prompt = (
            f"It is now {year}. You are a person living in the United States. "
            f"The following is your background and a record of key life events. "
            f"This person has genuine tensions and contradictions in their views "
            f"and does not always answer consistently with any single ideology.\n\n"
            f"MY BACKGROUND:\n{demographics}\n\n"
            f"MY LIFE EVENTS:\n{event_chain}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Respond with ONLY the number of your answer (e.g., \"1\" or \"2\"). "
            f"Do not explain your reasoning."
        )
    elif prompt_mode == 3:
        prompt = (
            f"It is now {year}. You are a person living in the United States. "
            f"The following is your personal background and a summary of your life experiences. "
            f"Answer the survey question as yourself, in the first person.\n\n"
            f"MY BACKGROUND:\n{demographics}\n\n"
            f"MY LIFE EXPERIENCES:\n{event_narrative}\n\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Respond with ONLY the number of your answer (e.g., \"1\" or \"2\"). "
            f"Do not explain your reasoning."
        )
    else:  # prompt_mode == 1 (default)
        sections = (
            f"MY BACKGROUND:\n{demographics}\n\n"
            f"MY LIFE HISTORY AND WORLDVIEW:\n{persona_card}\n\n"
        )
        if compressed_chain:
            sections += f"MY LIFE CHAIN SUMMARY:\n{compressed_chain}\n\n"
        if latent_state:
            sections += f"MY CURRENT BELIEF PROFILE:\n{latent_state}\n\n"
        prompt = (
            f"It is now {year}. You are a person living in the United States. "
            f"The following is your personal background and life history. "
            f"Answer the survey question as yourself, in the first person.\n\n"
            f"{sections}"
            f"Question: {question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Respond with ONLY the number of your answer (e.g., \"1\" or \"2\"). "
            f"Do not explain your reasoning."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_tokens": 50,
    }

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL, headers=headers, json=data, timeout=timeout
            )
            response.raise_for_status()

            result = response.json()
            answer_text = result["choices"][0]["message"]["content"].strip()

            try:
                answer = int(answer_text)
                if answer not in options:
                    return {
                        "answer": None,
                        "error": f"Out-of-range answer: {answer_text}",
                        "prompt_tokens":      result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens":  result.get("usage", {}).get("completion_tokens", 0),
                        "raw_response":       answer_text,
                    }
            except ValueError:
                return {
                    "answer": None,
                    "error": f"Non-numeric answer: {answer_text}",
                    "prompt_tokens":      result.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens":  result.get("usage", {}).get("completion_tokens", 0),
                    "raw_response":       answer_text,
                }

            return {
                "answer":            answer,
                "error":             None,
                "prompt_tokens":     result.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "raw_response":      answer_text,
            }

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except Exception as e:
            last_error = f"Unexpected error: {e}"
            break

    return {
        "answer": None, "error": last_error,
        "prompt_tokens": 0, "completion_tokens": 0, "raw_response": "",
    }


# ---------------------------------------------------------------------------
# Resume helper — identical logic to baseline
# ---------------------------------------------------------------------------

def load_completed_tasks(output_file: Path) -> Set[Tuple[int, str, int]]:
    if not output_file.exists():
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(
            df[df["answer"].notna()][["persona_id", "variable", "run"]]
            .apply(tuple, axis=1)
        )
    except Exception as e:
        print(f"Warning: could not load completed tasks: {e}")
        return set()


# ---------------------------------------------------------------------------
# Persona card loader
# ---------------------------------------------------------------------------

def load_persona_cards(
    summary_path: Path,
    chains_dir: Optional[Path],
    chain_model_filter: Optional[str],
) -> pd.DataFrame:
    """
    Load persona cards produced by Phase 3.

    Priority:
    1. life_chains_summary.csv  — flat CSV with respondent_id + persona_card
    2. Individual life_chains/<id>.json files — if summary is unavailable

    Returns a DataFrame indexed by respondent_id with columns:
        persona_card, chain_model
        (plus event_chain, compressed_chain, event_narrative, latent finals if present)
    """
    # Columns added by the multi-mode pipeline in 02_generate_life_chains.py
    _OPTIONAL_COLS = [
        "event_chain", "compressed_chain", "event_narrative",
        # latent final values for mode-1 belief profile
        "moral_traditionalism_final", "economic_redistribution_final",
        "racial_attitudes_final", "institutional_trust_final", "outgroup_affect_final",
    ]

    # Try summary CSV first
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if "persona_card" not in df.columns:
            raise ValueError(
                f"{summary_path} is missing a 'persona_card' column. "
                "Re-run 02_generate_life_chains.py to regenerate."
            )
        if chain_model_filter and "model" in df.columns:
            df = df[df["model"] == chain_model_filter]
            if df.empty:
                raise ValueError(
                    f"No chains with model='{chain_model_filter}' found in {summary_path}. "
                    f"Available: {pd.read_csv(summary_path)['model'].unique().tolist()}"
                )
        if "model" not in df.columns:
            df["chain_model"] = "unknown"
        else:
            df = df.rename(columns={"model": "chain_model"})
        keep = ["respondent_id", "persona_card", "chain_model"] + [
            c for c in _OPTIONAL_COLS if c in df.columns
        ]
        return df[keep].set_index("respondent_id")

    # Fallback: scan JSON files
    if chains_dir and chains_dir.exists():
        records = []
        for json_path in sorted(chains_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    chain = json.load(f)
                chain_model = chain.get("model", "unknown")
                if chain_model_filter and chain_model != chain_model_filter:
                    continue
                events = chain.get("events", [])
                compressed = chain.get("compressed_summaries", [])
                records.append({
                    "respondent_id":  chain["respondent_id"],
                    "persona_card":   chain.get("persona_card", ""),
                    "chain_model":    chain_model,
                    "event_chain":    "\n\n".join(
                        f"[Age {e['simulated_age']}] {e['event_text']}"
                        for e in events
                    ),
                    "compressed_chain": "\n\n".join(
                        s["summary"] for s in compressed
                    ),
                    "event_narrative":  chain.get("event_narrative", ""),
                    **{f"{v}_final": chain.get("final_latent_state", {}).get(v)
                       for v in _LATENT_LABELS},
                })
            except Exception:
                continue
        if not records:
            raise FileNotFoundError(
                f"No usable chain files found in {chains_dir}. "
                "Run 02_generate_life_chains.py first."
            )
        df = pd.DataFrame(records).set_index("respondent_id")
        return df

    raise FileNotFoundError(
        f"Neither {summary_path} nor {chains_dir} contains life-chain data. "
        "Run 02_generate_life_chains.py first."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: Query LLMs with GSS items using life-chain persona cards "
            "instead of static demographic strings."
        )
    )
    parser.add_argument(
        "--year", type=int, required=True,
        choices=[2024, 2016, 2008, 2000],
        help="Survey year for temporal framing in prompts"
    )
    parser.add_argument(
        "--models", type=str,
        help="Comma-separated list of survey-answering model names"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Use all models from the POPULAR_MODELS list"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Runs per persona-question pair (default: 1)"
    )
    parser.add_argument(
        "--personas", type=int, default=None,
        help="Limit to this many personas from the chain summary (default: all available)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Parallel API workers (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Results to buffer before writing to disk (default: 100)"
    )
    parser.add_argument(
        "--summary", default="life_chains_summary.csv",
        help="Path to life_chains_summary.csv from Phase 3 (default: life_chains_summary.csv)"
    )
    parser.add_argument(
        "--chains-dir", default="life_chains",
        help="Fallback: directory of per-persona JSON chain files (default: life_chains/)"
    )
    parser.add_argument(
        "--chain-model", default=None,
        help="If set, only use chains generated by this specific model"
    )
    parser.add_argument(
        "--personas-csv", default="gss2024_personas.csv",
        help="Path to base demographics CSV for Level 1 context (default: gss2024_personas.csv)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: synthetic_data_lifechains/year_<year>/mode<N>/)"
    )
    parser.add_argument(
        "--prompt-mode", type=int, default=1, choices=[1, 2, 3],
        help=(
            "Survey prompt context mode: "
            "1=persona card + compressed chain + latent state values (default); "
            "2=raw life events + cross-cutting contradiction instruction; "
            "3=experience-based narrative (no worldview labels). "
            "Modes 2 and 3 require chains regenerated with the updated 02 script."
        )
    )

    args = parser.parse_args()
    year = args.year

    # API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your key from https://openrouter.ai/keys"
        )

    # Models
    if args.all_models:
        models = POPULAR_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        raise ValueError("Specify --models <name> or --all-models")

    # Output directory — modes get separate subdirectories to avoid mixing results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            SCRIPT_DIR / "synthetic_data_lifechains"
            / f"year_{year}" / f"mode{args.prompt_mode}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load life-chain persona cards
    print("Loading life-chain persona cards ...")
    cards_df = load_persona_cards(
        summary_path      = SCRIPT_DIR / args.summary,
        chains_dir        = SCRIPT_DIR / args.chains_dir,
        chain_model_filter= args.chain_model,
    )
    print(f"  {len(cards_df)} personas with completed chains")
    if "chain_model" in cards_df.columns:
        for cm, n in cards_df["chain_model"].value_counts().items():
            print(f"    chain model: {cm}  ({n} personas)")
    print()

    # Load base demographics (Level 1 context)
    personas_csv = SCRIPT_DIR / args.personas_csv
    if not personas_csv.exists():
        personas_csv = SCRIPT_DIR / "gss2024_personas.csv"
    demographics_df = pd.read_csv(personas_csv).set_index("respondent_id")

    # Merge: keep only personas that have both a chain card and demographics
    valid_ids = cards_df.index.intersection(demographics_df.index)
    if len(valid_ids) < len(cards_df):
        print(
            f"  Warning: {len(cards_df) - len(valid_ids)} chain personas have no "
            f"matching demographics row — they will be skipped."
        )
    cards_df   = cards_df.loc[valid_ids]

    # Optional persona limit
    if args.personas and args.personas < len(cards_df):
        random.seed(42)
        sampled_ids = random.sample(list(cards_df.index), args.personas)
        cards_df = cards_df.loc[sampled_ids]

    print(f"Running survey on {len(cards_df)} personas")
    print(f"Questions:  {len(GSS_QUESTIONS_COMPREHENSIVE)} "
          f"({len(GSS_QUESTIONS_CULTUREWAR)} culture-war + "
          f"{len(GSS_QUESTIONS_NONCULTUREWAR)} non-culture-war)")
    print(f"Runs/item:  {args.runs}")
    print(f"Workers:    {args.max_workers}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Output dir: {output_dir}")
    print()

    # ---- Per-model loop (mirrors baseline exactly) ----
    for model in models:
        print("=" * 70)
        print(f"Model: {model}")
        print("=" * 70)

        model_filename = model.replace("/", "_")
        output_file    = output_dir / f"{model_filename}.csv"

        completed_tasks = load_completed_tasks(output_file)
        if completed_tasks:
            print(f"  Resuming: {len(completed_tasks)} tasks already done")

        # Build task list
        tasks: List[Dict] = []
        for persona_id, card_row in cards_df.iterrows():
            persona_card     = str(card_row["persona_card"])
            chain_model      = str(card_row.get("chain_model", "unknown"))
            demographics     = str(demographics_df.loc[persona_id, "persona"])
            event_chain      = str(card_row.get("event_chain", ""))
            compressed_chain = str(card_row.get("compressed_chain", ""))
            event_narrative  = str(card_row.get("event_narrative", ""))
            latent_state     = format_latent_state(card_row)

            for var_name, q_data in GSS_QUESTIONS_COMPREHENSIVE.items():
                for run in range(1, args.runs + 1):
                    if (persona_id, var_name, run) in completed_tasks:
                        continue
                    tasks.append({
                        "persona_id":      persona_id,
                        "demographics":    demographics,
                        "persona_card":    persona_card,
                        "chain_model":     chain_model,
                        "event_chain":     event_chain,
                        "compressed_chain": compressed_chain,
                        "event_narrative": event_narrative,
                        "latent_state":    latent_state,
                        "variable":        var_name,
                        "question":        q_data["text"],
                        "options":         q_data["options"],
                        "run":             run,
                    })

        total_tasks = len(tasks) + len(completed_tasks)
        if not tasks:
            print(f"  All {total_tasks} tasks already complete.\n")
            continue

        print(f"  Tasks: {len(tasks)} remaining of {total_tasks} total\n")

        results = []

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    query_openrouter,
                    model,
                    task["demographics"],
                    task["persona_card"],
                    task["question"],
                    task["options"],
                    api_key,
                    year,
                    args.prompt_mode,
                    task["event_chain"],
                    task["compressed_chain"],
                    task["event_narrative"],
                    task["latent_state"],
                ): task
                for task in tasks
            }

            pbar = tqdm(total=len(tasks), desc=model.split("/")[-1])

            for future in as_completed(future_to_task):
                task   = future_to_task[future]
                result = future.result()

                results.append({
                    "timestamp":         datetime.now(timezone.utc).isoformat(),
                    "model":             model,
                    "chain_model":       task["chain_model"],
                    "prompt_mode":       args.prompt_mode,
                    "persona_id":        task["persona_id"],
                    "variable":          task["variable"],
                    "question_short":    task["question"][:50] + "...",
                    "run":               task["run"],
                    "answer":            result.get("answer"),
                    "prompt_tokens":     result.get("prompt_tokens", 0),
                    "completion_tokens": result.get("completion_tokens", 0),
                    "total_tokens":      (result.get("prompt_tokens", 0)
                                         + result.get("completion_tokens", 0)),
                    "error":             result.get("error", ""),
                    "raw_response":      result.get("raw_response", ""),
                })

                pbar.update(1)

                # Batch save
                if len(results) >= args.batch_size:
                    _append_results(results, output_file)
                    results = []

            pbar.close()

        # Flush remainder
        if results:
            _append_results(results, output_file)

        # Summary
        full_df      = pd.read_csv(output_file)
        success_rate = full_df["answer"].notna().mean() * 100
        total_tokens = full_df["total_tokens"].sum()
        print(f"\n  Saved: {output_file}")
        print(f"  Responses:   {len(full_df):,}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total tokens: {total_tokens:,}\n")

    print("=" * 70)
    print("All models complete.")
    print("=" * 70)


def _append_results(results: list, output_file: Path) -> None:
    df = pd.DataFrame(results)
    if output_file.exists():
        df.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
