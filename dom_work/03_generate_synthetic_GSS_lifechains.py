#!/usr/bin/env python3
"""
03_generate_synthetic_GSS_lifechains.py

Phase 4: Query LLMs with the same 52 GSS attitudinal items as the baseline
(01_generate_synthetic_GSS.py), but using life-chain persona cards instead
of static demographic strings.

Prompt structure per survey item
---------------------------------
  Step 0  (once per persona): convert third-person persona card → first-person
  LIFE HISTORY AND WORLDVIEW:  (first-person persona, ~150 tokens)
  Question + options            (~30 tokens)
  → answer: single digit

The BACKGROUND / demographics section has been removed.  The first-person
conversion is done once per persona and cached in memory for the survey loop.

Temperature is set to 0.7 (medium) — lower than the life-chain generation
stage but with enough variance to avoid degenerate mode-seeking behaviour.

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
# First-person conversion — called once per persona before the survey loop
# ---------------------------------------------------------------------------

def convert_to_first_person(
    model: str,
    persona_card: str,
    api_key: str,
    timeout: int = 30,
    max_retries: int = 3,
) -> str:
    """
    Rewrite a third-person persona card as a first-person self-description.

    Returns the converted text, or the original card unchanged if the API
    call fails (so the survey can still proceed with the raw card).
    """
    prompt = (
        "Rewrite the following third-person description as a first-person "
        "self-description. Keep all factual details and preserve the tone. "
        "Do not add or remove any information. Output only the rewritten text, "
        "with no preamble or explanation.\n\n"
        f"{persona_card}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_API_URL, headers=headers, json=data, timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            converted = result["choices"][0]["message"]["content"].strip()
            if converted:
                return converted
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # Fallback: return the original card unmodified
    return persona_card


# ---------------------------------------------------------------------------
# Survey query — demographics removed; uses first-person persona directly
# ---------------------------------------------------------------------------

def query_openrouter(
    model: str,
    persona_card: str,
    question: str,
    options: Dict[int, str],
    api_key: str,
    year: int,
    timeout: int = 30,
    max_retries: int = 3,
) -> Dict:
    """
    Query OpenRouter for a single GSS item using the life-chain prompt.

    The persona is provided as a first-person self-description derived from
    the life-chain persona card.  The separate BACKGROUND/demographics block
    has been removed.
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    prompt = (
        f"It is now {year}. You are answering survey questions as the following person, "
        f"who is living in the United States.\n\n"
        f"LIFE HISTORY AND WORLDVIEW:\n{persona_card}\n\n"
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
        "temperature": 0.7,
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
# Resume helpers
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


def load_first_person_cache(cache_path: Path) -> Dict[int, str]:
    """
    Load any already-converted first-person persona cards from disk.

    Returns a dict mapping respondent_id (int) -> first-person text.
    """
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_csv(cache_path)
        return dict(zip(df["respondent_id"].astype(int), df["first_person_card"]))
    except Exception as e:
        print(f"Warning: could not load first-person cache ({cache_path}): {e}")
        return {}


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
    """
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
        # Use 'model' column as chain_model if present, else "unknown"
        if "model" not in df.columns:
            df["chain_model"] = "unknown"
        else:
            df = df.rename(columns={"model": "chain_model"})
        return df[["respondent_id", "persona_card", "chain_model"]].set_index("respondent_id")

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
                records.append({
                    "respondent_id": chain["respondent_id"],
                    "persona_card":  chain.get("persona_card", ""),
                    "chain_model":   chain_model,
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
        "--output-dir", default=None,
        help="Override output directory (default: synthetic_data_lifechains/year_<year>/)"
    )
    parser.add_argument(
        "--conversion-model", default=None,
        help=(
            "Model to use for the first-person conversion step. "
            "Defaults to the first survey model if not specified."
        )
    )
    parser.add_argument(
        "--fp-cache", default="life_chains_first_person.csv",
        help=(
            "Path to the first-person persona cache CSV "
            "(default: life_chains_first_person.csv in the script directory). "
            "Existing entries are reused; new conversions are appended."
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

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SCRIPT_DIR / "synthetic_data_lifechains" / f"year_{year}"
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
    print(f"Output dir: {output_dir}")
    print()

    # Determine conversion model (default to first survey model)
    conversion_model = args.conversion_model if args.conversion_model else models[0]

    # ── First-person conversion (resumable, persisted to CSV) ───────────────
    fp_cache_path = SCRIPT_DIR / args.fp_cache
    first_person_cards: Dict[int, str] = load_first_person_cache(fp_cache_path)

    personas_to_convert = [
        (pid, row) for pid, row in cards_df.iterrows()
        if pid not in first_person_cards
    ]

    if personas_to_convert:
        print(
            f"Converting {len(personas_to_convert)} persona cards to first-person "
            f"using '{conversion_model}' ..."
        )
        if first_person_cards:
            print(f"  (Resuming — {len(first_person_cards)} already cached in {fp_cache_path.name})")

        for persona_id, card_row in tqdm(personas_to_convert, desc="converting"):
            converted = convert_to_first_person(
                model        = conversion_model,
                persona_card = str(card_row["persona_card"]),
                api_key      = api_key,
            )
            first_person_cards[persona_id] = converted

            # Persist immediately so progress is not lost on crash
            row_df = pd.DataFrame([{
                "respondent_id":     persona_id,
                "first_person_card": converted,
                "conversion_model":  conversion_model,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            }])
            row_df.to_csv(
                fp_cache_path,
                mode="a",
                header=not fp_cache_path.exists(),
                index=False,
            )

        print(f"  Conversion complete. Cache saved to {fp_cache_path}\n")
    else:
        print(
            f"  All {len(first_person_cards)} persona cards already converted "
            f"(loaded from {fp_cache_path.name}).\n"
        )

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

        # Build task list (uses pre-converted first-person cards)
        tasks: List[Dict] = []
        for persona_id, card_row in cards_df.iterrows():
            persona_card = first_person_cards[persona_id]
            chain_model  = str(card_row.get("chain_model", "unknown"))

            for var_name, q_data in GSS_QUESTIONS_COMPREHENSIVE.items():
                for run in range(1, args.runs + 1):
                    if (persona_id, var_name, run) in completed_tasks:
                        continue
                    tasks.append({
                        "persona_id":   persona_id,
                        "persona_card": persona_card,
                        "chain_model":  chain_model,
                        "variable":     var_name,
                        "question":     q_data["text"],
                        "options":      q_data["options"],
                        "run":          run,
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
                    task["persona_card"],
                    task["question"],
                    task["options"],
                    api_key,
                    year,
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
