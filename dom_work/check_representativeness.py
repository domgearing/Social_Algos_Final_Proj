#!/usr/bin/env python3
"""
check_representativeness.py

Compares the demographic distribution of:
  (A) The full GSS 2024 sample  — parsed from gss2024_personas.csv (3,309 rows)
  (B) The life-chain subset     — parsed from life_chains/*.json (200 files)

against approximate US Census / ACS 2023 benchmarks for adults 18+.

Variables checked: sex, race, age group, education, party ID, political ideology.

Run from dom_work/:
    python check_representativeness.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# US Census / ACS 2023 benchmarks (adults 18+)
# ---------------------------------------------------------------------------

CENSUS = {
    "sex": {
        "male":   0.488,
        "female": 0.512,
    },
    "race": {
        "white_only": 0.596,   # White alone, non-Hispanic
        "black":      0.124,   # Black alone
        "other":      0.280,   # Hispanic + Asian + other (GSS conflates these into "other")
    },
    "age_group": {
        "18-34": 0.298,
        "35-54": 0.330,
        "55+":   0.372,
    },
    "educ_group": {
        "< High school": 0.089,
        "High school":   0.268,
        "Some college":  0.281,
        "Bachelor+":     0.362,
    },
}

# ---------------------------------------------------------------------------
# Demographic parser  (replicates initialize_latent_vars.parse_persona logic)
# ---------------------------------------------------------------------------

def parse_demo(text: str) -> dict:
    d: dict = {}

    # Age
    m = re.search(r"I am (\d+) years old", text)
    d["age"] = int(m.group(1)) if m else None

    # Sex
    if "I am male" in text:
        d["sex"] = "male"
    elif "I am female" in text:
        d["sex"] = "female"
    else:
        d["sex"] = None

    # Race
    if "I am Black or African American" in text:
        d["race"] = "black"
    elif "I am White" in text:
        d["race"] = "white_only"
    else:
        d["race"] = "other"

    # Education → years
    educ_rules = [
        (r"I completed high school",          12),
        (r"I had 1 year of college",          13),
        (r"I had 2 years of college",         14),
        (r"I had 3 years of college",         15),
        (r"I completed 4 years of college",   16),
        (r"I had 5 years of college",         17),
        (r"I had 6 years of college",         18),
        (r"I had 7 years of college",         19),
        (r"I had 8 or more years of college", 20),
        (r"I completed 1[01]th grade",        10),
        (r"I completed 9th grade",             9),
        (r"I completed 8th grade",             8),
    ]
    d["educ"] = None
    for pattern, val in educ_rules:
        if re.search(pattern, text):
            d["educ"] = val
            break

    # Party ID
    partyid_map = [
        ("I am a strong Democrat",                              "Strong Dem"),
        ("I am a not very strong Democrat",                     "Lean Dem"),
        ("I am an independent, close to Democrat",              "Lean Dem"),
        ("I am an independent (neither Democrat nor Republican)","Independent"),
        ("I am an independent, close to Republican",            "Lean Rep"),
        ("I am a not very strong Republican",                   "Lean Rep"),
        ("I am a strong Republican",                            "Strong Rep"),
    ]
    d["partyid"] = None
    for label, bucket in partyid_map:
        if label in text:
            d["partyid"] = bucket
            break

    # Political views
    polviews_map = [
        ("I am extremely liberal",      "Liberal"),
        ("I am slightly liberal",       "Liberal"),
        ("I am liberal",                "Liberal"),
        ("I am moderate",               "Moderate"),
        ("I am slightly conservative",  "Conservative"),
        ("I am extremely conservative", "Conservative"),
        ("I am conservative",           "Conservative"),
    ]
    d["polviews"] = None
    for label, bucket in polviews_map:
        if label in text:
            d["polviews"] = bucket
            break

    return d


def educ_group(yrs):
    if yrs is None:
        return None
    if yrs <= 11:
        return "< High school"
    if yrs == 12:
        return "High school"
    if yrs <= 15:
        return "Some college"
    return "Bachelor+"


def age_group(age):
    if age is None:
        return None
    if age < 35:
        return "18-34"
    if age < 55:
        return "35-54"
    return "55+"


# ---------------------------------------------------------------------------
# Load full GSS sample from personas CSV
# ---------------------------------------------------------------------------

print("Loading full GSS sample from gss2024_personas.csv ...")
personas_df = pd.read_csv(SCRIPT_DIR / "gss2024_personas.csv")
gss_records = [parse_demo(str(row["persona"])) for _, row in personas_df.iterrows()]
gss_df = pd.DataFrame(gss_records)
gss_df["age_grp"]   = gss_df["age"].apply(age_group)
gss_df["educ_grp"]  = gss_df["educ"].apply(educ_group)
print(f"  {len(gss_df)} respondents\n")

# ---------------------------------------------------------------------------
# Load life-chain subset from JSON files
# ---------------------------------------------------------------------------

chains_dir = SCRIPT_DIR / "life_chains"
json_files = sorted(chains_dir.glob("*.json"))
print(f"Loading life-chain personas from {chains_dir} ...")
chain_records = []
for jf in json_files:
    try:
        with open(jf) as f:
            data = json.load(f)
        demo_text = data.get("demographics", "")
        rec = parse_demo(demo_text)
        rec["respondent_id"] = data.get("respondent_id")
        chain_records.append(rec)
    except Exception as e:
        print(f"  Warning: could not read {jf.name}: {e}")

chain_df = pd.DataFrame(chain_records)
chain_df["age_grp"]  = chain_df["age"].apply(age_group)
chain_df["educ_grp"] = chain_df["educ"].apply(educ_group)
print(f"  {len(chain_df)} life-chain personas\n")

# ---------------------------------------------------------------------------
# Helper: distribution comparison table
# ---------------------------------------------------------------------------

def dist(series):
    counts = series.value_counts(dropna=True)
    return (counts / counts.sum()).round(4)


def comparison_table(var_label, full_dist, chain_dist, census_dict=None):
    all_keys = sorted(
        set(full_dist.index) | set(chain_dist.index) |
        (set(census_dict.keys()) if census_dict else set())
    )
    rows = []
    for k in all_keys:
        f  = full_dist.get(k,  0.0)
        c  = chain_dist.get(k, 0.0)
        bm = census_dict.get(k, float("nan")) if census_dict else float("nan")
        rows.append({
            "Category":          k,
            "GSS full %":        f"{f*100:.1f}",
            "Chain subset %":    f"{c*100:.1f}",
            "Diff (chain-GSS) %": f"{(c-f)*100:+.1f}",
            "US Census %":       f"{bm*100:.1f}" if not np.isnan(bm) else "—",
            "Chain vs Census %": f"{(c-bm)*100:+.1f}" if not np.isnan(bm) else "—",
        })
    df = pd.DataFrame(rows).set_index("Category")
    print(f"── {var_label} " + "─" * max(1, 50 - len(var_label)))
    print(df.to_string())
    print()


# ---------------------------------------------------------------------------
# Run comparisons
# ---------------------------------------------------------------------------

comparison_table("Sex",       dist(gss_df["sex"]),      dist(chain_df["sex"]),      CENSUS["sex"])
comparison_table("Race",      dist(gss_df["race"]),     dist(chain_df["race"]),     CENSUS["race"])
comparison_table("Age group", dist(gss_df["age_grp"]),  dist(chain_df["age_grp"]),  CENSUS["age_group"])
comparison_table("Education", dist(gss_df["educ_grp"]), dist(chain_df["educ_grp"]), CENSUS["educ_group"])
comparison_table("Party ID",  dist(gss_df["partyid"]),  dist(chain_df["partyid"]),  None)
comparison_table("Pol views", dist(gss_df["polviews"]), dist(chain_df["polviews"]), None)

# ---------------------------------------------------------------------------
# Age & education means
# ---------------------------------------------------------------------------

print("── Continuous means " + "─" * 31)
rows = []
for col, label in [("age", "Age (years)"), ("educ", "Education (years)")]:
    gf = gss_df[col].dropna().mean()
    cf = chain_df[col].dropna().mean()
    rows.append({"Variable": label, "GSS full": f"{gf:.1f}", "Chain subset": f"{cf:.1f}",
                 "Diff (chain-GSS)": f"{cf-gf:+.1f}"})
print(pd.DataFrame(rows).set_index("Variable").to_string())
print()

# ---------------------------------------------------------------------------
# Parse-rate check
# ---------------------------------------------------------------------------

print("── Parse rates " + "─" * 35)
for col in ["sex", "race", "age", "educ", "partyid", "polviews"]:
    gn = gss_df[col].notna().mean() * 100
    cn = chain_df[col].notna().mean() * 100
    print(f"  {col:10s}  GSS: {gn:.1f}%   Chain: {cn:.1f}%")
