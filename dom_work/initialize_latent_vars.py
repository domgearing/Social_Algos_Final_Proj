#!/usr/bin/env python3
"""
initialize_latent_vars.py

Phase 2: Initialize Beta-distribution priors for 5 latent belief variables
from the natural-language persona strings in gss2024_personas.csv.

Output: gss2024_latent_priors.csv
  One row per persona with columns:
    respondent_id
    For each variable V in {moral_traditionalism, economic_redistribution,
                            racial_attitudes, institutional_trust, outgroup_affect}:
      V_alpha, V_beta  — Beta distribution parameters
      V_mean           — alpha / (alpha + beta), the initialization point [0, 1]
      V_concentration  — alpha + beta, how "confident" the prior is

  Also includes the parsed demographic fields for auditability and use in Phase 3.

Variable directions (all 0→1):
  moral_traditionalism:    0 = progressive/permissive  → 1 = traditional/conservative
  economic_redistribution: 0 = anti-redistribution     → 1 = pro-redistribution
  racial_attitudes:        0 = individualist framing   → 1 = structuralist framing
  institutional_trust:     0 = low trust               → 1 = high trust
  outgroup_affect:         0 = closed / low tolerance  → 1 = open / welcoming

Usage:
    python initialize_latent_vars.py
    python initialize_latent_vars.py --input gss2024_personas.csv --output gss2024_latent_priors.csv
    python initialize_latent_vars.py --sample 200 --seed 42   # stratified sample
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Demographic parsing
# ---------------------------------------------------------------------------

def parse_persona(text: str) -> dict:
    """
    Parse a natural-language GSS persona string into a demographic dict.

    Returns a dict with keys:
      age (int|None), sex (str|None), race (int|None), educ (int|None),
      income (int|None), partyid (int|None), polviews (int|None),
      relig (int|None), attend (int|None)

    None means the field was absent from the persona string.
    """
    d: dict = {k: None for k in
               ["age", "sex", "race", "educ", "income",
                "partyid", "polviews", "relig", "attend"]}

    # --- Age ---
    m = re.search(r"I am (\d+) years old", text)
    if m:
        d["age"] = int(m.group(1))
    elif "89 years or older" in text:
        d["age"] = 89

    # --- Sex ---
    if "I am male" in text:
        d["sex"] = "male"
    elif "I am female" in text:
        d["sex"] = "female"

    # --- Race ---
    if "I am Black or African American" in text:
        d["race"] = 2
    elif "I am White" in text:
        d["race"] = 1
    elif "I am of another race" in text:
        d["race"] = 3

    # --- Education (years: 0–20) ---
    # Order matters: more specific patterns first
    educ_rules = [
        (r"I have no formal schooling", 0),
        (r"I completed 1st grade", 1),
        (r"I completed 2nd grade", 2),
        (r"I completed 3rd grade", 3),
        (r"I completed 4th grade", 4),
        (r"I completed 5th grade", 5),
        (r"I completed 6th grade", 6),
        (r"I completed 7th grade", 7),
        (r"I completed 8th grade", 8),
        (r"I completed 9th grade", 9),
        (r"I completed 10th grade", 10),
        (r"I completed 11th grade", 11),
        (r"I completed high school", 12),
        (r"I had 1 year of college", 13),
        (r"I had 2 years of college", 14),
        (r"I had 3 years of college", 15),
        (r"I completed 4 years of college", 16),
        (r"I had 5 years of college", 17),
        (r"I had 6 years of college", 18),
        (r"I had 7 years of college", 19),
        (r"I had 8 or more years of college", 20),
    ]
    for pattern, val in educ_rules:
        if re.search(pattern, text):
            d["educ"] = val
            break

    # --- Income bracket (1–26) ---
    income_rules = [
        ("less than $1,000", 1),
        ("$1,000 to $2,999", 2),
        ("$3,000 to $3,999", 3),
        ("$4,000 to $4,999", 4),
        ("$5,000 to $5,999", 5),
        ("$6,000 to $6,999", 6),
        ("$7,000 to $7,999", 7),
        ("$8,000 to $9,999", 8),
        ("$10,000 to $12,499", 9),
        ("$12,500 to $14,999", 10),
        ("$15,000 to $17,499", 11),
        ("$17,500 to $19,999", 12),
        ("$20,000 to $22,499", 13),
        ("$22,500 to $24,999", 14),
        ("$25,000 to $29,999", 15),
        ("$30,000 to $34,999", 16),
        ("$35,000 to $39,999", 17),
        ("$40,000 to $49,999", 18),
        ("$50,000 to $59,999", 19),
        ("$60,000 to $74,999", 20),
        ("$75,000 to $89,999", 21),
        ("$90,000 to $109,999", 22),
        ("$110,000 to $129,999", 23),
        ("$130,000 to $149,999", 24),
        ("$150,000 to $169,999", 25),
        ("$170,000 or more", 26),
    ]
    for label, val in income_rules:
        if label in text:
            d["income"] = val
            break

    # --- Party ID (0=strong Dem … 6=strong Rep) ---
    # Longer/more specific strings must come before shorter ones
    partyid_rules = [
        ("I am a strong Democrat", 0),
        ("I am a not very strong Democrat", 1),
        ("I am an independent, close to Democrat", 2),
        ("I am an independent (neither Democrat nor Republican)", 3),
        ("I am an independent, close to Republican", 4),
        ("I am a not very strong Republican", 5),
        ("I am a strong Republican", 6),
    ]
    for label, val in partyid_rules:
        if label in text:
            d["partyid"] = val
            break

    # --- Political views (1=ext liberal … 7=ext conservative) ---
    # "slightly" before plain "liberal"/"conservative" to avoid substring match
    polviews_rules = [
        ("I am extremely liberal", 1),
        ("I am slightly liberal", 3),
        ("I am liberal", 2),
        ("I am moderate", 4),
        ("I am slightly conservative", 5),
        ("I am extremely conservative", 7),
        ("I am conservative", 6),
    ]
    for label, val in polviews_rules:
        if label in text:
            d["polviews"] = val
            break

    # --- Religion ---
    relig_rules = [
        ("My religion is Protestant", 1),
        ("My religion is Catholic", 2),
        ("My religion is Jewish", 3),
        ("I have no religious affiliation", 4),
        ("My religion is Buddhism", 6),
        ("My religion is Hinduism", 7),
        ("My religion is Other Eastern", 8),
        ("My religion is Islam", 9),
        ("My religion is Orthodox Christian", 10),
        ("My religion is Christian", 11),
        ("My religion is Native American", 12),
        ("My religion is Inter/Nondenominational", 13),
        ("My religion is Other", 5),
    ]
    for label, val in relig_rules:
        if label in text:
            d["relig"] = val
            break

    # --- Religious attendance (0=never … 8=more than once/week) ---
    # More specific patterns first ("more than once" before "every week")
    attend_rules = [
        ("I never attend religious services", 0),
        ("I attend religious services less than once a year", 1),
        ("I attend religious services about once or twice a year", 2),
        ("I attend religious services several times a year", 3),
        ("I attend religious services about once a month", 4),
        ("I attend religious services 2-3 times a month", 5),
        ("I attend religious services nearly every week", 6),
        ("I attend religious services more than once a week", 8),
        ("I attend religious services every week", 7),
    ]
    for label, val in attend_rules:
        if label in text:
            d["attend"] = val
            break

    return d


# ---------------------------------------------------------------------------
# Beta prior computation
# ---------------------------------------------------------------------------

LATENT_VARS = [
    "moral_traditionalism",
    "economic_redistribution",
    "racial_attitudes",
    "institutional_trust",
    "outgroup_affect",
]


def _beta_params(mu: float, kappa: float) -> tuple[float, float]:
    """Return (alpha, beta) for a Beta distribution with given mean and concentration."""
    mu = float(np.clip(mu, 0.05, 0.95))
    kappa = max(kappa, 2.0)
    return mu * kappa, (1.0 - mu) * kappa


def infer_relig_trad(
    attend: int | None,
    race: int | None,
    pid_rep: float | None,
    pv_con: float | None,
) -> float | None:
    """
    Infer a religion-type traditionalism score (0–1) when the GSS relig code
    is unavailable, using attendance, race, and political identity as proxies.

    This is NOT the same as attend_n (raw commitment level) — it captures
    *denomination type*:
      - A high-attendance evangelical Protestant → high trad (~0.78)
      - A high-attendance Black Baptist → moderately high trad (~0.68)
      - A weekly-attending Unitarian → low trad (~0.30) despite high attendance
      - A low-attendance cultural Catholic → moderate (~0.45)
      - A secular none → low trad (~0.18)

    Grounding:
      - Pew (2023): ~26% of Americans are religiously unaffiliated
      - Black Americans: ~87% Christian, mostly Baptist/evangelical Protestant
      - White Republicans: ~60% evangelical/conservative Protestant
      - White Democrats: ~45% none or mainline Protestant
      - Hout & Fischer (2014): political polarization drives low-attend liberals
        toward "none" label, while high-attend conservatives stay in traditionalist
        denominations; attend × politics interaction is the key signal.
    """
    if attend is None and pid_rep is None and pv_con is None:
        return None

    # ---- Attendance base ----
    # Anchors: 0=never (likely none/secular) → 8=more than once/week (devout traditional)
    # At attend=0: ~65% are nones or nominally affiliated → expected trad ≈ 0.20
    # At attend=8: ~90% are in active traditional congregations → expected trad ≈ 0.76
    if attend is not None:
        attend_n = attend / 8.0
        # Non-linear: low attend → secularism drops fast; high attend → trad rises
        attend_base = 0.18 + 0.58 * attend_n
    else:
        attend_base = 0.48  # no attendance info → weak prior

    # ---- Race adjustment ----
    # Black Americans who attend are overwhelmingly Baptist/Pentecostal (trad on morality);
    # White low-attenders are more likely truly secular than culturally nominal.
    race_adj = 0.0
    if race == 2:      # Black — Baptist/Pentecostal predominates, trad on moral items
        race_adj = +0.08
    elif race == 1:    # White — low attenders more secular than Black equivalents
        if attend is not None and attend <= 2:
            race_adj = -0.06

    # ---- Politics × attendance interaction ----
    # At high attendance, conservative politics → evangelical (trad ↑)
    # At low attendance, liberal politics confirms secular/none (trad ↓)
    pol_signal = None
    if pid_rep is not None and pv_con is not None:
        pol_signal = pid_rep * 0.5 + pv_con * 0.5  # 0=liberal Dem, 1=conservative Rep
    elif pid_rep is not None:
        pol_signal = pid_rep
    elif pv_con is not None:
        pol_signal = pv_con

    pol_adj = 0.0
    if pol_signal is not None:
        centered = pol_signal - 0.5  # −0.5=max liberal, +0.5=max conservative
        if attend is not None and attend >= 5:
            # High attendance + conservative politics → evangelical denomination
            pol_adj = centered * 0.18
        elif attend is not None and attend <= 1:
            # Low attendance + liberal politics → confirmed secular
            pol_adj = centered * 0.14
        else:
            pol_adj = centered * 0.10

    return float(np.clip(attend_base + race_adj + pol_adj, 0.10, 0.90))


def compute_latent_priors(d: dict) -> dict[str, tuple[float, float]]:
    """
    Map parsed demographic dict → Beta(alpha, beta) for each of the 5 latent variables.

    Design principles
    -----------------
    * Only use correlates with robust empirical backing; keep weights sparse.
    * Missing demographics → fall back to uniform-ish prior (mu=0.5, low concentration).
    * Concentration (kappa = alpha+beta) reflects how confidently demographics
      predict the variable, not just directionality.
    * These are PRIORS that the life-event chain will update — they don't need to
      be perfect, just directionally correct and not too narrow.
    """

    # ---- Normalize raw codes to [0, 1] signals ----

    # partyid: 0=strong Dem, 6=strong Rep
    pid = d["partyid"]
    pid_rep   = (pid / 6.0) if pid is not None else None   # 0=Dem, 1=Rep
    pid_dem   = (1.0 - pid_rep) if pid_rep is not None else None
    pid_extr  = abs((pid / 3.0) - 1.0) if pid is not None else None  # 0=moderate, 1=extreme

    # polviews: 1=ext liberal, 7=ext conservative
    pv = d["polviews"]
    pv_con    = ((pv - 1) / 6.0) if pv is not None else None   # 0=liberal, 1=conservative
    pv_lib    = (1.0 - pv_con) if pv_con is not None else None

    # Combined conservative/liberal signal; prefer polviews > partyid for ideology
    def _avg(*vals):
        present = [v for v in vals if v is not None]
        return float(np.mean(present)) if present else None

    con_signal = _avg(
        pv_con * 0.6 + pid_rep * 0.4 if (pv_con is not None and pid_rep is not None) else pv_con,
        pid_rep if pv_con is None else None,
    )
    lib_signal = (1.0 - con_signal) if con_signal is not None else None

    # Education: 0–20 → [0, 1]
    educ = d["educ"]
    educ_n = (educ / 20.0) if educ is not None else None

    # Income: 1–26 → [0, 1]
    income = d["income"]
    income_n = ((income - 1) / 25.0) if income is not None else None

    # Religion traditionalism score: Protestant/Catholic/Evangelical → higher
    relig = d["relig"]
    _relig_trad_lookup = {
        1: 0.72,   # Protestant (broad; skews traditional)
        2: 0.65,   # Catholic
        3: 0.40,   # Jewish
        4: 0.18,   # No religion
        5: 0.50,   # Other
        6: 0.38,   # Buddhism
        7: 0.48,   # Hinduism
        8: 0.45,   # Other Eastern
        9: 0.68,   # Islam
        10: 0.70,  # Orthodox Christian
        11: 0.68,  # Christian (generic)
        12: 0.55,  # Native American
        13: 0.60,  # Inter/Nondenominational
    }
    if relig is not None:
        relig_trad = _relig_trad_lookup.get(relig, None)
    else:
        # GSS 2024 has no relig codes — infer denomination type from available proxies.
        # pid_rep and pv_con may not be computed yet at this point in the function,
        # so we compute them locally here (they're cheap).
        _pid_rep_local = (d["partyid"] / 6.0) if d["partyid"] is not None else None
        _pv_con_local  = ((d["polviews"] - 1) / 6.0) if d["polviews"] is not None else None
        relig_trad = infer_relig_trad(
            attend  = d["attend"],
            race    = d["race"],
            pid_rep = _pid_rep_local,
            pv_con  = _pv_con_local,
        )

    # Attendance: 0–8 → [0, 1]
    attend = d["attend"]
    attend_n = (attend / 8.0) if attend is not None else None

    # Race
    race = d["race"]

    # Age: 18–89 → [0, 1]
    age = d["age"]
    age_n = min(age / 89.0, 1.0) if age is not None else None

    # -----------------------------------------------------------------------
    # Helper: weighted average over (signal, weight) pairs, ignoring Nones
    # -----------------------------------------------------------------------
    def wavg(pairs: list[tuple]) -> float | None:
        """pairs = list of (signal_or_None, weight). Returns None if no signals."""
        vals = [(s, w) for s, w in pairs if s is not None]
        if not vals:
            return None
        signals, weights = zip(*vals)
        return float(np.average(signals, weights=weights))

    priors = {}

    # -------------------------------------------------------------------
    # 1. MORAL TRADITIONALISM
    #    Higher → more traditional (abortion restrictions, anti-homosex,
    #    traditional gender roles, biblical literalism)
    #
    #    Primary: polviews, partyid, relig, attend
    #    Secondary: age (older → slightly higher), educ (higher → slightly lower)
    # -------------------------------------------------------------------
    mu_mt = wavg([
        (pv_con,      0.35),
        (pid_rep,     0.25),
        (relig_trad,  0.22),
        (attend_n,    0.12),
        (age_n * 0.6 + 0.2 if age_n is not None else None, 0.04),   # weak age effect
        (1.0 - educ_n * 0.35 if educ_n is not None else None, 0.02), # weak educ effect
    ])
    if mu_mt is None:
        mu_mt, kappa_mt = 0.5, 2.5
    else:
        # Concentration scales with strength of political/religious signal
        signal_strength = max(
            abs(pv_con - 0.5) * 2 if pv_con is not None else 0,
            abs(pid_rep - 0.5) * 2 if pid_rep is not None else 0,
            attend_n if attend_n is not None else 0,
        )
        kappa_mt = 5.0 + 6.0 * signal_strength
    priors["moral_traditionalism"] = _beta_params(mu_mt, kappa_mt)

    # -------------------------------------------------------------------
    # 2. ECONOMIC REDISTRIBUTION
    #    Higher → more pro-redistribution (eqwlth, gov spending, unions)
    #
    #    Primary: polviews (liberal → high), partyid (Dem → high),
    #             income (low income → high — people vote their pocketbook)
    #    Secondary: educ (complex: net slight positive — more educated Dems,
    #               but college-educated are increasingly partisan, keep weak)
    # -------------------------------------------------------------------
    mu_er = wavg([
        (pv_lib,             0.32),
        (pid_dem,            0.32),
        (1.0 - income_n if income_n is not None else None, 0.26),
        (0.3 + educ_n * 0.4 if educ_n is not None else None, 0.10),
    ])
    if mu_er is None:
        mu_er, kappa_er = 0.5, 2.5
    else:
        signal_strength = max(
            abs(pv_lib - 0.5) * 2 if pv_lib is not None else 0,
            abs(pid_dem - 0.5) * 2 if pid_dem is not None else 0,
        )
        kappa_er = 4.5 + 5.0 * signal_strength
    priors["economic_redistribution"] = _beta_params(mu_er, kappa_er)

    # -------------------------------------------------------------------
    # 3. RACIAL ATTITUDES
    #    Higher → more structural/systemic explanations for racial inequality
    #
    #    Primary: race (Black → high), polviews, partyid
    #    Secondary: educ (higher → slightly more structural)
    #
    #    Race effect is large and well-documented (Kluegel & Smith 1986;
    #    Pew 2019 race & inequality survey).
    # -------------------------------------------------------------------
    race_structural = {1: 0.35, 2: 0.82, 3: 0.58}.get(race, None)  # White/Black/Other
    mu_ra = wavg([
        (race_structural,    0.40),
        (pv_lib,             0.28),
        (pid_dem,            0.20),
        (0.30 + educ_n * 0.40 if educ_n is not None else None, 0.12),
    ])
    if mu_ra is None:
        mu_ra, kappa_ra = 0.5, 2.5
    else:
        # Black respondents have a much stronger and more consistent prior
        race_boost = 5.0 if race == 2 else 0.0
        signal_strength = max(
            abs(pv_lib - 0.5) * 2 if pv_lib is not None else 0,
            abs(pid_dem - 0.5) * 2 if pid_dem is not None else 0,
        )
        kappa_ra = 4.5 + 4.0 * signal_strength + race_boost
    priors["racial_attitudes"] = _beta_params(mu_ra, kappa_ra)

    # -------------------------------------------------------------------
    # 4. INSTITUTIONAL TRUST
    #    Higher → more trust in government, press, science, religious institutions
    #
    #    This is the hardest to predict from demographics alone.
    #    Moderation (low partisanship extremity) → more trust.
    #    Higher SES (income + educ) → somewhat more trust in formal institutions.
    #    High religious attendance → more trust in religious institutions
    #    (which loads onto this composite variable).
    #
    #    Intentionally kept with LOW concentration: demographics predict the
    #    direction of institutional trust poorly; life events are the main driver.
    # -------------------------------------------------------------------
    moderation = (1.0 - pid_extr) if pid_extr is not None else None  # 0=extreme, 1=moderate
    mu_it = wavg([
        (0.35 + moderation * 0.35 if moderation is not None else None, 0.30),
        (0.30 + income_n * 0.40 if income_n is not None else None,     0.25),
        (0.32 + educ_n * 0.30 if educ_n is not None else None,         0.25),
        (0.38 + attend_n * 0.25 if attend_n is not None else None,     0.20),
    ])
    if mu_it is None:
        mu_it, kappa_it = 0.5, 2.5
    else:
        # Deliberately weak concentration — life events dominate this variable
        kappa_it = 3.5 + 1.5 * (moderation if moderation is not None else 0)
    priors["institutional_trust"] = _beta_params(mu_it, kappa_it)

    # -------------------------------------------------------------------
    # 5. OUTGROUP AFFECT
    #    Higher → more positive affect toward outgroups, immigrants, etc.
    #    (generalized social trust + cosmopolitanism)
    #
    #    Primary: educ (education is the most consistent predictor;
    #             Hainmueller & Hopkins 2014 on immigration attitudes),
    #             polviews (liberal → more open)
    #    Secondary: partyid, income, religion (non-exclusive denominations)
    # -------------------------------------------------------------------
    relig_open = {
        1: 0.42, 2: 0.47, 3: 0.65, 4: 0.68,
        5: 0.50, 6: 0.65, 7: 0.58, 8: 0.58,
        9: 0.50, 10: 0.45, 11: 0.45, 12: 0.55, 13: 0.50,
    }.get(relig, None)
    mu_oa = wavg([
        (0.22 + educ_n * 0.56 if educ_n is not None else None, 0.32),
        (pv_lib,                                                 0.28),
        (pid_dem,                                                0.20),
        (0.30 + income_n * 0.30 if income_n is not None else None, 0.10),
        (relig_open,                                              0.10),
    ])
    if mu_oa is None:
        mu_oa, kappa_oa = 0.5, 2.5
    else:
        educ_signal = educ_n if educ_n is not None else 0
        pol_signal = max(
            abs(pv_lib - 0.5) * 2 if pv_lib is not None else 0,
            abs(pid_dem - 0.5) * 2 if pid_dem is not None else 0,
        )
        kappa_oa = 4.0 + 4.0 * educ_signal + 2.0 * pol_signal
    priors["outgroup_affect"] = _beta_params(mu_oa, kappa_oa)

    return priors


# ---------------------------------------------------------------------------
# Stratified sampling helper
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Draw a stratified sample of n rows from df, preserving approximate
    marginals on partyid_bucket (Dem/Ind/Rep) and race_bucket (W/B/Other).

    Falls back to random sample if parsed demographic columns are missing.
    """
    rng = np.random.default_rng(seed)

    # We need the parsed demographics already in the df
    if "partyid" not in df.columns or "race" not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=seed)

    df = df.copy()
    df["_pid_bucket"] = df["partyid"].apply(
        lambda x: "dem" if (x is not None and x <= 2) else
                  ("rep" if (x is not None and x >= 4) else "ind")
    )
    df["_race_bucket"] = df["race"].apply(
        lambda x: "white" if x == 1 else ("black" if x == 2 else "other")
    )
    df["_stratum"] = df["_pid_bucket"] + "_" + df["_race_bucket"]

    strata = df["_stratum"].value_counts(normalize=True)
    samples = []
    remaining = n

    for i, (stratum, frac) in enumerate(strata.items()):
        stratum_df = df[df["_stratum"] == stratum]
        # Last stratum gets whatever's left to hit exactly n
        k = round(frac * n) if i < len(strata) - 1 else remaining
        k = min(k, len(stratum_df))
        samples.append(stratum_df.sample(n=k, random_state=int(rng.integers(0, 9999))))
        remaining -= k

    result = pd.concat(samples).drop(columns=["_pid_bucket", "_race_bucket", "_stratum"])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def row_to_demo_dict(row) -> dict:
    """
    Convert a raw GSS numeric row (from RDS) directly to the demographic dict
    expected by compute_latent_priors — bypasses natural-language parsing.
    """
    def _int(val):
        try:
            v = float(val)
            return None if pd.isna(v) else int(v)
        except (TypeError, ValueError):
            return None

    return {
        "age":      _int(row.get("age")),
        "sex":      ("male" if _int(row.get("sex")) == 1 else
                     "female" if _int(row.get("sex")) == 2 else None),
        "race":     _int(row.get("race")),
        "educ":     _int(row.get("educ")),
        "income":   _int(row.get("income")),
        "partyid":  _int(row.get("partyid")),
        "polviews": _int(row.get("polviews")),
        "relig":    _int(row.get("relig")),
        "attend":   _int(row.get("attend")),
    }


def build_priors_df(personas_df: pd.DataFrame, raw_numeric: bool = False) -> pd.DataFrame:
    """
    For each persona in personas_df, parse demographics and compute Beta priors.
    Returns a wide DataFrame with one row per persona.

    raw_numeric=True: input rows have GSS numeric columns directly (from RDS).
    raw_numeric=False: input rows have a 'persona' text column (from 01_generate_synthetic_GSS.py).
    """
    records = []
    for _, row in personas_df.iterrows():
        if raw_numeric:
            demo = row_to_demo_dict(row)
        else:
            demo = parse_persona(str(row["persona"]))
        priors = compute_latent_priors(demo)

        rec = {"respondent_id": row["respondent_id"]}

        # Parsed demographics (for auditability and Phase 3 context)
        rec.update({f"demo_{k}": v for k, v in demo.items()})

        # Beta parameters and derived statistics
        for var, (alpha, beta) in priors.items():
            mean = alpha / (alpha + beta)
            concentration = alpha + beta
            rec[f"{var}_alpha"]         = round(alpha, 4)
            rec[f"{var}_beta"]          = round(beta, 4)
            rec[f"{var}_mean"]          = round(mean, 4)
            rec[f"{var}_concentration"] = round(concentration, 4)

        records.append(rec)

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for each latent variable's prior means."""
    print("\nLatent Variable Prior Summary")
    print("=" * 60)
    for var in LATENT_VARS:
        col = f"{var}_mean"
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        print(f"\n{var}")
        print(f"  n={len(vals)}  mean={vals.mean():.3f}  std={vals.std():.3f}"
              f"  min={vals.min():.3f}  max={vals.max():.3f}")
        q = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  p10={q[0.1]:.3f}  p25={q[0.25]:.3f}  p50={q[0.5]:.3f}"
              f"  p75={q[0.75]:.3f}  p90={q[0.9]:.3f}")

    # Parse rate check
    print("\nDemographic Parse Rates")
    print("-" * 40)
    demo_cols = [c for c in df.columns if c.startswith("demo_")]
    for col in demo_cols:
        field = col.replace("demo_", "")
        n_present = df[col].notna().sum()
        print(f"  {field:12s}: {n_present}/{len(df)} ({100*n_present/len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize Beta-distribution latent variable priors from GSS personas."
    )
    parser.add_argument(
        "--input", default="gss2024_personas.csv",
        help="Path to personas CSV (default: gss2024_personas.csv)"
    )
    parser.add_argument(
        "--output", default="gss2024_latent_priors.csv",
        help="Output path for priors CSV (default: gss2024_latent_priors.csv)"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="If set, draw a stratified sample of this many personas"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading personas from {input_path} ...")
    raw_numeric = input_path.suffix.lower() == ".rds"
    if raw_numeric:
        try:
            import pyreadr
        except ImportError:
            raise SystemExit("pyreadr is required for .rds input: pip install pyreadr")
        result = pyreadr.read_r(str(input_path))
        personas_df = list(result.values())[0].reset_index(drop=True)
        personas_df["respondent_id"] = range(1, len(personas_df) + 1)
    else:
        personas_df = pd.read_csv(input_path)
    print(f"  Loaded {len(personas_df)} personas")

    if args.sample is not None:
        if raw_numeric:
            # partyid and race columns already present as numeric
            pass
        else:
            # Need parsed demographics for stratification; do a quick parse pass first
            print(f"  Parsing demographics for stratification ...")
            parsed = [parse_persona(str(r)) for r in personas_df["persona"]]
            personas_df["partyid"] = [p["partyid"] for p in parsed]
            personas_df["race"]    = [p["race"]    for p in parsed]
        personas_df = stratified_sample(personas_df, n=args.sample, seed=args.seed)
        if not raw_numeric:
            personas_df = personas_df.drop(columns=["partyid", "race"], errors="ignore")
        print(f"  Stratified sample: {len(personas_df)} personas")

    print("Computing latent variable priors ...")
    priors_df = build_priors_df(personas_df, raw_numeric=raw_numeric)

    print_summary(priors_df)

    priors_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(priors_df)} rows to {output_path}")
    print(f"Columns: {list(priors_df.columns)}")


if __name__ == "__main__":
    main()
