#!/usr/bin/env python3
"""
04_analysis.py

Phase 5: Compare the life-chain approach against the static-demographic
baseline and real GSS 2024 responses on three metrics:

  1. First-order fit  — do item means match real GSS marginals?
  2. Second-order fit — does variance on PC2+ increase toward the GSS benchmark?
  3. Covariance structure — is the item-item correlation matrix closer to GSS?
  4. Persona types — qualitative case studies of reinforcing / cross-cutting / rupture

Outputs (written to figures/)
------------------------------
  01_first_order_scatter.png
  02_pca_variance_bars.png
  03_pca_scatter_pc2_pc3.png
  04_correlation_heatmaps.png
  05_frobenius_bar.png
  06_persona_trajectories.png
  summary_stats.csv              — machine-readable results table

Prerequisites
-------------
  pip install pyreadr scikit-learn matplotlib seaborn

Data expected
-------------
  gss2024_dellaposta_extract.rds             — real GSS responses
  gss2024_personas.csv                       — base demographic strings
  gss2024_latent_priors_200.csv              — stratified 200-persona priors
  life_chains_summary.csv                    — Phase 3 chain output
  synthetic_data_lifechains/year_2024/*.csv  — Phase 4 life-chain responses
  synthetic_data/year_2024/*.csv             — Phase 1 baseline responses (optional)

To run the baseline on the same 200 stratified personas first:
  python 04_analysis.py --export-baseline-personas
  # then in your terminal:
  python 01_generate_synthetic_GSS.py --year 2024 \\
      --personas-file gss2024_personas_200.csv \\
      --models mistralai/mistral-nemo --personas 200

Usage
-----
  python 04_analysis.py
  python 04_analysis.py --lifechain-csv synthetic_data_lifechains/year_2024/mistralai_mistral-nemo.csv
  python 04_analysis.py --baseline-csv  synthetic_data/year_2024/mistralai_mistral-nemo.csv
  python 04_analysis.py --export-baseline-personas   # exports 200-persona CSV then exits
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports — checked at runtime
try:
    import pyreadr
    PYREADR_OK = True
except ImportError:
    PYREADR_OK = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR    = Path(__file__).parent
FIGURES_DIR   = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

RDS_PATH          = SCRIPT_DIR / "gss2024_dellaposta_extract.rds"
PERSONAS_CSV      = SCRIPT_DIR / "gss2024_personas.csv"
PRIORS_200_CSV    = SCRIPT_DIR / "gss2024_latent_priors_200.csv"
CHAINS_SUMMARY    = SCRIPT_DIR / "life_chains_summary.csv"
CHAINS_DIR        = SCRIPT_DIR / "life_chains"
LIFECHAIN_DIR     = SCRIPT_DIR / "synthetic_data_lifechains" / "year_2024"
BASELINE_DIR      = SCRIPT_DIR / "synthetic_data" / "year_2024"

LATENT_VARS = [
    "moral_traditionalism",
    "economic_redistribution",
    "racial_attitudes",
    "institutional_trust",
    "outgroup_affect",
]

# Items missing from the real GSS 2024 extract (economic well-being module)
MISSING_FROM_GSS = {"fair", "finalter", "getahead", "richwork", "satfin", "satjob"}

# Aesthetic palette
PALETTE = {"real_gss": "#2c7bb6", "lifechain": "#d7191c", "baseline": "#1a9641"}
LABEL   = {"real_gss": "Real GSS 2024", "lifechain": "Life-Chain", "baseline": "Baseline"}

sns.set_theme(style="whitegrid", font_scale=1.1)

# ---------------------------------------------------------------------------
# Utility: identify GSS item option range (for direction normalisation)
# ---------------------------------------------------------------------------

# For each GSS variable: the minimum valid option code.
# Items where the lowest code = the "liberal" or "pro-redistribution" end
# (so that higher code = more conservative) — used for direction consistency.
ITEM_MIN = {}  # populated dynamically from real data

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_real_gss(items: list[str]) -> pd.DataFrame:
    """Load real GSS 2024, return wide DataFrame indexed by respondent row."""
    if not PYREADR_OK:
        raise ImportError("pyreadr not installed. Run: pip install pyreadr")
    result = pyreadr.read_r(str(RDS_PATH))
    gss = list(result.values())[0]
    available = [v for v in items if v in gss.columns]
    df = gss[available].copy()
    # Coerce to numeric (some columns come in as float from R)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_synthetic_wide(csv_path: Path, items: list[str]) -> pd.DataFrame:
    """
    Load a synthetic response CSV, pivot to wide (persona_id × variable).
    Multi-run averages are collapsed to the mean.
    """
    df = pd.read_csv(csv_path)
    df = df[df["answer"].notna()].copy()
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
    wide = df.pivot_table(
        index="persona_id", columns="variable", values="answer", aggfunc="mean"
    )
    available = [v for v in items if v in wide.columns]
    return wide[available].copy()


def load_chains(chains_dir: Path) -> dict:
    """Load all per-persona chain JSONs. Returns {respondent_id: chain_dict}."""
    chains = {}
    for p in sorted(chains_dir.glob("*.json")):
        try:
            with open(p) as f:
                ch = json.load(f)
            chains[ch["respondent_id"]] = ch
        except Exception:
            pass
    return chains


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------

def get_common_items(*dataframes) -> list[str]:
    """Return items present in all supplied DataFrames (excl. MISSING_FROM_GSS)."""
    cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        cols &= set(df.columns)
    cols -= MISSING_FROM_GSS
    return sorted(cols)


def z_score_by_real(
    real_df: pd.DataFrame,
    *other_dfs,
    items: list[str],
) -> tuple:
    """
    Fit a StandardScaler on real_df[items], apply to all DataFrames.
    Returns (scaler, real_z, other_z1, other_z2, ...).
    """
    real_sub = real_df[items].copy()
    # Fill NaN per-column with column mean so listwise deletion doesn't discard
    # the many GSS respondents who skipped individual items.
    real_sub = real_sub.fillna(real_sub.mean())
    scaler = StandardScaler()
    real_z = pd.DataFrame(
        scaler.fit_transform(real_sub),
        columns=items,
        index=real_sub.index,
    )
    others_z = []
    for df in other_dfs:
        sub = df[items].fillna(df[items].mean())
        z = pd.DataFrame(scaler.transform(sub), columns=items, index=df.index)
        others_z.append(z)
    return (scaler, real_z, *others_z)


# ---------------------------------------------------------------------------
# 3. First-order fit
# ---------------------------------------------------------------------------

def first_order_analysis(
    real_df: pd.DataFrame,
    lifechain_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    items: list[str],
) -> pd.DataFrame:
    """
    Compute mean per item for each condition.
    Returns a DataFrame with columns: item, real_mean, lifechain_mean, [baseline_mean].
    """
    rows = []
    for item in items:
        row = {"item": item, "real_mean": real_df[item].mean()}
        row["lifechain_mean"] = lifechain_df[item].mean() if item in lifechain_df.columns else np.nan
        if baseline_df is not None:
            row["baseline_mean"] = baseline_df[item].mean() if item in baseline_df.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("item")


def plot_first_order(fo: pd.DataFrame, has_baseline: bool) -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        fo["real_mean"], fo["lifechain_mean"],
        color=PALETTE["lifechain"], alpha=0.75, s=55, zorder=3,
        label=f'{LABEL["lifechain"]} (r={fo["real_mean"].corr(fo["lifechain_mean"]):.3f})'
    )
    if has_baseline and "baseline_mean" in fo.columns:
        ax.scatter(
            fo["real_mean"], fo["baseline_mean"],
            color=PALETTE["baseline"], alpha=0.55, s=35, marker="^", zorder=2,
            label=f'{LABEL["baseline"]} (r={fo["real_mean"].corr(fo["baseline_mean"]):.3f})'
        )

    lo = min(fo["real_mean"].min(), fo["lifechain_mean"].min()) - 0.1
    hi = max(fo["real_mean"].max(), fo["lifechain_mean"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="Perfect fit")

    ax.set_xlabel("Real GSS 2024 — item mean")
    ax.set_ylabel("Simulated — item mean")
    ax.set_title("First-Order Fit: Item Marginals")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    out = FIGURES_DIR / "01_first_order_scatter.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4. Second-order fit: PCA
# ---------------------------------------------------------------------------

def run_pca(real_z: pd.DataFrame, lifechain_z: pd.DataFrame,
            baseline_z: pd.DataFrame | None, n_pcs: int = 6):
    """
    Fit PCA on real_z. Project all conditions onto the same PC axes.
    Returns: pca object, dict of {condition: projected_scores}, variance_ratio
    """
    pca = PCA(n_components=min(n_pcs, real_z.shape[1]))
    real_scores = pca.fit_transform(real_z.values)

    scores = {"real_gss": real_scores}
    scores["lifechain"] = pca.transform(lifechain_z.values)
    if baseline_z is not None:
        scores["baseline"] = pca.transform(baseline_z.values)

    return pca, scores


def pc_variance_in_condition(scores: np.ndarray, n_pcs: int = 6) -> np.ndarray:
    """Variance of projected scores on each PC axis (not explained ratio — actual spread)."""
    return np.var(scores[:, :n_pcs], axis=0)


def plot_pca_variance(pca, scores: dict, has_baseline: bool) -> Path:
    n_pcs = pca.n_components_
    labels = [f"PC{i+1}" for i in range(n_pcs)]
    x = np.arange(n_pcs)
    width = 0.25 if has_baseline else 0.35

    conditions = ["real_gss", "lifechain"]
    if has_baseline and "baseline" in scores:
        conditions.append("baseline")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: variance explained by the PCA itself (real GSS benchmark)
    ax = axes[0]
    ax.bar(x, pca.explained_variance_ratio_ * 100,
           color=PALETTE["real_gss"], alpha=0.8, label=LABEL["real_gss"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% variance explained")
    ax.set_title("PCA: Real GSS Scree Plot")
    ax.legend()

    # Right: actual score variance per PC per condition
    ax = axes[1]
    offsets = np.linspace(-width, width, len(conditions))
    for i, (cond, offset) in enumerate(zip(conditions, offsets)):
        var = pc_variance_in_condition(scores[cond], n_pcs)
        ax.bar(x + offset, var, width=width * 0.9,
               color=PALETTE[cond], alpha=0.8, label=LABEL[cond])

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score variance on PC axis")
    ax.set_title("Second-Order Fit: PC Score Variance\n(life-chain should match GSS on PC2+)")
    ax.legend()

    out = FIGURES_DIR / "02_pca_variance_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pca_scatter(scores: dict, pca, has_baseline: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    pv = pca.explained_variance_ratio_
    for cond, sc in scores.items():
        alpha = 0.25 if cond == "real_gss" else 0.55
        size  = 12  if cond == "real_gss" else 30
        ax.scatter(sc[:, 1], sc[:, 2], alpha=alpha, s=size,
                   color=PALETTE[cond], label=LABEL[cond])

    ax.set_xlabel(f"PC2 ({pv[1]*100:.1f}% of real GSS variance)")
    ax.set_ylabel(f"PC3 ({pv[2]*100:.1f}% of real GSS variance)")
    ax.set_title("Persona Distribution in PC Space\n(more spread = more within-cell variance)")
    ax.legend()

    out = FIGURES_DIR / "03_pca_scatter_pc2_pc3.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5. Covariance structure
# ---------------------------------------------------------------------------

def correlation_matrix(z_df: pd.DataFrame) -> pd.DataFrame:
    return z_df.corr(method="pearson")


def frobenius_distance(A: pd.DataFrame, B: pd.DataFrame) -> float:
    common = A.index.intersection(B.index)
    a = A.loc[common, common].values
    b = B.loc[common, common].values
    mask = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.sum((a[mask] - b[mask]) ** 2)))


def plot_correlation_heatmaps(
    real_corr: pd.DataFrame,
    lifechain_corr: pd.DataFrame,
    baseline_corr: pd.DataFrame | None,
    frob_lc: float,
    frob_bl: float | None,
) -> Path:
    n_panels = 3 if (baseline_corr is not None) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 2:
        axes = list(axes)

    items = real_corr.columns.tolist()
    kw = dict(vmin=-1, vmax=1, cmap="RdBu_r", square=True,
              xticklabels=False, yticklabels=False, cbar=False)

    sns.heatmap(real_corr, ax=axes[0], **kw)
    axes[0].set_title(f"{LABEL['real_gss']}\n({len(items)} items)")

    sns.heatmap(lifechain_corr.reindex(index=items, columns=items),
                ax=axes[1], **kw)
    axes[1].set_title(f"{LABEL['lifechain']}\nFrobenius dist = {frob_lc:.3f}")

    if baseline_corr is not None and frob_bl is not None:
        sns.heatmap(baseline_corr.reindex(index=items, columns=items),
                    ax=axes[2], **kw)
        axes[2].set_title(f"{LABEL['baseline']}\nFrobenius dist = {frob_bl:.3f}")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Pearson r")

    out = FIGURES_DIR / "04_correlation_heatmaps.png"
    fig.suptitle("Item-Item Correlation Structure", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_frobenius_bar(frob_lc: float, frob_bl: float | None) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    conditions = [LABEL["lifechain"]]
    values     = [frob_lc]
    colors     = [PALETTE["lifechain"]]
    if frob_bl is not None:
        conditions.append(LABEL["baseline"])
        values.append(frob_bl)
        colors.append(PALETTE["baseline"])

    bars = ax.bar(conditions, values, color=colors, width=0.4, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Frobenius distance from real GSS correlation matrix")
    ax.set_title("Covariance Structure Fit\n(lower = closer to real GSS)")
    ax.set_ylim(0, max(values) * 1.25)

    out = FIGURES_DIR / "05_frobenius_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6. Persona-type classification & trajectory plots
# ---------------------------------------------------------------------------

def classify_persona_type(chain: dict) -> str:
    """
    Classify one persona's chain as reinforcing, cross_cutting, or rupture.

    Reinforcing : most non-zero deltas push in a consistent direction
                  (|mean_delta| / mean(|delta|) > 0.55)
    Rupture     : one event accounts for > 40% of total absolute movement
    Cross-cutting: everything else (deltas cancel each other out)
    """
    all_deltas = []
    for var in LATENT_VARS:
        all_deltas.extend(chain["latent_trajectories"].get(var, []))

    nonzero = [d for d in all_deltas if abs(d) > 1e-6]
    if not nonzero:
        return "unclassified"

    abs_deltas = [abs(d) for d in nonzero]
    total_movement = sum(abs_deltas)

    # Rupture: single event dominates
    max_single = max(abs_deltas)
    if total_movement > 0 and max_single / total_movement > 0.40:
        return "rupture"

    # Reinforcing vs cross-cutting: alignment of directions
    mean_abs = np.mean(abs_deltas)
    mean_signed = abs(np.mean(nonzero))
    alignment = mean_signed / mean_abs if mean_abs > 0 else 0

    return "reinforcing" if alignment > 0.55 else "cross_cutting"


def plot_persona_trajectories(chains: dict, lifechain_wide: pd.DataFrame) -> Path:
    """
    Plot latent variable trajectories for one persona of each type.
    Shows: trajectory of latent means over life events, plus their
    survey response distribution on the most relevant items.
    """
    # Classify all loaded chains
    typed: dict[str, list] = {"reinforcing": [], "cross_cutting": [], "rupture": []}
    for rid, chain in chains.items():
        ptype = classify_persona_type(chain)
        if ptype in typed:
            typed[ptype].append(rid)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    type_labels = {
        "reinforcing":   "Reinforcing Chain\n(ideologically rigid)",
        "cross_cutting": "Cross-Cutting Chain\n(genuinely ambivalent)",
        "rupture":       "Rupture-Event Chain\n(dominant single event)",
    }
    type_colors = {
        "reinforcing": "#2c7bb6", "cross_cutting": "#d7191c", "rupture": "#fdae61"
    }

    for col_idx, ptype in enumerate(["reinforcing", "cross_cutting", "rupture"]):
        ids = typed[ptype]
        if not ids:
            continue
        rid = ids[0]
        chain = chains[rid]

        # ---- Top row: latent variable trajectories ----
        ax_top = fig.add_subplot(gs[0, col_idx])
        n_steps = chain["n_events"]
        for var in LATENT_VARS:
            traj = chain["latent_trajectories"].get(var, [])
            init = chain["initial_latent_state"].get(var, 0.5)
            # Reconstruct running means from deltas
            running = [init]
            for delta in traj:
                running.append(np.clip(running[-1] + delta, 0.03, 0.97))
            ax_top.plot(range(len(running)), running,
                        marker="o", markersize=3, lw=1.5,
                        label=var.replace("_", " ")[:12])

        ax_top.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax_top.set_ylim(0, 1)
        ax_top.set_xlabel("Life event step")
        ax_top.set_ylabel("Latent variable mean")
        ax_top.set_title(type_labels[ptype], fontsize=10,
                         color=type_colors[ptype], fontweight="bold")
        if col_idx == 0:
            ax_top.legend(fontsize=6, loc="lower left")

        # ---- Bottom row: survey response distribution for this persona ----
        ax_bot = fig.add_subplot(gs[1, col_idx])
        # For the 5-persona pilot, show actual answers; for larger runs, could show variance
        if rid in lifechain_wide.index:
            persona_answers = lifechain_wide.loc[rid].dropna()
            # Normalise answers to [0, 1] by dividing by item max (crude but visual)
            normed = persona_answers / persona_answers.index.map(
                lambda v: lifechain_wide[v].max() if v in lifechain_wide.columns else 1
            )
            ax_bot.hist(normed.values, bins=8, color=type_colors[ptype],
                        alpha=0.7, edgecolor="white")
            ax_bot.set_xlabel("Normalised response (0=low, 1=high)")
            ax_bot.set_ylabel("Item count")
            ax_bot.set_title(f"Survey responses\npersona {rid}")
        else:
            ax_bot.text(0.5, 0.5, f"Persona {rid}\n(no survey data)",
                        ha="center", va="center", transform=ax_bot.transAxes)

    out = FIGURES_DIR / "06_persona_trajectories.png"
    fig.suptitle("Persona Types from Life-Chain Simulation", fontsize=13, y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 7. 200-persona stratified setup helper
# ---------------------------------------------------------------------------

def export_baseline_personas(priors_200_path: Path, personas_path: Path) -> Path:
    """
    Create gss2024_personas_200.csv containing only the 200 stratified personas
    so that 01_generate_synthetic_GSS.py can be pointed at the same sample.
    """
    priors = pd.read_csv(priors_200_path)
    ids    = priors["respondent_id"].tolist()
    full   = pd.read_csv(personas_path)
    subset = full[full["respondent_id"].isin(ids)]
    out    = SCRIPT_DIR / "gss2024_personas_200.csv"
    subset.to_csv(out, index=False)
    print(f"Saved {len(subset)} personas to {out}")
    return out


# ---------------------------------------------------------------------------
# 8. Summary table
# ---------------------------------------------------------------------------

def build_summary(
    fo: pd.DataFrame,
    pca,
    scores: dict,
    frob_lc: float,
    frob_bl: float | None,
    persona_type_counts: dict,
    has_baseline: bool,
) -> pd.DataFrame:
    pv = pca.explained_variance_ratio_

    lc_var  = pc_variance_in_condition(scores["lifechain"])
    gss_var = pc_variance_in_condition(scores["real_gss"])

    rows = []

    # First-order: MAE between simulated and real means
    lc_mae = (fo["lifechain_mean"] - fo["real_mean"]).abs().mean()
    rows.append({"metric": "First-order MAE (lifechain vs real)", "value": round(lc_mae, 4)})
    if has_baseline and "baseline_mean" in fo.columns:
        bl_mae = (fo["baseline_mean"] - fo["real_mean"]).abs().mean()
        rows.append({"metric": "First-order MAE (baseline vs real)", "value": round(bl_mae, 4)})

    # Correlation of item means
    lc_r = fo["real_mean"].corr(fo["lifechain_mean"])
    rows.append({"metric": "First-order r (lifechain vs real)", "value": round(lc_r, 4)})

    # Second-order: variance ratio on PC2+ (lifechain / real)
    for i in range(min(5, len(lc_var))):
        rows.append({
            "metric": f"PC{i+1} score variance — lifechain",
            "value":  round(float(lc_var[i]), 4)
        })
        rows.append({
            "metric": f"PC{i+1} score variance — real_gss",
            "value":  round(float(gss_var[i]), 4)
        })
        rows.append({
            "metric": f"PC{i+1} variance ratio (lc/gss)",
            "value":  round(float(lc_var[i] / gss_var[i]) if gss_var[i] > 0 else np.nan, 4)
        })

    # Covariance
    rows.append({"metric": "Frobenius distance — lifechain", "value": round(frob_lc, 4)})
    if frob_bl is not None:
        rows.append({"metric": "Frobenius distance — baseline", "value": round(frob_bl, 4)})

    # Persona types
    for ptype, n in persona_type_counts.items():
        rows.append({"metric": f"Persona type: {ptype}", "value": n})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Constraint statistics: effective dependence & PC1 variance share
# ---------------------------------------------------------------------------

def load_educ_series(rds_path: Path) -> tuple[pd.Series, pd.Series]:
    """
    Load educ from the GSS RDS.
    Returns (educ_0based, educ_1based):
      educ_0based — index 0..N-1, aligned with real GSS DataFrame default index
      educ_1based — index 1..N, aligned with synthetic persona_id (= R respondent_id)
    """
    if not PYREADR_OK or not rds_path.exists():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    result = pyreadr.read_r(str(rds_path))
    gss = list(result.values())[0]
    if "educ" not in gss.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    educ = pd.to_numeric(gss["educ"], errors="coerce").reset_index(drop=True)
    educ_0 = educ.copy()
    educ_1 = educ.copy()
    educ_1.index = range(1, len(educ_1) + 1)
    return educ_0, educ_1


def split_wide_by_educ(
    wide_df: pd.DataFrame,
    educ_col: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Split wide_df into Overall / Low edu (educ≤12) / High edu (educ≥16) subsets."""
    groups: dict[str, pd.DataFrame] = {"Overall": wide_df}
    aligned = educ_col.reindex(wide_df.index).dropna()
    if aligned.empty:
        return groups
    low_ids  = aligned[aligned <= 12].index
    high_ids = aligned[aligned >= 16].index
    if len(low_ids) >= 5:
        groups["Low edu"]  = wide_df.loc[low_ids]
    if len(high_ids) >= 5:
        groups["High edu"] = wide_df.loc[high_ids]
    return groups


def _corr_eigvals(wide_df: pd.DataFrame, items: list[str]) -> np.ndarray | None:
    sub = wide_df[items].dropna()
    if len(sub) < 10:
        return None
    corr = sub.corr(method="pearson").values
    eigvals = np.linalg.eigvalsh(corr)  # ascending order
    return np.maximum(eigvals, 1e-10)


def compute_effective_dependence(wide_df: pd.DataFrame, items: list[str]) -> float:
    """De = 1 − geometric mean of eigenvalues of Pearson correlation matrix (Peña & Rodríguez 2003)."""
    eigvals = _corr_eigvals(wide_df, items)
    if eigvals is None:
        return np.nan
    p = len(eigvals)
    geo_mean = np.exp(np.sum(np.log(eigvals)) / p)
    return float(1.0 - geo_mean)


def compute_pc1_share(wide_df: pd.DataFrame, items: list[str]) -> float:
    """PC1 variance proportion: largest eigenvalue / sum of all eigenvalues."""
    eigvals = _corr_eigvals(wide_df, items)
    if eigvals is None or eigvals.sum() == 0:
        return np.nan
    return float(eigvals[-1] / eigvals.sum())


def bootstrap_constraint_stats(
    wide_df: pd.DataFrame,
    items: list[str],
    n_boots: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Cluster bootstrap: resample persona rows with replacement, compute De and PC1 each draw."""
    rng = np.random.default_rng(seed)
    ids = np.array(wide_df.index.tolist())
    n = len(ids)
    de_list: list[float] = []
    pc1_list: list[float] = []
    for _ in range(n_boots):
        boot_ids = rng.choice(ids, size=n, replace=True)
        boot = wide_df.loc[boot_ids]
        de_list.append(compute_effective_dependence(boot, items))
        pc1_list.append(compute_pc1_share(boot, items))
    return {"de": de_list, "pc1": pc1_list}


def plot_constraint_ridge(
    boot_results: dict,
    metric: str,
    out_path: Path,
    title: str,
    xlabel: str,
) -> Path:
    """
    Ridge/density plot comparing bootstrap distributions across models and education groups.

    boot_results: {model_label: {edu_group: {'de': [...], 'pc1': [...]}}}
    metric: 'de' or 'pc1'
    """
    edu_order  = ["Overall", "Low edu", "High edu"]
    edu_colors = {"Overall": "#555555", "Low edu": "#e08214", "High edu": "#4393c3"}

    model_labels = list(boot_results.keys())
    n_models = len(model_labels)

    all_vals = [
        v
        for md in boot_results.values()
        for gd in md.values()
        for v in gd[metric]
        if np.isfinite(v)
    ]
    if not all_vals:
        return out_path

    x_lo = max(0.0, np.percentile(all_vals, 0.5)  - 0.01)
    x_hi = min(1.0, np.percentile(all_vals, 99.5) + 0.01)
    x_grid = np.linspace(x_lo, x_hi, 500)

    fig, axes = plt.subplots(
        n_models, 1,
        figsize=(8, 1.8 * n_models + 1.2),
        sharex=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, label in zip(axes, model_labels):
        model_data = boot_results[label]
        for grp in edu_order:
            if grp not in model_data:
                continue
            vals = [v for v in model_data[grp][metric] if np.isfinite(v)]
            if len(vals) < 10:
                continue
            kde = gaussian_kde(vals, bw_method="scott")
            density = kde(x_grid)
            color = edu_colors[grp]
            ax.fill_between(x_grid, density, alpha=0.20, color=color)
            ax.plot(x_grid, density, lw=1.8, color=color, label=grp)

        ax.set_ylabel(label, fontsize=8.5, rotation=0, ha="right",
                      va="center", labelpad=4)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[-1].set_xlabel(xlabel, fontsize=10)
    axes[0].set_title(title, fontsize=11)

    handles = [
        plt.Line2D([0], [0], color=edu_colors[g], lw=2, label=g)
        for g in edu_order
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5 analysis")
    parser.add_argument(
        "--lifechain-csv", default=None,
        help="Path to life-chain survey CSV (default: auto-detect in synthetic_data_lifechains/year_2024/)"
    )
    parser.add_argument(
        "--baseline-csv", default=None,
        help="Path to baseline survey CSV (optional; enables baseline comparison)"
    )
    parser.add_argument(
        "--export-baseline-personas", action="store_true",
        help="Write gss2024_personas_200.csv for baseline run then exit"
    )
    parser.add_argument(
        "--n-boots", type=int, default=500,
        help="Bootstrap iterations for constraint statistics (default: 500)"
    )
    args = parser.parse_args()

    # ---- Setup helper ----
    if args.export_baseline_personas:
        if not PRIORS_200_CSV.exists():
            print("ERROR: gss2024_latent_priors_200.csv not found. Run:")
            print("  python initialize_latent_vars.py --sample 200 --seed 42 --output gss2024_latent_priors_200.csv")
            sys.exit(1)
        out = export_baseline_personas(PRIORS_200_CSV, PERSONAS_CSV)
        print("\nNow run the baseline on these 200 personas:")
        print(f"  python 01_generate_synthetic_GSS.py --year 2024 \\")
        print(f"    --models mistralai/mistral-nemo --personas 200")
        print(f"\nNOTE: 01_generate_synthetic_GSS.py loads personas from its default path.")
        print(f"Temporarily replace gss2024_personas.csv with {out.name}, or")
        print(f"edit the script's personas_file path to point to {out.name}.")
        sys.exit(0)

    # ---- Dependency checks ----
    if not PYREADR_OK:
        print("ERROR: pyreadr not installed. Run: pip install pyreadr"); sys.exit(1)
    if not SKLEARN_OK:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn"); sys.exit(1)

    # ---- Find synthetic CSVs ----
    if args.lifechain_csv:
        lc_path = Path(args.lifechain_csv)
    else:
        lc_csvs = sorted(LIFECHAIN_DIR.glob("*.csv")) if LIFECHAIN_DIR.exists() else []
        if not lc_csvs:
            print(f"ERROR: No life-chain CSV found in {LIFECHAIN_DIR}")
            print("Run: python 03_generate_synthetic_GSS_lifechains.py --year 2024 --models <model>")
            sys.exit(1)
        lc_path = lc_csvs[0]
        if len(lc_csvs) > 1:
            print(f"Multiple life-chain CSVs found; using {lc_path.name}. "
                  f"Pass --lifechain-csv to specify.")

    bl_path = None
    if args.baseline_csv:
        bl_path = Path(args.baseline_csv)
    else:
        bl_csvs = sorted(BASELINE_DIR.glob("*.csv")) if BASELINE_DIR.exists() else []
        if bl_csvs:
            bl_path = bl_csvs[0]
            if len(bl_csvs) > 1:
                print(f"Multiple baseline CSVs found; using {bl_path.name}.")

    has_baseline = bl_path is not None and bl_path.exists()

    print(f"\nLife-chain CSV : {lc_path}")
    print(f"Baseline CSV   : {bl_path if has_baseline else '(not found — baseline comparison skipped)'}")
    print()

    # ---- Load data ----
    print("Loading real GSS 2024 ...")
    # We'll determine the full item list from the synthetic data, then intersect with GSS
    lc_raw = pd.read_csv(lc_path)
    lc_items = sorted(lc_raw[lc_raw["answer"].notna()]["variable"].unique())

    gss_real = load_real_gss(lc_items)
    print(f"  Real GSS: {len(gss_real)} respondents × {len(gss_real.columns)} items")

    print("Loading life-chain synthetic responses ...")
    lc_wide = load_synthetic_wide(lc_path, lc_items)
    print(f"  Life-chain: {len(lc_wide)} personas × {len(lc_wide.columns)} items")

    bl_wide = None
    if has_baseline:
        print("Loading baseline synthetic responses ...")
        bl_wide = load_synthetic_wide(bl_path, lc_items)
        print(f"  Baseline: {len(bl_wide)} personas × {len(bl_wide.columns)} items")

    # ---- Common items ----
    dfs_to_intersect = [gss_real, lc_wide] + ([bl_wide] if bl_wide is not None else [])
    common_items = get_common_items(*dfs_to_intersect)
    print(f"\nCommon items for analysis: {len(common_items)}")

    # ---- Load chain JSONs for persona-type analysis ----
    print("Loading chain JSONs ...")
    chains = load_chains(CHAINS_DIR)
    print(f"  {len(chains)} chain files found")

    # Classify persona types
    type_counts: dict[str, int] = {}
    for rid, chain in chains.items():
        ptype = classify_persona_type(chain)
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    print(f"  Persona types: {type_counts}")

    # ---- Normalize ----
    print("\nNormalizing (z-score by real GSS distribution) ...")
    args_z = [gss_real, lc_wide] + ([bl_wide] if bl_wide is not None else [])
    z_results = z_score_by_real(gss_real, lc_wide,
                                *([bl_wide] if bl_wide is not None else []),
                                items=common_items)
    scaler_obj, real_z, lc_z = z_results[0], z_results[1], z_results[2]
    bl_z = z_results[3] if has_baseline else None

    # ---- Analysis ----
    print("Running first-order analysis ...")
    fo = first_order_analysis(
        gss_real[common_items], lc_wide[common_items],
        bl_wide[common_items] if bl_wide is not None else None,
        common_items,
    )
    lc_r   = fo["real_mean"].corr(fo["lifechain_mean"])
    lc_mae = (fo["lifechain_mean"] - fo["real_mean"]).abs().mean()
    print(f"  Life-chain first-order r={lc_r:.3f}, MAE={lc_mae:.4f}")
    if has_baseline and "baseline_mean" in fo.columns:
        bl_r   = fo["real_mean"].corr(fo["baseline_mean"])
        bl_mae = (fo["baseline_mean"] - fo["real_mean"]).abs().mean()
        print(f"  Baseline   first-order r={bl_r:.3f}, MAE={bl_mae:.4f}")

    print("Running PCA ...")
    pca, scores = run_pca(real_z, lc_z, bl_z)
    for i in range(min(4, pca.n_components_)):
        gss_v = np.var(scores["real_gss"][:, i])
        lc_v  = np.var(scores["lifechain"][:, i])
        ratio = lc_v / gss_v if gss_v > 0 else np.nan
        print(f"  PC{i+1}: GSS var={gss_v:.3f}  LC var={lc_v:.3f}  ratio={ratio:.3f}")

    print("Computing covariance structure ...")
    real_corr = correlation_matrix(real_z)
    lc_corr   = correlation_matrix(lc_z)
    bl_corr   = correlation_matrix(bl_z) if bl_z is not None else None

    frob_lc = frobenius_distance(real_corr, lc_corr)
    frob_bl = frobenius_distance(real_corr, bl_corr) if bl_corr is not None else None
    print(f"  Frobenius distance — life-chain: {frob_lc:.4f}")
    if frob_bl is not None:
        print(f"  Frobenius distance — baseline:   {frob_bl:.4f}")

    # ---- Constraint statistics (bootstrap) ----
    print("\nLoading educ for education-group splits ...")
    educ_0, educ_1 = load_educ_series(RDS_PATH)

    print(f"Running constraint statistics ({args.n_boots} bootstrap iterations) ...")
    constraint_models: dict[str, tuple[pd.DataFrame, pd.Series]] = {
        "Real GSS 2024": (gss_real[common_items], educ_0),
        "Life-chain":    (lc_wide[common_items],  educ_1),
    }
    if bl_wide is not None:
        constraint_models["Baseline"] = (bl_wide[common_items], educ_1)

    all_boots: dict[str, dict[str, dict]] = {}
    for model_label, (wide_df, educ_col) in constraint_models.items():
        print(f"  Bootstrapping {model_label} ...")
        groups = split_wide_by_educ(wide_df, educ_col)
        all_boots[model_label] = {}
        for grp_name, grp_df in groups.items():
            grp_items = [it for it in common_items if it in grp_df.columns]
            all_boots[model_label][grp_name] = bootstrap_constraint_stats(
                grp_df, grp_items, args.n_boots
            )

    # ---- Figures ----
    print("\nGenerating figures ...")

    p = plot_first_order(fo, has_baseline)
    print(f"  {p.name}")

    p = plot_pca_variance(pca, scores, has_baseline)
    print(f"  {p.name}")

    if pca.n_components_ >= 3:
        p = plot_pca_scatter(scores, pca, has_baseline)
        print(f"  {p.name}")

    p = plot_correlation_heatmaps(real_corr, lc_corr, bl_corr, frob_lc, frob_bl)
    print(f"  {p.name}")

    p = plot_frobenius_bar(frob_lc, frob_bl)
    print(f"  {p.name}")

    if chains:
        p = plot_persona_trajectories(chains, lc_wide)
        print(f"  {p.name}")

    p = plot_constraint_ridge(
        all_boots, "de",
        FIGURES_DIR / "07_effective_dependence_ridge.png",
        "Effective Dependence",
        "De  =  1 − geometric mean of eigenvalues  (higher → more constrained)",
    )
    print(f"  {p.name}")

    p = plot_constraint_ridge(
        all_boots, "pc1",
        FIGURES_DIR / "08_pc1_variance_ridge.png",
        "PC1 Variance Share",
        "λ₁ / Σλⱼ  (higher → more variance concentrated in first principal component)",
    )
    print(f"  {p.name}")

    # ---- Summary table ----
    summary = build_summary(fo, pca, scores, frob_lc, frob_bl, type_counts, has_baseline)
    summary_path = SCRIPT_DIR / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table: {summary_path}")

    print("\n" + "=" * 60)
    print("Analysis complete. Key results:")
    print(f"  First-order r (life-chain vs real GSS): {lc_r:.3f}")
    print(f"  Frobenius distance (life-chain):        {frob_lc:.4f}")
    if frob_bl:
        improvement = (frob_bl - frob_lc) / frob_bl * 100
        print(f"  Frobenius distance (baseline):          {frob_bl:.4f}")
        print(f"  Improvement in covariance fit:          {improvement:.1f}%")
    print(f"\nFigures saved to: {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
