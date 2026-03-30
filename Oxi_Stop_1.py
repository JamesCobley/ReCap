# ============================================
# STEP 0 — Setup
# ============================================
!pip install pyarrow --quiet

# STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import re

# ============================================
# STEP 1 — Load Parquet file
# ============================================
parquet_path = '/content/drive/MyDrive/report (2).parquet'  # <-- change if needed
df = pd.read_parquet(parquet_path)

print("Loaded rows:", len(df))
print("Columns:", len(df.columns))


# ============================================
# STEP 2 — Parse Protein.Sites → Site.Protein.ID, Site.AA, Site.Position
# ============================================
def parse_protein_sites(val):
    """
    Parse entries like:
      "[Q5RL51:C622]" or "[Q5RL51:C622, Q9XXX1:C105]"
    into a list of dicts:
      [{'Protein.ID': 'Q5RL51', 'AA': 'C', 'Position': 622}, ...]
    """
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []

    s = s.strip('[]')
    pattern = r'([A-Z0-9]+):([A-Z])(\d{1,5})'
    matches = re.findall(pattern, s)

    sites = []
    for prot, aa, pos in matches:
        sites.append({
            'Protein.ID': prot,
            'AA': aa,
            'Position': int(pos)
        })
    return sites

df['Parsed.Protein.Sites'] = df['Protein.Sites'].apply(parse_protein_sites)

df_sites = df.explode('Parsed.Protein.Sites').reset_index(drop=True)

df_sites['Site.Protein.ID'] = df_sites['Parsed.Protein.Sites'].apply(
    lambda d: d['Protein.ID'] if isinstance(d, dict) else None
)
df_sites['Site.AA'] = df_sites['Parsed.Protein.Sites'].apply(
    lambda d: d['AA'] if isinstance(d, dict) else None
)
df_sites['Site.Position'] = df_sites['Parsed.Protein.Sites'].apply(
    lambda d: d['Position'] if isinstance(d, dict) else None
)

df_cys = df_sites[df_sites['Site.AA'] == 'C'].copy()
print("Rows with Cys sites:", len(df_cys))


# ============================================
# STEP 3 — Identify MS2 fragment quant columns & compute MS2_Total
# ============================================
ms2_cols = [c for c in df_cys.columns if c.startswith('Fr.') and c.endswith('.Quantity')]
print("MS2 quant columns:", ms2_cols)

if len(ms2_cols) == 0:
    raise ValueError("No MS2 fragment quantity columns (Fr.X.Quantity) detected.")

df_cys['MS2_Total'] = df_cys[ms2_cols].sum(axis=1)


# ============================================
# STEP 4 — Classify NEM_L vs NEM_H by Modified.Sequence
# ============================================
def classify_label(seq):
    s = str(seq)
    if 'NEM_L' in s:
        return 'NEM_L'
    elif 'NEM_H' in s:
        return 'NEM_H'
    else:
        return None

df_cys['LabelType'] = df_cys['Modified.Sequence'].apply(classify_label)

df_cys = df_cys[df_cys['LabelType'].notna()].copy()
print("Cys rows with NEM label:", len(df_cys))


# ============================================
# STEP 5 — Build SiteKey and aggregate MS2_Total per (Run, SiteKey, LabelType)
# ============================================
df_cys['SiteKey'] = (
    df_cys['Site.Protein.ID'].astype(str) + "_" +
    df_cys['Site.Position'].astype(str)
)

grouped = (
    df_cys
    .groupby(['Run', 'SiteKey', 'LabelType'])['MS2_Total']
    .sum()
    .reset_index()
)

pivot = grouped.pivot_table(
    index=['Run', 'SiteKey'],
    columns='LabelType',
    values='MS2_Total',
    fill_value=0
).reset_index()

if 'NEM_L' not in pivot.columns:
    pivot['NEM_L'] = 0
if 'NEM_H' not in pivot.columns:
    pivot['NEM_H'] = 0

pivot.columns = [c if not isinstance(c, tuple) else c[0] for c in pivot.columns]


# ============================================
# STEP 6 — Compute % oxidation per site
# %Ox = NEM_H / (NEM_H + NEM_L) * 100
# ============================================
denom = pivot['NEM_H'] + pivot['NEM_L']
pivot['Percent_Ox'] = (pivot['NEM_H'] / denom) * 100
pivot.loc[denom == 0, 'Percent_Ox'] = 0

print("Example per-site redox values:")
print(pivot.head())


# ============================================
# STEP 7 — Exclude runs containing 'Air_3' or 'Tin_6'
# ============================================
exclude_mask = (
    pivot['Run'].astype(str).str.contains('Air_3', na=False) |
    pivot['Run'].astype(str).str.contains('Tin_6', na=False)
)
pivot_inc = pivot[~exclude_mask].copy()

included_runs = sorted(pivot_inc['Run'].unique())
print("\nIncluded runs (after excluding Air_3 and Tin_6):")
for r in included_runs:
    print("  ", r)
print("Number of included runs:", len(included_runs))


# ============================================
# STEP 8 — Determine shared cysteine sites across all included runs
# ============================================
site_run_counts = (
    pivot_inc
    .groupby('SiteKey')['Run']
    .nunique()
    .reset_index()
    .rename(columns={'Run': 'Num_Runs_With_Site'})
)

n_included = len(included_runs)
shared_sitekeys = site_run_counts.loc[
    site_run_counts['Num_Runs_With_Site'] == n_included, 'SiteKey'
]

print("\nNumber of cysteine sites shared across ALL included runs:", len(shared_sitekeys))

shared_pivot = pivot_inc[pivot_inc['SiteKey'].isin(shared_sitekeys)].copy()

print("\nExample rows for shared sites:")
print(shared_pivot.head())


# ============================================
# STEP 9 — Build shared-site redox matrix: SiteKey × Run
# ============================================
shared_matrix = shared_pivot.pivot(index='SiteKey', columns='Run', values='Percent_Ox')

print("\nShared-site redox matrix (head):")
print(shared_matrix.head())


# ============================================
# STEP 10 — Rename Air → CTRL, Tin → REDCAP
# ============================================
df_shared = shared_matrix.copy()

rename_map = {}
for col in df_shared.columns:
    new_col = str(col)
    if "Air" in new_col:
        new_col = new_col.replace("Air", "CTRL")
    if "Tin" in new_col:
        new_col = new_col.replace("Tin", "REDCAP")
    rename_map[col] = new_col

df_shared = df_shared.rename(columns=rename_map)

print("\nRenamed columns:")
print(df_shared.columns.tolist())

ctrl_cols = [c for c in df_shared.columns if "CTRL" in c]
redcap_cols = [c for c in df_shared.columns if "REDCAP" in c]

print("\nCTRL samples:", ctrl_cols)
print("REDCAP samples:", redcap_cols)

if len(ctrl_cols) == 0 or len(redcap_cols) == 0:
    raise ValueError("Could not identify CTRL and/or REDCAP replicate columns after renaming.")


# ============================================
# STEP 11 — Build site-level summary table
# Compatible with downstream exact identity analysis
# ============================================
stats_df = df_shared.copy().reset_index()

# Mean values
stats_df['Mean_CTRL'] = stats_df[ctrl_cols].mean(axis=1)
stats_df['Mean_REDCAP'] = stats_df[redcap_cols].mean(axis=1)
stats_df['Delta_Percent'] = stats_df['Mean_REDCAP'] - stats_df['Mean_CTRL']
stats_df['Abs_Delta_Percent'] = stats_df['Delta_Percent'].abs()

# Optional: move key summary columns to the front
front_cols = ['SiteKey', 'Mean_CTRL', 'Mean_REDCAP', 'Delta_Percent', 'Abs_Delta_Percent']
other_cols = [c for c in stats_df.columns if c not in front_cols]
stats_df = stats_df[front_cols + other_cols]

print("\nSite-level summary table (head):")
print(stats_df.head())


# ============================================
# STEP 12 — Proteome-level summary printout
# ============================================
n_sites = len(stats_df)
mean_ctrl_all = stats_df['Mean_CTRL'].mean()
mean_redcap_all = stats_df['Mean_REDCAP'].mean()
delta_all = mean_redcap_all - mean_ctrl_all

print("\n========== Shared-site cysteine proteome summary ==========")
print(f"Number of shared cysteine sites: {n_sites}")
print(f"Mean CTRL redox state of shared cysteine proteome: {mean_ctrl_all:.6f}")
print(f"Mean REDCAP redox state of shared cysteine proteome: {mean_redcap_all:.6f}")
print(f"Delta difference (REDCAP - CTRL): {delta_all:.6f}")


# ============================================
# STEP 13 — Save outputs
# ============================================
stats_output_path = '/content/shared_cysteine_redox_summary_CTRL_vs_REDCAP.csv'
matrix_output_path = '/content/shared_cysteine_redox_matrix_CTRL_vs_REDCAP.csv'

stats_df.to_csv(stats_output_path, index=False)
df_shared.reset_index().to_csv(matrix_output_path, index=False)

print("\nSaved site-level summary table to:", stats_output_path)
print("Saved shared-site replicate matrix to:", matrix_output_path)
