import pandas as pd
import numpy as np
import re
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# ============================================
# STEP 1 — Load Parquet file
# ============================================
parquet_path = '/content/drive/MyDrive/report (2).parquet'  # <-- change if needed
df = pd.read_parquet(parquet_path)

print("Loaded rows:", len(df))
print("Columns:", len(df.columns))
print("\nFirst 40 columns:")
print(df.columns[:40].tolist())


# ============================================
# STEP 2 — Helpers
# ============================================
def choose_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def first_accession(val):
    """
    Return the first accession only from a semicolon-separated protein group entry.
    Example:
      'P12345;Q99999' -> 'P12345'
      '[P12345;Q99999]' -> 'P12345'
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().strip('[]')
    if not s:
        return np.nan
    first = s.split(';')[0].strip()
    return first if first else np.nan

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

def assign_group(run_name):
    s = str(run_name)
    if 'Air' in s:
        return 'CTRL'
    elif 'Tin' in s:
        return 'REDCAP'
    else:
        return 'Other'


# ============================================
# STEP 3 — Parse Protein.Sites → Site.Protein.ID, Site.AA, Site.Position
# ============================================
if 'Protein.Sites' not in df.columns:
    raise ValueError("Column 'Protein.Sites' not found.")

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
print("\nRows with Cys sites:", len(df_cys))


# ============================================
# STEP 4 — Identify MS2 fragment quant columns & compute MS2_Total
# ============================================
ms2_cols = [c for c in df_cys.columns if c.startswith('Fr.') and c.endswith('.Quantity')]
print("MS2 quant columns:", ms2_cols)

if len(ms2_cols) == 0:
    raise ValueError("No MS2 fragment quantity columns (Fr.X.Quantity) detected.")

df_cys['MS2_Total'] = df_cys[ms2_cols].sum(axis=1)


# ============================================
# STEP 5 — Classify NEM_L vs NEM_H by Modified.Sequence
# ============================================
if 'Modified.Sequence' not in df_cys.columns:
    raise ValueError("Column 'Modified.Sequence' not found.")

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
# STEP 6 — Build SiteKey and aggregate MS2_Total per (Run, SiteKey, LabelType)
# ============================================
if 'Run' not in df_cys.columns:
    raise ValueError("Column 'Run' not found.")

df_cys['SiteKey'] = df_cys['Site.Protein.ID'].astype(str) + "_" + df_cys['Site.Position'].astype(str)

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
# STEP 7 — Compute % oxidation per site
# ============================================
denom = pivot['NEM_H'] + pivot['NEM_L']
pivot['Percent_Ox'] = (pivot['NEM_H'] / denom) * 100
pivot.loc[denom == 0, 'Percent_Ox'] = 0

print("\nExample per-site redox values:")
print(pivot.head())


# ============================================
# STEP 8 — Exclude runs containing 'Air_3' or 'Tin_6'
# ============================================
exclude_mask = pivot['Run'].str.contains('Air_3', na=False) | pivot['Run'].str.contains('Tin_6', na=False)
pivot_inc = pivot[~exclude_mask].copy()

included_runs = sorted(pivot_inc['Run'].dropna().unique())
print("\nIncluded runs (after excluding Air_3 and Tin_6):")
for r in included_runs:
    print("  ", r)
print("Number of included runs:", len(included_runs))


# ============================================
# STEP 9 — Determine shared cysteine sites across all included runs
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


# ============================================
# STEP 10 — Build shared-site redox matrix: SiteKey × Run
# ============================================
shared_matrix = (
    shared_pivot
    .pivot(index='SiteKey', columns='Run', values='Percent_Ox')
)

print("\nShared-site redox matrix shape:", shared_matrix.shape)


# ============================================
# STEP 11 — Rename Air → CTRL, Tin → REDCAP
# ============================================
df_shared = shared_matrix.copy()

rename_map = {}
for col in df_shared.columns:
    new_col = col
    if "Air" in new_col:
        new_col = new_col.replace("Air", "CTRL")
    if "Tin" in new_col:
        new_col = new_col.replace("Tin", "REDCAP")
    rename_map[col] = new_col

df_shared = df_shared.rename(columns=rename_map)

ctrl_cols = [c for c in df_shared.columns if "CTRL" in c]
redcap_cols = [c for c in df_shared.columns if "REDCAP" in c]

print("\nCTRL samples:", ctrl_cols)
print("REDCAP samples:", redcap_cols)


# ============================================
# STEP 12 — Site-wise stats: log2FC, Welch t-test, FDR
# ============================================
results = []

for site in df_shared.index:
    ctrl_vals = df_shared.loc[site, ctrl_cols].values.astype(float)
    redcap_vals = df_shared.loc[site, redcap_cols].values.astype(float)

    t_stat, p_val = ttest_ind(ctrl_vals, redcap_vals, equal_var=False, nan_policy='omit')

    mean_ctrl = np.nanmean(ctrl_vals)
    mean_redcap = np.nanmean(redcap_vals)

    fc = (mean_redcap + 1e-9) / (mean_ctrl + 1e-9)
    log2fc = np.log2(fc)

    results.append([site, mean_ctrl, mean_redcap, log2fc, p_val])

site_stats_df = pd.DataFrame(results, columns=[
    "SiteKey", "Mean_CTRL", "Mean_REDCAP", "log2FC", "pvalue"
])

site_stats_df['FDR'] = multipletests(site_stats_df['pvalue'], method='fdr_bh')[1]

print("\nSite-wise stats (head):")
print(site_stats_df.head())


# ============================================
# STEP 13 — Site-level volcano plot
# ============================================
x = site_stats_df['log2FC']
y = -np.log10(site_stats_df['FDR'] + 1e-300)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=10, alpha=0.5)

sig_mask = site_stats_df['FDR'] < 0.05
up_mask  = (site_stats_df['log2FC'] >  0.58) & sig_mask
down_mask= (site_stats_df['log2FC'] < -0.58) & sig_mask

plt.scatter(site_stats_df.loc[up_mask, 'log2FC'],
            -np.log10(site_stats_df.loc[up_mask, 'FDR'] + 1e-300),
            s=12, alpha=0.8, label='REDCAP > CTRL')

plt.scatter(site_stats_df.loc[down_mask, 'log2FC'],
            -np.log10(site_stats_df.loc[down_mask, 'FDR'] + 1e-300),
            s=12, alpha=0.8, label='CTRL > REDCAP')

plt.axvline(0, color='black', linewidth=1)
plt.xlabel('log2(REDCAP / CTRL)')
plt.ylabel('-log10(FDR)')
plt.title('Volcano Plot of Shared Cysteine Site Redox (MS2-based)')
plt.legend()
plt.tight_layout()
plt.show()

site_stats_output_path = '/content/shared_cysteine_redox_stats_CTRL_vs_REDCAP.csv'
site_stats_df.to_csv(site_stats_output_path, index=False)
print("\nSaved site stats table to:", site_stats_output_path)


# ============================================
# STEP 14 — Per-sample unique protein groups and unique precursors
# ============================================
df_inc = df[df['Run'].isin(included_runs)].copy()

# protein-group column
protein_candidates = [
    'Protein.Ids',
    'Protein.Group',
    'Protein.Groups',
    'Protein.Names'
]
protein_col = choose_first_existing(df_inc.columns, protein_candidates)
if protein_col is None:
    raise ValueError(
        f"Could not find a protein-group style column. Looked for: {protein_candidates}"
    )

df_inc['ProteinGroup_Accession'] = df_inc[protein_col].apply(first_accession)
print("\nUsing protein-group column:", protein_col)

# precursor column
precursor_candidates = [
    'Precursor.Id',
    'PrecursorID',
    'Precursor',
    'Transition.Group.Id'
]
precursor_col = choose_first_existing(df_inc.columns, precursor_candidates)

if precursor_col is not None:
    df_inc['PrecursorKey'] = df_inc[precursor_col].astype(str)
    print("Using precursor column:", precursor_col)
else:
    seq_candidates = ['Modified.Sequence', 'Stripped.Sequence', 'Sequence']
    charge_candidates = ['Precursor.Charge', 'Charge']

    seq_col = choose_first_existing(df_inc.columns, seq_candidates)
    charge_col = choose_first_existing(df_inc.columns, charge_candidates)

    if seq_col is None:
        raise ValueError("Could not find a precursor column or sequence column.")

    if charge_col is not None:
        df_inc['PrecursorKey'] = (
            df_inc[seq_col].astype(str) + "_z" + df_inc[charge_col].astype(str)
        )
        print(f"Built precursor key from: {seq_col} + {charge_col}")
    else:
        df_inc['PrecursorKey'] = df_inc[seq_col].astype(str)
        print(f"Built precursor key from sequence only: {seq_col}")

counts_df = (
    df_inc
    .groupby('Run')
    .agg(
        Unique_Protein_Groups=('ProteinGroup_Accession', lambda x: x.dropna().nunique()),
        Unique_Precursors=('PrecursorKey', lambda x: x.dropna().nunique())
    )
    .reset_index()
)

counts_df['Group'] = counts_df['Run'].apply(assign_group)
counts_df = counts_df[['Run', 'Group', 'Unique_Protein_Groups', 'Unique_Precursors']]

print("\nPer-sample counts:")
print(counts_df.to_string(index=False))

counts_out = '/content/per_sample_protein_and_precursor_counts.csv'
counts_df.to_csv(counts_out, index=False)
print("\nSaved per-sample counts to:", counts_out)


# ============================================
# STEP 15 — Protein-group differential analysis using LFQ / MaxLFQ
# ============================================
pg_quantity_candidates = [
    'PG.MaxLFQ',
    'MaxLFQ',
    'Protein.Group.MaxLFQ',
    'PG.Quantity',
    'PG.Normalised'
]
pg_quantity_col = choose_first_existing(df_inc.columns, pg_quantity_candidates)
if pg_quantity_col is None:
    raise ValueError(
        f"Could not find a protein-group LFQ/quantity column. Looked for: {pg_quantity_candidates}"
    )

print("\nUsing protein-group LFQ/quantity column:", pg_quantity_col)

work_pg = df_inc[['Run', 'ProteinGroup_Accession', pg_quantity_col]].copy()
work_pg[pg_quantity_col] = pd.to_numeric(work_pg[pg_quantity_col], errors='coerce')
work_pg = work_pg.dropna(subset=['ProteinGroup_Accession', pg_quantity_col])

# If the same protein-group quantity is repeated across multiple rows in a run,
# use max() to avoid summing duplicated protein-level values.
pg_run = (
    work_pg
    .groupby(['Run', 'ProteinGroup_Accession'])[pg_quantity_col]
    .max()
    .reset_index(name='ProteinGroup_Quantity')
)

pg_run['Group'] = pg_run['Run'].apply(assign_group)
pg_run = pg_run[pg_run['Group'].isin(['CTRL', 'REDCAP'])].copy()

ctrl_runs_pg = sorted(pg_run.loc[pg_run['Group'] == 'CTRL', 'Run'].unique())
redcap_runs_pg = sorted(pg_run.loc[pg_run['Group'] == 'REDCAP', 'Run'].unique())

print("\nCTRL runs used for protein-group stats:", ctrl_runs_pg)
print("REDCAP runs used for protein-group stats:", redcap_runs_pg)

# strict common proteins = present in all CTRL and all REDCAP runs
presence = (
    pg_run
    .groupby(['ProteinGroup_Accession', 'Group'])['Run']
    .nunique()
    .reset_index()
)

presence_wide = presence.pivot(
    index='ProteinGroup_Accession',
    columns='Group',
    values='Run'
).fillna(0)

common_pg = presence_wide.index[
    (presence_wide.get('CTRL', 0) == len(ctrl_runs_pg)) &
    (presence_wide.get('REDCAP', 0) == len(redcap_runs_pg))
]

print("\nN common protein groups detected in ALL CTRL and ALL REDCAP runs:", len(common_pg))

pg_common = pg_run[pg_run['ProteinGroup_Accession'].isin(common_pg)].copy()

pg_matrix = (
    pg_common
    .pivot(index='ProteinGroup_Accession', columns='Run', values='ProteinGroup_Quantity')
    .reindex(columns=ctrl_runs_pg + redcap_runs_pg)
)

print("Protein-group matrix shape:", pg_matrix.shape)

positive_vals = pg_matrix.values[np.isfinite(pg_matrix.values) & (pg_matrix.values > 0)]
if len(positive_vals) == 0:
    raise ValueError("No positive protein-group LFQ/quantity values found for stats.")

pseudocount = np.min(positive_vals) / 2.0
pg_log2 = np.log2(pg_matrix + pseudocount)

pg_results = []

for pg in pg_log2.index:
    ctrl_vals = pg_log2.loc[pg, ctrl_runs_pg].values.astype(float)
    redcap_vals = pg_log2.loc[pg, redcap_runs_pg].values.astype(float)

    t_stat, p_val = ttest_ind(ctrl_vals, redcap_vals, equal_var=False, nan_policy='omit')

    mean_ctrl = np.nanmean(ctrl_vals)
    mean_redcap = np.nanmean(redcap_vals)
    log2fc = mean_redcap - mean_ctrl

    pg_results.append([pg, mean_ctrl, mean_redcap, log2fc, p_val])

pg_stats = pd.DataFrame(pg_results, columns=[
    'ProteinGroup_Accession',
    'Mean_log2_CTRL',
    'Mean_log2_REDCAP',
    'log2FC',
    'pvalue'
])

pg_stats['FDR'] = multipletests(pg_stats['pvalue'], method='fdr_bh')[1]
pg_stats['neglog10FDR'] = -np.log10(pg_stats['FDR'] + 1e-300)

pg_stats['Significant'] = pg_stats['FDR'] < 0.05
pg_stats['Direction'] = 'NS'
pg_stats.loc[(pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] > 0), 'Direction'] = 'Higher in REDCAP'
pg_stats.loc[(pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] < 0), 'Direction'] = 'Higher in CTRL'

print("\nProtein-group stats (head):")
print(pg_stats.head())

print("\n========== PROTEIN-GROUP SUMMARY ==========")
print("N tested common protein groups:", len(pg_stats))
print("FDR < 0.05:", int((pg_stats['FDR'] < 0.05).sum()))
print("Higher in REDCAP:", int(((pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] > 0)).sum()))
print("Higher in CTRL:", int(((pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] < 0)).sum()))

pg_stats_out = '/content/common_protein_group_stats_CTRL_vs_REDCAP.csv'
pg_stats.to_csv(pg_stats_out, index=False)
print("\nSaved protein-group stats to:", pg_stats_out)


# ============================================
# STEP 16 — Protein-group volcano plot
# ============================================
top_label_df = (
    pg_stats[pg_stats['FDR'] < 0.05]
    .sort_values(['FDR', 'pvalue', 'neglog10FDR'], ascending=[True, True, False])
    .head(12)
)

plt.figure(figsize=(9, 7))

ns = pg_stats['FDR'] >= 0.05
sig_up = (pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] > 0)
sig_down = (pg_stats['FDR'] < 0.05) & (pg_stats['log2FC'] < 0)

plt.scatter(pg_stats.loc[ns, 'log2FC'],
            pg_stats.loc[ns, 'neglog10FDR'],
            s=18, alpha=0.35, label='Not significant')

plt.scatter(pg_stats.loc[sig_up, 'log2FC'],
            pg_stats.loc[sig_up, 'neglog10FDR'],
            s=24, alpha=0.85, label='Higher in REDCAP')

plt.scatter(pg_stats.loc[sig_down, 'log2FC'],
            pg_stats.loc[sig_down, 'neglog10FDR'],
            s=24, alpha=0.85, label='Higher in CTRL')

plt.axvline(0, color='black', linewidth=1)
plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
plt.axvline(1.0, color='black', linestyle=':', linewidth=0.8)
plt.axvline(-1.0, color='black', linestyle=':', linewidth=0.8)

for _, row in top_label_df.iterrows():
    plt.text(
        row['log2FC'],
        row['neglog10FDR'],
        row['ProteinGroup_Accession'],
        fontsize=8,
        ha='left',
        va='bottom'
    )

plt.xlabel('log2 fold-change (REDCAP / CTRL)')
plt.ylabel('-log10(FDR)')
plt.title(f'Protein-group volcano plot\nCommon proteins across all included runs (N = {len(pg_stats)})')
plt.legend(frameon=False)
plt.tight_layout()

volcano_out = '/content/protein_group_volcano_CTRL_vs_REDCAP.png'
plt.savefig(volcano_out, dpi=600, bbox_inches='tight')
plt.show()
