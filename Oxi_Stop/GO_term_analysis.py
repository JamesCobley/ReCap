# ============================================
# Protein-level GO enrichment
# Nonidentical proteins vs identical proteins
# Using protein_annotations_long_all_cys.csv
# ============================================

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------
# 1. Build protein-level class from site-level table
# df_id must already exist and contain:
#   SiteKey, Identity, NonIdentity
# ------------------------------------------------

df_prot = df_id.copy()

# Split SiteKey into accession + position
split_cols = df_prot['SiteKey'].astype(str).str.rsplit('_', n=1, expand=True)
df_prot['accession'] = split_cols[0].str.strip()

protein_class = (
    df_prot
    .groupby('accession')
    .agg(
        n_sites=('SiteKey', 'nunique'),
        any_nonidentical=('NonIdentity', 'max'),
        all_identical=('Identity', 'min')
    )
    .reset_index()
)

protein_class['ProteinClass'] = np.where(
    protein_class['any_nonidentical'],
    'Nonidentical',
    'Identical'
)

print("Protein class counts:")
print(protein_class['ProteinClass'].value_counts())
print("\nUnique measured proteins:", protein_class['accession'].nunique())

# ------------------------------------------------
# 2. Load annotation sheet and keep GO only
# ------------------------------------------------

ann_path = '/content/protein_annotations_long_all_cys.csv'
ann = pd.read_csv(ann_path)

print("\nAnnotation sheet shape:", ann.shape)
print("Columns:", ann.columns.tolist())

ann['accession'] = ann['accession'].astype(str).str.strip()
ann['annotation_source'] = ann['annotation_source'].astype(str).str.strip()
ann['annotation_class'] = ann['annotation_class'].astype(str).str.strip()
ann['annotation_id'] = ann['annotation_id'].astype(str).str.strip()
ann['annotation_name'] = ann['annotation_name'].astype(str).str.strip()

go = ann[ann['annotation_source'] == 'GO'].copy()

print("\nGO rows:", len(go))
print("Unique GO-annotated proteins:", go['accession'].nunique())
print("GO classes:")
print(go['annotation_class'].value_counts())

# Restrict to measured proteins only
measured_proteins = set(protein_class['accession'])
go = go[go['accession'].isin(measured_proteins)].copy()

print("\nGO rows after restricting to measured proteins:", len(go))
print("Unique measured proteins with GO:", go['accession'].nunique())

# Drop duplicate protein-term pairs
go = go.drop_duplicates(subset=['accession', 'annotation_id'])

# ------------------------------------------------
# 3. Define foreground/background sets
# ------------------------------------------------

nonident_proteins = set(
    protein_class.loc[protein_class['ProteinClass'] == 'Nonidentical', 'accession']
)
ident_proteins = set(
    protein_class.loc[protein_class['ProteinClass'] == 'Identical', 'accession']
)

print("\nProtein sets:")
print("Nonidentical proteins:", len(nonident_proteins))
print("Identical proteins:", len(ident_proteins))

# ------------------------------------------------
# 4. Fisher exact test for each GO term
# ------------------------------------------------

rows = []

for go_id, sub in go.groupby('annotation_id'):
    go_term = sub['annotation_name'].iloc[0]
    go_class = sub['annotation_class'].iloc[0]
    go_group = sub['annotation_group'].iloc[0] if 'annotation_group' in sub.columns else None

    term_proteins = set(sub['accession'])

    a = len(nonident_proteins & term_proteins)  # nonidentical with term
    b = len(nonident_proteins - term_proteins)  # nonidentical without term
    c = len(ident_proteins & term_proteins)     # identical with term
    d = len(ident_proteins - term_proteins)     # identical without term

    if (a + c) == 0:
        continue

    oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative='two-sided')

    rows.append({
        'go_id': go_id,
        'go_term': go_term,
        'go_class': go_class,
        'go_group': go_group,
        'nonident_with_term': a,
        'nonident_without_term': b,
        'ident_with_term': c,
        'ident_without_term': d,
        'odds_ratio': oddsratio,
        'pvalue': pvalue
    })

enrich = pd.DataFrame(rows)

# Multiple testing correction
enrich['FDR'] = multipletests(enrich['pvalue'], method='fdr_bh')[1]

# Fractions for interpretability
enrich['pct_nonident'] = enrich['nonident_with_term'] / (
    enrich['nonident_with_term'] + enrich['nonident_without_term']
)
enrich['pct_ident'] = enrich['ident_with_term'] / (
    enrich['ident_with_term'] + enrich['ident_without_term']
)
enrich['delta_pct'] = enrich['pct_nonident'] - enrich['pct_ident']

# Sort
enrich = enrich.sort_values(['FDR', 'odds_ratio'], ascending=[True, False])

print("\nTop enriched GO terms overall:")
print(enrich.head(30).to_string(index=False))

# ------------------------------------------------
# 5. Optional: split by GO class
# ------------------------------------------------

for cls in ['Biological Process', 'Molecular Function', 'Cellular Component']:
    sub = enrich[enrich['go_class'] == cls].copy()
    if len(sub):
        print(f"\nTop enriched terms: {cls}")
        print(sub.head(15).to_string(index=False))

# ------------------------------------------------
# 6. Save results
# ------------------------------------------------

out_path = '/content/GO_enrichment_nonidentical_vs_identical_proteins.csv'
enrich.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ============================================
# Significant GO terms + redox distortion summary
# ============================================

import pandas as pd
import numpy as np

# -------------------------------
# 1. Keep significant GO terms
# -------------------------------
sig_go = enrich[enrich['FDR'] < 0.05].copy()

print("Significant GO terms (FDR < 0.05):", len(sig_go))
if len(sig_go):
    print(sig_go[['go_id', 'go_term', 'go_class', 'odds_ratio', 'pvalue', 'FDR']].head(30).to_string(index=False))
else:
    print("No significant GO terms at FDR < 0.05")

# Save significant terms
sig_go.to_csv('/content/significant_GO_terms_FDR_lt_0.05.csv', index=False)

# -------------------------------
# 2. Prepare site-level table with accession
#    df_id must contain:
#    SiteKey, Identity, NonIdentity, Mean_CTRL, Mean_REDCAP, Delta_Percent
# -------------------------------
site_df = df_id.copy()

split_cols = site_df['SiteKey'].astype(str).str.rsplit('_', n=1, expand=True)
site_df['accession'] = split_cols[0].str.strip()
site_df['cys_position'] = pd.to_numeric(split_cols[1], errors='coerce').astype('Int64')

# Protein-level class table from earlier
protein_class2 = protein_class.copy()

# Restrict GO annotation table to GO only, measured proteins, unique protein-term pairs
go2 = ann[ann['annotation_source'] == 'GO'].copy()
go2['accession'] = go2['accession'].astype(str).str.strip()
go2['annotation_id'] = go2['annotation_id'].astype(str).str.strip()
go2['annotation_name'] = go2['annotation_name'].astype(str).str.strip()
go2['annotation_class'] = go2['annotation_class'].astype(str).str.strip()
if 'annotation_group' in go2.columns:
    go2['annotation_group'] = go2['annotation_group'].astype(str).str.strip()
else:
    go2['annotation_group'] = np.nan

go2 = go2[go2['accession'].isin(set(protein_class2['accession']))].copy()
go2 = go2.drop_duplicates(subset=['accession', 'annotation_id'])

# -------------------------------
# 3. Summarise distortion for each significant GO term
# -------------------------------
dist_rows = []

for _, row in sig_go.iterrows():
    go_id = row['go_id']
    go_term = row['go_term']
    go_class = row['go_class']
    go_group = row['go_group'] if 'go_group' in row else np.nan

    term_proteins = set(go2.loc[go2['annotation_id'] == go_id, 'accession'])

    # protein-level summary
    prot_sub = protein_class2[protein_class2['accession'].isin(term_proteins)].copy()
    n_term_proteins = len(prot_sub)
    n_nonident_proteins = int((prot_sub['ProteinClass'] == 'Nonidentical').sum())
    n_ident_proteins = int((prot_sub['ProteinClass'] == 'Identical').sum())

    # site-level summary
    site_sub = site_df[site_df['accession'].isin(term_proteins)].copy()
    n_term_sites = len(site_sub)
    n_nonident_sites = int(site_sub['NonIdentity'].sum())
    n_ident_sites = int(site_sub['Identity'].sum())

    # direction among sites
    n_sites_ctrl_higher = int((site_sub['Delta_Percent'] < 0).sum())
    n_sites_redcap_higher = int((site_sub['Delta_Percent'] > 0).sum())
    n_sites_equal = int((site_sub['Delta_Percent'] == 0).sum())

    # magnitude summaries
    mean_delta_all_sites = site_sub['Delta_Percent'].mean() if len(site_sub) else np.nan
    median_delta_all_sites = site_sub['Delta_Percent'].median() if len(site_sub) else np.nan

    nonident_site_sub = site_sub[site_sub['NonIdentity']].copy()
    mean_delta_nonident_sites = nonident_site_sub['Delta_Percent'].mean() if len(nonident_site_sub) else np.nan
    median_delta_nonident_sites = nonident_site_sub['Delta_Percent'].median() if len(nonident_site_sub) else np.nan
    mean_abs_delta_nonident_sites = nonident_site_sub['Delta_Percent'].abs().mean() if len(nonident_site_sub) else np.nan

    # proportions
    pct_nonident_proteins = n_nonident_proteins / n_term_proteins if n_term_proteins else np.nan
    pct_nonident_sites = n_nonident_sites / n_term_sites if n_term_sites else np.nan
    pct_ctrl_higher_sites = n_sites_ctrl_higher / n_term_sites if n_term_sites else np.nan
    pct_redcap_higher_sites = n_sites_redcap_higher / n_term_sites if n_term_sites else np.nan

    dist_rows.append({
        'go_id': go_id,
        'go_term': go_term,
        'go_class': go_class,
        'go_group': go_group,
        'odds_ratio': row['odds_ratio'],
        'pvalue': row['pvalue'],
        'FDR': row['FDR'],

        'n_term_proteins': n_term_proteins,
        'n_nonident_proteins': n_nonident_proteins,
        'n_ident_proteins': n_ident_proteins,
        'pct_nonident_proteins': pct_nonident_proteins,

        'n_term_sites': n_term_sites,
        'n_nonident_sites': n_nonident_sites,
        'n_ident_sites': n_ident_sites,
        'pct_nonident_sites': pct_nonident_sites,

        'n_sites_ctrl_higher': n_sites_ctrl_higher,
        'n_sites_redcap_higher': n_sites_redcap_higher,
        'n_sites_equal': n_sites_equal,
        'pct_ctrl_higher_sites': pct_ctrl_higher_sites,
        'pct_redcap_higher_sites': pct_redcap_higher_sites,

        'mean_delta_all_sites': mean_delta_all_sites,
        'median_delta_all_sites': median_delta_all_sites,
        'mean_delta_nonident_sites': mean_delta_nonident_sites,
        'median_delta_nonident_sites': median_delta_nonident_sites,
        'mean_abs_delta_nonident_sites': mean_abs_delta_nonident_sites
    })

dist_df = pd.DataFrame(dist_rows)

# -------------------------------
# 4. Sort and inspect
# -------------------------------
if len(dist_df):
    dist_df = dist_df.sort_values(['FDR', 'mean_abs_delta_nonident_sites'], ascending=[True, False])

    print("\nSignificant GO terms with distortion summary:")
    cols = [
        'go_id', 'go_term', 'go_class', 'FDR', 'odds_ratio',
        'n_term_proteins', 'pct_nonident_proteins',
        'n_term_sites', 'pct_nonident_sites',
        'pct_ctrl_higher_sites', 'pct_redcap_higher_sites',
        'mean_delta_all_sites', 'mean_delta_nonident_sites',
        'mean_abs_delta_nonident_sites'
    ]
    print(dist_df[cols].head(30).to_string(index=False))
else:
    print("\nNo significant GO terms to summarise.")

# Save
dist_df.to_csv('/content/significant_GO_terms_with_redox_distortion_summary.csv', index=False)
print("\nSaved:")
print("/content/significant_GO_terms_FDR_lt_0.05.csv")
print("/content/significant_GO_terms_with_redox_distortion_summary.csv")
