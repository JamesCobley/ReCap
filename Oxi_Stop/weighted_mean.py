# --- A. Build site metadata for shared/common sites only ---

site_meta = df_cys[['Run', 'Site.Protein.ID', 'Site.Position']].copy()
site_meta['SiteKey'] = (
    site_meta['Site.Protein.ID'].astype(str) + "_" + site_meta['Site.Position'].astype(str)
)

# keep only common/shared sites
site_meta = site_meta[site_meta['SiteKey'].isin(shared_sitekeys)].copy()

# unique Run-SiteKey to protein mapping
site_meta = (
    site_meta[['Run', 'SiteKey', 'Site.Protein.ID']]
    .drop_duplicates()
    .rename(columns={'Site.Protein.ID': 'ProteinGroup_Accession'})
)

# --- B. Shared-site redox values only ---
site_redox = pivot_inc[['Run', 'SiteKey', 'Percent_Ox']].copy()
site_redox = site_redox[site_redox['SiteKey'].isin(shared_sitekeys)].copy()

# merge site → protein mapping
site_redox = site_redox.merge(
    site_meta,
    on=['Run', 'SiteKey'],
    how='left'
)

# --- C. Protein LFQ per run ---
lfq_work = df_inc[['Run', 'ProteinGroup_Accession', pg_quantity_col]].copy()
lfq_work[pg_quantity_col] = pd.to_numeric(lfq_work[pg_quantity_col], errors='coerce')
lfq_work = lfq_work.dropna(subset=['Run', 'ProteinGroup_Accession', pg_quantity_col]).copy()

# avoid duplicating repeated protein-level LFQ values across rows
lfq_run = (
    lfq_work
    .groupby(['Run', 'ProteinGroup_Accession'])[pg_quantity_col]
    .max()
    .reset_index(name='Protein_LFQ')
)

# --- D. Attach LFQ weights to each shared site ---
weighted_sites = site_redox.merge(
    lfq_run,
    on=['Run', 'ProteinGroup_Accession'],
    how='left'
)

weighted_sites['Percent_Ox'] = pd.to_numeric(weighted_sites['Percent_Ox'], errors='coerce')
weighted_sites['Protein_LFQ'] = pd.to_numeric(weighted_sites['Protein_LFQ'], errors='coerce')

weighted_sites = weighted_sites.dropna(subset=['Percent_Ox', 'Protein_LFQ']).copy()
weighted_sites = weighted_sites[weighted_sites['Protein_LFQ'] > 0].copy()

print("Weighted shared-site rows:", len(weighted_sites))
print("Number of common shared SiteKeys:", len(shared_sitekeys))

# --- E. Compute LFQ-weighted mean redox per sample on common sites only ---
weighted_summary = (
    weighted_sites
    .assign(weighted_redox = weighted_sites['Percent_Ox'] * weighted_sites['Protein_LFQ'])
    .groupby('Run')
    .agg(
        Group=('Run', lambda x: assign_group(x.iloc[0])),
        N_common_sites_used=('SiteKey', 'nunique'),
        N_proteins_used=('ProteinGroup_Accession', 'nunique'),
        Sum_weights=('Protein_LFQ', 'sum'),
        Sum_weighted_redox=('weighted_redox', 'sum')
    )
    .reset_index()
)

weighted_summary['LFQ_weighted_mean_cysteine_redox_common_sites'] = (
    weighted_summary['Sum_weighted_redox'] / weighted_summary['Sum_weights']
)

weighted_summary = weighted_summary[
    weighted_summary['Group'].isin(['CTRL', 'REDCAP'])
].copy()

weighted_summary = weighted_summary.sort_values(['Group', 'Run']).reset_index(drop=True)

print("\nLFQ-weighted mean cysteine redox state per sample (common sites only):")
print(
    weighted_summary[
        [
            'Run',
            'Group',
            'LFQ_weighted_mean_cysteine_redox_common_sites',
            'N_common_sites_used',
            'N_proteins_used'
        ]
    ].to_string(index=False)
)

# save table
weighted_out = '/content/LFQ_weighted_mean_cysteine_redox_common_sites_per_sample.csv'
weighted_summary.to_csv(weighted_out, index=False)
print("\nSaved weighted summary to:", weighted_out)

# --- F. Group comparison ---
ctrl_vals = weighted_summary.loc[
    weighted_summary['Group'] == 'CTRL',
    'LFQ_weighted_mean_cysteine_redox_common_sites'
].values.astype(float)

redcap_vals = weighted_summary.loc[
    weighted_summary['Group'] == 'REDCAP',
    'LFQ_weighted_mean_cysteine_redox_common_sites'
].values.astype(float)

mean_ctrl = np.nanmean(ctrl_vals)
mean_redcap = np.nanmean(redcap_vals)

if len(ctrl_vals) >= 2 and len(redcap_vals) >= 2:
    t_stat, p_val = ttest_ind(ctrl_vals, redcap_vals, equal_var=False, nan_policy='omit')
else:
    t_stat, p_val = np.nan, np.nan

print("\n========== LFQ-WEIGHTED REDOX SUMMARY (COMMON SITES ONLY) ==========")
print(f"CTRL mean:   {mean_ctrl:.4f}")
print(f"REDCAP mean: {mean_redcap:.4f}")
print(f"Difference (REDCAP - CTRL): {mean_redcap - mean_ctrl:.4f}")
print(f"Welch t-test p-value: {p_val}")

# --- G. Simple plot ---
plt.figure(figsize=(6, 5))

for grp in ['CTRL', 'REDCAP']:
    y = weighted_summary.loc[
        weighted_summary['Group'] == grp,
        'LFQ_weighted_mean_cysteine_redox_common_sites'
    ].values
    x = np.repeat(grp, len(y))
    plt.scatter(x, y, s=55, alpha=0.85)

group_means = (
    weighted_summary
    .groupby('Group')['LFQ_weighted_mean_cysteine_redox_common_sites']
    .mean()
)

for grp, val in group_means.items():
    plt.scatter([grp], [val], s=200, marker='_', linewidths=3)

plt.ylabel('LFQ-weighted mean cysteine redox (%)')
plt.title('Weighted mean cysteine redox state\n(common shared sites only)')
plt.tight_layout()
plt.show()
