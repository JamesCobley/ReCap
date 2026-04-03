# ============================================
# STEP 19 — Check whether site sampling confounds
#            the abundance–redox relationship
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

FASTA_PATH = '/content/Mus musculus_Sp_canonical_20240408.fasta'

# --------------------------------------------
# A. Parse FASTA and count total cysteines per accession
# --------------------------------------------
def parse_fasta_cys_counts(fasta_path):
    records = []
    header = None
    seq_chunks = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    seq = ''.join(seq_chunks)
                    records.append((header, seq))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if header is not None:
            seq = ''.join(seq_chunks)
            records.append((header, seq))

    out = []
    for header, seq in records:
        # SwissProt canonical style often: sp|Q9Z1M0|NAME ...
        parts = header.split('|')
        if len(parts) >= 2:
            accession = parts[1]
        else:
            accession = header.split()[0]

        total_cys_in_sequence = seq.count('C')
        protein_length = len(seq)

        out.append({
            'ProteinGroup_Accession': accession,
            'Protein_Length': protein_length,
            'Total_Cys_Sequence': total_cys_in_sequence
        })

    return pd.DataFrame(out)

fasta_df = parse_fasta_cys_counts(FASTA_PATH)
print("FASTA proteins parsed:", len(fasta_df))
print(fasta_df.head())

# --------------------------------------------
# B. Build per-condition measured cysteine counts
#    protein_condition should already exist from the rank/redox step
#    It contains:
#      Group, ProteinGroup_Accession, Mean_log2_LFQ, Mean_Protein_Redox,
#      Mean_N_common_sites, N_runs
# --------------------------------------------
confound_df = protein_condition.copy()

# rename for clarity
confound_df = confound_df.rename(columns={
    'Mean_N_common_sites': 'Measured_Common_Cys_Sites'
})

# merge in FASTA-derived counts
confound_df = confound_df.merge(
    fasta_df,
    on='ProteinGroup_Accession',
    how='left'
)

# add coverage metrics
confound_df['Measured_Common_Cys_Sites'] = pd.to_numeric(
    confound_df['Measured_Common_Cys_Sites'], errors='coerce'
)
confound_df['Total_Cys_Sequence'] = pd.to_numeric(
    confound_df['Total_Cys_Sequence'], errors='coerce'
)
confound_df['Protein_Length'] = pd.to_numeric(
    confound_df['Protein_Length'], errors='coerce'
)

confound_df['Frac_Cys_Measured'] = (
    confound_df['Measured_Common_Cys_Sites'] / confound_df['Total_Cys_Sequence']
)
confound_df.loc[confound_df['Total_Cys_Sequence'] <= 0, 'Frac_Cys_Measured'] = np.nan

print("\nMerged confound table shape:", confound_df.shape)
print(confound_df.head())

# --------------------------------------------
# C. Spearman checks per group
# --------------------------------------------
rows = []

for grp in ['CTRL', 'REDCAP']:
    sub = confound_df[confound_df['Group'] == grp].copy()

    tests = [
        ('Abundance_vs_Redox', 'Mean_log2_LFQ', 'Mean_Protein_Redox'),
        ('Abundance_vs_MeasuredSites', 'Mean_log2_LFQ', 'Measured_Common_Cys_Sites'),
        ('Abundance_vs_TotalCys', 'Mean_log2_LFQ', 'Total_Cys_Sequence'),
        ('MeasuredSites_vs_Redox', 'Measured_Common_Cys_Sites', 'Mean_Protein_Redox'),
        ('TotalCys_vs_Redox', 'Total_Cys_Sequence', 'Mean_Protein_Redox'),
        ('FracMeasured_vs_Redox', 'Frac_Cys_Measured', 'Mean_Protein_Redox'),
    ]

    for test_name, xcol, ycol in tests:
        tmp = sub[[xcol, ycol]].dropna()
        if len(tmp) >= 3:
            rho, pval = spearmanr(tmp[xcol], tmp[ycol], nan_policy='omit')
        else:
            rho, pval = np.nan, np.nan

        rows.append({
            'Group': grp,
            'Test': test_name,
            'N': len(tmp),
            'Spearman_rho': rho,
            'pvalue': pval
        })

corr_checks = pd.DataFrame(rows)
print("\nConfounder checks:")
print(corr_checks.to_string(index=False))

corr_checks_out = '/content/abundance_redox_sitecount_confound_checks.csv'
corr_checks.to_csv(corr_checks_out, index=False)
print("\nSaved confounder checks to:", corr_checks_out)

# --------------------------------------------
# D. Residualization:
#    Does abundance still associate with redox after regressing out
#    measured common site count?
# --------------------------------------------
def residualize(y, x):
    """
    Return residuals from simple linear regression y ~ x
    using numpy polyfit.
    """
    mask = np.isfinite(y) & np.isfinite(x)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    resid = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 3:
        return resid

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    pred = slope * x[mask] + intercept
    resid[mask] = y[mask] - pred
    return resid

partial_rows = []

for grp in ['CTRL', 'REDCAP']:
    sub = confound_df[confound_df['Group'] == grp].copy()

    # residualize abundance and redox against measured site count
    sub['Redox_resid_given_sites'] = residualize(
        sub['Mean_Protein_Redox'].values,
        sub['Measured_Common_Cys_Sites'].values
    )
    sub['Abundance_resid_given_sites'] = residualize(
        sub['Mean_log2_LFQ'].values,
        sub['Measured_Common_Cys_Sites'].values
    )

    tmp = sub[['Abundance_resid_given_sites', 'Redox_resid_given_sites']].dropna()
    if len(tmp) >= 3:
        rho, pval = spearmanr(
            tmp['Abundance_resid_given_sites'],
            tmp['Redox_resid_given_sites'],
            nan_policy='omit'
        )
    else:
        rho, pval = np.nan, np.nan

    partial_rows.append({
        'Group': grp,
        'N': len(tmp),
        'Partial_like_rho_controlling_measured_sites': rho,
        'pvalue': pval
    })

partial_df = pd.DataFrame(partial_rows)
print("\nAbundance–redox relationship after controlling for measured common site count:")
print(partial_df.to_string(index=False))

partial_out = '/content/partial_like_abundance_redox_controlling_measured_sites.csv'
partial_df.to_csv(partial_out, index=False)
print("\nSaved partial-like results to:", partial_out)

# --------------------------------------------
# E. Plots
# --------------------------------------------
for grp in ['CTRL', 'REDCAP']:
    sub = confound_df[confound_df['Group'] == grp].copy()

    # 1) abundance vs measured site count
    plt.figure(figsize=(6,5))
    plt.scatter(
        sub['Mean_log2_LFQ'],
        sub['Measured_Common_Cys_Sites'],
        s=20,
        alpha=0.5
    )
    rho, pval = spearmanr(
        sub[['Mean_log2_LFQ', 'Measured_Common_Cys_Sites']].dropna()['Mean_log2_LFQ'],
        sub[['Mean_log2_LFQ', 'Measured_Common_Cys_Sites']].dropna()['Measured_Common_Cys_Sites']
    )
    plt.xlabel('Mean log2 LFQ')
    plt.ylabel('Measured common cysteine sites')
    plt.title(f'{grp}: abundance vs measured common cysteine count\nSpearman rho = {rho:.3f}, p = {pval:.3g}')
    plt.tight_layout()
    plt.savefig(f'/content/abundance_vs_measured_sites_{grp}.png', dpi=300)
    plt.show()

    # 2) measured site count vs redox
    plt.figure(figsize=(6,5))
    plt.scatter(
        sub['Measured_Common_Cys_Sites'],
        sub['Mean_Protein_Redox'],
        s=20,
        alpha=0.5
    )
    tmp = sub[['Measured_Common_Cys_Sites', 'Mean_Protein_Redox']].dropna()
    rho, pval = spearmanr(tmp['Measured_Common_Cys_Sites'], tmp['Mean_Protein_Redox'])
    plt.xlabel('Measured common cysteine sites')
    plt.ylabel('Mean protein redox (%)')
    plt.title(f'{grp}: measured site count vs redox\nSpearman rho = {rho:.3f}, p = {pval:.3g}')
    plt.tight_layout()
    plt.savefig(f'/content/measured_sites_vs_redox_{grp}.png', dpi=300)
    plt.show()

    # 3) abundance vs redox colored by measured site count
    plt.figure(figsize=(6.5,5.5))
    sc = plt.scatter(
        sub['Mean_log2_LFQ'],
        sub['Mean_Protein_Redox'],
        c=sub['Measured_Common_Cys_Sites'],
        cmap='viridis',
        s=24,
        alpha=0.8
    )
    tmp = sub[['Mean_log2_LFQ', 'Mean_Protein_Redox']].dropna()
    rho, pval = spearmanr(tmp['Mean_log2_LFQ'], tmp['Mean_Protein_Redox'])
    plt.xlabel('Mean log2 LFQ')
    plt.ylabel('Mean protein redox (%)')
    plt.title(f'{grp}: abundance vs redox\ncolored by measured common site count\nSpearman rho = {rho:.3f}, p = {pval:.3g}')
    cbar = plt.colorbar(sc)
    cbar.set_label('Measured common cysteine sites')
    plt.tight_layout()
    plt.savefig(f'/content/abundance_vs_redox_{grp}.png', dpi=300)
    plt.show()

# --------------------------------------------
# F. Save merged table
# --------------------------------------------
confound_out = '/content/protein_abundance_redox_with_fasta_sitecounts.csv'
confound_df.to_csv(confound_out, index=False)
print("\nSaved merged abundance/redox/site-count table to:", confound_out)
