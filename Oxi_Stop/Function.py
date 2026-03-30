# ============================================
# Load functional cysteine annotation resource
# ============================================

import pandas as pd
import numpy as np

func_path = '/content/cysteine_functional_sites.csv'  # adjust if needed
func_df = pd.read_csv(func_path)

print("Functional resource shape:", func_df.shape)
print("Columns:", func_df.columns.tolist())

# Build SiteKey to match your stats table
func_df['SiteKey'] = func_df['accession'].astype(str) + "_" + func_df['cys_position'].astype(str)

# Keep a compact annotation table
func_annot = func_df[[
    'SiteKey', 'accession', 'gene_primary', 'protein_name',
    'cys_position', 'feature_type', 'feature_description',
    'ligand_name', 'ligand_id', 'ligand_part_name',
    'evidence', 'source_note'
]].copy()

print("\nUnique annotated SiteKeys:", func_annot['SiteKey'].nunique())
print(func_annot.head())

# ============================================
# Robust merge: split SiteKey into accession + cys_position
# ============================================

import pandas as pd
import numpy as np

# Load functional resource
func_df = pd.read_csv('/content/cysteine_functional_sites.csv')

# Clean annotation columns
func_df['accession_clean'] = func_df['accession'].astype(str).str.strip()
func_df['cys_position_clean'] = pd.to_numeric(func_df['cys_position'], errors='coerce').astype('Int64')

print("Functional rows:", len(func_df))
print("Unique accessions:", func_df['accession_clean'].nunique())

# Split SiteKey from your redox table
df_map = df_id.copy()

split_cols = df_map['SiteKey'].astype(str).str.rsplit('_', n=1, expand=True)
df_map['accession_clean'] = split_cols[0].str.strip()
df_map['cys_position_clean'] = pd.to_numeric(split_cols[1], errors='coerce').astype('Int64')

print("\nRedox rows:", len(df_map))
print("Unique parsed accessions:", df_map['accession_clean'].nunique())

# Merge on the two clean columns
merged = df_map.merge(
    func_df,
    on=['accession_clean', 'cys_position_clean'],
    how='left',
    suffixes=('', '_func')
)

print("\nMerged rows:", len(merged))
print("Rows with annotation:", merged['feature_type'].notna().sum())
print("Unique annotated SiteKeys:", merged.loc[merged['feature_type'].notna(), 'SiteKey'].nunique())

# Nonidentical annotated subset
nonident_annot = merged[(merged['NonIdentity']) & (merged['feature_type'].notna())].copy()

print("\nAnnotated nonidentical rows:", len(nonident_annot))
print("Unique annotated nonidentical SiteKeys:", nonident_annot['SiteKey'].nunique())

# Quick look
print("\nTop feature types:")
print(nonident_annot['feature_type'].value_counts().head(20))

print("\nTop genes:")
print(nonident_annot['gene_primary'].value_counts().head(20))

# Save output
nonident_annot.to_csv('/content/nonidentical_functionally_annotated_sites.csv', index=False)
print("\nSaved: /content/nonidentical_functionally_annotated_sites.csv")

# Unique-site direction summary among annotated nonidentical sites
annot_unique = nonident_annot.drop_duplicates(subset=['SiteKey']).copy()

ctrl_higher_unique = annot_unique[annot_unique['Delta_Percent'] < 0]
redcap_higher_unique = annot_unique[annot_unique['Delta_Percent'] > 0]

print("Unique annotated nonidentical SiteKeys:", annot_unique['SiteKey'].nunique())
print("More oxidised in CTRL:", ctrl_higher_unique['SiteKey'].nunique())
print("More oxidised in ReCap:", redcap_higher_unique['SiteKey'].nunique())

print("\nFeature types among unique annotated sites more oxidised in CTRL:")
print(ctrl_higher_unique['feature_type'].value_counts())

print("\nFeature types among unique annotated sites more oxidised in ReCap:")
print(redcap_higher_unique['feature_type'].value_counts())
