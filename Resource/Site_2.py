import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# INPUT FILES
# -----------------------------
long_path = "/content/cysteine_feature_annotations_long.csv"
summary_path = "/content/cysteine_feature_annotations_summary.csv"  # optional, not required here

# Your known total from the FASTA count
TOTAL_CYS = 277404

# -----------------------------
# LOAD LONG ANNOTATION TABLE
# -----------------------------
annot_df = pd.read_csv(long_path)

print("Columns in annot_df:")
print(annot_df.columns.tolist())

# -----------------------------
# CHECK / STANDARDIZE COLUMN NAMES
# -----------------------------
# Expected:
# accession, cys_position, feature_type
req = {"accession", "cys_position", "feature_type"}
missing = req - set(annot_df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found: {annot_df.columns.tolist()}")

# -----------------------------
# KEEP FEATURES OF INTEREST
# -----------------------------
keep_features = ["Active site", "Binding site", "Disulfide bond", "Zinc finger"]

tmp = annot_df[annot_df["feature_type"].isin(keep_features)].copy()

# Unique cysteine ID = protein accession + residue position
tmp["cys_id"] = (
    tmp["accession"].astype(str) + "|" +
    tmp["cys_position"].astype(str)
)

# Union of all annotated cysteines
annotated_union = tmp["cys_id"].nunique()
unannotated = TOTAL_CYS - annotated_union

if unannotated < 0:
    raise ValueError(
        f"Unannotated count is negative ({unannotated}). "
        "This suggests TOTAL_CYS is wrong or the annotation table includes duplicates from a different source."
    )

# -----------------------------
# PRINT SUMMARY
# -----------------------------
summary = pd.DataFrame({
    "Category": ["Functionally annotated", "Functionally unannotated"],
    "Count": [annotated_union, unannotated],
    "Percent": [annotated_union / TOTAL_CYS * 100, unannotated / TOTAL_CYS * 100]
})

print("\nSummary:")
print(summary)

# -----------------------------
# PIE CHART
# -----------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.pie(
    summary["Count"],
    labels=[f"{cat}\n({cnt:,})" for cat, cnt in zip(summary["Category"], summary["Count"])],
    autopct=lambda p: f"{p:.1f}%",
    startangle=90
)

ax.set_title("Mouse cysteines: functionally annotated vs unannotated")
plt.tight_layout()

out_png = "/content/cysteine_annotated_vs_unannotated_pie_300dpi.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {out_png}")
