# ============================================
# 1) Count cysteines in FASTA and proteins
# ============================================
!pip -q install biopython requests pandas tqdm

from Bio import SeqIO
import pandas as pd

fasta_path = "/content/Mus musculus_Sp_canonical_20240408.fasta"

records = list(SeqIO.parse(fasta_path, "fasta"))

rows = []
for rec in records:
    seq = str(rec.seq)
    n_cys = seq.count("C")

    # UniProt FASTA headers usually look like:
    # sp|Q9D0M3|NAME_MOUSE ...
    parts = rec.id.split("|")
    accession = parts[1] if len(parts) >= 2 else rec.id

    rows.append({
        "fasta_id": rec.id,
        "accession": accession,
        "length": len(seq),
        "n_cys": n_cys,
        "has_cys": n_cys > 0,
        "sequence": seq
    })

fasta_df = pd.DataFrame(rows)

total_cys = int(fasta_df["n_cys"].sum())
proteins_with_cys = int(fasta_df["has_cys"].sum())
total_proteins = len(fasta_df)

print("Total proteins in FASTA:", total_proteins)
print("Proteins containing >=1 cysteine:", proteins_with_cys)
print("Total cysteines across FASTA:", total_cys)

# Optional export
fasta_df[["accession","length","n_cys"]].to_csv("/content/fasta_cys_counts.csv", index=False)

# ============================================
# 2) Query UniProtKB JSON features
#    and overlap with cysteine positions
# ============================================
import requests
import time
from tqdm.auto import tqdm

def get_uniprot_json(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None

def cys_positions(seq):
    # 1-based positions
    return [i+1 for i, aa in enumerate(seq) if aa == "C"]

def parse_features(entry_json):
    feats = entry_json.get("features", []) if entry_json else []
    out = []
    for f in feats:
        ftype = f.get("type", "")
        desc = f.get("description", "")

        loc = f.get("location", {})
        start = loc.get("start", {}).get("value", None)
        end   = loc.get("end", {}).get("value", None)

        # some site-like features may only have a position
        pos = loc.get("position", {}).get("value", None)
        if pos is not None:
            start = end = pos

        if start is not None and end is not None:
            out.append({
                "type": ftype,
                "description": desc,
                "start": int(start),
                "end": int(end)
            })
    return out

# Feature groups of interest
SITE_LIKE = {
    "Active site",
    "Binding site",
    "Metal binding",
    "Modified residue",
    "Mutagenesis",
    "Site"
}

REGION_LIKE = {
    "Zinc finger",
    "Domain",
    "Region",
    "Repeat"
}

PAIR_LIKE = {
    "Disulfide bond",
    "Cross-link"
}

all_cys_annotations = []
summary_rows = []

for _, row in tqdm(fasta_df.iterrows(), total=len(fasta_df)):
    acc = row["accession"]
    seq = row["sequence"]
    cys_pos = set(cys_positions(seq))

    entry = get_uniprot_json(acc)
    if entry is None:
        continue

    features = parse_features(entry)

    # keep track per protein
    protein_counts = {
        "accession": acc,
        "n_cys_total": len(cys_pos),
        "active_site_cys": 0,
        "binding_site_cys": 0,
        "metal_binding_cys": 0,
        "disulfide_cys": 0,
        "zinc_finger_region_cys": 0
    }

    seen = {
        "active_site_cys": set(),
        "binding_site_cys": set(),
        "metal_binding_cys": set(),
        "disulfide_cys": set(),
        "zinc_finger_region_cys": set()
    }

    for feat in features:
        ftype = feat["type"]
        desc = feat["description"] or ""
        start, end = feat["start"], feat["end"]
        overlap = {p for p in cys_pos if start <= p <= end}
        if not overlap:
            continue

        # Active site cysteines
        if ftype == "Active site":
            seen["active_site_cys"].update(overlap)

        # Generic binding sites
        if ftype == "Binding site":
            seen["binding_site_cys"].update(overlap)

        # Metal binding cysteines (nice for Zn-bound cysteines)
        if ftype == "Metal binding":
            seen["metal_binding_cys"].update(overlap)

        # Disulfide bond cysteines
        if ftype == "Disulfide bond":
            seen["disulfide_cys"].update(overlap)

        # Zinc finger region cysteines
        if ftype == "Zinc finger":
            seen["zinc_finger_region_cys"].update(overlap)

        for p in sorted(overlap):
            all_cys_annotations.append({
                "accession": acc,
                "cys_position": p,
                "feature_type": ftype,
                "feature_description": desc,
                "feature_start": start,
                "feature_end": end
            })

    for k in seen:
        protein_counts[k] = len(seen[k])

    summary_rows.append(protein_counts)

    time.sleep(0.05)  # polite pacing

annot_df = pd.DataFrame(all_cys_annotations).drop_duplicates()
summary_df = pd.DataFrame(summary_rows)

print("\n=== Global cysteine annotation totals ===")
for col in [
    "active_site_cys",
    "binding_site_cys",
    "metal_binding_cys",
    "disulfide_cys",
    "zinc_finger_region_cys"
]:
    print(f"{col}: {int(summary_df[col].sum())}")

annot_df.to_csv("/content/cysteine_feature_annotations_long.csv", index=False)
summary_df.to_csv("/content/cysteine_feature_annotations_summary.csv", index=False)

# ============================================
# 3) Global summary
# ============================================
global_summary = {
    "total_proteins_in_fasta": total_proteins,
    "proteins_with_cys": proteins_with_cys,
    "total_cysteines_in_fasta": total_cys,
    "annotated_active_site_cys": int(summary_df["active_site_cys"].sum()),
    "annotated_binding_site_cys": int(summary_df["binding_site_cys"].sum()),
    "annotated_metal_binding_cys": int(summary_df["metal_binding_cys"].sum()),
    "annotated_disulfide_cys": int(summary_df["disulfide_cys"].sum()),
    "cys_within_zinc_finger_regions": int(summary_df["zinc_finger_region_cys"].sum())
}

pd.Series(global_summary)
