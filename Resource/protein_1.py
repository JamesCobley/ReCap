import pandas as pd
import requests
import time
from tqdm.auto import tqdm
summary_path = "/content/cysteine_feature_annotations_summary.csv"
summary_df = pd.read_csv(summary_path)

req = {
    "accession",
    "n_cys_total",
    "active_site_cys",
    "binding_site_cys",
    "disulfide_cys",
    "zinc_finger_region_cys",
}
missing = req - set(summary_df.columns)
if missing:
    raise ValueError(f"Missing columns in summary_df: {missing}. Found: {summary_df.columns.tolist()}")

protein_base = (
    summary_df[
        [
            "accession",
            "n_cys_total",
            "active_site_cys",
            "binding_site_cys",
            "disulfide_cys",
            "zinc_finger_region_cys",
        ]
    ]
    .drop_duplicates("accession")
    .copy()
)

# keep all proteins that contain >=1 cysteine
protein_base = protein_base[protein_base["n_cys_total"] > 0].copy()

protein_base["has_any_functional_cys"] = (
    protein_base[["active_site_cys", "binding_site_cys", "disulfide_cys", "zinc_finger_region_cys"]]
    .sum(axis=1) > 0
)

print("Cysteine-containing proteins:", len(protein_base))
protein_base.head()

def get_uniprot_json(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None

def parse_gene_primary(entry):
    genes = entry.get("genes", [])
    if not genes:
        return None
    gene = genes[0]
    gene_name = gene.get("geneName", {})
    if isinstance(gene_name, dict):
        return gene_name.get("value")
    return None

def parse_protein_name(entry):
    pdsc = entry.get("proteinDescription", {})
    rec = pdsc.get("recommendedName", {})
    full = rec.get("fullName", {})
    if isinstance(full, dict):
        return full.get("value")
    return None

def parse_organism(entry):
    return entry.get("organism", {}).get("scientificName")

def parse_reviewed(entry):
    entry_type = str(entry.get("entryType", ""))
    return ("Swiss-Prot" in entry_type) or ("reviewed" in entry_type.lower())

def parse_uniprot_xrefs(entry):
    """
    Returns:
      go_rows: list of dicts
      kegg_rows: list of dicts
      string_rows: list of dicts
    """
    xrefs = entry.get("uniProtKBCrossReferences", []) or []

    go_rows = []
    kegg_rows = []
    string_rows = []

    for xr in xrefs:
        db = xr.get("database")
        xid = xr.get("id")
        props = xr.get("properties", []) or []

        prop_dict = {}
        for p in props:
            key = p.get("key")
            val = p.get("value")
            if key:
                prop_dict[key] = val

        # ---------------- GO ----------------
        if db == "GO":
            # UniProt GO cross-references typically include GoTerm like:
            # P:mitochondrial matrix
            # F:ATP binding
            # C:mitochondrion
            goterm = prop_dict.get("GoTerm", "")
            goevidence = prop_dict.get("GoEvidenceType", "")

            go_class = None
            go_term_name = goterm

            if isinstance(goterm, str) and ":" in goterm:
                prefix, rest = goterm.split(":", 1)
                go_term_name = rest.strip()
                go_class = {
                    "C": "Cellular Component",
                    "P": "Biological Process",
                    "F": "Molecular Function",
                }.get(prefix, prefix)

            go_rows.append({
                "annotation_source": "GO",
                "annotation_class": go_class,
                "annotation_id": xid,
                "annotation_name": go_term_name,
                "evidence": goevidence,
                "source_note": "UniProt GO cross-reference"
            })

        # ---------------- KEGG ----------------
        elif db == "KEGG":
            # Often the ID itself is something like mmu:xxxx or just a KEGG-linked identifier
            kegg_rows.append({
                "annotation_source": "KEGG",
                "annotation_class": "Pathway/Xref",
                "annotation_id": xid,
                "annotation_name": xid,
                "evidence": None,
                "source_note": "UniProt KEGG cross-reference"
            })

        # ---------------- STRING ----------------
        elif db == "STRING":
            string_rows.append({
                "annotation_source": "STRING_XREF",
                "annotation_class": "Network ID",
                "annotation_id": xid,
                "annotation_name": xid,
                "evidence": None,
                "source_note": "UniProt STRING cross-reference"
            })

    return go_rows, kegg_rows, string_rows
  master_rows = []
annot_rows = []
failed = []

for _, row in tqdm(protein_base.iterrows(), total=len(protein_base)):
    acc = str(row["accession"])
    entry = get_uniprot_json(acc)

    if entry is None:
        failed.append(acc)
        continue

    gene_primary = parse_gene_primary(entry)
    protein_name = parse_protein_name(entry)
    organism = parse_organism(entry)
    reviewed = parse_reviewed(entry)

    go_rows, kegg_rows, string_rows = parse_uniprot_xrefs(entry)

    # protein-level master row
    master_rows.append({
        "accession": acc,
        "gene_primary": gene_primary,
        "protein_name": protein_name,
        "organism": organism,
        "reviewed": reviewed,
        "n_cys_total": row["n_cys_total"],
        "active_site_cys": row["active_site_cys"],
        "binding_site_cys": row["binding_site_cys"],
        "disulfide_cys": row["disulfide_cys"],
        "zinc_finger_region_cys": row["zinc_finger_region_cys"],
        "has_any_functional_cys": bool(row["has_any_functional_cys"]),
        "n_go_terms": len(go_rows),
        "n_kegg_xrefs": len(kegg_rows),
        "n_string_xrefs": len(string_rows),
    })

    # annotation rows
    for r in go_rows + kegg_rows + string_rows:
        annot_rows.append({
            "accession": acc,
            "gene_primary": gene_primary,
            "protein_name": protein_name,
            **r
        })

    time.sleep(0.05)

protein_master = pd.DataFrame(master_rows).drop_duplicates("accession")
protein_annotations_long = pd.DataFrame(annot_rows).drop_duplicates()

print("protein_master:", protein_master.shape)
print("protein_annotations_long:", protein_annotations_long.shape)
print("failed:", len(failed))

print("\nAnnotation source counts:")
print(protein_annotations_long["annotation_source"].value_counts(dropna=False))

print("\nGO class counts:")
print(
    protein_annotations_long.loc[
        protein_annotations_long["annotation_source"] == "GO",
        "annotation_class"
    ].value_counts(dropna=False)
)
protein_annotations_long["annotation_group"] = protein_annotations_long["annotation_class"].map({
    "Cellular Component": "location",
    "Biological Process": "process",
    "Molecular Function": "molecular_function",
    "Pathway/Xref": "pathway",
    "Network ID": "network"
}).fillna(protein_annotations_long["annotation_class"])

protein_annotations_long.head()
protein_master.to_csv("/content/protein_master_all_cys.csv", index=False)
protein_annotations_long.to_csv("/content/protein_annotations_long_all_cys.csv", index=False)

if failed:
    pd.DataFrame({"accession": failed}).to_csv("/content/protein_annotation_failed_accessions.csv", index=False)

with pd.ExcelWriter("/content/redox_annotation_resource_all_cys_v1.xlsx", engine="openpyxl") as writer:
    protein_master.to_excel(writer, sheet_name="protein_master", index=False)
    protein_annotations_long.to_excel(writer, sheet_name="annotations_long", index=False)
    if failed:
        pd.DataFrame({"accession": failed}).to_excel(writer, sheet_name="failed_accessions", index=False)

print("Saved:")
print("/content/protein_master_all_cys.csv")
print("/content/protein_annotations_long_all_cys.csv")
print("/content/redox_annotation_resource_all_cys_v1.xlsx")
if failed:
    print("/content/protein_annotation_failed_accessions.csv")
