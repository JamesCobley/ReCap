!pip -q install requests pandas tqdm openpyxl
import pandas as pd
import requests
import time
from tqdm.auto import tqdm

# --------------------------------------------------
# INPUT
# --------------------------------------------------
summary_path = "/content/cysteine_feature_annotations_summary.csv"
summary_df = pd.read_csv(summary_path)

if "accession" not in summary_df.columns:
    raise ValueError(f"'accession' not found. Columns are: {summary_df.columns.tolist()}")

protein_list = summary_df["accession"].dropna().astype(str).drop_duplicates().tolist()
print("Unique accessions to query:", len(protein_list))
import pandas as pd
import requests
import time
from tqdm.auto import tqdm

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
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

def get_feature_positions(feature):
    loc = feature.get("location", {}) or {}

    start = loc.get("start", {}).get("value")
    end = loc.get("end", {}).get("value")
    pos = loc.get("position", {}).get("value")

    if pos is not None:
        start = pos
        end = pos

    if start is None or end is None:
        return None, None

    return int(start), int(end)

def get_feature_evidence(feature):
    # UniProt JSON may contain evidences list
    evs = feature.get("evidences", []) or []
    if not evs:
        return None
    vals = []
    for e in evs:
        code = e.get("evidenceCode")
        source = e.get("source")
        if code and source:
            vals.append(f"{code}|{source}")
        elif code:
            vals.append(str(code))
        elif source:
            vals.append(str(source))
    return "; ".join(vals) if vals else None

def extract_site_rows(entry):
    accession = entry.get("primaryAccession")
    gene_primary = parse_gene_primary(entry)
    protein_name = parse_protein_name(entry)
    organism = parse_organism(entry)
    reviewed = parse_reviewed(entry)

    sequence = entry.get("sequence", {}).get("value", "")
    if not sequence:
        return []

    features = entry.get("features", []) or []
    rows = []

    keep_types = {"Active site", "Binding site", "Disulfide bond", "Zinc finger"}

    for f in features:
        ftype = f.get("type", "")
        if ftype not in keep_types:
            continue

        start, end = get_feature_positions(f)
        if start is None or end is None:
            continue

        desc = f.get("description", "")

        ligand_name = None
        ligand_id = None
        ligand_part_name = None

        ligand = f.get("ligand", {})
        if isinstance(ligand, dict):
            ligand_name = ligand.get("name")
            ligand_id = ligand.get("id")

        ligand_part = f.get("ligandPart", {})
        if isinstance(ligand_part, dict):
            ligand_part_name = ligand_part.get("name")

        evidence = get_feature_evidence(f)

        # We only keep cysteines overlapping the annotated feature span
        for pos in range(start, end + 1):
            if pos <= len(sequence) and sequence[pos - 1] == "C":
                rows.append({
                    "accession": accession,
                    "gene_primary": gene_primary,
                    "protein_name": protein_name,
                    "organism": organism,
                    "reviewed": reviewed,
                    "cys_position": pos,
                    "cys_label": f"Cys{pos}",
                    "feature_type": ftype,
                    "feature_description": desc,
                    "feature_start": start,
                    "feature_end": end,
                    "ligand_name": ligand_name,
                    "ligand_id": ligand_id,
                    "ligand_part_name": ligand_part_name,
                    "evidence": evidence,
                    "source_note": "UniProt positional feature annotation"
                })

    return rows
  # --------------------------------------------------
# BUILD SITE-LEVEL RESOURCE
# --------------------------------------------------
site_rows = []
failed = []

for acc in tqdm(protein_list):
    entry = get_uniprot_json(acc)
    if entry is None:
        failed.append(acc)
        continue

    try:
        rows = extract_site_rows(entry)
        site_rows.extend(rows)
    except Exception as e:
        failed.append(acc)

    time.sleep(0.05)

cysteine_functional_sites = pd.DataFrame(site_rows).drop_duplicates()

print("Site rows:", cysteine_functional_sites.shape)
print("Failed accessions:", len(failed))

if len(cysteine_functional_sites) > 0:
    print("\nFeature counts:")
    print(cysteine_functional_sites["feature_type"].value_counts())

    print("\nUnique cysteine sites:")
    print(
        cysteine_functional_sites[["accession", "cys_position"]]
        .drop_duplicates()
        .shape[0]
    )

cysteine_functional_sites.to_csv("/content/cysteine_functional_sites.csv", index=False)

if failed:
    pd.DataFrame({"accession": failed}).to_csv("/content/cysteine_functional_sites_failed_accessions.csv", index=False)

print("\nSaved:")
print("/content/cysteine_functional_sites.csv")
if failed:
    print("/content/cysteine_functional_sites_failed_accessions.csv")
