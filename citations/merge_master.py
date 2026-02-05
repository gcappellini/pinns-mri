import pandas as pd
import re
import os

# Define the output schema
SCHEMA = [
    "work_id", "title", "authors", "year", "publication_type", "venue",
    "volume", "issue", "doi", "pmid", "arxiv_id", "ieee_doc_id", "source_dbs"
]

def normalize_doi(doi):
    """Normalize DOI: lowercase, remove prefixes."""
    if pd.isna(doi) or not isinstance(doi, str):
        return None
    doi = doi.strip().lower()
    # Remove common prefixes
    doi = re.sub(r'^https://doi\.org/', '', doi)
    doi = re.sub(r'^doi:', '', doi)
    doi = doi.strip()
    return doi if doi else None

def normalize_title(title):
    """Normalize title: lowercase, remove punctuation, collapse spaces."""
    if pd.isna(title) or not isinstance(title, str):
        return None
    title = title.lower()
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title

# ============================================================================
# 1. READ AND PROCESS PUBMED
# ============================================================================
def process_pubmed():
    df = pd.read_csv("Citations_PubMed.csv")
    result = pd.DataFrame()
    result["title"] = df["Title"]
    result["authors"] = df["Authors"]
    result["year"] = pd.to_numeric(df["Publication Year"], errors="coerce").astype("Int64")
    result["publication_type"] = "journal"
    result["venue"] = df["Journal/Book"]
    result["volume"] = None  # Could parse from Citation, but optional
    result["issue"] = None
    result["doi"] = df["DOI"].apply(normalize_doi)
    result["pmid"] = df["PMID"]
    result["arxiv_id"] = None
    result["ieee_doc_id"] = None
    result["source_dbs"] = "PubMed"
    return result

# ============================================================================
# 2. READ AND PROCESS IEEE XPLORE
# ============================================================================
def process_ieee():
    df = pd.read_csv("Citations_IEEEXplore.csv")
    result = pd.DataFrame()
    result["title"] = df["Document Title"]
    result["authors"] = df["Authors"]
    result["year"] = pd.to_numeric(df["Publication Year"], errors="coerce").astype("Int64")
    # Determine publication type based on Publisher
    result["publication_type"] = df["Publisher"].apply(
        lambda x: "journal" if (isinstance(x, str) and "Journals" in x) else "conference"
    )
    result["venue"] = df["Publication Title"]
    result["volume"] = df["Volume"]
    result["issue"] = df["Issue"]
    result["doi"] = df["DOI"].apply(normalize_doi)
    result["pmid"] = None
    result["arxiv_id"] = None
    result["ieee_doc_id"] = df["Document Identifier"]
    result["source_dbs"] = "IEEEXplore"
    return result

# ============================================================================
# 3. READ AND PROCESS SCOPUS
# ============================================================================
def process_scopus():
    # Read the file to find the header row
    with open("Citations_Scopus.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Find the header row (contains "Publication Year,Document Title,...")
    header_idx = None
    for i, line in enumerate(lines):
        if "Publication Year" in line and "Document Title" in line:
            header_idx = i
            break
    
    if header_idx is None:
        # Fallback: assume standard header at line 0
        df = pd.read_csv("Citations_Scopus.csv")
    else:
        df = pd.read_csv("Citations_Scopus.csv", skiprows=header_idx)
    
    result = pd.DataFrame()
    result["title"] = df["Document Title"]
    result["authors"] = None
    result["year"] = pd.to_numeric(df["Publication Year"], errors="coerce").astype("Int64")
    result["publication_type"] = "journal"
    result["venue"] = df["Journal Title"]
    result["volume"] = df["Volume"]
    result["issue"] = df["Issue"]
    result["doi"] = None
    result["pmid"] = None
    result["arxiv_id"] = None
    result["ieee_doc_id"] = None
    result["source_dbs"] = "Scopus"
    return result

# ============================================================================
# 4. READ AND PROCESS ARXIV
# ============================================================================
def process_arxiv():
    df = pd.read_csv(
        "Citations_arXiv_short.csv",
        header=None,
        names=["raw_id", "title", "abstract", "published", "link"]
    )
    result = pd.DataFrame()
    result["title"] = df["title"]
    result["authors"] = None
    # Extract year from published (first 4 characters)
    result["year"] = df["published"].str[:4].astype("Int64")
    result["publication_type"] = "preprint"
    result["venue"] = "arXiv"
    result["volume"] = None
    result["issue"] = None
    result["doi"] = None
    result["pmid"] = None
    # Extract arXiv ID: e.g., "http://arxiv.org/abs/2102.07271v1" -> "2102.07271"
    result["arxiv_id"] = df["raw_id"].apply(lambda x: re.search(r'(\d+\.\d+)', x).group(1) if pd.notna(x) else None)
    result["ieee_doc_id"] = None
    result["source_dbs"] = "arXiv"
    return result

# ============================================================================
# 5. MAIN PIPELINE
# ============================================================================
def main():
    print("Reading and processing PubMed...")
    pubmed_df = process_pubmed()
    print(f"  -> {len(pubmed_df)} records")
    
    print("Reading and processing IEEE Xplore...")
    ieee_df = process_ieee()
    print(f"  -> {len(ieee_df)} records")
    
    print("Reading and processing Scopus...")
    scopus_df = process_scopus()
    print(f"  -> {len(scopus_df)} records")
    
    print("Reading and processing arXiv...")
    arxiv_df = process_arxiv()
    print(f"  -> {len(arxiv_df)} records")
    
    # ========================================================================
    # 3. Concatenate all DataFrames (skipping deduplication for now)
    # ========================================================================
    print("Concatenating all records...")
    final_df = pd.concat([pubmed_df, ieee_df, scopus_df, arxiv_df], ignore_index=True)
    print(f"  -> Total: {len(final_df)} records")
    
    # ========================================================================
    # 4. Assign work_id
    # ========================================================================
    print("Assigning work IDs...")
    final_df["work_id"] = [f"W{i+1:05d}" for i in range(len(final_df))]
    
    # ========================================================================
    # 5. Select and order final columns
    # ========================================================================
    final_df = final_df[SCHEMA].copy()
    
    # ========================================================================
    # 4. Save to CSV
    # ========================================================================
    print("Saving to PINN_MRI_master.csv...")
    final_df.to_csv("PINN_MRI_master.csv", encoding="utf-8", index=False)
    print(f"  -> Saved {len(final_df)} records to PINN_MRI_master.csv")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
