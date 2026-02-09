import pandas as pd
import requests
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configuration
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF_API = "https://api.crossref.org/works"
REQUEST_DELAY = 0.5  # seconds between requests

def fetch_pubmed_abstract(pmid):
    """Fetch abstract from PubMed using NCBI E-utilities API."""
    if pd.isna(pmid) or pmid == "":
        return None
    
    try:
        pmid_str = str(int(pmid)) if pd.notna(pmid) else None
        if not pmid_str:
            return None
            
        url = f"{PUBMED_API}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": pmid_str,
            "rettype": "abstract",
            "retmode": "text"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        text = response.text.strip()
        if text and len(text) > 50:  # Ensure we have meaningful content
            return text
    except Exception as e:
        print(f"  Error fetching PubMed abstract for {pmid}: {str(e)[:50]}")
    
    return None

def fetch_ieee_abstract(doi):
    """Fetch abstract from IEEE Xplore page."""
    if pd.isna(doi) or doi == "":
        return None
    
    try:
        # IEEE DOI format: 10.1109/...
        if "10.1109" not in str(doi).lower():
            return None
        
        # Try to construct IEEE Xplore URL from DOI
        url = f"https://ieeexplore.ieee.org/document/{doi.split('/')[-1]}/"
        
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for abstract in various common locations
        # Try meta description tag
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            abstract = meta_desc.get("content")
            if len(abstract) > 50:
                return abstract
        
        # Try abstract div
        abstract_div = soup.find("div", {"class": re.compile(r".*abstract.*", re.I)})
        if abstract_div:
            abstract_text = abstract_div.get_text(strip=True)
            if len(abstract_text) > 50:
                return abstract_text
        
        # Try p tag that might contain abstract
        for tag in soup.find_all("p"):
            text = tag.get_text(strip=True)
            if "abstract" in text.lower() and len(text) > 100:
                # Extract from " Abstract: text here" format
                match = re.search(r"abstract\s*:\s*(.+)", text, re.I)
                if match:
                    return match.group(1)[:500]
    except Exception as e:
        print(f"  Error fetching IEEE abstract for {doi}: {str(e)[:50]}")
    
    return None

def fetch_acm_abstract(doi):
    """Fetch abstract from ACM Digital Library page."""
    if pd.isna(doi) or doi == "":
        return None
    
    try:
        # ACM DOI format: 10.1145/...
        if "10.1145" not in str(doi).lower():
            return None
        
        url = f"https://dl.acm.org/doi/{doi}"
        
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Look for abstract section
        abstract_div = soup.find("div", {"class": re.compile(r".*abstract.*", re.I)})
        if abstract_div:
            abstract_text = abstract_div.get_text(strip=True)
            if len(abstract_text) > 50:
                return abstract_text
        
        # Look for meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            abstract = meta_desc.get("content")
            if len(abstract) > 50:
                return abstract
    except Exception as e:
        print(f"  Error fetching ACM abstract for {doi}: {str(e)[:50]}")
    
    return None

def fetch_crossref_abstract(doi):
    """Fetch abstract from CrossRef API."""
    if pd.isna(doi) or doi == "":
        return None
    
    try:
        url = f"{CROSSREF_API}/{doi}"
        
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        data = response.json()
        if "message" in data and "abstract" in data["message"]:
            abstract = data["message"]["abstract"]
            if abstract and len(abstract) > 50:
                return abstract
    except Exception as e:
        print(f"  Error fetching CrossRef abstract for {doi}: {str(e)[:50]}")
    
    return None

def fetch_abstract(row):
    """Fetch abstract for a single row using available methods."""
    # Try PubMed first if PMID is available
    if pd.notna(row.get("pmid")) and row["pmid"] != "":
        abstract = fetch_pubmed_abstract(row["pmid"])
        if abstract:
            return abstract
        time.sleep(REQUEST_DELAY)
    
    # Try DOI-specific sources
    if pd.notna(row.get("doi")) and row["doi"] != "":
        doi = str(row["doi"]).lower()
        
        # Try IEEE
        if "10.1109" in doi:
            abstract = fetch_ieee_abstract(row["doi"])
            if abstract:
                return abstract
            time.sleep(REQUEST_DELAY)
        
        # Try ACM
        if "10.1145" in doi:
            abstract = fetch_acm_abstract(row["doi"])
            if abstract:
                return abstract
            time.sleep(REQUEST_DELAY)
        
        # Try CrossRef (general fallback)
        abstract = fetch_crossref_abstract(row["doi"])
        if abstract:
            return abstract
        time.sleep(REQUEST_DELAY)
    
    return None

def main():
    print("Reading PINN_MRI_master.csv...")
    df = pd.read_csv("PINN_MRI_master.csv")
    print(f"  -> Loaded {len(df)} records")
    
    # Initialize abstract column if it doesn't exist
    if "abstract" not in df.columns:
        df["abstract"] = None
    
    # Count existing abstracts
    existing_count = df["abstract"].notna().sum()
    print(f"  -> {existing_count} records already have abstracts")
    
    # Fetch abstracts for rows that don't have them
    print("\nFetching abstracts...")
    total = len(df)
    fetched_count = 0
    
    for idx, row in df.iterrows():
        # Skip if already has abstract
        if pd.notna(row.get("abstract")) and row["abstract"] != "":
            continue
        
        # Progress indicator
        print(f"  [{idx+1}/{total}] {row['work_id']}: ", end="", flush=True)
        
        # Fetch abstract
        abstract = fetch_abstract(row)
        if abstract:
            df.at[idx, "abstract"] = abstract
            fetched_count += 1
            print(f"✓ ({len(abstract)} chars)")
        else:
            print("✗")
    
    print(f"\n  -> Fetched {fetched_count} new abstracts")
    print(f"  -> Total abstracts: {df['abstract'].notna().sum()}")
    
    # Save updated CSV
    output_file = "PINN_MRI_master_with_abstracts.csv"
    df.to_csv(output_file, encoding="utf-8", index=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
