import pandas as pd
import webbrowser
import time

# Load MASTER.csv
df = pd.read_csv('MASTER.csv')

# Filter for entries with DOI
df_with_doi = df[df['doi'].notna()]

print(f"Found {len(df_with_doi)} entries with DOIs")
print(f"Opening DOIs in browser windows...\n")

# Open each DOI in a browser window
for idx, (_, row) in enumerate(df_with_doi.iterrows(), 1):
    work_id = row['work_id']
    doi = row['doi']
    title = row['title'][:60] + ('...' if len(row['title']) > 60 else '')
    
    # Construct DOI URL
    doi_url = f"https://doi.org/{doi}"
    
    print(f"{idx}. [{work_id}] {title}")
    print(f"   URL: {doi_url}")
    
    # Open in browser
    webbrowser.open(doi_url)
    
    # Small delay to avoid overwhelming the browser
    time.sleep(0.5)

print(f"\nDone! Opened {len(df_with_doi)} DOI links.")
