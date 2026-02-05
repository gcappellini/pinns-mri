import pandas as pd
import sys

# Read the master CSV
print("Reading PINN_MRI_master.csv...")
df = pd.read_csv("PINN_MRI_master.csv")
print(f"  -> Total records: {len(df)}")

# ============================================================================
# 1. Verify all entries have DOI
# ============================================================================
print("\nVerifying all entries have DOI...")
missing_doi = df[(df["doi"].isna()) | (df["doi"] == "")]
if len(missing_doi) > 0:
    print(f"  ❌ ERROR: Found {len(missing_doi)} entries without DOI!")
    print("\nEntries without DOI:")
    print(missing_doi[["work_id", "title", "venue", "source_dbs"]])
    sys.exit(1)
else:
    print(f"  ✓ All {len(df)} entries have DOI")

# ============================================================================
# 2. Find entries with same DOI
# ============================================================================
print("\nFinding duplicate DOIs...")
# Normalize DOI for comparison (lowercase)
df["doi_normalized"] = df["doi"].str.lower().str.strip()

# Find duplicates
duplicate_dois = df[df.duplicated(subset=["doi_normalized"], keep=False)]
if len(duplicate_dois) > 0:
    print(f"  -> Found {len(duplicate_dois)} entries with duplicate DOIs")
    
    # Group by DOI to see duplicates
    dup_groups = duplicate_dois.groupby("doi_normalized")
    print(f"  -> {len(dup_groups)} unique DOIs have duplicates\n")
    
    # Show summary of duplicates
    for doi, group in dup_groups:
        print(f"  DOI: {doi}")
        for _, row in group.iterrows():
            print(f"    - {row['work_id']}: {row['venue']} ({row['source_dbs']})")
else:
    print("  -> No duplicate DOIs found")

# ============================================================================
# 3. Among duplicates, delete entries with venue "arXiv"
# ============================================================================
print("\nRemoving arXiv preprints that have published versions...")

# Strategy: For each DOI, if there are duplicates, keep non-arXiv version(s)
# and remove arXiv versions
deduplicated_df = df.copy()
removed_count = 0

# Group by normalized DOI
for doi_norm, group in deduplicated_df.groupby("doi_normalized"):
    if len(group) > 1:  # Duplicate DOI
        # Check if there's at least one non-arXiv version
        non_arxiv = group[group["venue"] != "arXiv"]
        arxiv_versions = group[group["venue"] == "arXiv"]
        
        if len(non_arxiv) > 0 and len(arxiv_versions) > 0:
            # Remove arXiv versions
            for idx in arxiv_versions.index:
                work_id = deduplicated_df.loc[idx, "work_id"]
                print(f"  Removing {work_id} (arXiv) - published as DOI: {doi_norm}")
                deduplicated_df = deduplicated_df.drop(idx)
                removed_count += 1

print(f"  -> Removed {removed_count} arXiv preprints with published versions")
print(f"  -> Remaining records: {len(deduplicated_df)}")

# ============================================================================
# 4. Verify no duplicated DOIs remain
# ============================================================================
print("\nVerifying no duplicate DOIs remain...")
deduplicated_df["doi_normalized"] = deduplicated_df["doi"].str.lower().str.strip()
remaining_duplicates = deduplicated_df[deduplicated_df.duplicated(subset=["doi_normalized"], keep=False)]

if len(remaining_duplicates) > 0:
    print(f"  ❌ ERROR: Still have {len(remaining_duplicates)} duplicate entries!")
    print("\nRemaining duplicates:")
    for doi, group in remaining_duplicates.groupby("doi_normalized"):
        print(f"\n  DOI: {doi}")
        for _, row in group.iterrows():
            print(f"    - {row['work_id']}: {row['title'][:50]}... ({row['venue']})")
    sys.exit(1)
else:
    print(f"  ✓ No duplicate DOIs remain")

# ============================================================================
# 5. Save to new deduplicated CSV
# ============================================================================
print("\nSaving deduplicated data...")
# Drop the temporary normalized DOI column
deduplicated_df = deduplicated_df.drop(columns=["doi_normalized"])

# Save to new file
output_file = "PINN_MRI_master_deduplicated.csv"
deduplicated_df.to_csv(output_file, encoding="utf-8", index=False)
print(f"  -> Saved {len(deduplicated_df)} records to {output_file}")

print("\n✓ Deduplication complete!")
print(f"  Original: {len(df)} records")
print(f"  Removed: {removed_count} arXiv duplicates")
print(f"  Final: {len(deduplicated_df)} records")
