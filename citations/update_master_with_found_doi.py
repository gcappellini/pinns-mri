import pandas as pd

# Read the master CSV and found DOI CSV
print("Reading PINN_MRI_master.csv...")
master_df = pd.read_csv("PINN_MRI_master.csv")

print("Reading PINN_MRI_found_doi.csv...")
found_doi_df = pd.read_csv("PINN_MRI_found_doi.csv")

# Create a lookup dictionary from work_id to doi_found
doi_lookup = dict(zip(found_doi_df["work_id"], found_doi_df["doi_found"]))

print(f"Found {len(doi_lookup)} DOIs to merge...")

# Update the master CSV with found DOIs where DOI is missing or empty
updated_count = 0
for work_id, new_doi in doi_lookup.items():
    # Find the row with this work_id
    mask = master_df["work_id"] == work_id
    if mask.any():
        # Check if current DOI is missing or empty
        current_doi = master_df.loc[mask, "doi"].values[0]
        if pd.isna(current_doi) or current_doi == "":
            master_df.loc[mask, "doi"] = new_doi
            updated_count += 1
        else:
            print(f"  {work_id}: Already has DOI {current_doi}, skipping")

print(f"  -> Updated {updated_count} entries")

# Save the updated master CSV
print("Saving updated PINN_MRI_master.csv...")
master_df.to_csv("PINN_MRI_master.csv", encoding="utf-8", index=False)
print(f"  -> Saved {len(master_df)} records to PINN_MRI_master.csv")

print("\nDone!")
