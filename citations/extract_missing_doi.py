import pandas as pd

# Read the master CSV
print("Reading PINN_MRI_master.csv...")
df = pd.read_csv("PINN_MRI_master.csv")

# Identify rows with missing DOI
print("Filtering rows with missing DOI...")
missing_doi = df[(df["doi"].isna()) | (df["doi"] == "")].copy()
print(f"  -> Found {len(missing_doi)} records with missing DOI")

# Select only the needed fields
output_fields = ["work_id", "title", "authors", "year", "venue", "source_dbs"]
output_df = missing_doi[output_fields].copy()

# Save to new CSV
print("Saving to PINN_MRI_missing_doi.csv...")
output_df.to_csv("PINN_MRI_missing_doi.csv", encoding="utf-8", index=False)
print(f"  -> Saved {len(output_df)} records to PINN_MRI_missing_doi.csv")

print("\nDone!")
