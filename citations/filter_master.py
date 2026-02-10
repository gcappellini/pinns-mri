import pandas as pd

# Read the master CSV
print("Reading PINN_MRI_master.csv...")
df = pd.read_csv("PINN_MRI_master.csv")

# # Identify rows with missing DOI
# print("Filtering rows with missing DOI...")
# missing_doi = df[(df["doi"].isna()) | (df["doi"] == "")].copy()
# print(f"  -> Found {len(missing_doi)} records with missing DOI")

# # Select only the needed fields
# output_fields = ["work_id", "title", "authors", "year", "venue", "source_dbs"]
# output_df = missing_doi[output_fields].copy()

# # Save to new CSV
# print("Saving to PINN_MRI_missing_doi.csv...")
# output_df.to_csv("PINN_MRI_missing_doi.csv", encoding="utf-8", index=False)
# print(f"  -> Saved {len(output_df)} records to PINN_MRI_missing_doi.csv")

# print("\nDone!")

# # Identify rows with publication_type "journal"
# print("Filtering rows with publication_type 'journal'...")
# journal = df[(df["publication_type"] == "journal")].copy()
# print(f"  -> Found {len(journal)} records with publication_type 'journal'")

# # Select only the needed fields
# output_df = journal.copy()

# # Save to new CSV
# print("Saving to PINN_MRI_journal.csv...")
# output_df.to_csv("PINN_MRI_journal.csv", encoding="utf-8", index=False)
# print(f"  -> Saved {len(output_df)} records to PINN_MRI_journal.csv")

# print("\nDone!\n")

# print DOIs of entries with work_id W00015, 16, 21, 26, 27, 30, 36, 37, 38, 40, 43, 44, 45, 52, 53, 86, 87, 109
target_work_ids = ["W00015", "W00016", "W00021", "W00026", "W00027", "W00030", "W00036", "W00037", "W00038",
                   "W00040", "W00043", "W00044", "W00045", "W00052", "W00053", "W00086", "W00087", "W00109"]
print("DOIs for target work_ids:")
for wid in target_work_ids:
    row = df[df["work_id"] == wid]
    if not row.empty:
        doi = row["doi"].values[0]
        print(f"  {wid}: {doi}")
    else:
        print(f"  {wid}: Not found in master CSV")