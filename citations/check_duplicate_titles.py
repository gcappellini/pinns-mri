import pandas as pd
import re

# Read the deduplicated master CSV
print("Reading PINN_MRI_master_deduplicated.csv...")
df = pd.read_csv("PINN_MRI_master_deduplicated.csv")
print(f"  -> Total records: {len(df)}")

# Normalize titles for comparison
print("\nNormalizing titles...")
def normalize_title(title):
    """Normalize title: lowercase, remove punctuation, collapse spaces."""
    if pd.isna(title) or not isinstance(title, str):
        return ""
    title = title.lower()
    # Remove punctuation
    title = re.sub(r'[^\w\s]', '', title)
    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title

df["title_normalized"] = df["title"].apply(normalize_title)

# Find entries with duplicate titles
print("Checking for identical titles...")
duplicate_titles = df[df.duplicated(subset=["title_normalized"], keep=False)]

if len(duplicate_titles) > 0:
    print(f"\n⚠️  Found {len(duplicate_titles)} entries with duplicate titles!")
    
    # Group by normalized title to show duplicates
    dup_groups = duplicate_titles.groupby("title_normalized")
    print(f"  -> {len(dup_groups)} unique titles have duplicates\n")
    
    # Show details of each duplicate group
    for title_norm, group in dup_groups:
        print(f"\n{'='*80}")
        print(f"Title (normalized): {title_norm[:70]}...")
        print(f"{'='*80}")
        for _, row in group.iterrows():
            print(f"\n  work_id: {row['work_id']}")
            print(f"  title: {row['title'][:70]}...")
            print(f"  authors: {row['authors']}")
            print(f"  year: {row['year']}")
            print(f"  venue: {row['venue']}")
            print(f"  doi: {row['doi']}")
            print(f"  source_dbs: {row['source_dbs']}")
    
    # Save duplicate titles to a CSV for review
    output_file = "PINN_MRI_duplicate_titles.csv"
    duplicate_titles_output = duplicate_titles.drop(columns=["title_normalized"]).sort_values(["title", "year"])
    duplicate_titles_output.to_csv(output_file, encoding="utf-8", index=False)
    print(f"\n  -> Saved duplicate titles to {output_file}")
else:
    print("  ✓ No duplicate titles found")

print("\nDone!")
