import pandas as pd

# Load the merged CSV
csv_path = 'PINN_MRI_master_merged.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("DESCRIPTIVE STATISTICS - PINN MRI MASTER MERGED")
print("=" * 80)
print(f"\nTotal entries: {len(df)}\n")

# 1. Counts by year
print("\n" + "=" * 80)
print("COUNTS BY YEAR")
print("=" * 80)
year_counts = df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"{year}: {count}")
print(f"Total: {year_counts.sum()}")

# 2. Counts by mri_type
print("\n" + "=" * 80)
print("COUNTS BY MRI TYPE")
print("=" * 80)
mri_counts = df['mri_type'].value_counts().sort_values(ascending=False)
for mri_type, count in mri_counts.items():
    print(f"{mri_type}: {count}")
print(f"Total: {mri_counts.sum()}")

# 3. Counts by technical_task
print("\n" + "=" * 80)
print("COUNTS BY TECHNICAL TASK")
print("=" * 80)
task_counts = df['technical_task'].value_counts().sort_values(ascending=False)
for task, count in task_counts.items():
    print(f"{task}: {count}")
print(f"Total: {task_counts.sum()}")

# 4. Counts by physics_model
print("\n" + "=" * 80)
print("COUNTS BY PHYSICS MODEL")
print("=" * 80)
physics_counts = df['physics_model'].value_counts().sort_values(ascending=False)
for physics, count in physics_counts.items():
    print(f"{physics}: {count}")
print(f"Total: {physics_counts.sum()}")

# 5. Counts by architecture_class
print("\n" + "=" * 80)
print("COUNTS BY ARCHITECTURE CLASS")
print("=" * 80)
arch_counts = df['architecture_class'].value_counts().sort_values(ascending=False)
for arch, count in arch_counts.items():
    print(f"{arch}: {count}")
print(f"Total: {arch_counts.sum()}")

# 6. Counts by clinical_research_context
print("\n" + "=" * 80)
print("COUNTS BY CLINICAL RESEARCH CONTEXT")
print("=" * 80)
context_counts = df['clinical_research_context'].value_counts().sort_values(ascending=False)
for context, count in context_counts.items():
    print(f"{context}: {count}")
print(f"Total: {context_counts.sum()}")

# 7. Counts by data_type
print("\n" + "=" * 80)
print("COUNTS BY DATA TYPE")
print("=" * 80)
data_counts = df['data_type'].value_counts().sort_values(ascending=False)
for data_type, count in data_counts.items():
    print(f"{data_type}: {count}")
print(f"Total: {data_counts.sum()}")

print("\n" + "=" * 80)