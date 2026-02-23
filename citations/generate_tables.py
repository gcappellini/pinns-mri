import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Set font to support better rendering
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 8

# 1) Read MASTER.csv
df = pd.read_csv('MASTER.csv')

# 2) Select and rename columns for the table
columns_to_include = [
    'work_id',
    # 'quantitative_evaluation',
    'physics_model',
    'mri_type',
    'technical_task',
    'clinical_research_context'
]

table_df = df[columns_to_include].copy()                #[df['technical_task'] == 'other_task']

# Rename work_id to Work ID for display
table_df.columns = [
    'Work ID',
    # 'Quant Eval',
    'Physics Model',
    'MRI Type',
    'Technical Task',
    'Clinical Context'
]

# 3) Print the table
print("=" * 120)
print("TABLE: Work ID, Quantitative Evaluation, and Study Characteristics")
print("=" * 120)
print(table_df.to_string(index=False))
print("=" * 120)

# Save table as PNG
fig, ax = plt.subplots(figsize=(16, 20))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc='left',
    loc='center',
    colWidths=[0.08, 0.18, 0.15, 0.20, 0.25]
    # colWidths=[0.08, 0.10, 0.18, 0.15, 0.20, 0.25]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.5)

# Color header row
for i in range(len(table_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_df) + 1):
    for j in range(len(table_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')

plt.title('Quantitative Evaluation and Study Characteristics', 
          fontsize=14, fontweight='bold', pad=20)

plt.savefig('table_quantitative_evaluation.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.2)

print(f"\nTable saved as 'table_quantitative_evaluation.png'")
print(f"Total entries: {len(table_df)}")
