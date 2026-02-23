import pandas as pd

# Define the list of metric columns to analyze
metric_cols = [
    "context_clarity",
    "physics_model_clarity",
    "data_transparency",
    "baseline_comparison",
    "quantitative_evaluation",
    "validation_beyond_training",
    "robustness_or_uncertainty",
    "reproducibility_signal",
]

# Read MASTER.csv
df = pd.read_csv('MASTER.csv')

print("=" * 80)
print("METRICS QUALITY ASSESSMENT SUMMARY")
print("=" * 80)
print()

# Store results for summary DataFrame
summary_data = []

# Analyze each metric
for metric in metric_cols:
    # Total papers
    N = len(df)
    
    # Count each category
    met_count = (df[metric] == 1.0).sum()
    partial_count = (df[metric] == 0.5).sum()
    not_count = (df[metric] == 0.0).sum()
    
    # Calculate percentages
    met_pct = (met_count / N) * 100
    partial_pct = (partial_count / N) * 100
    not_pct = (not_count / N) * 100
    
    # Print summary for this metric
    print(f"Metric: {metric} (N = {N})")
    print(f"  met (1.0):      {met_count:2d} papers ({met_pct:5.1f}%)")
    print(f"  partial (0.5):  {partial_count:2d} papers ({partial_pct:5.1f}%)")
    print(f"  not met (0.0):  {not_count:2d} papers ({not_pct:5.1f}%)")
    print()
    
    # Store for summary DataFrame
    summary_data.append({
        'metric': metric,
        'N': N,
        'met_count': met_count,
        'met_pct': round(met_pct, 1),
        'partial_count': partial_count,
        'partial_pct': round(partial_pct, 1),
        'not_count': not_count,
        'not_pct': round(not_pct, 1)
    })

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save to CSV
summary_df.to_csv('MASTER_metrics_summary.csv', index=False)

print("=" * 80)
print(f"Summary saved to MASTER_metrics_summary.csv")
print("=" * 80)
