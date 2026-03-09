import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = r'C:\Users\visha\OneDrive\Desktop\BA PROJECT\Indigo_Collapse_Analysis\outputs\01_validation\01_cleaned_data_no_pii.csv'
OUTPUT_DIR = r'C:\Users\visha\OneDrive\Desktop\BA PROJECT\Indigo_Collapse_Analysis\outputs\02_analysis'

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MULTICOLLINEARITY CHECK & COMPOSITE SCORE CREATION")
print("=" * 80)

# Load cleaned data
df = pd.read_csv(INPUT_FILE)
print(f"\n✓ Loaded data: {df.shape}")

# ============================================================================
# MAP ACTUAL COLUMN NAMES (from your questionnaire)
# ============================================================================

# Define the actual column names with full question text
emotion_actual_items = {
    'Anger': 'I experienced anger regarding the flight disruption.  ',
    'Stress': 'I experienced stress or anxiety while waiting for updates.  ',
    'Frustration': 'I experienced frustration with how the situation was managed.  '
}

trust_actual_items = {
    'Reliable': 'The airline is generally reliable.  ',
    'Competence': 'The airline demonstrates operational competence.  ',
    'Integrity': 'The airline acted with integrity during the situation.  '
}

emotion_heard_items = {
    'Concerned': 'Hearing about the disruptions made me feel concerned for affected passengers.  ',
    'Influence_Flight': 'Hearing about the disruptions influenced my perception of flying with this airline.  ',
    'Severity': 'The disruptions appeared significant in scale.  '
}

trust_heard_items = {
    'Reliability': 'The event influenced my perception of the airline\'s reliability.  ',
    'Trust': 'The event influenced my level of trust in the airline.  ',
    'Planning': 'The event influenced my perception of the airline\'s operational planning.  '
}

choice_items = {
    'Choice_Influence': 'The event would influence my choice of airline for a similar trip.  ',
    'Reconsider': 'I would reconsider flying with this airline in the near future.  ',
    'Switch': 'I am likely to switch airlines based on negative information.  '
}

# ============================================================================
# STEP 1: MULTICOLLINEARITY CHECK (Using Correlations)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: MULTICOLLINEARITY CHECK (Correlation-Based)")
print("=" * 80)

scales = {
    'Emotion (Actual)': emotion_actual_items,
    'Trust (Actual)': trust_actual_items,
    'Emotion (Heard)': emotion_heard_items,
    'Trust (Heard)': trust_heard_items,
    'Choice Intent (Heard)': choice_items
}

multicollinearity_report = []

for scale_name, items_dict in scales.items():
    print(f"\n{scale_name}:")
    print("-" * 70)
    
    # Get actual column names
    actual_cols = [col for col in items_dict.values() if col in df.columns]
    
    if not actual_cols:
        print(f"⚠ No columns found for this scale")
        print(f"  Looking for: {list(items_dict.values())}")
        continue
    
    # Get data for this scale
    scale_data = df[actual_cols].dropna()
    
    if len(scale_data) < 2:
        print(f"⚠ Not enough data points")
        continue
    
    # Calculate correlation matrix
    print(f"\nInter-Item Correlations ({len(actual_cols)} items, n={len(scale_data)}):")
    corr_matrix = scale_data.corr()
    print(corr_matrix.round(3))
    
    # Check for high correlations
    max_corr = 0
    high_corr = []
    for i in range(len(actual_cols)):
        for j in range(i+1, len(actual_cols)):
            corr_val = corr_matrix.iloc[i, j]
            max_corr = max(max_corr, abs(corr_val))
            if abs(corr_val) > 0.85:
                high_corr.append((actual_cols[i][:30], actual_cols[j][:30], corr_val))
    
    if high_corr:
        print("\n⚠ High correlations (> 0.85):")
        for item1, item2, corr in high_corr:
            print(f"  r = {corr:.3f}")
    else:
        print("\n✅ No high correlations (all < 0.85)")
    
    multicollinearity_report.append({
        'Scale': scale_name,
        'N_Items': len(actual_cols),
        'N_Cases': len(scale_data),
        'Max_Correlation': round(max_corr, 3),
        'Status': 'ACCEPTABLE' if max_corr < 0.85 else 'WARNING'
    })

# Save multicollinearity report
if multicollinearity_report:
    df_multicollinearity = pd.DataFrame(multicollinearity_report)
    df_multicollinearity.to_csv(
        os.path.join(OUTPUT_DIR, '01_multicollinearity_check.csv'), 
        index=False
    )
    print("\n✓ Saved: 01_multicollinearity_check.csv")

# ============================================================================
# STEP 2: CREATE COMPOSITE SCORES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING COMPOSITE SCORES")
print("=" * 80)

df_composite = df.copy()

composite_scales = {
    'Emotion_Actual': emotion_actual_items,
    'Trust_Actual': trust_actual_items,
    'Emotion_Heard': emotion_heard_items,
    'Trust_Heard': trust_heard_items,
    'Choice_Intent_Heard': choice_items
}

composite_summary = []

for score_name, items_dict in composite_scales.items():
    # Get actual column names
    actual_cols = [col for col in items_dict.values() if col in df_composite.columns]
    
    if not actual_cols:
        print(f"\n⚠ Skipping {score_name} - Columns not found")
        continue
    
    # Calculate composite score (mean of items)
    df_composite[score_name] = df_composite[actual_cols].mean(axis=1)
    
    # Get statistics
    valid_n = df_composite[score_name].notna().sum()
    mean = df_composite[score_name].mean()
    std = df_composite[score_name].std()
    min_val = df_composite[score_name].min()
    max_val = df_composite[score_name].max()
    
    print(f"\n✓ {score_name}:")
    print(f"  Items: {len(actual_cols)}")
    print(f"  N = {valid_n}")
    print(f"  Mean = {mean:.2f}, SD = {std:.2f}")
    print(f"  Range = {min_val:.2f} - {max_val:.2f}")
    
    composite_summary.append({
        'Score_Name': score_name,
        'Items': len(actual_cols),
        'N': valid_n,
        'Mean': round(mean, 2),
        'SD': round(std, 2),
        'Min': round(min_val, 2),
        'Max': round(max_val, 2)
    })

# Save composite scores
df_composite.to_csv(
    os.path.join(OUTPUT_DIR, '02_composite_scores.csv'), 
    index=False
)
print("\n✓ Saved: 02_composite_scores.csv")

# Save summary
if composite_summary:
    df_composite_summary = pd.DataFrame(composite_summary)
    df_composite_summary.to_csv(
        os.path.join(OUTPUT_DIR, '02_composite_scores_summary.csv'),
        index=False
    )
    print("✓ Saved: 02_composite_scores_summary.csv")

# ============================================================================
# STEP 3: CHECK FOR OUTLIERS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: OUTLIER DETECTION (Z-score method)")
print("=" * 80)

outlier_summary = []

for score_name in composite_scales.keys():
    if score_name not in df_composite.columns:
        continue
    
    valid_data = df_composite[score_name].dropna()
    if len(valid_data) < 2:
        continue
    
    # Calculate Z-scores
    z_scores = np.abs(zscore(valid_data))
    
    # Count outliers (|Z| > 3)
    outliers_count = (z_scores > 3).sum()
    outliers_mild = (z_scores > 2.5).sum()
    
    print(f"\n{score_name}:")
    print(f"  Extreme outliers (|Z| > 3): {outliers_count} ({100*outliers_count/len(valid_data):.1f}%)")
    print(f"  Mild outliers (|Z| > 2.5): {outliers_mild} ({100*outliers_mild/len(valid_data):.1f}%)")
    
    if outliers_count > 0:
        print(f"  ⚠ Consider investigating extreme outliers")
    else:
        print(f"  ✅ No extreme outliers")
    
    outlier_summary.append({
        'Score_Name': score_name,
        'Extreme_Outliers_Count': outliers_count,
        'Extreme_Outliers_Pct': round(100*outliers_count/len(valid_data), 2),
        'Mild_Outliers_Count': outliers_mild,
        'Mild_Outliers_Pct': round(100*outliers_mild/len(valid_data), 2)
    })

# Save outlier report
if outlier_summary:
    df_outliers = pd.DataFrame(outlier_summary)
    df_outliers.to_csv(
        os.path.join(OUTPUT_DIR, '03_outlier_detection.csv'),
        index=False
    )
    print("\n✓ Saved: 03_outlier_detection.csv")

# ============================================================================
# STEP 4: VISUALIZATION - COMPOSITE SCORES DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CREATING VISUALIZATIONS")
print("=" * 80)

try:
    # Create distribution plots
    scores_to_plot = [col for col in df_composite.columns if col in composite_scales.keys()]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Composite Scores Distribution', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, score_name in enumerate(scores_to_plot):
        ax = axes[idx]
        
        data = df_composite[score_name].dropna()
        ax.hist(data, bins=15, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add normal distribution curve
        mu, sigma = data.mean(), data.std()
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        ax.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2), 
                'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{score_name}\n(M={mu:.2f}, SD={sigma:.2f}, n={len(data)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Turn off unused subplots
    for i in range(len(scores_to_plot), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_composite_scores_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_composite_scores_distribution.png")
    plt.close()
except Exception as e:
    print(f"⚠ Could not create visualization: {str(e)}")

# ============================================================================
# STEP 5: CORRELATION BETWEEN SCALES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: SCALE-TO-SCALE CORRELATIONS")
print("=" * 80)

# Select only composite score columns
composite_cols = [col for col in df_composite.columns if col in composite_scales.keys()]

if len(composite_cols) > 1:
    corr_between_scales = df_composite[composite_cols].corr()
    
    print("\nCorrelation between composite scales:")
    print(corr_between_scales.round(3))
    
    # Visualize
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_between_scales, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
        plt.title('Correlations Between Composite Scales')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '05_scale_correlations.png'),
                    dpi=300, bbox_inches='tight')
        print("\n✓ Saved: 05_scale_correlations.png")
        plt.close()
    except Exception as e:
        print(f"⚠ Could not create correlation heatmap: {str(e)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PROCESSING COMPLETE ✓")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated Files:")
print("1. 01_multicollinearity_check.csv - Correlation analysis")
print("2. 02_composite_scores.csv - Composite scores for all respondents")
print("3. 02_composite_scores_summary.csv - Summary statistics")
print("4. 03_outlier_detection.csv - Outlier detection report")
print("5. 04_composite_scores_distribution.png - Distribution plots")
print("6. 05_scale_correlations.png - Correlation heatmap")

print("\n" + "=" * 80)
print("NEXT STEPS FOR HYPOTHESIS TESTING:")
print("=" * 80)
print("1. Use composite scores from: 02_composite_scores.csv")
print("2. Available for analysis:")
print("   ✓ Emotion_Actual")
print("   ✓ Emotion_Heard")
print("   ✓ Trust_Heard")
print("   ✓ Choice_Intent_Heard")
print("3. NOTE: Trust_Actual has low reliability - use with caution")
print("\n✓ Ready for hypothesis testing!")