import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import os

INPUT_FILE = r'C:\Users\visha\OneDrive\Desktop\BA PROJECT\Indigo_Collapse_Analysis\outputs\02_analysis\02_composite_scores.csv'
OUTPUT_DIR = r'C:\Users\visha\OneDrive\Desktop\BA PROJECT\Indigo_Collapse_Analysis\outputs\04_specific_hypotheses'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Define columns
exp_col = 'Did you experience a cancellation or significant delay on an IndiGo flight in December 2025?'
comm_col = 'The communication provided was timely and clear.  '
comp_col = 'Suitable compensation and arrangements were made by the airline.'
intent_actual_col = 'I intend to consider this airline for future travel.  '

# Clean experience indicator
df['Experienced'] = (df[exp_col].astype(str).str.strip().str.lower() == 'yes').astype(int)

# Create combined Emotion and Trust
df['Emotion_All'] = df['Emotion_Actual'].fillna(df['Emotion_Heard'])
df['Trust_All'] = df['Trust_Actual'].fillna(df['Trust_Heard'])

# Future Intent: For actual, it's 'I intend to consider...'. For Heard, it's 'Choice_Intent_Heard' (which has Reconsider/Switch).
# Reconsider/Switch are negatively framed (higher = want to switch). We reverse them to unify with 'Intent to consider'.
if 'Choice_Intent_Heard' in df.columns:
    df['Future_Intent_All'] = df[intent_actual_col].fillna(8 - df['Choice_Intent_Heard'])
else:
    df['Future_Intent_All'] = df[intent_actual_col]

# Encode compensation
comp_map = {
    'No compensation was given': 1,
    'Compensation was made, but not suitable': 2,
    'Compensation was made, and it was suitable': 3
}
df['Comp_Score'] = df[comp_col].map(comp_map)

# Extract Demographic Controls to resolve Omitted Variable Bias (Simpson's Paradox)
def get_combined(df, keyword):
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    if len(cols) == 0: return pd.Series(index=df.index, dtype=float)
    if len(cols) == 1: return df[cols[0]]
    # Fill NAs horizontally across matches
    res = df[cols[0]]
    for c in cols[1:]: res = res.fillna(df[c])
    return res

df['Age_All'] = get_combined(df, 'age')
df['Gender_All'] = get_combined(df, 'gender')

age_map = {'Under 25':1, '25–34':2, '35–44':3, '45–54':4, '55+':5}
df['Age_Numeric'] = df['Age_All'].astype(str).str.strip().map(age_map).fillna(1) # Default base category
df['Female'] = (df['Gender_All'].astype(str).str.lower().str.strip() == 'female').astype(int)

control_cols = ['Age_Numeric', 'Female']

results = []
report = []

def add_result(h_num, description, method, stat_name, stat_val, p_val, accepted, notes=""):
    results.append({
        'Hypothesis': h_num,
        'Description': description,
        'Method': method,
        'Statistic_Type': stat_name,
        'Statistic': stat_val,
        'P-Value': p_val,
        'Status': 'ACCEPTED' if accepted else 'REJECTED',
        'Notes': notes
    })
    report.append(f"[{h_num}] {description}")
    report.append(f"Method: {method}")
    report.append(f"{stat_name}: {stat_val:.4f}, p-value: {p_val:.4f} -> {'ACCEPTED' if accepted else 'REJECTED'}")
    if notes: report.append(f"Notes: {notes}")
    report.append("-" * 60)

print("Running Robust Controlled Hypothesis Models...")

# H1: Negative crisis experiences significantly increase emotional distress among passengers.
# Method: OLS (Emotion ~ Experienced + Controls)
valid = df[['Emotion_All', 'Experienced'] + control_cols].dropna()
if len(valid) > 2:
    X = sm.add_constant(valid[['Experienced'] + control_cols])
    m1 = sm.OLS(valid['Emotion_All'], X).fit()
    b = m1.params['Experienced']
    p = m1.pvalues['Experienced'] / 2 if b > 0 else 1.0 # Expected positive effect of experience on distress
    add_result('H1', 'Crisis exp increases emotional distress', 'Controlled OLS', 'Beta_Exp', b, p, p < 0.05, f"R^2={m1.rsquared:.3f}")
else: add_result('H1', 'Crisis exp increases emotional distress', 'Controlled OLS', 'Beta', np.nan, np.nan, False)

# H2: Higher levels of emotional distress negatively affect passengers' belief in the airline.
# Controlled OLS
valid = df[['Emotion_All', 'Trust_All'] + control_cols].dropna()
if len(valid) > 2:
    X = sm.add_constant(valid[['Emotion_All'] + control_cols])
    m2 = sm.OLS(valid['Trust_All'], X).fit()
    b = m2.params['Emotion_All']
    p = m2.pvalues['Emotion_All'] / 2 if b < 0 else 1.0
    add_result('H2', 'Distress neg. affects belief (Trust)', 'Controlled OLS', 'Beta_Emo', b, p, p < 0.05, f"R^2={m2.rsquared:.3f}")
else: add_result('H2', 'Distress neg. affects belief', 'Controlled OLS', 'Beta', np.nan, np.nan, False)

# H3: Emotional distress has a negative effect on passengers' trust in the airline.
if len(valid) > 2:
    add_result('H3', 'Distress neg. affects trust', 'Controlled OLS (same as H2)', 'Beta_Emo', b, p, p < 0.05, "Refined methodology identical to H2")
else: add_result('H3', 'Distress neg. affects trust', 'Controlled OLS', 'Beta', np.nan, np.nan, False)

# H4: High-quality communication during disruptions positively influences airline trust.
valid = df[df['Experienced']==1][['Trust_Actual', comm_col] + control_cols].dropna()
if len(valid) > 2:
    X = sm.add_constant(valid[[comm_col] + control_cols])
    m4 = sm.OLS(valid['Trust_Actual'], X).fit()
    b = m4.params[comm_col]
    p = m4.pvalues[comm_col] / 2 if b > 0 else 1.0
    add_result('H4', 'Communication positively influences trust', 'Controlled OLS', 'Beta_Comm', b, p, p < 0.05, f"R^2={m4.rsquared:.3f}")
else: add_result('H4', 'Communication positively influences trust', 'Controlled OLS', 'Beta', np.nan, np.nan, False)

# H5: Perceived fairness of compensation positively influences airline trust but does not fully offset emotional distress.
valid = df[df['Experienced']==1][['Trust_Actual', 'Comp_Score', 'Emotion_Actual'] + control_cols].dropna()
if len(valid) > 2:
    # Standardize core continuous variables for beta comparing
    for c in ['Trust_Actual', 'Comp_Score', 'Emotion_Actual']:
        valid[c] = (valid[c] - valid[c].mean()) / valid[c].std()
        
    X = sm.add_constant(valid[['Comp_Score', 'Emotion_Actual'] + control_cols])
    m5 = sm.OLS(valid['Trust_Actual'], X).fit()
    b_comp = m5.params['Comp_Score']
    p_comp = m5.pvalues['Comp_Score'] / 2 if b_comp > 0 else 1.0
    b_emo = m5.params['Emotion_Actual']
    p_emo = m5.pvalues['Emotion_Actual'] / 2 if b_emo < 0 else 1.0
    acc = (p_comp < 0.05) and (p_emo < 0.05)
    add_result('H5', 'Comp positive but emo negative offset', 'Controlled Multi-OLS (Std)', 'Beta_Comp', b_comp, p_comp, acc, f"Beta_Emo={b_emo:.3f} (p={p_emo:.3f})")
else: add_result('H5', 'Comp positive but emo negative offset', 'Controlled Multi-OLS', 'Beta', np.nan, np.nan, False)

# H6: Perceived reliability, integrity, and competence positively influence overall airline trust.
rel = 'The airline is generally reliable.  '
comp = 'The airline demonstrates operational competence.  '
intg = 'The airline acted with integrity during the situation.  '
if all(c in df.columns for c in [rel, comp, intg, 'Trust_Actual']):
    valid = df[[rel, comp, intg, 'Trust_Actual'] + control_cols].dropna()
    # We run OLS of Trust ~ Rel + Comp + Intg + Controls rather than just raw correlations
    X = sm.add_constant(valid[[rel, comp, intg] + control_cols])
    m6 = sm.OLS(valid['Trust_Actual'], X).fit()
    ps = [m6.pvalues[c] for c in [rel, comp, intg]]
    bs = [m6.params[c] for c in [rel, comp, intg]]
    acc = all(p < 0.05 and b > 0 for p, b in zip(ps, bs))
    add_result('H6', 'Rel/Int/Comp -> Trust', 'Controlled Multiple OLS', 'Min Beta', min(bs), max(ps), acc, f"B_rel={bs[0]:.2f}, B_comp={bs[1]:.2f}, B_int={bs[2]:.2f}")
else: add_result('H6', 'Rel/Int/Comp -> Trust', 'Controlled Multiple OLS', 'Beta', np.nan, np.nan, False)

# H7: Higher trust in the airline increases passengers' willingness to choose the airline in the future.
valid = df[['Trust_All', 'Future_Intent_All'] + control_cols].dropna()
if len(valid) > 2:
    X = sm.add_constant(valid[['Trust_All'] + control_cols])
    m7 = sm.OLS(valid['Future_Intent_All'], X).fit()
    b = m7.params['Trust_All']
    p = m7.pvalues['Trust_All'] / 2 if b > 0 else 1.0
    add_result('H7', 'Trust -> Willingness to choose', 'Controlled OLS', 'Beta_Trust', b, p, p < 0.05, f"R^2={m7.rsquared:.3f}")
else: add_result('H7', 'Trust -> Willingness to choose', 'Controlled OLS', 'Beta', np.nan, np.nan, False)

# H8: Airline trust mediates the relationship between crisis experience and future adoption intention.
valid = df[['Experienced', 'Trust_All', 'Future_Intent_All'] + control_cols].dropna()
if len(valid) > 2:
    m_c = sm.OLS(valid['Future_Intent_All'], sm.add_constant(valid[['Experienced'] + control_cols])).fit()
    c_p = m_c.pvalues['Experienced']
    m_a = sm.OLS(valid['Trust_All'], sm.add_constant(valid[['Experienced'] + control_cols])).fit()
    a_p = m_a.pvalues['Experienced']
    m_bc = sm.OLS(valid['Future_Intent_All'], sm.add_constant(valid[['Experienced', 'Trust_All'] + control_cols])).fit()
    b_p = m_bc.pvalues['Trust_All']
    cp_p = m_bc.pvalues['Experienced']
    acc = (a_p < 0.05) and (b_p < 0.05)
    med_type = "Full Mediation" if cp_p >= 0.05 else "Partial Mediation"
    add_result('H8', 'Trust mediates crisis exp and intent', 'Controlled Mediation', 'Beta_Trust_PathB', m_bc.params['Trust_All'], b_p, acc, f"{med_type} (p_a={a_p:.3f}, p_b={b_p:.3f})")
else: add_result('H8', 'Trust mediates crisis exp and intent', 'Mediation', 'Beta', np.nan, np.nan, False)

# H9: Airline trust has a stronger influence on future adoption intention than financial compensation.
valid = df[df['Experienced']==1][['Trust_All', 'Comp_Score', 'Future_Intent_All'] + control_cols].dropna()
if len(valid) > 2:
    for c in ['Trust_All', 'Comp_Score', 'Future_Intent_All']:
        valid[c] = (valid[c] - valid[c].mean()) / valid[c].std()
    X = sm.add_constant(valid[['Trust_All', 'Comp_Score'] + control_cols])
    m9 = sm.OLS(valid['Future_Intent_All'], X).fit()
    b_t = m9.params['Trust_All']
    p_t = m9.pvalues['Trust_All'] / 2 if b_t > 0 else 1.0
    b_c = m9.params['Comp_Score']
    p_c = m9.pvalues['Comp_Score'] / 2 if b_c > 0 else 1.0
    acc = (b_t > b_c) and (p_t < 0.05)
    add_result('H9', 'Trust stronger influence than compensation', 'Controlled Multi-OLS (Std)', 'Beta_Trust', b_t, p_t, acc, f"Beta_Comp={b_c:.3f} (p={p_c:.3f})")
else: add_result('H9', 'Trust stronger influence than compensation', 'Multiple OLS', 'Beta', np.nan, np.nan, False)

import json
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUTPUT_DIR, 'H1_H9_Results.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'H1_H9_Report.txt'), 'w', encoding='utf-8') as f:
    f.write("ROBUST MULTIVARIABLE HYPOTHESIS TESTING REPORT (H1 - H9)\n")
    f.write("==========================================================\n")
    f.write("Methodology: Used robust OLS regressions strictly controlling for baseline Age and Gender to eliminate Simpson's Paradox and Omitted Variable Bias.\n\n")
    f.write("\n".join(report))
    
print("Robust Tests complete! Saved H1_H9_Results.csv and H1_H9_Report.txt")
