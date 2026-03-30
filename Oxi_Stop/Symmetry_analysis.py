# ============================================
# Exact identity / nonidentity analysis
# No tolerance: identity = exactly equal
# ============================================

import numpy as np
import pandas as pd

# Make sure the core columns exist
required = ['SiteKey', 'Mean_CTRL', 'Mean_REDCAP']
missing = [c for c in required if c not in stats_df.columns]
if missing:
    raise ValueError(f"Missing required columns in stats_df: {missing}")

df_id = stats_df.copy()

# Exact delta
df_id['Delta_Percent'] = df_id['Mean_REDCAP'] - df_id['Mean_CTRL']
df_id['Abs_Delta_Percent'] = df_id['Delta_Percent'].abs()

# --------------------------------------------
# Exact identity / nonidentity
# --------------------------------------------
df_id['Identity'] = df_id['Mean_REDCAP'] == df_id['Mean_CTRL']
df_id['NonIdentity'] = ~df_id['Identity']

n_total = len(df_id)
n_ident = int(df_id['Identity'].sum())
n_nonident = int(df_id['NonIdentity'].sum())

print("========== Exact identity summary ==========")
print(f"Total shared sites: {n_total}")
print(f"Identical (exactly equal): {n_ident} ({n_ident/n_total:.3%})")
print(f"Nonidentical (different): {n_nonident} ({n_nonident/n_total:.3%})")

# --------------------------------------------
# Identity subclasses
# --------------------------------------------
ident_df = df_id[df_id['Identity']].copy()

ident_0 = int(((ident_df['Mean_CTRL'] == 0) & (ident_df['Mean_REDCAP'] == 0)).sum())
ident_100 = int(((ident_df['Mean_CTRL'] == 100) & (ident_df['Mean_REDCAP'] == 100)).sum())
ident_partial = int(n_ident - ident_0 - ident_100)

print("\n========== Identity subclasses ==========")
print(f"Identical at 0%: {ident_0} ({ident_0/n_total:.3%} of all; {ident_0/n_ident:.3%} of identical)" if n_ident else "Identical at 0%: 0")
print(f"Identical at 100%: {ident_100} ({ident_100/n_total:.3%} of all; {ident_100/n_ident:.3%} of identical)" if n_ident else "Identical at 100%: 0")
print(f"Identical at same partial state: {ident_partial} ({ident_partial/n_total:.3%} of all; {ident_partial/n_ident:.3%} of identical)" if n_ident else "Identical at same partial state: 0")

# --------------------------------------------
# Nonidentity direction
# Positive delta = REDCAP higher
# Negative delta = CTRL higher
# --------------------------------------------
nonident_df = df_id[df_id['NonIdentity']].copy()

up_redcap_df = nonident_df[nonident_df['Delta_Percent'] > 0].copy()
down_redcap_df = nonident_df[nonident_df['Delta_Percent'] < 0].copy()

n_up_redcap = len(up_redcap_df)
n_down_redcap = len(down_redcap_df)

sum_up_redcap = up_redcap_df['Delta_Percent'].sum()
sum_down_redcap = down_redcap_df['Delta_Percent'].sum()  # negative number
net_delta_from_nonident = nonident_df['Delta_Percent'].sum()

mean_ctrl_all = df_id['Mean_CTRL'].mean()
mean_redcap_all = df_id['Mean_REDCAP'].mean()
mean_delta_all = mean_redcap_all - mean_ctrl_all

# Sanity check:
# mean_delta_all should equal net_delta_from_nonident / n_total
sanity_expected = net_delta_from_nonident / n_total
sanity_diff = mean_delta_all - sanity_expected

print("\n========== Nonidentity direction ==========")
print(f"Nonidentical with REDCAP > CTRL: {n_up_redcap} ({n_up_redcap/n_nonident:.3%} of nonidentical)" if n_nonident else "Nonidentical with REDCAP > CTRL: 0")
print(f"Nonidentical with CTRL > REDCAP: {n_down_redcap} ({n_down_redcap/n_nonident:.3%} of nonidentical)" if n_nonident else "Nonidentical with CTRL > REDCAP: 0")

print("\n========== Magnitude carried by nonidentity ==========")
print(f"Sum of positive deltas (REDCAP > CTRL): {sum_up_redcap:.6f}")
print(f"Sum of negative deltas (CTRL > REDCAP): {sum_down_redcap:.6f}")
print(f"Net delta from nonidentical sites: {net_delta_from_nonident:.6f}")

print("\n========== Mean-level sanity check ==========")
print(f"Mean CTRL across shared sites: {mean_ctrl_all:.6f}")
print(f"Mean REDCAP across shared sites: {mean_redcap_all:.6f}")
print(f"Observed mean delta (REDCAP - CTRL): {mean_delta_all:.6f}")
print(f"Net delta from nonidentical / total sites: {sanity_expected:.6f}")
print(f"Difference (should be ~0): {sanity_diff:.12f}")

# --------------------------------------------
# Boundary-state helpers
# --------------------------------------------
def boundary_state(x):
    if x == 0:
        return '0'
    elif x == 100:
        return '100'
    else:
        return 'partial'

nonident_df['CTRL_state'] = nonident_df['Mean_CTRL'].apply(boundary_state)
nonident_df['REDCAP_state'] = nonident_df['Mean_REDCAP'].apply(boundary_state)

# --------------------------------------------
# Switch-like vs toggle-like
# switch-like = any nonidentical move involving at least one boundary
# toggle-like = partial -> partial
# --------------------------------------------
nonident_df['Switch_like'] = (
    (nonident_df['CTRL_state'].isin(['0', '100'])) |
    (nonident_df['REDCAP_state'].isin(['0', '100']))
)

nonident_df['Toggle_like'] = (
    (nonident_df['CTRL_state'] == 'partial') &
    (nonident_df['REDCAP_state'] == 'partial')
)

n_switch = int(nonident_df['Switch_like'].sum())
n_toggle = int(nonident_df['Toggle_like'].sum())

print("\n========== Nonidentity mechanism class ==========")
print(f"Switch-like (involves boundary 0 or 100): {n_switch} ({n_switch/n_nonident:.3%} of nonidentical)" if n_nonident else "Switch-like: 0")
print(f"Toggle-like (partial -> partial): {n_toggle} ({n_toggle/n_nonident:.3%} of nonidentical)" if n_nonident else "Toggle-like: 0")

# --------------------------------------------
# More granular switch classes
# --------------------------------------------
switch_0_to_partial = int(((nonident_df['CTRL_state'] == '0') & (nonident_df['REDCAP_state'] == 'partial')).sum())
switch_partial_to_0 = int(((nonident_df['CTRL_state'] == 'partial') & (nonident_df['REDCAP_state'] == '0')).sum())
switch_100_to_partial = int(((nonident_df['CTRL_state'] == '100') & (nonident_df['REDCAP_state'] == 'partial')).sum())
switch_partial_to_100 = int(((nonident_df['CTRL_state'] == 'partial') & (nonident_df['REDCAP_state'] == '100')).sum())
switch_0_to_100 = int(((nonident_df['CTRL_state'] == '0') & (nonident_df['REDCAP_state'] == '100')).sum())
switch_100_to_0 = int(((nonident_df['CTRL_state'] == '100') & (nonident_df['REDCAP_state'] == '0')).sum())

print("\n========== Granular nonidentity classes ==========")
print(f"0 -> partial: {switch_0_to_partial}")
print(f"partial -> 0: {switch_partial_to_0}")
print(f"100 -> partial: {switch_100_to_partial}")
print(f"partial -> 100: {switch_partial_to_100}")
print(f"0 -> 100: {switch_0_to_100}")
print(f"100 -> 0: {switch_100_to_0}")
print(f"partial -> partial (toggle-like): {n_toggle}")

# --------------------------------------------
# Optional: save detailed classification table
# --------------------------------------------
out_path = '/content/shared_cysteine_exact_identity_analysis.csv'
nonident_cols = [
    'SiteKey', 'Mean_CTRL', 'Mean_REDCAP', 'Delta_Percent', 'Abs_Delta_Percent',
    'Identity', 'NonIdentity', 'CTRL_state', 'REDCAP_state', 'Switch_like', 'Toggle_like'
]

# For identical rows, CTRL_state / REDCAP_state are still useful
df_id['CTRL_state'] = df_id['Mean_CTRL'].apply(boundary_state)
df_id['REDCAP_state'] = df_id['Mean_REDCAP'].apply(boundary_state)
df_id['Switch_like'] = False
df_id['Toggle_like'] = False
df_id.to_csv(out_path, index=False)

print(f"\nSaved detailed classification table to: {out_path}")
