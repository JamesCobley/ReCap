# ============================================
# Direction breakdown within switch-like vs toggle-like
# ============================================

# Rebuild directional masks on nonident_df
switch_up_redcap = ((nonident_df['Switch_like']) & (nonident_df['Delta_Percent'] > 0)).sum()
switch_up_ctrl   = ((nonident_df['Switch_like']) & (nonident_df['Delta_Percent'] < 0)).sum()

toggle_up_redcap = ((nonident_df['Toggle_like']) & (nonident_df['Delta_Percent'] > 0)).sum()
toggle_up_ctrl   = ((nonident_df['Toggle_like']) & (nonident_df['Delta_Percent'] < 0)).sum()

n_switch = int(nonident_df['Switch_like'].sum())
n_toggle = int(nonident_df['Toggle_like'].sum())

print("\n========== Direction within mechanism class ==========")
print(f"Switch-like total: {n_switch}")
print(f"  More oxidised in REDCAP: {switch_up_redcap} ({switch_up_redcap/n_switch:.3%})" if n_switch else "  More oxidised in REDCAP: 0")
print(f"  More oxidised in CTRL:   {switch_up_ctrl} ({switch_up_ctrl/n_switch:.3%})" if n_switch else "  More oxidised in CTRL: 0")

print(f"\nToggle-like total: {n_toggle}")
print(f"  More oxidised in REDCAP: {toggle_up_redcap} ({toggle_up_redcap/n_toggle:.3%})" if n_toggle else "  More oxidised in REDCAP: 0")
print(f"  More oxidised in CTRL:   {toggle_up_ctrl} ({toggle_up_ctrl/n_toggle:.3%})" if n_toggle else "  More oxidised in CTRL: 0")

# Optional: magnitude within each class
switch_mag_redcap = nonident_df.loc[nonident_df['Switch_like'] & (nonident_df['Delta_Percent'] > 0), 'Delta_Percent'].sum()
switch_mag_ctrl   = nonident_df.loc[nonident_df['Switch_like'] & (nonident_df['Delta_Percent'] < 0), 'Delta_Percent'].sum()

toggle_mag_redcap = nonident_df.loc[nonident_df['Toggle_like'] & (nonident_df['Delta_Percent'] > 0), 'Delta_Percent'].sum()
toggle_mag_ctrl   = nonident_df.loc[nonident_df['Toggle_like'] & (nonident_df['Delta_Percent'] < 0), 'Delta_Percent'].sum()

print("\n========== Magnitude within mechanism class ==========")
print(f"Switch-like positive delta sum (REDCAP > CTRL): {switch_mag_redcap:.6f}")
print(f"Switch-like negative delta sum (CTRL > REDCAP): {switch_mag_ctrl:.6f}")
print(f"Toggle-like positive delta sum (REDCAP > CTRL): {toggle_mag_redcap:.6f}")
print(f"Toggle-like negative delta sum (CTRL > REDCAP): {toggle_mag_ctrl:.6f}")
