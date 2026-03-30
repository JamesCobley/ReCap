# ============================================
# Nonidentity within the nonzero-oxidation subset
# ============================================

df_nz = df_id.copy()

# A site is in the oxidised subset if it is >0 in either group
nz_either_mask = (df_nz['Mean_CTRL'] > 0) | (df_nz['Mean_REDCAP'] > 0)
nz_either_df = df_nz[nz_either_mask].copy()

n_nz_either = len(nz_either_df)
n_nz_either_ident = int(nz_either_df['Identity'].sum())
n_nz_either_nonident = int(nz_either_df['NonIdentity'].sum())

print("\n========== Nonzero oxidation subset (either group > 0) ==========")
print(f"Sites with nonzero oxidation in either group: {n_nz_either}")
print(f"Identical: {n_nz_either_ident} ({n_nz_either_ident/n_nz_either:.3%})" if n_nz_either else "Identical: 0")
print(f"Nonidentical: {n_nz_either_nonident} ({n_nz_either_nonident/n_nz_either:.3%})" if n_nz_either else "Nonidentical: 0")

# Stricter version: >0 in both groups
nz_both_mask = (df_nz['Mean_CTRL'] > 0) & (df_nz['Mean_REDCAP'] > 0)
nz_both_df = df_nz[nz_both_mask].copy()

n_nz_both = len(nz_both_df)
n_nz_both_ident = int(nz_both_df['Identity'].sum())
n_nz_both_nonident = int(nz_both_df['NonIdentity'].sum())

print("\n========== Nonzero oxidation subset (both groups > 0) ==========")
print(f"Sites with nonzero oxidation in both groups: {n_nz_both}")
print(f"Identical: {n_nz_both_ident} ({n_nz_both_ident/n_nz_both:.3%})" if n_nz_both else "Identical: 0")
print(f"Nonidentical: {n_nz_both_nonident} ({n_nz_both_nonident/n_nz_both:.3%})" if n_nz_both else "Nonidentical: 0")
