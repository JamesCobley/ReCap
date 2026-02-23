from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load full DIA-NN report
df = pd.read_parquet('/content/drive/MyDrive/reportnewcal.parquet')

# Only peptides with C
df['Has_C'] = df['Stripped.Sequence'].str.contains("C")
df_cys = df[df['Has_C']].copy()

# Deduplicate per Run and Stripped.Sequence
df_unique = df_cys.drop_duplicates(subset=['Run', 'Stripped.Sequence'])

# Assign label type
def get_label(seq):
    if "_L" in seq:
        return "L"
    elif "_H" in seq:
        return "H"
    else:
        return "None"

df_unique['Label'] = df_unique['Modified.Sequence'].apply(get_label)

# Summarize per run
summary = df_unique.groupby('Run').agg(
    Total_C_Peptides=('Stripped.Sequence', 'count'),
    Light_C=('Label', lambda x: (x == 'L').sum()),
    Heavy_C=('Label', lambda x: (x == 'H').sum())
).reset_index()

summary['Unlabeled_C'] = summary['Total_C_Peptides'] - summary['Light_C'] - summary['Heavy_C']
summary['%Unlabeled_C'] = (summary['Unlabeled_C'] / summary['Total_C_Peptides']) * 100

# Filter to mixing runs (exclude pure 0 and 100)
mixes_only = summary['Run'].str.contains("James_")
mixes_only = summary[mixes_only]

# Display result
print(mixes_only.to_string(index=False))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === 1. Input your data here ===
# Replace this with your actual CSV path or data
data = {
    'Run': [
        'James_0_S22', 'James_0_S23', 'James_0_S24',
        'James_100_S1', 'James_100_S2', 'James_100_S3',
        'James_20_S19', 'James_20_S20', 'James_20_S21',
        'James_40_S16', 'James_40_S17', 'James_40_S18',
        'James_60_S13', 'James_60_S14', 'James_60_S15',
        'James_80_S10', 'James_80_S11', 'James_80_S12',
        'James_90_S7', 'James_90_S8', 'James_90_S9',
        'James_95_S4', 'James_95_S5', 'James_95_S6'
    ],
    'Total_C_Peptides': [
        9341, 9884, 9749,
        18560, 18233, 18119,
        10805, 11302, 11158,
        12064, 12439, 12737,
        14286, 14777, 15087,
        16291, 17156, 16623,
        17341, 17262, 16973,
        17865, 17884, 17780
    ],
    'Light_C': [
        41, 58, 75,
        18077, 17752, 17611,
        1992, 2256, 2330,
        4246, 4633, 4924,
        7756, 8288, 8747,
        12374, 12777, 12645,
        15078, 15155, 14672,
        16352, 16327, 16111
    ],
    'Heavy_C': [
        7708, 8192, 8020,
        472, 472, 499,
        7839, 8016, 7848,
        7209, 7199, 7201,
        6185, 6136, 5983,
        3772, 4220, 3830,
        2209, 2049, 2238,
        1474, 1515, 1636
    ],
    'Unlabeled_C': [
        1592, 1634, 1654,
        11, 9, 9,
        974, 1030, 980,
        609, 607, 612,
        345, 353, 357,
        145, 159, 148,
        54, 58, 63,
        39, 42, 33
    ]
}
df = pd.DataFrame(data)

# === 2. Extract target % from run name ===
df['Target_%_Ox'] = df['Run'].str.extract(r'James_(\d+)_')[0].astype(int)

# === 3. Compute observed reduced states ===
df['Total_C_Labeled'] = df['Light_C'] + df['Heavy_C']
df['Total_C'] = df['Total_C_Labeled'] + df['Unlabeled_C']

df['Observed_Reduced_MS1'] = df['Light_C'] / df['Total_C']
df['Observed_Reduced_MS2'] = df['Light_C'] / (df['Light_C'] + df['Heavy_C'])  # MS2 proxy
df['Observed_Reduced_Combined'] = (df['Observed_Reduced_MS1'] + df['Observed_Reduced_MS2']) / 2

# === 4. Model expected reduced with adjustment for unlabelled ===
# Estimate average unlabeled from pure 0% and 100% sets
zero_unlab = df[df['Target_%_Ox'] == 0]['Unlabeled_C'] / df[df['Target_%_Ox'] == 0]['Total_C_Peptides']
hundred_unlab = df[df['Target_%_Ox'] == 100]['Unlabeled_C'] / df[df['Target_%_Ox'] == 100]['Total_C_Peptides']

avg_unlab_0 = zero_unlab.mean()
avg_unlab_100 = hundred_unlab.mean()

# Compute linear mix model for expected unlabelled fraction
df['Expected_Unlabeled'] = (
    (100 - df['Target_%_Ox']) * avg_unlab_0 +
    df['Target_%_Ox'] * avg_unlab_100
) / 100

# ✅ Model expected reduced fraction correctly: %reduced × (1 - unlabelled)
df['Expected_Reduced'] = df['Target_%_Ox'] / 100 * (1 - df['Expected_Unlabeled'])

# === 5. Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(df['Target_%_Ox'], df['Observed_Reduced_MS1'], 'o-', label='Observed MS1', color='tab:blue')
plt.plot(df['Target_%_Ox'], df['Observed_Reduced_MS2'], 's--', label='Observed MS2', color='tab:orange')
plt.plot(df['Target_%_Ox'], df['Observed_Reduced_Combined'], '^-', label='Observed Combined', color='tab:green')
plt.plot(df['Target_%_Ox'], df['Expected_Reduced'], 'k--', label='Expected Reduced')

plt.xlabel('Cysteine oxidation (%)')
plt.ylabel('% Reduced (computed)')
plt.title('Cysteine Redox State: Observed vs. Expected')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('redox_comparison.png',dpi=300)
plt.show()

# === 6. Save result ===
df[['Run', 'Target_%_Ox', 'Observed_Reduced_MS1', 'Observed_Reduced_MS2',
    'Observed_Reduced_Combined', 'Expected_Reduced']].to_csv("redox_comparison.csv", index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === 1. Load your data ===
# Replace with your actual CSV path
df = pd.read_csv("redox_comparison.csv")  # Ensure columns: Expected_Reduced, Observed_Reduced_MS1, Observed_Reduced_MS2, Observed_Reduced_Combined

# === 2. Safe MAPE implementation ===
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.any(mask):
        return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))
    else:
        return np.nan  # or 0 if you prefer

# === 3. Define metric computation ===
def compute_metrics(observed, expected, label):
    r2 = r2_score(expected, observed)
    mse = mean_squared_error(expected, observed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(expected, observed)
    mape = safe_mape(expected, observed)  # <--- using safe version here
    residuals = observed - expected
    bias = np.mean(residuals)
    sd_residuals = np.std(residuals)

    print(f"\nMetrics for {label}:")
    print(f"  R²     = {r2:.4f}")
    print(f"  RMSE   = {rmse:.4f}")
    print(f"  MAE    = {mae:.4f}")
    print(f"  MAPE   = {mape * 100:.2f}%")
    print(f"  Bias   = {bias:.4f}")
    print(f"  SD(residuals) = {sd_residuals:.4f}")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
        "sd": sd_residuals,
        "residuals": residuals
    }

# === 4. Compute metrics for each mode ===
results = {}
for col in ['Observed_Reduced_MS1', 'Observed_Reduced_MS2', 'Observed_Reduced_Combined']:
    label = col.split('_')[-1] if 'Combined' in col else col.split('_')[-1]
    results[label] = compute_metrics(df[col], df['Expected_Reduced'], label)

plt.figure(figsize=(10, 6))
for label, color, marker in zip(['MS1', 'MS2', 'Combined'], ['blue', 'green', 'red'], ['o', 's', '^']):
    plt.scatter(df['Expected_Reduced'], results[label]['residuals'], label=f"{label} Residuals", color=color, marker=marker, alpha=0.7)

plt.axhline(0, color='black', linestyle='--')
plt.axhspan(-0.05, 0.05, color='gray', alpha=0.1, label='±5% Band')
plt.xlabel("Expected Reduced (%)")
plt.ylabel("Residual (Observed - Expected)")
plt.title("Residuals of Oxi-DIA Quantification Across Reduction Range")
plt.ylim(-0.1, 0.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('residuals_clean.png', dpi=300)
plt.show()


# === 6. Export residuals ===
df['Residual_MS1'] = results['MS1']['residuals']
df['Residual_MS2'] = results['MS2']['residuals']
df['Residual_Combined'] = results['Combined']['residuals']
df.to_csv("oxi_dia_with_residuals.csv", index=False)
