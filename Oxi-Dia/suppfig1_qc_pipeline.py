#!/usr/bin/env python3
"""
Supplementary Figure 1 QC pipeline for Oxi-DIA (DIA-NN outputs)

Inputs:
  - DIA-NN report in Parquet format

Outputs (in out_dir):
  - summary_metrics.csv
  - protein_groups_per_run.csv
  - SuppFig1_metrics_table.png
  - SuppFig1_protein_groups_per_run.png
  - SuppFig1_rt_drift_cys_box.png
  - SuppFig1_rt_drift_cys_median_bar.png
  - (optional) Same-precursor-set QC panels:
      SuppFig1_rt_drift_complete_precursors_box.png
      SuppFig1_rt_drift_complete_precursors_median_bar.png
      SuppFig1_mass_accuracy_complete_precursors_box.png
      SuppFig1_mass_accuracy_complete_precursors_median_bar.png
      SuppFig1_fwhm_complete_precursors_box.png
      SuppFig1_fwhm_complete_precursors_median_bar.png

Notes:
  - "Complete precursors" are defined by (Modified.Sequence, Precursor.Charge) observed in ALL runs.
  - RT drift is computed by centering each peptide/precursor's RT by its median across runs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str:
    """Pick the first existing column from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns were found: {candidates}")
    return ""


def savefig(path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def compute_global_metrics(df: pd.DataFrame, rt_col: str) -> dict:
    # Unique Protein Groups (global)
    unique_proteins = (
        df["Protein.Ids"]
        .dropna()
        .astype(str)
        .str.split(";")
        .explode()
        .nunique()
    )

    # Avg Fragment Ion Coverage
    fragment_cols = [c for c in df.columns if ("Fr." in c and c.endswith(".Quantity"))]
    if len(fragment_cols) == 0:
        avg_frag_coverage = np.nan
    else:
        avg_frag_coverage = df[fragment_cols].notna().sum(axis=1).mean()

    # Mass accuracy (Mean + SD)
    ma_col = pick_col(df, ["Ms1.Apex.Mz.Delta", "Ms1.Apex.Mz.Delta(ppm)"], required=True)
    mass_accuracy_mean = df[ma_col].mean()
    mass_accuracy_sd = df[ma_col].std()

    # Peptide completeness across runs (by Modified.Sequence)
    runs = df["Run"].nunique()
    peptides_per_run = df.groupby(["Modified.Sequence", "Run"]).size().unstack(fill_value=0)
    completeness = (peptides_per_run > 0).sum(axis=1).mean() / runs

    # FWHM
    fwhm_col = pick_col(df, ["FWHM", "Fwhm", "Peak.Width", "PeakWidth"], required=True)
    fwhm_mean = df[fwhm_col].mean()
    fwhm_sd = df[fwhm_col].std()

    # Q-values
    q_col = pick_col(df, ["Q.Value", "QValue", "q.value"], required=True)
    q_vals = df[q_col].dropna()
    total_peptides = len(q_vals)
    q_mean = q_vals.mean()
    q_sd = q_vals.std()
    q_001 = (q_vals < 0.01).sum()
    q_0001 = (q_vals < 0.001).sum()
    q_00001 = (q_vals < 0.0001).sum()

    # RT CV for C-containing peptides (by stripped sequence)
    stripped_col = pick_col(df, ["Stripped.Sequence", "Stripped.Sequence"], required=True)
    df_cys = df[df[stripped_col].astype(str).str.contains("C", na=False)].copy()

    rt_matrix = (
        df_cys
        .groupby([stripped_col, "Run"])[rt_col]
        .median()
        .unstack()
    )
    rt_cv = rt_matrix.std(axis=1) / rt_matrix.mean(axis=1)
    rt_cv_median = np.nanmedian(rt_cv.values)

    return dict(
        unique_protein_groups_global=int(unique_proteins),
        avg_fragment_ion_coverage=float(avg_frag_coverage) if pd.notna(avg_frag_coverage) else np.nan,
        mass_accuracy_col=ma_col,
        mass_accuracy_mean=float(mass_accuracy_mean),
        mass_accuracy_sd=float(mass_accuracy_sd),
        peptide_completeness=float(completeness),
        fwhm_col=fwhm_col,
        fwhm_mean=float(fwhm_mean),
        fwhm_sd=float(fwhm_sd),
        q_col=q_col,
        total_peptides_with_q=int(total_peptides),
        q_lt_0_01=int(q_001),
        q_lt_0_001=int(q_0001),
        q_lt_0_0001=int(q_00001),
        q_mean=float(q_mean),
        q_sd=float(q_sd),
        rt_col=rt_col,
        median_rt_cv_cys_peptides=float(rt_cv_median),
    )


def protein_groups_per_run(df: pd.DataFrame) -> pd.Series:
    df_expanded = df.assign(ProteinID=df["Protein.Ids"].astype(str).str.split(";")).explode("ProteinID")
    # remove empty strings if any
    df_expanded["ProteinID"] = df_expanded["ProteinID"].replace({"": np.nan})
    return df_expanded.groupby("Run")["ProteinID"].nunique().sort_index()


def rt_drift_by_run(df: pd.DataFrame, rt_col: str, key_cols: list[str]) -> tuple[pd.DataFrame, pd.Series, float]:
    """
    Returns:
      rt_centered: matrix [peptide x run] centered per peptide
      per_run_median_drift
      median_rt_cv
    """
    rt_matrix = (
        df
        .groupby(key_cols + ["Run"])[rt_col]
        .median()
        .unstack()
    )
    rt_centered = rt_matrix.sub(rt_matrix.median(axis=1), axis=0)
    per_run_median_drift = rt_centered.median(axis=0)
    rt_cv = rt_matrix.std(axis=1) / rt_matrix.mean(axis=1)
    rt_cv_median = np.nanmedian(rt_cv.values)
    return rt_centered, per_run_median_drift, float(rt_cv_median)


def complete_precursor_subset(df: pd.DataFrame, rt_col: str, key_cols: list[str]) -> pd.DataFrame:
    df_pep = df.dropna(subset=key_cols + [rt_col]).copy()
    presence = (
        df_pep
        .groupby(key_cols + ["Run"])
        .size()
        .unstack(fill_value=0)
    )
    peptides_in_all_runs = presence[presence.gt(0).all(axis=1)].index
    df_complete = (
        df_pep
        .set_index(key_cols)
        .loc[peptides_in_all_runs]
        .reset_index()
    )
    return df_complete


def plot_metrics_table(metrics_rows: list[tuple[str, str]], out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    table = ax.table(
        cellText=metrics_rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 1.5)
    plt.title(title, fontsize=14, weight="bold", pad=20)
    savefig(out_png, dpi=300)


def plot_bar(series: pd.Series, out_png: Path, title: str, ylabel: str, xlabel: str = "Run"):
    plt.figure(figsize=(10, 5))
    series.sort_index().plot(kind="bar")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    savefig(out_png, dpi=300)


def plot_boxplot(matrix: pd.DataFrame, out_png: Path, title: str, ylabel: str, hline0: bool = True):
    plt.figure(figsize=(12, 5))
    matrix.boxplot()
    if hline0:
        plt.axhline(0, linestyle="--")
    plt.ylabel(ylabel)
    plt.xlabel("Run")
    plt.title(title)
    savefig(out_png, dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to DIA-NN report parquet")
    ap.add_argument("--out_dir", required=True, help="Output directory for SuppFig1 panels + tables")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    ap.add_argument("--make_complete_precursor_panels", action="store_true",
                    help="Also compute QC panels restricted to precursors present in all runs")
    args = ap.parse_args()

    report_path = Path(args.report)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load once
    df = pd.read_parquet(report_path)

    # Column picks (robust)
    rt_col = pick_col(df, ["RT", "RT.Apex", "Rt", "Retention.Time", "iRT"], required=True)
    stripped_col = pick_col(df, ["Stripped.Sequence"], required=True)
    ma_col = pick_col(df, ["Ms1.Apex.Mz.Delta", "Ms1.Apex.Mz.Delta(ppm)"], required=True)
    fwhm_col = pick_col(df, ["FWHM", "Fwhm", "Peak.Width", "PeakWidth"], required=True)

    # ---------------------------
    # Global metrics + summary table
    # ---------------------------
    g = compute_global_metrics(df, rt_col=rt_col)

    pg_run = protein_groups_per_run(df)
    g["avg_protein_groups_per_run"] = float(pg_run.mean())
    g["min_protein_groups_per_run"] = int(pg_run.min())
    g["max_protein_groups_per_run"] = int(pg_run.max())

    # Save summary metrics as CSV
    metrics_df = pd.DataFrame([g])
    metrics_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    # Save per-run protein group counts
    pg_run.rename("protein_groups").to_csv(out_dir / "protein_groups_per_run.csv")

    # Build nice table rows
    metrics_rows = [
        ("Unique protein groups (global)", f"{g['unique_protein_groups_global']:,}"),
        ("Protein groups / run (mean)", f"{g['avg_protein_groups_per_run']:.0f}"),
        ("Protein groups / run (min–max)", f"{g['min_protein_groups_per_run']:,}–{g['max_protein_groups_per_run']:,}"),
        ("Avg fragment ion coverage", f"{g['avg_fragment_ion_coverage']:.2f}" if pd.notna(g["avg_fragment_ion_coverage"]) else "NA"),
        (f"Mass accuracy mean ({g['mass_accuracy_col']})", f"{g['mass_accuracy_mean']:.4f}"),
        (f"Mass accuracy SD ({g['mass_accuracy_col']})", f"{g['mass_accuracy_sd']:.4f}"),
        ("Peptide completeness across runs", f"{g['peptide_completeness']*100:.2f}%"),
        (f"FWHM mean ({g['fwhm_col']})", f"{g['fwhm_mean']:.2f}"),
        (f"FWHM SD ({g['fwhm_col']})", f"{g['fwhm_sd']:.2f}"),
        ("Q-value < 0.01", f"{g['q_lt_0_01']:,} ({g['q_lt_0_01']/g['total_peptides_with_q']*100:.2f}%)"),
        ("Q-value < 0.001", f"{g['q_lt_0_001']:,} ({g['q_lt_0_001']/g['total_peptides_with_q']*100:.2f}%)"),
        ("Q-value < 0.0001", f"{g['q_lt_0_0001']:,} ({g['q_lt_0_0001']/g['total_peptides_with_q']*100:.2f}%)"),
        ("Q-value mean", f"{g['q_mean']:.5e}"),
        ("Q-value SD", f"{g['q_sd']:.5e}"),
        (f"Median RT CV (C-peptides; {g['rt_col']})", f"{g['median_rt_cv_cys_peptides']:.4f}"),
    ]

    plot_metrics_table(
        metrics_rows,
        out_png=out_dir / "SuppFig1_metrics_table.png",
        title="Oxi-DIA performance summary (DIA-NN report)"
    )

    # ---------------------------
    # Protein groups per run plot
    # ---------------------------
    plot_bar(
        pg_run,
        out_png=out_dir / "SuppFig1_protein_groups_per_run.png",
        title="Protein groups identified per run",
        ylabel="Protein groups (unique IDs)"
    )

    # ---------------------------
    # RT drift for cysteine peptides (Stripped.Sequence level)
    # ---------------------------
    df_cys = df[df[stripped_col].astype(str).str.contains("C", na=False)].copy()
    rt_centered_cys, per_run_median_drift_cys, rt_cv_median_cys = rt_drift_by_run(
        df_cys, rt_col=rt_col, key_cols=[stripped_col]
    )
    # Save RT drift summaries
    per_run_median_drift_cys.rename("median_rt_drift_min").to_csv(out_dir / "rt_drift_cys_median_per_run.csv")

    plot_boxplot(
        rt_centered_cys,
        out_png=out_dir / "SuppFig1_rt_drift_cys_box.png",
        title="RT drift across runs (Cys peptides; centered per peptide)",
        ylabel="RT drift (min)"
    )
    plot_bar(
        per_run_median_drift_cys,
        out_png=out_dir / "SuppFig1_rt_drift_cys_median_bar.png",
        title="Median RT drift per run (Cys peptides; centered per peptide)",
        ylabel="Median RT drift (min)"
    )

    # ---------------------------
    # Optional: same-precursor-set panels (Modified.Sequence + Charge)
    # ---------------------------
    if args.make_complete_precursor_panels:
        key_cols = ["Modified.Sequence", "Precursor.Charge"]
        df_complete = complete_precursor_subset(df, rt_col=rt_col, key_cols=key_cols)

        # RT drift
        rt_centered, per_run_rt = rt_drift_by_run(df_complete, rt_col=rt_col, key_cols=key_cols)
        plot_boxplot(
            rt_centered,
            out_png=out_dir / "SuppFig1_rt_drift_complete_precursors_box.png",
            title="RT drift across runs (complete precursors; centered per precursor)",
            ylabel="RT drift (min)"
        )
        plot_bar(
            per_run_rt,
            out_png=out_dir / "SuppFig1_rt_drift_complete_precursors_median_bar.png",
            title="Median RT drift per run (complete precursors)",
            ylabel="Median RT drift (min)"
        )

        # Mass accuracy (same precursors)
        ma_matrix = (
            df_complete
            .groupby(key_cols + ["Run"])[ma_col]
            .median()
            .unstack()
            .dropna(how="any", axis=0)
        )
        plot_boxplot(
            ma_matrix,
            out_png=out_dir / "SuppFig1_mass_accuracy_complete_precursors_box.png",
            title="Mass accuracy across runs (complete precursors)",
            ylabel=f"{ma_col} (Da)",
            hline0=True
        )
        plot_bar(
            ma_matrix.median(axis=0),
            out_png=out_dir / "SuppFig1_mass_accuracy_complete_precursors_median_bar.png",
            title="Median mass accuracy per run (complete precursors)",
            ylabel=f"Median {ma_col} (Da)"
        )

        # FWHM (same precursors)
        fwhm_matrix = (
            df_complete
            .groupby(key_cols + ["Run"])[fwhm_col]
            .median()
            .unstack()
            .dropna(how="any", axis=0)
        )
        plot_boxplot(
            fwhm_matrix,
            out_png=out_dir / "SuppFig1_fwhm_complete_precursors_box.png",
            title="Peak width (FWHM) across runs (complete precursors)",
            ylabel=f"{fwhm_col} (min)",
            hline0=False
        )
        plot_bar(
            fwhm_matrix.median(axis=0),
            out_png=out_dir / "SuppFig1_fwhm_complete_precursors_median_bar.png",
            title="Median FWHM per run (complete precursors)",
            ylabel=f"Median {fwhm_col} (min)"
        )

        # Record how many complete precursors you had
        n_runs = df["Run"].nunique()
        n_complete = df_complete.drop_duplicates(subset=key_cols).shape[0]
        pd.DataFrame([{"runs": n_runs, "complete_precursors": n_complete}]).to_csv(
            out_dir / "complete_precursor_counts.csv", index=False
        )

    print(f"✅ SuppFig1 outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
