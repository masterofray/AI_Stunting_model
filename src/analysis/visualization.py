#!/usr/bin/env python3

__author__     = "Aryanto"
__copyright__  = "Copyright 2026, masterofray/AI_Stunting_model"
__credits__    = ["aryanto"]
__license__    = "GNU_Public"
__version__    = "0.2.0"
__maintainer__ = "Aryanto, M.Si"
__email__      = "aryanto.dandan@gmail.com"
__created__    = "2026-08-31"
__modified__   = "2026-09-18"


"""Data Visualization Module"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from config.settings import REQUIRED_COLUMNS, BINARY_VARS, TARGET_VAR, DISTRICT_COL, MALNUTRITION_COL, plot_config

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = plot_config.FIG_SIZE_STANDARD

    def plot_district_prevalence(self, df_summary: Optional[pd.DataFrame] = None, output_path: Optional[Path] = None) -> plt.Figure:
        if df_summary is None:
            from analysis.descriptive import DescriptiveAnalyzer
            df_summary = DescriptiveAnalyzer(self.df).district_summary()
        df_summary = df_summary.sort_values("stunting_prev")
        x = np.arange(len(df_summary))
        vals = df_summary["stunting_prev"].to_numpy()
        low = vals - df_summary["ci95_low"].to_numpy()
        high = df_summary["ci95_high"].to_numpy() - vals
        fig, ax = plt.subplots(figsize=plot_config.FIG_SIZE_WIDE)
        bars = ax.bar(x, vals, color=plot_config.COLOR_STUNTING, alpha=0.7)
        ax.errorbar(x, vals, yerr=[low, high], fmt="none", capsize=4, color='black', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(df_summary[DISTRICT_COL], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Stunting Prevalence")
        ax.set_xlabel("District")
        ax.set_title("Stunting Prevalence by District with 95% Wilson CI")
        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=plot_config.FIG_DPI, bbox_inches='tight')
        return fig

    def plot_malnutrition_levels(self, df_summary: Optional[pd.DataFrame] = None, output_path: Optional[Path] = None) -> plt.Figure:
        if df_summary is None:
            from analysis.descriptive import DescriptiveAnalyzer
            df_summary, _ = DescriptiveAnalyzer(self.df).malnutrition_summary()
        fig, ax = plt.subplots(figsize=plot_config.FIG_SIZE_STANDARD)
        x = df_summary[MALNUTRITION_COL].astype(str)
        y = df_summary["stunting_prev"]
        colors = [plot_config.COLOR_STUNTING if val > 0.5 else plot_config.COLOR_NON_STUNTING for val in y]
        ax.bar(x, y, color=colors, alpha=0.7)
        ax.set_xlabel("Malnutrition Level")
        ax.set_ylabel("Stunting Prevalence")
        ax.set_title("Stunting Prevalence by Malnutrition Level")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=plot_config.FIG_DPI, bbox_inches='tight')
        return fig

    def plot_binary_covariates_by_stunting(self, output_path: Optional[Path] = None) -> plt.Figure:
        cov = self.df.groupby(TARGET_VAR)[BINARY_VARS].mean().T
        fig, ax = plt.subplots(figsize=plot_config.FIG_SIZE_WIDE)
        cov.plot(kind="bar", ax=ax, alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion Coded 1")
        ax.set_xlabel("Binary Variable")
        ax.set_title("Binary Variable Distribution by Stunting Status")
        ax.legend(title="Stunting Status", labels=["Non-Stunting", "Stunting"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=plot_config.FIG_DPI, bbox_inches='tight')
        return fig

    def plot_odds_ratios(self, df_associations: Optional[pd.DataFrame] = None, output_path: Optional[Path] = None) -> plt.Figure:
        if df_associations is None:
            from analysis.statistical import StatisticalAnalyzer
            df_associations = StatisticalAnalyzer(self.df).all_binary_associations()
        fig, ax = plt.subplots(figsize=plot_config.FIG_SIZE_STANDARD)
        df_associations = df_associations.sort_values("odds_ratio")
        y = np.arange(len(df_associations))
        ors = df_associations["odds_ratio"].values
        ci_low = df_associations["or_ci_low"].values
        ci_high = df_associations["or_ci_high"].values
        ax.hlines(y=y, xmin=ci_low, xmax=ci_high, color='black', alpha=0.5, linewidth=2)
        ax.scatter(ors, y, color=plot_config.COLOR_STUNTING, s=100, alpha=0.7)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Odds Ratio (log scale)")
        ax.set_ylabel("Variable")
        ax.set_title("Odds Ratios for Binary Variables with 95% CI")
        ax.set_yticks(y)
        ax.set_yticklabels(df_associations["variable"])
        ax.set_xscale('log')
        plt.tight_layout()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=plot_config.FIG_DPI, bbox_inches='tight')
        return fig

    def save_all_plots(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        self.plot_district_prevalence(output_path=figures_dir / "district_prevalence.png")
        self.plot_malnutrition_levels(output_path=figures_dir / "malnutrition_levels.png")
        self.plot_binary_covariates_by_stunting(output_path=figures_dir / "binary_covariates_by_stunting.png")
        self.plot_odds_ratios(output_path=figures_dir / "odds_ratios.png")
        print(f"Plots saved to: {figures_dir}")