"""
Plotting utilities to generate figures consistent with the exploratory analysis
performed in the notebook `notebooks/MLOPS_FASE1.ipynb`.

This module centralizes plotting so the same visuals can be reproduced from
pipeline code. All functions save figures to disk and optionally return the
figure object for further manipulation in interactive sessions.
"""
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class Plotter:
    """Helper class to generate and save common EDA and model plots."""

    def __init__(self, figures_dir: Path | str = "reports/figures"):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Set a consistent style
        sns.set(style="whitegrid", context="notebook")

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        out_path = self.figures_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        logger.info(f"Figure saved: {out_path}")
        return out_path

    def plot_histograms(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[Sequence[str]] = None,
        bins: int = 50,
        cols_per_row: int = 4,
        figsize: Tuple[int, int] = (16, 10),
        filename: str = "histograms_numeric.png",
    ) -> Path:
        """Plot KDE-enhanced histograms for numeric columns.

        Mirrors the histogram section in the notebook.
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = cols_per_row
        n_rows = int(np.ceil(len(numeric_columns) / n_cols)) or 1

        fig = plt.figure(figsize=figsize)
        for idx, col in enumerate(numeric_columns, start=1):
            ax = plt.subplot(n_rows, n_cols, idx)
            sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=ax)
            ax.set_title(f"Histograma de {col}")
        plt.tight_layout()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    def plot_boxplots(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[Sequence[str]] = None,
        cols_per_row: int = 4,
        figsize: Tuple[int, int] = (16, 10),
        filename: str = "boxplots_numeric.png",
    ) -> Path:
        """Plot boxplots for numeric columns (before/after cleaning)."""
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = cols_per_row
        n_rows = int(np.ceil(len(numeric_columns) / n_cols)) or 1

        fig = plt.figure(figsize=figsize)
        for idx, col in enumerate(numeric_columns, start=1):
            ax = plt.subplot(n_rows, n_cols, idx)
            sns.boxplot(y=df[col].dropna(), ax=ax)
            ax.set_title(f"Boxplot de {col}")
        plt.tight_layout()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    def plot_categorical_counts(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[Sequence[str]] = None,
        cols_per_row: int = 2,
        figsize: Tuple[int, int] = (12, 8),
        filename: str = "categorical_counts.png",
    ) -> Path:
        """Plot count plots for categorical columns."""
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if not categorical_columns:
            raise ValueError("No categorical columns found to plot")

        n_cols = cols_per_row
        n_rows = int(np.ceil(len(categorical_columns) / n_cols)) or 1

        fig = plt.figure(figsize=figsize)
        for idx, col in enumerate(categorical_columns, start=1):
            ax = plt.subplot(n_rows, n_cols, idx)
            sns.countplot(x=col, data=df, ax=ax)
            ax.set_title(f"Frecuencia de categorías en {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment("right")
        plt.tight_layout()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[Sequence[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        filename: str = "correlation_heatmap.png",
    ) -> Path:
        """Plot a correlation heatmap for numeric variables."""
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_columns:
            raise ValueError("No numeric columns found for correlation heatmap")

        corr_matrix = df[numeric_columns].corr()

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0,
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Matriz de correlaciones (variables acústicas)")
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[Sequence[str]] = None,
        normalize: bool = False,
        cmap: str = "Blues",
        figsize: Tuple[int, int] = (6, 5),
        filename: str = "confusion_matrix.png",
    ) -> Path:
        """Plot and save a confusion matrix."""
        if normalize:
            with np.errstate(all="ignore"):
                cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            cbar=False,
            xticklabels=class_names if class_names is not None else True,
            yticklabels=class_names if class_names is not None else True,
            ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    def plot_pca_variance(
        self,
        explained_variance_ratio: Sequence[float],
        figsize: Tuple[int, int] = (8, 4),
        filename: str = "pca_variance.png",
    ) -> Path:
        """Plot explained variance ratio and cumulative sum for PCA components."""
        evr = np.asarray(explained_variance_ratio)
        cumulative = evr.cumsum()

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].plot(np.arange(1, len(evr) + 1), evr, marker="o")
        ax[0].set_title("Varianza explicada por componente")
        ax[0].set_xlabel("Componente")
        ax[0].set_ylabel("Varianza explicada")

        ax[1].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")
        ax[1].set_title("Varianza acumulada")
        ax[1].set_xlabel("Componente")
        ax[1].set_ylabel("Varianza acumulada")
        ax[1].set_ylim(0, 1.05)

        plt.tight_layout()
        out = self._save(fig, filename)
        plt.close(fig)
        return out


