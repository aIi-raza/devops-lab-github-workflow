"""
visualize.py

Plots confusion matrices for the trained models and a bar chart comparing
your results against the paper's reported results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrices(y_test, dt_preds, rf_preds):
    """Plot Decision Tree and Random Forest confusion matrices side by side."""
    dt_cm = confusion_matrix(y_test, dt_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=axes[0])
    axes[0].set_title('Decision Tree - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=axes[1])
    axes[1].set_title('Random Forest - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


def plot_comparison_chart(dt_results: dict, rf_results: dict):
    """
    Plot a grouped bar chart comparing your DT/RF results against the
    paper's reported DT/RF results, across all 4 main metrics.
    """
    paper_dt = [99.51, 99.49, 99.46, 99.47]
    paper_rf = [99.72, 99.84, 99.56, 99.70]

    your_dt = [dt_results["accuracy"], dt_results["precision"],
               dt_results["recall"], dt_results["f1"]]
    your_rf = [rf_results["accuracy"], rf_results["precision"],
               rf_results["recall"], rf_results["f1"]]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, paper_dt, width, label='Paper - DT', color='#a8c8f0')
    ax.bar(x - 0.5 * width, your_dt, width, label='Yours - DT', color='#2c5f9e')
    ax.bar(x + 0.5 * width, paper_rf, width, label='Paper - RF', color='#f5b895')
    ax.bar(x + 1.5 * width, your_rf, width, label='Yours - RF', color='#d9622b')

    ax.set_ylabel('Score (%)')
    ax.set_title('Your Results vs. Paper (Avci & Koca, 2023)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(98, 100)
    ax.legend()

    plt.tight_layout()
    plt.show()
