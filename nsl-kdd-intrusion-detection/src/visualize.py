"""
visualize.py

Plots confusion matrices for the trained models and a bar chart comparing
your results against the paper's reported results.
"""

import numpy as np
import matplotlib.pyplot as plt   # for drawing charts
import seaborn as sns             # built on matplotlib, makes nicer-looking heatmaps
from sklearn.metrics import confusion_matrix  # builds the correct/incorrect prediction grid


def plot_confusion_matrices(y_test, dt_preds, rf_preds):
    """Plot Decision Tree and Random Forest confusion matrices side by side."""
    # Compares real answers (y_test) to predictions, returns a 2x2 grid:
    # [[correct normals, wrongly flagged attacks], [missed attacks, correctly caught attacks]]
    dt_cm = confusion_matrix(y_test, dt_preds)
    rf_cm = confusion_matrix(y_test, rf_preds)

    # Create one figure with 2 plot areas side by side (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # annot=True prints the actual numbers on each cell
    # fmt='d' formats them as whole numbers (not decimals)
    # cmap='Blues' sets the color scheme
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=axes[0])  # draw this one in the first (left) plot area
    axes[0].set_title('Decision Tree - Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Same thing again for Random Forest, drawn in the second (right) plot area
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=axes[1])
    axes[1].set_title('Random Forest - Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()  # keeps the two plots from overlapping/looking cramped
    plt.show()           # actually opens the window showing the charts


def plot_comparison_chart(dt_results: dict, rf_results: dict):
    """
    Plot a grouped bar chart comparing your DT/RF results against the
    paper's reported DT/RF results, across all 4 main metrics.
    """
    # The paper's fixed, published numbers (hardcoded here just for this chart)
    paper_dt = [99.51, 99.49, 99.46, 99.47]
    paper_rf = [99.72, 99.84, 99.56, 99.70]

    # Pull out our own 4 metrics in the same order, from the results dictionaries
    your_dt = [dt_results["accuracy"], dt_results["precision"],
               dt_results["recall"], dt_results["f1"]]
    your_rf = [rf_results["accuracy"], rf_results["precision"],
               rf_results["recall"], rf_results["f1"]]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    # x = [0, 1, 2, 3] - one position per metric on the chart
    x = np.arange(len(metrics))
    width = 0.2  # how wide each bar is

    fig, ax = plt.subplots(figsize=(10, 6))
    # Each .bar() call draws one group of 4 bars (one per metric)
    # We shift each group left/right using x - 1.5*width etc, so all 4 bars
    # per metric sit side by side instead of on top of each other
    ax.bar(x - 1.5 * width, paper_dt, width, label='Paper - DT', color='#a8c8f0')
    ax.bar(x - 0.5 * width, your_dt, width, label='Yours - DT', color='#2c5f9e')
    ax.bar(x + 0.5 * width, paper_rf, width, label='Paper - RF', color='#f5b895')
    ax.bar(x + 1.5 * width, your_rf, width, label='Yours - RF', color='#d9622b')

    ax.set_ylabel('Score (%)')
    ax.set_title('Your Results vs. Paper (Avci & Koca, 2023)')
    ax.set_xticks(x)                   # put a tick mark at each metric position
    ax.set_xticklabels(metrics)        # label those ticks with the metric names
    ax.set_ylim(98, 100)               # zoom in on 98-100%, since all scores are in that range
    ax.legend()                        # show the color key (which bar is which)

    plt.tight_layout()
    plt.show()