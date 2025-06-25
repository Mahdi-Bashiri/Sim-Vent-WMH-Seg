import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class SegmentationVisualizer:
    """
    A class to create visualizations for segmentation results comparison.
    """

    def __init__(self, results: Dict):
        """
        Initialize visualizer with results dictionary.

        Args:
            results: Nested dictionary containing evaluation metrics for all methods and cases
        """
        self.results = results
        self.methods = list(results.keys())
        self.metrics = list(next(iter(next(iter(results.values())).values())).keys())

        # Set global font sizes for all plots
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20
        })

    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Convert nested results dictionary to a pandas DataFrame.

        Returns:
            pd.DataFrame: Formatted DataFrame for plotting
        """
        data = []
        for method in self.results:
            for case in self.results[method]:
                for metric, value in self.results[method][case].items():
                    if metric != 'confusion_matrix':  # Skip confusion matrix for this dataframe
                        data.append({
                            'Method': method,
                            'Case': case,
                            'Metric': metric,
                            'Value': value
                        })
        return pd.DataFrame(data)

    def create_boxplots(self, save_path: Optional[str] = None) -> None:
        """
        Create boxplots comparing all methods for each metric with turquoise color scheme.

        Args:
            save_path: Optional path to save the figure
        """
        df = self._prepare_dataframe()

        # Create subplot for each metric in a single column
        n_metrics = len([m for m in self.metrics if m != 'confusion_matrix'])
        fig, axes = plt.subplots(nrows=n_metrics, ncols=1,
                                 figsize=(14, 6 * n_metrics))

        if n_metrics == 1:
            axes = [axes]  # Make axes iterable if there's only one metric

        # Custom turquoise color palette
        turquoise_palette = sns.light_palette('#40E0D0', n_colors=len(self.methods))

        metric_idx = 0
        for metric in self.metrics:
            if metric == 'confusion_matrix':
                continue

            metric_data = df[df['Metric'] == metric]
            sns.boxplot(data=metric_data, x='Method', y='Value', ax=axes[metric_idx], palette=turquoise_palette,
                        linewidth=2.5)

            # Increase font sizes
            # axes[metric_idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=18, fontweight='bold', pad=15)
            axes[metric_idx].set_xticklabels(axes[metric_idx].get_xticklabels(), rotation=45, ha='right', fontsize=14)
            axes[metric_idx].set_xlabel('', fontsize=16)
            axes[metric_idx].set_ylabel(metric.replace("_", " ").title(), fontsize=16, fontweight='bold')
            axes[metric_idx].grid(axis='y', linestyle='--', alpha=0.7)

            # Set y-axis limit based on metric
            if metric != 'hausdorff95':  # For all metrics except hausdorff95
                axes[metric_idx].set_ylim(0, 1)

            # Enhance tick parameters
            axes[metric_idx].tick_params(axis='both', which='major', labelsize=14, width=2)

            metric_idx += 1

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_violin_plots(self, save_path: Optional[str] = None) -> None:
        """
        Create violin plots comparing all methods for each metric with pinkish burgundy color scheme.

        Args:
            save_path: Optional path to save the figure
        """
        df = self._prepare_dataframe()

        n_metrics = len([m for m in self.metrics if m != 'confusion_matrix'])
        fig, axes = plt.subplots(nrows=n_metrics, ncols=1,
                                 figsize=(14, 6 * n_metrics))

        if n_metrics == 1:
            axes = [axes]  # Make axes iterable if there's only one metric

        # Custom pinkish burgundy color palette
        pink_burgundy_palette = sns.light_palette('#DB005B', n_colors=len(self.methods))

        metric_idx = 0
        for metric in self.metrics:
            if metric == 'confusion_matrix':
                continue

            metric_data = df[df['Metric'] == metric]

            sns.violinplot(
                data=metric_data,
                x='Method',
                y='Value',
                ax=axes[metric_idx],
                palette=pink_burgundy_palette,
                linewidth=2
            )

            # Increase font sizes
            # axes[metric_idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=18, fontweight='bold', pad=15)
            axes[metric_idx].set_xticklabels(axes[metric_idx].get_xticklabels(), rotation=45, ha='right', fontsize=14)
            axes[metric_idx].set_xlabel('', fontsize=16)
            axes[metric_idx].set_ylabel(metric.replace("_", " ").title(), fontsize=16, fontweight='bold')
            axes[metric_idx].grid(axis='y', linestyle='--', alpha=0.7)

            # Set y-axis limit based on metric
            if metric != 'hausdorff95':  # For all metrics except hausdorff95
                axes[metric_idx].set_ylim(0, 1)

            # Enhance tick parameters
            axes[metric_idx].tick_params(axis='both', which='major', labelsize=14, width=2)

            metric_idx += 1

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_heatmap(self, metric: str, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing pairwise statistical significance between methods.

        Args:
            metric: Name of the metric to compare
            save_path: Optional path to save the figure
        """
        if metric == 'confusion_matrix':
            return

        df = self._prepare_dataframe()
        metric_data = df[df['Metric'] == metric]

        # Compute p-values for all method pairs
        methods = sorted(metric_data['Method'].unique())
        n_methods = len(methods)
        p_values = np.zeros((n_methods, n_methods))

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    values1 = metric_data[metric_data['Method'] == method1]['Value']
                    values2 = metric_data[metric_data['Method'] == method2]['Value']
                    _, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    p_values[i, j] = p_value
                else:
                    p_values[i, j] = 1.0

        # Create heatmap
        plt.figure(figsize=(12, 10))

        # Create heatmap with larger font sizes
        ax = sns.heatmap(p_values, xticklabels=methods, yticklabels=methods,
                         annot=True, fmt='.3f', cmap='RdYlBu_r', annot_kws={"size": 14})

        # Increase font sizes
        plt.title(f'P-values for {metric.replace("_", " ").title()} Comparison', fontsize=20, pad=20)
        plt.xlabel('Method', fontsize=16, labelpad=15)
        plt.ylabel('Method', fontsize=16, labelpad=15)

        # Increase tick label sizes
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_roc_curves(self, roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                          task: str, save_path: Optional[str] = None) -> None:
        """
        Create ROC curves for all methods in a task.

        Args:
            roc_data: Dictionary mapping method names to (fpr, tpr, auc) tuples
            task: Task name (WMH or VENT)
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 10))

        for method, (fpr, tpr, auc_value) in roc_data.items():
            plt.plot(fpr, tpr, lw=3, label=f'{method} (AUC = {auc_value:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        # Increase font sizes
        plt.xlabel('False Positive Rate', fontsize=16, labelpad=12)
        plt.ylabel('True Positive Rate', fontsize=16, labelpad=12)
        plt.title(f'ROC Curves for {task} Segmentation', fontsize=20, pad=20)

        # Enhance legend
        plt.legend(loc="lower right", fontsize=14, frameon=True, fancybox=True, framealpha=0.9)
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        # Enhance tick parameters
        plt.tick_params(axis='both', which='major', labelsize=14, width=2)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_pr_curves(self, pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                         task: str, save_path: Optional[str] = None) -> None:
        """
        Create Precision-Recall curves for all methods in a task.

        Args:
            pr_data: Dictionary mapping method names to (precision, recall, auc) tuples
            task: Task name (WMH or VENT)
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(12, 10))

        for method, (precision, recall, auc_value) in pr_data.items():
            plt.plot(recall, precision, lw=3, label=f'{method} (AUC = {auc_value:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        # Increase font sizes
        plt.xlabel('Recall', fontsize=16, labelpad=12)
        plt.ylabel('Precision', fontsize=16, labelpad=12)
        plt.title(f'Precision-Recall Curves for {task} Segmentation', fontsize=20, pad=20)

        # Enhance legend
        plt.legend(loc="upper right", fontsize=14, frameon=True, fancybox=True, framealpha=0.9)
        plt.grid(axis='both', linestyle='--', alpha=0.7)

        # Enhance tick parameters
        plt.tick_params(axis='both', which='major', labelsize=14, width=2)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_confusion_matrix_plots(self, task_results: Dict[str, Dict], task: str,
                                      save_path: Optional[str] = None) -> None:
        """
        Create confusion matrix plots for all methods in a task.

        Args:
            task_results: Dictionary containing results for all methods
            task: Task name (WMH or VENT)
            save_path: Optional path to save the figure
        """
        methods = list(task_results.keys())

        # Aggregate confusion matrices across all subjects for each method
        aggregated_cms = {}
        for method in methods:
            # Initialize with zeros [TN, FP, FN, TP]
            agg_cm = np.zeros(4)

            # Sum confusion matrices across all subjects
            for subject_id in task_results[method]:
                if 'confusion_matrix' in task_results[method][subject_id]:
                    cm = np.array(task_results[method][subject_id]['confusion_matrix'])
                    agg_cm += cm

            aggregated_cms[method] = agg_cm

        # Create a normalized version of the confusion matrices
        normalized_cms = {}
        for method, cm in aggregated_cms.items():
            # Convert to 2x2 format
            cm_2x2 = np.array([[cm[0], cm[1]], [cm[2], cm[3]]])

            # Normalize by row (true condition)
            row_sums = cm_2x2.sum(axis=1, keepdims=True)
            norm_cm = np.zeros_like(cm_2x2, dtype=float)

            # Avoid division by zero
            for i in range(2):
                if row_sums[i] > 0:
                    norm_cm[i] = cm_2x2[i] / row_sums[i]

            normalized_cms[method] = norm_cm

        # Plot the confusion matrices
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 12))

        # Set all subplot backgrounds to white
        for row in axes:
            for ax in row:
                ax.set_facecolor('white')
                ax.set_visible(False)  # Hide all initially

        task_configs = {
            'VENT': {
                'row': 0,
                'start_col': 1,  # Start in column 2 (0-indexed)
                'end_col': 3  # End in column 4 (0-indexed)
            },
            'WMH': {
                'row': 1,
                'start_col': 0,  # Start in column 1 (0-indexed)
                'end_col': 4  # End in column 5 (0-indexed)
            }
        }

        config = task_configs[task]
        row_idx = config['row']

        # Get methods for this task and sort them
        task_methods = list(aggregated_cms.keys())

        # Plot each method's confusion matrix
        col_idx = config['start_col']
        for method in task_methods:
            if col_idx > config['end_col']:
                break

            ax = axes[row_idx, col_idx]
            ax.set_visible(True)

            # Get normalized confusion matrix
            cm = normalized_cms[method]

            # Create heatmap with larger font sizes
            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar=False,
                ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 16}
            )

            # Add raw counts as text in the cells
            raw_cm = aggregated_cms[method].reshape(2, 2)
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j + 0.5, i + 0.7, f'n={int(raw_cm[i, j])}',
                        ha='center', va='center',
                        color='black' if cm[i, j] < 0.7 else 'white',
                        fontsize=14  # Increased font size
                    )

            # Set labels with increased font sizes
            ax.set_title(f'{method} - {task}', fontsize=18, pad=15)
            ax.set_xlabel('Predicted', fontsize=16, labelpad=10)
            ax.set_ylabel('True', fontsize=16, labelpad=10)

            # Increase tick label sizes
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

            col_idx += 1

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def create_overlay_visualization(original: np.ndarray,
                                     ground_truth: np.ndarray,
                                     prediction: np.ndarray,
                                     slice_idx: int,
                                     save_path: Optional[str] = None) -> None:
        """
        Create a visualization showing original image with ground truth and prediction overlays.

        Args:
            original: Original T2-FLAIR image
            ground_truth: Ground truth segmentation mask
            prediction: Predicted segmentation mask
            slice_idx: Index of the slice to visualize
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(18, 7))

        # Original image
        plt.subplot(131)
        plt.imshow(np.rot90(original[:, :, slice_idx]), cmap='gray')
        plt.title('Original T2-FLAIR', fontsize=18, pad=15)
        plt.axis('off')

        # Ground truth overlay
        plt.subplot(132)
        plt.imshow(np.rot90(original[:, :, slice_idx]), cmap='gray')
        plt.imshow(np.rot90(ground_truth[:, :, slice_idx]), alpha=0.3, cmap='Reds')
        plt.title('Ground Truth Overlay', fontsize=18, pad=15)
        plt.axis('off')

        # Prediction overlay
        plt.subplot(133)
        plt.imshow(np.rot90(original[:, :, slice_idx]), cmap='gray')
        plt.imshow(np.rot90(prediction[:, :, slice_idx]), alpha=0.3, cmap='Blues')
        plt.title('Prediction Overlay', fontsize=18, pad=15)
        plt.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

