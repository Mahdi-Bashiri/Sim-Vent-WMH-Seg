import os
import time
import json
import pickle
import numpy as np
import nibabel as nib
import seaborn as sns
from pathlib import Path
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score



class MetricsComputer:
    """Separate class for computing metrics without plotting functionality"""
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        
    @staticmethod
    def calculate_loss(pred, target):
        """Calculate binary cross-entropy loss"""
        epsilon = 1e-7
        pred = np.clip(pred, epsilon, 1 - epsilon)
        
        if np.any(np.isin(target, [0.25, 0.75])):
            return np.mean((pred - target) ** 2)
        
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    
    @staticmethod
    def calculate_metrics(pred, target, key):
        """Calculate metrics based on mask type"""
        loss = MetricsComputer.calculate_loss(pred, target)
        
        auc_ = None
        if key != 'main':
            pred_flat = pred.flatten()
            gt_flat = target.flatten()
            gt_flat = gt_flat / np.max(gt_flat) if np.max(gt_flat) > 0 else gt_flat / 1
            # Now compute p-r ROC curve on clean data
            precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
            auc_ = auc(recall, precision)
            # auc_ = roc_auc_score(gt_flat, pred_flat)
            
        return loss, auc_

    def process_single_subject(self, args):
        """Process a single subject - designed for parallel processing"""
        subject_dir, epoch, split, refined = args
        subject_dir = Path(subject_dir)
        
        try:
            # Load ground truth masks
            gt_dir = subject_dir.parent.parent.parent.parent / 'groundtruth' / subject_dir.name
            if not gt_dir.exists():
                print(f"Ground truth directory not found: {gt_dir}")
                return None
            
            metrics = {
                'awmh': {'pred': None, 'gt': nib.load(gt_dir / f'{subject_dir.name.split("_")[-1]}_abWMHmask.nii.gz').get_fdata()},
                'nwmh': {'pred': None, 'gt': nib.load(gt_dir / f'{subject_dir.name.split("_")[-1]}_nWMHmask.nii.gz').get_fdata()},
                'vent': {'pred': None, 'gt': nib.load(gt_dir / f'{subject_dir.name.split("_")[-1]}_VENTmask.nii.gz').get_fdata()},
                'main': {'pred': None, 'gt': nib.load(gt_dir / f'{subject_dir.name.split("_")[-1]}_MAINmask.nii.gz').get_fdata()}
            }
            
            # Load predictions
            suffix = '_rf' if refined else ''
            results = {}
            
            for key in metrics.keys():
                if key == 'main':
                    pred_file = f'{subject_dir.name.split("_")[-1]}_our_main{suffix}.nii.gz'
                else:
                    pred_file = f'{subject_dir.name.split("_")[-1]}_our_{key}{suffix}.nii.gz'
                
                pred_path = subject_dir / pred_file
                if pred_path.exists():
                    metrics[key]['pred'] = nib.load(pred_path).get_fdata()
                    loss, auc_ = self.calculate_metrics(metrics[key]['pred'], metrics[key]['gt'], key)
                    results[key] = {'loss': loss, 'auc': auc_}
                else:
                    print(f"Prediction file not found: {pred_path}")
                        
            return {
                'subject': subject_dir.name,
                'epoch': epoch,
                'split': split,
                'metrics': results
            }
            
        except Exception as e:
            print(f"Error processing subject {subject_dir.name}: {str(e)}")
            return None

    def process_epoch_range(self, start_epoch, end_epoch, pred, refined=False, num_processes=None):
        """Process a specific range of epochs using parallel processing"""
        if num_processes is None:
            num_processes = mp.cpu_count()
        
        tasks = []
        for split in ['train', 'test']:
            split_dir = self.base_dir / split / 'predict' / pred
            print(f"\nProcessing {split} directory: {split_dir}")
            if not split_dir.exists():
                print(f"Directory not found: {split_dir}")
                continue
                
            epoch_dirs = [d for d in split_dir.glob('epoch_*') 
                         if start_epoch <= int(d.name.split('_')[1]) <= end_epoch]
            
            for epoch_dir in epoch_dirs:
                epoch_num = int(epoch_dir.name.split('_')[1])
                print(f"Processing Epoch: {epoch_num}")
                for subject_dir in epoch_dir.glob('subj_*'):
                    tasks.append((str(subject_dir), epoch_num, split, refined))
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self.process_single_subject, tasks)
        
        # Filter out None results and save
        results = [r for r in results if r is not None]
        return results

def save_results(results, output_file):
    """Save computed results to a file"""
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

class MetricsPlotter:
    """Separate class for plotting from saved results"""

    @staticmethod
    def load_results(results_files):
        """Load and combine results from multiple files"""
        all_results = []
        for file in results_files:
            with open(file, 'rb') as f:
                all_results.extend(pickle.load(f))
        return all_results

    @staticmethod
    def process_results_for_plotting(results):
        """Convert raw results into plottable format"""
        processed = {
            'train': {'loss': {}, 'auc': {}},
            'test': {'loss': {}, 'auc': {}}
        }

        for result in results:
            split = result['split']
            epoch = result['epoch']

            for metric_type in ['loss', 'auc']:
                if epoch not in processed[split][metric_type]:
                    processed[split][metric_type][epoch] = {'awmh': [], 'nwmh': [], 'vent': [], 'main': []}

                for key, values in result['metrics'].items():
                    if values[metric_type] is not None:
                        processed[split][metric_type][epoch][key].append(values[metric_type])

        return processed

    @staticmethod
    def plot_metrics(processed_results, metric_type='loss', refined=False):
        """Generate publication-quality plots for the specified metric"""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 16
        })

        # Changed from 2x2 grid to 1x4 grid (all plots in a single row)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'{"Refined " if refined else ""}Model {metric_type.upper()} Over Training Epochs',
                     y=1.05)

        categories = ['awmh', 'nwmh', 'vent', 'main']
        titles = ['Abnormal WMH', 'Normal WMH', 'Ventricle', 'Main Prediction']

        for idx, (cat, title) in enumerate(zip(categories, titles)):
            ax = axes[idx]

            # Skip if no data for this category (e.g., main mask for AUC)
            if metric_type == 'auc' and cat == 'main':
                ax.text(0.5, 0.5, 'AUC not applicable\nfor multi-class mask',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # Get all available epochs for both train and test
            train_epochs = sorted(processed_results['train'][metric_type].keys())
            test_epochs = sorted(processed_results['test'][metric_type].keys())
            all_epochs = sorted(set(train_epochs + test_epochs))

            if not all_epochs:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # Prepare data for plotting
            train_means = []
            test_means = []

            for epoch in all_epochs:
                # Training data
                if epoch in train_epochs and cat in processed_results['train'][metric_type][epoch]:
                    train_data = processed_results['train'][metric_type][epoch][cat]
                    if train_data:
                        train_means.append(np.mean(train_data))
                    else:
                        train_means.append(None)
                else:
                    train_means.append(None)

                # Testing data
                if epoch in test_epochs and cat in processed_results['test'][metric_type][epoch]:
                    test_data = processed_results['test'][metric_type][epoch][cat]
                    if test_data:
                        test_means.append(np.mean(test_data))
                    else:
                        test_means.append(None)
                else:
                    test_means.append(None)

            # Plot
            if any(x is not None for x in train_means):
                ax.plot(all_epochs, train_means, 'b-', label='Training',
                        linewidth=2, marker='.')
            if any(x is not None for x in test_means):
                ax.plot(all_epochs, test_means, 'r-', label='Testing',
                        linewidth=2, marker='.')

            ax.set_title(title)
            ax.set_xlabel('Epoch')
            
            # Updated y-axis label for AUC plots
            if metric_type == 'auc':
                ax.set_ylabel('AUC (of P-R Curve)', fontweight='bold')
            else:
                ax.set_ylabel(f'{metric_type.capitalize()}', fontweight='bold')
            
            # Set x-axis ticks to increment by 4
            min_epoch = min(all_epochs) if all_epochs else 1
            max_epoch = max(all_epochs) if all_epochs else 50
            
            # Create some padding on both sides
            x_min = max(0, min_epoch - 1)  # At least 0
            x_max = max_epoch + 1  # Add one extra tick beyond the max epoch

            # Generate ticks from x_min to x_max incrementing by 2
            # Starting from 2 and even numbers only
            if x_min % 2 == 1:  # If x_min is odd
                x_min = x_min - 1 if x_min > 0 else 0
            ticks = range(x_min+4, x_max + 1, 4)  # +2 to not show 0; and +1 to ensure x_max is included if it's even

            ax.set_xticks(ticks)
            ax.set_xlim(x_min, x_max)
                
            ax.legend()
            ax.grid(True, alpha=0.3)

            if metric_type == 'auc':
                ax.set_ylim(0, 1)

        plt.tight_layout()
        return fig

def main(base_dir='/home/sai/challenge/codes/paper2/data', pred='our_bs1', epochs_all=50):

    output_dir = os.path.join(os.path.dirname(base_dir), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Example of parallel processing for different epoch ranges
    computer = MetricsComputer(base_dir)
    
    # Process epochs in parallel on different machines
    # Machine 1 could run:

    from_epoch = 1
    to_epoch = epochs_all // 2
    results_1 = computer.process_epoch_range(from_epoch, to_epoch, pred, refined=False, num_processes=4)
    sub_path1 = f"results_{from_epoch}_{to_epoch}.pkl"
    save_results(results_1, os.path.join(output_dir, sub_path1))

    # # Machine 2 could run:
    from_epoch = epochs_all // 2 + 1
    to_epoch = epochs_all
    results_2 = computer.process_epoch_range(from_epoch, to_epoch, pred, refined=False, num_processes=4)
    sub_path2 = f"results_{from_epoch}_{to_epoch}.pkl"
    save_results(results_2, os.path.join(output_dir, sub_path2))

    # Later, combine results and plot
    plotter = MetricsPlotter()
    all_results = plotter.load_results([
        os.path.join(output_dir, sub_path1),
        os.path.join(output_dir, sub_path2)
    ])

    processed_results = plotter.process_results_for_plotting(all_results)

    loss_fig = plotter.plot_metrics(processed_results, metric_type='loss', refined=False)
    loss_fig.savefig(os.path.join(output_dir, 'loss_plots.png'), dpi=300, bbox_inches='tight')

    auc_fig = plotter.plot_metrics(processed_results, metric_type='auc', refined=False)
    auc_fig.savefig(os.path.join(output_dir, 'auc_plots.png'), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total Elapsed Time of Evaluation: {time.time() - start_time}")