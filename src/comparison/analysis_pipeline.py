import os
import re
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from segmentation_visualization import SegmentationVisualizer
from segmentation_metrics import SegmentationMetrics, ResultsAnalyzer


class AnalysisPipeline:
    """
    Orchestrates the complete analysis pipeline for both WMH and ventricle segmentation.
    """

    def __init__(self, base_dir: str, output_dir: str):
        """
        Initialize the pipeline with directory paths.

        Args:
            base_dir: Base directory containing all data
            output_dir: Directory to save analysis results
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "stats").mkdir(exist_ok=True)
        (self.output_dir / "overlays").mkdir(exist_ok=True)

        # Define method directories for binary masks
        self.wmh_methods = {
            'BIANCA': self.base_dir / 'WMH/bianca_results_ab',
            # 'BIANCA-AB': self.base_dir / 'WMH/bianca_results_ab',
            'LGA': self.base_dir / 'WMH/LGA',
            'LPA': self.base_dir / 'WMH/LPA',
            'WMH SynthSeg': self.base_dir / 'WMH/WMH_SynthSeg_masks',
            'Our Method': self.base_dir / 'WMH/our_model_all_WMH_035_ep19',
            # 'Our-Method-AB-WMH': self.base_dir / 'WMH/our_model_ab_WMH',
            # 'Our-Method-N-WMH': self.base_dir / 'WMH/our_model_n_WMH'
        }

        self.vent_methods = {
            'Atlas Matching': self.base_dir / 'VENT/Atlas_Matching_masks',
            'SynthSeg': self.base_dir / 'VENT/Vent_SynthSeg_masks',
            'Our Method  ': self.base_dir / 'VENT/our_model_vent_035_ep19'
        }

        # Define method directories for probability masks
        self.wmh_prob_methods = {
            'BIANCA': self.base_dir / 'WMH/bianca_results_ab/prob',
            # 'BIANCA-AB': self.base_dir / 'WMH/bianca_results_ab/prob',
            'LGA': self.base_dir / 'WMH/LGA/prob',
            'LPA': self.base_dir / 'WMH/LPA/prob',
            'WMH SynthSeg': self.base_dir / 'WMH/WMH_SynthSeg_masks/prob',
            'Our Method': self.base_dir / 'WMH/our_model_all_WMH_035_ep19/prob',
            # 'Our-Method-AB-WMH': self.base_dir / 'WMH/our_model_ab_WMH/prob',
            # 'Our-Method-N-WMH': self.base_dir / 'WMH/our_model_n_WMH/prob'
        }

        self.vent_prob_methods = {
            'Atlas Matching': self.base_dir / 'VENT/Atlas_Matching_masks/prob',
            'SynthSeg': self.base_dir / 'VENT/Vent_SynthSeg_masks/prob',
            'Our Method  ': self.base_dir / 'VENT/our_model_vent_035_ep19/prob'
        }

        # Initialize metrics
        self.metrics = SegmentationMetrics()

    def get_subject_ids(self) -> List[str]:
        """Get all subject IDs from the subjects directory."""
        subjects_dir = self.base_dir / 'subjects'
        # Extract 6-digit IDs from subject folder names
        subject_ids = []
        for d in subjects_dir.iterdir():
            if d.is_dir():
                # Extract 6-digit number from folder name
                match = re.search(r'subj_(\d{6})', d.name)
                if match:
                    subject_ids.append(match.group(1))
        return sorted(subject_ids)

    def load_ground_truth(self, subject_id: str, task: str) -> np.ndarray:
        """
        Load ground truth mask for a subject.

        Args:
            subject_id: 6-digit subject identifier
            task: Either 'WMH' or 'VENT'
        """
        subject_dir = self.base_dir / 'subjects' / f'subj_{subject_id}'
        mask_file = f"{subject_id}_{'abWMHmask' if task == 'WMH' else 'VENTmask'}.nii.gz"

        loaded_data = nib.load(str(subject_dir / mask_file)).get_fdata()
        loaded_data = np.where(loaded_data > 0.5, 1, 0).astype(np.uint8)

        # choosing specific slice range: for ventricles [4, 16]
        if task == 'VENT':
            loaded_data[..., :3] = 0
            loaded_data[..., 16:] = 0
        else:
            # choosing specific slice range: for WMH [8, 17]
            loaded_data[..., :7] = 0
            loaded_data[..., 17:] = 0

        return loaded_data

    def load_flair(self, subject_id: str) -> np.ndarray:
        """Load FLAIR image for a subject."""
        subject_dir = self.base_dir / 'subjects' / f'subj_{subject_id}'
        flair_file = f"{subject_id}_FLAIR.nii.gz"
        return nib.load(str(subject_dir / flair_file)).get_fdata()

    def load_prediction(self, subject_id: str, method: str, method_dir: Path) -> np.ndarray:
        """Load prediction mask for a subject from a specific method."""
        # Handle different naming conventions for each method
        method_file_patterns = {
            'BIANCA': f"{subject_id}_bianca_output_thr09bin.nii.gz",
            # 'BIANCA-AB': f"{subject_id}_bianca_output_thr09bin.nii.gz",
            'LGA': f"{subject_id}_lga_wmh.nii.gz",
            'LPA': f"{subject_id}_lpa_wmh.nii.gz",
            'WMH SynthSeg': f"{subject_id}_synthseg_wmh_mask.nii.gz",
            'Our Method': f"{subject_id}_our_awmh_mask.nii.gz",
            # 'Our-Method-AB-WMH': f"{subject_id}_our_awmh_mask.nii.gz",
            # 'Our-Method-N-WMH': f"{subject_id}_our_nwmh_mask.nii.gz",
            'Atlas Matching': f"c3mni{subject_id}_AXFLAIR_vents_on_FLAIR.nii.gz",
            'SynthSeg': f"{subject_id}_T1_to_FLAIR_seg_vent_T1_mask.nii.gz",
            'Our Method  ': f"{subject_id}_our_vent_mask.nii.gz"
        }

        filename = method_file_patterns[method]
        try:
            loaded_data = nib.load(str(method_dir / filename)).get_fdata()
            loaded_data = np.where(loaded_data > 0.5, 1, 0).astype(np.uint8)
            if method == 'Atlas Matching' or method == 'SynthSeg':
                loaded_data = np.flip(loaded_data, axis=1)

            # choosing specific slice range: for ventricles [4, 16]
            if method == 'Atlas Matching' or method == 'SynthSeg' or method == 'Our Method  ':
                loaded_data[..., :3] = 0
                loaded_data[..., 16:] = 0
            else:
                # choosing specific slice range: for WMH [8, 17]
                loaded_data[..., :7] = 0
                loaded_data[..., 17:] = 0

            return loaded_data
        except FileNotFoundError:
            print(f"Warning: Could not find file {method_dir / filename}")
            # Return empty array with same shape as FLAIR
            flair = self.load_flair(subject_id)
            return np.zeros_like(flair, dtype=np.uint8)

    def load_prediction_prob(self, subject_id: str, method: str, method_dir: Path) -> np.ndarray:
        """Load probability prediction mask for a subject from a specific method."""
        # Handle different naming conventions for each method
        method_file_patterns = {
            'BIANCA': f"{subject_id}_bianca_output.nii.gz",
            # 'BIANCA-AB': f"{subject_id}_bianca_output.nii.gz",
            'LGA': f"ples_lga_0.3_rm{subject_id}_F.nii.gz",
            'LPA': f"ples_lpa_m{subject_id}.nii.gz",
            'WMH SynthSeg': f"{subject_id}_synthseg_wmh.nii.gz",
            'Our Method': f"{subject_id}_our_awmh.nii.gz",
            # 'Our-Method-AB-WMH': f"{subject_id}_our_awmh.nii.gz",
            # 'Our-Method-N-WMH': f"{subject_id}_our_nwmh.nii.gz",
            'Atlas Matching': f"c3mni{subject_id}_AXFLAIR_vents_on_FLAIR.nii.gz",
            'SynthSeg': f"{subject_id}_T1_to_FLAIR_seg_vent_T1.nii.gz",
            'Our Method  ': f"{subject_id}_our_vent.nii.gz"
        }

        filename = method_file_patterns[method]
        try:
            loaded_data = nib.load(str(method_dir / filename)).get_fdata()
            if method == 'Atlas Matching' or method == 'SynthSeg':
                loaded_data = np.flip(loaded_data, axis=1)

            # choosing specific slice range: for ventricles [4, 16]
            if method == 'Atlas Matching' or method == 'SynthSeg' or method == 'Our Method  ':
                loaded_data[..., :3] = 0
                loaded_data[..., 16:] = 0
            else:
                # choosing specific slice range: for WMH [8, 17]
                loaded_data[..., :7] = 0
                loaded_data[..., 17:] = 0

            return loaded_data
        except FileNotFoundError:
            print(f"Warning: Could not find file {method_dir / filename}")
            # Return empty array with same shape as FLAIR
            flair = self.load_flair(subject_id)
            return np.zeros_like(flair, dtype=np.float32)

    def evaluate_all_subjects(self, task: str) -> Dict:
        """
        Evaluate all subjects for a specific task.

        Args:
            task: Either 'WMH' or 'VENT'

        Returns:
            Dict: Results for all methods and subjects
        """
        methods = self.wmh_methods if task == 'WMH' else self.vent_methods
        results = {method: {} for method in methods}
        
        subjects = self.get_subject_ids()
        
        for subject_id in subjects:
            ground_truth = self.load_ground_truth(subject_id, task)
            
            for method, method_dir in methods.items():
                prediction = self.load_prediction(subject_id, method, method_dir)
                
                # Flatten masks for metric calculation
                gt_flat = ground_truth.flatten()
                pred_flat = prediction.flatten()
                
                # Calculate metrics
                dice = self.metrics.dice_coefficient(gt_flat, pred_flat)
                jaccard = self.metrics.jaccard_index(gt_flat, pred_flat)
                precision = self.metrics.precision(gt_flat, pred_flat)
                recall = self.metrics.recall(gt_flat, pred_flat)
                # accuracy = self.metrics.accuracy(gt_flat, pred_flat)
                hausdorff95 = self.metrics.hausdorff95(ground_truth, prediction)  # Added Hausdorff95 metric
                
                # Calculate confusion matrix
                confusion_values = self.metrics.get_confusion_matrix(gt_flat, pred_flat)
                
                results[method][subject_id] = {
                    'precision': precision,
                    'recall': recall,
                    # 'accuracy': accuracy,
                    'dice': dice,
                    'jaccard': jaccard,
                    'hausdorff95': hausdorff95,
                    'confusion_matrix': confusion_values.tolist()  # Store confusion matrix values
                }
        
        return results

    def calculate_roc_curves(self, task: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Calculate ROC curves for all methods in a task.

        Args:
            task: Either 'WMH' or 'VENT'

        Returns:
            Dict: Method-wise ROC curve data
        """
        methods = self.wmh_prob_methods if task == 'WMH' else self.vent_prob_methods
        roc_data = {}
        
        subjects = self.get_subject_ids()
        
        for method, method_dir in methods.items():
            all_gt = []
            all_pred = []
            
            for subject_id in subjects:
                ground_truth = self.load_ground_truth(subject_id, task)
                prediction_prob = self.load_prediction_prob(subject_id, method, method_dir)

                # Skip cases with empty or all-NaN masks
                if ground_truth is None or prediction_prob is None or np.all(np.isnan(ground_truth)) or np.all(np.isnan(prediction_prob)):
                    print(f"Skipping case {subject_id} for method {method} due to invalid data")
                    continue

                all_gt.append(ground_truth)
                all_pred.append(prediction_prob)

            if not all_gt or not all_pred:
                print(f"No valid data found for method {method}")
                continue

            # Flatten and concatenate all cases
            try:
                all_gt_flat = np.concatenate([g.flatten() for g in all_gt])
                all_pred_flat = np.concatenate([p.flatten() for p in all_pred])

                # Compute ROC curve
                fpr, tpr, auc_value = self.metrics.compute_roc_curve(all_gt_flat, all_pred_flat)
                roc_data[method] = (fpr, tpr, auc_value)

            except Exception as e:
                print(f"Error calculating ROC curve for method {method}: {e}")
                continue

        return roc_data

    def calculate_pr_curves(self, task: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Calculate PR curves for all methods in a task.

        Args:
            task: Either 'WMH' or 'VENT'

        Returns:
            Dict: Method-wise PR curve data
        """
        methods = self.wmh_prob_methods if task == 'WMH' else self.vent_prob_methods
        pr_data = {}
        
        subjects = self.get_subject_ids()
        
        for method, method_dir in methods.items():
            all_gt = []
            all_pred = []
            
            for subject_id in subjects:
                ground_truth = self.load_ground_truth(subject_id, task)
                prediction_prob = self.load_prediction_prob(subject_id, method, method_dir)

                # Skip cases with empty or all-NaN masks
                if ground_truth is None or prediction_prob is None or np.all(np.isnan(ground_truth)) or np.all(
                        np.isnan(prediction_prob)):
                    print(f"Skipping case {subject_id} for method {method} due to invalid data")
                    continue

                all_gt.append(ground_truth)
                all_pred.append(prediction_prob)

            if not all_gt or not all_pred:
                print(f"No valid data found for method {method}")
                continue

            # Flatten and concatenate all cases
            try:
                all_gt_flat = np.concatenate([g.flatten() for g in all_gt])
                all_pred_flat = np.concatenate([p.flatten() for p in all_pred])

                # Calculate PR
                precision, recall, auc_value = self.metrics.compute_pr_curve(all_gt_flat, all_pred_flat)
                pr_data[method] = (precision, recall, auc_value)

            except Exception as e:
                print(f"Error calculating p-r ROC curve for method {method}: {e}")
                continue

        return pr_data

    def generate_visualizations(self, task: str) -> None:
        """
        Generate visualizations for a task.

        Args:
            task: Either 'WMH' or 'VENT'
        """
        # Evaluate all subjects
        results = self.evaluate_all_subjects(task)
        
        # Create visualizer
        visualizer = SegmentationVisualizer(results)
        
        # Generate boxplots with turquoise color palette
        visualizer.create_boxplots(save_path=str(self.output_dir / "plots" / f"{task.lower()}_boxplots.png"))
        
        # Generate violin plots with pinkish burgundy color palette
        visualizer.create_violin_plots(save_path=str(self.output_dir / "plots" / f"{task.lower()}_violinplots.png"))
        
        # Generate heatmaps for each metric
        for metric in ['dice', 'jaccard', 'precision', 'recall', 'hausdorff95']: #  'accuracy'
            visualizer.create_heatmap(metric, 
                save_path=str(self.output_dir / "plots" / f"{task.lower()}_{metric}_heatmap.png"))
        
        # Calculate and generate ROC curves
        roc_data = self.calculate_roc_curves(task)
        visualizer.create_roc_curves(roc_data, task, 
            save_path=str(self.output_dir / "plots" / f"{task.lower()}_roc_curves.png"))
        
        # Calculate and generate PR curves
        pr_data = self.calculate_pr_curves(task)
        visualizer.create_pr_curves(pr_data, task, 
            save_path=str(self.output_dir / "plots" / f"{task.lower()}_pr_curves.png"))
        
        # Generate confusion matrix plots
        visualizer.create_confusion_matrix_plots(results, task,
            save_path=str(self.output_dir / "plots" / f"{task.lower()}_confusion_matrices.png"))

    def generate_overlay_examples(self, task: str, n_examples: int = 10) -> None:
        """
        Generate example overlays for visual comparison.

        Args:
            task: Either 'WMH' or 'VENT'
            n_examples: Number of examples to generate
        """
        methods = self.wmh_methods if task == 'WMH' else self.vent_methods
        subjects = self.get_subject_ids()[:n_examples]  # Use first n subjects
        
        for subject_id in subjects:
            flair = self.load_flair(subject_id)
            ground_truth = self.load_ground_truth(subject_id, task)
            
            # Determine best slice to visualize (one with most ground truth)
            slice_sums = np.sum(ground_truth, axis=(0, 1))
            best_slice = np.argmax(slice_sums)
            best_slice = 11

            for method, method_dir in methods.items():
                prediction = self.load_prediction(subject_id, method, method_dir)
                
                # Create and save overlay
                save_path = str(self.output_dir / "overlays" / f"{task.lower()}_{subject_id}_{method.replace(' ', '_')}.png")
                SegmentationVisualizer.create_overlay_visualization(
                    flair, ground_truth, prediction, best_slice, save_path=save_path
                )

    def export_results_to_json(self, task: str) -> None:
        """
        Export results to JSON file.

        Args:
            task: Either 'WMH' or 'VENT'
        """
        results = self.evaluate_all_subjects(task)
        
        # Calculate means across subjects
        summary = {}
        for method in results:
            summary[method] = {}
            for metric in ['dice', 'jaccard', 'precision', 'recall', 'hausdorff95']:    #  'accuracy'
                values = [results[method][subj][metric] for subj in results[method]]
                summary[method][f"mean_{metric}"] = float(np.mean(values))
                summary[method][f"std_{metric}"] = float(np.std(values))
        
        # Save detailed results
        with open(str(self.output_dir / "stats" / f"{task.lower()}_detailed_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        with open(str(self.output_dir / "stats" / f"{task.lower()}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    def run_pipeline(self) -> None:
        """Run the complete analysis pipeline for both tasks."""
        start_time = datetime.now()
        print(f"Starting analysis pipeline at {start_time}")

        # Process WMH segmentation
        print("Processing WMH segmentation...")
        self.generate_visualizations('WMH')
        self.generate_overlay_examples('WMH')
        self.export_results_to_json('WMH')

        # Process Ventricle segmentation
        print("Processing Ventricle segmentation...")
        self.generate_visualizations('VENT')
        self.generate_overlay_examples('VENT')
        self.export_results_to_json('VENT')

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Analysis pipeline completed at {end_time}")
        print(f"Total duration: {duration}")


if __name__ == "__main__":
    
    # on Linux machine
    base_dir = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test"
    output_dir = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/analysis_results"

    # on Windows machine
    base_dir = r"C:/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test"
    output_dir = r"C:/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/analysis_results_abWMH_vent_035_ep19"

    pipeline = AnalysisPipeline(base_dir, output_dir)
    pipeline.run_pipeline()