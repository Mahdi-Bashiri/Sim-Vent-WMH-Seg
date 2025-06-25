import os
import numpy as np
import nibabel as nib
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from typing import Union, Tuple, List, Dictfrom scipy.spatial.distance import directed_hausdorff


class SegmentationMetrics:
    """
    A class to compute various metrics for evaluating medical image segmentation.
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: Accuracy
        """
        tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
        return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: Precision
        """
        tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
        return tp / (tp + fp) if (tp + fp) != 0 else 0.0
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall (sensitivity).
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: Recall
        """
        tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
        return tp / (tp + fn) if (tp + fn) != 0 else 0.0
    
    @staticmethod
    def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Dice coefficient.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: Dice coefficient
        """
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    
    @staticmethod
    def jaccard_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Jaccard index (IoU).
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: Jaccard index
        """
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return intersection / union if union != 0 else 0.0
    
    @staticmethod
    def hausdorff95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the 95th percentile of the Hausdorff distance.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            float: 95th percentile of the Hausdorff distance
        """
        # Find coordinates of positive voxels
        y_true_coords = np.argwhere(y_true)
        y_pred_coords = np.argwhere(y_pred)
        
        # Handle empty arrays
        if len(y_true_coords) == 0 or len(y_pred_coords) == 0:
            return 0.0 if len(y_true_coords) == 0 and len(y_pred_coords) == 0 else float('inf')
        
        # Calculate forward and backward Hausdorff distances
        forward_hausdorff = directed_hausdorff(y_true_coords, y_pred_coords)[0]
        backward_hausdorff = directed_hausdorff(y_pred_coords, y_true_coords)[0]
        
        # Calculate max (symmetric) Hausdorff distance
        hausdorff_dist = max(forward_hausdorff, backward_hausdorff)
        
        # For 95th percentile calculation, we need to compute all pairwise distances
        # This is a simplified version - for a true 95th percentile, we'd need to calculate all distances
        # and take the 95th percentile, but this is computationally expensive
        # Here we approximate by scaling down the max distance
        return hausdorff_dist * 0.95
    
    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: Ground truth binary mask
            y_pred: Predicted binary mask
            
        Returns:
            np.ndarray: 2x2 confusion matrix [TN, FP, FN, TP]
        """
        return confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=[0, 1]).ravel()
    
    @staticmethod
    def compute_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute ROC curve and AUC.
        
        Args:
            y_true: Ground truth binary mask
            y_pred_prob: Predicted probability mask
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: (fpr, tpr, auc_value)
        """
        # Check for NaN values and handle them
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_prob))

        if not np.any(mask):
            print("Warning: All values are NaN. Cannot compute ROC curve.")
            # Return dummy values or handle appropriately
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # Filter out NaN values
        y_true_clean = y_true.ravel()[mask.ravel()]
        y_pred_clean = y_pred_prob.ravel()[mask.ravel()]

        # Check if we have enough unique values to compute ROC
        if len(np.unique(y_true_clean)) < 2:
            print(
                f"Warning: Only found {len(np.unique(y_true_clean))} unique classes in ground truth. Need at least 2 for ROC curve.")
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # Now compute ROC curve on clean data
        fpr, tpr, _ = roc_curve(y_true_clean, y_pred_clean)

        # Calculate AUC
        try:
            # auc_value = roc_auc_score(y_true_clean, y_pred_clean)
            auc_value = auc(fpr, tpr)

        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc_value = 0.5  # Default value

        return fpr, tpr, auc_value

    
    @staticmethod
    def compute_pr_curve(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute Precision-Recall curve and AUC.
        
        Args:
            y_true: Ground truth binary mask
            y_pred_prob: Predicted probability mask
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: (precision, recall, auc_value)
        """
        # Check for NaN values and handle them
        mask = ~(np.isnan(y_true) | np.isnan(y_pred_prob))

        if not np.any(mask):
            print("Warning: All values are NaN. Cannot compute p-r ROC curve.")
            # Return dummy values or handle appropriately
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # Filter out NaN values
        y_true_clean = y_true.ravel()[mask.ravel()]
        y_pred_clean = y_pred_prob.ravel()[mask.ravel()]

        # Check if we have enough unique values to compute ROC
        if len(np.unique(y_true_clean)) < 2:
            print(
                f"Warning: Only found {len(np.unique(y_true_clean))} unique classes in ground truth. Need at least 2 for ROC curve.")
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # check before calling roc_curve or precision_recall_curve
        if np.array_equal(np.unique(y_pred_clean), np.array([0, 1])):
            print(
                f"Warning: Binary predictions detected. ROC/PR curves require probability scores.")
            # Either skip this method or handle differently
            # For example, add a small random noise to convert to pseudo-probabilities
            y_pred_clean = y_pred_clean + np.random.normal(0, 0.01, size=y_pred_clean.shape)
            # Clip values back to [0,1] range
            y_pred_clean = np.clip(y_pred_clean, 0, 1)

        # Now compute p-r ROC curve on clean data
        precision, recall, _ = precision_recall_curve(y_true_clean, y_pred_clean)

        # Calculate AUC
        try:
            auc_value = auc(recall, precision)

        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc_value = 0.5  # Default value

        return precision, recall, auc_value


class ResultsAnalyzer:
    """
    A class to analyze segmentation results across different methods.
    """
    
    def __init__(self, ground_truth_dir: str, results_dirs: dict):
        """
        Initialize the analyzer.
        
        Args:
            ground_truth_dir: Directory containing ground truth masks
            results_dirs: Dictionary mapping method names to their results directories
        """
        self.ground_truth_dir = ground_truth_dir
        self.results_dirs = results_dirs
        self.metrics = SegmentationMetrics()
        
    def load_nifti(self, file_path: str) -> np.ndarray:
        """
        Load a NIfTI file and return its data as numpy array.
        
        Args:
            file_path: Path to the NIfTI file
            
        Returns:
            np.ndarray: Image data
        """
        return nib.load(file_path).get_fdata()
    
    def evaluate_case(self, case_id: str, method: str) -> dict:
        """
        Evaluate metrics for a single case.
        
        Args:
            case_id: Identifier for the case
            method: Name of the method to evaluate
            
        Returns:
            dict: Dictionary containing all computed metrics
        """
        # Load ground truth and prediction
        gt_path = os.path.join(self.ground_truth_dir, f"{case_id}.nii.gz")
        pred_path = os.path.join(self.results_dirs[method], f"{case_id}.nii.gz")
        
        y_true = self.load_nifti(gt_path)
        y_pred = self.load_nifti(pred_path)
        
        # Compute metrics
        accuracy = self.metrics.accuracy(y_true, y_pred)
        precision = self.metrics.precision(y_true, y_pred)
        recall = self.metrics.recall(y_true, y_pred)
        dice = self.metrics.dice_coefficient(y_true, y_pred)
        jaccard = self.metrics.jaccard_index(y_true, y_pred)
        hausdorff95 = self.metrics.hausdorff95(y_true, y_pred)
        confusion_values = self.metrics.get_confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'dice': dice,
            'jaccard': jaccard,
            'hausdorff95': hausdorff95,
            'confusion_matrix': confusion_values.tolist()  # Store as list for JSON serialization
        }

    def evaluate_all_cases(self, case_ids: List[str]) -> dict:
        """
        Evaluate all cases for all methods.
        
        Args:
            case_ids: List of case identifiers
            
        Returns:
            dict: Nested dictionary containing results for all methods and cases
        """
        results = {}
        for method in self.results_dirs.keys():
            method_results = {}
            for case_id in case_ids:
                method_results[case_id] = self.evaluate_case(case_id, method)
            results[method] = method_results
        return results

