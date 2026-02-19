"""
P2 Article - Compute Class Weights from Training Data
Utility script to calculate inverse frequency weights for class balancing

Usage:
    python p2_compute_class_weights.py --fold 0 --scenario 4class --preprocessing standard
    
Output:
    Saves class weights to JSON file for reproducibility
    Prints weights for use in training
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Import data loader
from p2_data_loader import DataConfig, P2DataLoader


def compute_class_frequencies(dataset, num_classes, total_samples=None):
    """
    Compute class frequencies from dataset
    
    Args:
        dataset: TensorFlow dataset yielding (paired_input, target_mask)
        num_classes: Number of classes (3 or 4)
        total_samples: Total number of samples (for progress bar)
        
    Returns:
        class_pixel_counts: Array of pixel counts per class
        total_pixels: Total number of pixels analyzed
    """
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    
    print(f"Computing class frequencies for {num_classes}-class scenario...")
    
    iterator = tqdm(dataset, total=total_samples, desc="Processing") if total_samples else dataset
    
    for paired_input, target_mask in iterator:
        # target_mask shape: (batch_size, 256, 256)
        masks = target_mask.numpy()
        
        for mask in masks:
            # Count pixels for each class
            for class_id in range(num_classes):
                class_pixel_counts[class_id] += np.sum(mask == class_id)
            
            total_pixels += mask.size
    
    return class_pixel_counts, total_pixels


def compute_inverse_frequency_weights(class_pixel_counts, num_classes):
    """
    Compute inverse frequency weights with normalization
    
    Args:
        class_pixel_counts: Array of pixel counts per class
        num_classes: Number of classes
        
    Returns:
        class_weights: Normalized inverse frequency weights
        class_frequencies: Class frequencies (for reference)
    """
    total_pixels = np.sum(class_pixel_counts)
    
    # Class frequencies
    class_frequencies = class_pixel_counts / total_pixels
    
    # Inverse frequency (with small epsilon to avoid division by zero)
    epsilon = 1e-6
    inverse_freq = 1.0 / (class_frequencies + epsilon)
    
    # Normalize weights to sum = num_classes
    # This keeps weights in a reasonable range while maintaining relative importance
    class_weights = inverse_freq / np.sum(inverse_freq) * num_classes
    
    return class_weights, class_frequencies


def compute_and_save_class_weights(fold_id, class_scenario, preprocessing, 
                                   output_dir='class_weights'):
    """
    Compute class weights for a specific fold and scenario
    
    Args:
        fold_id: Fold number (0-4)
        class_scenario: '3class' or '4class'
        preprocessing: 'standard' or 'zoomed'
        output_dir: Directory to save weights
        
    Returns:
        Dictionary with weights and statistics
    """
    print("\n" + "="*70)
    print(f"COMPUTING CLASS WEIGHTS")
    print("="*70)
    print(f"Fold: {fold_id}")
    print(f"Scenario: {class_scenario}")
    print(f"Preprocessing: {preprocessing}")
    print("="*70 + "\n")
    
    # Initialize data loader
    config = DataConfig()
    data_loader = P2DataLoader(config)
    
    # Determine number of classes
    num_classes = 3 if class_scenario == '3class' else 4
    
    # Load training dataset
    print("Loading training dataset...")
    train_dataset = data_loader.create_dataset_for_fold(
        fold_id=fold_id,
        split='train',
        preprocessing=preprocessing,
        class_scenario=class_scenario,
        batch_size=4,  # Larger batch for faster processing
        shuffle=False  # No need to shuffle for counting
    )
    
    # Get dataset size
    train_size = sum(1 for _ in train_dataset)
    print(f"Training samples: {train_size}")
    
    # Recreate dataset after consuming
    train_dataset = data_loader.create_dataset_for_fold(
        fold_id=fold_id,
        split='train',
        preprocessing=preprocessing,
        class_scenario=class_scenario,
        batch_size=4,
        shuffle=False
    )
    
    # Compute class frequencies
    class_pixel_counts, total_pixels = compute_class_frequencies(
        train_dataset, num_classes, train_size
    )
    
    # Compute inverse frequency weights
    class_weights, class_frequencies = compute_inverse_frequency_weights(
        class_pixel_counts, num_classes
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    class_names = {
        3: ['Background', 'Ventricles', 'Abnormal WMH'],
        4: ['Background', 'Ventricles', 'Normal WMH', 'Abnormal WMH']
    }
    
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    print(f"\nClass Statistics:")
    print("-" * 70)
    
    for i in range(num_classes):
        print(f"Class {i} ({class_names[num_classes][i]}):")
        print(f"  Pixel count:  {class_pixel_counts[i]:,}")
        print(f"  Frequency:    {class_frequencies[i]:.6f} ({class_frequencies[i]*100:.2f}%)")
        print(f"  Weight:       {class_weights[i]:.4f}")
        print()
    
    # Save to JSON
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'fold_id': fold_id,
        'class_scenario': class_scenario,
        'preprocessing': preprocessing,
        'num_classes': num_classes,
        'total_pixels': int(total_pixels),
        'class_pixel_counts': class_pixel_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'class_weights': class_weights.tolist(),
        'class_names': class_names[num_classes]
    }
    
    filename = f"class_weights_fold{fold_id}_{preprocessing}_{class_scenario}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("="*70)
    print(f"✅ Class weights saved to: {filepath}")
    print("="*70)
    
    # Print weights in format ready for code
    print("\nFor use in training script:")
    print("-" * 70)
    print(f"class_weights = tf.constant({class_weights.tolist()}, dtype=tf.float32)")
    print()
    
    return results


def compute_all_scenarios_for_fold(fold_id):
    """
    Compute class weights for all 4 scenarios of a given fold
    
    Args:
        fold_id: Fold number (0-4)
    """
    scenarios = [
        {'preprocessing': 'standard', 'class_scenario': '3class'},
        {'preprocessing': 'standard', 'class_scenario': '4class'},
        {'preprocessing': 'zoomed', 'class_scenario': '3class'},
        {'preprocessing': 'zoomed', 'class_scenario': '4class'},
    ]
    
    all_results = {}
    
    for scenario in scenarios:
        results = compute_and_save_class_weights(
            fold_id=fold_id,
            class_scenario=scenario['class_scenario'],
            preprocessing=scenario['preprocessing']
        )
        
        key = f"{scenario['preprocessing']}_{scenario['class_scenario']}"
        all_results[key] = results
        
        print("\n" + "="*70 + "\n")
    
    return all_results


def load_class_weights(fold_id, class_scenario, preprocessing, weights_dir='class_weights'):
    """
    Load previously computed class weights
    
    Args:
        fold_id: Fold number (0-4)
        class_scenario: '3class' or '4class'
        preprocessing: 'standard' or 'zoomed'
        weights_dir: Directory containing weights files
        
    Returns:
        class_weights: NumPy array of weights
    """
    weights_path = Path(weights_dir)
    filename = f"class_weights_fold{fold_id}_{preprocessing}_{class_scenario}.json"
    filepath = weights_path / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Class weights not found: {filepath}\n"
            f"Run compute_and_save_class_weights() first."
        )
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    class_weights = np.array(results['class_weights'], dtype=np.float32)
    
    return class_weights


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Compute class weights from training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single scenario
    python p2_compute_class_weights.py --fold 0 --scenario 4class --preprocessing standard
    
    # All scenarios for one fold
    python p2_compute_class_weights.py --fold 0 --all
    
    # All folds (for completeness)
    python p2_compute_class_weights.py --all-folds
        """
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        choices=[0, 1, 2, 3, 4],
        help='Fold number (0-4)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['3class', '4class'],
        help='Class scenario'
    )
    
    parser.add_argument(
        '--preprocessing',
        type=str,
        choices=['standard', 'zoomed'],
        help='Preprocessing type'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Compute for all scenarios of specified fold'
    )
    
    parser.add_argument(
        '--all-folds',
        action='store_true',
        help='Compute for all scenarios of all folds'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.all_folds:
        # Compute for all folds
        for fold_id in range(5):
            print(f"\n{'='*70}")
            print(f"PROCESSING FOLD {fold_id}")
            print(f"{'='*70}\n")
            compute_all_scenarios_for_fold(fold_id)
    
    elif args.all:
        # Compute all scenarios for one fold
        if args.fold is None:
            parser.error("--fold is required when using --all")
        compute_all_scenarios_for_fold(args.fold)
    
    else:
        # Compute single scenario
        if args.fold is None or args.scenario is None or args.preprocessing is None:
            parser.error("--fold, --scenario, and --preprocessing are required")
        
        compute_and_save_class_weights(
            fold_id=args.fold,
            class_scenario=args.scenario,
            preprocessing=args.preprocessing
        )


if __name__ == "__main__":
    main()
