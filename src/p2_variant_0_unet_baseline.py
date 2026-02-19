"""
P2 Baseline - U-Net with Weighted Categorical Cross-Entropy Loss

Simple baseline for comparison with pix2pix variants.

Features:
- U-Net architecture (same as pix2pix generator)
- Weighted Categorical Cross-Entropy loss
- One-hot encoded targets
- Class weight computation per fold
- No GAN components (pure segmentation)
"""

import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from unet_model import build_unet_3class

# Import data loader
from p2_data_loader import DataConfig, P2DataLoader


# Import class weights utility
from p2_compute_class_weights import compute_and_save_class_weights, load_class_weights

print("TensorFlow Version:", tf.__version__)

###################### GPU Configuration ######################

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("✅ GPU memory growth enabled")
        print(f"   Available GPUs: {len(physical_devices)}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠️  No GPU detected - training will be slow")
 
"""
GPU Memory Management for Sequential Experiments
To properly release memory between experiments
"""

import gc
from tensorflow.keras import backend as K

def clear_gpu_memory():
    """
    Comprehensive GPU memory cleanup between experiments
    Call this after each experiment completes
    """
    print("\n" + "="*70)
    print("CLEANING UP GPU MEMORY")
    print("="*70)
    
    # Clear Keras session
    K.clear_session()
    print("✅ Cleared Keras session")
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    print("✅ Ran garbage collection (3 passes)")
    
    # Reset TensorFlow graphs
    tf.compat.v1.reset_default_graph()
    print("✅ Reset default graph")
    
    # Additional cleanup for TF 2.x
    try:
        # Clear any cached tensors
        tf.config.experimental.reset_memory_stats('GPU:0')
        print("✅ Reset GPU memory stats")
    except:
        pass
    
    # CRITICAL: Reset GPU memory allocator
    # This forces TensorFlow to release memory back to the system
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            # Disable and re-enable memory growth to flush allocator
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, False)
                tf.config.experimental.set_memory_growth(device, True)
            print("✅ Reset memory growth (flushed allocator)")
    except Exception as e:
        print(f"⚠️  Could not reset memory growth: {e}")
    
    print("="*70 + "\n")


def get_gpu_memory_info():
    """
    Print current GPU memory usage
    Useful for monitoring memory leaks
    """
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            for device in gpu_devices:
                details = tf.config.experimental.get_memory_info(device.name.replace('/physical_device:', ''))
                current_mb = details['current'] / 1024**2
                peak_mb = details['peak'] / 1024**2
                print(f"GPU Memory - Current: {current_mb:.1f} MB, Peak: {peak_mb:.1f} MB")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")


###################### Target Preparation ######################

def prepare_inputs(paired_input, target_mask, num_classes):
    """
    Prepare inputs for training
    
    Args:
        paired_input: (bs, 256, 512, 1) with FLAIR + mask
        target_mask: (bs, 256, 256) with class labels [0, num_classes-1]
        num_classes: number of classes
        
    Returns:
        flair_normalized: FLAIR normalized to [-1, 1]
        target_onehot: One-hot encoded mask (bs, 256, 256, num_classes)
    """
    # Extract FLAIR, previously normalized to [-1, 1]
    flair_normalized = paired_input[:, :, :256, :]

    # One-hot encode target
    target_onehot = tf.one_hot(target_mask, depth=num_classes, dtype=tf.float32)
    
    return flair_normalized, target_onehot

###################### Metrics Calculation ######################

def compute_classwise_metrics(all_val_true, all_val_pred, num_classes, exclude_class=None):
    """
    Compute class-wise Dice, Precision, and Recall for validation predictions.
    
    Args:
        all_val_true: List of one-hot encoded ground truth tensors
        all_val_pred: List of softmax output tensors from model
        num_classes: Number of classes (3 or 4)
        exclude_class: Class to exclude from metric calculation (e.g., 2 for background)
    
    Returns:
        Dictionary containing class-wise and mean metrics
    """
    # Concatenate all batches
    y_true_concat = tf.concat(all_val_true, axis=0)  # Shape: (N, H, W, num_classes)
    y_pred_concat = tf.concat(all_val_pred, axis=0)  # Shape: (N, H, W, num_classes)
    
    # Flatten spatial dimensions: (N*H*W, num_classes)
    y_true_flat = tf.reshape(y_true_concat, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred_concat, [-1, num_classes])
    
    # Convert predictions to one-hot (argmax)
    y_pred_classes = tf.argmax(y_pred_flat, axis=-1)
    y_pred_onehot = tf.one_hot(y_pred_classes, depth=num_classes)
    
    # Convert to numpy for easier computation
    y_true_np = y_true_flat.numpy()
    y_pred_np = y_pred_onehot.numpy()
    
    metrics = {
        'dice': {},
        'precision': {},
        'recall': {}
    }
    
    classes_to_evaluate = [c for c in range(num_classes) if c != exclude_class]
    
    for class_idx in classes_to_evaluate:
        # Extract binary masks for this class
        true_class = y_true_np[:, class_idx]
        pred_class = y_pred_np[:, class_idx]
        
        # True Positives, False Positives, False Negatives
        TP = np.sum((true_class == 1) & (pred_class == 1))
        FP = np.sum((true_class == 0) & (pred_class == 1))
        FN = np.sum((true_class == 1) & (pred_class == 0))
        
        # Dice Score: 2*TP / (2*TP + FP + FN)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        
        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP + 1e-7)
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = TP / (TP + FN + 1e-7)
        
        metrics['dice'][f'class_{class_idx}'] = float(dice)
        metrics['precision'][f'class_{class_idx}'] = float(precision)
        metrics['recall'][f'class_{class_idx}'] = float(recall)
    
    # Compute mean metrics (excluding the excluded class)
    metrics['dice']['mean'] = np.mean([v for v in metrics['dice'].values()])
    metrics['precision']['mean'] = np.mean([v for v in metrics['precision'].values()])
    metrics['recall']['mean'] = np.mean([v for v in metrics['recall'].values()])
    
    return metrics

###################### Experiment Configuration ######################

class ExperimentConfig:
    """Configuration for U-Net baseline experiment"""
    
    def __init__(self, 
                 variant: int = 0,
                 preprocessing: str = 'standard',
                 class_scenario: str = '4class',
                 fold_id: int = 0):
        
        # Experiment identification
        self.variant = variant
        self.preprocessing = preprocessing  # 'standard' or 'zoomed'
        self.class_scenario = class_scenario  # '3class' or '4class'
        self.fold_id = fold_id
        
        # Experiment name
        self.exp_name = f"exp_unet_baseline_{preprocessing}_{class_scenario}_fold{fold_id}"
        
        # Number of classes
        self.num_classes = 3 if class_scenario == '3class' else 4
        
        # Training hyperparameters
        self.batch_size = 4
        self.img_width = 256
        self.img_height = 256
        self.epochs = 60
        
        # Optimizer parameters
        self.learning_rate = 2e-4
        self.beta_1 = 0.9

        # Learning rate schedule parameters
        self.use_lr_schedule = True
        self.lr_schedule_type = 'plateau'  # Options: 'exponential', 'plateau', 'cosine'

        # ReduceLROnPlateau parameters
        self.lr_patience = 5          # Wait 5 epochs before reducing
        self.lr_reduction_factor = 0.5  # Reduce LR by half
        self.lr_min = 1e-6            # Don't go below this
        self.lr_monitor = 'val_loss'  # Or 'val_dice_mean'
        
        # Paths
        self.results_dir = Path(f"results_fold_{fold_id}_var_{variant}_bet_zscore")
        self.models_dir = self.results_dir / "models" / f"{preprocessing}_{class_scenario}"
        self.figures_dir = self.results_dir / "figures" / f"{preprocessing}_{class_scenario}" / f"fold_{fold_id}"
        self.logs_dir = self.results_dir / "logs" / f"{preprocessing}_{class_scenario}" / f"fold_{fold_id}"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint configuration
        self.checkpoint_dir = self.models_dir / f"fold_{fold_id}"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Class weights directory
        self.weights_dir = Path("class_weights")
        self.weights_dir.mkdir(exist_ok=True)

        # Save configuration
        self.save_config()
    
    def save_config(self):
        """Save experiment configuration to JSON"""
        config_dict = {
            'variant': self.variant,
            'variant_name': 'UNet_Baseline',
            'preprocessing': self.preprocessing,
            'class_scenario': self.class_scenario,
            'fold_id': self.fold_id,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'loss': 'Weighted Categorical Cross-Entropy'
        }
        
        config_file = self.checkpoint_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

###################### Loss Functions ######################

def weighted_categorical_crossentropy(y_true, y_pred, class_weights, exclude_class=None):
    """
    Weighted categorical cross-entropy loss
    
    Args:
        y_true: (bs, 256, 256, num_classes) one-hot encoded
        y_pred: (bs, 256, 256, num_classes) softmax probabilities
        class_weights: (num_classes,) weight per class
        exclude_class: Optional int, class index to exclude from loss (e.g., 2 for CSF)
    
    Returns:
        Scalar loss value
    """
    # Clip predictions to prevent log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Cross-entropy per pixel: -sum(y_true * log(y_pred))
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  # (bs, 256, 256)
    
    # Apply class weights
    # class_weights shape: (num_classes,) -> (1, 1, 1, num_classes) for broadcasting
    weights_tensor = tf.cast(class_weights, dtype=tf.float32)
    weights_tensor = tf.reshape(weights_tensor, [1, 1, 1, -1])
    
    # Weight map: (bs, 256, 256)
    pixel_weights = tf.reduce_sum(y_true * weights_tensor, axis=-1)
    
    # Weighted cross-entropy
    # Exclude specific class if specified
    if exclude_class is not None:
        class_mask = tf.argmax(y_true, axis=-1)  # (bs, 256, 256)
        valid_mask = tf.cast(class_mask != exclude_class, tf.float32)
        weighted_ce = ce * pixel_weights * valid_mask
        return tf.reduce_sum(weighted_ce) / (tf.reduce_sum(valid_mask) + 1e-7)
    else:
        weighted_ce = ce * pixel_weights
        return tf.reduce_mean(weighted_ce)

###################### Training Functions ######################

@tf.function
def train_step(input_image, target_onehot, model, optimizer, class_weights):
    """
    Single training step for U-Net
    
    Args:
        input_image: Input FLAIR (bs, 256, 256, 1) in [-1, 1]
        target_onehot: Target mask (bs, 256, 256, num_classes) one-hot
        model: U-Net model
        optimizer: Optimizer
        class_weights: (num_classes,) weight per class
    
    Returns:
        loss: Training loss value
    """
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(input_image, training=True)
        
        # Compute loss
        loss = weighted_categorical_crossentropy(target_onehot, predictions, class_weights)
    
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

def generate_and_save_images(model, test_input, test_target, 
                            epoch, save_path, num_classes):
    """
    Generate predictions and save visualization
    
    Args:
        model: U-Net model
        test_input: Test input image (bs, 256, 512, 1)
        test_target: Test target mask (bs, 256, 256)
        epoch: Current epoch number
        save_path: Path to save figure
        num_classes: Number of classes
    """
    for ik in range(test_input.numpy().shape[0]):
        # Extract FLAIR
        flair_normalized = test_input[ik, :, :256, :]
        flair_normalized = tf.expand_dims(flair_normalized, axis=0)
        
        # Generate prediction
        prediction_softmax = model(flair_normalized, training=False)
        
        # Convert to class labels
        pred_classes = tf.argmax(prediction_softmax, axis=-1).numpy()
        target_mask = test_target[ik].numpy()
        
        # Create figure
        plt.figure(figsize=(20, 5))
        
        # Input FLAIR
        plt.subplot(1, 5, 1)
        plt.title('Input FLAIR')
        plt.imshow(flair_normalized[0, :, :, 0], cmap='gray')
        plt.axis('off')
        
        # Ground truth
        plt.subplot(1, 5, 2)
        plt.title('Ground Truth')
        plt.imshow(target_mask, cmap='jet', vmin=0, vmax=num_classes-1)
        plt.colorbar()
        plt.axis('off')
        
        # Prediction
        plt.subplot(1, 5, 3)
        plt.title('Predicted Classes')
        plt.imshow(pred_classes[0], cmap='jet', vmin=0, vmax=num_classes-1)
        plt.colorbar()
        plt.axis('off')
        
        # Class probabilities for most confident prediction
        plt.subplot(1, 5, 4)
        plt.title('Max Probability')
        max_prob = tf.reduce_max(prediction_softmax[0], axis=-1).numpy()
        plt.imshow(max_prob, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        
        # Difference map
        plt.subplot(1, 5, 5)
        plt.title('Error Map (Red=Wrong)')
        error_map = (pred_classes[0] != target_mask).astype(float)
        plt.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / f'epoch_{epoch:03d}_{ik+1}.png', dpi=300, bbox_inches='tight')
        plt.close()

###################### Main Training Function ######################

def train_baseline_unet(config: ExperimentConfig):
    """
    Main training function for baseline U-Net
    
    Args:
        config: ExperimentConfig object
    """
    print("\n" + "="*70)
    print(f"TRAINING BASELINE U-NET: {config.exp_name}")
    print("="*70)
    print(f"Variant: {config.variant}")
    print(f"Preprocessing: {config.preprocessing}")
    print(f"Class scenario: {config.class_scenario} ({config.num_classes} classes)")
    print(f"Fold: {config.fold_id}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Loss: Weighted Categorical Cross-Entropy")
    print("="*70 + "\n")

    # Check initial GPU memory
    get_gpu_memory_info()
    
    # Initialize data loader
    data_config = DataConfig()
    data_loader = P2DataLoader(data_config)
    
    # Load datasets
    print("Loading training data...")
    train_dataset = data_loader.create_dataset_for_fold(
        fold_id=config.fold_id,
        split='train',
        preprocessing=config.preprocessing,
        class_scenario=config.class_scenario,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    print("Loading validation data...")
    val_dataset = data_loader.create_dataset_for_fold(
        fold_id=config.fold_id,
        split='val',
        preprocessing=config.preprocessing,
        class_scenario=config.class_scenario,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Get dataset sizes
    train_size = tf.data.experimental.cardinality(train_dataset).numpy()
    val_size = tf.data.experimental.cardinality(val_dataset).numpy()
    
    # If cardinality returns -2 (unknown), fall back to counting
    if train_size < 0:
        train_size = sum(1 for _ in train_dataset)
        # Rebuild if we had to count
        train_dataset = data_loader.create_dataset_for_fold(
            fold_id=config.fold_id, split='train',
            preprocessing=config.preprocessing,
            class_scenario=config.class_scenario,
            batch_size=config.batch_size, shuffle=True
        )
    if val_size < 0:
        val_size = sum(1 for _ in val_dataset)
        # Rebuild if we had to count
        val_dataset = data_loader.create_dataset_for_fold(
            fold_id=config.fold_id, split='val',
            preprocessing=config.preprocessing,
            class_scenario=config.class_scenario,
            batch_size=config.batch_size, shuffle=False
        )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}\n")
    
    # Compute or load class weights
    print("Computing class weights from training data...")
    try:
        class_weights = load_class_weights(
            config.fold_id, config.class_scenario, 
            config.preprocessing, config.weights_dir
        )
        print("✅ Loaded pre-computed class weights")
    except FileNotFoundError:
        print("Computing class weights (this may take a few minutes)...")
        results = compute_and_save_class_weights(
            config.fold_id, config.class_scenario, 
            config.preprocessing, str(config.weights_dir)
        )
        class_weights = np.array(results['class_weights'], dtype=np.float32)
    
    print(f"Class weights: {class_weights}")

    # Build model
    print("\n🏗️  Building U-Net model...")
    model = build_unet_3class(input_shape=(256, 256, 1), num_classes=config.num_classes)
    
    print(f"Model parameters: {model.count_params():,}\n")
        
    # Optimizer (will be updated with ReduceLROnPlateau)
    optimizer = tf.keras.optimizers.legacy.Adam(
        config.learning_rate, beta_1=config.beta_1
    )
    print(f"Initial learning rate: {config.learning_rate}")

    # ReduceLROnPlateau tracking
    if config.use_lr_schedule and config.lr_schedule_type == 'plateau':
        plateau_counter = 0
        best_monitored_value = float('inf') if config.lr_monitor == 'val_loss' else 0.0
        current_lr = config.learning_rate
        print(f"✅ Using ReduceLROnPlateau (patience={config.lr_patience}, factor={config.lr_reduction_factor})")
    
    # Initialize optimizer variables
    print("Initializing optimizer variables...")
    dummy_input = tf.zeros((1, 256, 256, 1))
    
    with tf.GradientTape() as tape:
        output = model(dummy_input, training=True)
        dummy_loss = tf.reduce_mean(output)

    # Apply dummy gradients to build optimizer variables
    grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("✅ Optimizer variables initialized\n")
    
    # Checkpoint
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )
    
    checkpoint_prefix = config.checkpoint_dir / "ckpt"
    manager = tf.train.CheckpointManager(
        checkpoint, config.checkpoint_dir, max_to_keep=1
    )
    
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"✅ Restored from checkpoint: {manager.latest_checkpoint}\n")
    else:
        print("Starting training from scratch\n")
    
    # Get example for visualization
    skip_n = 1 # min(100 // config.batch_size, val_size - 1)
    example_paired, example_target = next(iter(val_dataset.skip(skip_n).take(20)))
    
    print("Initializing metrics computer...")
    if config.num_classes == 4:
        class_names = ['Background', 'Ventricles', 'Normal_WMH', 'Abnormal_WMH']
    elif config.num_classes == 3:
        class_names = ['Background', 'Ventricles', 'Abnormal_WMH']

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    best_val_dice = 0.0
    exclude_class = 2 if config.num_classes == 4 else None  # Exclude class 2 only in 4-class
    
    try:
        for epoch in range(config.epochs):
            start_time = time.time()
            
            # Training metrics
            epoch_losses = []
            
            # Training loop
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            train_bar = tqdm(train_dataset, total=train_size, desc="Training")
            
            for paired_input, target_mask in train_bar:
                # Prepare inputs: normalize FLAIR + one-hot encode target
                flair_normalized, target_onehot = prepare_inputs(
                    paired_input, target_mask, config.num_classes
                )
                
                # Train step
                loss = train_step(
                    flair_normalized, target_onehot,
                    model, optimizer, class_weights
                )
                
                epoch_losses.append(loss.numpy())
                
                # Update progress bar
                train_bar.set_postfix({
                    'loss': f"{loss.numpy():.4f}"
                })
            
            # Calculate epoch average
            avg_train_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_losses = []
            all_val_true = []
            all_val_pred = []
            
            for val_paired, val_target in val_dataset:
                try:
                    val_flair_norm, val_target_onehot = prepare_inputs(
                        val_paired, val_target, config.num_classes
                    )
                    val_pred = model(val_flair_norm, training=False)
                    val_loss = weighted_categorical_crossentropy(
                        val_target_onehot, val_pred, class_weights, 
                        exclude_class=exclude_class
                    )

                    # Store true and prediction values for metrics calculation
                    all_val_true.append(val_target_onehot)
                    all_val_pred.append(val_pred)
                    
                    if not tf.math.is_nan(val_loss):
                        val_losses.append(val_loss.numpy())
                except:
                    continue
            
            if len(val_losses) > 0:
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Compute class-wise metrics
                val_metrics = compute_classwise_metrics(
                    all_val_true, all_val_pred, 
                    config.num_classes
                )
                history['val_metrics'].append(val_metrics)
                
                # Print validation results
                epoch_time = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{config.epochs} Summary (Time: {epoch_time:.2f}s)")
                print(f"{'='*70}")
                print(f"Training Loss: {avg_train_loss:.4f}")
                print(f"Validation Loss: {avg_val_loss:.4f}")
                print(f"\nClass-wise Dice Scores:")
                for class_name, dice_val in val_metrics['dice'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {dice_val:.4f}")
                        if class_name == f"class_{config.num_classes - 1}":
                            abwmh_val_dice = dice_val
                        elif class_name == f"class_1":
                            vent_val_dice = dice_val
                print(f"  Mean Dice: {val_metrics['dice']['mean']:.4f}")
                print(f"\nClass-wise Precision:")
                for class_name, prec_val in val_metrics['precision'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {prec_val:.4f}")
                print(f"  Mean Precision: {val_metrics['precision']['mean']:.4f}")
                print(f"\nClass-wise Recall:")
                for class_name, rec_val in val_metrics['recall'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {rec_val:.4f}")
                print(f"  Mean Recall: {val_metrics['recall']['mean']:.4f}")
                print(f"{'='*70}\n")
                
                # Save best model based on overall validation performance
                overal_val_performance = 0.6 * abwmh_val_dice + 0.3 * vent_val_dice + 0.1 * (1 - 100*avg_val_loss)
                if overal_val_performance > best_val_dice:
                    best_val_dice = overal_val_performance
                    model.save_weights(f"{config.checkpoint_dir}/best_dice_model.h5")
                    print(f"✓ Best model saved (performance: {best_val_dice:.4f})")
            else:
                print("Warning: No valid validation batches")
                history['val_loss'].append(float('nan'))
                history['val_metrics'].append({})

            # ReduceLROnPlateau logic
            if config.use_lr_schedule and config.lr_schedule_type == 'plateau':
                # Get monitored value
                if config.lr_monitor == 'val_loss':
                    monitored_value = avg_val_loss
                    improved = monitored_value < best_monitored_value
                else:  # 'val_dice_mean'
                    monitored_value = val_metrics['dice']['mean']
                    improved = monitored_value > best_monitored_value
                
                if improved:
                    best_monitored_value = monitored_value
                    plateau_counter = 0
                    print(f"  📈 Validation improved: {config.lr_monitor}={monitored_value:.4f}")
                else:
                    plateau_counter += 1
                    print(f"  📉 No improvement for {plateau_counter}/{config.lr_patience} epochs")
                    
                    if plateau_counter >= config.lr_patience:
                        old_lr = current_lr
                        current_lr = max(current_lr * config.lr_reduction_factor, config.lr_min)
                        optimizer.learning_rate.assign(current_lr)
                        plateau_counter = 0  # Reset counter
                        print(f"  🔽 Reducing LR: {old_lr:.6f} → {current_lr:.6f}")
                
                print(f"  Current LR: {current_lr:.6f}")
                
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                manager.save()
                print(f"  💾 Saved checkpoint")
            
            # Generate sample images
            if (epoch + 1) % 5 == 0 or epoch == 0 or True:
                generate_and_save_images(
                    model, example_paired, example_target,
                    epoch + 1, config.figures_dir, config.num_classes
                )
                print(f"  📊 Saved visualization")
        
        # Save final model
        final_model_path = config.checkpoint_dir / "final_model.h5"
        model.save(final_model_path)
        print(f"\n✅ Training complete! Final model saved to {final_model_path}")
        
        # Save history
        history_serializable = {
            key: [float(val) if isinstance(val, (int, float, np.number)) else val 
                  for val in values]
            for key, values in history.items()
        }
        
        history_file = config.checkpoint_dir / "history.json"
        with open(history_file, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        return history, history_file
    
    finally:
        # CRITICAL: Always cleanup, even if training fails
        print("\n🧹 Cleaning up resources...")

        # Delete models explicitly to break references
        try:
            del model
            del optimizer
            del checkpoint
            del manager
            del train_dataset
            del val_dataset
            print("✅ Deleted model objects")
        except Exception as e:
            print(f"⚠️  Error deleting objects: {e}")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Check final GPU memory
        get_gpu_memory_info()

###################### Main Execution ######################

if __name__ == "__main__":
    # Example: Train baseline U-Net for 3-class, zoomed preprocessing, fold 0
    config = ExperimentConfig(
        variant=0,
        preprocessing='standard',
        class_scenario='4class',
        fold_id=0
    )
    
    history, history_path = train_baseline_unet(config)
    
    print("\n" + "="*70)
    print("BASELINE U-NET TRAINING COMPLETE")
    print("="*70)
