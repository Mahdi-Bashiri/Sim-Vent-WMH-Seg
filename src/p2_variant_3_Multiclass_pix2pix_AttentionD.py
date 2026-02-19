"""
P2 Article - Multi-Class Baseline Pix2pix with Attention on Discriminator (Variant 3)

Features:
- Multi-channel Generator output (softmax)
- Attention-Weighted PatchGAN Discriminator
- One-hot encoded targets
- Class weight computation per fold
- Optimized for severe class imbalance
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
        all_val_pred: List of softmax output tensors from generator
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
    """Configuration for multi-class pix2pix experiment"""
    
    def __init__(self, 
                 variant: int = 3,
                 preprocessing: str = 'standard',
                 class_scenario: str = '4class',
                 fold_id: int = 0):
        
        # Experiment identification
        self.variant = variant
        self.preprocessing = preprocessing  # 'standard' or 'zoomed'
        self.class_scenario = class_scenario  # '3class' or '4class'
        self.fold_id = fold_id
        
        # Experiment name
        self.exp_name = f"exp_{variant}_multiclass_{preprocessing}_{class_scenario}_fold{fold_id}"
        
        # Number of classes
        self.num_classes = 3 if class_scenario == '3class' else 4
        
        # Training hyperparameters
        self.batch_size = 4
        self.img_width = 256
        self.img_height = 256
        self.epochs = 20
        
        # Loss weights
        self.lambda_seg = 50   # seg loss weight
        self.lambda_gan = 1     # GAN loss weight
        
        # Optimizer parameters
        self.learning_rate = 5e-5
        self.beta_1 = 0.9
        
        # Attention parameters
        self.attention_weight = 2.0  # How much to upweight lesion regions
        
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
            'variant_name': 'Multiclass_AttentionD',
            'preprocessing': self.preprocessing,
            'class_scenario': self.class_scenario,
            'fold_id': self.fold_id,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'lambda_seg': self.lambda_seg,
            'lambda_gan': self.lambda_gan,
            'learning_rate': self.learning_rate,
            'beta_1': self.beta_1,
            'attention_weight': self.attention_weight,
            'innovation': 'Attention Map on Discriminator'
        }
        
        config_file = self.checkpoint_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)


###################### Model Architecture ######################

def downsample(filters, size, apply_norm=True, use_groupnorm=True):
    """
    Downsample block for encoder
    
    Args:
        filters: Number of filters
        size: Kernel size
        apply_norm: Whether to apply normalization
        use_groupnorm: If True, use GroupNorm (better for batch_size=1)
                       If False, use BatchNorm (original pix2pix)
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    
    if apply_norm:
        if use_groupnorm:
            # ✅ GroupNorm: Independent of batch size, no train/inference mismatch
            # Use 32 groups (standard), or filters//8 if filters < 32
            groups = min(32, max(1, filters // 8))
            result.add(tf.keras.layers.GroupNormalization(groups=groups))
        else:
            # Original BatchNorm (can cause NaN with batch_size=1 at inference)
            result.add(tf.keras.layers.BatchNormalization(momentum=0.99))
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result


def upsample(filters, size, apply_dropout=False, use_groupnorm=True):
    """
    Upsample block for decoder
    
    Args:
        filters: Number of filters
        size: Kernel size
        apply_dropout: Whether to apply dropout
        use_groupnorm: If True, use GroupNorm (better for batch_size=1)
                       If False, use BatchNorm (original pix2pix)
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    
    if use_groupnorm:
        # ✅ GroupNorm: Independent of batch size, no train/inference mismatch
        groups = min(32, max(1, filters // 8))
        result.add(tf.keras.layers.GroupNormalization(groups=groups))
    else:
        # Original BatchNorm (can cause NaN with batch_size=1 at inference)
        result.add(tf.keras.layers.BatchNormalization(momentum=0.99))
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())
    
    return result


def build_attention_discriminator(num_classes: int, input_channels: int = 1, 
                                   attention_weight: float = 2.0, use_groupnorm: bool = True):
    """
    Build Attention-Weighted PatchGAN Discriminator
    
    Args:
        num_classes: Number of classes in target mask
        input_channels: Number of input channels
        attention_weight: Multiplier for lesion regions (>1.0 upweights lesions)
        use_groupnorm: If True, use GroupNorm instead of BatchNorm
    
    Returns:
        Discriminator model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Input: FLAIR image
    inp = tf.keras.layers.Input(
        shape=[256, 256, input_channels], 
        name='input_image'
    )
    
    # ✅ Target: Multi-channel one-hot mask
    tar = tf.keras.layers.Input(
        shape=[256, 256, num_classes],
        name='target_mask'
    )
    
    # ✅ Compute spatial attention map from target mask
    # attention_map: (bs, 256, 256, 1)
    # Background (class 0) gets weight 1.0, lesions get attention_weight
    class_indices = tf.argmax(tar, axis=-1, output_type=tf.int32)  # (bs, 256, 256)
    attention_map = tf.where(
        class_indices == 0,
        tf.ones_like(class_indices, dtype=tf.float32),  # Background: weight 1.0
        tf.ones_like(class_indices, dtype=tf.float32) * attention_weight  # Lesions: upweighted
    )
    attention_map = tf.expand_dims(attention_map, axis=-1)  # (bs, 256, 256, 1)
    
    # Concatenate input and target
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, 1+num_classes)
    
    # Standard PatchGAN backbone
    down1 = downsample(64, 4, apply_norm=False, use_groupnorm=use_groupnorm)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, use_groupnorm=use_groupnorm)(down1)     # (bs, 64, 64, 128)
    down3 = downsample(256, 4, use_groupnorm=use_groupnorm)(down2)     # (bs, 32, 32, 256)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)  # (bs, 31, 31, 512)
    
    if use_groupnorm:
        batchnorm1 = tf.keras.layers.GroupNormalization(groups=8)(conv)
    else:
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    
    # Output patch predictions
    patch_output = tf.keras.layers.Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer,
        name='patch_predictions'
    )(zero_pad2)  # (bs, 30, 30, 1)
    
    # ✅ Apply spatial attention to patch predictions
    # Downsample attention map to match patch size (256 -> 30)
    attention_downsampled = tf.keras.layers.AveragePooling2D(
        pool_size=(9, 9), strides=(8, 8), padding='same'
    )(attention_map)  # Approximate (bs, 30, 30, 1)
    
    # Ensure exact shape match
    attention_resized = tf.image.resize(
        attention_downsampled, 
        [tf.shape(patch_output)[1], tf.shape(patch_output)[2]],
        method='bilinear'
    )
    
    # Apply attention weighting
    weighted_output = patch_output * attention_resized
    
    return tf.keras.Model(
        inputs=[inp, tar], 
        outputs=weighted_output, 
        name='AttentionDiscriminator'
    )


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

# Binary cross-entropy for GAN loss
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target_onehot, 
                  class_weights, lambda_gan=1, lambda_seg=100
                  ):
    """
    Generator loss: GAN loss + Weighted CCE
    
    Args:
        disc_generated_output: Discriminator output for generated mask
        gen_output: Generated mask (bs, 256, 256, num_classes) softmax
        target_onehot: Target mask (bs, 256, 256, num_classes) one-hot
        class_weights: (num_classes,) weight per class
        lambda_gan: Weight for GAN loss (default 1.0)
        lambda_seg: Weight for segmentation loss (default 100.0)
    
    Returns:
        total_gen_loss, gan_loss, seg_loss
    """
    # GAN loss: fool the discriminator
    gan_loss = bce_loss(
        tf.ones_like(disc_generated_output), 
        disc_generated_output
    )
    
    # Weighted categorical cross-entropy
    seg_loss = weighted_categorical_crossentropy(target_onehot, gen_output, class_weights)
    
    # Total generator loss
    total_gen_loss = (lambda_gan * gan_loss) + (lambda_seg * seg_loss)
    
    return total_gen_loss, gan_loss, seg_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Discriminator loss: distinguish real from fake
    
    Args:
        disc_real_output: Discriminator output for real mask
        disc_generated_output: Discriminator output for generated mask
    
    Returns:
        total_disc_loss
    """
    real_loss = bce_loss(
        tf.ones_like(disc_real_output), 
        disc_real_output
    )
    
    generated_loss = bce_loss(
        tf.zeros_like(disc_generated_output), 
        disc_generated_output
    )
    
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss


###################### Training Functions ######################

@tf.function
def train_step(input_image, target_onehot, generator, discriminator, 
               generator_optimizer, discriminator_optimizer, 
               class_weights_np, lambda_gan, lambda_seg):
    """
    Single training step
    
    Args:
        input_image: Input FLAIR (bs, 256, 256, 1) in [-1, 1]
        target_onehot: Target mask (bs, 256, 256, num_classes) one-hot
        generator, discriminator, optimizers
        class_weights_np: (num_classes,) weight per class
        lambda_gan, lambda_seg: Loss weights
    
    Returns:
        gen_total_loss, gen_gan_loss, gen_seg_loss, disc_loss
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate output
        gen_output = generator(input_image, training=True)
        
        # Discriminator outputs
        disc_real_output = discriminator(
            [input_image, target_onehot], training=True
        )
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True
        )
        
        # Generator loss (adaptive)
        gen_total_loss, gen_gan_loss, gen_seg_loss = \
            generator_loss(
                disc_generated_output, gen_output, target_onehot, 
                class_weights_np, lambda_gan, lambda_seg
            )

        # Discriminator loss
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    # Calculate gradients
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    
    # Apply gradients
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    
    # return gen_total_loss, gen_gan_loss, gen_seg_loss, disc_loss
    return (gen_total_loss, gen_gan_loss, gen_seg_loss,
            disc_loss, class_weights_np)


def generate_and_save_images(generator, test_input, test_target, 
                            epoch, save_path, num_classes):
    """
    Generate predictions and save visualization
    
    Args:
        generator: Generator model
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
        prediction_softmax = generator(flair_normalized, training=False)
        
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

def train_experiment_with_metrics(config: ExperimentConfig):
    """
    Main training function for baseline multi-class pix2pix with attention on discriminator
    
    Args:
        config: ExperimentConfig object
    """
    print("\n" + "="*70)
    print(f"TRAINING EXPERIMENT: {config.exp_name}")
    print("="*70)
    print(f"Variant: {config.variant} (Baseline + AttentionD)")
    print(f"Preprocessing: {config.preprocessing}")
    print(f"Class scenario: {config.class_scenario} ({config.num_classes} classes)")
    print(f"Fold: {config.fold_id}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Loss weights: λ_SEG={config.lambda_seg}, λ_GAN={config.lambda_gan}")
    print(f"Attention weight: {config.attention_weight}")
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

    # Build models
    print("\n🏗️  Building models...")
    generator = build_unet_3class(input_shape=(256, 256, 1), num_classes=config.num_classes)
    discriminator = build_attention_discriminator(
        config.num_classes, 
        input_channels=1,
        attention_weight=config.attention_weight,
        use_groupnorm=True  # ✅ Consistent with generator
    )
    
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}\n")
    
    # Optimizers
    generator_optimizer = tf.keras.optimizers.legacy.Adam(
        config.learning_rate, beta_1=config.beta_1
    )
    discriminator_optimizer = tf.keras.optimizers.legacy.Adam(
        config.learning_rate, beta_1=config.beta_1
    )
    
    # Initialize optimizer variables
    # CRITICAL: Build optimizer variables by calling them once with dummy data
    # This prevents the "tf.function only supports singleton tf.Variables" error
    print("Initializing optimizer variables...")
    dummy_input = tf.zeros((1, 256, 256, 1))
    dummy_target = tf.zeros((1, 256, 256, config.num_classes))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(dummy_input, training=True)
        disc_output = discriminator([dummy_input, dummy_target], training=True)
        # Dummy losses
        dummy_gen_loss = tf.reduce_mean(gen_output)
        dummy_disc_loss = tf.reduce_mean(disc_output)

    # Apply dummy gradients to build optimizer variables
    # Don't include class_weights since they're not trainable
    gen_grads = gen_tape.gradient(dummy_gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(dummy_disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
    print("✅ Optimizer variables initialized\n")
    
    # Checkpoint
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
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
        'gen_total_loss': [],
        'gen_gan_loss': [],
        'gen_seg_loss': [],
        'disc_loss': [],
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
            epoch_gen_total_loss = []
            epoch_gen_gan_loss = []
            epoch_gen_seg_loss = []
            epoch_disc_loss = []
            
            # Training loop

            # Update learning rate based on epoch
            new_lr = config.learning_rate * ((1-(1-0.1e-1)*(epoch / config.epochs)))  # Exponential decay from 2e-4 to 1e-6
            generator_optimizer.learning_rate.assign(new_lr)
            discriminator_optimizer.learning_rate.assign(new_lr)

            print(f"\nEpoch {epoch+1}/{config.epochs} (lr={new_lr:.6f})")
            train_bar = tqdm(train_dataset, total=train_size, desc="Training")
            
            for paired_input, target_mask in train_bar:
                # ✅ Prepare inputs: normalize FLAIR + one-hot encode target
                flair_normalized, target_onehot = prepare_inputs(
                    paired_input, target_mask, config.num_classes
                )
                
                # Train step
                gen_total, gen_gan, gen_seg, disc, cw = train_step(
                    flair_normalized, target_onehot,
                    generator, discriminator,
                    generator_optimizer, discriminator_optimizer,
                    class_weights, config.lambda_gan, config.lambda_seg, 
                )
                
                epoch_gen_total_loss.append(gen_total.numpy())
                epoch_gen_gan_loss.append(gen_gan.numpy())
                epoch_gen_seg_loss.append(gen_seg.numpy())
                epoch_disc_loss.append(disc.numpy())
                
                # Update progress bar
                train_bar.set_postfix({
                    'G_loss': f"{gen_total.numpy():.4f}",
                    'D_loss': f"{disc.numpy():.4f}",
                    'SEG': f"{gen_seg.numpy():.4f}"
                })
            
            # Calculate epoch averages
            avg_gen_total = np.mean(epoch_gen_total_loss)
            avg_gen_gan = np.mean(epoch_gen_gan_loss)
            avg_gen_seg = np.mean(epoch_gen_seg_loss)
            avg_disc = np.mean(epoch_disc_loss)
            
            history['gen_total_loss'].append(avg_gen_total)
            history['gen_gan_loss'].append(avg_gen_gan)
            history['gen_seg_loss'].append(avg_gen_seg)
            history['disc_loss'].append(avg_disc)
            
            # Validation
            val_losses = []
            all_val_true = []
            all_val_pred = []
            for val_paired, val_target in val_dataset:
                try:
                    val_flair_norm, val_target_onehot = prepare_inputs(
                        val_paired, val_target, config.num_classes
                    )
                    val_pred = generator(val_flair_norm, training=False)  # ✅ Now safe!

                    val_seg_loss = weighted_categorical_crossentropy(val_target_onehot, val_pred, class_weights, exclude_class=exclude_class)

                    # Store true and prediction values for final metrics calculation
                    all_val_true.append(val_target_onehot)
                    all_val_pred.append(val_pred)
                    
                    if not tf.math.is_nan(val_seg_loss):
                        val_losses.append(val_seg_loss.numpy())
                except:
                    continue
            

            if len(val_losses) > 0:
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Compute class-wise metrics
                val_metrics = compute_classwise_metrics(
                    all_val_true, all_val_pred, 
                    config.num_classes#, exclude_class=exclude_class
                )
                history['val_metrics'].append(val_metrics)
                
                # Print validation results
                epoch_time = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{config.epochs} Summary (Time: {epoch_time:.2f}s)")
                print(f"{'='*70}")
                print(f"Training Losses:")
                print(f"  Generator Total: {avg_gen_total:.4f} | GAN: {avg_gen_gan:.4f} | SEG: {avg_gen_seg:.4f} | Discriminator: {avg_disc:.4f}")
                print(f"\nValidation Loss: {avg_val_loss:.4f}")
                print(f"\nClass-wise Dice Scores:")
                for class_name, dice_val in val_metrics['dice'].items():
                    if class_name != 'mean':
                        print(f"  {class_name}: {dice_val:.4f}")
                        if class_name == f"class_{config.num_classes -1}":
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
                
                # Save best model based on validation loss
                overal_val_performance = 0.6 * abwmh_val_dice + 0.3 * vent_val_dice + 0.1 * (1-10*avg_val_loss)
                if overal_val_performance > best_val_dice:
                    best_val_dice = overal_val_performance
                    generator.save_weights(f"{config.checkpoint_dir}/best_dice_generator.h5")
                    discriminator.save_weights(f"{config.checkpoint_dir}/best_dice_discriminator.h5")
                    print(f"✓ Best model saved (performance: {best_val_dice:.4f})")
            else:
                print("Warning: No valid validation batches")
                history['val_loss'].append(float('nan'))
                history['val_metrics'].append({})
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Gen Total Loss: {avg_gen_total:.4f}")
            print(f"  Gen GAN Loss: {avg_gen_gan:.4f}")
            print(f"  Gen Seg Loss: {avg_gen_seg:.4f}")
            print(f"  Disc Loss: {avg_disc:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                manager.save()
                print(f"  💾 Saved checkpoint")
            
            # Generate sample images
            if (epoch + 1) % 5 == 0 or epoch == 0 or True:
                generate_and_save_images(
                    generator, example_paired, example_target,
                    epoch + 1, config.figures_dir, config.num_classes
                )
                print(f"  📊 Saved visualization")

        # Save final model
        final_model_path = config.checkpoint_dir / "final_model.h5"
        generator.save(final_model_path)
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
        # This runs whether training succeeds or fails
        print("\n🧹 Cleaning up resources...")

        # Delete models explicitly to break references
        try:
            del generator
            del discriminator
            del generator_optimizer
            del discriminator_optimizer
            del checkpoint
            del manager
            del train_dataset
            del val_dataset
            # class_weights don't need deletion (they're constants, not variables)
            print("✅ Deleted model objects")
        except Exception as e:
            print(f"⚠️  Error deleting objects: {e}")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        # Check final GPU memory
        get_gpu_memory_info()


###################### Main Execution ######################

if __name__ == "__main__":
    # Example: Train multi-class model for 4-class, standard preprocessing, fold 0
    config = ExperimentConfig(
        variant=3,
        preprocessing='standard',
        class_scenario='4class',
        fold_id=0
    )
    
    history, history_path = train_experiment_with_metrics(config)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)