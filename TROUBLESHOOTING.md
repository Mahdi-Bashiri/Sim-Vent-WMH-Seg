# Troubleshooting Guide

## Common Installation Issues

### TensorFlow Installation Problems

**Problem**: TensorFlow installation fails or GPU not detected
```bash
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Solutions**:
```bash
# 1. Update pip and try again
pip install --upgrade pip
pip install tensorflow==2.10.0

# 2. If GPU issues, check CUDA compatibility
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 3. Install CPU-only version if GPU unavailable
pip install tensorflow-cpu==2.10.0

# 4. For older systems, use compatible version
pip install tensorflow==2.8.0
```

### Dependencies Conflicts

**Problem**: Package version conflicts during installation
```bash
ERROR: pip's dependency resolver does not currently take into account all packages
```

**Solutions**:
```bash
# 1. Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
# fresh_env\Scripts\activate  # Windows

# 2. Install packages individually
pip install numpy==1.21.0
pip install tensorflow==2.10.0
pip install nibabel==3.2.0

# 3. Use conda for better dependency resolution
conda create -n sim-vent-wmh python=3.9
conda activate sim-vent-wmh
conda install tensorflow-gpu
pip install nibabel opencv-python scikit-image
```

### Memory Issues During Installation

**Problem**: Installation fails due to memory constraints
```bash
ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
```

**Solutions**:
```bash
# 1. Clear pip cache
pip cache purge

# 2. Install without cache
pip install --no-cache-dir -r requirements.txt

# 3. Install packages one by one
pip install --no-cache-dir tensorflow
pip install --no-cache-dir nibabel
# ... continue for each package
```

## Data Processing Issues

### FLAIR Image Loading Problems

**Problem**: Cannot load NIfTI files
```python
FileNotFoundError: [Errno 2] No such file or directory: 'image.nii.gz'
```

**Solutions**:
```python
# 1. Check file existence and permissions
import os
print(os.path.exists("path/to/image.nii.gz"))
print(os.access("path/to/image.nii.gz", os.R_OK))

# 2. Use absolute paths
import os
abs_path = os.path.abspath("path/to/image.nii.gz")

# 3. Check file format
import nibabel as nib
try:
    img = nib.load("path/to/image.nii.gz")
    print("Image loaded successfully:", img.shape)
except Exception as e:
    print("Loading error:", e)
```

### Preprocessing Failures

**Problem**: Brain extraction or preprocessing fails
```python
ValueError: Input image has incorrect dimensions
```

**Solutions**:
```python
# 1. Check image dimensions and orientation
import nibabel as nib
img = nib.load("image.nii.gz")
print("Shape:", img.shape)
print("Affine:", img.affine)
print("Header:", img.header)

# 2. Verify image intensity range
data = img.get_fdata()
print("Min:", data.min(), "Max:", data.max())
print("Mean:", data.mean(), "Std:", data.std())

# 3. Check for NaN or infinite values
import numpy as np
print("NaN values:", np.isnan(data).sum())
print("Inf values:", np.isinf(data).sum())
```

### Memory Issues During Processing

**Problem**: Out of memory errors during processing
```python
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
```python
# 1. Reduce batch size (if training)
batch_size = 1  # Use minimum batch size

# 2. Enable GPU memory growth (TensorFlow)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# 3. Process images slice by slice
def process_slices_individually(image_3d):
    results = []
    for slice_idx in range(image_3d.shape[2]):
        slice_2d = image_3d[:, :, slice_idx]
        result = process_single_slice(slice_2d)
        results.append(result)
    return np.stack(results, axis=2)

# 4. Clear GPU memory between processes
tf.keras.backend.clear_session()
```

## Model Inference Issues

### Model Loading Problems

**Problem**: Cannot load pre-trained model
```python
ValueError: SavedModel file does not exist at: model_path
```

**Solutions**:
```python
# 1. Check model path and files
import os
model_path = "src/models/pix2pix_generator_4L"
print("Model directory exists:", os.path.exists(model_path))
print("Contents:", os.listdir(model_path) if os.path.exists(model_path) else "Directory not found")

# 2. Load model with error handling
import tensorflow as tf
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading error:", e)
    # Try loading with custom objects
    model = tf.keras.models.load_model(model_path, compile=False)

# 3. Check TensorFlow version compatibility
print("TensorFlow version:", tf.__version__)
```

### Inference Prediction Issues

**Problem**: Model produces incorrect or empty predictions
```python
# All predictions are zeros or have wrong shape
```

**Solutions**:
```python
# 1. Check input preprocessing
def debug_preprocessing(input_image):
    print("Input shape:", input_image.shape)
    print("Input dtype:", input_image.dtype)
    print("Input range:", input_image.min(), "to", input_image.max())
    print("Input mean:", input_image.mean())
    
    # Visualize input
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image[:, :, 0] if len(input_image.shape) == 3 else input_image, cmap='gray')
    plt.title('Input Image')
    plt.colorbar()
    plt.show()

# 2. Check model input/output shapes
def debug_model_shapes(model, sample_input):
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    print("Sample input shape:", sample_input.shape)
    
    # Ensure input shape matches
    if len(sample_input.shape) == 2:
        sample_input = np.expand_dims(sample_input, axis=[0, -1])
    elif len(sample_input.shape) == 3:
        sample_input = np.expand_dims(sample_input, axis=0)

# 3. Test with simple input
def test_model_basic(model):
    # Create simple test input
    test_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
    try:
        prediction = model.predict(test_input)
        print("Test prediction successful:", prediction.shape)
        return True
    except Exception as e:
        print("Test prediction failed:", e)
        return False
```

## Performance Issues

### Slow Processing Speed

**Problem**: Processing takes much longer than expected (>10 seconds per case)

**Solutions**:
```python
# 1. Check GPU utilization
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Enable GPU if available
if tf.config.list_physical_devices('GPU'):
    print("Using GPU acceleration")
else:
    print("Using CPU - consider enabling GPU")

# 2. Optimize batch processing
def optimize_batch_processing(images, model, batch_size=4):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model.predict(batch)
        results.extend(batch_results)
    return results

# 3. Profile processing time
import time
def profile_processing_steps(image_path):
    times = {}
    
    start = time.time()
    # Preprocessing
    preprocessed = preprocess_image(image_path)
    times['preprocessing'] = time.time() - start
    
    start = time.time()
    # Inference
    result = model.predict(preprocessed)
    times['inference'] = time.time() - start
    
    start = time.time()
    # Postprocessing
    final_result = postprocess_result(result)
    times['postprocessing'] = time.time() - start
    
    print("Processing times:", times)
    return final_result
```

### High Memory Usage

**Problem**: System runs out of RAM during batch processing

**Solutions**:
```python
# 1. Process images one at a time
def process_individually(image_list):
    for image_path in image_list:
        # Process single image
        result = process_single_image(image_path)
        # Save immediately and clear memory
        save_result(result, get_output_path(image_path))
        del result  # Explicit memory cleanup

# 2. Use generators for large datasets
def image_generator(image_paths, batch_size=1):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [load_and_preprocess(path) for path in batch_paths]
        yield np.array(batch_images), batch_paths

# 3. Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    return memory_mb
```

## Visualization Issues

### Cannot Display Results

**Problem**: Matplotlib or visualization issues
```python
ImportError: No module named 'matplotlib'
```

**Solutions**:
```bash
# 1. Install missing packages
pip install matplotlib seaborn

# 2. For headless servers
pip install matplotlib
# Add to your Python script:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

### Poor Image Quality in Results

**Problem**: Segmentation results look noisy or incorrect

**Solutions**:
```python
# 1. Check image intensity normalization
def check_normalization(image):
    print("Original range:", image.min(), "to", image.max())
    
    # Apply proper normalization
    normalized = (image - image.min()) / (image.max() - image.min())
    print("Normalized range:", normalized.min(), "to", normalized.max())
    
    return normalized

# 2. Apply post-processing filters
import scipy.ndimage as ndimage

def postprocess_segmentation(seg_mask):
    # Remove small connected components
    from skimage.morphology import remove_small_objects
    from skimage.measure import label
    
    labeled = label(seg_mask)
    cleaned = remove_small_objects(labeled, min_size=50)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(cleaned > 0)
    
    return filled.astype(seg_mask.dtype)

# 3. Validate segmentation quality
def validate_segmentation(seg_mask, original_image):
    unique_values = np.unique(seg_mask)
    print("Segmentation classes found:", unique_values)
    
    # Check if segmentation is reasonable
    background_ratio = np.sum(seg_mask == 0) / seg_mask.size
    print("Background ratio:", background_ratio)
    
    if background_ratio > 0.9:
        print("Warning: Too much background, check preprocessing")
    elif background_ratio < 0.3:
        print("Warning: Too little background, check thresholds")
```

## File System Issues

### Permission Errors

**Problem**: Cannot write to output directories
```python
PermissionError: [Errno 13] Permission denied: 'output/file.nii.gz'
```

**Solutions**:
```bash
# 1. Check and fix directory permissions
chmod -R 755 output_directory/

# 2. Create directories with proper permissions
mkdir -p output_directory
chmod 755 output_directory

# 3. Run with appropriate user permissions
sudo chown -R $USER:$USER output_directory/
```

### Path Issues

**Problem**: Path-related errors on different operating systems

**Solutions**:
```python
# 1. Use os.path.join for cross-platform compatibility
import os
output_path = os.path.join("results", "patient_001", "segmentation.nii.gz")

# 2. Use pathlib for modern path handling
from pathlib import Path
output_path = Path("results") / "patient_001" / "segmentation.nii.gz"
output_path.parent.mkdir(parents=True, exist_ok=True)

# 3. Handle spaces and special characters in paths
import shlex
safe_path = shlex.quote(str(output_path))
```

## Baseline Method Issues

### SynthSeg Problems

**Problem**: SynthSeg fails to run or produces errors
```bash
Command 'mri_synthseg' not found
```

**Solutions**:
```bash
# 1. Install FreeSurfer and SynthSeg
# Download FreeSurfer from: https://surfer.nmr.mgh.harvard.edu/
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# 2. Alternative: Use Docker version
docker pull freesurfer/synthseg
docker run --rm -v $(pwd):/data freesurfer/synthseg \
    --i /data/input.nii.gz --o /data/output.nii.gz

# 3. Check Python version compatibility
python --version  # SynthSeg requires Python 3.6+
```

### FSL BIANCA Issues

**Problem**: BIANCA installation or execution problems
```bash
fsl: command not found
```

**Solutions**:
```bash
# 1. Install FSL
# Download from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
source /path/to/fsl/etc/fslconf/fsl.sh

# 2. Set environment variables
export FSLDIR=/path/to/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

# 3. Test FSL installation
flirt -version
```

### LST Methods Problems

**Problem**: MATLAB or SPM-related errors for LST methods
```matlab
Error: SPM not found
```

**Solutions**:
```matlab
% 1. Install SPM12
% Download from: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

% 2. Add SPM to MATLAB path
addpath('/path/to/spm12');

% 3. Install LST toolbox
% Download from: https://www.statistical-modelling.de/lst.html

% 4. Alternative: Use SPM standalone
% Download SPM standalone version that doesn't require MATLAB license
```

## Advanced Troubleshooting

### Debug Mode Processing

**Problem**: Need detailed debugging information

**Solutions**:
```python
import logging

# 1. Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def debug_process_image(image_path):
    logger.info(f"Processing image: {image_path}")
    
    try:
        # Load image
        logger.debug("Loading image...")
        image = load_image(image_path)
        logger.debug(f"Image loaded successfully: {image.shape}")
        
        # Preprocess
        logger.debug("Preprocessing...")
        preprocessed = preprocess_image(image)
        logger.debug(f"Preprocessing complete: {preprocessed.shape}")
        
        # Inference
        logger.debug("Running inference...")
        result = model.predict(preprocessed)
        logger.debug(f"Inference complete: {result.shape}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        logger.exception("Full traceback:")
        raise

# 2. Save intermediate results for debugging
def debug_save_intermediates(image_path, output_dir):
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save original
    original = load_image(image_path)
    save_image(original, os.path.join(debug_dir, "01_original.nii.gz"))
    
    # Save after each preprocessing step
    denoised = denoise_image(original)
    save_image(denoised, os.path.join(debug_dir, "02_denoised.nii.gz"))
    
    brain_extracted = extract_brain(denoised)
    save_image(brain_extracted, os.path.join(debug_dir, "03_brain_extracted.nii.gz"))
    
    normalized = normalize_intensity(brain_extracted)
    save_image(normalized, os.path.join(debug_dir, "04_normalized.nii.gz"))
```

### Performance Profiling

**Problem**: Need to identify bottlenecks

**Solutions**:
```python
import cProfile
import pstats
import time
from functools import wraps

# 1. Profile entire processing pipeline
def profile_processing():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your processing
    process_images(image_list)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

# 2. Time individual functions
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def preprocess_image(image):
    # Your preprocessing code
    return processed_image

# 3. Monitor system resources
import psutil
import matplotlib.pyplot as plt
from threading import Thread
import time

class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.monitoring = False
    
    def start_monitoring(self):
        self.monitoring = True
        Thread(target=self._monitor).start()
    
    def stop_monitoring(self):
        self.monitoring = False
    
    def _monitor(self):
        while self.monitoring:
            self.cpu_usage.append(psutil.cpu_percent())
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.timestamps.append(time.time())
            time.sleep(1)
    
    def plot_usage(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(self.timestamps, self.cpu_usage)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Resource Usage')
        
        ax2.plot(self.timestamps, self.memory_usage)
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()

# Usage
monitor = ResourceMonitor()
monitor.start_monitoring()
# Run your processing
process_images(image_list)
monitor.stop_monitoring()
monitor.plot_usage()
```

### Network and Environment Issues

**Problem**: Issues with downloading models or dependencies

**Solutions**:
```bash
# 1. Check internet connectivity
ping google.com

# 2. Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 3. Use alternative package sources
pip install -i https://pypi.org/simple/ tensorflow

# 4. Download manually if needed
wget https://github.com/user/repo/releases/download/v1.0/model.zip
unzip model.zip -d src/models/
```

## Platform-Specific Issues

### Windows-Specific Problems

**Problem**: Path separator and encoding issues

**Solutions**:
```python
# 1. Use raw strings for Windows paths
import os
path = r"C:\Users\Username\Documents\data\image.nii.gz"

# 2. Handle Unicode issues
import locale
print("System encoding:", locale.getpreferredencoding())

# 3. Use pathlib for better Windows compatibility
from pathlib import Path
path = Path("C:/Users/Username/Documents/data/image.nii.gz")
```

### macOS-Specific Problems

**Problem**: Permission issues and library conflicts

**Solutions**:
```bash
# 1. Install Xcode command line tools
xcode-select --install

# 2. Use Homebrew for system dependencies
brew install python@3.9
brew install cmake

# 3. Handle SIP (System Integrity Protection) issues
# Use virtual environments to avoid system conflicts
```

### Linux-Specific Problems

**Problem**: Missing system libraries

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install python3-devel python3-pip
sudo yum install mesa-libGL glib2

# Check for missing libraries
ldd /path/to/problematic/library.so
```

## Getting Help

### Before Seeking Help

1. **Check this troubleshooting guide** thoroughly
2. **Search existing GitHub issues** for similar problems
3. **Test with sample data** to isolate the issue
4. **Check system requirements** and compatibility
5. **Try with a fresh virtual environment**

### When Creating GitHub Issues

Include the following information:

```markdown
## System Information
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- TensorFlow version: [e.g., 2.10.0]
- GPU: [e.g., NVIDIA RTX 3060, None]
- CUDA version: [e.g., 11.7, N/A]

## Problem Description
[Clear description of the issue]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [Third step]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Error Messages
```
[Full error message and traceback]
```

## Additional Context
[Any other relevant information]
```

### Community Resources

- **GitHub Issues**: Primary support channel
- **GitHub Discussions**: General questions and community help
- **Documentation**: Check all documentation files
- **Example Scripts**: Review provided examples

### Emergency Workarounds

If you need immediate results while troubleshooting:

1. **Use pre-computed results** from the `results/` directory
2. **Process smaller image batches** to avoid memory issues
3. **Use CPU-only mode** if GPU problems persist
4. **Skip problematic preprocessing steps** temporarily
5. **Use alternative baseline methods** for comparison

Remember: Most issues are related to environment setup, data format, or path problems. Systematic debugging usually resolves them quickly.