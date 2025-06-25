# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: Intel Core i7 or equivalent (multi-core recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: CUDA-capable GPU with 4GB+ VRAM (recommended for training)
- **Storage**: 10GB+ free disk space

### Software Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **Python**: 3.9 or higher (3.9-3.11 recommended)
- **CUDA**: 11.2+ (if using GPU acceleration)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg.git
cd Sim-Vent-WMH-Seg
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n sim-vent-wmh python=3.9
conda activate sim-vent-wmh
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import nibabel as nib; print('nibabel installed successfully')"
python -c "import cv2; print('OpenCV installed successfully')"
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Setup

1. **Install NVIDIA drivers** (latest version recommended)
2. **Install CUDA Toolkit 11.2+**:
   ```bash
   # Check CUDA installation
   nvcc --version
   ```

3. **Install cuDNN** (compatible with your CUDA version)

4. **Verify GPU detection**:
   ```bash
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

## Directory Structure Setup

After installation, your directory should look like:

```
Sim-Vent-WMH-Seg/
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── preparation/
│   └── comparison/
├── baselines/
├── results/
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

## Download Pre-trained Models

The pre-trained model (epoch 19) should be available in the repository. If not:

1. Create the models directory:
   ```bash
   mkdir -p src/models/pix2pix_generator_4L
   ```

2. The model files will be located in `Phase3_model_training_and_inferencing_and_evaluation/pix2pix_generator_4L/`

## Data Directory Setup

Create directories for your data:

```bash
mkdir -p data/{raw,preprocessed,results}
```

## Troubleshooting Installation

### Common Issues

**TensorFlow Installation Issues:**
```bash
# If TensorFlow installation fails, try:
pip install --upgrade pip
pip install tensorflow==2.10.0  # or compatible version
```

**OpenCV Installation Issues:**
```bash
# If OpenCV fails to install:
pip install opencv-python-headless
```

**Memory Issues During Installation:**
```bash
# Use pip with limited memory:
pip install --no-cache-dir -r requirements.txt
```

### Platform-Specific Notes

**Windows:**
- Ensure Visual Studio Build Tools are installed
- Use Anaconda/Miniconda for easier dependency management

**macOS:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies if needed

**Linux:**
- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  ```

## Verification Tests

Run the test suite to ensure everything is working:

```bash
# Navigate to the repository root
cd Sim-Vent-WMH-Seg

# Run basic tests (if available)
python -m pytest tests/ -v

# Or run a simple verification
python -c "
import sys
sys.path.append('src')
print('Installation verified successfully!')
"
```

## Docker Installation (Alternative)

If you prefer Docker:

```dockerfile
# Create Dockerfile (example)
FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
```

```bash
# Build and run
docker build -t sim-vent-wmh .
docker run --gpus all -v $(pwd)/data:/app/data sim-vent-wmh
```

## Next Steps

After successful installation:

1. **Read the Usage Tutorial** for step-by-step usage instructions
2. **Prepare your FLAIR MRI data** in NIfTI format
3. **Run the preprocessing pipeline** on sample data
4. **Test inference** with the pre-trained model

## Support

If you encounter installation issues:

1. Check the **Troubleshooting Guide**
2. Search existing [GitHub Issues](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/issues)
3. Create a new issue with your system details and error messages