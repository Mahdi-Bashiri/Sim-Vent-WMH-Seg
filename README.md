# Simultaneous Segmentation of Brain Ventricles and White Matter Hyperintensities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)

## 🧠 Overview

This repository implements a novel **2D pix2pix-based deep learning framework** for simultaneous segmentation of brain ventricles and white matter hyperintensities (WMHs) in Multiple Sclerosis (MS) patients. Our approach uniquely distinguishes between **normal** and **pathological** hyperintensities, enabling more accurate clinical diagnosis.

### 🎯 Key Features

- **🔄 Simultaneous Multi-Structure Segmentation**: First approach to jointly segment ventricles and WMHs in a unified framework
- **🎨 Normal vs. Abnormal WMH Classification**: Distinguish CSF-contaminated hyperintensities from true MS lesions
- **⚡ Exceptional Speed**: 4-second processing time (18-36x faster than existing methods)
- **🏥 Clinical Optimization**: Designed for anisotropic clinical MRI data
- **📊 Comprehensive Evaluation**: Compared against 6 state-of-the-art baseline methods

### 📈 Performance Highlights

| Metric | Ventricles | WMH Segmentation | Normal/Abnormal WMH |
|--------|------------|------------------|---------------------|
| **Dice Coefficient** | 0.801 ± 0.025 | 0.624 ± 0.061 | 0.647 |
| **HD95 (mm)** | 18.46 ± 7.1 | 23.0 ± 10.06 | - |
| **AUC-PR** | 0.857 | 0.68 | - |
| **Clinical Accuracy** | 92% | - | 81% valuable |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Clinical FLAIR MRI images

### Installation

```bash
# Clone the repository
git clone https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg.git
cd Sim-Vent-WMH-Seg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.inference import predict
from src.preprocessing import preprocess_flair

# Preprocess FLAIR image
preprocessed_image = preprocess_flair("path/to/flair.nii.gz")

# Run segmentation
result = predict.segment_simultaneous(preprocessed_image)

# Results contain:
# - Ventricle segmentation
# - Normal WMH segmentation  
# - Abnormal WMH segmentation
# - Combined 4-class output
```

---

## 🏗️ Repository Structure

```
├── 📁 src/                          # Core implementation
│   ├── 📁 models/                   # pix2pix architecture
│   ├── 📁 preprocessing/            # Data preprocessing pipeline
│   ├── 📁 training/                 # Training scripts
│   ├── 📁 inference/                # Prediction and evaluation
│   └── 📁 utils/                    # Utility functions
├── 📁 baselines/                    # Comparison methods
│   ├── 📁 SynthSeg/                 # SynthSeg implementation
│   ├── 📁 BIANCA/                   # FSL BIANCA method
│   ├── 📁 LST_methods/              # LST-LPA and LST-LGA
│   └── 📁 Atlas_Matching/           # Template-based approach
├── 📁 examples/                     # Tutorials and demos
│   ├── 📄 quickstart_tutorial.ipynb
│   └── 📄 baseline_comparison.ipynb
├── 📁 results/                      # Performance data and figures
├── 📁 docs/                         # Documentation
└── 📁 tests/                        # Unit tests
```

---

## 🧪 Methodology

### Architecture

Our approach uses a **conditional Generative Adversarial Network (cGAN)** based on pix2pix:

- **Generator**: Modified U-Net with encoder-decoder structure and skip connections
- **Discriminator**: PatchGAN for realistic segmentation evaluation
- **Input**: FLAIR MRI sequences
- **Output**: 4-class segmentation (background, ventricles, normal WMH, abnormal WMH)

### Preprocessing Pipeline

1. **Noise Reduction**: Median filter (3×3) + selective Gaussian filter (σ=1.0)
2. **Brain Extraction**: Morphology-based approach with elliptical masking
3. **Intensity Normalization**: Slice-based adaptive normalization
4. **Paired-Image Generation**: 256×512 composite images for pix2pix input

### Training Details

- **Dataset**: 300 MS patients (1.5-Tesla TOSHIBA Vantage)
- **Expert Annotations**: 20+ years neuroradiologist experience
- **Training Time**: 214 minutes (50 epochs)
- **Optimization**: Adam optimizer (lr=0.0002)
- **Loss Function**: Combined adversarial + L1 loss (λ_adv=1, λ_L1=100)

---

## 📊 Baseline Comparisons

We compared our method against 6 state-of-the-art approaches:

### Ventricle Segmentation
- **SynthSeg**: Deep learning synthetic data approach
- **Atlas Matching**: MNI152 template-based registration

### WMH Segmentation  
- **BIANCA**: FSL-based supervised method
- **LST-LPA**: Unsupervised lesion prediction algorithm
- **LST-LGA**: Lesion growth algorithm
- **WMH-SynthSeg**: SynthSeg extension for WMH

**Result**: Our method achieved superior performance across all metrics while being 18-36x faster.

---

## 🏥 Clinical Applications

### Diagnostic Benefits
- **MS Diagnosis**: Improved accuracy in distinguishing pathological from normal hyperintensities
- **Disease Monitoring**: Quantitative assessment of both atrophy and lesion burden
- **Treatment Planning**: Rapid biomarker quantification for therapy decisions

### Deployment Features
- **Real-time Processing**: 4-second analysis enables same-session decisions
- **Minimal Hardware**: 15% CPU, ~1GB RAM, standard GPU
- **Clinical Integration**: Compatible with standard MRI protocols
- **DICOM Support**: Direct integration with hospital systems

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Usage Tutorial](docs/USAGE.md)**: Step-by-step usage guide
- **[Clinical Guide](docs/CLINICAL_GUIDE.md)**: Clinical interpretation and validation
- **[API Documentation](docs/API.md)**: Complete API reference
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
flake8 src/
```

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{bawil2025,
      title={Simultaneous Segmentation of Ventricles and Normal/Abnormal White Matter Hyperintensities in Clinical MRI using Deep Learning}, 
      author={Mahdi Bashiri Bawil and Mousa Shamsi and Abolhassan Shakeri Bavil},
      year={2025},
      eprint={2506.07123},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.07123}, 
}
```

See [CITATION.bib](CITATION.bib) for detailed citation information.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golgasht Medical Imaging Center**, Tabriz, Iran for providing the clinical dataset
- **Expert neuroradiologists** for manual annotations and clinical validation
- **Open-source community** for foundational tools and libraries

---

## 📞 Contact

- **Repository**: [https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg)
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/discussions)

---

## 🌟 Star History

If you find this work useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/Sim-Vent-WMH-Seg&type=Date)](https://star-history.com/#Mahdi-Bashiri/Sim-Vent-WMH-Seg&Date)
