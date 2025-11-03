[README_v2.md](https://github.com/user-attachments/files/22440134/README_v2.md)
# Simultaneous Segmentation of Brain Ventricles and White Matter Hyperintensities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![arXiv](https://img.shields.io/badge/arXiv-2506.07123-b31b1b.svg)](https://arxiv.org/abs/2506.07123)

## 🧠 Overview

This repository implements a novel **2D pix2pix-based deep learning framework** for simultaneous segmentation of brain ventricles and white matter hyperintensities (WMHs) in Multiple Sclerosis (MS) patients. Our approach uniquely distinguishes between **normal** and **pathological** hyperintensities, enabling more accurate clinical diagnosis.

### 🎯 Key Features

- **🔄 Simultaneous Multi-Structure Segmentation**: First approach to jointly segment ventricles and WMHs in a unified framework
- **🎨 Normal vs. Abnormal WMH Classification**: Distinguish CSF-contaminated hyperintensities from true MS lesions
- **⚡ Exceptional Speed**: 4-second processing time (18-36x faster than existing methods)
- **🏥 Clinical Optimization**: Designed for anisotropic clinical MRI data
- **🌐 Cross-Dataset Validation**: Validated on both local clinical data and public MSSEG2016 dataset
- **📊 Comprehensive Evaluation**: Compared against 6 state-of-the-art baseline methods

### 📈 Performance Highlights

#### Local Dataset Performance
| Metric | Ventricles | WMH Segmentation | Normal/Abnormal WMH |
|--------|------------|------------------|---------------------|
| **Dice Coefficient** | 0.801 ± 0.025 | 0.624 ± 0.061 | 0.647 |
| **HD95 (mm)** | 18.46 ± 7.1 | 23.0 ± 10.06 | - |
| **AUC-PR** | 0.857 | 0.68 | - |
| **Clinical Accuracy** | 92% | - | 81% valuable |

#### Cross-Dataset Validation (MSSEG2016)
| Metric | Ventricles | WMH Segmentation |
|--------|------------|------------------|
| **Dice Coefficient** | 0.798 ± 0.101 | 0.484 ± 0.153 |
| **HD95 (mm)** | 24.39 ± 20.03 | 24.59 ± 4.62 |
| **AUC-PR** | 0.877 | 0.596 |

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
from src.models import inferring
from src.preprocessing import preprocess_flair

# Preprocess FLAIR image
preprocessed_image, brain_mask_info = preprocess_flair.main("path/to/flair.nii.gz", "path/to/save")

# Run segmentation
result = inferring.main(preprocessed_image, brain_mask_info, "path/to/pre-trained-model", "path/to/save")

# Results contain:
# - Ventricle segmentation
# - Normal WMH segmentation  
# - Abnormal WMH segmentation
# - Combined 4-class output
```

### Using Pre-trained Models

We provide two pre-trained models:

1. **Local Model (Epoch 19)**: Trained on our clinical dataset
2. **Fine-tuned Model (Epoch 28)**: Fine-tuned on MSSEG2016 for cross-dataset generalization

```python
# For clinical data similar to our training set
model_path = "models/pix2pix_generator_4L_epoch19/"

# For cross-dataset applications
model_path = "models/pix2pix_generator_4L_epoch28_finetuned/"
```

---

## 🗂️ Repository Structure

```
├── 📁 src/                          # Core implementation
│   ├── 📁 models/                   # pix2pix architecture, training, inference, and evaluation
│   ├── 📁 preprocessing/            # Data preprocessing pipeline
│   ├── 📁 preparation/              # Data preparation pipeline
│   └── 📁 comparison/               # Analytical comparison analysis pipeline
├── 📁 baselines/                    # Comparison methods
│   ├── 📁 SynthSeg/                 # SynthSeg implementation
│   ├── 📁 BIANCA/                   # FSL BIANCA method
│   ├── 📁 LST_methods/              # LST-LPA and LST-LGA
│   ├── 📁 Atlas_Matching/           # Template-based approach
│   └── 📁 raw_data/                 # Sample raw data from cohorts
├── 📁 results/                      # Performance data and figures
│   ├── 📁 local_dataset/            # Local dataset results
│   ├── 📁 msseg2016_dataset/        # Public dataset results
│   ├── 📁 cross_validation/         # Cross-dataset analysis
│   └── 📁 figures/                  # Updated manuscript figures
├── 📁 models/                       # Pre-trained models
│   ├── 📁 epoch19_local/            # Local dataset model
│   └── 📁 epoch28_finetuned/        # Fine-tuned model
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

### Dataset and Validation Strategy

#### Primary Dataset
- **Size**: 300 MS patients from Golgasht Medical Imaging Center, Tabriz, Iran
- **Demographics**: 79 males (18-57 years), 221 females (18-68 years)
- **Scanner**: 1.5-Tesla TOSHIBA Vantage
- **Expert Annotations**: 20+ years neuroradiologist experience

#### Cross-Dataset Validation
- **Public Dataset**: MSSEG2016 challenge dataset (15 patients, 3 centers)
- **Fine-tuning Strategy**: 3 patients for adaptation, 12 for testing
- **Validation Protocol**: Patient-level stratified sampling to prevent data leakage

### Training Details

#### Initial Training
- **Training Time**: 214 minutes (50 epochs)
- **Optimization**: Adam optimizer (lr=0.0002)
- **Loss Function**: Combined adversarial + L1 loss (λ_adv=1, λ_L1=100)
- **Optimal Epoch**: 19 (based on validation performance)

#### Fine-tuning Protocol
- **Base Model**: Epoch 19 from local training
- **Fine-tuning Duration**: Epochs 19-29 on MSSEG2016
- **Optimal Fine-tuned Epoch**: 28
- **Adaptation Strategy**: Limited data fine-tuning (3 patients)

### Preprocessing Pipeline

1. **Noise Reduction**: Median filter (3×3) + selective Gaussian filter (σ=1.0)
2. **Brain Extraction**: Utilizing BET FSL
3. **Intensity Normalization**: Slice-based adaptive normalization
4. **Paired-Image Generation**: 256×512 composite images for pix2pix input

---

## 📊 Comprehensive Evaluation

### Baseline Comparisons

We compared our method against 6 state-of-the-art approaches across both datasets:

#### Ventricle Segmentation
- **SynthSeg**: Deep learning synthetic data approach
- **Atlas Matching**: MNI152 template-based registration

#### WMH Segmentation  
- **BIANCA**: FSL-based supervised method
- **LST-LPA**: Unsupervised lesion prediction algorithm
- **LST-LGA**: Lesion growth algorithm
- **WMH-SynthSeg**: SynthSeg extension for WMH

### Key Findings

#### Cross-Dataset Performance Insights
1. **Ventricle Segmentation**: Consistent performance across datasets
2. **WMH Segmentation**: Dataset-dependent variability highlights generalization challenges
3. **Method-Specific Adaptations**: Different methods show varying responses to dataset characteristics

#### Computational Efficiency
- **Processing Time**: ~4 seconds per case
- **Speed Advantage**: 18-36x faster than existing methods
- **Resource Requirements**: 15% CPU, ~1GB RAM, 80% GPU utilization
- **Hardware Accessibility**: Suitable for diverse clinical environments

---

## 🏥 Clinical Applications

### Diagnostic Benefits
- **MS Diagnosis**: Improved accuracy in distinguishing pathological from normal hyperintensities
- **Disease Monitoring**: Quantitative assessment of both atrophy and lesion burden
- **Treatment Planning**: Rapid biomarker quantification for therapy decisions
- **Cross-Protocol Compatibility**: Validated across different acquisition parameters

### Clinical Validation
- **Expert Assessment**: 92% clinical accuracy for ventricle segmentation
- **WMH Classification**: 81% deemed "clinically valuable" or "highly valuable"
- **Real-world Application**: Optimized for routine clinical MRI protocols

### Deployment Features
- **Real-time Processing**: 4-second analysis enables same-session decisions
- **Minimal Hardware**: Compatible with standard clinical computing infrastructure
- **Clinical Integration**: Direct compatibility with DICOM workflows
- **Scalability**: Suitable for both high-resource and resource-limited settings

---

## 🔬 Research Applications

### Academic Use
- **Reproducible Research**: Complete codebase with documentation
- **Baseline Comparisons**: Standardized evaluation against established methods
- **Cross-Dataset Analysis**: Framework for generalization studies
- **Method Development**: Foundation for advanced segmentation approaches

### Extension Opportunities
- **Multi-Modal Integration**: Framework for incorporating additional MRI sequences
- **Longitudinal Analysis**: Adaptation for disease progression studies
- **Multi-Site Validation**: Protocol for broader clinical validation
- **Clinical Decision Support**: Integration with automated reporting systems

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Usage Tutorial](docs/USAGE.md)**: Step-by-step usage guide
- **[Cross-Dataset Validation](docs/CROSS_DATASET.md)**: Multi-dataset evaluation protocol
- **[Clinical Guide](docs/CLINICAL_GUIDE.md)**: Clinical interpretation and validation
- **[Baseline Methods](docs/BASELINES.md)**: Implementation details for comparison methods
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

## 🔄 Version History

### v2.0.0 (Current)
- ✅ Cross-dataset validation on MSSEG2016
- ✅ Fine-tuning methodology for generalization
- ✅ Enhanced performance analysis across datasets
- ✅ Updated evaluation metrics and comparisons
- ✅ Comprehensive generalizability insights

### v1.0.0
- Initial release with local dataset validation
- Core pix2pix implementation
- Baseline method comparisons
- Clinical validation results

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golgasht Medical Imaging Center**, Tabriz, Iran for providing the clinical dataset
- **MSSEG2016 Challenge** organizers for the public dataset
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
