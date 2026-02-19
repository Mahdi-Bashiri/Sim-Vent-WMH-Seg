[README.md](https://github.com/user-attachments/files/25402907/README.md)
# Adversarial Deep Learning for Simultaneous Segmentation of Ventricular and White Matter Hyperintensities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![arXiv](https://img.shields.io/badge/arXiv-2506.07123-b31b1b.svg)](https://arxiv.org/abs/2506.07123)
[![Models on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Bawil/neuro-ai/tree/main)

## 🧠 Overview

This repository implements a novel **adversarial deep learning framework** based on the pix2pix conditional GAN architecture for simultaneous segmentation of brain ventricles and white matter hyperintensities (WMHs) in Multiple Sclerosis (MS) patients. Our approach uniquely distinguishes between **normal periventricular hyperintensities** (CSF-contaminated) and **pathological MS lesions**, enabling more accurate clinical diagnosis.

### 🎯 Key Features

- **🔄 Simultaneous Multi-Structure Segmentation**: First adversarial approach to jointly segment ventricles and WMH in a unified framework
- **🎨 Normal vs. Pathological WMH Classification**: Distinguish CSF-contaminated hyperintensities from true MS lesions
- **🏗️ Systematic Architecture Ablation**: Six variants (V0-V5) progressively integrating adversarial training, attention mechanisms, and adaptive loss
- **⚡ Exceptional Speed**: ~4-second processing time (up to 36× faster than existing methods)
- **🥼 Clinical Optimization**: Designed for anisotropic clinical MRI data with 2D slice-based approach
- **🌍 Cross-Dataset Validation**: Trained on 300 local patients + MSSEG2016 (15 patients) with 5-fold cross-validation
- **📊 Comprehensive Evaluation**: Compared against 6 state-of-the-art baseline methods

### 📈 Performance Highlights

#### Final Model (V5) - Combined Dataset Performance
| Metric | Overall | Ventricles | Abnormal WMH | Normal WMH |
|--------|---------|------------|--------------|------------|
| **Dice Coefficient** | 0.852 ± 0.004 | 0.907 ± 0.002 | 0.825 ± 0.009 | 0.677 ± 0.007 |
| **HD95 (mm)** | 4.87 ± 0.13 | 3.00 ± 0.51 | 4.51 ± 0.32 | 4.87 ± 0.24 |
| **Precision** | 0.856 ± 0.006 | 0.916 ± 0.005 | 0.849 ± 0.013 | 0.660 ± 0.019 |
| **Recall** | 0.850 ± 0.006 | 0.899 ± 0.007 | 0.804 ± 0.021 | 0.696 ± 0.010 |
| **IoU** | 0.760 ± 0.006 | 0.830 ± 0.004 | 0.703 ± 0.013 | 0.512 ± 0.008 |

#### Normal vs. Abnormal WMH Classification
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 84.28% |
| **Sensitivity (Abnormal WMH)** | 79.85% |
| **Specificity (Normal WMH)** | 93.65% |
| **Precision** | 96.38% |
| **F1-Score** | 0.8734 |
| **Cohen's Kappa** | 0.6707 |

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
preprocessed_image, brain_mask_info = preprocess_flair.main(
    "path/to/flair.nii.gz", 
    "path/to/save"
)

# Run segmentation with V5 model
result = inferring.main(
    preprocessed_image, 
    brain_mask_info, 
    "path/to/v5-model", 
    "path/to/save"
)

# Results contain:
# - Ventricle segmentation
# - Normal WMH segmentation  
# - Abnormal WMH segmentation
# - Combined 4-class output
```

### Using Pre-trained Models

We provide the final V5 model trained on combined dataset:

```python
# V5: Attention Discriminator + Adaptive Hybrid Loss (Recommended)
model_path = "models/pix2pix_v5_combined/"
```

---

## 🗂️ Repository Structure

```
├── 📁 src/                          # Core implementation
│   ├── 📁 models/                   # pix2pix architecture variants (V0-V5)
│   │   ├── training.py              # Training scripts for all variants
│   │   ├── inference.py             # Inference pipeline
│   │   ├── evaluation.py            # Evaluation metrics
│   │   └── architectures/           # Generator and discriminator definitions
│   ├── 📁 preprocessing/            # Data preprocessing pipeline
│   │   ├── noise_reduction.py       # Median + Gaussian filtering
│   │   ├── brain_extraction.py      # FSL BET integration
│   │   └── normalization.py         # Slice-based intensity normalization
│   ├── 📁 preparation/              # Data preparation
│   │   └── paired_image.py          # 256×512 composite generation
│   └── 📁 comparison/               # Baseline comparison analysis
├── 📁 baselines/                    # Comparison methods
│   ├── 📁 SynthSeg/                 # SynthSeg implementation
│   ├── 📁 BIANCA/                   # FSL BIANCA method
│   ├── 📁 LST_methods/              # LST-LPA and LST-LGA
│   ├── 📁 Atlas_Matching/           # Template-based approach
│   └── 📁 WMH-SynthSeg/             # WMH-SynthSeg extension
├── 📁 results/                      # Performance data and figures
│   ├── 📁 ablation_study/           # V0-V5 comparison results
│   ├── 📁 baseline_comparison/      # Results vs. 6 baselines
│   ├── 📁 cross_validation/         # 5-fold CV analysis
│   └── 📁 figures/                  # Manuscript figures
├── 📁 models/                       # Pre-trained models
│   └── 📁 v5_combined/              # Final V5 model weights
├── 📁 docs/                         # Documentation
└── 📁 tests/                        # Unit tests
```

---

## 🧪 Methodology

### Architecture Evolution: V0 to V5

Our systematic ablation study evaluated six architectural variants:

| Variant | Description | Key Components | Mean Dice | HD95 (mm) |
|---------|-------------|----------------|-----------|-----------|
| **V0** | Baseline U-Net | Standard U-Net + WCE | 0.714 ± 0.018 | 6.50 ± 0.46 |
| **V1** | Baseline Pix2Pix | U-Net + PatchGAN + WCE | 0.823 ± 0.011 | 5.31 ± 0.20 |
| **V2** | Pix2Pix + UFL | U-Net + PatchGAN + UFL | 0.817 ± 0.010 | 5.33 ± 0.13 |
| **V3** | Attention Discriminator | U-Net + Attention PatchGAN + WCE | 0.824 ± 0.008 | 5.23 ± 0.31 |
| **V4** | Adaptive Hybrid Loss | U-Net + PatchGAN + Adaptive Loss | 0.844 ± 0.002 | 4.81 ± 0.05 |
| **V5** | **Final Model** | **All components combined** | **0.852 ± 0.004** | **4.87 ± 0.13** |

**Key Findings from Ablation:**
- Adversarial training (V0→V1): **+0.109 Dice** - largest single improvement
- Attention discriminator (V1→V3): **+0.001 Dice, -0.08mm HD95** - focused foreground learning
- Adaptive hybrid loss (V1→V4): **+0.021 Dice, -0.50mm HD95** - optimal balance
- Combined V5: **Best overall performance** with stable convergence

### Pix2Pix Framework (V5 Architecture)

#### Generator: Modified U-Net
- **Encoder**: 4 downsampling blocks (Conv → BatchNorm → LeakyReLU)
- **Bottleneck**: Feature compression and representation
- **Decoder**: 4 upsampling blocks (TransposeConv → BatchNorm → ReLU → Dropout)
- **Skip Connections**: Preserve fine-grained structural detail
- **Output**: 4-class segmentation (background, ventricles, normal WMH, abnormal WMH)

#### Discriminator: Attention-Weighted PatchGAN
- **PatchGAN Architecture**: Evaluates overlapping 70×70 patches for local boundary quality
- **Attention Mechanism**: 2× weight for foreground classes vs. background
- **Purpose**: Directs adversarial signal toward anatomically critical structures

#### Loss Functions

**Total Loss (V5):**
```
L_total = λ_adv × L_adversarial + λ_seg × L_adaptive_hybrid
```

**Adaptive Hybrid Loss:**
```
L_adaptive_hybrid = α(t) × L_WCE + (1 - α(t)) × L_UFL

α(t) = 1 / (1 + exp(k × (t - h)))
```
Where:
- `t`: current epoch
- `k`: transition speed constant
- `h`: transition midpoint
- Smoothly transitions from WCE (early training) to UFL (late training)

### Dataset and Validation Strategy

#### Primary Dataset
- **Size**: 300 MS patients from Golgasht Medical Imaging Center, Tabriz, Iran
- **Demographics**: 79 males (18-57 years), 221 females (18-68 years)
- **Scanner**: 1.5-Tesla TOSHIBA Vantage
- **Voxel Dimensions**: 0.9 × 0.9 × 6 mm (anisotropic)
- **Expert Annotations**: Neuroradiologist with 20+ years experience

#### Public Dataset Integration
- **Dataset**: MSSEG2016 challenge (15 patients, 3 centers)
- **Annotation**: Expert-provided WMH + our manual ventricle/normal WMH annotations
- **Purpose**: Cross-dataset validation and generalization testing

#### Training Configuration
- **Split**: 
  - Training: 210 local + 9 public patients (4,650 images)
  - Validation: 30 local + 3 public patients (750 images)
  - Test: 60 local + 3 public patients (1,350 images)
- **Cross-Validation**: 5-fold with patient-level stratification
- **Batch Size**: 4
- **Epochs**: 60
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, lr=2×10⁻⁴)
- **Training Time**: ~4.35 hours per fold (total ~22 hours for 5-fold CV)

### Preprocessing Pipeline

1. **Noise Reduction**
   - Median filter (3×3 kernel) for salt-and-pepper noise
   - Selective Gaussian filter (σ=1.0) for smoothing

2. **Brain Extraction**
   - FSL BET (Brain Extraction Tool) with default parameters
   - Binary mask generation for brain region

3. **Intensity Normalization**
   - Slice-based adaptive normalization
   - Min: Average background intensity (slice-specific)
   - Max: Peripheral structures intensity (skull/scalp)
   - Z-score normalization on scaled [0,1] values

4. **Paired-Image Generation**
   - Horizontal concatenation: FLAIR (256×256) + Mask (256×256)
   - Output: 256×512 composite for pix2pix input

### Ground Truth Annotation Protocol

#### Four-Phase Workflow

**Phase 1 - Primary Manual Annotation:**
- Ventricles: Low FLAIR intensity + anatomical location
- Abnormal WMH: High FLAIR intensity + MS-characteristic features
  - Periventricular/juxtacortical location
  - Ovoid morphology perpendicular to ventricles
  - Minimum diameter: 3mm
  - T2-weighted confirmation
- Normal WMH: Non-MS hyperintensities
  - Thin periventricular rims
  - Bilateral symmetric age-related changes

**Phase 2 - Statistical Morphological Refinement:**
- Ventricles: Morphological closing (3×3) + connected component analysis
- WMH: Conservative opening (3×3) + area change verification (>10% flagged)
- Intensity validation: Ventricles <25th percentile, WMH >75th percentile

**Phase 3 - Expert Consensus Review:**
- Independent review by senior neuroradiologist
- Discrepancy resolution through discussion
- Systematic documentation of refinement criteria

**Phase 4 - Final Quality Control:**
- Comprehensive review by both experts
- Detailed re-examination of flagged cases
- Manual correction as needed

#### Periventricular Normal WMH Definition
- **Boundary**: 5-pixel radius dilation (~4.5mm) from ventricle masks
- **Intensity**: >75th percentile of brain tissue
- **Exclusion**: Overlap with abnormal WMH removed
- **Verification**: Bilateral symmetry, smooth margins, no MS features

---

## 📊 Comprehensive Evaluation

### Baseline Comparisons

We compared V5 against 6 state-of-the-art methods:

#### Ventricle Segmentation Methods
1. **SynthSeg**: Deep learning on synthetic data [14]
2. **Atlas Matching**: MNI152 template-based registration [38]

#### WMH Segmentation Methods
3. **BIANCA**: FSL-based supervised method [31]
4. **LST-LPA**: Unsupervised lesion prediction [39]
5. **LST-LGA**: Lesion growth algorithm [40]
6. **WMH-SynthSeg**: SynthSeg extension for WMH [6]

### Performance Summary

#### Ventricle Segmentation (Local Dataset)
| Method | Dice | HD95 (mm) | Precision | Recall |
|--------|------|-----------|-----------|--------|
| **V5 (Ours)** | **0.907 ± 0.002** | **3.00 ± 0.51** | **0.916 ± 0.005** | **0.899 ± 0.007** |
| SynthSeg | 0.838 ± 0.026 | 6.1 ± 6.57 | 0.848 ± 0.032 | 0.848 ± 0.051 |
| Atlas Matching | 0.669 ± 0.062 | 14.7 ± 5.23 | 0.672 ± 0.088 | 0.713 ± 0.083 |

#### WMH Segmentation (Local Dataset)
| Method | Dice | HD95 (mm) | Precision | Recall |
|--------|------|-----------|-----------|--------|
| **V5 (Ours)** | **0.825 ± 0.009** | **4.51 ± 0.32** | **0.849 ± 0.013** | **0.804 ± 0.021** |
| WMH-SynthSeg | 0.576 ± 0.067 | 6.73 ± 1.88 | 0.571 ± 0.082 | 0.602 ± 0.088 |
| BIANCA | 0.485 ± 0.081 | 10.8 ± 3.46 | 0.519 ± 0.107 | 0.488 ± 0.097 |
| LST-LPA | 0.462 ± 0.079 | 11.6 ± 3.89 | 0.457 ± 0.093 | 0.498 ± 0.104 |
| LST-LGA | 0.273 ± 0.058 | 18.9 ± 5.67 | 0.312 ± 0.083 | 0.265 ± 0.062 |

### Computational Efficiency

| Method | Processing Time | CPU Usage | RAM Usage | GPU Usage |
|--------|----------------|-----------|-----------|-----------|
| **V5 (Ours)** | **~4 sec** | **15%** | **~1 GB** | **80%** |
| SynthSeg | 124 sec | 100% | ~5 GB | 100% |
| WMH-SynthSeg | 78 sec | 100% | ~5 GB | 100% |
| LST-LGA | 147 sec | 85% | ~3 GB | N/A |
| Atlas Matching | 115 sec | 70% | ~4 GB | N/A |
| BIANCA | 11 sec | 40% | ~2 GB | N/A |
| LST-LPA | 9 sec | 35% | ~2 GB | N/A |

**Speed Advantage**: 18-36× faster than most baseline methods!

---

## 🥼 Clinical Applications

### Diagnostic Benefits

1. **MS Diagnosis**: Improved accuracy in distinguishing pathological from normal hyperintensities
2. **Disease Monitoring**: Simultaneous quantitative assessment of atrophy (ventricles) and lesion burden (WMH)
3. **Treatment Planning**: Rapid biomarker quantification for therapy decisions
4. **Clinical Workflow Integration**: 4-second processing enables same-session analysis

### Clinical Validation Results

- **Normal vs. Abnormal WMH Classification**: 84.28% accuracy
- **Conservative Bias**: 6.35% false positives (reliable when predicting pathology)
- **Sensitivity for Abnormal WMH**: 79.85% (captures majority of true lesions)
- **Specificity for Normal WMH**: 93.65% (avoids over-diagnosis)

### Deployment Features

- **Real-time Processing**: ~4 seconds per case
- **Minimal Hardware**: 1GB RAM, consumer-grade GPU
- **DICOM Compatible**: Direct integration with clinical workflows
- **Scalability**: Suitable for both resource-rich and resource-limited settings

---

## 📬 Research Applications

### Academic Use

- **Reproducible Research**: Complete codebase with documentation
- **Systematic Ablation**: V0-V5 variants for architecture studies
- **Cross-Dataset Protocol**: Framework for generalization studies
- **Baseline Comparisons**: Standardized evaluation suite

### Extension Opportunities

- **Multi-Modal Integration**: Incorporate T1, T2 sequences
- **Longitudinal Analysis**: Disease progression tracking
- **3D Extension**: Adapt 2D approach to 3D volumes
- **Transfer Learning**: Apply to other neurological conditions

---

## 📚 Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Usage Tutorial](docs/USAGE.md)**: Step-by-step usage guide
- **[Architecture Ablation](docs/ABLATION_STUDY.md)**: V0-V5 variant comparison
- **[Training Guide](docs/TRAINING.md)**: How to train your own models
- **[Clinical Guide](docs/CLINICAL_GUIDE.md)**: Clinical interpretation
- **[Baseline Methods](docs/BASELINES.md)**: Implementation details
- **[API Reference](docs/API.md)**: Complete API documentation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

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

### Contribution Areas

- 🐛 Bug fixes and improvements
- 📝 Documentation enhancements
- 🧪 Additional test coverage
- 🚀 Performance optimizations
- 🎨 New visualization tools
- 🔬 Extended validation studies

---

## 📜 Citation

If you use this work in your research, please cite:

```bibtex
@article{bashiri2025adversarial,
      title={Adversarial Deep Learning for Simultaneous Segmentation of Ventricular and White Matter Hyperintensities in Clinical MRI}, 
      author={Mahdi Bashiri Bawil and Mousa Shamsi and Abolhassan Shakeri Bavil},
      year={2025},
      eprint={2506.07123},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.07123}, 
      note={Trial Registration: IR.TBZMED.REC.1402.902}
}
```

See [CITATION.bib](CITATION.bib) for detailed citation information.

---

## 📄 Version History

### v3.0.0 (Current)
- ✅ Systematic architectural ablation study (V0-V5)
- ✅ Attention-weighted PatchGAN discriminator
- ✅ Adaptive hybrid loss function (WCE→UFL transition)
- ✅ 5-fold cross-validation with patient-level stratification
- ✅ Combined local (300) + public (15) dataset training
- ✅ Comprehensive baseline comparison (6 methods)
- ✅ Enhanced normal vs. abnormal WMH classification

### v2.0.0
- Cross-dataset validation on MSSEG2016
- Fine-tuning methodology for generalization
- Updated evaluation metrics and comparisons

### v1.0.0
- Initial release with local dataset validation
- Core pix2pix implementation
- Basic baseline comparisons

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Golgasht Medical Imaging Center**, Tabriz, Iran for providing the clinical dataset
- **MSSEG2016 Challenge** organizers for the public dataset
- **Expert neuroradiologists** for comprehensive manual annotations
- **Tabriz University of Medical Sciences** Research Ethics Committee (IR.TBZMED.REC.1402.902)
- **Open-source community** for foundational tools (TensorFlow, FSL, SPM)

---

## 📞 Contact

- **Lead Author**: Mahdi Bashiri Bawil (m_bashiri99@sut.ac.ir)
- **Corresponding Author**: Mousa Shamsi (shamsi@sut.ac.ir)
- **Repository**: [https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg)
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/Sim-Vent-WMH-Seg/discussions)

---

## 🔑 Key Takeaways

### For Clinicians
✅ **Fast & Accurate**: 4-second processing with 0.907 Dice for ventricles, 0.825 for WMH  
✅ **Clinically Relevant**: 84.28% accuracy distinguishing normal from pathological hyperintensities  
✅ **Easy Integration**: Compatible with standard clinical MRI protocols (1.5T FLAIR)  
✅ **Reliable**: 96.38% precision when predicting pathological lesions

### For Researchers
✅ **Systematic Approach**: Complete ablation study (V0-V5) showing component contributions  
✅ **Rigorous Validation**: 5-fold CV + 6 baseline comparisons + cross-dataset testing  
✅ **Open Source**: Full code, pre-trained models, and documentation available  
✅ **Reproducible**: Detailed methodology from data acquisition to evaluation

### For Developers
✅ **Modern Stack**: TensorFlow 2.x, Python 3.9+, modular architecture  
✅ **Well-Documented**: Comprehensive API docs and usage examples  
✅ **Efficient**: 2D slice-based approach, GPU-optimized inference  
✅ **Extensible**: Clear architecture for adding new features

---

## 🌟 Star History

If you find this work useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Mahdi-Bashiri/Sim-Vent-WMH-Seg&type=Date)](https://star-history.com/#Mahdi-Bashiri/Sim-Vent-WMH-Seg&Date)

---

## 📊 Quick Reference Tables

### Evaluation Metrics Formulas

| Metric | Formula | Range | Best Value |
|--------|---------|-------|------------|
| Dice | 2TP / (2TP + FP + FN) | [0, 1] | 1 |
| IoU | TP / (TP + FP + FN) | [0, 1] | 1 |
| Precision | TP / (TP + FP) | [0, 1] | 1 |
| Recall | TP / (TP + FN) | [0, 1] | 1 |
| HD95 | 95th percentile distance | [0, ∞) mm | 0 |

### Dataset Distribution

| Split | Local | Public | Total Patients | Total Images |
|-------|-------|--------|----------------|--------------|
| Training | 210 | 9 | 219 | 4,650 |
| Validation | 30 | 3 | 33 | 750 |
| Test | 60 | 3 | 63 | 1,350 |
| **Total** | **300** | **15** | **315** | **6,750** |

---

**Last Updated**: February 2025  
**Manuscript Version**: R1 (Revised)  
**Ethics Approval**: IR.TBZMED.REC.1402.902
