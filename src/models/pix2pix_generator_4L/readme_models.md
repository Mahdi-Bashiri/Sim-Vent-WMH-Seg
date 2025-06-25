# Model Setup Instructions

## Pre-trained Models

Due to file size limitations, the pre-trained models are hosted externally.

### Quick Setup:
```bash
# Run the download script
python download_models.py
```

### Manual Download:
1. **Pix2Pix Generator Model** (200MB)
   - Download: [Google Drive Link](https://drive.google.com/drive/folders/1vDjKp9K9JCnIWBs0NTjdnVAcSbatb8S-?usp=sharing)
   - Move to: `models/pix2pix_generator_4L/`

### Model Structure:
```
models/
└── pix2pix_generator_4L/
    ├── saved_model.pb
    ├── variables/
    │   ├── variables.data-00000-of-00001
    │   └── variables.index
    └── assets/
```

### Troubleshooting:
- If download fails, try downloading manually from the links above
- Ensure you have sufficient disk space (>500MB)
- Check internet connection for large file downloads
