import os
import gdown
import zipfile
import requests
from pathlib import Path


def download_from_google_drive(output_path):
    """Download file from Google Drive"""
    try:
        print(f"Downloading model to {output_path}...")
        gdown.download(f"https://drive.google.com/drive/folders/1vDjKp9K9JCnIWBs0NTjdnVAcSbatb8S-?usp=drive_link", output_path, quiet=False)
        print("Download completed successfully!")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def setup_models():
    """Main function to download and setup models"""
    
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    success = download_from_google_drive(model_path)
    
    if not success:
        print("Failed to download!")
        return False
    
    print("\n✅ All models downloaded successfully!")
    return True


if __name__ == "__main__":
    # Install required package if not available
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown")
        import gdown
    
    setup_models()
