"""
RefCOCO / RefCOCO+ Annotation Downloader and Converter for Strikler SFT.
Reuses existing COCO 2017 images already in data/user_dataset/images.
Fetches referring expressions (e.g., 'man in yellow shirt', 'black racket') 
and formats them into SFT instruction-tuning conversations.
"""

import os
import json
import urllib.request
import zipfile

REFCOCO_URL = "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"
REFCOCO_G_URL = "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"

def download_and_extract_refcoco(target_dir="data/refcoco"):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "refcoco.zip")
    
    if not os.path.exists(zip_path):
        print(f"[*] Downloading RefCOCO annotations from {REFCOCO_URL}...")
        try:
            urllib.request.urlretrieve(REFCOCO_URL, zip_path)
            print("[✓] Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("[✓] Extracted RefCOCO successfully.")
        except Exception as e:
            print(f"[!] Direct download error: {e}")
            print("[*] Alternative: can be loaded via HuggingFace datasets library 'lmms-lab/RefCOCO'")

if __name__ == "__main__":
    download_and_extract_refcoco()
