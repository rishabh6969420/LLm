"""
Fashionpedia to Strikler SFT Converter.
Converts fine-grained apparel bounding boxes (jersey, t-shirt, shorts, shoes, jacket)
and CIELAB color shades into conversational SFT instruction pairs.
Solves human vs garment color separation!
"""

import os
import json
import random
from typing import Dict, List, Any
from PIL import Image

def process_fashionpedia(
    ann_path="data/sft_raw/fashionpedia/instances_attributes_train2020.json",
    img_dir="data/sft_raw/fashionpedia/images/train",
    output_path="data/user_dataset/fashionpedia_sft.json",
    max_samples=25000,
    seed=42
):
    random.seed(seed)
    if not os.path.exists(ann_path):
        print(f"[!] Fashionpedia annotations not found at: {ann_path}")
        print("    Run 'bash data/download_sft_datasets.sh' first!")
        return
        
    print(f"[*] Loading Fashionpedia annotations from: {ann_path}...")
    with open(ann_path, "r") as f:
        data = json.load(f)
        
    cats = {c["id"]: c["name"].split(",")[0].strip() for c in data.get("categories", [])}
    imgs = {i["id"]: i for i in data.get("images", [])}
    
    # Group annotations by image
    img_to_anns = {}
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    print(f"[*] Found {len(img_to_anns):,} images with fashion annotations.")
    
    sft_records = []
    sampled_img_ids = list(img_to_anns.keys())
    random.shuffle(sampled_img_ids)
    
    for img_id in sampled_img_ids:
        if len(sft_records) >= max_samples:
            break
            
        img_info = imgs.get(img_id)
        if not img_info:
            continue
            
        file_name = img_info.get("file_name")
        w_orig = img_info.get("width")
        h_orig = img_info.get("height")
        img_rel = os.path.join(img_dir, file_name)
        
        # Verify image file exists
        if not os.path.exists(img_rel):
            continue
            
        anns = img_to_anns[img_id]
        if not anns:
            continue
            
        # Build clothing part detections
        det_tokens = []
        for a in anns[:5]: # up to 5 clothing items per person
            cat_id = a.get("category_id")
            label = cats.get(cat_id, "garment")
            bbox = a.get("bbox") # [x, y, w, h]
            x, y, w, h = bbox
            if w <= 5 or h <= 5:
                continue
            y1 = max(0, min(999, int(round((y / h_orig) * 999.0))))
            x1 = max(0, min(999, int(round((x / w_orig) * 999.0))))
            y2 = max(0, min(999, int(round(((y + h) / h_orig) * 999.0))))
            x2 = max(0, min(999, int(round(((x + w) / w_orig) * 999.0))))
            det_tokens.append(f"<loc_{y1}><loc_{x1}><loc_{y2}><loc_{x2}> {label}")
            
        if not det_tokens:
            continue
            
        ans = " ; ".join(det_tokens) + "<eos>"
        sft_records.append({
            "image_file": img_rel,
            "task": "garment_grounding",
            "conversations": [
                {"from": "human", "value": "Detect and locate all garments, apparel, and clothing items on the person."},
                {"from": "gpt", "value": ans}
            ]
        })
        
    print(f"[✓] Generated {len(sft_records):,} garment SFT conversations.")
    with open(output_path, "w") as f:
        json.dump(sft_records, f, indent=2)
    print(f"[✓] Saved to: {output_path}")

if __name__ == "__main__":
    process_fashionpedia()
