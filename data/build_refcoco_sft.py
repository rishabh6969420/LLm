"""
RefCOCO to Strikler SFT Converter.
Streams 'jxu124/refcoco' from Hugging Face and creates referring expression
grounding conversations matching local images in data/user_dataset/images.
"""

import os
import json
from datasets import load_dataset
from PIL import Image

def process_refcoco_to_sft(output_path="data/user_dataset/refcoco_sft.json", max_samples=50000):
    print("[*] Streaming RefCOCO dataset from Hugging Face ('jxu124/refcoco')...")
    ds = load_dataset("jxu124/refcoco", split="train", streaming=True)
    
    sft_data = []
    local_images_dir = "data/user_dataset/images"
    
    count = 0
    skipped = 0
    for sample in ds:
        img_id = sample.get("image_id")
        # Match with local file naming convention: clean_000000XXXXXX.jpg or clean_XXXXXX.jpg
        cand1 = f"images/clean_{img_id:06d}.jpg"
        cand2 = f"images/clean_{img_id:012d}.jpg"
        cand3 = f"images/clean_{img_id}.jpg"
        
        target_file = None
        for cand in [cand1, cand2, cand3]:
            if os.path.exists(os.path.join("data/user_dataset", cand)):
                target_file = cand
                break
                
        if not target_file:
            skipped += 1
            continue
            
        bbox = sample.get("bbox") # [x, y, w, h] in pixels
        raw_info = sample.get("raw_image_info", {})
        if isinstance(raw_info, str):
            try:
                raw_info = json.loads(raw_info)
            except Exception:
                raw_info = {}
        h_orig = raw_info.get("height", 0) if isinstance(raw_info, dict) else 0
        w_orig = raw_info.get("width", 0) if isinstance(raw_info, dict) else 0
        
        if h_orig <= 0 or w_orig <= 0:
            # Read image size from local file
            full_path = os.path.join("data/user_dataset", target_file)
            try:
                with Image.open(full_path) as im:
                    w_orig, h_orig = im.size
            except Exception:
                continue
                
        x, y, w, h = bbox
        y1 = int(round((y / h_orig) * 999.0))
        x1 = int(round((x / w_orig) * 999.0))
        y2 = int(round(((y + h) / h_orig) * 999.0))
        x2 = int(round(((x + w) / w_orig) * 999.0))
        
        # Clamp coordinates [0, 999]
        y1 = max(0, min(999, y1))
        x1 = max(0, min(999, x1))
        y2 = max(0, min(999, y2))
        x2 = max(0, min(999, x2))
        
        box_str = f"<loc_{y1}><loc_{x1}><loc_{y2}><loc_{x2}>"
        
        sentences = sample.get("sentences", [])
        for s in sentences:
            sent_text = s.get("sent", "").strip().lower()
            if not sent_text:
                continue
                
            sft_data.append({
                "image_file": target_file,
                "task": "referring_expression",
                "conversations": [
                    {"from": "human", "value": f"Where is {sent_text}?"},
                    {"from": "gpt", "value": f"{sent_text.capitalize()} is located at {box_str}.<eos>"}
                ]
            })
            count += 1
            if count >= max_samples:
                break
                
        if count >= max_samples:
            break
            
    print(f"[*] Processed {count:,} RefCOCO referring expression conversations (skipped {skipped:,} non-local).")
    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=2)
    print(f"[✓] Saved RefCOCO SFT data to: {output_path}")

if __name__ == "__main__":
    process_refcoco_to_sft()
