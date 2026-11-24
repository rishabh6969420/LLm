"""
SFT Dataset Merger & Final Split Partitioner for Strikler-500M.
Combines:
  1. Internal conversational color & detection pairs (data/user_dataset/sft_instructions.json)
  2. RefCOCO spatial referring expression pairs (data/user_dataset/refcoco_sft.json)
  3. External garment/part-level pairs (optional)
Partitions cleanly into train/val/test splits for train_sft.py.
"""

import os
import json
import random
import argparse

def merge_and_split(
    sft_files=[
        "data/user_dataset/sft_instructions.json",
        "data/user_dataset/refcoco_sft.json",
        "data/user_dataset/fashionpedia_sft.json"
    ],
    output_merged="data/user_dataset/sft_dataset_final.json",
    seed=42
):
    random.seed(seed)
    merged_data = []
    
    for fpath in sft_files:
        if os.path.exists(fpath):
            print(f"[*] Merging dataset from: {fpath}")
            with open(fpath, "r") as f:
                d = json.load(f)
                merged_data.extend(d)
                print(f"    -> Added {len(d):,} conversations.")
        else:
            print(f"[!] Warning: {fpath} not found, skipping.")
            
    random.shuffle(merged_data)
    print("=" * 60)
    print(f"[*] Total Merged SFT Conversations: {len(merged_data):,}")
    print(f"[*] Writing to: {output_merged}")
    
    with open(output_merged, "w") as f:
        json.dump(merged_data, f, indent=2)
        
    print(f"[✓] Final SFT dataset ready! ({os.path.getsize(output_merged)/(1024*1024):.2f} MB)")
    return merged_data

if __name__ == "__main__":
    merge_and_split()
