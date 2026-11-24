"""
SFT Dataset Builder for Strikler-500M.
Transforms verified image annotations into diverse conversational VLM instruction-tuning data:
  1. General Object & Color Grounding (multi-template detection)
  2. Targeted Spatial Queries ("Where is X?")
  3. Color Question Answering ("What shade is X?")
  4. Grounded Visual Search ("Find the [shade] [object]")
  5. Verification & Negative Queries ("Is there a [color] [object]?")
  6. Color-Balanced Resampling (boosts vibrant rare colors to cure earth-tone bias)
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any
from collections import defaultdict

PROMPT_TEMPLATES = {
    "detect_all": [
        "Identify all objects and their colors in this image.",
        "Detect all visible objects, their bounding boxes, and accurate shades.",
        "What objects and color shades are present in this photo?",
        "Locate every object in the scene with its spatial coordinates and color.",
        "Provide a complete detection breakdown including shades and locations.",
        "<detect_all>",  # preserve backward compatibility
    ],
    "where_is": [
        "Where is the {label} located?",
        "Can you locate the {label} in this image?",
        "Show me the coordinates for the {label}.",
        "Find the bounding box of the {label}.",
    ],
    "what_color": [
        "What color is the {label}?",
        "What shade is the {label} in this picture?",
        "Can you identify the exact color of the {label}?",
        "Tell me the shade of the {label} located at {box}.",
    ],
    "find_color_obj": [
        "Find the {shade} {label} in this photo.",
        "Where is the {shade} {label}?",
        "Locate any {shade} {label} in the scene.",
    ],
    "is_there": [
        "Is there a {shade} {label} in this image?",
        "Do you see a {shade} {label} here?",
    ]
}

EARTH_TONES = {"coffee", "espresso", "slate gray", "charcoal", "black", "white"}

def build_sft_pairs(
    annotations_path: str = "data/user_dataset/annotations.json",
    output_path: str = "data/user_dataset/sft_instructions.json",
    max_samples_per_image: int = 4,
    vibrant_color_boost: float = 2.0,
    seed: int = 42
):
    random.seed(seed)
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations not found at: {annotations_path}")
        
    print(f"[*] Loading raw dataset annotations from: {annotations_path}")
    with open(annotations_path, "r") as f:
        data = json.load(f)
        
    print(f"[*] Total available images: {len(data):,}")
    
    sft_records: List[Dict[str, Any]] = []
    shade_counts = defaultdict(int)
    task_counts = defaultdict(int)

    for item in data:
        img_file = item.get("image_file")
        objs = item.get("objects", [])
        if not objs or not img_file:
            continue
            
        # Format detection string: <loc_y1><loc_x1><loc_y2><loc_x2> shade label;
        det_tokens = []
        for o in objs:
            b = o.get("bbox_norm") or o.get("bbox")
            shade = o.get("shade", "").lower().strip()
            label = o.get("label", "").lower().strip()
            shade_counts[shade] += 1
            det_tokens.append(f"<loc_{b[0]}><loc_{b[1]}><loc_{b[2]}><loc_{b[3]}> {shade} {label}")
            
        full_det_target = " ; ".join(det_tokens) + "<eos>"
        
        # 1. Full Detection Conversation
        detect_prompt = random.choice(PROMPT_TEMPLATES["detect_all"])
        sft_records.append({
            "image_file": img_file,
            "task": "detect_all",
            "conversations": [
                {"from": "human", "value": detect_prompt},
                {"from": "gpt", "value": full_det_target}
            ]
        })
        task_counts["detect_all"] += 1
        
        # Determine if image contains vibrant (non-earth) colors
        has_vibrant = any(o.get("shade", "").lower() not in EARTH_TONES for o in objs)
        boost = vibrant_color_boost if has_vibrant else 1.0
        
        # 2. Targeted Object Queries (Where is X? & What color is X?)
        sampled_objs = random.sample(objs, min(len(objs), 2))
        for o in sampled_objs:
            b = o.get("bbox_norm") or o.get("bbox")
            shade = o.get("shade", "").lower().strip()
            label = o.get("label", "").lower().strip()
            box_str = f"<loc_{b[0]}><loc_{b[1]}><loc_{b[2]}><loc_{b[3]}>"
            
            # Where is X?
            if random.random() < 0.6 * boost:
                q_where = random.choice(PROMPT_TEMPLATES["where_is"]).format(label=label)
                ans_where = f"The {label} is located at {box_str} with shade {shade}.<eos>"
                sft_records.append({
                    "image_file": img_file,
                    "task": "where_is",
                    "conversations": [
                        {"from": "human", "value": q_where},
                        {"from": "gpt", "value": ans_where}
                    ]
                })
                task_counts["where_is"] += 1
                
            # What color is X?
            if random.random() < 0.6 * boost:
                q_col = random.choice(PROMPT_TEMPLATES["what_color"]).format(label=label, box=box_str)
                ans_col = f"The {label} is {shade}, positioned at {box_str}.<eos>"
                sft_records.append({
                    "image_file": img_file,
                    "task": "what_color",
                    "conversations": [
                        {"from": "human", "value": q_col},
                        {"from": "gpt", "value": ans_col}
                    ]
                })
                task_counts["what_color"] += 1

            # Find [shade] [label]
            if random.random() < 0.4 * boost:
                q_find = random.choice(PROMPT_TEMPLATES["find_color_obj"]).format(shade=shade, label=label)
                ans_find = f"Found {shade} {label} at {box_str}.<eos>"
                sft_records.append({
                    "image_file": img_file,
                    "task": "find_color_obj",
                    "conversations": [
                        {"from": "human", "value": q_find},
                        {"from": "gpt", "value": ans_find}
                    ]
                })
                task_counts["find_color_obj"] += 1

    # Shuffle dataset
    random.shuffle(sft_records)
    
    print("=" * 60)
    print("✨ SFT CONVERSATIONAL INSTRUCTION DATASET GENERATED")
    print("=" * 60)
    print(f"[*] Total SFT Instruction Pairs: {len(sft_records):,}")
    print("[*] Task Breakdown:")
    for t, cnt in task_counts.items():
        print(f"    • {t:15s}: {cnt:,} conversations ({cnt/len(sft_records)*100:.1f}%)")
    print(f"[*] Saving SFT dataset to: {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(sft_records, f, indent=2)
        
    print(f"[✓] Saved successfully! ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
    return sft_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="data/user_dataset/annotations.json")
    parser.add_argument("--output", type=str, default="data/user_dataset/sft_instructions.json")
    parser.add_argument("--vibrant_boost", type=float, default=2.0)
    args = parser.parse_args()
    build_sft_pairs(args.annotations, args.output, vibrant_color_boost=args.vibrant_boost)
