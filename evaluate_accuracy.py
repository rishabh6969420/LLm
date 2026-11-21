"""
Comprehensive Accuracy & Validation Benchmark for Strikler-100M.
Evaluates:
  1. Mean Intersection over Union (mIoU) for Bounding Boxes
  2. Bounding Box Detection Precision / Recall / F1 @ IoU >= 0.50 (AP50)
  3. Fine-Grained Exact Shade Accuracy (Turquoise, Blood Red, Wine Red, etc.)
  4. Parent Color Family Accuracy (Red, Blue, Green, etc.)
  5. Cross-Entropy Loss & Perplexity on 15% Val & 10% Test Splits
  6. Per-Task Breakdown: <detect_all>, <detect_color>, <color_at>
"""

import os
import re
import math
import json
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_model_preset
from model.vision_llm import StriklerVLM
from data.dataset import get_dataloaders
from data.tokenizer_utils import VLMTokenizer
from data.color_utils import COLOR_SHADE_DATABASE, delta_e_cie76, rgb_to_cielab
from engine.visualize import parse_vlm_detection_output


def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes [y1, x1, y2, x2]."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    inter_area = max(0, yB - yA) * max(0, xB - xA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_model(
    checkpoint_path: str = "checkpoints/vlm_step_3000.pt",
    model_size: str = "100M",
    eval_samples: int = 500,
    batch_size: int = 16,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 75)
    print("🎯 STRIKLER-100M: COMPREHENSIVE ACCURACY & VALIDATION BENCHMARK")
    print("=" * 75)
    print(f"[*] Evaluation Device : {device}")
    print(f"[*] Checkpoint Target : {checkpoint_path}")
    print(f"[*] Evaluation Pool   : {eval_samples} held-out unseen samples")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load Model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        v_cfg = ckpt["config"]["vision"]
        p_cfg = ckpt["config"]["projector"]
        l_cfg = ckpt["config"]["llm"]
    else:
        v_cfg, p_cfg, l_cfg = get_model_preset(model_size)

    model = StriklerVLM(v_cfg, p_cfg, l_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model Loaded: Strikler-{model_size} ({total_params / 1e6:.2f}M parameters)")
    print(f"[*] Checkpoint Training Step: {ckpt.get('step', 'N/A')}\n")

    # 2. Prepare DataLoaders
    tokenizer = VLMTokenizer()
    _, val_loader, test_loader, stats = get_dataloaders(
        total_samples=max(eval_samples * 2, 2000),
        batch_size=batch_size,
        tokenizer=tokenizer,
        num_workers=2,
    )

    # 3. Compute Cross-Entropy Loss on 15% Val & 10% Test
    print("[PHASE 1] Computing Loss & Perplexity on Validation & Test Splits...")
    val_loss_total = 0.0
    val_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 25:
                break
            img = batch["image"].to(device)
            txt = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(img, txt, lbl)["loss"]
            val_loss_total += loss.item()
            val_batches += 1

    val_loss = val_loss_total / max(1, val_batches)
    perplexity = math.exp(min(val_loss, 20))
    print(f"  [✓] 15% Validation Split Loss : {val_loss:.4f} (Perplexity: {perplexity:.2f})")

    test_loss_total = 0.0
    test_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 25:
                break
            img = batch["image"].to(device)
            txt = batch["input_ids"].to(device)
            lbl = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(img, txt, lbl)["loss"]
            test_loss_total += loss.item()
            test_batches += 1

    test_loss = test_loss_total / max(1, test_batches)
    print(f"  [✓] 10% Testing Split Loss    : {test_loss:.4f}")

    # 4. Compute Bounding Box IoU & Shade Classification Accuracy
    print("\n[PHASE 2] Evaluating Spatial Grounding (IoU) & Accurate Color Shades...")
    total_evaluated_objects = 0
    total_iou = 0.0
    tp_iou50 = 0
    tp_iou75 = 0
    total_predicted_boxes = 0
    total_ground_truth_boxes = 0

    exact_shade_matches = 0
    parent_family_matches = 0
    task_counts = {"detect_all": 0, "detect_color": 0, "color_at": 0}
    task_success = {"detect_all": 0, "detect_color": 0, "color_at": 0}

    # Sort shades by descending length to match compound shades properly
    sorted_shades = sorted(COLOR_SHADE_DATABASE.keys(), key=lambda s: len(s), reverse=True)

    test_iter = iter(test_loader)
    samples_processed = 0

    with torch.no_grad():
        while samples_processed < eval_samples:
            try:
                batch = next(test_iter)
            except StopIteration:
                break

            images = batch["image"].to(device)
            prompts = batch["prompt_text"]
            targets = batch["target_text"]
            raw_annotations = batch["annotations"]

            for b in range(images.size(0)):
                if samples_processed >= eval_samples:
                    break
                samples_processed += 1

                img = images[b:b+1]
                prompt_text = prompts[b]
                gt_anns = raw_annotations[b]

                # Determine task
                if "<detect_all>" in prompt_text:
                    task = "detect_all"
                elif "<detect_color>" in prompt_text:
                    task = "detect_color"
                else:
                    task = "color_at"

                task_counts[task] += 1

                # Generate model prediction
                prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    gen_ids = model.generate(img, prompt_ids, max_new_tokens=64, temperature=0.2)

                pred_text = tokenizer.decode(gen_ids[0].tolist())
                pred_dets = parse_vlm_detection_output(pred_text, image_size=224)

                total_ground_truth_boxes += len(gt_anns)
                total_predicted_boxes += len(pred_dets)

                # Match each GT annotation with best predicted box
                sample_success = True
                for gt in gt_anns:
                    total_evaluated_objects += 1
                    gt_box = gt["bbox_norm"]
                    gt_shade = gt["shade"].lower()
                    gt_parent = COLOR_SHADE_DATABASE.get(gt_shade, {}).get("parent", "")

                    best_iou = 0.0
                    best_pred = None
                    for pred in pred_dets:
                        iou = calculate_iou(gt_box, pred["norm_bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = pred

                    total_iou += best_iou
                    if best_iou >= 0.50:
                        tp_iou50 += 1
                    if best_iou >= 0.75:
                        tp_iou75 += 1

                    # Check shade match
                    if best_pred:
                        pred_shade = best_pred["shade"].lower()
                        pred_parent = best_pred.get("parent_color", "").lower()

                        if pred_shade == gt_shade:
                            exact_shade_matches += 1
                        if pred_parent == gt_parent and gt_parent != "":
                            parent_family_matches += 1
                    else:
                        sample_success = False

                if sample_success and len(pred_dets) > 0:
                    task_success[task] += 1

    # Metrics computation
    mean_iou = (total_iou / max(1, total_evaluated_objects)) * 100.0
    precision_50 = (tp_iou50 / max(1, total_predicted_boxes)) * 100.0
    recall_50 = (tp_iou50 / max(1, total_ground_truth_boxes)) * 100.0
    f1_50 = (2 * precision_50 * recall_50) / max(1e-5, (precision_50 + recall_50))

    precision_75 = (tp_iou75 / max(1, total_predicted_boxes)) * 100.0
    recall_75 = (tp_iou75 / max(1, total_ground_truth_boxes)) * 100.0

    exact_shade_acc = (exact_shade_matches / max(1, total_evaluated_objects)) * 100.0
    parent_color_acc = (parent_family_matches / max(1, total_evaluated_objects)) * 100.0

    results = {
        "model_scale": f"Strikler-{model_size}",
        "total_parameters": total_params,
        "checkpoint": checkpoint_path,
        "eval_samples": samples_processed,
        "loss_metrics": {
            "val_split_loss": round(val_loss, 4),
            "test_split_loss": round(test_loss, 4),
            "perplexity": round(perplexity, 2),
        },
        "grounding_metrics": {
            "mean_iou_pct": round(mean_iou, 2),
            "precision_iou50_pct": round(precision_50, 2),
            "recall_iou50_pct": round(recall_50, 2),
            "f1_score_iou50_pct": round(f1_50, 2),
            "precision_iou75_pct": round(precision_75, 2),
            "recall_iou75_pct": round(recall_75, 2),
        },
        "color_accuracy_metrics": {
            "exact_shade_accuracy_pct": round(exact_shade_acc, 2),
            "parent_color_accuracy_pct": round(parent_color_acc, 2),
        },
        "task_breakdown_pct": {
            task: round((task_success[task] / max(1, task_counts[task])) * 100.0, 2)
            for task in task_counts
        },
    }

    print("\n" + "=" * 75)
    print("📊 BENCHMARK ACCURACY RESULTS REPORT")
    print("=" * 75)
    print(f"• Mean IoU (Spatial Accuracy)     : {mean_iou:.2f}%")
    print(f"• Bounding Box AP@50 (IoU >= 0.50): Precision: {precision_50:.2f}% | Recall: {recall_50:.2f}% | F1: {f1_50:.2f}%")
    print(f"• Bounding Box AP@75 (IoU >= 0.75): Precision: {precision_75:.2f}% | Recall: {recall_75:.2f}%")
    print(f"• Exact Shade Accuracy (80+ shades): {exact_shade_acc:.2f}% (Turquoise, Blood Red, Wine Red, etc.)")
    print(f"• Parent Color Accuracy           : {parent_color_acc:.2f}% (Red, Blue, Green, etc.)")
    print(f"• 15% Validation Loss             : {val_loss:.4f}")
    print(f"• 10% Unseen Test Loss            : {test_loss:.4f}")
    print("-" * 75)
    print("Task-Specific Success Rates:")
    for task, rate in results["task_breakdown_pct"].items():
        print(f"  - Task <{task}> : {rate:.2f}%")
    print("=" * 75)

    os.makedirs("logs", exist_ok=True)
    report_file = f"logs/accuracy_report_{model_size}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Full accuracy report saved to: {report_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Strikler-100M accuracy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vlm_step_3000.pt", help="Path to checkpoint")
    parser.add_argument("--model_size", type=str, default="100M", help="Model scale")
    parser.add_argument("--samples", type=int, default=300, help="Number of test samples to benchmark")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        model_size=args.model_size,
        eval_samples=args.samples,
        batch_size=args.batch_size,
    )

