"""
Inference Script for ChromaDetect-VLM.
Detects objects, boundaries, and colors in any target image.
Usage:
  python infer.py --image path/to/image.jpg --output annotated_result.png
"""

import argparse
import os
from typing import Tuple, Optional
import torch
import numpy as np
from PIL import Image

from config import VisionConfig, ProjectorConfig, LLMConfig
from model.vision_llm import ChromaDetectVLM
from data.tokenizer_utils import VLMTokenizer
from engine.visualize import parse_vlm_detection_output, draw_detections_and_palette


def preprocess_image(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess image for VLM."""
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = raw_img.resize((image_size, image_size))

    arr = np.array(resized_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0), resized_img


def run_inference(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: str = "detected_output.png",
    prompt_task: str = "<detect_all>",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = VLMTokenizer()

    # Model Configuration
    vision_cfg = VisionConfig()
    projector_cfg = ProjectorConfig()
    llm_cfg = LLMConfig()

    model = ChromaDetectVLM(vision_cfg, projector_cfg, llm_cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[*] Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[!] No checkpoint provided. Running with initialized weights for demonstration.")

    model.eval()

    img_tensor, resized_img = preprocess_image(image_path, vision_cfg.image_size)
    img_tensor = img_tensor.to(device)

    # Encode prompt
    full_prompt = f"<bos>{prompt_task}"
    prompt_ids = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long, device=device)

    print(f"[*] Running VLM image and color detection on {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(img_tensor, prompt_ids, max_new_tokens=64)

    gen_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"[*] Raw Model Output: {gen_text}")

    # Parse and visualize
    detections = parse_vlm_detection_output(gen_text, image_size=vision_cfg.image_size)
    print(f"[*] Parsed {len(detections)} object & color detections:")
    for i, det in enumerate(detections, 1):
        print(f"    {i}. {det['description']} | Color: {det['color']} ({det['hex']}) | Box: {det['bbox']}")

    annotated = draw_detections_and_palette(resized_img, detections, output_path=output_path)
    print(f"[✓] Annotated detection output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ChromaDetect-VLM inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--output", type=str, default="annotated_detection.png", help="Path to save output")
    parser.add_argument("--prompt", type=str, default="<detect_all>", help="Task prompt: <detect_all>, <detect_color>, or <color_at>")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.output, args.prompt)
