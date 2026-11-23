"""
Interactive Testing Utility for Strikler-500M.
Keeps model loaded in memory for instant (<100ms) multi-image testing.
Allows testing custom images and prompts interactively.
Usage:
  python test_interactive.py
"""

import os
import glob
import torch
from PIL import Image
import numpy as np

from config import get_model_preset
from model.vision_llm import StriklerVLM
from data.tokenizer_utils import VLMTokenizer
from engine.visualize import parse_vlm_detection_output, draw_detections_and_palette


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    ckpts = glob.glob(os.path.join(checkpoint_dir, "vlm_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # Sort by step number
    def get_step(path):
        base = os.path.basename(path).replace("vlm_step_", "").replace(".pt", "")
        return int(base) if base.isdigit() else 0
    ckpts.sort(key=get_step, reverse=True)
    return ckpts[0]


def main():
    print("=" * 75)
    print("🚀 STRIKLER-500M: INTERACTIVE LIVE TESTING CLI")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latest_ckpt = find_latest_checkpoint()
    print(f"[*] Found Latest Checkpoint: {latest_ckpt}")
    print(f"[*] Loading weights onto {device} (takes ~5 seconds once)...")

    # Load Model once into VRAM
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
    if "config" in ckpt:
        v_cfg = ckpt["config"]["vision"]
        p_cfg = ckpt["config"]["projector"]
        l_cfg = ckpt["config"]["llm"]
    else:
        v_cfg, p_cfg, l_cfg = get_model_preset("500M")

    model = StriklerVLM(v_cfg, p_cfg, l_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = VLMTokenizer()

    print(f"[✓] Model Loaded Successfully! (Step: {ckpt.get('step', 'N/A')})")
    print("-" * 75)
    print("Commands:")
    print("  • Type an image path (e.g. /home/piet/Downloads/test.jpeg)")
    print("  • Type 'random' to test a random photo from the dataset")
    print("  • Type 'quit' or 'exit' to stop")
    print("=" * 75)

    test_counter = 1
    while True:
        try:
            user_input = input("\n👉 Enter image path (or 'random' / 'quit'): ").strip()
            if not user_input or user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting interactive testing. Good job!")
                break

            if user_input.lower() == "random":
                images = glob.glob("data/user_dataset/images/*.jpg")
                if not images:
                    print("[!] No images found in data/user_dataset/images/")
                    continue
                import random
                img_path = random.choice(images)
                print(f"[*] Selected Random Image: {img_path}")
            else:
                img_path = os.path.expanduser(user_input)
                if not os.path.exists(img_path):
                    print(f"[!] File not found: {img_path}")
                    continue

            # Prompt selection
            prompt_input = input("👉 Enter prompt task [Press Enter for '<detect_all>']: ").strip()
            task_prompt = prompt_input if prompt_input else "<detect_all>"

            # Load and Preprocess
            raw_img = Image.open(img_path).convert("RGB")
            resized_img = raw_img.resize((v_cfg.image_size, v_cfg.image_size))

            arr = np.array(resized_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

            # Generate
            full_prompt = f"<bos>{task_prompt}"
            prompt_ids = torch.tensor([tokenizer.encode(full_prompt)], dtype=torch.long, device=device)

            print("\n⏳ Model is thinking & generating tokens...")
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    generated_ids = model.generate(img_tensor, prompt_ids, max_new_tokens=64, temperature=0.7)

            gen_text = tokenizer.decode(generated_ids[0].tolist())
            print(f"\n[*] 🤖 Raw Generated Output:\n    {gen_text}")

            # Parse and Visualize
            detections = parse_vlm_detection_output(gen_text, image_size=v_cfg.image_size)
            print(f"\n[*] 📋 Parsed Detections ({len(detections)} objects found):")
            if not detections:
                print("    (No distinct bounding boxes detected)")
            for i, det in enumerate(detections, 1):
                print(f"    {i}. {det['description']:<20} | Shade: {det['shade']:<15} ({det['hex']}) | Box: {det['bbox']}")

            out_file = f"interactive_test_{test_counter}.png"
            draw_detections_and_palette(resized_img, detections, output_path=out_file)
            print(f"\n[✓] Annotated photo with color swatches saved to: {out_file}")
            test_counter += 1

        except KeyboardInterrupt:
            print("\nExiting interactive testing.")
            break
        except Exception as e:
            print(f"[!] Error during inference: {e}")


if __name__ == "__main__":
    main()
