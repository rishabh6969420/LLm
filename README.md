# 🎨 ChromaDetect-VLM: Multimodal Vision-Language Model for Object Detection & Color Grounding

**ChromaDetect-VLM** is a research-grade Multimodal Vision-Language Model (VLM) engineered from first principles in PyTorch. It bridges computer vision and autoregressive causal language modeling to perform simultaneous **spatial object detection, bounding-box grounding, and fine-grained color analysis**.

Designed specifically for training on high-performance accelerators like the **NVIDIA DGX Spark (NVIDIA GB10, 121 GB VRAM)**.

---

## 🏛️ Model Architecture

```
                                  ┌───────────────────────────────┐
                                  │      Input RGB Image          │
                                  │       (224 x 224 x 3)         │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Vision Transformer (ViT)    │
                                  │  • 14x14 Patch Embeddings     │
                                  │  • 12 Layers, 12 Heads (768d) │
                                  └───────────────┬───────────────┘
                                                  │ (256 Visual Tokens)
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │  Multimodal MLP Projector     │
                                  │  • Linear(768 -> 2048) + GELU │
                                  │  • Linear(2048 -> 1024)       │
                                  └───────────────┬───────────────┘
                                                  │
┌──────────────────────────────┐                  │
│       Prompt Tokens          │                  │
│  <detect_all> / <color_at>   │                  │
└──────────────┬───────────────┘                  │
               │ (Token Embeddings)               │
               ▼                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │         Concatenated Multimodal Embeddings             │
      │        [Visual Tokens (256) | Text Tokens (N)]         │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │               Causal LLM Decoder                       │
      │  • 16 Layers, 16 Query Heads, 4 KV Heads (GQA)         │
      │  • Hidden Dim: 1024 | Intermediate Dim: 2816 (SwiGLU)  │
      │  • Rotary Positional Embeddings (RoPE) & RMSNorm       │
      │  • FlashAttention-2 / Scaled Dot-Product Attention     │
      └───────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                 Output Token Sequence                  │
      │  <loc_y1><loc_x1><loc_y2><loc_x2> [color] [object]     │
      │  Hex code, Swatches, and Detailed Attributes           │
      └────────────────────────────────────────────────────────┘
```

### Key Technical Innovations
1. **Discrete Coordinate Tokenization**: Special tokens `<loc_0>` through `<loc_999>` represent continuous bounding box coordinates $[y_{min}, x_{min}, y_{max}, x_{max}]$ normalized to $1000 \times 1000$ coordinate grids.
2. **Color Attribute Grounding**: Jointly learns visual spectral features alongside object geometry, enabling natural queries like:
   - `<detect_all>`: Detects all objects, coordinates, and dominant colors.
   - `<detect_color> red`: Finds and localizes specifically red objects.
   - `<color_at> <loc_y1><loc_x1><loc_y2><loc_x2>`: Identifies exact color name and Hex value for a target bounding box.
3. **Hardware-Optimized (BF16 & FlashAttention)**: Utilizes `torch.amp.autocast` in Bfloat16, Grouped Query Attention (GQA), and gradient accumulation to saturate Tensor Cores on NVIDIA DGX.

---

## 🚀 Quickstart

### 1. Installation & Environment
Ensure you have PyTorch with CUDA support enabled:
```bash
# If using the pre-configured ai-lab-env
source /home/piet/ai-lab-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Demo
Generates a test image with synthetic colored objects, passes it through the coordinate and color extraction pipeline, and saves an annotated result with a color palette sidebar:
```bash
python demo.py
```
Outputs:
- `sample_test_image.png`: The raw input image.
- `demo_annotated_detection.png`: The annotated image with bounding boxes, color tags, and hex swatches.

---

## 🏋️ Training on NVIDIA DGX Spark (~48-Hour Run)

The training engine supports multiple profiles tailored from sanity checks to a full multi-day continuous pretraining run:

### Profile 1: Quick Sanity Check (100 Steps, ~1 minute)
Verifies CUDA memory, backward pass, and checkpoint generation:
```bash
python train.py --mode quick_test
```

### Profile 2: Standard Training (5,000 Steps, ~1-2 hours)
Trains on high-throughput synthetic color-geometry streams:
```bash
python train.py --mode standard --batch_size 32
```

### Profile 3: 2-Day Continuous Training Run (~48 Hours on DGX Spark GB10)
Runs **100,000 steps** with gradient accumulation and cosine annealing learning rate schedule:
```bash
python train.py --mode dgx_2day --batch_size 32 --lr 2e-4
```

### Training Telemetry
During training, live telemetry tracks loss, learning rate, throughput, and VRAM:
```text
[Step   2000/100000] Loss: 1.8421 | LR: 1.95e-04 | Speed: 42.10 step/s | VRAM: 34.20 GB
[✓] Saved model checkpoint: checkpoints/vlm_step_2000.pt
```

---

## 🔍 Inference on Custom Images

Run inference on any image (`.jpg`, `.png`):
```bash
python infer.py \
  --image path/to/your/image.jpg \
  --checkpoint checkpoints/vlm_step_10000.pt \
  --output annotated_result.png \
  --prompt "<detect_all>"
```

### Query Variants:
- Detect all objects and colors: `--prompt "<detect_all>"`
- Detect only blue objects: `--prompt "<detect_color> blue"`
- Interrogate a specific region: `--prompt "<color_at> <loc_100><loc_200><loc_400><loc_600>"`

---

## 📂 Repository Structure

```
.
├── config.py                 # Hyperparameters (Vision, Projector, LLM, Training)
├── train.py                  # Main training entrypoint
├── infer.py                  # Single-image inference and evaluation
├── demo.py                   # End-to-end demo and visualization script
├── requirements.txt          # Python dependencies
├── model/
│   ├── __init__.py
│   ├── vision_encoder.py     # Vision Transformer (ViT) patch encoder
│   ├── projector.py          # Multimodal 2-layer MLP projection adapter
│   ├── llm_backbone.py       # Causal LLM Decoder (RoPE, RMSNorm, SwiGLU, GQA)
│   └── vision_llm.py         # Unified ChromaDetect-VLM architecture
├── data/
│   ├── color_utils.py        # Color palette distance matching & Hex utilities
│   ├── tokenizer_utils.py    # Spatial coordinate tokenization (<loc_0> to <loc_999>)
│   ├── synthetic_generator.py# Diverse geometric and color dataset generator
│   └── dataset.py            # High-throughput PyTorch DataLoader
└── engine/
    ├── __init__.py
    ├── trainer.py            # Multi-GPU/BF16 training loop & checkpoint manager
    └── visualize.py          # Bounding box & color swatch rendering engine
```

---

## 🔬 Scientific & Educational Value

For students and researchers in AI/ML, this repository demonstrates:
1. **Multimodal Alignment**: How continuous image representations are mapped into a discrete autoregressive token vocabulary.
2. **Object Grounding as Language Modeling**: Unifying bounding box detection into next-token prediction without requiring anchor boxes or NMS (Non-Maximum Suppression).
3. **Spatial Representation**: How discrete location tokens preserve sub-pixel spatial resolution.
4. **Color Perception in Neural Networks**: How cross-attention layers disentangle color attributes from shape geometry.
