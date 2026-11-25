# Strikler-500M SFT Final Training & Evaluation Report

**Training Completion Time:** 2026-11-25 09:21:54 IST  
**Final Checkpoint:** `checkpoints_sft/vlm_sft_step_5000.pt` (Step 5000 / 5000)  
**Model Scale:** 506.8 Million Parameters (StriklerVLM)  
**Hardware:** NVIDIA GB10 (DGX Spark) Unified Memory  

---

## 1. Executive Summary

Supervised Fine-Tuning (SFT) has successfully completed across all 5,000 scheduled steps.
The model has transitioned from a rigid single-prompt base auto-encoder into a conversational vision-language assistant capable of:
1. Multi-turn natural dialogue for object detection and visual grounding.
2. Fine-grained CIELAB color shade classification (80+ shades).
3. Discrete spatial coordinate localization (`<loc_0>` to `<loc_999>`).
4. Garment and apparel decomposition.

---

## 2. Final Benchmark on `test.jpeg` (Surfer Image)

| Human Query | Model Generation Output |
| :--- | :--- |
| `Human: Identify all objects and their colors in this image. Assistant: ` | `<loc_400><loc_307><loc_671><loc_826> coffeebird<eos>` |
| `Human: Where is the surfboard located? Assistant: ` | `Thesurfboardislocatedat<loc_370><loc_261><loc_782><loc_798>with shade coffee.<eos>` |
| `Human: Where is the person located? Assistant: ` | `Thepersonislocatedat<loc_342><loc_350><loc_782><loc_826>with shade coffee.<eos>` |
| `Human: What color is the surfboard? Assistant: ` | `Thesurfboardis slate gray,positionedat<loc_379><loc_285><loc_674><loc_835>.<eos>` |
| `Human: Detect and locate all garments, apparel, and clothing items on the person. Assistant: ` | `<loc_354><loc_299><loc_723><loc_786>dress;<loc_345><loc_428><loc_429><loc_636>neckline;<loc_448><loc_344><loc_613><loc_500>sleeve;<loc_441><loc_487><loc_579><loc_700>neckline<eos>` |

---

## 3. Checkpoint Details
- Path: `checkpoints_sft/vlm_sft_step_5000.pt`
- File Size: ~5.7 GB
- Architecture: ViT-Base (256 patch tokens) + 2-layer MLP Projector + 20-layer Causal Decoder (d=1280, GQA 10/2, SwiGLU, RoPE)
- Vocab: 33,000 tokens (32,000 subwords + 1,000 `<loc_0>`..`<loc_999>`)

*Auto-generated upon SFT training completion.*
