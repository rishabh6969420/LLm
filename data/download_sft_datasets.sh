#!/bin/bash
# =============================================================================
# Master SFT Dataset Downloader for Strikler-500M
# Downloads:
#   1. LLaVA-Instruct-150K (Conversational VQA - Reuses local COCO images)
#   2. Fashionpedia (Official CVDF S3 - Garments & Clothing Decomposition)
# =============================================================================

set -e

PROJECT_ROOT="$(pwd)"
LLAVA_DIR="${PROJECT_ROOT}/data/sft_raw/llava"
FASHION_DIR="${PROJECT_ROOT}/data/sft_raw/fashionpedia"

mkdir -p "${LLAVA_DIR}"
mkdir -p "${FASHION_DIR}"

echo "==========================================================================="
echo "📥 1. DOWNLOADING LLAVA-INSTRUCT-150K (Reuses existing COCO 2017 images)"
echo "==========================================================================="
cd "${LLAVA_DIR}"

if [ ! -f "conversation_58k.json" ]; then
    echo "[*] Downloading conversation_58k.json (126 MB)..."
    wget -c -q --show-progress https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json
fi

if [ ! -f "detail_23k.json" ]; then
    echo "[*] Downloading detail_23k.json (20.5 MB)..."
    wget -c -q --show-progress https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json
fi

if [ ! -f "complex_reasoning_77k.json" ]; then
    echo "[*] Downloading complex_reasoning_77k.json (79.6 MB)..."
    wget -c -q --show-progress https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json
fi

echo "[✓] LLaVA-Instruct-150K downloaded successfully!"

echo "==========================================================================="
echo "📥 2. DOWNLOADING FASHIONPEDIA (Garment Decomposition: Jersey vs Skin)"
echo "==========================================================================="
cd "${FASHION_DIR}"

if [ ! -f "instances_attributes_train2020.json" ]; then
    echo "[*] Downloading Fashionpedia Annotations (542 MB)..."
    wget -c -q --show-progress https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
fi

if [ ! -f "train2020.zip" ]; then
    echo "[*] Downloading Fashionpedia Images (3.34 GB)..."
    wget -c -q --show-progress https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
fi

if [ ! -d "images" ] && [ -f "train2020.zip" ]; then
    echo "[*] Extracting train2020.zip..."
    mkdir -p images
    unzip -q train2020.zip -d images/
fi

cd "${PROJECT_ROOT}"

echo "==========================================================================="
echo "🎉 ALL EXTERNAL SFT DATASETS DOWNLOADED AND EXTRACTED SUCCESSFULLY!"
echo "==========================================================================="
