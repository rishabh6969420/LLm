#!/usr/bin/env bash
# ==============================================================================
# Auto Watcher & Git Push Script for Strikler SFT Completion
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

LOG_FILE="${SCRIPT_DIR}/auto_push.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SFT Auto-Push Watcher Started"
echo "Repository: ${REPO_DIR}"
echo "=================================================================="

# Check if train_sft is running
echo "[*] Monitoring train_sft.py process..."
while pgrep -f "python.*train_sft.py" > /dev/null 2>&1; do
    # Log progress every ~5 minutes
    sleep 30
done

echo "[*] [$(date '+%Y-%m-%d %H:%M:%S')] train_sft.py process has finished!"

# Wait a few seconds for file buffers to flush
sleep 10

# Verify checkpoint
FINAL_CKPT="${REPO_DIR}/checkpoints_sft/vlm_sft_step_5000.pt"
if [ -f "${FINAL_CKPT}" ]; then
    echo "[✓] Found final SFT checkpoint: ${FINAL_CKPT} ($(du -h "${FINAL_CKPT}" | cut -f1))"
else
    echo "[!] Warning: vlm_sft_step_5000.pt not found, checking latest checkpoint in checkpoints_sft/..."
    ls -lh "${REPO_DIR}/checkpoints_sft/"
fi

# Run final automated evaluation
echo "[*] Running automated final evaluation..."
/home/piet/ai-lab-env/bin/python "${REPO_DIR}/scripts/evaluate_sft_final.py" || true

# Git Commit and Push
echo "[*] Staging evaluation report and pipeline scripts..."
git add SFT_FINAL_REPORT.md scripts/evaluate_sft_final.py scripts/auto_push_on_sft_complete.sh .gitignore

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "[*] No new changes to commit."
else
    echo "[*] Committing changes..."
    git commit -m "Auto-commit: Complete SFT training (Step 5000), evaluation report, and final benchmarks"
    
    echo "[*] Pushing to GitHub (origin/main)..."
    git push origin main
    echo "[✓] [$(date '+%Y-%m-%d %H:%M:%S')] Successfully pushed SFT completion updates to GitHub!"
fi

echo "=================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SFT AUTO-PUSH WATCHER FINISHED SUCCESSFULLY"
echo "=================================================================="
