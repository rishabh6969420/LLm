# 📁 Custom Dataset Directory for Strikler

Place your self-collected and cleaned dataset in this folder.

---

## 🗂️ Expected Directory Structure

```text
data/user_dataset/
├── images/
│   ├── image_0001.jpg (or .png)
│   ├── image_0002.jpg
│   └── ...
└── annotations.json  (or annotations.jsonl)
```

---

## 📝 Annotation Format Options

### Option 1: Simple JSON (`annotations.json`) - Recommended
```json
[
  {
    "image_file": "images/image_0001.jpg",
    "objects": [
      {
        "label": "car",
        "shade": "blood red",
        "bbox": [120, 340, 450, 600]
      },
      {
        "label": "cup",
        "shade": "turquoise",
        "bbox": [500, 100, 800, 300]
      }
    ]
  }
]
```
> **Coordinates:** Normalized to $[0, 999]$ as `[ymin, xmin, ymax, xmax]`, or normalized floats $[0.0, 1.0]$. Both are automatically detected.

### Option 2: JSON Lines (`annotations.jsonl`)
Each line is a self-contained JSON object:
```json
{"image_file": "images/image_0001.jpg", "objects": [{"label": "car", "shade": "blood red", "bbox": [120, 340, 450, 600]}]}
{"image_file": "images/image_0002.jpg", "objects": [{"label": "bottle", "shade": "wine red", "bbox": [50, 120, 300, 400]}]}
```

---

## 🔍 Eligibility Check & Pretraining Gate

Once you have added your data, run:
```bash
python check_and_train.py --model_size 500M
```
* **If not ready / insufficient data:** It tells you exactly what is missing and how many more samples to prepare.
* **If ready & passed:** It automatically splits into **75% Train / 15% Val / 10% Test** and begins pretraining!
