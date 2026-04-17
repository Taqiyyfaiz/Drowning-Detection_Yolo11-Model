# Drowning Detection System

A computer vision project that detects drowning incidents in swimming pool footage using YOLOv11 object detection and a temporal state machine to reduce false alarms.

Built as a final-year B.Tech project at Dr. MGR Educational and Research Institute, Chennai.

---

## What It Does

The system takes a video feed (recorded or live), runs YOLOv11 to detect people in the water, and classifies each detected person as "normal" (swimming) or "drowning". It doesn't trigger an alert on a single suspicious frame — it watches across multiple frames before deciding, which cuts down on false positives from someone just diving in or floating.

Output is annotated video/images saved to `output_images/` with bounding boxes, labels, and confidence scores.

---

## Tech Stack

| Component | Tool |
|---|---|
| Detection Model | YOLOv11 Nano |
| Framework | PyTorch + Ultralytics |
| Computer Vision | OpenCV |
| Language | Python 3.10 |
| GPU (optional) | CUDA 11.8+ with cuDNN |

---

## Project Structure

```
drowning_dataset/
├── dataset/            # YOLO format images & labels (train/val/test splits)
├── models/             # best.pt — trained model weights
├── src/
│   ├── image_inference.py
│   ├── temporal.py     # Frame buffer and state machine logic
│   └── system.py
├── output_images/      # Annotated results saved here after inference
├── runs/               # Training logs and metrics from Ultralytics
├── train.py            # Script to retrain the model
├── main.py             # Entry point — run this
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/drowning-detection.git
cd drowning-detection
```

### 2. Make sure you have Python 3.10

Check your version:
```bash
python --version
```

If it shows anything other than 3.10.x, install Python 3.10 from [python.org](https://www.python.org/downloads/) before continuing.

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` in your terminal prompt. All installs go here — do not skip this step.

### 4. Install dependencies

```bash
pip install ultralytics lap opencv-python numpy pillow
```

If you have a CUDA-compatible GPU and want to use it:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only (slower but works fine for testing):
```bash
pip install torch torchvision
```

### 5. Add your model weights

Place your trained `best.pt` file inside the `models/` folder. If you don't have one, you can train from scratch — see the Training section below.

---

## Running the Project

### Run inference on a video

```bash
python main.py
```

By default, `main.py` looks for a video file at the path set inside the script. Open `main.py` and update this line to point to your video:

```python
VIDEO_PATH = "path/to/your/video.mp4"
```

Then run again:
```bash
python main.py
```

Annotated output frames are saved to `output_images/`.

### Run inference on an image

```bash
python src/image_inference.py --source path/to/image.jpg
```

---

## Training the Model

If you want to retrain on your own dataset:

```bash
python train.py
```

Make sure your dataset is in YOLO format inside the `dataset/` folder with `train/`, `val/`, and `test/` splits. Update the dataset path inside `train.py` if needed.

Training logs and metrics (loss curves, mAP) are saved to `runs/`.

---

## How the Detection Works

```
Input Video
    ↓
YOLOv11 — detects people frame by frame
    ↓
Semantic Interpretation — labels each detection as normal or drowning
    ↓
Temporal Buffer — holds the last N frames of each tracked person
    ↓
State Machine — only flags danger if distress persists across frames
    ↓
Alert / Annotated Output
```

The temporal layer is the important part. A single frame of someone face-down in the water doesn't mean they're drowning — they could be doing a flip turn. The state machine only escalates if the distress signal holds over time.

---

## Common Errors

**`ModuleNotFoundError: No module named 'lap'`**
You installed `lap` in the wrong Python environment. Make sure your venv is activated, then run:
```bash
pip install lap
```

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python
```

**Model not found error**
Check that `best.pt` exists inside `models/`. The script expects it there.

**CUDA errors**
If you don't have a GPU, make sure you installed the CPU version of PyTorch. The model runs on CPU — it'll just be slower.

---

## Disclaimer

This is a research project built for academic purposes. It has not been tested in real pool environments and is not suitable for actual life-safety use. False positives and false negatives will occur. Do not deploy this as a substitute for trained lifeguards or certified safety equipment.

---

## Author

Mohamed Taqiyy Faiz
B.Tech — Computer Science & Engineering (Data Science & AI)
Dr. MGR Educational and Research Institute, Chennai — 2026
