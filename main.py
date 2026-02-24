import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# ===============================
# CONFIG
# ===============================

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "videos/drowning1.mp4"

WINDOW_SIZE = 30        # frames

HIGH_RISK_FRAMES = 15   # threshold

# ===============================
# LOAD MODEL
# ===============================

model = YOLO(MODEL_PATH)

# ===============================
# MEMORY FOR TRACKING
# ===============================

memory = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

# ===============================
# DISTANCE FUNCTION
# ===============================

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ===============================
# RISK ANALYSIS
# ===============================

def analyze_behavior(track):
    if len(track) < 5:
        return "NORMAL_ACTIVITY"

    movement = 0
    for i in range(1, len(track)):
        movement += distance(track[i-1], track[i])

    avg_movement = movement / len(track)

    if avg_movement < 2:
        return "HIGH_RISK_DROWNING"

    if avg_movement < 5:
        return "POSSIBLE_DROWNING"

    return "NORMAL_ACTIVITY"

# ===============================
# VIDEO PIPELINE
# ===============================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

print("✅ Video loaded")

# Built-in YOLO tracking
results = model.track(source=VIDEO_PATH, stream=True, persist=True)

for r in results:

    frame = r.orig_img

    if r.boxes is None:
        continue

    boxes = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id

    if ids is None:
        continue

    ids = ids.cpu().numpy().astype(int)

    for box, pid in zip(boxes, ids):

        x1, y1, x2, y2 = map(int, box)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        memory[pid].append((cx, cy))

        state = analyze_behavior(memory[pid])

        # ===============================
        # COLOR BY STATE
        # ===============================

        color_map = {
            "NORMAL_ACTIVITY": (0, 255, 0),
            "POSSIBLE_DROWNING": (0, 255, 255),
            "HIGH_RISK_DROWNING": (0, 0, 255),
        }

        color = color_map.get(state, (255, 255, 255))

        # ===============================
        # DRAW
        # ===============================

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            frame,
            f"ID {pid}: {state}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Drowning Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()