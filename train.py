from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,              # ↓ lower batch for CPU
        optimizer="SGD",
        lr0=0.01,
        patience=15,
        device="cpu",         # ✅ THIS FIXES THE ERROR
        project="runs/drowning",
        name="yolov11_binary",
        verbose=True
    )

if __name__ == "__main__":
    main()
