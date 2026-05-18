import numpy as np
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None

    def load(self):
        self.model = YOLO(self.model_path)
        print(f"[YOLODetector] Model loaded: {self.model_path} on {self.device}")

    def detect(self, frame: np.ndarray) -> list:
        if self.model is None:
            raise RuntimeError("model not loaded call load() first")

        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "label": label,
                        "confidence": conf,
                        "class_id": cls_id,
                    }
                )

        return detections
