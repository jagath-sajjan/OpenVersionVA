import sys
import time

import cv2

from core.camera import Camera
from core.renderer import Renderer
from detectors.yolo_detector import YOLODetector
from trackers.tracker_manager import TrackerManager

WINDOW_NAME = "OpenVisionVA — Phase 2"
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.4
CAMERA_SOURCE = 2


def main():
    print("[OpenVisionVA] Starting...")

    camera = Camera(source=CAMERA_SOURCE, width=1280, height=720)
    renderer = Renderer()
    detector = YOLODetector(model_path=MODEL_PATH, confidence=CONFIDENCE, device="cpu")
    tracker = TrackerManager(max_disappeared=30, max_distance=80.0, trail_length=40)

    detector.load()

    if not camera.open():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("[OpenVisionVA] Running. Press Q or ESC to quit.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Empty frame. Retrying...")
                continue

            detections = detector.detect(frame)

            tracked = tracker.update(detections)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            frame = renderer.draw_trails(frame, tracked)
            frame = renderer.draw_tracked(frame, tracked)
            frame = renderer.draw_fps(frame, fps)
            frame = renderer.draw_hud(frame, len(tracked))

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[OpenVisionVA] Shutting down.")
                break

    except KeyboardInterrupt:
        print("[OpenVisionVA] Interrupted.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[OpenVisionVA] Cleanup complete.")


if __name__ == "__main__":
    main()
