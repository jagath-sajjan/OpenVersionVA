import cv2
import numpy as np


class Renderer:
    BOX_COLOR = (0, 255, 0)
    LABEL_BG_COLOR = (0, 255, 0)
    LABEL_TEXT_COLOR = (0, 0, 0)
    FPS_COLOR = (0, 200, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BOX_THICKNESS = 2
    FONT_SCALE = 0.55
    FONT_THICKNESS = 1

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, self.BOX_THICKNESS)

            # Build label string
            text = f"{label} {confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                text, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w + 4, y1),
                self.LABEL_BG_COLOR,
                -1,
            )

            cv2.putText(
                frame,
                text,
                (x1 + 2, y1 - baseline - 2),
                self.FONT,
                self.FONT_SCALE,
                self.LABEL_TEXT_COLOR,
                self.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (12, 32),
            self.FONT,
            0.9,
            self.FPS_COLOR,
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_hud(self, frame: np.ndarray, count: int) -> np.ndarray:
        h, w = frame.shape[:2]
        text = f"Objects: {count}"
        (text_w, text_h), _ = cv2.getTextSize(text, self.FONT, 0.7, 2)
        cv2.putText(
            frame,
            text,
            (w - text_w - 12, 32),
            self.FONT,
            0.7,
            self.FPS_COLOR,
            2,
            cv2.LINE_AA,
        )
        return frame
