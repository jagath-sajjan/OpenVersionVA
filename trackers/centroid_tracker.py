from collections import OrderedDict

import numpy as np

from trackers.base_tracker import BaseTracker


class CentroidTracker(BaseTracker):
    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.max_disappered = max_disappeared
        self.max_distance = max_distance

        self.next_id = 0
        self.objects = OrderedDict()
        self.disappered = OrderedDict()

    def _register(self, centroid: tuple, det: dict):
        obj_id = self.next_id
        self.objects[obj_id] = {
            "centroid": centroid,
            "bbox": det["bbox"],
            "label": det["label"],
            "confidence": det["confidence"],
            "class_id": det.get("class_id", -1),
        }
        self.disappered[obj_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappered[obj_id]

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detections: list) -> dict:
        if len(detections) == 0:
            for obj_id in list(self.disappered.keys()):
                self.disappered[obj_id] += 1
                if self.disappered[obj_id] > self.max_disappered:
                    self._deregister(obj_id)
            return dict(self.objects)

        input_centroids = []
        for det in detections:
            input_centroids.append(self._centroid(det["bbox"]))

        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(input_centroids[i], det)
            return dict(self.objects)

        existing_ids = list(self.objects.keys())
        existing_centroids = np.array(
            [self.objects[oid]["centroid"] for oid in existing_ids], dtype=float
        )
        input_arr = np.array(input_centroids, dtype=float)

        D = np.linalg.norm(
            existing_centroids[:, np.newaxis] - input_arr[np.newaxis, :], axis=2
        )

        row_order = D.min(axis=1).argsort()
        col_order = D.argmin(axis=1)[row_order]

        matched_rows = set()
        matched_cols = set()

        for row, col in zip(row_order, col_order):
            if row in matched_rows or col in matched_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = existing_ids[row]
            det = detections[col]
            self.objects[obj_id].update(
                {
                    "centroid": input_centroids[col],
                    "bbox": det["bbox"],
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "class_id": det.get("class_id", -1),
                }
            )
            self.disappered[obj_id] = 0

            matched_rows.add(row)
            matched_cols.add(col)

        unmatched_rows = set(range(len(existing_ids))) - matched_rows
        for row in unmatched_rows:
            obj_id = existing_ids[row]
            self.disappered[obj_id] += 1
            if self.disappered[obj_id] > self.max_disappered:
                self._deregister(obj_id)

        unmatched_cols = set(range(len(input_centroids))) - matched_cols
        for col in unmatched_cols:
            self._register(input_centroids[col], detections[col])

        return dict(self.objects)

    def reset(self):
        self.objects.clear()
        self.disappered.clear()
        self.next_id = 0
