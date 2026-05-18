from collections import defaultdict, deque

from trackers.centroid_tracker import CentroidTracker

TRAIL_LENGTH = 40
TRAIL_FADE_STEPS = 10


class TrackerManager:
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 80.0,
        trail_length: int = TRAIL_LENGTH,
    ):
        self.tracker = CentroidTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
        )
        self.trail_length = trail_length
        self.trails: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )

    def update(self, detections: list) -> dict:
        tracked = self.tracker.update(detections)

        active_ids = set(tracked.keys())
        stale_ids = set(self.trails.keys()) - active_ids
        for sid in stale_ids:
            del self.trails[sid]

        for obj_id, info in tracked.items():
            self.trails[obj_id].append(info["centroid"])
            info["trail"] = list(self.trails[obj_id])

        return tracked

    def reset(self):
        self.tracker.reset()
        self.trails.clear()
