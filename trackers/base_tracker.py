from abc import ABC, abstractmethod

import numpy as np


class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: list) -> dict:
        pass

    @abstractmethod
    def reset(self):
        pass
