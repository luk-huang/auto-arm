from abc import ABC, abstractmethod

class MeasurementTool(ABC):
    @abstractmethod
    def measure(self, data):
        pass


