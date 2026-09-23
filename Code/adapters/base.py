from abc import ABC, abstractmethod

class DatasetAdapter(ABC):

    @abstractmethod
    def normalize_messages(self, row):
        pass

    @abstractmethod
    def is_supported_row(self, row):
        pass