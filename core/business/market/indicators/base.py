from abc import ABC, abstractmethod

class MarketInfoBase(ABC):
    def __init__(self, crypto_queryset):
        self.crypto_queryset = crypto_queryset

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def get_flag(self):
        pass

    @abstractmethod
    def get_label(self):
        pass

    def get_info(self):
        return {
            "indicator": self.get_label(),
            "indicatorValue": self.compute(),
            "message": self.get_message(),
            "colorFlag": self.get_flag(),
        }

    def get_message(self):
        return f"Information for {self.get_label()} non disponible."
