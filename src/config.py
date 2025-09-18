from src.data.log_standardizer import LogStandardizer

class Config:
    def __init__(self):
        self.default = object()
        self.scaler = LogStandardizer
        self.time_slices = 50000

cfg = Config()
