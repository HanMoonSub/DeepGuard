class PredictorError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        super().__init__(self.message)
