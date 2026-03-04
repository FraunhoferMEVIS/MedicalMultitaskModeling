class NoUsefulWindowException(Exception):
    def __init__(self, message: str, anno):
        self.anno = anno
        super().__init__(message)
