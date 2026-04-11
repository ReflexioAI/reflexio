class StorageError(Exception):
    """
    Exception raised for storage errors
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"StorageError: {self.message}"
