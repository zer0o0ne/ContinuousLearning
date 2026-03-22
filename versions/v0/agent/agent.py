class ASI:
    def __init__(self, log, config = None):
        self.log = log
        self.config = config

    def test(self):
        self.log("Hello")