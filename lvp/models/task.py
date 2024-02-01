class Task:
    def __init__(self, step=None, compl=0):
        self.step = step
        self.compl = compl

    def to_dict(self):
        return {
            "complexity": self.compl,
            "step": self.step
        }