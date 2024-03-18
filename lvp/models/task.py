class Task:
    def __init__(self, step=None, compl=0):
        self.step = step
        self.compl = compl
        self.initial_compl = compl
        self.completed_step = None

    def to_dict(self):
        return {
            "complexity": self.compl,
            "step": self.step,
            "completed_step": self.completed_step,
            "initial_compl": self.initial_compl
        }
