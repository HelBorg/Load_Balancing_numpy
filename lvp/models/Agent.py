from typing import List

import numpy as np

from lvp.models.Task import Task
from lvp.tools import upload_pickle, save_pickle

DEFAULT_PATH_SAVE = "cache/alg_params/"
DEFAULT_TASKS_FILE = DEFAULT_PATH_SAVE + "tasks_{id}.pkl"
DEFAULT_PROD_FILE = DEFAULT_PATH_SAVE + "productivity_{id}.pkl"

STEPS_LAM = 2
COMPL_MEAN = 5
COMPL_DISTR = 10
SIZE_COEF = 800
SIZE_BIAS = 10
PRODUC_DISTR = 5


class Agent:
    def __init__(
            self,
            id: int,
            produc: float,
            generate: bool = True,
            num_steps: int = None,
            tasks_file_raw=DEFAULT_TASKS_FILE,
            prods_file_raw=DEFAULT_PROD_FILE
    ):
        self.id = id

        # Generate or upload tasks for each step of the experiment
        tasks_file = tasks_file_raw.format(id=self.id)
        self.all_tasks = self.generate_or_upload(generate, tasks_file, num_steps, self.generate_tasks)
        self.all_tasks.sort(key=lambda x: x.step)

        # Generate or upload productivities for each step of the experiment
        produc_file = prods_file_raw.format(id=self.id)
        self.avg_produc = produc
        self.prods = self.generate_or_upload(generate, produc_file, num_steps, self.generate_productivities)

        self.tasks = []
        self.theta_hat = len(self.tasks)
        self.task_on_comp = Task()

        self.neighb = []

    def generate_or_upload(self, generate: bool, file: str, num_steps, function):
        return function(file, num_steps) if generate else upload_pickle(file)

    def generate_tasks(self, tasks_file: str, num_steps: int) -> list:
        """
        Generate all_tasks
        :param tasks_file: path to the file to save tasks to
        :return: list of entities Task
        """
        size = np.random.poisson(lam=self.id * SIZE_COEF + SIZE_BIAS)
        steps = np.random.randint(num_steps, size=size//20)
        tasks = [Task(step, abs(np.random.normal(COMPL_MEAN, COMPL_DISTR))) for step in steps]

        compl = np.random.normal(COMPL_MEAN, COMPL_DISTR, size=size)
        add = [Task(0, comp) for comp in compl]
        tasks.extend(add)
        save_pickle(tasks, tasks_file)
        return tasks

    def generate_productivities(self, file: str, num_steps: int):
        """
        Generate productivities for each step of the experiment
        :param file: file to save to after generation
        :param num_steps: number of steps to generate to
        :return: productivities for each step
        """
        producs = np.random.normal(self.avg_produc, PRODUC_DISTR, size=num_steps)
        save_pickle(producs, file)
        return producs

    def update_with_new_tasks(self, step: int) -> None:
        """
        Append new tasks that appear at step
        :param step:
        :return:
        """
        new_tasks = self.get_new_tasks(step)
        self.tasks.extend(new_tasks)
        self.theta_hat += len(new_tasks)

    def get_new_tasks(self, step: int) -> List[Task]:
        """
        Return tasks that appear on step (self.all_tasks should be sorted by step)
        :param step: the step at which appear new tasks
        :return: list of tasks that should appear on step step
        """
        res = []
        ind = 0
        while ind < len(self.all_tasks) and self.all_tasks[ind].step <= step:
            if self.all_tasks[ind].step == step:
                res.append(self.all_tasks[ind])
            ind += 1
        return res

    def complete_tasks(self) -> None:
        """
        Complete tasks with taking into account productivity
        """
        to_complete = self.produc

        # Complete task that wasn't done fully before
        task_compl = self.task_on_comp.compl
        if to_complete > task_compl:
            to_complete -= task_compl
            self.task_on_comp = self.tasks.pop(0) if self.tasks else Task()
        else:
            self.task_on_comp.compl -= to_complete
            to_complete = 0

        if not to_complete:
            self.update_theta_hat()
            return

        # Complete other tasks
        while self.task_on_comp.compl and to_complete - self.task_on_comp.compl > 0:
            to_complete -= self.task_on_comp.compl
            self.task_on_comp = self.tasks.pop(0) if len(self.tasks) > 0 else Task()

        # Remember the task that wasn't done fully (maybe no task in the queue)
        self.task_on_comp.compl = max(0, self.task_on_comp.compl - to_complete)
        self.update_theta_hat()

    def tasks_to_send(self, number: int) -> List[Task]:
        """
        Extract tasks to send
        :param number: number of tasks to return
        :return: list of tasks
        """
        res, self.tasks = self.tasks[:number], self.tasks[number:]
        return res

    def receive_tasks(self, tasks: List[Task]) -> None:
        """
        Add received tasks to the queue
        :param tasks: list of tasks
        """
        self.tasks.extend(tasks)

    def update_theta_hat(self) -> None:
        """
        Update queue length
        """
        self.theta_hat = len(self.tasks)

    def get_real_queue_length(self) -> float:
        """
        Count queue computing time
        :return:
        """
        return sum([task.compl for task in self.tasks])
