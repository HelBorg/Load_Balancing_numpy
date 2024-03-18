from typing import List

import numpy as np

from lvp.models.task import Task
from lvp.models.task_pool import TaskPool
from lvp.tools import upload_pickle, save_pickle

DEFAULT_PATH_SAVE = "/cache/alg_params/"
DEFAULT_PROD_FILE = DEFAULT_PATH_SAVE + "productivity_{id}.pkl"

PRODUC_DISTR = 5


class Agent:
    def __init__(
            self,
            id: int,
            produc: float,
            num_steps: int = None,
            generate: bool = True,
            task_pool: TaskPool = None,
            prods_file_raw=DEFAULT_PROD_FILE
    ):
        self.id = id

        # Tasks
        self.all_tasks = task_pool.tasks.get(self.id, [])
        self.all_tasks.sort(key=lambda x: x.step)

        # Productivity
        produc_file = prods_file_raw.format(id=self.id)
        self.avg_produc = produc
        self.prods = self.generate_or_upload(self.generate_productivities, generate, produc_file, num_steps=num_steps)

        self.tasks = []
        self.theta_hat = len(self.tasks)
        self.task_on_comp = Task()

        self.neighb = []

    @staticmethod
    def generate_or_upload(function, generate: bool, file: str, **kwargs) -> list:
        return function(file, **kwargs) if generate else upload_pickle(file)

    def generate_productivities(self, file: str, num_steps: int):
        """
        Generate productivities for each step of the experiment

        Parameters
        ----------
        file: to save to after generation
        num_steps: number of steps to generate
        """
        producs = np.random.normal(self.avg_produc, PRODUC_DISTR, size=num_steps)
        save_pickle(producs, file)
        return producs

    def update_with_new_tasks(self, step: int) -> None:
        """
        Append new tasks that appear at step
        """
        new_tasks = self.get_new_tasks(step)
        self.tasks.extend(new_tasks)
        self.theta_hat += len(new_tasks)

    def get_new_tasks(self, step: int) -> List[Task]:
        """
        Return tasks that appear on step (self.all_tasks should be sorted by step)

        Parameters
        step: the step at which appear new tasks
        """
        res = []
        ind = 0
        while ind < len(self.all_tasks) and self.all_tasks[ind].step <= step:
            if self.all_tasks[ind].step == step:
                res.append(self.all_tasks[ind])
            ind += 1
        return res

    def complete_tasks(self, step) -> None:
        """
        Complete tasks with taking into account productivity
        """
        to_complete = self.produc

        # Complete task that wasn't done fully before
        task_compl = self.task_on_comp.compl
        if to_complete > task_compl:
            to_complete -= task_compl
            self.task_on_comp.completed_step = step
            self.task_on_comp = self.tasks.pop(0) if self.tasks else Task()
        else:
            self.task_on_comp.compl -= to_complete
            self.update_theta_hat()
            return

        # Complete other tasks
        while self.task_on_comp.compl and to_complete - self.task_on_comp.compl > 0:
            to_complete -= self.task_on_comp.compl
            self.task_on_comp.completed_step = step

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
        self.tasks, res = self.tasks[:-number], self.tasks[-number:]
        return res

    def receive_tasks(self, tasks: List[Task]) -> None:
        """
        Add received tasks to the queue
        :param tasks: list of tasks
        """
        self.tasks.extend(tasks)
        self.tasks = sorted(self.tasks, key=lambda x: x.step)

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
