from typing import List

import numpy as np

from lvp.services import upload_pickle, save_pickle

DEFAULT_PATH_SAVE = "cache/"
DEFAULT_TASKS_FILE = DEFAULT_PATH_SAVE + "tasks_{id}"

STEPS_LAM = 2
COMPL_MEAN = 10
COMPL_DISTR = 3
SIZE_COEF = 10
SIZE_BIAS = 20


class Parameters:
    n: int
    b: np.matrix
    product: np.matrix
    queue_gen: np.matrix
    theta_hat: np.matrix = np.matrix([[0], [0], [0]])

    params_dict: dict


class Task:
    def __init__(self, step=None, compl=0):
        self.step = step
        self.compl = compl

    def to_dict(self):
        return {
            "complexity": self.compl,
            "step": self.step
        }


class Agent:
    def __init__(self, id, produc, generate_tasks=True, tasks_file_raw=DEFAULT_TASKS_FILE):
        self.id = id
        self.produc = produc
        tasks_file = tasks_file_raw.format(id=self.id)
        self.all_tasks = self.get_all_tasks(generate_tasks, tasks_file)
        self.all_tasks.sort(key=lambda x: x.step)

        self.tasks = self.get_new_tasks(0)
        self.theta_hat = len(self.tasks)
        self.task_on_comp = self.tasks.pop(0) if len(self.tasks) > 0 else Task()

        self.neighb = []

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
            return

        # Complete other tasks
        while self.task_on_comp.compl and to_complete - self.task_on_comp.compl > 0:
            to_complete -= self.task_on_comp.compl
            self.task_on_comp = self.tasks.pop(0) if len(self.tasks) > 0 else Task()

        # Remember the task that wasn't done fully (maybe no task in the queue)
        self.task_on_comp.compl = max(0, self.task_on_comp.compl - to_complete)
        self.update_theta_hat()

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

    def get_all_tasks(self, generate_tasks: bool, tasks_file: str) -> list:
        """
        Get all tasks at initialization
        :param generate_tasks: generate tasks or get from file
        :param tasks_file: file to save to or to upload from
        :return:
        """
        if generate_tasks:
            return self.generate_tasks(tasks_file)
        else:
            return upload_pickle(tasks_file)

    def get_tasks(self, number: int) -> List[Task]:
        """
        Extract tasks to send
        :param number: number of tasks to return
        :return: list of tasks
        """
        res, self.tasks = self.tasks[:number], self.tasks[number:]
        return res

    def generate_tasks(self, tasks_file: str) -> list:
        """
        Generate all_tasks
        :param tasks_file: path to the file to save tasks to
        :return: list of entities Task
        """
        size = np.random.poisson(lam=self.id * SIZE_COEF + SIZE_BIAS)
        steps = np.random.poisson(lam=STEPS_LAM, size=size)
        compl = np.random.normal(COMPL_MEAN, COMPL_DISTR, size=size)
        tasks = [Task(step, comp) for step, comp in zip(steps, compl)]
        save_pickle(tasks, tasks_file)
        return tasks

    def get_new_tasks(self, step: int) -> List[Task]:
        """
        Return tasks that appear on step (self.all_tasks should be sorted by step)
        :param step: the step at which appear new tasks
        :return: list of tasks that should appear on step step
        """
        res = []
        while self.all_tasks and self.all_tasks[0].step == step:
            res.append(self.all_tasks.pop(0))

        return res

    def receive_tasks(self, tasks: List[Task]) -> None:
        """
        Add received tasks to the queue
        :param tasks: list of tasks
        """
        self.tasks.extend(tasks)

    def update_with_new_tasks(self, step: int) -> None:
        """
        Append new tasks that appear at step
        :param step:
        :return:
        """
        new_tasks = self.get_new_tasks(step)
        self.tasks.extend(new_tasks)
        self.theta_hat += len(new_tasks)
