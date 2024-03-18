import numpy as np

from lvp.models.task import Task
from lvp.tools import upload_pickle, save_pickle

DEFAULT_PATH_SAVE = "\\cache\\alg_params\\"
DEFAULT_DISTR_TASKS_FILE = DEFAULT_PATH_SAVE + "tasks_distribution_{distribution}.pkl"
DEFAULT_ALL_TASKS_FILE = DEFAULT_PATH_SAVE + "tasks.pkl"

STEPS_LAM = 2
COMPL_MEAN = 5
COMPL_DISTR = 5
SIZE_COEF = 100
SIZE_BIAS = 100

UNIFORM = "UNIFORM"
POISSON = "POISSON"
DISTR_METHODS = {
    UNIFORM: "distribute_uniform",
    POISSON: "distribute_poisson"
}


class TaskPool:
    """
    Initial task generation and distribution between agents
    """

    def __init__(
            self,
            num_agents: int,
            num_steps: int,
            generate: bool = True,
            distribute: bool = True,
            distribution: str = POISSON,
            all_tasks_file: str = DEFAULT_ALL_TASKS_FILE,
            tasks_distr_raw: str = DEFAULT_DISTR_TASKS_FILE

    ):
        self.tasks = {agent_id: [] for agent_id in range(num_agents)}

        if generate:
            all_tasks = self.generate_tasks(num_agents, num_steps)
            save_pickle(all_tasks, all_tasks_file)
        else:
            all_tasks = upload_pickle(all_tasks_file)

        # get distribution
        tasks_distr_file = tasks_distr_raw.format(distribution=distribution)
        if distribute:
            tasks = self.distribute_initial(all_tasks, num_agents)
            getattr(self, DISTR_METHODS.get(distribution))(tasks, all_tasks, num_agents)
            save_pickle(self.tasks, tasks_distr_file)

        else:
            self.tasks = upload_pickle(tasks_distr_file)

    def generate_tasks(self, num_agents, num_steps) -> dict:
        """
        Generate random number of tasks
        Returns dictionary of tasks by step
        """
        size = np.random.poisson(lam=SIZE_COEF * num_agents * 5 / 2 + num_agents * SIZE_BIAS)  # todo: change
        steps = np.random.randint(num_steps, size=size // 20)
        tasks = {step: [] for step in range(num_steps)}
        for step in steps:
            tasks[step].append(Task(step, abs(np.random.normal(COMPL_MEAN, COMPL_DISTR))))

        # Create initial tasks
        compl = np.random.uniform(COMPL_MEAN - COMPL_DISTR, COMPL_MEAN + COMPL_DISTR, size=size)
        add = [Task(0, comp) for comp in compl]
        tasks[0].extend(add)
        return tasks

    def distribute_initial(self, all_tasks, num_agents):
        initial_tasks = all_tasks[0]
        del all_tasks[0]
        return {agent_id: self.assign_tasks_to_agent(agent_id, initial_tasks) for agent_id in range(num_agents)}

    def assign_tasks_to_agent(self, agent_id, initial_tasks, assigned_tasks=None):
        if assigned_tasks is None:
            assigned_tasks = dict()

        size = np.random.poisson(lam=SIZE_COEF * agent_id + SIZE_BIAS)
        inds = np.random.randint(len(initial_tasks), size=min(size, len(initial_tasks)))
        return [initial_tasks.pop(min(ind, len(initial_tasks) - 1)) for ind in inds] + assigned_tasks.get(agent_id, [])

    def distribute_poisson(self, tasks, all_tasks, num_agents):
        tasks_list = [task for subtasks in all_tasks.values() for task in subtasks]
        self.tasks = {agent_id: self.assign_tasks_to_agent(agent_id, tasks_list, assigned_tasks=tasks) for agent_id in
                       range(num_agents)}

    def distribute_uniform(self, tasks, all_tasks, num_agents):
        """
        Distribute tasks between agents evenly
        Each step start with the one on which ended previously

        Parameters
        ----------
        all_tasks: all tasks that was generated
        num_agents: number of agents to distribute between

        Returns
        -------

        """
        start_agent = 0
        for step, step_tasks in all_tasks.items():
            num_add = len(step_tasks) // num_agents
            residual = len(step_tasks) % num_agents

            for agent_id in range(num_agents):
                agent_residual = int(self.add_residual(agent_id, num_agents, residual, start_agent))
                tasks[agent_id].extend(step_tasks[:num_add + agent_residual + 1])
                step_tasks = step_tasks[num_add + agent_residual + 1:]

            start_agent = (start_agent + residual) % num_agents
        self.tasks = tasks

    def add_residual(self, agent_id: int, num_agents: int, residual: int, start_point: int) -> bool:
        """
        Need to distribute residual. Is this agent will get extra task?
        Parameters
        ----------
        agent_id: agent that is being investigated
        num_agents: number of agents
        residual: number of agents to distribute between
        start_point: distribute residual starting with this agent_id

        Returns do we need to add extra task to this agent?
        -------

        """
        end_point = (start_point + residual) % num_agents
        return (start_point <= agent_id < end_point
                or agent_id >= start_point > end_point
                or start_point > end_point > agent_id)

    def get_tasks_by_id(self, agent_id):
        return self.tasks.get(agent_id, [])
