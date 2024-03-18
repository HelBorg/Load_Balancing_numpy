import logging
import random
from typing import Dict, List

import numpy as np

from lvp.algorithms.abstract_lb import LbAlgorithm
from lvp.jobs.distribute_parallel import DistributeParallel
from lvp.models.parameters import Parameters
from lvp.models.task import Task
from lvp.models.task_pool import TaskPool, POISSON
from lvp.tools import save_pickle, upload_pickle

DEFAULT_CACHE_PATH = "/cache/"
DEFAULT_LOGGS_PATH = DEFAULT_CACHE_PATH + "logs/"
DEFAULT_NEIGH_FILE = DEFAULT_CACHE_PATH + "alg_params/neigh.pkl"
DEFAULT_NOISE_FILE = DEFAULT_CACHE_PATH + "alg_params/noise.pkl"

NOISE_AVG = 0
NOISE_DISTR = 10


class LocalVotingProtocol(LbAlgorithm):
    def __init__(self, params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        super().__init__(params, **kwargs)

        # Topology
        self.adj_mat = params.b
        self.neib_add = params.neib_add  # number of added edges at each step
        self.add_neib_val = params.add_neib_val  # value for added edges

        # Algorithm parameter
        self.h = params.params_dict.get("h", None)

        # Technical variables
        self.distr_parallel = DistributeParallel()

    # --------------------------- Prepare before run ---------------------------------
    def create_task_pool(self, num_steps: int, generate: bool):
        return TaskPool(self.n, num_steps, generate, distribute=generate, distribution=POISSON)

    def prepare(self, generate, neigh_file, num_steps):
        self.generate_new_neighbours(num_steps, generate, neigh_file)
        self.generate_noise(num_steps, generate)

    def generate_new_neighbours(self, num_steps: int, generate_neigh: bool, neigh_file: str) -> None:
        """
        Generate neighbours for each step at start
        For each step choice randomly int(self.neib_add) edges and add them to matrices
        :param num_steps: number of steps
        :param generate_neigh: either to generate or to read from file
        :param neigh_file: file to read from or to save to
        """
        zeros = [(i, j) for i, j in np.argwhere(self.adj_mat == 0) if i < j]
        if generate_neigh:
            add_zeros = [[tuple(random.choice(zeros)) for _ in range(self.neib_add)] for _ in range(num_steps)]

            b_value = lambda i, j, step: \
                self.add_neib_val if i != j and ((i, j) in add_zeros[step] or (j, i) in add_zeros[step]) else 0
            self.adj_mat_by_step = \
                {
                    step: self.adj_mat + [
                        [b_value(i, j, step) for i in range(self.n)] for j in range(self.n)
                    ] for step in range(num_steps)
                }
            save_pickle(self.adj_mat_by_step, neigh_file)
        else:
            self.adj_mat_by_step = upload_pickle(neigh_file)

    def generate_noise(self, num_steps: int, generate: bool, noise_file: str = DEFAULT_NOISE_FILE):
        if generate:
            non_zeros = np.array([[i, j] for i, j in np.argwhere(self.adj_mat != 0)])
            values_num = len(non_zeros)

            noises_mat = np.zeros((num_steps, self.n, self.n), int)
            for step in range(num_steps):
                noises = np.random.normal(NOISE_AVG, NOISE_DISTR, size=values_num)
                noises_mat[[step] * values_num, non_zeros[:, 0], non_zeros[:, 1]] = noises

            save_pickle(noises_mat, noise_file)
        else:
            noises_mat = upload_pickle(noise_file)
        self.noise_mat = noises_mat

    # --------------------------- Extract step info ----------------------------------

    def extract_step_info(self, step: int) -> None:
        """
        Extract information about neighbours and productivity according to the step
        :param step:
        """
        self.b = self.adj_mat_by_step[step]
        self.D = np.diagflat(self.b.sum(axis=1))
        for agent in self.agents:
            agent.neighb = np.where(self.b[agent.id] > 0)[-1]

            agent.produc = agent.prods[step]

    # --------------------------- Algorithm step -------------------------------------

    def algorithm_step(self, is_logging, step):
        # Compute local voting protocol
        self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
        self.log(is_logging, f"Current distribution on {step} step: {self.theta_hat}")

        # Compute redistribution protocol
        u_lvp = self.compute_change(step)
        self.log(is_logging, f"Local voting protocol redistribution: {u_lvp}")
        requests_dic = {ind: u for ind, u in enumerate(u_lvp) if u < 0}

        # Distribute tasks
        response = self.distribute_tasks(requests_dic, u_lvp, step)

        # Receive tasks
        self.receive_tasks(response)

        self.log(is_logging, f"Local voting protocol redistributed:")
        for agent in self.agents:
            agent.update_theta_hat()
            self.log(is_logging, f"Agent {agent.id}: {agent.theta_hat}")

    def compute_change(self, step):
        u_lvp = self.local_voting_protocol(self.theta_hat, step)
        u_lvp = (self.h * u_lvp).round()
        return u_lvp

    def local_voting_protocol(self, x, step) -> np.array:
        """
        Compute local voting protocol
        :param x: agents queue lengths
        :return: lvp result
        """
        lvp = [[self.lvp_step((x.T + self.noise_mat[step, agent_id]).T, agent_id)] for agent_id in range(self.n)]
        return np.matrix(lvp)

    def lvp_step(self, x, agent_id):
        return ((self.D[agent_id] - self.b[agent_id]) * x).item(0)

    def distribute_tasks(self, requests_dic: Dict[int, int], u_lvp: np.array, step: int) -> Dict[int, List[Task]]:
        """
        For each agent that want to send tasks find agent to send to
        :param requests_dic: {agent id: number of tasks to send}
        :param u_lvp: local voting protocol result
        :return: {agent id: tasks to send}
        """
        resp_age = [a for a in self.agents if u_lvp[a.id] > 0]
        params_list = []
        for agent in resp_age:
            params_list.append((agent, requests_dic, u_lvp[agent.id], step, self.loggs_path,))

        response = self.distr_parallel.run(args_list=params_list, shared_vars=requests_dic)

        if not response:
            return {}

        send_tasks = {}
        keys = requests_dic.keys()
        response_inv = {
            send_to: {
                send_from: response[send_from][send_to]
                for send_from in response if send_to in response[send_from]
            } for send_to in keys
        }

        for send_to_id, res in response_inv.items():
            if not res:
                continue

            sum_to_send = sum(res.values())
            mx_num, add = -1, -1
            if sum_to_send < abs(requests_dic[send_to_id]):
                mx_num = max(res, key=res.get)
                add = abs(requests_dic[send_to_id]) - sum_to_send

            for agent_id, num in res.items():
                if mx_num == agent_id:
                    num = num + add
                    logging.info(f"Agent {send_to_id} need {add} more getting from {mx_num}")
                    mx_num, add = -1, -1

                if num == 0:
                    continue

                tasks = self.agents[agent_id].tasks_to_send(int(num))
                send_tasks.setdefault(send_to_id, []).extend(tasks)

        return send_tasks

    def receive_tasks(self, response: Dict[int, List[Task]]) -> None:
        """
        Distribute received tasks
        :param response: tasks that was sent
        """
        for agent_id, tasks in response.items():
            self.agents[agent_id].receive_tasks(tasks)


if __name__ == "__main__":
    generate = True
    num_agents = 5
    productivities = [10] * num_agents
    num_steps = 20

    pars = Parameters()
    pars.n = num_agents
    Adj = np.array([
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0]
    ])
    pars.b = Adj / 2

    pars.neib_add = 5
    pars.add_neib_val = 0.3
    pars.params_dict = {
        "h": 0.2,
    }

    alg_lvp = LocalVotingProtocol(params=pars)
    alg_lvp.run(num_steps=num_steps, generate=generate, productivities=productivities)
    lvp_seq = alg_lvp.sequence
    logging.info("lvp_seq")
