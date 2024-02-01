import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from lvp.jobs.AlvpParallel import AlvpParallel
from lvp.jobs.DistributeParallel import DistributeParallel
from lvp.models.Agent import Agent
from lvp.models.Parameters import Parameters
from lvp.models.Task import Task
from lvp.tools import save_pickle, upload_pickle

DEFAULT_CACHE_PATH = "cache/"
DEFAULT_LOGGS_PATH = DEFAULT_CACHE_PATH + "logs/"
DEFAULT_NEIGH_FILE = DEFAULT_CACHE_PATH + "alg_params/neigh.pkl"
DEFAULT_NOISE_FILE = DEFAULT_CACHE_PATH + "alg_params/noise.pkl"

NOISE_AVG = 0
NOISE_DISTR = 10


class LbAlgorithm:
    def __init__(self, agents: List[Agent], params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        # Topology
        self.n = params.n
        self.agents = agents
        self.adj_mat = params.b
        self.neib_add = params.neib_add  # number of added edges at each step
        self.add_neib_val = params.add_neib_val  # value for added edges

        # Algorithm parameter
        self.h = params.params_dict.get("h", None)

        # Accelerated local protocol variables
        self.nesterov_step = np.array([[0]] * self.n)
        self.L = params.params_dict.get("L", None)
        self.mu = params.params_dict.get("mu", None)
        self.eta = params.params_dict.get("eta", None)
        self.gamma = params.params_dict.get("gamma", [])
        self.alpha = params.params_dict.get("alpha", None)

        # Technical variables
        self.distr_parallel = DistributeParallel()
        self.alvp_parallel = AlvpParallel(self.alpha, self.mu, self.eta, self.h, self.L)

        # Resuts
        self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
        self.sequence = []
        self.sequence_2 = []

        # Set up logging
        loggs_id = max([int(i) for i in os.listdir(DEFAULT_LOGGS_PATH) if i.isdigit()] + [-1])
        self.loggs_path = DEFAULT_LOGGS_PATH + f"{int(loggs_id) + 1}/"
        os.mkdir(self.loggs_path)
        logging.basicConfig(filename=f'{self.loggs_path}_loggs_lvp.log', filemode='w', level=logging.INFO, force=True)

    def run(
            self,
            num_steps: int = 100,
            eps: int = 0.1,
            accelerate: bool = False,
            generate: bool = True,
            neigh_file: str = DEFAULT_NEIGH_FILE
    ):
        self.generate_new_neighbours(num_steps, generate, neigh_file)
        self.generate_noise(num_steps, generate)

        for step in range(num_steps):
            logging.info(f"\n\nStep: {step}")
            for agent in self.agents:
                df = [task.to_dict() for task in agent.tasks]
                logging.info(pd.DataFrame(df).sum())

            # Add some neighbour edges
            self.extract_step_info(step)

            # Get new tasks
            [agent.update_with_new_tasks(step) for agent in self.agents]

            self.sequence.append([agent.get_real_queue_length() for agent in self.agents])
            self.sequence_2.append([agent.theta_hat for agent in self.agents])

            # Complete some tasks
            [agent.complete_tasks() for agent in self.agents]

            # Compute local voting protocol
            self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
            logging.info(f"Current distribution on {step} step: {self.theta_hat}")

            if not accelerate:
                u_lvp = self.local_voting_protocol(self.theta_hat, step)
                u_lvp = (self.h * u_lvp).round()
            else:
                u_lvp = self.acc_local_voting_protocol(self.theta_hat, step)
                u_lvp = u_lvp.round()
            logging.info(f"Local voting protocol redistribution: {u_lvp}")

            requests_dic = {ind: u for ind, u in enumerate(u_lvp) if u < 0}

            # Distribute tasks
            response = self.distribute_tasks(requests_dic, u_lvp, step)

            # Receive tasks
            self.receive_tasks(response)

            logging.info(f"Local voting protocol redistributed:")
            for agent in self.agents:
                agent.update_theta_hat()
                logging.info(f"Agent {agent.id}: {agent.theta_hat}")

            print(f"Step {step} is completed")

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

    def acc_local_voting_protocol(self, x: np.array, step) -> np.array:
        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))

        params_list = []
        for agent_id in range(self.n):
            params_list.append(
                (
                    agent_id,
                    (x.T + self.noise_mat[step, agent_id]).T,
                    self.nesterov_step.item((agent_id, 0)),
                    self.gamma,
                    self.D[agent_id],
                    self.b[agent_id],
                    self.loggs_path,)
            )
        response = self.alvp_parallel.run(args_list=params_list)

        self.nesterov_step = AlvpParallel.extract_response_to_array(response, "nesterov_step", range(self.n))

        return AlvpParallel.extract_response_to_array(response, "x", range(self.n))

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


def is_neib(i, j):
    if i in [0, num_agents - 1] and i + j == num_agents - 1:
        return 1
    return 1 if abs(i - j) == 1 else 0


if __name__ == "__main__":
    generate = True
    num_agents = 20
    productivity = 10
    num_steps = 20
    agents = [Agent(id, productivity, generate=generate, num_steps=num_steps) for id in range(num_agents)]

    pars = Parameters()
    pars.n = num_agents
    pars.theta_hat = np.matrix([len(agent.tasks) for agent in agents]).transpose()
    Adj = np.matrix([[is_neib(i, j) for i in range(num_agents)] for j in range(num_agents)])
    pars.b = Adj / 2

    pars.neib_add = 5
    pars.add_neib_val = 0.3
    pars.params_dict = {
        "L": 7.1,
        "mu": 0.9,
        "h": 0.2,
        "eta": 0.8,
        "gamma": [[0.07, 0.09, 0.11][0]],
        "alpha": [0.07, 0.09, 0.11][1]
    }

    alg_lvp = LbAlgorithm(agents=agents, params=pars)
    alg_lvp.run(num_steps=20, accelerate=False, generate=generate)
    lvp_seq = alg_lvp.sequence
    logging.info("lvp_seq")

    # alg_lvp.theta_hat = np.matrix([[44], [715], [912], [1304], [2442]])
    # alg_lvp.nesterov_step = np.array([[0], [0], [0], [0], [0]])
    # alg_lvp.b = np.matrix([
    #     [0, 0, 1, 1, 0],
    #     [0, 0, 0, 1, 1],
    #     [1, 0, 0, 0, 1],
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 0]
    # ]) / 2
    # alg_lvp.D = np.diagflat(alg_lvp.b.sum(axis=1))
    # alg_lvp.acc_local_voting_protocol(alg_lvp.theta_hat)
    # print(alg_lvp.gamma)
    #
    # alg_lvp.theta_hat = np.matrix([[80], [586], [1195], [1608], [2105]])
    # alg_lvp.local_voting_protocol(alg_lvp.theta_hat)
