import logging
import random
from typing import Dict, List

import numpy as np
import pandas as pd

from lvp.LvpParallel import LvpParallel
from lvp.models.Agent import Agent
from lvp.models.Parameters import Parameters
from lvp.models.Task import Task
from lvp.tools import save_pickle, upload_pickle
import os

DEFAULT_CACHE_PATH = "cache/"
DEFAULT_LOGGS_PATH = DEFAULT_CACHE_PATH + "logs/"
DEFAULT_NEIGH_FILE = DEFAULT_CACHE_PATH + "neigh.pkl"


class LbAlgorithm:
    def __init__(self, agents: List[Agent], params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        # Topology
        self.n = params.n
        self.agents = agents
        self.adj_mat = params.b
        self.neib_add = params.neib_add
        self.add_neib_val = params.add_neib_val

        # Algorithm parameter
        self.h = params.params_dict.get("h", None)

        # Accelerated local protocol variables
        self.nesterov_step = np.array([[0]] * self.n)
        self.L = params.params_dict.get("L", None)
        self.mu = params.params_dict.get("mu", None)
        self.eta = params.params_dict.get("eta", None)
        self.gamma = params.params_dict.get("gamma", [])
        self.alpha = params.params_dict.get("alpha", None)

        self.parallel = LvpParallel()

        self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
        self.sequence = []
        self.sequence_2 = []

        loggs_id = max(os.listdir(DEFAULT_LOGGS_PATH))
        self.loggs_path = DEFAULT_LOGGS_PATH + f"{int(loggs_id) + 1}/"
        os.mkdir(self.loggs_path)
        logging.basicConfig(filename=f'{self.loggs_path}_loggs_lvp.log', filemode='w', level=logging.INFO, force=True)

    def run(
            self,
            num_steps: int = 100,
            eps: int = 0.1,
            accelerate: bool = False,
            generate_neigh: bool = True,
            neigh_file: str = DEFAULT_NEIGH_FILE
    ):
        self.generate_new_neighbours(num_steps, generate_neigh, neigh_file)
        for step in range(num_steps):
            logging.info(f"\n\nStep: {step}")
            for agent in self.agents:
                df = [task.to_dict() for task in agent.tasks]
                logging.info(pd.DataFrame(df).sum())

            # Add some neighbour edges
            self.extract_step_info(step)

            self.sequence.append([agent.get_real_queue_length() for agent in self.agents])
            self.sequence_2.append([agent.theta_hat for agent in self.agents])

            # Get new tasks
            [agent.update_with_new_tasks(step) for agent in self.agents]

            # Complete some tasks
            [agent.complete_tasks() for agent in self.agents]

            # Compute local voting protocol
            self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
            logging.info(f"Current distribution on {step} step: {self.theta_hat}")

            if not accelerate:
                u_lvp = self.local_voting_protocol(self.theta_hat)
            else:
                u_lvp = self.acc_local_voting_protocol(self.theta_hat)
            u_lvp = (self.h * u_lvp).round()
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

    def extract_step_info(self, step: int) -> None:
        """
        Extract information about neighbours and productivity according to the step
        :param step:
        """
        self.b = self.adj_mat_by_step[step]
        self.D = np.diagflat(self.b.sum(axis=1))
        for agent in self.agents:
            # print(np.where(self.b[agent.id] > 0))
            agent.neighb = np.where(self.b[agent.id] > 0)[0]

            agent.produc = agent.prods[step]

    def local_voting_protocol(self, x: np.array) -> np.array:
        """
        Compute local voting protocol
        :param x: agents queue lengths
        :return: lvp result
        """
        lvp = (self.D - self.b) * x
        return np.squeeze(np.asarray(lvp))

    def acc_local_voting_protocol(self, x: np.array) -> np.array:
        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))
        x_n = 1 / (self.gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * self.gamma[0] * self.nesterov_step + self.gamma[1] * x)

        x_to_lvp = np.tile(x, (1, 5)).astype(float)
        np.fill_diagonal(x_to_lvp, x_n)
        y_n = np.diag(self.local_voting_protocol(x_to_lvp))

        y_n_vec = np.matrix(y_n).transpose() if y_n.shape == (self.n,) else y_n
        self.nesterov_step = 1 / self.gamma[0] * \
                             ((1 - self.alpha) * self.gamma[0] * self.nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * y_n_vec)

        H = self.h - self.h * self.h * self.L / 2
        if H - self.alpha * self.alpha / (2 * self.gamma[1]) < 0:
            logging.warning(H)
            print(f"H {H}")
            logging.exception(f"Oh no: {H - self.alpha * self.alpha / (2 * self.gamma[1])}")
            print(f"Oh no: {H - self.alpha * self.alpha / (2 * self.gamma[1])}")
            # raise BaseException()

        return y_n

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

        response = self.parallel.run(args_list=params_list, shared_vars=requests_dic)

        if not response:
            return {}

        send_tasks = {}
        for agent_id, res in response.items():
            for send_to_id, num in res:
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
    generate = False
    number_of_agents = 5
    productivity = 10
    num_steps = 20
    agents = [Agent(id, productivity, generate=generate, num_steps=num_steps) for id in range(number_of_agents)]

    pars = Parameters()
    pars.n = number_of_agents
    pars.theta_hat = np.matrix([len(agent.tasks) for agent in agents]).transpose()
    pars.b = np.matrix([[0, 0.3, 0.3, 0, 0, 0],
                        [0.3, 0, 0, 0, 0, 0.3],
                        [0.3, 0, 0, 0.3, 0, 0],
                        [0, 0, 0.3, 0, 0.3, 0],
                        [0, 0, 0, 0.3, 0, 0.3],
                        [0, 0.3, 0, 0, 0.3, 0]])
    pars.neib_add = 3
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
    alg_lvp.run(num_steps=1, accelerate=True, generate_neigh=generate)
    lvp_seq = alg_lvp.sequence
    logging.info("lvp_seq")

