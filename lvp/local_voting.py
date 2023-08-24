import logging
from multiprocessing import Manager

import numpy as np
import pandas as pd

from lvp.objects import Parameters, Agent
from lvp.parallel import ParallelProcessing


class LvpParallel(ParallelProcessing):
    def __init__(self, request_dic, *args, **kwargs):
        super(LvpParallel, self).__init__(*args, **kwargs)
        self.request_dic = request_dic

    def process(self, agent: Agent, requests_dic, get_tasks: int, step: int, response_dict: dict) -> None:
        """
        Agent distribute it's tasks among neibours
        :param agent:  agent instance
        :param requests_dic: dictionary of requests
        :param get_tasks: result of counting local voting protocol for agent
        :return: dict {id send to: tasks to send}
        """
        agent_id = agent.id
        ag_tasks = len(agent.tasks)

        logging.basicConfig(filename=f'cache/loggs/_loggs_lvp_{agent_id}.log', filemode='a', level=logging.INFO)
        logging.info(f"\n\nStep {step}")

        requests_neib = [ind for ind in requests_dic.keys() if ind in agent.neib]
        response = []
        for req_id in requests_neib:
            if get_tasks == 0:
                break
            num_tasks = min(self._enter_crit_sec(req_id, get_tasks), ag_tasks)

            if not num_tasks:
                continue

            logging.info(f"Agent {agent_id} send {num_tasks} tasks to {req_id}")
            response.append((req_id, num_tasks))
            get_tasks -= num_tasks
            ag_tasks -= num_tasks

        response_dict[agent_id] = response
        logging.info(f"Agent {agent_id} ended sending")

    def init_child(self, par_lock_, request_dic_):
        super(LvpParallel, self).init_child(par_lock_)
        global request_dic
        request_dic = request_dic_

    def get_shared_vars(self, manager: Manager):
        request_dic = manager.dict()
        request_dic.update(self.request_dic)
        return (request_dic,)

    def critical_section(self, req_id: int, can_send: int):
        """
        Change shared dictionary requests
        :param req_id: neighbour from whom want to take tasks
        :param can_send: number of tasks that can send
        :return:
        """
        if req_id not in request_dic:
            return 0

        req = abs(request_dic[req_id])
        if req > can_send:
            request_dic[req_id] = -(req - can_send)
            send = can_send
        else:
            del request_dic[req_id]
            send = req
        return send


class LbAlgorithm:
    def __init__(self, agents, params: Parameters, **kwargs):
        """
        :param parameters: parameters of class Parameters
        """
        # Topology
        self.n = params.n
        self.agents = agents
        self.b = params.b
        self.neibours = {ind: np.where(self.b[ind] > 0)[1] for ind in range(self.n)}
        for ind, neib in self.neibours.items():
            self.agents[ind].neib = neib

        # Algorithm parameter
        self.h = params.params_dict.get("h", None)

        # Accelerated local protocol variables
        self.nesterov_step = np.array([[0]] * self.n)
        self.L = params.params_dict.get("L", None)
        self.mu = params.params_dict.get("mu", None)
        self.eta = params.params_dict.get("eta", None)
        self.gamma = params.params_dict.get("gamma", [])
        self.alpha = params.params_dict.get("alpha", None)

        self.D = np.diagflat(self.b.sum(axis=1))
        self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
        self.sequence = []
        self.sequence_2 = []

    def local_voting_protocol(self, x):
        lvp = (self.D - self.b) * x
        return np.squeeze(np.asarray(lvp))

    def acc_local_voting_protocol(self):
        self.gamma = [self.gamma[-1]]
        self.gamma.append((1 - self.alpha) * self.gamma[0] + self.alpha * (self.mu - self.eta))
        x_n = 1 / (self.gamma[0] + self.alpha * (self.mu - self.eta)) \
              * (self.alpha * self.gamma[0] * self.nesterov_step + self.gamma[1] * self.theta_hat)

        y_n = self.local_voting_protocol(x_n)
        y_n_vec = np.matrix(y_n).transpose() if y_n.shape == (self.n,) else y_n

        self.nesterov_step = 1 / self.gamma[0] * \
                             ((1 - self.alpha) * self.gamma[0] * self.nesterov_step
                              + self.alpha * (self.mu - self.eta) * x_n
                              - self.alpha * y_n_vec)

        H = self.h - self.h * self.h * self.L / 2
        if H - self.alpha * self.alpha / (2 * self.gamma[1]) < 0:
            logging.warning(H)
            logging.exception(f"Oh no: {H - self.alpha * self.alpha / (2 * self.gamma[1])}")
            logging.info("Exception")
            raise BaseException()

        return y_n

    def run(self, num_steps=100, eps=0.1, accelerate=False):
        for step in range(num_steps):
            logging.info(f"\n\nStep: {step}")
            for agent in self.agents:
                df = [task.to_dict() for task in agent.tasks]
                logging.info(pd.DataFrame(df).sum())

            # Get new tasks
            [agent.update_with_new_tasks(step) for agent in self.agents]

            # Complete some tasks
            [agent.complete_tasks() for agent in self.agents]

            # Compute local voting protocol
            self.theta_hat = np.matrix([[agent.theta_hat] for agent in self.agents])
            logging.info(f"Current distribution on {step} step: {self.theta_hat}")
            logging.info(f"Neighbors: {self.neibours}")

            if not accelerate:
                u_lvp = self.local_voting_protocol(self.theta_hat)
            else:
                u_lvp = self.acc_local_voting_protocol()
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

            self.sequence.append([agent.get_real_queue_length() for agent in self.agents])
            self.sequence_2.append([agent.theta_hat for agent in self.agents])
            print(f"Step {step} is completed")

    def receive_tasks(self, response):
        for agent_id, tasks in response.items():
            self.agents[agent_id].receive_tasks(tasks)

    def distribute_tasks(self, requests_dic, u_lvp, step):
        """
        For each agent that want to send tasks find agent to send to
        :param requests_dic: {agent id: number of tasks to send}
        :param u_lvp: local voting protocol result
        :return: {agent id: tasks to send}
        """
        resp_age = [a for a in self.agents if u_lvp[a.id] > 0]
        params_list = []
        for agent in resp_age:
            params_list.append((agent, requests_dic, u_lvp[agent.id], step,))

            # response.extend(self.distribute_agents_tasks(agent, requests_dic, u_lvp[agent.id]))

        mpp = LvpParallel(requests_dic)
        response = mpp.run(args_list=params_list)

        if not response:
            return {}

        send_tasks = {}
        for agent_id, res in response.items():
            for send_to_id, num in res:
                tasks = self.agents[agent_id].get_tasks(int(num))
                send_tasks.setdefault(send_to_id, []).extend(tasks)

        return send_tasks

    def get_request(self, requests: dict, req_id: int, can_send: int):
        """
        Change shared dictionary requests
        :param requests: {agent_id: -number of needed tasks}
        :param req_id: neighbour from whom want to take tasks
        :param can_send: number of tasks that can send
        :return:
        """
        req = abs(requests[req_id])
        if req > can_send:
            requests[req_id] = -(req - can_send)
            send = can_send
        else:
            del requests[req_id]
            send = req
        return send

    def distribute_agents_tasks(self, agent: Agent, requests_dic, get_tasks: int):
        """
        Agent distribute it's tasks among neibours
        :param agent:  agent instance
        :param requests_dic: dictionary of requests
        :param get_tasks: result of counting local voting protocol for agent
        :return: dict {id send to: tasks to send}
        """
        agent_id = agent.id
        requests_neib = [ind for ind in requests_dic.keys() if ind in self.neibours[agent_id]]
        response = []
        for req_id in requests_neib:
            if get_tasks == 0:
                break

            num_tasks = self.get_request(requests_dic, req_id, get_tasks)

            if not num_tasks:
                continue

            tasks = agent.get_tasks(int(num_tasks))
            logging.info(f"Agent {agent_id} send {num_tasks} tasks to {req_id}")
            response.append((req_id, tasks))
            get_tasks -= num_tasks
        return response


def safe_list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default


if __name__ == "__main__":
    logging.basicConfig(filename=f'cache/loggs/_loggs_lvp.log', filemode='w', level=logging.INFO)
    number_of_agents = 6
    productivity = 10
    agents = [Agent(id, productivity) for id in range(number_of_agents)]

    pars = Parameters()
    pars.n = number_of_agents
    pars.theta_hat = np.matrix([len(agent.tasks) for agent in agents]).transpose()
    pars.b = np.matrix([[0, 0.3, 0.3, 0, 0, 0],
                        [0.3, 0, 0, 0.3, 0, 0.3],
                        [0.3, 0, 0, 0.3, 0.3, 0],
                        [0, 0.3, 0.3, 0, 0.3, 0],
                        [0, 0, 0.3, 0.3, 0, 0.3],
                        [0, 0.3, 0, 0, 0.3, 0]])
    pars.params_dict = {
        "L": 7.1,
        "mu": 0.9,
        "h": 0.2,
        "eta": 0.8,
        "gamma": [[0.07, 0.09, 0.11][0]],
        "alpha": [0.07, 0.09, 0.11][1]
    }
    alg = LbAlgorithm(agents=agents, params=pars)
    alg.run(num_steps=20, accelerate=False)
