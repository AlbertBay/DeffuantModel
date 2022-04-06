from mesa import Model
from mesa.datacollection import DataCollector
from Agent import DeffuantAgent
from mesa.time import StagedActivation
import numpy as np


# stage activation


class DeffuantModel(Model):
    def __init__(self, n, mu, gen_u, ext_u=None, extremist_ratio=None, d=None, regime="p2p"):
        """
        :param n: int -- Колличество агентов
        :param mu: float -- Константный параметр, описывающий насколько мы восприимчивы к чужому мнению
        :param gen_u: float -- Неопределенность агентов
        :param d: float -- Отношение экстремистов
        :param regime: str -- "p2p" or "all" -- режимы работы модели "парный" и "все со всеми"
        """
        super().__init__()

        self.num_agents = n
        self.gen_u = gen_u

        self.ext_num = int(extremist_ratio * self.num_agents)
        self.ext_u = ext_u
        self.ext_plus = int((d + 1) * self.ext_num / 2)
        self.ext_minus = self.ext_num - self.ext_plus

        self.mu = mu
        self.regime = regime
        self.stage_list = ['interaction_' + regime, 'step']
        self.schedule = StagedActivation(model=self, stage_list=self.stage_list, shuffle=False,
                                         shuffle_between_stages=False)
        self.opinions = np.random.uniform(-1.01, 1.01, self.num_agents)
        uncertaincies = np.full(self.num_agents, gen_u).astype(float)

        extr_ind = np.concatenate([np.argsort(self.opinions)[:self.ext_minus],
                                   np.argsort(self.opinions)[-self.ext_plus:]])
        uncertaincies[extr_ind] = self.ext_u
        self.uncertaincies = uncertaincies

        for id, (opinion, uncertaincy) in enumerate(zip(self.opinions, self.uncertaincies)):
            a = DeffuantAgent(unique_id=id, model=self, x=opinion, u=uncertaincy, mu=self.mu, model_regime=self.regime)
            self.schedule.add(a)
        self.datacollector = DataCollector(agent_reporters={"Opinion": "x"})

    def step(self):

        self.datacollector.collect(self)
        self.schedule.step()