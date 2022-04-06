from mesa import Agent
import numpy as np


class DeffuantAgent(Agent):
    def __init__(self, unique_id, model, x, u, mu, model_regime='p2p'):
        """
        :param unique_id: int -- ID агента
        :param model: mesa.Model -- Среда для агента
        :param x: float -- Мнение агента
        :param u: float -- Уровень сомнения агента
        :param mu: float -- Константный параметр, описывающий насколько мы восприимчивы к чужому мнению
        :param model_regime: str -- "p2p" или "all" -- режимы работы модели "парный" и "все со всеми"
        """

        super().__init__(unique_id, model)

        self.x = x
        self.u = u
        self.mu = mu
        self.delta_x = 0
        self.delta_u = 0
        self.regime = model_regime

    def interaction_p2p(self):
        other_agent = self.model.random.choice(self.model.schedule.agents)
        overlap = min(self.x + self.u, other_agent.x + other_agent.u) - \
                  max(self.x - self.u, other_agent.x - other_agent.u)

        if overlap > other_agent.u:
            delta_us = overlap / other_agent.u - 1
            self.delta_x = self.mu * delta_us * (other_agent.x - self.x)
            self.delta_u = self.mu * delta_us * (other_agent.u - self.u)

    def interaction_all(self):
        other_agents = [agent for agent in self.model.schedule.agents if agent != self]
        for other_agent in other_agents:
            overlap = min(self.x + self.u, other_agent.x + other_agent.u) - \
                      max(self.x - self.u, other_agent.x - other_agent.u)

            if overlap > other_agent.u:
                delta_us = overlap / other_agent.u - 1
                self.delta_x += (self.mu/self.model.num_agents * delta_us * (other_agent.x - self.x))
                self.delta_u += (self.mu/self.model.num_agents * delta_us * (other_agent.u - self.u))

    def step(self):
        self.x += self.delta_x
        self.x = max(self.x, -1)
        self.x = min(self.x, 1)
        self.u += self.delta_u
        self.u = max(self.u, 0.001)
        self.u = min(self.u, 2)
        self.delta_x = 0
        self.delta_u = 0
