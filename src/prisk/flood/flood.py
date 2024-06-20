import dataclasses
from typing import List

import numpy as np

from prisk.kernel.message import FloodEvent

@dataclasses.dataclass
class FloodExposure:
    return_period: float
    depth: float

    @property
    def probability(self):
        return 1 / self.return_period

    @property
    def poisson_probability(self):
        return 1 - np.exp(-1 / self.return_period)

    def __str__(self):
        return f"FloodExposure({self.return_period}, {self.depth})"



class FloodEntitySim:
    """ The FloodEntitySim allows the simulation of floods based on
    the exposures of a certain entity """
    def __init__(self, entity, model: str = "poisson"):
        self.entity = entity
        self.exposures = entity.flood_exposure
        self.model = model


    def _simulate_poisson(self, time_horizon: float, kernel):
        """ Simulate the floodings using the Poisson model 
        
        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        for exposure in self.exposures:
            time = np.random.exponential(exposure.return_period)
            while time < time_horizon:
                FloodEvent(time, exposure.depth, self.entity).send(kernel=kernel)
                time += np.random.exponential(exposure.return_period)

    def simulate(self, time_horizon: float, kernel):
        """ Simulate the floodings 
        
        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        if self.model == "poisson":
            self._simulate_poisson(time_horizon, kernel)