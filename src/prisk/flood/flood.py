import dataclasses
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats import poisson

from prisk.kernel.message import FloodEvent

@dataclasses.dataclass
class FloodExposure:
    """
    The FloodExposure class represents the exposure of an entity to a flood event expressed
    in terms of return period and depth.
    """
    return_period: float
    depth: float

    @property
    def probability(self) -> float:
        """ The probability of the flood event based on the return period"""
        return 1 / self.return_period

    @property
    def poisson_probability(self) -> float:
        """ The probability of the flood event based on the return period under the poisson distribution.
        Both of the probabilities converge to the same value as the return period increases"""
        return 1 - np.exp(-1 / self.return_period)

    def __str__(self) -> str:
        return f"FloodExposure({self.return_period}, {self.depth})"

class FloodBasinSim:
    def __init__(self, entity, events):
        """
        The FloodBasinSim class simulates the flood events based on the return periods
        and the depth associated to these return periods. The simulations are done
        at the basin-level.
        """
        self.entity = entity
        self.events = events

    def simulate(self, kernel):
        """
        Simulates flood events and adds them to the kernal queue

        Parameters
        ----------
        kernel : Kernel
            The kernel object that manages the simulation
        """
        for exposure in self.entity.flood_exposure:
            rp = exposure.return_period
            rp_events = self.events[self.events.return_period == rp].to_dict("records")
            for event in rp_events:
                for i in range(int(event["events"])):
                    FloodEvent(event["year"]-1, exposure.depth, self.entity).send(kernel=kernel)

    @classmethod
    def generate_events_set(
            self, 
            random_numbers: pd.DataFrame,
            years:int = 25):
        """
        Generate a set of flood events based on the return periods and the number of events
        for each return period

        Parameters
        ----------
        years : int
            The number of years to generate the events for
        random_numbers : pd.DataFrame
            The random numbers to use for the simulation.
        """
        return_periods = [5, 10, 25, 50, 100, 200, 500, 1000]
        events = pd.DataFrame()
        for return_period in return_periods:
            simulated_data = random_numbers.sample(years)
            simulated_data = simulated_data.apply(lambda x: poisson.ppf(x, 1/return_period)).reset_index().clip(0, 1)
            simulated_data = simulated_data.replace(0, pd.NA).melt(id_vars="index").dropna()
            if simulated_data.empty:
                continue
            simulated_data.loc[:, "return_period"] = return_period
            events = pd.concat([events, simulated_data])
        events.columns = ["year", "basin", "events", "return_period"]
        events.basin = events.basin.astype(str)
        return events

    @classmethod
    def events_df(self, random_numbers, years=25):
        return_periods = [5, 10, 25, 50, 100, 200, 500, 1000]
        events = pd.DataFrame()
        for return_period in return_periods:
            simulated_data = random_numbers.sample(years)
            simulated_data = simulated_data.apply(lambda x: poisson.ppf(x, 1/return_period)).reset_index().clip(0, 1)
            simulated_data = simulated_data.replace(0, pd.NA).melt(id_vars="index").dropna()
            if simulated_data.empty:
                continue
            simulated_data.loc[:, "return_period"] = return_period
            events = pd.concat([events, simulated_data])
        events.columns = ["year", "basin", "events", "return_period"]
        events.basin = events.basin.astype(str)
        return events


class FloodEntitySim:
    """ The FloodEntitySim allows the simulation of floods based on
    the exposures of a certain entity """
    def __init__(self, entity, model: str = "poisson", random_seed: Optional[int] = None):
        self.entity = entity
        self.exposures = entity.flood_exposure
        self.model = model
        self.random_seed = random_seed


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
        if self.random_seed:
            np.random.seed(self.random_seed)
        if self.model == "poisson":
            self._simulate_poisson(time_horizon, kernel)

