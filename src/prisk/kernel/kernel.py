from queue import PriorityQueue

from numpy import floor

from prisk.kernel.message import FloodEvent, StartofYearEvent, InsuranceDropoutEvent

class Kernel:
    """
        The Kernel provides the environment in which the simulations
        will take place
    """
    def __init__(self, assets, insurers):
        self.messages = PriorityQueue()
        self.internal_time = 0 # Expressed in years
        self.assets = assets
        self.insurers = insurers

    def run(self, time_horizon, verbose: int=0):
        """ Run the simulation """
        if verbose > 0:
            print("Starting simulation")
            print("-------------------")
            print("Adding End of Year events...")
        for i in range(0, time_horizon):
            self.messages.put(StartofYearEvent(i))

        while not self.messages.empty() and self.internal_time < time_horizon:
            message = self.messages.get()
            self.internal_time = message.time
            if isinstance(message, FloodEvent):
                if verbose:
                    print(f"Flood event at year {int(floor(self.internal_time))} at {message.asset} with depth {message.depth}")
                message.asset.flood(time=message.time, depth=message.depth)
            elif isinstance(message, StartofYearEvent):
                for insurer in self.insurers:
                    insurer.collect_premiums(message.time)
            elif isinstance(message, InsuranceDropoutEvent):
                if verbose:
                    print(f"Insurance dropout at year {int(floor(self.internal_time))} at {message.asset}")
                message.asset.remove_insurer()
        
        self.internal_time = time_horizon
        if verbose > 0:
            print("-------------------")
            print("Simulation finished")
            print(f"Simulation time: {round(self.internal_time, 4)} years")



    