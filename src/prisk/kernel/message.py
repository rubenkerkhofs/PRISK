from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from prisk.asset import Asset

@dataclass
class Message:
    def __init__(self, time: pd.Timestamp):
        self.time = time

    def send(self, kernel):
        kernel.messages.put(self)

    def __lt__(self, other: "Message") -> bool:  # type: ignore
        return self.time < other.time

    def __eq__(self, other: "Message") -> bool:  # type: ignore
        return self.time == other.time

    def __gt__(self, other: "Message") -> bool:  # type: ignore
        return self.time > other.time

@dataclass
class FloodEvent(Message):
    time: pd.Timestamp
    depth: float
    asset: "Asset"
    
@dataclass
class StartofYearEvent(Message):
    time: pd.Timestamp


@dataclass
class InsuranceDropoutEvent(Message):
    time: pd.Timestamp
    asset: "Asset"