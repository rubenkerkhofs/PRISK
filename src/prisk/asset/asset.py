from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
import plotly.graph_objects as go

from prisk.flood import FloodExposure


@dataclass
class Location:
    """A location is a geographical point on the Earth's surface."""

    latitude: float
    longitude: float


class Asset:
    """An asset typically corresponds to a physical entity that generates revenue."""

    def __init__(
        self,
        name: str,
        status: str,
        country: str,
        location: Location,
        value: float,
        flood_exposure: List[FloodExposure],
        flood_damages: pd.DataFrame,
        discount_rate: float = 0.09,
        growth_rate: float = 0.05,
    ):
        self.name = name
        self.status = status
        self.country = country
        self.location = location
        self.flood_exposure = flood_exposure
        self.flood_damages = flood_damages
        self.value = value
        self.discount_rate = discount_rate
        self.growth_rate = growth_rate

    def __repr__(self):
        return f"Asset({self.name})"

    def __str__(self):
        return self.name
    
    @property
    def net_profit(self) -> float:
        return self.value * (self.discount_rate - self.growth_rate)
    
    @property
    def cash_flow(self) -> float:
        return self.value * (self.discount_rate - self.growth_rate)


class PowerAsset(Asset):
    """A power asset is a specific type of asset that produces electricity."""

    def __init__(
        self,
        capacity: float, # In MW
        type: Literal["bioenery", "oil/gas", "coal", "nuclear"],
        net_profit_margin: float = 0.1,
        capacity_factor: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.capacity = capacity
        self.type = type
        self.net_profit_margin = net_profit_margin
        self.capacity_factor = capacity_factor

    @property
    def annual_capacity(self) -> float:
        return self.capacity * 24 * 365 * self.capacity_factor
    
    @property
    def average_price(self)-> float:
        return self.net_profit / self.annual_capacity
    
    @property
    def expected_replacement_cost(self) -> float:
        expected_damages = 0
        total_probability = 0
        for exposure in self.flood_exposure:
            total_probability += exposure.poisson_probability
            damage = self.flood_damages.loc[exposure.depth].damage \
                * self.value
            expected_damages += exposure.poisson_probability * damage
        return expected_damages

    @property
    def expected_disruption_cost(self) -> float:
        expected_damages = 0
        for exposure in self.flood_exposure:
            disruption_days = self.flood_damages.loc[exposure.depth].production
            volume_change =(disruption_days/365)*self.annual_capacity
            # TO DO: price impact function
            price_change = 0
            damages = self.annual_capacity * price_change \
                + self.average_price * volume_change \
                + volume_change * price_change
            expected_damages += damages * exposure.poisson_probability
        return expected_damages
    
    @property
    def replacement_impact(self):
        return self.expected_replacement_cost
    
    @property
    def disruption_impact(self):
        return self.expected_disruption_cost * (1/(self.discount_rate - self.growth_rate))
    
    @property
    def expected_impact(self) -> float:
        return self.replacement_impact + self.disruption_impact
    
    @property
    def prisk(self) -> float:
        return (self.value - self.expected_impact) / self.value


    def plot_risk(self):
        fig = go.Figure(go.Waterfall(
            name = "PRISK - Waterfall", 
            orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Base Value", "Capital Damagages", "Business disruptions", "Adjusted Value"],
            textposition = "outside",
            text = ["{:,.2f}M".format(self.value/1e6), 
                    "{:,.2f}M".format(-self.replacement_impact/1e6), 
                    "{:,.2f}M".format(-self.disruption_impact/1e6), 
                    "{:,.2f}M".format((self.value - self.expected_impact)/1e6)],
            y = [self.value, 
                 -self.replacement_impact,
                -self.disruption_impact, 
                 self.value - self.expected_impact],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
                title = "PRISK - Waterfall chart",
                showlegend = False,
                template="simple_white",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Asset value impacts",
                yaxis_title="Value",
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                yaxis_showgrid=True,
        )

        fig.show()

