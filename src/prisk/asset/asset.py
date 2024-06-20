""" Contains an  """
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class Asset:
    def __init__(
            self,
            name: str,
            flood_damage_curve: pd.Series,
            flood_exposure: float,
            production_path: np.ndarray,
            replacement_cost: float,
            unit_price: float=60,
            margin: float=0.2,
            discount_rate: float=0.05,
            insurer=None
            ):
        """ Initialize the AssetSim object

        Parameters
        ----------
        name : str
            Name of the asset
        flood_damage_curve : pd.Series
            A pandas series with the depth of flood as index and the damage as values.
            There are two columns: (1) damage, (2) production. The damage is the fraction 
            of the replacement cost that is lost, and the production is the fraction of
            production that is lost.
        flood_exposure : float
            The flood exposure of the asset. This is needed to compute the expected 
            damage needed to determine the insurer's premium.
        production_path : np.ndarray
            The production path of the asset. This is used to compute the revenue path.
        replacement_cost : float
            The replacement cost of the asset. This is used to compute the damage in case of a flood.
        unit_price : float
            The unit price of the asset. This is used to compute the revenue path.
        margin : float
            The margin of the asset. This is used to compute the cost path.
        discount_rate : float
            The discount rate of the asset. This is used to compute the NPV.
        insurer : Insurance
            The insurance company that insures the asset. This is used to compute the premium.
        """
        self.name = name
        self.flood_damage_curve = flood_damage_curve
        self.production_path = production_path
        self._TIME_HORIZON = len(production_path)
        self.damages = np.repeat(0.0, self._TIME_HORIZON)
        self.disruptions = np.repeat(0.0, self._TIME_HORIZON)
        self.discount_rate = discount_rate
        self.replacement_cost = replacement_cost
        self.flood_exposure = flood_exposure
        self._insurer = insurer

        self.unit_price = unit_price
        self._MARGIN = margin
        self.cost_path = self.revenue_path * (1 - self._MARGIN)
        self.flood_protection = 0.0
        self.update_expected_damage()
        self.replacement_cost_path = np.repeat(0, self._TIME_HORIZON)
        self.business_disruption_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_fair_premium_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_adjustment_path = np.repeat(0, self._TIME_HORIZON)
        self.base_value = self.npv



    def __str__(self):
        return self.name
    
    @property
    def revenue_path(self):
        return self.production_path * self.unit_price
    
    @property
    def cash_flow_path(self):
        return self.revenue_path - self.cost_path - self.climate_cost_path
    
    @property
    def discounted_cash_flow(self):
        return np.sum(self.cash_flow_path * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON))

    @property
    def climate_cost_path(self):
        return self.replacement_cost_path \
                + self.business_disruption_path \
                + self.insurance_fair_premium_path \
                + self.insurance_adjustment_path

    @property
    def terminal_value(self):
        return (self.cash_flow_path[-1] + self.climate_cost_path[-1]) / (self.discount_rate)

    @property
    def npv(self):
        return self.discounted_cash_flow + self.terminal_value/ (1 + self.discount_rate) ** (self._TIME_HORIZON+1)

    @property
    def total_replacement_costs(self):
        return np.sum(self.replacement_cost_path * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON))
    
    @property
    def total_business_disruption(self):
        return np.sum(self.business_disruption_path * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON))
    
    @property
    def total_fair_insurance_premiums(self):
        return np.sum(self.insurance_fair_premium_path * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON))

    @property
    def total_insurance_adjustments(self):
        return np.sum(self.insurance_adjustment_path * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON))

    @property
    def expected_damage(self):
        return self.__expected_damage

    def update_expected_damage(self):
        expected_damage = 0
        for flood_exposure in self.flood_exposure:
            impact_depth = round(max(0, flood_exposure.depth - self.flood_protection), 2)
            expected_damage += self.flood_damage_curve.loc[impact_depth].damage\
                                     * flood_exposure.poisson_probability \
                                     * self.replacement_cost
        self.__expected_damage = expected_damage

    def add_insurer(self, insurer):
        self._insurer = insurer
        insurer.add_subscriber(self)

    def remove_insurer(self):
        self._insurer.remove_subscriber(self)
        self._insurer = None
 
    def pay_insurance_premium(self, time):
        premium = self._insurer.premium(self)*self.replacement_cost
        fair_premium = self._insurer.get_fair_premium(self)*self.replacement_cost
        year = int(np.floor(time))
        self.insurance_fair_premium_path[year] += fair_premium
        self.insurance_adjustment_path[year] += premium - fair_premium

    def flood(self, depth: float, time: pd.Timestamp):
        """ Simulate the flood event """
        impact_depth = round(max(0, depth - self.flood_protection), 2)
        damage = self.flood_damage_curve.loc[impact_depth].damage
        production = self.flood_damage_curve.loc[impact_depth].production
        year = int(np.floor(time))
        if self._insurer is None:
            self.replacement_cost_path[year] += self.replacement_cost*damage
        else:
            self._insurer.payout(self.replacement_cost*damage)
        # Install flood protection
        if damage > 0:
            self.flood_protection = depth
            self.update_expected_damage()
        
        self.business_disruption_path[year] += self.revenue_path[year]*production/365
        # For now, we asusme a simple relation with costs. This needs to be
        # changed to a more sophisticated model when more realistic data
        # on production disruptions is available.
        self.business_disruption_path[year] -= self.cost_path[year]*min(0.5, production/(365*2))
        
    def plot_risk(self):
        fig = go.Figure(go.Waterfall(
            name = "PRISK - Waterfall", 
            orientation = "v",
            measure = ["relative", "relative", "relative", "relative", "relative", "total"],
            x = ["Base Value", "Capital Damagages", "Business disruptions", "Fair insurance premiums",
                 "Insurance adjustments", "Adjusted Value"],
            textposition = "outside",
            text = ["{:,.2f}M".format(self.base_value/1e6), 
                    "{:,.2f}M".format(-self.total_replacement_costs/1e6), 
                    "{:,.2f}M".format(-self.total_business_disruption/1e6), 
                    "{:,.2f}M".format(-self.total_fair_insurance_premiums/1e6),
                    "{:,.2f}M".format(-self.total_insurance_adjustments/1e6),
                    "{:,.2f}M".format(self.npv/1e6)],
            y = [self.base_value, 
                 -self.total_replacement_costs,
                -self.total_business_disruption,
                -self.total_fair_insurance_premiums, 
                -self.total_insurance_adjustments,
                 self.npv],
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
    

class PowerPlant(Asset):
    def __init__(
            self,
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)

    
    
    

    
    

    