import plotly.graph_objects as go

from prisk.asset.asset import Asset

class Holding:
    """
    The holding firm consists of a collection of assets in which
    it has a certain percentage of ownership.
    """
    def __init__(
            self, 
            name: str, 
            leverage_ratio: float=0.50) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the holding firm
        leverage_ratio : float
            The leverage ratio of the holding firm. The default value is 0.407867 which is based
            on an industry average obtained through FactSet.
        """
        
        self.name = name
        self.assets = []
        self.ownership = {}
        self.leverage_ratio = leverage_ratio
    
    def __str__(self):
        return self.name

    def add_asset(
            self,
            asset: "Asset", 
            ownership: float) -> None:
        """
        Add an asset to the holding firm

        Parameters
        ----------
        asset : Asset
            The asset to be added to the holding firm
        ownership : float
            The percentage ownership of the holding firm in the asset
        """
        self.assets.append(asset)
        self.ownership[asset] = ownership
        asset.parents.append({"firm": self, "ownership": ownership})

    def remove_asset(self, asset: "Asset") -> None:
        """
        Parameters
        ----------
        asset : Asset
            The asset to be removed from the holding firm

        """
        self.assets.remove(asset)
        del self.ownership[asset]
        asset.parents = [parent for parent in asset.parents if parent["firm"] != self]
    
    def get_asset_ownership(self, asset: "Asset") -> float:
        """
        Parameters
        ----------
        asset : Asset
            The asset to get the ownership of
        """
        return self.ownership.get(asset, 0)

    @property
    def base_value(self) -> float:
        return sum([asset.base_value * self.ownership[asset] for asset in self.assets])
    
    @property
    def total_replacement_costs(self) -> float:
        return sum([asset.total_replacement_costs * self.ownership[asset] for asset in self.assets])
    
    @property
    def total_business_disruption(self) -> float:
        return sum([asset.total_business_disruption * self.ownership[asset] for asset in self.assets])
    
    @property
    def total_fair_insurance_premiums(self) -> float:
        return sum([asset.total_fair_insurance_premiums * self.ownership[asset] for asset in self.assets])
    
    @property
    def total_insurance_adjustments(self) -> float:
        return sum([asset.total_insurance_adjustments * self.ownership[asset] for asset in self.assets])
    
    @property
    def npv(self) -> float:
        return sum([asset.npv * self.ownership[asset] for asset in self.assets])
    
    @property
    def liabilities(self) -> float:
        return self.leverage_ratio*self.original_assets
    
    @property
    def original_liabilities(self) -> float:
        return self.leverage_ratio*self.original_assets
    
    @property
    def leverage(self) -> float:
        return self.liabilities / self.total_assets
    
    @property
    def original_leverage(self) -> float:
        return self.original_liabilities / self.original_assets
    
    @property
    def delta_leverage(self) -> float:
        return self.leverage - self.original_leverage
    
    @property
    def profitability(self) -> float:
        return self.revenue / self.original_assets
    
    @property
    def original_profitability(self) -> float:
        return self.original_revenue / self.original_assets
    
    @property
    def delta_profitability(self) -> float:
        return self.profitability - self.original_profitability
    
    @property
    def original_revenue(self)  -> float:
        return sum([sum(asset.revenue_path) * self.ownership[asset]/len(asset.revenue_path) for asset in self.assets])
    
    @property
    def revenue(self) -> float:
        return sum([sum(asset.cash_flow_path + asset.cost_path) * self.ownership[asset]/len(asset.revenue_path) for asset in self.assets])
    
    @property
    def original_assets(self) -> float:
        return sum([asset.base_value * self.ownership[asset] for asset in self.assets])
    
    @property
    def total_assets(self) -> float:
        return sum([asset.npv * self.ownership[asset] for asset in self.assets])
    
    @property
    def delta_pd(self) -> float:
        # Parameters obtained from a regression analysis conducted by the ECB (2021 Economy
        # wide stress test). The parameters have been adjusted such that the delta_pd
        # is comparable to the merton model
        return 0.454*self.delta_leverage - 0.533*self.delta_profitability
    


    def plot_risk(self) -> None:
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
                xaxis_title="Firm value impacts: {0}".format(self.name),
                yaxis_title="Value",
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                yaxis_showgrid=True,
        )

        fig.show()
    

    
    