import pandas as pd

class Portfolio:
    def __init__(self, name: str=""):
        self.name = name
        self.__positions = []

    def add_position(self, firm, value):
        self.__positions.append(
            {"firm": firm,
             "exposure": value}
        )
    
    @property
    def underlying_value(self):
        return round(sum([p["exposure"]*p["firm"].npv for p in self.__positions]), 4)

    @property
    def assets(self):
        assets = []
        firms = []
        firm_exposures = []
        asset_ownerships = []
        exposures = []
        values = []
        for p in self.__positions:
            for a in p["firm"].assets:
                assets.append(a)
                firms.append(p["firm"])
                firm_exposures.append(p["exposure"])
                asset_ownerships.append(p["firm"].ownership[a])
                exposures.append(p["exposure"]*p["firm"].ownership[a])
                values.append(a.base_value)

        return pd.DataFrame({
            "asset": assets,
            "firm": firms,
            "firm_exposure": firm_exposures,
            "asset_ownership": asset_ownerships,
            "exposure": exposures,
            "value": values})

    @property
    def positions(self):
        return pd.DataFrame(self.__positions)
    
    def reset(self):
        for asset in self.assets:
            asset.reset()
    
    