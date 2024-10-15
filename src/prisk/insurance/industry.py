from typing import List

class Insurance:
    def __init__(
            self,
            name: str,
            capital: float,
            subscribers: List=[],
            adjust_premiums: bool = True,
            sensitivity: float = 1):
        self.name = name
        self.capital = capital
        self.start_capital = capital
        self.subscribers = subscribers.copy() # avoid default value being mutated, very important!
        self.adjust_premiums = adjust_premiums
        self.premium_adj_history = []
        self.sensitivity = sensitivity

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)

    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)

    def payout(self, amount: float):
        self.capital -= amount

    @property
    def total_expected_damage(self):
        return sum(asset.expected_damage for asset in self.subscribers)
    
    @property
    def total_insured_value(self):
        return sum(asset.replacement_cost for asset in self.subscribers)

    def premium(self, asset):
        fair_premium = self.get_fair_premium(asset)
        if not self.adjust_premiums:
            return fair_premium
        minp = 0.7*fair_premium
        maxp = 1.35*fair_premium
        s = self.sensitivity
        capital_ratio = self.start_capital/self.capital
        p = fair_premium*(1 + s*(capital_ratio-1))
        return min(maxp, max(minp, p))
    
    def get_fair_premium(self, asset):
        return asset.expected_damage/asset.replacement_cost


    def collect_premiums(self, time):
        for subscriber in self.subscribers:
            subscriber.pay_insurance_premium(time)
            self.capital += self.premium(subscriber)*subscriber.replacement_cost


